"""SageMaker managed MLflow through the official ``mlflow`` + ``sagemaker-mlflow`` libraries.

Opt-in (``aws-mlflow`` extra): when both packages are installed and ``MLFLOW_TRACKING_URI`` is a
SageMaker MLflow App / Tracking Server ARN, a run report is published with the regular MLflow client:

- experiment tracking: experiment, resumable run (keyed by ``zia_run_key``), params, tags, metrics;
- detailed plots: matplotlib dashboards through ``mlflow.log_figure()``;
- dataset versioning: ``mlflow.log_input`` with a content-addressed (sha256) dataset;
- model versioning: the final model logged as an MLflow ``pyfunc`` model plus one Model Registry
  version per continual-learning checkpoint, with aliases (``initial``, ``step-N``, ``final``,
  ``current``; MLflow reserves ``latest``);
- artifacts: the run directory, mirrored into the run's artifact root.

The ``sagemaker-mlflow`` plugin (entry point ``arn``) signs every request with SigV4, so no HTTP code
lives here. Everything imports lazily: the default AWS image has no MLflow and keeps using the REST
tracker in ``mlflow_tracking.py``, which is also the fallback when this publisher raises.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from continual_bench.infrastructure.tracking import MlflowClientBackend
from continual_bench.infrastructure.tracking.artifacts import artifact_plan, upload_concurrently
from continual_bench.infrastructure.tracking.rounds import publish_training_rounds
from continual_bench.infrastructure.zia_model_bundle import (
    PYFUNC_LOADER,
    pyfunc_code_paths,
    pyfunc_dependencies,
)

from .mlflow_tracking import ARTIFACT_ROOT, report_metric_items, report_params


class LibraryPublishError(RuntimeError):
    """The library path failed; the caller falls back to the REST tracker."""


def is_sagemaker_arn(uri: str | None) -> bool:
    return bool(uri) and str(uri).startswith("arn:aws") and ":sagemaker:" in str(uri)


def library_available() -> bool:
    """True when both ``mlflow`` and the ``sagemaker-mlflow`` plugin can be imported."""
    try:
        import mlflow  # noqa: F401
        import sagemaker_mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-")


def registry_names(tracker, report: dict[str, Any]) -> tuple[str, str]:
    """(per-policy model name, cross-policy family model name), same scheme as the REST tracker."""
    dataset = _slug(report.get("dataset") or tracker.dataset)
    policy = _slug(report.get("policy") or tracker.policy)
    suffix = "-smoke" if str(report.get("scope") or tracker.scope) == "smoke" else ""
    return (
        f"{tracker.project}-zia-{dataset}-{policy}{suffix}"[:57].rstrip("-"),
        f"{tracker.project}-zia-{dataset}{suffix}"[:57].rstrip("-"),
    )


def _log_dataset(mlflow, tracker, report: dict[str, Any], outputs_uri: str | None) -> bool:
    """Dataset lineage with a content digest, so a dataset revision is a distinct MLflow input."""
    inp = (report.get("contract") or {}).get("input") or {}
    source = inp.get("source") or {}
    sha = str(inp.get("sha256") or "")
    if not (source.get("repo") and sha):
        return False
    from mlflow.data.http_dataset_source import HTTPDatasetSource
    from mlflow.data.meta_dataset import MetaDataset

    name = f"ziadatalabs/{source['repo']}"
    revision, filename = source.get("revision", "main"), source.get("file", "")
    url = f"https://huggingface.co/datasets/{name}/resolve/{revision}/{filename}"
    dataset = MetaDataset(HTTPDatasetSource(url), name=name, digest=sha[:32])
    tags = {
        "dataset_sha256": sha,
        "dataset_revision": str(source.get("revision", "")),
        "dataset_rows": str(inp.get("rows", "")),
    }
    if outputs_uri:
        tags["outputs_uri"] = str(outputs_uri)
    mlflow.log_input(dataset, context=str(report.get("protocol", "training")), tags=tags)
    return True


def _log_artifacts(client, run_id: str, output_dir: Path | None, *, exclude=()) -> int:
    """Upload the shared artifact layout once, concurrently (``client`` is thread-safe; the
    fluent ``mlflow.log_artifact`` is bound to the caller thread's active run)."""
    if output_dir is None or not Path(output_dir).is_dir():
        return 0

    def put(path: Path, target: str) -> None:
        parent = target.rpartition("/")[0]
        client.log_artifact(run_id, str(path), artifact_path=parent or None)

    return len(upload_concurrently(put, artifact_plan(output_dir, exclude=exclude)))


def _saved_figures(output_dir: Path | None) -> list[str]:
    """Figures ``prepare_artifacts`` already rendered under ``artifacts/plots``."""
    root = Path(output_dir) / "artifacts" if output_dir is not None else None
    if root is None or not (root / "plots").is_dir():
        return []
    return sorted(path.relative_to(root).as_posix() for path in (root / "plots").rglob("*.png"))


def _ensure_registered_model(client, name: str, description: str, tags: dict[str, str]) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(name, tags=tags, description=description)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise


def _register_checkpoints(
    client, tracker, report, run, model_package_arn, pyfunc_logged=False
) -> dict[str, Any]:
    """One registry version per checkpoint (idempotent per run) plus the cross-policy family version."""
    policy_name, family_name = registry_names(tracker, report)
    contract = report.get("contract") or {}
    sha = str((contract.get("input") or {}).get("sha256", ""))
    metrics = report.get("metrics") or {}
    holdout = report.get("holdout") or {}
    versions = list(report.get("model_versions") or []) or [
        {
            "action": "final",
            "end_seq": int((contract.get("input") or {}).get("rows") or 0),
            "file": "model/model.tar.gz",
        }
    ]
    base = run.info.artifact_uri.rstrip("/")
    common = {
        "dataset": str(report.get("dataset") or tracker.dataset),
        "policy": str(report.get("policy") or tracker.policy),
        "scope": str(report.get("scope") or tracker.scope),
        "model": str(report.get("model") or tracker.model_name),
        "project": tracker.project,
    }
    _ensure_registered_model(client, policy_name, f"Continual-Bench checkpoints ({common['policy']})", common)
    _ensure_registered_model(
        client,
        family_name,
        "Continual-Bench policies side by side",
        {k: v for k, v in common.items() if k != "policy"},
    )
    done = {
        mv.tags.get("ordinal"): mv
        for mv in client.search_model_versions(f"name='{policy_name}'")
        if mv.run_id == run.info.run_id
    }
    created: list[str] = []
    for index, version in enumerate(versions):
        final = index == len(versions) - 1
        if str(index) in done:
            created.append(str(done[str(index)].version))
            continue
        end_seq = int(version.get("end_seq", 0))
        tags = {
            "ordinal": str(index),
            "action": str(version.get("action") or ("initial" if index == 0 else "update")),
            "start_seq": str(version.get("start_seq", 0)),
            "end_seq": str(end_seq),
            "train_samples": str(version.get("train_samples", 0)),
            "model_sha256": str(version.get("sha256", "")),
            "dataset_version": sha,
            "is_final": str(final).lower(),
            "artifact_format": "mlflow_pyfunc" if final and pyfunc_logged else "zia_native_checkpoint",
            "prequential_accuracy": str(metrics.get("accuracy", "")),
            "holdout_accuracy": str(holdout.get("accuracy", "")),
        }
        if final and model_package_arn:
            tags["sagemaker_model_package_arn"] = str(model_package_arn)
        relative = "model_pyfunc" if final and pyfunc_logged else version.get("file") or "model/model.tar.gz"
        created_version = client.create_model_version(
            policy_name,
            f"{base}/{relative}",
            run_id=run.info.run_id,
            tags={k: v for k, v in tags.items() if v != ""},
            description=(
                f"Checkpoint {index} ({tags['action']}) seq<= {end_seq}, sha256={tags['model_sha256'][:12]}"
            ),
        )
        number = str(created_version.version)
        created.append(number)
        aliases = (
            (["initial"] if index == 0 else [])
            + [f"step-{end_seq}"]
            + (["final", "current"] if final else [])
        )
        for alias in aliases:
            client.set_registered_model_alias(policy_name, alias, number)
    family_version = ""
    existing = [
        mv for mv in client.search_model_versions(f"name='{family_name}'") if mv.run_id == run.info.run_id
    ]
    if existing:
        family_version = str(existing[0].version)
    else:
        last = versions[-1]
        final_path = "model_pyfunc" if pyfunc_logged else last.get("file") or "model/model.tar.gz"
        fam = client.create_model_version(
            family_name,
            f"{base}/{final_path}",
            run_id=run.info.run_id,
            tags={
                "policy": common["policy"],
                "ordinal": "final",
                "artifact_format": "mlflow_pyfunc" if pyfunc_logged else "zia_native_checkpoint",
                "dataset_version": sha,
                "model_sha256": str(last.get("sha256", "")),
                "prequential_accuracy": str(metrics.get("accuracy", "")),
            },
            description=f"policy={common['policy']} ({len(versions)} checkpoints)",
        )
        family_version = str(fam.version)
        client.set_registered_model_alias(family_name, _slug(common["policy"]), family_version)
        client.set_registered_model_alias(family_name, "current", family_version)
    return {
        "registered_model": policy_name,
        "registered_model_version": created[-1] if created else "",
        "registered_model_versions": created,
        "registered_family_model": family_name,
        "registered_family_version": family_version,
    }


def _log_pyfunc_model(mlflow, output_dir: Path | None, report) -> bool:
    """Log the final model as an MLflow pyfunc model so it is loadable with ``mlflow.pyfunc.load_model``."""
    if output_dir is None:
        return False
    output_dir = Path(output_dir)
    metadata_path = output_dir / "model-metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    relative = report["model_versions"][-1]["file"]
    with TemporaryDirectory(prefix="zia-pyfunc-") as temporary:
        staging = Path(temporary)
        data = staging / "state"
        data.mkdir()
        shutil.copy2(metadata_path, data / "model-metadata.json")
        shutil.copy2(output_dir / relative, data / metadata["payload"])
        path = staging / "model"
        mlflow.pyfunc.save_model(
            path=str(path),
            loader_module=PYFUNC_LOADER,
            data_path=str(data),
            code_paths=pyfunc_code_paths(staging),
            pip_requirements=pyfunc_dependencies(metadata["model"]),
        )
        mlflow.pyfunc.load_model(str(path))
        mlflow.log_artifacts(str(path), artifact_path="model_pyfunc")
    return True


def publish_with_library(
    tracker,
    report: dict[str, Any],
    *,
    adapted: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    outputs_uri: str | None = None,
    model_package_arn: str | None = None,
) -> dict[str, Any]:
    """Publish ``report`` with the MLflow client; raise ``LibraryPublishError`` on failure."""
    if not is_sagemaker_arn(tracker.tracking_uri):
        raise LibraryPublishError("tracking URI is not a SageMaker MLflow ARN")
    if not library_available():
        raise LibraryPublishError("mlflow and sagemaker-mlflow are not installed")
    import mlflow
    from mlflow.tracking import MlflowClient

    from continual_bench.infrastructure.mlflow_figures import log_report_figures
    from continual_bench.infrastructure.observation_tracking import publish_sdk_observations
    from continual_bench.infrastructure.tracking import missing_metric_points

    try:
        mlflow.set_tracking_uri(tracker.tracking_uri)
        client = MlflowClient()
        backend = MlflowClientBackend(client)
        experiment = mlflow.set_experiment(tracker.experiment_name)
        tags = {t["key"]: t["value"] for t in tracker.mlflow_tags}
        resumed = tracker.mlflow_run_id
        if resumed is None:
            found = client.search_runs(
                [experiment.experiment_id],
                f"tags.zia_run_key = '{tracker.run_key}'",
                max_results=2,
            )
            if len(found) > 1:
                raise ValueError(f"duplicate MLflow parent runs for {tracker.run_key}")
            resumed = found[0].info.run_id if found else None
        summary, holdout = report.get("metrics") or {}, report.get("holdout") or {}
        note = (
            f"### Continual-Bench run `{tracker.dataset}` | `{tracker.policy}` | `{tracker.scope}`\n"
            f"- Prequential accuracy: `{summary.get('accuracy', 'n/a')}`\n"
            f"- Holdout accuracy: `{holdout.get('accuracy', 'n/a')}`\n"
            f"- Outputs: `{outputs_uri or 'n/a'}`"
        )
        with mlflow.start_run(
            run_id=resumed,
            run_name=None if resumed else tracker.run_name,
            experiment_id=None if resumed else experiment.experiment_id,
            tags=tags,
        ) as run:
            run_id = run.info.run_id
            # REST fallback must retain this identity even if SDK publication fails below.
            tracker.experiment_id = str(experiment.experiment_id)
            tracker.mlflow_run_id = run_id
            tracker.artifact_uri = run.info.artifact_uri
            mlflow.set_tags(
                {
                    "mlflow.note.content": note,
                    "tracking_library": "sagemaker-mlflow",
                    "benchmark_complete": "false",
                    "zia_artifacts_status": "publishing",
                }
            )
            params = report_params(tracker, report)
            points = report_metric_items(report, 0, adapted, include_observations=output_dir is None)
            desired = {(p["key"], p["step"], p["value"]): p for p in points}
            missing = missing_metric_points(backend, run_id, desired)
            # Params and metrics share log-batch requests (MLflow limits in tracking.batching).
            backend.log_batch(run_id, metrics=[desired[key] for key in missing], params=params)
            dataset_logged = _log_dataset(mlflow, tracker, report, outputs_uri)
            child_runs = publish_training_rounds(
                backend,
                experiment.experiment_id,
                run_id,
                report,
                now_ms=int(time.time() * 1000),
            )
            observed = output_dir is not None and publish_sdk_observations(
                client,
                run_id,
                output_dir,
                report,
                child_runs,
            )
            # prepare_artifacts already rendered the dashboards: upload them instead of re-plotting.
            figures = _saved_figures(output_dir) or log_report_figures(report)
            uploaded = _log_artifacts(
                client,
                run_id,
                output_dir,
                exclude=(ARTIFACT_ROOT,) if observed else (),
            )
            if output_dir is not None and not uploaded:
                raise ValueError("MLflow run has no uploaded artifacts")
            pyfunc_logged = _log_pyfunc_model(mlflow, output_dir, report)
            run = client.get_run(run_id)  # artifact_uri is final once the run exists
            registry = _register_checkpoints(client, tracker, report, run, model_package_arn, pyfunc_logged)
            tags_done = {
                "benchmark_complete": "true",
                "zia_artifacts_status": "complete",
                "cb-training-rounds": str(len(child_runs)),
                "figures_logged": str(len(figures)),
                "registered_model": registry["registered_model"],
            }
            if registry["registered_model_version"]:
                tags_done["registered_model_version"] = registry["registered_model_version"]
            if outputs_uri:
                tags_done["cb-outputs-uri"] = str(outputs_uri)[:500]
        backend.log_batch(run_id, tags=tags_done)
    except Exception as error:
        raise LibraryPublishError(f"{type(error).__name__}: {error}") from error
    return {
        "published": True,
        "publisher": "sagemaker-mlflow",
        "mlflow_experiment_id": tracker.experiment_id,
        "mlflow_run_id": run_id,
        "artifact_uri": tracker.artifact_uri,
        "artifacts_uploaded": {"files": uploaded, "artifact_uri": tracker.artifact_uri},
        "figures": figures,
        "dataset_logged": dataset_logged,
        "pyfunc_model_logged": pyfunc_logged,
        "child_runs": child_runs,
        **registry,
    }


__all__ = [
    "LibraryPublishError",
    "is_sagemaker_arn",
    "library_available",
    "publish_with_library",
    "registry_names",
]
