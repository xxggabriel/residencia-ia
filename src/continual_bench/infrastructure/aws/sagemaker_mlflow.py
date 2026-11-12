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

import re
import time
from pathlib import Path
from typing import Any

from .mlflow_tracking import _SKIP_PREFIXES, _SKIP_SUFFIXES, report_metric_items, report_params

BATCH_METRICS = 1000
BATCH_PARAMS = 100
MODEL_REQUIREMENTS = ["numpy", "pandas", "scipy", "xgboost"]
PYFUNC_LOADER = "continual_bench.models.zia"


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


def _chunks(items: list, size: int):
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


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


def _log_artifacts(mlflow, output_dir: Path | None) -> int:
    if output_dir is None or not Path(output_dir).is_dir():
        return 0
    root = Path(output_dir)
    uploaded = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(_SKIP_SUFFIXES):
            continue
        relative = path.relative_to(root)
        if any(part.startswith(_SKIP_PREFIXES) for part in relative.parts):
            continue
        # artifacts/plots -> plots/, artifacts/data -> data/ (same layout as every other publisher)
        parts = (
            relative.parts[1:]
            if relative.parts[0] == "artifacts" and len(relative.parts) > 1
            else relative.parts
        )
        parent = "/".join(parts[:-1])
        mlflow.log_artifact(str(path), artifact_path=parent or None)
        uploaded += 1
    return uploaded


def _ensure_registered_model(client, name: str, description: str, tags: dict[str, str]) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(name, tags=tags, description=description)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise


def _register_checkpoints(client, tracker, report, run, model_package_arn) -> dict[str, Any]:
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
            "prequential_accuracy": str(metrics.get("accuracy", "")),
            "holdout_accuracy": str(holdout.get("accuracy", "")),
        }
        if final and model_package_arn:
            tags["sagemaker_model_package_arn"] = str(model_package_arn)
        created_version = client.create_model_version(
            policy_name,
            f"{base}/{str(version.get('file') or 'model/model.tar.gz').lstrip('/')}",
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
        fam = client.create_model_version(
            family_name,
            f"{base}/{str(last.get('file') or 'model/model.tar.gz').lstrip('/')}",
            run_id=run.info.run_id,
            tags={
                "policy": common["policy"],
                "ordinal": "final",
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
    versions = report.get("model_versions") or []
    relative = str((versions[-1].get("file") if versions else None) or "model/model.tar.gz")
    path = Path(output_dir) / relative
    if not path.is_file():
        return False
    mlflow.pyfunc.log_model(
        artifact_path="model_pyfunc",
        loader_module=PYFUNC_LOADER,
        data_path=str(path),
        pip_requirements=MODEL_REQUIREMENTS,
    )
    return True


def publish_with_library(
    tracker,
    report: dict[str, Any],
    *,
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
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    from continual_bench.infrastructure.mlflow_figures import log_report_figures

    try:
        mlflow.set_tracking_uri(tracker.tracking_uri)
        client = MlflowClient()
        experiment = mlflow.set_experiment(tracker.experiment_name)
        tags = {t["key"]: t["value"] for t in tracker.mlflow_tags}
        found = client.search_runs(
            [experiment.experiment_id],
            f"tags.zia_run_key = '{tracker.run_id}'",
            max_results=2,
        )
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
        ) as run:
            run_id = run.info.run_id
            mlflow.set_tags({**tags, "mlflow.note.content": note, "tracking_library": "sagemaker-mlflow"})
            params = report_params(tracker, report)
            for chunk in _chunks(list(params.items()), BATCH_PARAMS):
                mlflow.log_params(dict(chunk))
            points = report_metric_items(report, int(time.time() * 1000))
            for chunk in _chunks(points, BATCH_METRICS):
                client.log_batch(
                    run_id,
                    metrics=[Metric(p["key"], p["value"], p["timestamp"], p["step"]) for p in chunk],
                    synchronous=True,
                )
            dataset_logged = _log_dataset(mlflow, tracker, report, outputs_uri)
            figures = log_report_figures(report)  # mlflow.log_figure on this active run
            uploaded = _log_artifacts(mlflow, output_dir)
            pyfunc_logged = False
            try:
                pyfunc_logged = _log_pyfunc_model(mlflow, output_dir, report)
            except Exception as error:  # the registry checkpoints below are the source of truth
                tracker._record_error("log_pyfunc_model", error)
            run = client.get_run(run_id)  # artifact_uri is final once the run exists
            registry = _register_checkpoints(client, tracker, report, run, model_package_arn)
            tags_done = {
                "benchmark_complete": "true",
                "zia_artifacts_status": "complete",
                "figures_logged": str(len(figures)),
                "registered_model": registry["registered_model"],
            }
            if registry["registered_model_version"]:
                tags_done["registered_model_version"] = registry["registered_model_version"]
            if outputs_uri:
                tags_done["cb-outputs-uri"] = str(outputs_uri)[:500]
            mlflow.set_tags(tags_done)
    except Exception as error:
        raise LibraryPublishError(f"{type(error).__name__}: {error}") from error
    tracker.experiment_id = str(experiment.experiment_id)
    tracker.mlflow_run_id = run_id
    tracker.artifact_uri = run.info.artifact_uri
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
        **registry,
    }


__all__ = [
    "LibraryPublishError",
    "is_sagemaker_arn",
    "library_available",
    "publish_with_library",
    "registry_names",
]
