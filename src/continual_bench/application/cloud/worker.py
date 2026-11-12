"""One AWS Batch job = one run: load the verified snapshot, evaluate, publish, exit.

The evaluator, artifact generators and metrics are the project's own (`evaluation.zia.run`,
`write_zia_artifacts`, `prediction_manifest`, `application.matrix`). This module only moves
inputs and outputs between object storage and a scratch directory, records the attempt and
writes the final manifest with a conditional create, so retries reuse a completed run and
failed attempts keep their own evidence.
"""

import io
import json
import logging
import os
import sys
import tarfile
import time
import traceback
from pathlib import Path

from continual_bench.infrastructure.object_store import (
    ObjectExists,
    get_json,
    put_json,
    sha256_path,
    split_uri,
)
from continual_bench.models.training.callbacks import GracefulTerminationHandler

from . import layout
from .contracts import RunUnit, SnapshotRef
from .records import ROWS_FILE, ensure_rows, matrix_rows, zia_rows
from .snapshots import SnapshotNotReady, materialize, verify_snapshot

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

EXIT_OK, EXIT_FAILED, EXIT_CONTRACT, EXIT_SNAPSHOT = 0, 1, 2, 3
SKIP_SUFFIXES = (".lock",)
SKIP_PREFIXES = (".pending-", ".predictions-", ".model-", ".prepare-")


class ContractError(ValueError):
    pass


class JsonFormatter(logging.Formatter):
    def __init__(self, context):
        super().__init__()
        self.context = context

    def format(self, record):
        entry = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": self.formatTime(record),
            **self.context,
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def logger_for(context) -> logging.Logger:
    logger = logging.getLogger(f"continual_bench.aws.{context.get('run_id', 'unit')}")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(context))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def ecs_task_arn(env) -> str | None:
    """Task ARN from the ECS task metadata endpoint v4 (available inside Fargate tasks)."""
    uri = env.get("ECS_CONTAINER_METADATA_URI_V4")
    if not uri:
        return None
    import urllib.request

    try:
        with urllib.request.urlopen(uri + "/task", timeout=2) as response:
            return json.loads(response.read()).get("TaskARN")
    except OSError:
        return None


def publish_directory(store, directory: Path, prefix: str) -> dict:
    files, total = 0, 0
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if (
            not path.is_file()
            or path.name.endswith(SKIP_SUFFIXES)
            or any(part.startswith(SKIP_PREFIXES) for part in relative.parts)
        ):
            continue
        key = f"{prefix}/{relative.as_posix()}"
        digest = sha256_path(path)
        existing = store.head(key)
        if existing is None:
            try:
                store.put_file(key, path, if_absent=True, sha256=digest)
            except ObjectExists:
                existing = store.head(key)
        if existing is not None and existing.sha256 != digest:
            raise ObjectExists(f"{key} already holds different attempt evidence")
        files += 1
        total += path.stat().st_size
    return {"files": files, "bytes": total, "prefix": prefix}


def model_tarball(directory: Path, metadata: dict) -> bytes:
    """SageMaker model archive: metadata + native payload at the archive root."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, source in (
            ("model-metadata.json", directory / "model-metadata.json"),
            (metadata["payload"], directory / metadata["native_path"]),
        ):
            info = archive.gettarinfo(str(source), arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    return buffer.getvalue()


def read_row_range(path, columns, start: int, count: int):
    """Rows [start, start + count) reading only the row groups that contain them (bounded memory)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables, offset = [], 0
    with pq.ParquetFile(path) as source:
        for index in range(source.metadata.num_row_groups):
            size = source.metadata.row_group(index).num_rows
            if offset + size > start and offset < start + count:
                group = source.read_row_group(index, columns=columns, use_threads=False)
                low = max(0, start - offset)
                tables.append(group.slice(low, min(size, start + count - offset) - low))
            offset += size
            if offset >= start + count:
                break
    if not tables:
        raise ValueError(f"rows {start}..{start + count} are outside {path}")
    return pa.concat_tables(tables)


def verify_native_loader(directory: Path, report: dict, data_root: Path, rows: int = 256) -> dict:
    """The project's loader must reproduce the archived holdout predictions of the final model."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from continual_bench.datasets.zia import load_manifest
    from continual_bench.models.zia import ZiaPredictor

    final = report["model_versions"][-1]
    staging = directory / "inference-check"
    staging.mkdir(exist_ok=True)
    metadata = json.loads((directory / "model-metadata.json").read_text())
    (staging / "model-metadata.json").write_bytes((directory / "model-metadata.json").read_bytes())
    (staging / metadata["payload"]).write_bytes((directory / final["file"]).read_bytes())
    predictor = ZiaPredictor(staging)
    path, manifest = load_manifest(data_root, report["dataset"], report["scope"])
    start = report["contract"]["holdout_start"]
    features = read_row_range(path, manifest["features"], start, rows).to_pandas().astype("float64")
    expected = pd.concat(
        [
            pq.read_table(part, filters=[("seq", ">=", start), ("seq", "<", start + rows)]).to_pandas()
            for part in sorted((directory / "predictions").glob("part-*.parquet"))
        ],
        ignore_index=True,
    ).sort_values("seq")
    if len(expected) != len(features) or set(expected["model_end_seq"]) != {final["end_seq"]}:
        raise ValueError("holdout predictions were not produced by the final model version")
    probabilities = predictor.predict(features)
    archived = expected[[f"p_{i}" for i in range(probabilities.shape[1])]].to_numpy()
    np.testing.assert_allclose(probabilities, archived, rtol=1e-6, atol=1e-9)
    for item in staging.iterdir():
        item.unlink()
    staging.rmdir()
    return {
        "rows": len(features),
        "max_abs_difference": float(np.max(np.abs(probabilities - archived))),
        "model_sha256": final["sha256"],
    }


def execute_zia(unit: RunUnit, store, workdir: Path, log, *, tracker=None) -> dict:
    import yaml

    from continual_bench.config import TrackingConfig
    from continual_bench.evaluation.zia import run
    from continual_bench.evaluation.zia_artifacts import tracking_report, write_zia_artifacts
    from continual_bench.evaluation.zia_predictions import prediction_manifest
    from continual_bench.evaluation.zia_publication import run_documents

    params = unit.params
    snapshot = SnapshotRef(**params["snapshot"])
    manifest = verify_snapshot(store, snapshot)
    if manifest["manifest_sha256"] != unit.contract["snapshot"]["manifest_sha256"]:
        raise SnapshotNotReady("incompatible", "snapshot manifest differs from the campaign plan")
    tracking_path = Path(os.environ.get("BENCH_PROJECT_ROOT", ".")) / params["tracking"]
    import hashlib

    if hashlib.sha256(tracking_path.read_bytes()).hexdigest() != unit.contract["tracking_sha256"]:
        raise ContractError("tracking profile differs from the campaign plan")
    settings = TrackingConfig.model_validate(yaml.safe_load(tracking_path.read_text())).model_dump()
    data_root = materialize(store, manifest, workdir / "data")
    output = workdir / "run"
    log.info("evaluation starting", extra={"fields": {"rows": manifest["rows"], "policy": params["policy"]}})
    report = run(
        data_root,
        params["dataset"],
        params["scope"],
        params["policy"],
        output,
        settings,
        tracker=tracker,
    )
    evidence = prediction_manifest(output, report)
    adapted = tracking_report(report, evidence)
    write_zia_artifacts(output, report, adapted)
    return {
        "report": report,
        "adapted": adapted,
        "output": output,
        "data_root": data_root,
        "documents": run_documents,
        "snapshot": manifest,
    }


def execute_matrix_case(store, unit: RunUnit, workdir: Path) -> dict:
    from continual_bench.application.matrix import _execute_case, _headline
    from continual_bench.config import Config, EpochConfig, TypedConfig

    evaluator = unit.params["evaluator"]
    schema = {"binary": Config, "typed": TypedConfig, "epochs": EpochConfig}[evaluator]
    config = schema.model_validate(unit.params["config"])
    output = workdir / "run"
    source = None
    if "source_sha256" in unit.params:
        from .matrix_sources import materialize_source

        source = materialize_source(
            store, unit.labels["dataset"], unit.params["source_sha256"], workdir / "source", config, evaluator
        )
    report = _execute_case(config, output, str(workdir / "cache"), None, source=source)
    return {"report": report, "headline": _headline(report), "output": output}


def flush_partial_state(
    store,
    output: Path,
    prefix: str,
    status_key: str,
    status: dict,
    *,
    started: float,
    clock=time.time,
) -> dict:
    """Persist partial run artifacts and interrupted status to S3 on Fargate SIGTERM."""
    partial_prefix = f"{prefix}/partial"
    files, total = 0, 0
    if output.exists():
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.name.endswith(SKIP_SUFFIXES):
                continue
            relative = path.relative_to(output)
            key = f"{partial_prefix}/{relative.as_posix()}"
            digest = sha256_path(path)
            store.put_file(key, path, if_absent=False, sha256=digest)
            files += 1
            total += path.stat().st_size
    payload = {
        **status,
        "state": "INTERRUPTED",
        "finished_at": clock(),
        "wall_seconds": time.monotonic() - started,
        "partial_prefix": store.uri(partial_prefix),
        "partial_files": files,
        "partial_bytes": total,
    }
    put_json(store, status_key, payload)
    return payload


def run_unit(
    store,
    unit_ref: str,
    workdir,
    *,
    env=None,
    registry=None,
    tracker=None,
    emit=None,
    clock=time.time,
) -> int:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    env = os.environ if env is None else env
    workdir = Path(workdir)
    key = split_uri(unit_ref)[1] if "://" in unit_ref else unit_ref
    unit = RunUnit(**get_json(store, key))
    job_id, attempt = env.get("AWS_BATCH_JOB_ID"), env.get("AWS_BATCH_JOB_ATTEMPT")
    attempt_key = layout.attempt_key_for(job_id, attempt)
    context = {
        "campaign_id": unit.campaign_id,
        "run_id": unit.run_id,
        "job_id": job_id or "local",
        "attempt": attempt_key,
    }
    log = logger_for(context)
    final_key = layout.final_key(unit.campaign_id, unit.run_id)
    status_key = layout.attempt_status_key(unit.campaign_id, unit.run_id, attempt_key)
    status = {
        "run_id": unit.run_id,
        "campaign_id": unit.campaign_id,
        "attempt_key": attempt_key,
        "job_id": job_id,
        "attempt": attempt,
        "task_arn": ecs_task_arn(env),
        "image_digest": env.get("BENCH_IMAGE_DIGEST"),
        "git_sha": env.get("BENCH_GIT_REVISION"),
        "started_at": clock(),
        "labels": unit.labels,
    }
    if store.head(final_key) is not None:
        try:
            analytics = ensure_rows(store, get_json(store, final_key))
        except Exception as error:  # Repair failed: report it, keep the final manifest untouched.
            log.error("analytics repair failed", exc_info=True)
            put_json(store, status_key, {**status, "state": "REUSED_ANALYTICS_FAILED", "finished_at": clock(),
                                         "diagnostic": {"error_type": type(error).__name__,
                                                        "message": str(error)[:2000]}})
            return EXIT_FAILED
        put_json(
            store,
            status_key,
            {**status, "state": "REUSED", "finished_at": clock(), "final_uri": store.uri(final_key),
             "analytics": analytics},
        )
        log.info("run already final; reusing published evidence", extra={"fields": {"analytics": analytics}})
        return EXIT_OK
    put_json(store, status_key, {**status, "state": "RUNNING"})
    started = time.monotonic()
    prefix = layout.attempt_prefix(unit.campaign_id, unit.run_id, attempt_key)
    sigterm_handler = GracefulTerminationHandler(
        on_terminate=lambda _sig: flush_partial_state(
            store, workdir / "run", prefix, status_key, status, started=started, clock=clock
        )
    )
    active_tracker = None
    try:
        with sigterm_handler:
            expected_image = unit.contract["release"]["runtime_image"]
            if env.get("BENCH_IMAGE_DIGEST") and env["BENCH_IMAGE_DIGEST"] != expected_image:
                raise ContractError("job runs an image other than the campaign release")
            if unit.kind == "zia":
                active_tracker = tracker(unit) if callable(tracker) else tracker
                outcome = execute_zia(unit, store, workdir, log, tracker=active_tracker)
            else:
                outcome = execute_matrix_case(store, unit, workdir)
        evaluation_seconds = time.monotonic() - started
        outputs_uri = store.uri(prefix)
        final = {
            "version": "run-final-v1",
            "run_id": unit.run_id,
            "campaign_id": unit.campaign_id,
            "kind": unit.kind,
            "contract_sha256": unit.contract_sha256,
            "unit_uri": store.uri(unit.unit_key),
            "labels": unit.labels,
            "attempt": {k: status[k] for k in ("attempt_key", "job_id", "attempt", "task_arn")},
            "release": unit.contract["release"],
            "image_digest": status["image_digest"],
            "outputs_uri": outputs_uri,
            "evaluation_seconds": evaluation_seconds,
        }
        if unit.kind == "zia":
            report, output = outcome["report"], outcome["output"]
            metadata, links, documents = outcome["documents"](report, outcome["adapted"], outputs_uri)
            for name, value in documents.items():
                (output / name).write_text(json.dumps(value, sort_keys=True, indent=2))
            loader = verify_native_loader(output, report, outcome["data_root"])
            (output / "model").mkdir(exist_ok=True)
            (output / "model" / "model.tar.gz").write_bytes(model_tarball(output, metadata))
            rows = zia_rows(_base(unit, attempt_key), report, store.uri(final_key), outputs_uri)
            (output / ROWS_FILE).write_text(json.dumps(rows, sort_keys=True, allow_nan=False))
            published = publish_directory(store, output, prefix)
            model_package = None
            if registry is not None:
                model_package = registry(
                    unit, report, metadata, f"{outputs_uri}/model/model.tar.gz", outcome["snapshot"]
                )
            if active_tracker is not None and hasattr(active_tracker, "publish_run_report"):
                try:
                    mlflow_info = active_tracker.publish_run_report(
                        report,
                        adapted=outcome["adapted"],
                        output_dir=output,
                        store=store,
                        outputs_uri=outputs_uri,
                        model_package_arn=(model_package or {}).get("model_package_arn"),
                    )
                    if isinstance(mlflow_info, dict) and mlflow_info.get("mlflow_run_id"):
                        links["mlflow_run_id"] = mlflow_info["mlflow_run_id"]
                        links["mlflow_experiment_id"] = mlflow_info.get("mlflow_experiment_id")
                        links["mlflow_artifact_uri"] = mlflow_info.get("artifact_uri")
                except Exception:
                    log.warning(
                        "mlflow report publication failed; run evidence in S3 is unaffected",
                        exc_info=True,
                    )
            final.update(
                {
                    "snapshot": unit.contract["snapshot"],
                    "report_uri": f"{outputs_uri}/report.json",
                    "report_sha256": sha256_path(output / "report.json"),
                    "predictions_manifest_uri": f"{outputs_uri}/predictions-manifest.json",
                    "artifacts_index_uri": f"{outputs_uri}/artifacts/data/index.json",
                    "model": {
                        "state_sha256": metadata["state_sha256"],
                        "native_uri": f"{outputs_uri}/{metadata['native_path']}",
                        "archive_uri": f"{outputs_uri}/model/model.tar.gz",
                        "loader_check": loader,
                        "model_package_arn": (model_package or {}).get("model_package_arn"),
                    },
                    "metrics": {
                        k: report["metrics"].get(k)
                        for k in (
                            "accuracy",
                            "balanced_accuracy",
                            "f1_macro",
                            "roc_auc",
                            "pr_auc",
                            "macro_roc_auc",
                            "macro_pr_auc",
                            "log_loss",
                            "brier",
                            "n",
                            "update_count",
                            "retrain_count",
                            "train_seconds",
                            "prediction_seconds",
                            "aws_cost_total_usd",
                            "scenario_total_cost",
                        )
                    },
                    "holdout": {
                        k: report["holdout"].get(k)
                        for k in (
                            "accuracy",
                            "balanced_accuracy",
                            "roc_auc",
                            "pr_auc",
                            "macro_roc_auc",
                            "macro_pr_auc",
                            "log_loss",
                            "n",
                        )
                    },
                    "decision_threshold": report["contract"].get("decision_threshold"),
                    "links": links,
                    "published": published,
                }
            )
        else:
            from continual_bench.models import default_model

            dimensions = ("model", "mode", "detector", "seed")
            coordinates = {k: unit.labels[k] for k in dimensions if k in unit.labels}
            if "model" not in coordinates and unit.params["evaluator"] == "epochs":
                coordinates["model"] = default_model(outcome["report"]["contract"]["source"]["task"])
            coordinates["seed"] = int(coordinates["seed"])
            rows = matrix_rows(
                {**_base(unit, attempt_key), **coordinates},
                outcome["headline"],
                outcome["report"],
                store.uri(final_key),
                outputs_uri,
            )
            (outcome["output"] / ROWS_FILE).write_text(json.dumps(rows, sort_keys=True, allow_nan=False))
            published = publish_directory(store, outcome["output"], prefix)
            final.update(
                {
                    "metrics": outcome["headline"],
                    "matrix_coordinates": coordinates,
                    "published": published,
                    "run_key": outcome["report"].get("run_key"),
                }
            )
        try:
            put_json(store, final_key, final, if_absent=True)
            state = "SUCCEEDED"
        except ObjectExists:
            state = "SUPERSEDED"  # Another attempt published first; this attempt's evidence stays.
            log.warning("final manifest already exists; attempt kept as superseded evidence")
        # Rows come from the winning attempt's archived file; a failure here is repaired later.
        analytics = ensure_rows(store, get_json(store, final_key))
        finished = clock()
        put_json(
            store,
            status_key,
            {
                **status,
                "state": state,
                "finished_at": finished,
                "wall_seconds": time.monotonic() - started,
                "final_uri": store.uri(final_key),
                "published_bytes": final["published"]["bytes"],
                "analytics": analytics,
            },
        )
        if emit:
            emit(
                {
                    "RunSucceeded": 1,
                    "EvaluationSeconds": evaluation_seconds,
                    "PublishedBytes": final["published"]["bytes"],
                    "HoldoutAccuracy": final.get("holdout", {}).get("accuracy"),
                },
                {
                    "Dataset": unit.labels.get("dataset", "matrix"),
                    "Scope": unit.labels.get("scope", "matrix"),
                    "Policy": unit.labels.get("policy", "matrix"),
                },
                context,
                units={"EvaluationSeconds": "Seconds", "PublishedBytes": "Bytes"},
            )
        log.info("run published", extra={"fields": {"state": state, "final_uri": store.uri(final_key)}})
        return EXIT_OK
    except Exception as error:
        code = (
            EXIT_SNAPSHOT
            if isinstance(error, SnapshotNotReady)
            else EXIT_CONTRACT
            if isinstance(error, ContractError)
            else EXIT_FAILED
        )
        diagnostic = {
            "error_type": type(error).__name__,
            "message": str(error)[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "exit_code": code,
        }
        log.error("run failed", exc_info=True, extra={"fields": {"exit_code": code}})
        if active_tracker is not None and hasattr(active_tracker, "mark_failed"):
            try:
                active_tracker.mark_failed(str(error)[:1000])
            except Exception:
                pass
        put_json(
            store,
            status_key,
            {
                **status,
                "state": "FAILED",
                "finished_at": clock(),
                "wall_seconds": time.monotonic() - started,
                "diagnostic": diagnostic,
            },
        )
        if emit:
            emit(
                {"RunFailed": 1},
                {
                    "Dataset": unit.labels.get("dataset", "matrix"),
                    "Scope": unit.labels.get("scope", "matrix"),
                    "Policy": unit.labels.get("policy", "matrix"),
                },
                context,
            )
        return code


def _base(unit: RunUnit, attempt_key: str) -> dict:
    return {
        "campaign_id": unit.campaign_id,
        "run_id": unit.run_id,
        "attempt_key": attempt_key,
        "dataset": unit.labels.get("dataset"),
        "scope": unit.labels.get("scope"),
        "policy": unit.labels.get("policy"),
        "kind": unit.kind,
        "git_sha": unit.contract["release"]["git_sha"],
    }


def sagemaker_registrar(model_registry, inference_image: str):
    """Adapter from a finished Zia run to a pending Model Registry candidate."""

    def register(unit, report, metadata, archive_uri, snapshot):
        policy_slug = report["policy"].replace("_", "-")
        group = f"continual-bench-zia-{report['dataset']}-{policy_slug}" + (
            "-smoke" if report["scope"] == "smoke" else ""
        )
        model_registry.ensure_group(
            group,
            f"Zia {report['dataset']} {report['policy']} candidates",
            {"dataset": report["dataset"], "policy": report["policy"]},
        )
        return model_registry.register(
            group=group,
            run_id=unit.run_id,
            image=inference_image,
            model_data_uri=archive_uri,
            description=f"run {unit.run_id}",
            metadata={
                "campaign_id": unit.campaign_id,
                "snapshot_id": snapshot["snapshot_id"],
                "dataset_sha256": report["contract"]["input"]["sha256"],
                "model_sha256": metadata["state_sha256"],
                "report_uri": archive_uri.rsplit("/model/", 1)[0] + "/report.json",
                "git_sha": unit.contract["release"]["git_sha"],
            },
        )

    return register


def unit_from_store(store, campaign_id, run_id) -> RunUnit:
    return RunUnit(**get_json(store, layout.unit_key(campaign_id, run_id)))
