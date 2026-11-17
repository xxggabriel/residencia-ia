"""Load the verified snapshot, evaluate, and publish a campaign unit.

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
from contextlib import ExitStack, nullcontext
from pathlib import Path

from continual_bench.infrastructure.object_store import (
    ObjectExists,
    get_json,
    put_json,
    sha256_bytes,
    sha256_path,
    split_uri,
)
from continual_bench.models.training.callbacks import GracefulTerminationHandler

from . import layout
from .checkpoints import (
    SKIP_SUFFIXES,
    CampaignCancelled,
    CheckpointError,
    SpotAttempt,
    SpotCheckpoint,
    SpotGrantError,
    output_files,
)
from .contracts import ReleaseManifest, RunUnit, SnapshotRef, run_identity
from .records import ROWS_FILE, ensure_rows, matrix_rows, zia_rows
from .snapshots import SnapshotNotReady, materialize, verify_snapshot

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

EXIT_OK, EXIT_FAILED, EXIT_CONTRACT, EXIT_SNAPSHOT = 0, 1, 2, 3
EXIT_INTERRUPTED, EXIT_CANCELLED = 75, 76


class ContractError(ValueError):
    pass


class TrackingPublicationError(RuntimeError):
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


def publish_directory(store, directory: Path, prefix: str, *, check_stop=None) -> dict:
    files, total = 0, 0
    for relative, path in output_files(directory):
        if check_stop is not None:
            check_stop()
        key = f"{prefix}/{relative}"
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


def zia_snapshot(store, unit):
    manifest = verify_snapshot(store, SnapshotRef(**unit.params["snapshot"]))
    if manifest["manifest_sha256"] != unit.contract["snapshot"]["manifest_sha256"]:
        raise SnapshotNotReady("incompatible", "snapshot manifest differs from the campaign plan")
    return manifest


def execute_zia(
    unit: RunUnit, store, workdir: Path, log, *, stop_requested=None, progress=None, checkpoint=None,
    snapshot_manifest=None,
) -> dict:
    import yaml

    from continual_bench.config import TrackingConfig
    from continual_bench.evaluation.zia import run
    from continual_bench.evaluation.zia_artifacts import tracking_report, write_zia_artifacts
    from continual_bench.evaluation.zia_predictions import prediction_manifest
    from continual_bench.evaluation.zia_publication import run_documents

    params = unit.params
    manifest = snapshot_manifest if snapshot_manifest is not None else zia_snapshot(store, unit)
    tracking_path = Path(os.environ.get("BENCH_PROJECT_ROOT", ".")) / params["tracking"]
    import hashlib

    if hashlib.sha256(tracking_path.read_bytes()).hexdigest() != unit.contract["tracking_sha256"]:
        raise ContractError("tracking profile differs from the campaign plan")
    settings = TrackingConfig.model_validate(yaml.safe_load(tracking_path.read_text())).model_dump()
    data_root = materialize(store, manifest, workdir / "data")
    if stop_requested is not None and stop_requested():
        raise InterruptedError("Zia stopped before evaluation after SIGTERM")
    output = workdir / "run"
    log.info("evaluation starting", extra={"fields": {"rows": manifest["rows"], "policy": params["policy"]}})
    report = run(
        data_root,
        params["dataset"],
        params["scope"],
        params["policy"],
        output,
        settings,
        stop_requested=stop_requested if checkpoint is None else None,
        progress=progress,
        **({"checkpoint": checkpoint} if checkpoint is not None else {}),
    )
    if stop_requested is not None and stop_requested():
        raise InterruptedError("Zia evaluation stopped after SIGTERM")
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


def restore_benchmark_evidence(store, final, directory, *, report_path="report.json", check_stop=None):
    """Materialize immutable benchmark evidence after an interrupted publication."""
    prefix = final["published"]["prefix"] + "/"
    files = 0
    for key in store.list(prefix):
        if check_stop is not None:
            check_stop()
        relative = key.removeprefix(prefix)
        if relative == "status.json" or relative.startswith("partial/"):
            continue
        store.download(key, directory / relative)
        files += 1
    if files != final["published"]["files"]:
        raise TrackingPublicationError("The archived attempt has missing benchmark artifacts")
    if sha256_path(directory / report_path) != final["report_sha256"]:
        raise TrackingPublicationError("The archived report differs from the publication manifest")
    return directory


def start_zia_tracking(tracker, attempt_key, log, *, spot=False):
    start_live = getattr(tracker, "start_live", None)
    if not callable(start_live):
        return False
    bind_attempt = getattr(tracker, "bind_attempt", None)
    if callable(bind_attempt) and not spot:
        bind_attempt(attempt_key)
    try:
        start_live()
    except Exception:
        log.warning("could not start live Zia MLflow tracking", exc_info=True)
    return True


def stop_zia_tracking(tracker, log):
    mark_killed = getattr(tracker, "mark_killed", None)
    try:
        if callable(mark_killed):
            mark_killed()
        elif callable(finish := getattr(tracker, "finish", None)):
            finish("KILLED")
    except Exception:
        log.warning("could not mark Zia MLflow run as killed", exc_info=True)


def complete_zia_tracking(tracker, store, final, output, report, adapted):
    if hasattr(tracker, "bind_attempt") and not final["attempt"].get("instance_id"):
        tracker.bind_attempt(final["attempt"]["attempt_key"])
    result = tracker.publish_run_report(
        report,
        adapted=adapted,
        output_dir=output,
        store=store,
        outputs_uri=final["outputs_uri"],
        model_package_arn=final["model"]["model_package_arn"],
    ) or {}
    run_id = result.get("mlflow_run_id")
    artifacts = result.get("artifacts_uploaded") or {}
    children = result.get("child_runs") or []
    expected = [fit["fit_id"] for fit in report.get("training_history", [])]
    if (
        result.get("published") is not True
        or not run_id
        or artifacts.get("files", 0) < final["published"]["files"]
        or not expected
        or [child["fit_id"] for child in children] != expected
    ):
        errors = result.get("errors") or []
        details = ": " + " | ".join(errors[:3]) if errors else ""
        raise TrackingPublicationError(f"Incomplete MLflow publication{details}")
    links = final["links"]
    links.update(
        mlflow_run_id=run_id,
        mlflow_experiment_id=result.get("mlflow_experiment_id"),
        mlflow_artifact_uri=result.get("artifact_uri"),
    )
    links.update({key: result[key] for key in (
        "registered_model", "registered_model_version",
        "registered_family_model", "registered_family_version",
    ) if result.get(key)})
    final["tracking"] = {
        "required": True,
        "complete": True,
        "mlflow_run_id": run_id,
        "child_runs": children,
        "artifacts": artifacts,
    }


def execute_matrix_case(store, unit: RunUnit, workdir: Path, *, progress=None, checkpoint=None) -> dict:
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
    report = _execute_case(
        config, output, str(workdir / "cache"), None, source=source, progress=progress,
        **({"checkpoint": checkpoint} if checkpoint is not None else {}),
    )
    directory = output / evaluator / report["run_key"]
    report_path = directory / ("epoch-report.json" if evaluator == "epochs" else "report.json")
    return {"report": report, "headline": _headline(report), "output": output, "report_path": report_path}


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


def verify_spot_inputs(store, key, unit, env):
    expected_id, contract_hash = run_identity(unit.campaign_id, unit.contract)
    if (
        unit.run_id != expected_id
        or unit.contract_sha256 != contract_hash
        or unit.unit_key != key
        or unit.unit_key != layout.unit_key(unit.campaign_id, unit.run_id)
        or unit.params != unit.contract.get("params")
        or unit.kind != unit.contract.get("kind")
    ):
        raise ContractError("unit document differs from its campaign contract")
    revision = unit.contract["release"]["git_sha"]
    release = ReleaseManifest(**get_json(store, layout.release_key(revision)))
    image = release.images["runtime"]
    if (
        env.get("BENCH_GIT_REVISION") != revision
        or release.git_sha != revision
        or image.digest != unit.contract["release"]["runtime_image"]
        or env.get("BENCH_IMAGE_DIGEST") != image.digest
        or env.get("BENCH_IMAGE_URI", image.uri) != image.uri
    ):
        raise ContractError("instance image or revision differs from the campaign release")
    if unit.kind == "zia":
        manifest = zia_snapshot(store, unit)
        if manifest["prepared"]["sha256"] != unit.contract["snapshot"]["prepared_sha256"]:
            raise SnapshotNotReady("incompatible", "prepared dataset differs from the campaign plan")
        return manifest
    if "source_sha256" in unit.params:
        from .matrix_sources import verify_source

        verify_source(store, unit.labels["dataset"], unit.params["source_sha256"])
    return None


def reuse_final(store, final_key, status_key, status, log, clock):
    try:
        analytics = ensure_rows(store, get_json(store, final_key))
    except Exception as error:  # Repair failed: report it, keep the final manifest untouched.
        log.error("analytics repair failed", exc_info=True)
        put_json(
            store, status_key,
            {**status, "state": "REUSED_ANALYTICS_FAILED", "finished_at": clock(),
             "diagnostic": {"error_type": type(error).__name__, "message": str(error)[:2000]}},
        )
        return EXIT_FAILED
    put_json(
        store,
        status_key,
        {**status, "state": "REUSED", "finished_at": clock(), "final_uri": store.uri(final_key),
         "analytics": analytics},
    )
    log.info("run already final; reusing published evidence", extra={"fields": {"analytics": analytics}})
    return EXIT_OK


def stage_publication(store, output, prefix, checkpoint, *, check_stop=None):
    if checkpoint is None:
        return {"published": publish_directory(store, output, prefix, check_stop=check_stop)}
    files = checkpoint.archive_files(private=False, check_stop=check_stop)
    summary = {"files": len(files), "bytes": sum(file["bytes"] for file in files.values()), "prefix": prefix}
    return {"publication_files": files, "published": summary}


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
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    env = os.environ if env is None else env
    workdir = Path(workdir)
    key = split_uri(unit_ref)[1] if "://" in unit_ref else unit_ref
    unit = RunUnit(**get_json(store, key))
    instance_id = env.get("BENCH_EC2_INSTANCE_ID")
    if instance_id:
        job_id, attempt = instance_id, int(env["BENCH_EC2_ATTEMPT"])
    else:
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
    if not instance_id and store.head(final_key) is not None:
        return reuse_final(store, final_key, status_key, status, log, clock)
    started = time.monotonic()
    prefix = layout.attempt_prefix(unit.campaign_id, unit.run_id, attempt_key)
    active_tracker = None
    matrix_tracker = None
    pending_key = f"{layout.run_prefix(unit.campaign_id, unit.run_id)}/pending-publication.json"
    tracking_uri = env.get("MLFLOW_TRACKING_URI")
    tracking_required = bool(tracking_uri or (unit.kind == "zia" and tracker is not None))
    publication_required = bool(
        instance_id or tracking_required or (unit.kind == "zia" and registry is not None)
    )
    resuming = False
    live_attempt = None
    spot = None
    checkpoint = None
    publishing = False

    def on_terminate(_signal):
        if instance_id or unit.kind == "zia":
            return  # Zia stops at the next training step before flushing its last recorded metrics.
        flush_partial_state(store, workdir / "run", prefix, status_key, status, started=started, clock=clock)
        if matrix_tracker is not None:
            matrix_tracker.finish("KILLED")

    sigterm_handler = GracefulTerminationHandler(on_terminate=on_terminate)
    zia_signal_scope = ExitStack()

    def check_stop():
        if spot:
            spot.check_publication()
        if sigterm_handler.received:
            raise InterruptedError("Zia execution stopped after SIGTERM")

    def stop_requested():
        return bool(spot.reason() if spot else sigterm_handler.received)

    def bind_spot_tracking(active):
        if active is not None and spot:
            active.bind_spot_attempt(
                attempt_key, instance_id=instance_id, attempt=attempt,
                resumed_checkpoint=status.get("resumed_checkpoint"),
                previous_attempt_key=status.get("previous_attempt_key"),
                recovery_reason=status.get("recovery_reason"),
                log_group=status.get("log_group"),
                log_stream=status.get("log_stream"),
            )

    def record_spot_tracking(state, reason=None):
        if active_tracker is not None and spot:
            return active_tracker.record_spot_attempt(
                state, reason=reason, checkpoint=checkpoint.receipt if checkpoint else None,
            )
        return True

    try:
        if instance_id or unit.kind == "zia":
            zia_signal_scope.enter_context(sigterm_handler)
        with nullcontext() if instance_id or unit.kind == "zia" else sigterm_handler:
            snapshot_manifest = None
            if instance_id:
                spot = SpotAttempt(
                    store, unit, instance_id, attempt, env.get("BENCH_SPOT_NOTICE_FILE"),
                    stop_requested=lambda: sigterm_handler.received,
                )
                try:
                    grant = spot.acquire()
                except SpotGrantError:
                    log.error("worker could not acquire exclusive EC2 attempt", exc_info=True)
                    return EXIT_CONTRACT
                except CampaignCancelled:
                    return EXIT_CANCELLED
                except InterruptedError:
                    return EXIT_INTERRUPTED
                status["instance_id"] = instance_id
                status.update({
                    key: grant.get(key)
                    for key in ("log_group", "log_stream", "previous_attempt_key", "recovery_reason")
                })
                put_json(store, status_key, {**status, "state": "RUNNING"})
                snapshot_manifest = verify_spot_inputs(store, key, unit, env)
                if store.head(final_key) is not None:
                    check_stop()
                    return reuse_final(store, final_key, status_key, status, log, clock)
                checkpoint = SpotCheckpoint(
                    store, unit, workdir / "run", attempt, requested=stop_requested, logger=log,
                    interval=float(env.get("BENCH_CHECKPOINT_INTERVAL_SECONDS", "60")),
                )
                if store.head(pending_key) is not None:
                    status["resumed_checkpoint"] = store.uri(pending_key)
                else:
                    checkpoint.prepare_restore()
                    status["resumed_checkpoint"] = checkpoint.resumed
            put_json(store, status_key, {**status, "state": "RUNNING"})
            expected_image = unit.contract["release"]["runtime_image"]
            if env.get("BENCH_IMAGE_DIGEST") and env["BENCH_IMAGE_DIGEST"] != expected_image:
                raise ContractError("job runs an image other than the campaign release")
            if unit.kind == "zia":
                active_tracker = tracker(unit) if callable(tracker) else tracker
                if tracking_required and not hasattr(active_tracker, "publish_run_report"):
                    raise TrackingPublicationError("MLflow is configured but the worker has no publisher")
                bind_spot_tracking(active_tracker)
                resuming = publication_required and store.head(pending_key) is not None
                if not resuming:
                    if start_zia_tracking(active_tracker, attempt_key, log, spot=bool(spot)):
                        live_attempt = attempt_key
                    callback = getattr(active_tracker, "progress", None)

                    def live_progress(step, metrics):
                        if stop_requested():
                            return
                        try:
                            callback(step, metrics)
                        except Exception:
                            log.warning("could not publish Zia progress to MLflow", exc_info=True)

                    check_stop()
                    options = (
                        {"checkpoint": checkpoint, "snapshot_manifest": snapshot_manifest} if spot else {}
                    )
                    outcome = execute_zia(
                        unit, store, workdir, log, stop_requested=stop_requested,
                        progress=live_progress if callable(callback) else None,
                        **options,
                    )
                    check_stop()
            else:
                resuming = publication_required and store.head(pending_key) is not None
                options = {"checkpoint": checkpoint} if spot else {}
                if tracking_required and not resuming:
                    from continual_bench.infrastructure.aws.matrix_tracking import MatrixMlflowRun

                    matrix_tracker = MatrixMlflowRun(unit, attempt_key, job_id, tracking_uri, log)
                    bind_spot_tracking(matrix_tracker)
                    if spot:
                        matrix_tracker.start_live()
                    else:
                        matrix_tracker.start()
                    active_tracker = matrix_tracker
                    outcome = execute_matrix_case(
                        store, unit, workdir, progress=matrix_tracker.metrics, **options,
                    )
                elif not resuming:
                    outcome = execute_matrix_case(store, unit, workdir, **options)
                if stop_requested():
                    raise InterruptedError("Matrix evaluation stopped after SIGTERM")
        publishing = True
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
        if spot:
            final["attempt"].update({name: status.get(name) for name in (
                "instance_id", "log_group", "log_stream", "previous_attempt_key", "recovery_reason",
                "resumed_checkpoint",
            )})
            final["attempt"]["checkpoint"] = checkpoint.receipt
        if unit.kind == "zia" and not resuming:
            report, output = outcome["report"], outcome["output"]
            metadata, links, documents = outcome["documents"](report, outcome["adapted"], outputs_uri)
            for name, value in documents.items():
                (output / name).write_text(json.dumps(value, sort_keys=True, indent=2))
            check_stop()
            loader = verify_native_loader(output, report, outcome["data_root"])
            (output / "model").mkdir(exist_ok=True)
            (output / "model" / "model.tar.gz").write_bytes(model_tarball(output, metadata))
            rows = zia_rows(_base(unit, attempt_key), report, store.uri(final_key), outputs_uri)
            (output / ROWS_FILE).write_text(json.dumps(rows, sort_keys=True, allow_nan=False))
            if hasattr(active_tracker, "prepare_artifacts"):
                active_tracker.prepare_artifacts(report, output)
            check_stop()
            final.update(
                {
                    "snapshot": unit.contract["snapshot"],
                    "report_uri": f"{outputs_uri}/report.json",
                    "report_sha256": sha256_path(output / "report.json"),
                    "training_history_uri": f"{outputs_uri}/training-history.json",
                    "predictions_manifest_uri": f"{outputs_uri}/predictions-manifest.json",
                    "artifacts_index_uri": f"{outputs_uri}/artifacts/data/index.json",
                    "model": {
                        "state_sha256": metadata["state_sha256"],
                        "native_uri": f"{outputs_uri}/{metadata['native_path']}",
                        "archive_uri": f"{outputs_uri}/model/model.tar.gz",
                        "loader_check": loader,
                        "model_package_arn": None,
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
                            "f1_macro",
                            "roc_auc",
                            "pr_auc",
                            "macro_roc_auc",
                            "macro_pr_auc",
                            "log_loss",
                            "brier",
                            "n",
                        )
                    },
                    "decision_threshold": report["contract"].get("decision_threshold"),
                    "links": links,
                }
            )
            final.update(stage_publication(store, output, prefix, checkpoint, check_stop=check_stop))
            adapted = outcome["adapted"]
            final["tracking"] = {"required": tracking_required, "complete": False}
        elif unit.kind != "zia" and not resuming:
            from continual_bench.models import default_model

            report = outcome["report"]
            report_path = outcome["report_path"]
            directory = report_path.parent
            relative_report = report_path.relative_to(outcome["output"]).as_posix()
            relative_evidence = directory.relative_to(outcome["output"]).as_posix()
            dimensions = ("model", "mode", "detector", "seed")
            coordinates = {k: unit.labels[k] for k in dimensions if k in unit.labels}
            if "model" not in coordinates and unit.params["evaluator"] == "epochs":
                coordinates["model"] = default_model(report["contract"]["source"]["task"])
            coordinates["seed"] = int(coordinates["seed"])
            rows = matrix_rows(
                {**_base(unit, attempt_key), **coordinates},
                outcome["headline"],
                report,
                store.uri(final_key),
                outputs_uri,
            )
            (outcome["output"] / ROWS_FILE).write_text(json.dumps(rows, sort_keys=True, allow_nan=False))
            final.update(
                {
                    "metrics": outcome["headline"],
                    "matrix_coordinates": coordinates,
                    "run_key": report["run_key"],
                    "report_uri": f"{outputs_uri}/{relative_report}",
                    "report_sha256": sha256_path(report_path),
                    "report_path": relative_report,
                    "artifacts_index_uri": f"{outputs_uri}/{relative_evidence}/artifacts/data/index.json",
                    "tracking": {"required": tracking_required, "complete": False},
                }
            )
            final.update(stage_publication(
                store, outcome["output"], prefix, checkpoint, check_stop=check_stop if spot else None,
            ))
            if "training_history" in report:
                final["training_history_uri"] = f"{outputs_uri}/{relative_evidence}/training-history.json"
        if publication_required and not resuming:
            if spot:
                spot.check_publication(strict=True)
            try:
                pending = put_json(store, pending_key, final, if_absent=True)
                if spot:
                    log.info("publication receipt confirmed", extra={"fields": {"receipt": pending.uri}})
            except ObjectExists:
                resuming = True
        publication_root = None
        if spot:
            if resuming:
                data = store.get_bytes(pending_key)
                if store.head(pending_key).sha256 != sha256_bytes(data):
                    raise CheckpointError("Publication receipt checksum could not be verified")
                final = json.loads(data)
                if final["contract_sha256"] != unit.contract_sha256:
                    raise ContractError("Archived evidence does not match the unit contract")
                publication_root = workdir / "publication"
                publication_root.mkdir(parents=True, exist_ok=True)
                checkpoint.restore_files(
                    final["publication_files"], publication_root, check_stop=check_stop,
                )
            else:
                publication_root = outcome["output"]
            final.pop("publication_files")
            published = final["published"]
            archived_prefix = layout.attempt_prefix(
                unit.campaign_id, unit.run_id, final["attempt"]["attempt_key"],
            )
            if published["prefix"] != archived_prefix or final["outputs_uri"] != store.uri(archived_prefix):
                raise ContractError("Archived evidence is outside its campaign attempt")
            replicated = publish_directory(
                store, publication_root, archived_prefix, check_stop=check_stop,
            )
            if replicated != published:
                raise TrackingPublicationError("Restored files differ from the publication receipt")
        if unit.kind == "zia" and publication_required:
            check_stop()
            if resuming:
                from continual_bench.evaluation.zia_artifacts import tracking_report
                from continual_bench.evaluation.zia_predictions import prediction_manifest

                if not spot:
                    final = get_json(store, pending_key)
                if final["contract_sha256"] != unit.contract_sha256:
                    raise ContractError("Archived attempt does not match the unit contract")
                archived_attempt = final["attempt"]["attempt_key"]
                if spot:
                    start_zia_tracking(active_tracker, attempt_key, log, spot=True)
                elif live_attempt != archived_attempt:
                    if live_attempt is not None:
                        stop_zia_tracking(active_tracker, log)
                        if callable(tracker):
                            active_tracker = tracker(unit)
                    start_zia_tracking(active_tracker, archived_attempt, log)
                output = publication_root or restore_benchmark_evidence(
                    store, final, workdir / "publication", check_stop=check_stop
                )
                report = json.loads((output / "report.json").read_text())
                metadata = json.loads((output / "model-metadata.json").read_text())
                adapted = tracking_report(report, prediction_manifest(output, report))
                evaluation_seconds = final["evaluation_seconds"]
                check_stop()
            if registry is not None:
                if spot:
                    spot.check_publication(strict=True)
                model_package = registry(
                    unit, report, metadata, final["model"]["archive_uri"],
                    {"snapshot_id": final["snapshot"]["id"]},
                )
                if not model_package or not model_package.get("model_package_arn"):
                    raise RuntimeError("SageMaker registry returned no model package ARN")
                final["model"]["model_package_arn"] = model_package["model_package_arn"]
            if tracking_required:
                if spot:
                    spot.check_publication(strict=True)
                else:
                    check_stop()
                complete_zia_tracking(active_tracker, store, final, output, report, adapted)
                check_stop()
        elif unit.kind != "zia" and tracking_required:
            from continual_bench.infrastructure.aws.matrix_tracking import MatrixMlflowRun

            if resuming:
                if not spot:
                    final = get_json(store, pending_key)
                if final["contract_sha256"] != unit.contract_sha256:
                    raise ContractError("Archived attempt does not match the unit contract")
                archived = final["attempt"]
                if matrix_tracker is not None and archived["attempt_key"] != attempt_key:
                    matrix_tracker.finish("KILLED")
                root = publication_root or restore_benchmark_evidence(
                    store, final, workdir / "publication", report_path=final["report_path"]
                )
                report_path = root / final["report_path"]
                directory = report_path.parent
                report = json.loads(report_path.read_text())
                evaluation_seconds = final["evaluation_seconds"]
                if spot:
                    matrix_tracker = MatrixMlflowRun(unit, attempt_key, job_id, tracking_uri, log)
                    bind_spot_tracking(matrix_tracker)
                    matrix_tracker.start_live()
                else:
                    matrix_tracker = MatrixMlflowRun(
                        unit, archived["attempt_key"], archived["job_id"], tracking_uri, log,
                    )
                active_tracker = matrix_tracker
            try:
                if spot:
                    spot.check_publication(strict=True)
                receipt = matrix_tracker.publish_training_evidence(
                    report, directory, store=store, headline=final["metrics"],
                    outputs_uri=final["outputs_uri"],
                )
                matrix_tracker.finish_publication()
            except InterruptedError:
                raise
            except Exception as error:
                raise TrackingPublicationError(f"Matrix MLflow publication failed: {error}") from error
            final["tracking"] = {"required": True, "complete": True, **receipt}
            final["links"] = {
                "mlflow_run_id": receipt["mlflow_run_id"],
                "mlflow_experiment_id": receipt["mlflow_experiment_id"],
                "mlflow_artifact_uri": receipt["artifact_uri"],
            }

        if spot:
            spot.check_publication(strict=True)
            analytics = ensure_rows(store, final)
            if analytics not in {"present", "repaired"}:
                raise TrackingPublicationError("Could not publish benchmark analytics")
            if not record_spot_tracking("SUCCEEDED"):
                raise TrackingPublicationError("Could not record the completed EC2 attempt in MLflow")
            final["publication_attempt"] = {
                "attempt_key": attempt_key, "instance_id": instance_id, "attempt": attempt,
            }
            spot.check_publication(strict=True)
        try:
            put_json(store, final_key, final, if_absent=True)
            state = "SUCCEEDED"
        except ObjectExists:
            state = "SUPERSEDED"  # Another attempt published first; this attempt's evidence stays.
            log.warning("final manifest already exists; attempt kept as superseded evidence")
        # Rows come from the winning attempt's archived file; a failure here is repaired later.
        if not spot:
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
                **({"checkpoint": checkpoint.receipt} if checkpoint else {}),
            },
        )
        holdout_m = final.get("holdout", {})
        stream_m = final.get("metrics", {})
        holdout_ap = holdout_m.get("pr_auc")
        if holdout_ap is None:
            holdout_ap = holdout_m.get("macro_pr_auc")
        stream_ap = stream_m.get("pr_auc")
        if stream_ap is None:
            stream_ap = stream_m.get("macro_pr_auc")
        if emit:
            emit(
                {
                    "RunSucceeded": 1,
                    "EvaluationSeconds": evaluation_seconds,
                    "PublishedBytes": final["published"]["bytes"],
                    "HoldoutAccuracy": holdout_m.get("accuracy"),
                    "HoldoutBalancedAccuracy": holdout_m.get("balanced_accuracy"),
                    "HoldoutPRAUC": holdout_ap,
                    "HoldoutMacroF1": holdout_m.get("f1_macro"),
                    "StreamBalancedAccuracy": stream_m.get("balanced_accuracy"),
                    "StreamPRAUC": stream_ap,
                    "StreamMacroF1": stream_m.get("f1_macro"),
                },
                {
                    "Dataset": unit.labels.get("dataset", "matrix"),
                    "Scope": unit.labels.get("scope", "matrix"),
                    "Policy": unit.labels.get("policy", "matrix"),
                },
                context,
                units={"EvaluationSeconds": "Seconds", "PublishedBytes": "Bytes"},
            )
        log.info(
            "run published",
            extra={
                "fields": {
                    "state": state,
                    "final_uri": store.uri(final_key),
                    "dataset": unit.labels.get("dataset"),
                    "policy": unit.labels.get("policy"),
                    "scope": unit.labels.get("scope"),
                    "holdout_accuracy": holdout_m.get("accuracy"),
                    "holdout_balanced_accuracy": holdout_m.get("balanced_accuracy"),
                    "holdout_pr_auc": holdout_ap,
                    "holdout_macro_f1": holdout_m.get("f1_macro"),
                    "stream_balanced_accuracy": stream_m.get("balanced_accuracy"),
                    "stream_pr_auc": stream_ap,
                    "stream_macro_f1": stream_m.get("f1_macro"),
                }
            },
        )
        return EXIT_OK
    except Exception as error:
        reason = None
        if spot:
            try:
                reason = spot.reason()
            except Exception:
                log.warning("could not verify the current EC2 interruption state", exc_info=True)
        if isinstance(error, CampaignCancelled):
            reason = "cancelled"
        code = (
            EXIT_CANCELLED
            if reason == "cancelled" or isinstance(error, CampaignCancelled)
            else EXIT_INTERRUPTED
            if reason
            else EXIT_SNAPSHOT
            if isinstance(error, SnapshotNotReady)
            else EXIT_CONTRACT
            if isinstance(error, (ContractError, CheckpointError, SpotGrantError))
            else EXIT_FAILED
        )
        diagnostic = {
            "error_type": type(error).__name__,
            "message": str(error)[:2000],
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "exit_code": code,
        }
        log.error("run failed", exc_info=True, extra={"fields": {"exit_code": code}})
        if (
            active_tracker is not None and hasattr(active_tracker, "mark_failed")
            and not spot and not sigterm_handler.received
        ):
            try:
                active_tracker.mark_failed(str(error)[:1000])
            except Exception:
                pass
        if not spot and unit.kind == "zia" and sigterm_handler.received:
            stop_zia_tracking(active_tracker, log)
            flush_partial_state(
                store, workdir / "run", prefix, status_key, status, started=started, clock=clock
            )
            return code
        failure_state = {EXIT_CANCELLED: "CANCELLED", EXIT_INTERRUPTED: "INTERRUPTED"}.get(code, "FAILED")
        if code == EXIT_FAILED and publication_required and (publishing or store.head(pending_key)):
            failure_state = (
                "TRACKING_FAILED" if isinstance(error, TrackingPublicationError) else "PUBLICATION_FAILED"
            )
        put_json(
            store,
            status_key,
            {
                **status,
                "state": failure_state,
                "finished_at": clock(),
                "wall_seconds": time.monotonic() - started,
                "diagnostic": diagnostic,
                **({"interruption_reason": reason, "checkpoint": checkpoint.receipt if checkpoint else None}
                   if spot else {}),
            },
        )
        if spot and not isinstance(error, SpotGrantError):
            try:
                record_spot_tracking(failure_state, reason=reason or str(error)[:1000])
                if code == EXIT_CANCELLED:
                    stop_zia_tracking(active_tracker, log)
                elif failure_state == "FAILED" and callable(
                    mark_failed := getattr(active_tracker, "mark_failed", None)
                ):
                    mark_failed(str(error)[:1000])
            except Exception:
                log.warning("could not record the failed EC2 attempt in MLflow", exc_info=True)
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
    finally:
        zia_signal_scope.close()


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
