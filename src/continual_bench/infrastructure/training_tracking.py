"""Shared MLflow tracking and checkpoint verification for recorded training rounds."""

import hashlib
import json
import math
import re
import time
from pathlib import Path, PurePosixPath

HISTORY_FILE = "training-history.json"
STEP_KINDS = {"epoch", "minibatch", "update", "example"}
_INVALID_NAME = re.compile(r"[^/\w.\- ]")
_FIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def sanitize_mlflow_name(name: str) -> str:
    """Convert metric/parameter names into valid MLflow identifiers (`:` -> `/`)."""
    cleaned = _INVALID_NAME.sub("_", str(name).replace(":", "/").strip())
    return cleaned[:250] or "metric"


def training_round_mlflow(parent_run_id, report, now_ms=0):
    """Yield the tags, parameters and metric points of each completed fit."""
    fit_ids = set()
    last_global = 0
    for fit in report.get("training_history", []):
        fit_id = fit["fit_id"]
        if not _FIT_ID.fullmatch(fit_id) or fit_id in fit_ids:
            raise ValueError(f"invalid or duplicate fit: {fit_id}")
        fit_ids.add(fit_id)
        action = fit["action"]
        tags = {
            "mlflow.parentRunId": parent_run_id,
            "mlflow.runName": f"{fit_id}-{action}",
            "cb-fit-id": fit_id,
            "cb-fit-action": action,
        }
        checkpoint = fit.get("checkpoint") or {}
        if fit.get("deployed_version") is not None:
            tags["cb-model-version"] = str(fit["deployed_version"])
        params = {
            key: str(fit[key])
            for key in (
                "start_seq", "end_seq", "block_index", "train_samples",
                "validation_samples", "training_presentations",
            ) if fit.get(key) is not None
        }
        if checkpoint.get("sha256"):
            params["checkpoint_sha256"] = checkpoint["sha256"]
        if checkpoint.get("file"):
            params["checkpoint"] = checkpoint["file"]
        if checkpoint.get("offset") is not None:
            params["checkpoint_offset"] = str(checkpoint["offset"])
        if checkpoint.get("bytes") is not None:
            params["checkpoint_bytes"] = str(checkpoint["bytes"])
        steps = fit["steps"]
        if not steps:
            raise ValueError(f"fit {fit_id} has no steps")
        kind = steps[0]["kind"]
        if kind not in STEP_KINDS:
            raise ValueError(f"fit {fit_id} has an invalid step kind")
        tags["cb-step-kind"] = kind
        metrics = {}

        def add(name, value, step):
            if isinstance(value, (int, float)) and math.isfinite(value):
                key = sanitize_mlflow_name(name)
                point = {
                    "key": key,
                    "value": float(value),
                    "timestamp": now_ms,
                    "step": step,
                }
                if (key, step) in metrics and metrics[key, step] != point:
                    raise ValueError(f"fit {fit_id} has conflicting metric values for {key} at step {step}")
                metrics[key, step] = point

        for index, row in enumerate(steps, start=1):
            global_step = row["global_step"]
            if row["step"] != index or row["kind"] != kind or type(global_step) is not int:
                raise ValueError(f"invalid step sequence for fit {fit_id}")
            if global_step <= last_global:
                raise ValueError(f"global step did not advance for fit {fit_id}")
            last_global = global_step
            add("global_step", global_step, index)
            for key, value in row["metrics"].items():
                add(key, value, index)
        for key, value in fit["summary"].items():
            add(f"fit_summary/{key}", value, 0)
        evidence = {
            "params": params, "kind": kind, "action": action,
            "metrics": [[key, step, point["value"]] for (key, step), point in sorted(metrics.items())],
        }
        tags["cb-fit-evidence-sha256"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        yield tags, params, list(metrics.values())


def training_checkpoint_paths(directory, report):
    """Verify retained checkpoint bytes, including slices of shared model journals."""
    root = Path(directory).resolve()
    paths = set()
    for fit in report.get("training_history", []):
        checkpoint = fit.get("checkpoint") or {}
        name = checkpoint.get("file")
        if not name:
            continue
        relative = PurePosixPath(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or ".." in relative.parts:
            raise ValueError("invalid training checkpoint path")
        offset = checkpoint.get("offset")
        size = checkpoint.get("bytes")
        with path.open("rb") as stream:
            if offset is None:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                observed = path.stat().st_size
            else:
                if type(offset) is not int or type(size) is not int or min(offset, size) < 0:
                    raise ValueError("invalid checkpoint journal segment")
                stream.seek(offset)
                part = stream.read(size)
                digest, observed = hashlib.sha256(part).hexdigest(), len(part)
        if digest != checkpoint.get("sha256") or (size is not None and size != observed):
            raise ValueError(f"training checkpoint checksum or size differs: {name}")
        paths.add(relative)
    return sorted(paths)


def publish_training_history(client, run_id, directory, report, *, include_checkpoints=True):
    """Verify and publish the fit history before marking the parent complete."""
    if "training_history" not in report:
        return []
    directory = Path(directory)
    path = directory / HISTORY_FILE
    if json.loads(path.read_bytes()) != {"version": 1, "fits": report["training_history"]}:
        raise ValueError("retained training history differs from the report")
    checkpoints = training_checkpoint_paths(directory, report) if include_checkpoints else []
    client.log_artifact(run_id, str(path))
    for relative in checkpoints:
        folder = None if relative.parent == PurePosixPath(".") else relative.parent.as_posix()
        client.log_artifact(run_id, str(directory / relative), artifact_path=folder)
    run = client.get_run(run_id)
    children = publish_sdk_training_rounds(client, run.info.experiment_id, run_id, report)
    client.set_tag(run_id, "cb-training-round-count", str(len(children)))
    return children


def publish_sdk_training_rounds(client, experiment_id, parent_run_id, report, now_ms=None):
    """Synchronously publish one child run per fit, keyed by parent attempt and fit ID."""
    from mlflow.entities import Metric, Param

    from .metric_tracking import missing_metric_points

    if not report.get("training_history"):
        return []
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    existing = {}
    token = None
    while True:
        found = client.search_runs(
            [experiment_id], f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
            max_results=500, page_token=token,
        )
        for run in found:
            if fit_id := run.data.tags.get("cb-fit-id"):
                if fit_id in existing:
                    raise ValueError(f"duplicate MLflow child runs for {parent_run_id}/{fit_id}")
                existing[fit_id] = run
        token = found.token
        if not token:
            break
    receipts = []
    for tags, params, metrics in training_round_mlflow(parent_run_id, report, now_ms):
        fit_id = tags["cb-fit-id"]
        run = existing.get(fit_id) or client.create_run(experiment_id, tags=tags)
        run_id = run.info.run_id
        remote_evidence = run.data.tags.get("cb-fit-evidence-sha256")
        if remote_evidence is not None and remote_evidence != tags["cb-fit-evidence-sha256"]:
            raise ValueError(f"MLflow child {fit_id} already tracks different training evidence")
        if run.info.status != "FINISHED":
            missing = missing_metric_points(
                client, run_id, [(p["key"], p["step"], p["value"]) for p in metrics],
                known_keys=set(run.data.metrics),
            )
            for offset in range(0, max(1, len(missing)), 500):
                batch = [Metric(k, v, now_ms, s) for k, s, v in missing[offset:offset + 500]]
                client.log_batch(
                    run_id,
                    metrics=batch,
                    params=[Param(k, v) for k, v in params.items()] if offset == 0 else (),
                    synchronous=True,
                )
            client.set_terminated(run_id, "FINISHED")
        receipts.append({"fit_id": fit_id, "run_id": run_id, "metric_points": len(metrics)})
    return receipts
