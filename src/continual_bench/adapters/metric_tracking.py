"""Append versioned evaluation evidence to an existing run without retraining."""

import copy
import hashlib
import json
import math
from pathlib import Path

from .catalog import put_immutable
from .local import atomic_write

VERSION = "residency-metrics-v1"
EXECUTION_SCOPE = "worker_subprocess_including_materialization_evaluation_publication_excluding_this_sidecar"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _source_sha256():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric_points(tracking):
    """Validate the full payload before remote writes; retain unavailable nulls."""
    if not isinstance(tracking, dict) or tracking.get("version") != VERSION:
        raise ValueError("unsupported tracking metrics version")
    required = ("metrics", "series", "availability", "settings")
    if not all(isinstance(tracking.get(key), dict) for key in required):
        raise ValueError("tracking requires metrics, series, availability and settings mappings")
    points = {}

    def add(key, step, value):
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("tracking metric step must be a nonnegative integer")
        if value is not None and not _finite(value):
            raise ValueError("tracking metric must be finite numeric or null")
        identity = key, step
        if identity in points and points[identity] != value:
            raise ValueError(f"conflicting tracking values for {key} at step {step}")
        points[identity] = value

    for name, value in tracking["metrics"].items():
        if not isinstance(name, str) or not name or name.startswith("/"):
            raise ValueError("tracking metric name must be a nonempty relative name")
        # This single metric is filled after the publication subprocess exits.
        # Replaying its original report must not conflict with the late value.
        if name != "pipeline_elapsed_seconds" or value is not None:
            add("tracking_v1/summary/" + name, 0, value)
    for name, entries in tracking["series"].items():
        if not isinstance(name, str) or not name or name.startswith("/"):
            raise ValueError("tracking series name must be a nonempty relative name")
        if not isinstance(entries, list):
            raise ValueError("tracking series must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or "step" not in entry or "value" not in entry:
                raise ValueError("tracking series entries require step and value")
            add("tracking_v1/" + name, entry["step"], entry["value"])
    _json_bytes(tracking)  # Reject non-JSON or nonfinite values in metadata too.
    return [(key, step, value) for (key, step), value in sorted(points.items())]


def missing_metric_points(client, run_id, points, known_keys=None):
    """Read each history once and reject conflicting values, ignoring timestamp."""
    desired = {}
    for key, step, value in points:
        if value is not None and not _finite(value):
            raise ValueError(f"nonfinite metric {key}")
        by_step = desired.setdefault(key, {})
        if step in by_step and by_step[step] != value:
            raise ValueError(f"conflicting metric {key} at step {step}")
        by_step[step] = value
    if known_keys is None and callable(getattr(client, "get_run", None)):
        run = client.get_run(run_id)
        metrics = getattr(getattr(run, "data", None), "metrics", None)
        if isinstance(metrics, dict):
            known_keys = set(metrics)
    missing = []
    for key, by_step in desired.items():
        existing = {}
        history = client.get_metric_history(run_id, key) if known_keys is None or key in known_keys else ()
        for metric in history:
            if metric.step in by_step:
                if not _finite(metric.value) or metric.value != by_step[metric.step]:
                    raise ValueError(f"remote metric conflicts for {key} at step {metric.step}")
                existing[metric.step] = metric.value
        missing.extend((key, step, value) for step, value in by_step.items()
                       if step not in existing and value is not None)
    return missing


def log_metric_points(client, run_id, points):
    missing = missing_metric_points(client, run_id, points)
    _log_verified_points(client, run_id, missing)


def _log_verified_points(client, run_id, missing):
    """Append preflight-verified points while the caller holds the run lock."""
    if missing:
        from mlflow.entities import Metric

        for offset in range(0, len(missing), 500):
            client.log_batch(run_id, metrics=[Metric(key, value, 0, step)
                                             for key, step, value in missing[offset:offset + 500]],
                             synchronous=True)


def publish_tracking(client, run_id, report, directory, objects=None, derived=False):
    """Publish a content-addressed sidecar; never mutate the original run files.

    The caller must serialize publication for a run. Read/check/append on MLflow
    histories cannot provide an atomic compare-and-set across concurrent writers.
    """
    tracking = report.get("tracking")
    if tracking is None:
        return None
    points = metric_points(tracking)
    # Detect conflicts before publishing any side effects or completion tags.
    missing = missing_metric_points(client, run_id, points)
    directory = Path(directory)
    report_path = directory / "report.json"
    original = report_path.read_bytes() if report_path.exists() else _json_bytes(
        {key: value for key, value in report.items() if key != "tracking"}
    )
    extra_paths = []
    for name in ("predictions.jsonl", "metric-definitions.json", "continual-probes.json",
                 "tracking/execution-v1.json"):
        path = directory / name
        if path.is_file():
            extra_paths.append(path)
    references = directory / "references"
    if references.is_dir():
        extra_paths.extend(path for path in sorted(references.rglob("*")) if path.is_file())
    files = []
    for path in extra_paths:
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("tracking evidence must be an ordinary file inside the run directory")
        blob = path.read_bytes()
        files.append({"path": path.relative_to(directory).as_posix(), "sha256": _sha(blob),
                      "bytes": len(blob)})
    envelope = {
        "version": VERSION,
        "derived": bool(derived),
        "report_sha256": _sha(original),
        "training_code_sha256": report["environment"]["code_sha256"],
        "evaluation_code_sha256": _source_sha256(),
        "dataset_sha256": report["dataset"]["sha256"],
        "settings_sha256": _sha(_json_bytes(tracking["settings"])),
        "tracking": tracking,
        "files": files,
    }
    payload = _json_bytes(envelope)
    payload_sha = _sha(payload)
    artifact_path = f"tracking/{VERSION}/{payload_sha}"
    sidecar = directory / artifact_path / "payload.json"
    if sidecar.exists() and sidecar.read_bytes() != payload:
        raise ValueError("immutable local tracking sidecar differs")
    if not sidecar.exists():
        atomic_write(sidecar, payload)
    client.set_tag(run_id, "tracking_v1_status", "publishing")
    uris = {}
    artifacts = [(sidecar, "payload.json", _sha(payload))]
    artifacts.extend((directory / item["path"], item["path"], item["sha256"]) for item in files)
    for path, relative, expected in artifacts:
        if _sha(path.read_bytes()) != expected:
            raise ValueError("tracking evidence changed during publication")
        if objects is not None:
            uris[relative] = put_immutable(objects, f"{artifact_path}/{relative}", path)
        parent = Path(relative).parent.as_posix()
        destination = artifact_path if parent == "." else f"{artifact_path}/{parent}"
        client.log_artifact(run_id, str(path), artifact_path=destination)
    # Explicitly synchronous: a completion tag must never precede queued writes.
    _log_verified_points(client, run_id, missing)
    client.set_tag(run_id, "tracking_v1_settings_sha256", envelope["settings_sha256"])
    if tracking.get("definitions_sha256"):
        client.set_tag(run_id, "tracking_v1_definitions_sha256", tracking["definitions_sha256"])
    client.set_tag(run_id, "tracking_v1_origin", "derived" if derived else "native")
    client.set_tag(run_id, "tracking_v1_unavailable_count",
                   str(sum(value is None for value in tracking["metrics"].values())))
    client.set_tag(run_id, "tracking_v1_payload_sha256", payload_sha)
    client.set_tag(run_id, "tracking_v1_status", "complete")
    return {"payload_sha256": payload_sha, "artifact_path": artifact_path,
            "sidecar_path": str(sidecar), "object_uris": uris}


def publish_execution(client, run_id, report, directory, elapsed_seconds, objects=None):
    """Publish first-success subprocess time without changing the saved report.

    The caller holds the same run lock as publish_tracking. Cache creation occurs
    after a successful subprocess and before its supplementary publication, so a
    failed upload can retry without substituting a faster cached replay time.
    """
    if report.get("tracking") is None:
        return None
    if not _finite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("execution elapsed seconds must be finite and nonnegative")
    directory = Path(directory)
    report_path = directory / "report.json"
    original = report_path.read_bytes() if report_path.exists() else _json_bytes(
        {key: value for key, value in report.items() if key != "tracking"}
    )
    source_report_sha = _sha(original)
    cache_path = directory / "tracking" / "execution-v1.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_bytes())
        if (cached.get("version") != "execution-v1"
                or cached.get("source_report_sha256") != source_report_sha
                or cached.get("run_id") != run_id
                or cached.get("scope") != EXECUTION_SCOPE
                or not _finite(cached.get("elapsed_seconds"))
                or cached["elapsed_seconds"] < 0):
            raise ValueError("execution cache provenance or elapsed time mismatch")
    else:
        cached = {"version": "execution-v1", "source_report_sha256": source_report_sha,
                  "run_id": run_id, "elapsed_seconds": elapsed_seconds, "scope": EXECUTION_SCOPE}
        atomic_write(cache_path, _json_bytes(cached))
    supplemental = dict(report)
    supplemental["tracking"] = copy.deepcopy(report["tracking"])
    tracking = supplemental["tracking"]
    tracking["metrics"]["pipeline_elapsed_seconds"] = cached["elapsed_seconds"]
    tracking["availability"]["pipeline_elapsed_seconds"] = {
        "status": "available", "reason": None, "scope": EXECUTION_SCOPE,
    }
    tracking["execution_scope"] = EXECUTION_SCOPE
    return publish_tracking(client, run_id, supplemental, directory, objects=objects, derived=False)
