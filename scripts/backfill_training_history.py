"""Inspect or recover grouped training evidence from existing benchmark runs."""

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from continual_bench.evaluation.historical_training import recover_training_history
from continual_bench.infrastructure.storage import atomic_write, publication_lock

REPORT_NAMES = ("report.json", "epoch-report.json", "result.json")
BACKFILL_SHA = "cb-training-backfill-sha256"
BACKFILL_STATUS = "cb-training-backfill-status"
NATIVE_EPOCH_KEYS = {
    "epoch",
    "early_stopped",
    "best_epoch",
    "patience_consumed",
    "training_duration_seconds",
    "validation_duration_seconds",
    "peak_rss_mb",
    "best_validation_loss",
    "best_score",
    "train_samples",
    "validation_samples",
}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _relative_path(value):
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"invalid relative artifact path: {value}")
    return path


def _validated_run_id(value):
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("expected a 32-character MLflow run ID")
    return value


def _read_s3(uri):
    from continual_bench.infrastructure.aws.s3 import S3ObjectStore

    parts = urlsplit(uri)
    if parts.scheme != "s3" or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("an explicit S3 object URI is required")
    key = str(_relative_path(parts.path.removeprefix("/")))
    return S3ObjectStore(parts.netloc).get_bytes(key)


@dataclass
class RetainedReport:
    source: str
    payload: bytes
    directory: Path | None = None
    run_id: str | None = None
    final_manifest: dict | None = None
    origin: str = "retained_report"


def _final_report_uri(uri, final):
    if final.get("version") != "run-final-v1":
        raise ValueError("unsupported final manifest format")
    outputs = str(final["outputs_uri"]).rstrip("/")
    base = uri.rsplit("/", 1)[0]
    if (
        not outputs.startswith(base + "/attempts/")
        or len(_relative_path(outputs.removeprefix(base + "/")).parts) != 2
    ):
        raise ValueError("final manifest output must be inside its own attempt prefix")
    if final.get("report_uri"):
        path = final["report_uri"]
        if not path.startswith(outputs + "/"):
            raise ValueError("report URI differs from the final manifest attempt")
        _relative_path(path.removeprefix(outputs + "/"))
        return path
    if final.get("kind") == "zia":
        return f"{outputs}/report.json"
    run_key = str(final.get("run_key") or "")
    if not re.fullmatch(r"[0-9a-f]{16,64}", run_key):
        raise ValueError("legacy final manifest lacks a retained report key")
    coordinates = final.get("matrix_coordinates", {})
    mode = coordinates.get("mode") or final.get("labels", {}).get("mode")
    if mode == "epochs":
        relative = f"epochs/{run_key}/epoch-report.json"
    elif mode == "typed_prequential":
        relative = f"typed/{run_key}/report.json"
    else:
        relative = f"binary/{run_key}/report.json"
    return f"{outputs}/{relative}"


def _sidecar_names(report):
    if "training_history" in report:
        return ()
    if report.get("schema_version") == "task_evaluation-typed-v1":
        return ("samples.jsonl",)
    if isinstance(report.get("model_versions"), dict) and report["model_versions"].get("index_sha256"):
        return ("model-versions.json",)
    return ()


def _cache_s3_sidecars(path, report, staging, read_s3):
    parent = path.rsplit("/", 1)[0]
    for name in _sidecar_names(report):
        try:
            payload = read_s3(f"{parent}/{name}")
        except FileNotFoundError:
            continue
        atomic_write(staging / name, payload)


def _load_s3(uri, staging, read_s3):
    raw = read_s3(uri)
    if PurePosixPath(urlsplit(uri).path).name != "final.json":
        _cache_s3_sidecars(uri, json.loads(raw), staging, read_s3)
        return RetainedReport(uri, raw, staging)
    final = json.loads(raw)
    path = _final_report_uri(uri, final)
    payload = read_s3(path)
    if final.get("report_sha256") and _sha(payload) != final["report_sha256"]:
        raise ValueError("retained report differs from the final manifest checksum")
    report = json.loads(payload)
    if final.get("run_key") and final["run_key"] != report.get("run_key"):
        raise ValueError("retained report differs from the final manifest run key")
    _cache_s3_sidecars(path, report, staging, read_s3)
    # The immutable attempt and the run link in final.json bind this report to its parent.
    return RetainedReport(
        path,
        payload,
        staging,
        final.get("links", {}).get("mlflow_run_id"),
        final,
    )


def _local_files(source):
    source = Path(source).resolve(strict=True)
    if source.is_file():
        yield source
    elif source.is_dir():
        for path in sorted(source.rglob("*.json")):
            if path.name in REPORT_NAMES and not path.is_symlink():
                yield path


def _points_by_step(client, run_id, name):
    points = {}
    for point in client.get_metric_history(run_id, name):
        if math.isfinite(point.value):
            if point.step in points and points[point.step] != point.value:
                raise ValueError(f"conflicting historical points for {name} at step {point.step}")
            points[point.step] = point.value
    return points


def _retained_run_metrics(client, run):
    """Use recorded epoch coordinates for known single-fit jobs without a report artifact."""
    run_id = run.info.run_id
    labels = run.data.tags
    keys = getattr(run.data, "metrics", {})
    reason = "no report artifact or independently identifiable epoch history is retained"
    report = {"recovery_limitations": [reason]}
    if (
        run.info.status == "FINISHED"
        and labels.get("cb-native-stage") in {"training", "tuning"}
        and "epoch" in keys
    ):
        coords = _points_by_step(client, run_id, "epoch")
        if (
            coords
            and sorted(coords) == list(range(1, len(coords) + 1))
            and all(step == epoch for step, epoch in coords.items())
        ):
            history = {step: {"epoch": step} for step in sorted(coords)}
            for name in sorted(keys):
                epoch_metric = name in NATIVE_EPOCH_KEYS or name.startswith(("train/", "validation/"))
                if name == "epoch" or not epoch_metric:
                    continue
                for step, value in _points_by_step(client, run_id, name).items():
                    if step in history:
                        history[step][name] = value
            report = {
                "training": {"history": list(history.values())},
                "recovery_limitations": [
                    "only the MLflow points retained by this job are recoverable; "
                    "checkpoints and training-window boundaries are not recorded here",
                ],
            }
    return RetainedReport(
        f"runs:/{run_id}#metrics",
        _json_bytes(report),
        run_id=run_id,
        origin="mlflow_metric_history",
    )


def _download_run_report(client, run_id, staging, path=None):
    run_id = _validated_run_id(run_id)
    run = client.get_run(run_id)
    if run.info.run_id != run_id or run.data.tags.get("mlflow.parentRunId"):
        raise ValueError("select an existing parent MLflow run")
    if path:
        path = str(_relative_path(path))
    else:
        candidates = [
            info.path
            for info in client.list_artifacts(run_id)
            if not info.is_dir and info.path in REPORT_NAMES
        ]
        if not candidates:
            return _retained_run_metrics(client, run)
        if len(candidates) > 1:
            raise ValueError("select the retained report using --report-artifact")
        path = candidates[0]

    def download(artifact):
        downloaded = Path(client.download_artifacts(run_id, artifact, str(staging))).resolve()
        if not downloaded.is_relative_to(staging.resolve()) or not downloaded.is_file():
            raise ValueError("downloaded artifact is outside its staging directory")
        return downloaded

    report_path = download(path)
    payload = report_path.read_bytes()
    if names := _sidecar_names(json.loads(payload)):
        parent = PurePosixPath(path).parent
        artifact_dir = None if parent == PurePosixPath(".") else str(parent)
        artifacts = {info.path for info in client.list_artifacts(run_id, artifact_dir) if not info.is_dir}
        for name in names:
            if (key := str(parent / name)) in artifacts:
                download(key)
    return RetainedReport(f"runs:/{run_id}/{path}", payload, report_path.parent, run_id)


def _plan(retained):
    report = json.loads(retained.payload)
    if not isinstance(report, dict):
        raise ValueError("expected a benchmark report object")
    fits, scheme, limitations = recover_training_history(report, retained.directory)
    if retained.origin == "mlflow_metric_history":
        limitations = report["recovery_limitations"]
    sidecars = {}
    if retained.directory:
        for name in _sidecar_names(report):
            if (path := retained.directory / name).is_file():
                sidecars[name] = _sha(path.read_bytes())
    evidence = {
        "version": 1,
        "origin": retained.origin,
        "source_sha256": _sha(retained.payload),
        "scheme": scheme,
        "limitations": limitations,
        "fits": fits,
        "auxiliary_sha256": sidecars,
    }
    if retained.run_id:
        evidence["parent_run_id"] = retained.run_id
    details = {
        "source": retained.source,
        "run_id": retained.run_id,
        "source_sha256": evidence["source_sha256"],
        "scheme": scheme,
        "availability": "partial" if fits and limitations else "available" if fits else "unavailable",
        "groups": len(fits),
        "steps": sum(len(fit["steps"]) for fit in fits),
        "limitations": limitations,
    }
    return report, evidence, details


def _materialize(report, evidence, destination):
    from continual_bench.evaluation.training_evidence import write_training_artifacts

    destination.mkdir(parents=True, exist_ok=True)
    write_training_artifacts(destination, {**report, "training_history": evidence["fits"]})
    atomic_write(destination / "backfill-provenance.json", _json_bytes(evidence))


def _publish(client, retained, report, evidence, destination):
    from continual_bench.infrastructure.training_tracking import publish_sdk_training_rounds

    run_id = _validated_run_id(retained.run_id)
    run = client.get_run(run_id)
    if run.info.run_id != run_id or run.info.status != "FINISHED" or run.data.tags.get("mlflow.parentRunId"):
        raise ValueError("historical evidence can only be attached to its finished parent run")
    if retained.final_manifest:
        final = retained.final_manifest
        if final["links"]["mlflow_run_id"] != run_id:
            raise ValueError("final manifest differs from the selected MLflow run")
        if experiment := final["links"].get("mlflow_experiment_id"):
            if str(experiment) != str(run.info.experiment_id):
                raise ValueError("final manifest differs from the selected MLflow experiment")
        for key, value in (
            ("cb-unit-id", final.get("run_id")),
            ("cb-attempt", final.get("attempt", {}).get("attempt_key")),
            ("cb-contract-sha256", final.get("contract_sha256")),
        ):
            if key in run.data.tags and run.data.tags[key] != value:
                raise ValueError(f"final manifest differs from parent run tag {key}")
    fingerprint = _sha(_json_bytes(evidence))
    previous = run.data.tags.get(BACKFILL_SHA)
    if previous and previous != fingerprint:
        raise ValueError("parent already has different historical evidence")
    if run.data.tags.get(BACKFILL_STATUS) == "complete" and previous == fingerprint:
        return "already_applied"
    if not previous:
        children = client.search_runs(
            [run.info.experiment_id],
            f"tags.`mlflow.parentRunId` = '{run_id}'",
            max_results=1,
        )
        if children:
            raise ValueError("parent already has grouped training runs")
    # A durable fingerprint reserves this plan for retries before any child or plot is published.
    client.set_tag(run_id, BACKFILL_SHA, fingerprint)
    client.set_tag(run_id, BACKFILL_STATUS, "publishing")
    prefix = f"training-backfill/v1/{fingerprint}"
    client.log_artifacts(run_id, str(destination), artifact_path=prefix)
    publish_sdk_training_rounds(
        client,
        run.info.experiment_id,
        run_id,
        {**report, "training_history": evidence["fits"]},
    )
    client.set_tag(run_id, BACKFILL_STATUS, "complete")
    return "applied"


def _handle(retained, args, client):
    report, evidence, details = _plan(retained)
    if not args.apply or not evidence["fits"]:
        return details
    if not retained.run_id and not args.output_dir:
        raise ValueError("local or unlinked evidence requires --output-dir")
    fingerprint = _sha(_json_bytes(evidence))
    lock_root = args.output_dir or Path("state/training-backfill")
    if (
        args.output_dir
        and retained.directory
        and lock_root.resolve().is_relative_to(retained.directory.resolve())
    ):
        raise ValueError("backfill output must be outside the original run directory")
    with publication_lock(lock_root, retained.run_id or fingerprint):
        if args.output_dir:
            destination = args.output_dir / fingerprint
            if destination.resolve().parent != args.output_dir.resolve():
                raise ValueError("backfill output is outside its chosen directory")
            _materialize(report, evidence, destination)
            details["output"] = str(destination)
        if retained.run_id:
            # Upload only files generated here, never other contents of a user-selected directory.
            with TemporaryDirectory(prefix="training-history-") as temp:
                destination = Path(temp)
                _materialize(report, evidence, destination)
                details["publication"] = _publish(client(), retained, report, evidence, destination)
    return details


def main(argv=None, *, mlflow_client=None, read_s3=_read_s3):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="local report/directory, S3 report or S3 final.json")
    parser.add_argument("--run-id", action="append", default=[], help="exact parent MLflow run")
    parser.add_argument("--report-artifact", help="relative report artifact path for a selected MLflow run")
    parser.add_argument("--apply", action="store_true", help="write derived evidence (default: inspect)")
    parser.add_argument("--output-dir", type=Path, help="local output for derived evidence (with --apply)")
    args = parser.parse_args(argv)
    if not args.sources and not args.run_id:
        parser.error("specify a report source or --run-id")
    if args.report_artifact and not args.run_id:
        parser.error("--report-artifact requires --run-id")
    if args.output_dir and not args.apply:
        parser.error("--output-dir requires --apply")

    def client():
        nonlocal mlflow_client
        if mlflow_client is None:
            from mlflow.tracking import MlflowClient

            mlflow_client = MlflowClient()
        return mlflow_client

    rows = []
    for source in args.sources:
        if source.startswith("s3://"):
            with TemporaryDirectory(prefix="training-source-") as temp:
                retained = _load_s3(source, Path(temp), read_s3)
                rows.append(_handle(retained, args, client))
        else:
            for path in _local_files(source):
                rows.append(_handle(RetainedReport(str(path), path.read_bytes(), path.parent), args, client))
    for run_id in args.run_id:
        with TemporaryDirectory(prefix="training-source-") as temp:
            retained = _download_run_report(client(), run_id, Path(temp), args.report_artifact)
            rows.append(_handle(retained, args, client))
    print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
