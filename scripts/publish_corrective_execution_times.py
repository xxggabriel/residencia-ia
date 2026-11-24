"""Derive recorded job-body wall times after a complete corrective campaign.

This post-processing step never trains, changes reports or fills legacy timers.
Only --tracking writes a new metric and provenance tags to the 60 matrix parents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from continual_bench.infrastructure.storage import atomic_write

SCHEMA = "corrective-execution-times-v1"
METRIC = "corrective_campaign/job_body_wall_seconds"
SCOPE = "execute_job_body_materialization_evaluation_diagnostics_publication"
SCOPE_DETAIL = (
    "Recorded finished_at minus started_at in execute_job. Includes materialization, evaluation, "
    "diagnostics, imports inside this interval and tracking publication. Excludes process startup, "
    "module imports and code/dependency validation before started_at, plus artifact hashing and "
    "final job-result writing after finished_at. Not a CPU timer or a per-trial timer."
)
SHA256 = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
RUN_ID = re.compile(r"[0-9a-f]{32}")
JSON_NAME = "corrective-execution-times.json"
CSV_NAME = "corrective-execution-times.csv"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def contained(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("evidence paths must be relative to the campaign")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("evidence path escapes the campaign or is missing")
    return path


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("recorded job timestamps must be timezone-aware ISO strings")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("invalid recorded job timestamp") from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("recorded job timestamp has no timezone")
    return result.astimezone(timezone.utc)


def epoch_ms(value):
    delta = value - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def helper_identity():
    source = Path(__file__).resolve()
    checksum = digest(source)
    explicit = os.environ.get("BENCH_POSTPROCESS_REVISION")
    if explicit:
        if not REVISION.fullmatch(explicit):
            raise ValueError("BENCH_POSTPROCESS_REVISION must be a full commit SHA")
        return {"sha256": checksum, "git_revision": explicit, "revision_source": "explicit_environment"}
    try:
        root = source.parents[1]
        relative = source.relative_to(root).as_posix()
        revision = (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%H", "--", relative], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        committed = subprocess.check_output(
            ["git", "show", f"{revision}:{relative}"], cwd=root, stderr=subprocess.DEVNULL
        )
        if REVISION.fullmatch(revision) and hashlib.sha256(committed).hexdigest() == checksum:
            return {"sha256": checksum, "git_revision": revision, "revision_source": "matching_source_blob"}
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return {"sha256": checksum, "git_revision": None, "revision_source": "uncommitted_or_unavailable"}


def validate_code(code):
    fields = ("code_sha256", "launcher_sha256", "replication_cli_sha256")
    if not isinstance(code, dict) or not REVISION.fullmatch(str(code.get("git_revision", ""))):
        raise ValueError("campaign requires an immutable evaluation revision")
    if any(not SHA256.fullmatch(str(code.get(name, ""))) for name in fields):
        raise ValueError("campaign code hashes are missing or invalid")


def validate_report_code(report, code):
    environment = report.get("environment") or (report.get("contract") or {}).get("environment") or {}
    if any(environment.get(key) != code[key] for key in ("git_revision", "code_sha256")):
        raise ValueError("report code differs from its frozen campaign")


def validate_tracking_report(state, value, directory, code):
    if value.get("complete") is not True or not re.fullmatch(r"[a-f0-9]{24}", str(value.get("run_key", ""))):
        raise ValueError("replication tracking report is incomplete")
    validate_report_code(value, code)
    receipt = value.get("mlflow")
    if not isinstance(receipt, dict) or receipt.get("complete") is not True:
        raise ValueError("replication tracking receipt is incomplete")
    if not RUN_ID.fullmatch(str(receipt.get("mlflow_run_id", ""))):
        raise ValueError("replication tracking receipt has no valid parent ID")
    relative = f"replications/tracking/{directory}/{value['run_key']}/report.json"
    native = contained(state, relative)
    if digest(native) != receipt.get("report_sha256"):
        raise ValueError("replication native report hash differs from its receipt")


def validate_replication(state, payload, job, seeds, code):
    if not isinstance(payload, list) or len(payload) != 1 or payload[0].get("dataset") != job["dataset"]:
        raise ValueError("replication output does not match its planned dataset")
    item = payload[0]
    if item.get("seeds") != seeds or item.get("trials") != 3:
        raise ValueError("replication job must retain its three planned seeds")
    trials, candidates = item.get("trials_detail", []), item.get("candidate_tracking", [])
    if [trial.get("seed") for trial in trials] != seeds or len(candidates) != 3:
        raise ValueError("replication output has incomplete per-seed evidence")
    for candidate, trial in zip(candidates, trials, strict=True):
        if candidate.get("contract", {}).get("seed") != trial["seed"]:
            raise ValueError("candidate seed differs from its trial")
        validate_tracking_report(state, candidate, "retraining-replication", code)
        validate_tracking_report(state, trial["policy_tracking"], "retraining-policy-results", code)


def build_execution_times(state):
    state = Path(state).resolve()
    plan_path, status_path = state / "campaign-plan.json", state / "campaign-status.json"
    plan, status = read(plan_path), read(status_path)
    if status.get("complete") is not True:
        raise ValueError("campaign-status.complete must be true before post-processing")
    cases, replications = plan["matrix"]["cases"], plan["replications"]["jobs"]
    if len(cases) != 60 or len(replications) != 8 or plan["matrix"].get("skipped"):
        raise ValueError("complete campaign must contain exactly 60 matrix cases and 8 replication jobs")
    seeds = plan["replications"].get("seeds")
    if not isinstance(seeds, list) or len(set(seeds)) != 3 or len(seeds) != 3:
        raise ValueError("campaign requires three distinct explicit seeds")
    if plan["replications"].get("trials") != 3 or plan.get("tracking_required") is not True:
        raise ValueError("complete tracked campaign with three replications is required")
    validate_code(plan["code"])
    jobs = [
        {"job_id": "matrix-" + case["case_key"], "family": "matrix", **case} for case in cases
    ] + replications
    if len({job["job_id"] for job in jobs}) != 68:
        raise ValueError("campaign contains duplicate job identities")
    if set(status.get("matrix", {})) != {case["case_key"] for case in cases}:
        raise ValueError("matrix status does not cover the exact planned cases")
    if set(status.get("replications", {})) != {job["job_id"] for job in replications}:
        raise ValueError("replication status does not cover the exact planned jobs")
    campaign_finished = timestamp(status["finished_at"])
    rows = []
    for job in jobs:
        relative_result = f"job-results/{job['job_id']}.json"
        result_path = contained(state, relative_result)
        result = read(result_path)
        category = "matrix" if job["family"] == "matrix" else "replications"
        key = job.get("case_key", job["job_id"])
        if result != status[category][key] or result.get("status") != "passed":
            raise ValueError("job-result differs from its passed campaign status")
        if result.get("job_id") != job["job_id"] or result.get("code") != plan["code"]:
            raise ValueError("job-result identity or code differs from the campaign")
        started, finished = timestamp(result["started_at"]), timestamp(result["finished_at"])
        if finished <= started or finished > campaign_finished:
            raise ValueError("job timestamps are reversed, empty or exceed campaign completion")
        artifact_relative = result.get("report") if category == "matrix" else result.get("output")
        artifact = contained(state, artifact_relative)
        artifact_sha = digest(artifact)
        if artifact_sha != result.get("artifact_sha256"):
            raise ValueError("job artifact hash differs from its execution receipt")
        parent_id = receipt_sha = None
        if category == "matrix":
            report = read(artifact)
            if report.get("complete") is not True or report.get("run_key") != job["case_key"]:
                raise ValueError("matrix report is incomplete or belongs to another case")
            if report.get("config") != job["config"]:
                raise ValueError("matrix report configuration differs from the plan")
            validate_report_code(report, plan["code"])
            coordinates = job["coordinates"]
            seed = job["config"]["dataset"]["seed"]
            if seed not in seeds or coordinates.get("seed") != seed:
                raise ValueError("matrix seed differs from its planned coordinates")
            raw_config = json.dumps(job["config"], sort_keys=True, separators=(",", ":")).encode()
            if hashlib.sha256(raw_config).hexdigest()[:24] != job["case_key"]:
                raise ValueError("case key is not the canonical planned configuration hash")
            receipt_path = artifact.parent / "benchmark-receipt.json"
            receipt = read(receipt_path)
            parent_id = receipt.get("mlflow_run_id")
            if receipt != result.get("tracking") or receipt.get("complete") is not True:
                raise ValueError("matrix tracking receipt differs from its job-result")
            if receipt.get("report_sha256") != artifact_sha or not RUN_ID.fullmatch(str(parent_id)):
                raise ValueError("matrix report hash or tracked parent ID is invalid")
            receipt_sha = digest(receipt_path)
            selected_seeds = [seed]
        else:
            validate_replication(state, read(artifact), job, seeds, plan["code"])
            coordinates, selected_seeds, seed = {"dataset": job["dataset"]}, seeds, None
        rows.append(
            {
                "campaign_id": plan["campaign_id"],
                "job_id": job["job_id"],
                "granularity": "matrix_case" if category == "matrix" else "replication_job",
                "dataset": coordinates["dataset"],
                "model": coordinates.get("model"),
                "policy": coordinates.get("policy"),
                "mode": coordinates.get("mode"),
                "detector": coordinates.get("detector"),
                "seed": seed,
                "seeds": selected_seeds,
                "seed_count": len(selected_seeds),
                "decision_timing": job.get("timing"),
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "finished_at_timestamp_ms": epoch_ms(finished),
                "job_body_wall_seconds": (finished - started).total_seconds(),
                "scope": SCOPE,
                "source_job_result": relative_result,
                "source_job_result_sha256": digest(result_path),
                "source_artifact": artifact_relative,
                "source_artifact_sha256": artifact_sha,
                "source_tracking_receipt_sha256": receipt_sha,
                "mlflow_parent_run_id": parent_id,
            }
        )
    matrix_ids = [row["mlflow_parent_run_id"] for row in rows if row["granularity"] == "matrix_case"]
    if len(set(matrix_ids)) != 60:
        raise ValueError("matrix jobs do not identify 60 distinct tracked parent runs")
    return {
        "schema": SCHEMA,
        "campaign_id": plan["campaign_id"],
        "source_campaign_complete": True,
        "matrix_cases": 60,
        "replication_jobs": 8,
        "evaluation_code": plan["code"],
        "helper": helper_identity(),
        "source_plan_sha256": digest(plan_path),
        "source_status_sha256": digest(status_path),
        "scope": SCOPE,
        "scope_detail": SCOPE_DETAIL,
        "clock": "timezone-aware recorded wall clock, not process CPU time",
        "attempt_scope": (
            "Recorded successful execute_job body only; does not reconstruct earlier interrupted attempts, "
            "prove fresh training instead of reuse, or infer a per-trial duration."
        ),
        "granularity": {
            "matrix_case": "one configured seed",
            "replication_job": "one subprocess job for all three seeds; no inferred per-trial duration",
        },
        "tracking": {
            "metric": METRIC,
            "step": 0,
            "timestamp": "finished_at_timestamp_ms",
            "matrix_only": True,
        },
        "rows": rows,
    }


def _origin_tags(document, row):
    return {
        "corrective_campaign/time_source_sha256": row["source_job_result_sha256"],
        "corrective_campaign/time_source_path": row["source_job_result"],
        "corrective_campaign/time_scope": SCOPE,
        "corrective_campaign/time_granularity": "matrix_case",
        "corrective_campaign/time_helper_sha256": document["helper"]["sha256"],
    }


def publish_execution_times(document, client):
    """Preflight every target; one synchronous batch per parent needing a change."""
    from mlflow.entities import Metric, RunTag

    rows = [row for row in document["rows"] if row["granularity"] == "matrix_case"]
    if len(rows) != 60 or len({row["mlflow_parent_run_id"] for row in rows}) != 60:
        raise ValueError("tracking requires exactly the 60 distinct matrix parents")
    prepared = []
    for row in rows:
        run_id = row["mlflow_parent_run_id"]
        run = client.get_run(run_id)
        tags = run.data.tags
        if run.info.status != "FINISHED" or tags.get("benchmark_complete") != "true":
            raise ValueError("target parent is not a completed benchmark")
        if tags.get("cb-report-sha256") != row["source_artifact_sha256"]:
            raise ValueError("target parent report hash differs from the verified source")
        if tags.get("cb-local-run-key") != row["job_id"].removeprefix("matrix-"):
            raise ValueError("target parent key differs from the planned matrix case")
        expected_tags = _origin_tags(document, row)
        if any(name in tags and tags[name] != value for name, value in expected_tags.items()):
            raise ValueError("conflicting execution-time provenance tags")
        history = list(client.get_metric_history(run_id, METRIC))
        if len(history) > 1 or any(
            point.step != 0
            or point.timestamp != row["finished_at_timestamp_ms"]
            or not math.isclose(point.value, row["job_body_wall_seconds"], rel_tol=1e-12, abs_tol=1e-12)
            for point in history
        ):
            raise ValueError("conflicting execution-time metric already exists")
        changed = {name: value for name, value in expected_tags.items() if tags.get(name) != value}
        prepared.append((row, not history, changed))
    results = []
    for row, missing, tags in prepared:
        run_id = row["mlflow_parent_run_id"]
        if missing or tags:
            metric = Metric(METRIC, row["job_body_wall_seconds"], row["finished_at_timestamp_ms"], 0)
            client.log_batch(
                run_id,
                metrics=[metric] if missing else [],
                tags=[RunTag(name, value) for name, value in tags.items()],
                synchronous=True,
            )
        history = list(client.get_metric_history(run_id, METRIC))
        actual_tags = client.get_run(run_id).data.tags
        if len(history) != 1 or history[0].step != 0:
            raise ValueError("execution-time publication did not retain one point")
        if (
            history[0].timestamp != row["finished_at_timestamp_ms"]
            or not math.isclose(history[0].value, row["job_body_wall_seconds"], rel_tol=1e-12, abs_tol=1e-12)
            or any(actual_tags.get(name) != value for name, value in _origin_tags(document, row).items())
        ):
            raise ValueError("execution-time publication readback differs from its source")
        results.append(
            {"run_id": run_id, "job_id": row["job_id"], "written": missing or bool(tags), "verified": True}
        )
    return {"complete": True, "metric": METRIC, "parent_count": 60, "results": results}


def write_outputs(document, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False).encode()
    atomic_write(output / JSON_NAME, raw)
    stream = io.StringIO(newline="")
    table = csv.DictWriter(stream, fieldnames=list(document["rows"][0]))
    table.writeheader()
    for row in document["rows"]:
        table.writerow({**row, "seeds": json.dumps(row["seeds"], separators=(",", ":"))})
    atomic_write(output / CSV_NAME, stream.getvalue().encode())
    manifest = {
        "schema": SCHEMA,
        "helper": document["helper"],
        "files": [
            {"path": name, "sha256": digest(output / name), "bytes": (output / name).stat().st_size}
            for name in (JSON_NAME, CSV_NAME)
        ],
        "tracking_metric": METRIC,
        "scope": SCOPE,
    }
    atomic_write(output / "corrective-execution-times-manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking", action="store_true")
    args = parser.parse_args()
    state, output = args.state.resolve(), args.output.resolve()
    if output == state or any(
        output.is_relative_to(state / name) for name in ("matrix", "replications", "job-results")
    ):
        raise ValueError("derived evidence cannot overwrite campaign input directories")
    document = build_execution_times(state)
    manifest = write_outputs(document, output)
    if args.tracking:
        from mlflow.tracking import MlflowClient

        receipt = publish_execution_times(document, MlflowClient())
        atomic_write(
            output / "corrective-execution-times-tracking.json", json.dumps(receipt, indent=2).encode()
        )
    print(
        json.dumps(
            {"matrix_cases": 60, "replication_jobs": 8, "files": manifest["files"], "tracking": args.tracking}
        )
    )


if __name__ == "__main__":
    main()
