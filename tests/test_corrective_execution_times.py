"""Post-processing validates complete campaign receipts and never invents per-trial clocks."""

import copy
import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import publish_corrective_execution_times as times

CODE = {
    "git_revision": "a" * 40,
    "code_sha256": "b" * 64,
    "launcher_sha256": "c" * 64,
    "replication_cli_sha256": "d" * 64,
}
ENVIRONMENT = {key: CODE[key] for key in ("git_revision", "code_sha256")}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2))


def tracking_report(state, job, seed, role):
    key = hashlib.sha256(f"{job}-{seed}-{role}".encode()).hexdigest()[:24]
    report = {"run_key": key, "complete": True, "contract": {"seed": seed, "environment": ENVIRONMENT}}
    directory = "retraining-replication" if role == "candidate" else "retraining-policy-results"
    path = state / "replications" / "tracking" / directory / key / "report.json"
    save(path, report)
    receipt = {
        "complete": True,
        "mlflow_run_id": hashlib.sha256(key.encode()).hexdigest()[:32],
        "report_sha256": times.digest(path),
    }
    return {**report, "mlflow": receipt}


@pytest.fixture
def campaign(tmp_path):
    state = tmp_path / "campaign"
    seeds, cases = [42, 43, 44], []
    status = {"complete": True, "finished_at": "2026-11-23T10:01:00+00:00", "matrix": {}, "replications": {}}
    for dataset in ("insects_a", "insects_b"):
        for model in ("logistic", "hoeffding_tree"):
            for policy in ("none", "drift", "cara", "upf", "oracle"):
                for seed in seeds:
                    config = {
                        "dataset": {"name": dataset, "seed": seed},
                        "model": {"name": model},
                        "policy": {"name": policy},
                        "detector": {"name": "none"},
                        "mode": "incremental",
                        "artifact_mode": "metrics_only",
                    }
                    key = hashlib.sha256(
                        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()[:24]
                    coordinates = {
                        "dataset": dataset,
                        "seed": seed,
                        "model": model,
                        "policy": policy,
                        "mode": "incremental",
                        "detector": "none",
                    }
                    case = {"case_key": key, "config": config, "coordinates": coordinates}
                    cases.append(case)
                    report = {
                        "complete": True,
                        "run_key": key,
                        "config": config,
                        "environment": ENVIRONMENT,
                        "tracking": {"metrics": {"pipeline_elapsed_seconds": None}},
                    }
                    relative = f"matrix/cases/{key}/binary/{key}/report.json"
                    path = state / relative
                    save(path, report)
                    receipt = {
                        "complete": True,
                        "mlflow_run_id": f"{len(cases):032x}",
                        "report_sha256": times.digest(path),
                    }
                    save(path.parent / "benchmark-receipt.json", receipt)
                    result = {
                        "job_id": "matrix-" + key,
                        "code": CODE,
                        "status": "passed",
                        "report": relative,
                        "tracking": receipt,
                        "artifact_sha256": times.digest(path),
                        "started_at": "2026-11-23T10:00:00+00:00",
                        "finished_at": "2026-11-23T10:00:10.250000+00:00",
                    }
                    save(state / "job-results" / (result["job_id"] + ".json"), result)
                    status["matrix"][key] = result
    jobs = [
        {"job_id": "upf-" + dataset, "family": "upf", "dataset": dataset}
        for dataset in ("electricity", "airlines", "gauss", "circles")
    ]
    jobs += [
        {"job_id": timing + "-" + dataset, "family": "paired", "dataset": dataset, "timing": timing}
        for timing in ("same-batch", "next-batch")
        for dataset in ("electricity", "airlines")
    ]
    for job in jobs:
        item = {
            "dataset": job["dataset"],
            "trials": 3,
            "seeds": seeds,
            "candidate_tracking": [
                tracking_report(state, job["job_id"], seed, "candidate") for seed in seeds
            ],
            "trials_detail": [
                {"seed": seed, "policy_tracking": tracking_report(state, job["job_id"], seed, "policy")}
                for seed in seeds
            ],
        }
        relative = f"replications/{job['job_id']}.json"
        path = state / relative
        save(path, [item])
        result = {
            "job_id": job["job_id"],
            "code": CODE,
            "status": "passed",
            "output": relative,
            "artifact_sha256": times.digest(path),
            "started_at": "2026-11-23T10:00:00+00:00",
            "finished_at": "2026-11-23T10:00:30+00:00",
        }
        save(state / "job-results" / (job["job_id"] + ".json"), result)
        status["replications"][job["job_id"]] = result
    plan = {
        "campaign_id": "fixture-campaign",
        "code": CODE,
        "tracking_required": True,
        "matrix": {"cases": cases, "skipped": []},
        "replications": {"jobs": jobs, "trials": 3, "seeds": seeds},
    }
    save(state / "campaign-plan.json", plan)
    save(state / "campaign-status.json", status)
    return state


def replace_result(state, job_id, change):
    path = state / "job-results" / (job_id + ".json")
    result = times.read(path)
    change(result)
    save(path, result)
    status = times.read(state / "campaign-status.json")
    category = "matrix" if job_id.startswith("matrix-") else "replications"
    key = job_id.removeprefix("matrix-") if category == "matrix" else job_id
    status[category][key] = result
    save(state / "campaign-status.json", status)


def first_job(state):
    return "matrix-" + times.read(state / "campaign-plan.json")["matrix"]["cases"][0]["case_key"]


def test_complete_wall_time_sidecar_has_real_grains_and_preserves_every_source(campaign, tmp_path):
    before = {path.relative_to(campaign): times.digest(path) for path in campaign.rglob("*.json")}
    document = times.build_execution_times(campaign)
    rows = document["rows"]
    matrix = [row for row in rows if row["granularity"] == "matrix_case"]
    replications = [row for row in rows if row["granularity"] == "replication_job"]
    assert len(matrix) == 60 and len(replications) == 8
    assert all(row["seed_count"] == 1 and row["job_body_wall_seconds"] == 10.25 for row in matrix)
    assert all(row["seeds"] == [42, 43, 44] and row["job_body_wall_seconds"] == 30.0 for row in replications)
    assert all(row["mlflow_parent_run_id"] is None and row["seed"] is None for row in replications)
    assert "per-trial" in document["scope_detail"] and "startup" in document["scope_detail"]
    output = tmp_path / "derived"
    manifest = times.write_outputs(document, output)
    with (output / times.CSV_NAME).open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 68
    assert all(times.digest(output / row["path"]) == row["sha256"] for row in manifest["files"])
    assert before == {path.relative_to(campaign): times.digest(path) for path in campaign.rglob("*.json")}
    assert "/Users/" not in json.dumps(document) and str(campaign) not in json.dumps(document)


@pytest.mark.parametrize(
    "problem", ["incomplete", "missing_case", "missing_job", "status_missing", "receipt_missing"]
)
def test_incomplete_campaign_is_never_postprocessed(campaign, problem):
    plan, status = times.read(campaign / "campaign-plan.json"), times.read(campaign / "campaign-status.json")
    if problem == "incomplete":
        status["complete"] = False
    elif problem == "missing_case":
        plan["matrix"]["cases"].pop()
    elif problem == "missing_job":
        plan["replications"]["jobs"].pop()
    elif problem == "status_missing":
        status["matrix"].pop(next(iter(status["matrix"])))
    else:
        (campaign / "job-results" / (first_job(campaign) + ".json")).unlink()
    save(campaign / "campaign-plan.json", plan)
    save(campaign / "campaign-status.json", status)
    with pytest.raises((ValueError, FileNotFoundError)):
        times.build_execution_times(campaign)


@pytest.mark.parametrize(
    "finished",
    [
        "invalid",
        "2026-11-23T10:00:10",
        "2026-11-23T09:59:59Z",
        "2026-11-23T10:00:00Z",
        "2026-11-23T10:02:00Z",
    ],
)
def test_invalid_or_incoherent_recorded_times_are_rejected(campaign, finished):
    replace_result(campaign, first_job(campaign), lambda result: result.update(finished_at=finished))
    with pytest.raises(ValueError, match="timestamp"):
        times.build_execution_times(campaign)


def test_timezone_offsets_preserve_the_actual_interval(campaign):
    replace_result(
        campaign,
        first_job(campaign),
        lambda result: result.update(
            started_at="2026-11-23T07:00:00-03:00", finished_at="2026-11-23T07:00:10.250000-03:00"
        ),
    )
    row = times.build_execution_times(campaign)["rows"][0]
    assert row["job_body_wall_seconds"] == 10.25
    assert row["started_at"] == "2026-11-23T10:00:00+00:00"


@pytest.mark.parametrize("problem", ["code", "artifact_hash", "tracking_receipt", "report_code"])
def test_changed_code_or_artifacts_cannot_be_relabelled_as_verified_time(campaign, problem):
    job_id = first_job(campaign)
    if problem == "code":
        replace_result(campaign, job_id, lambda result: result["code"].update(code_sha256="f" * 64))
    elif problem == "artifact_hash":
        replace_result(campaign, job_id, lambda result: result.update(artifact_sha256="f" * 64))
    elif problem == "tracking_receipt":
        replace_result(campaign, job_id, lambda result: result["tracking"].update(report_sha256="f" * 64))
    else:

        def replace_report(result):
            path = campaign / result["report"]
            report = times.read(path)
            report["environment"]["git_revision"] = "f" * 40
            save(path, report)
            result["artifact_sha256"] = times.digest(path)
            result["tracking"]["report_sha256"] = times.digest(path)
            save(path.parent / "benchmark-receipt.json", result["tracking"])

        replace_result(campaign, job_id, replace_report)
    with pytest.raises(ValueError, match="code|hash|receipt"):
        times.build_execution_times(campaign)


class FakeClient:
    def __init__(self, document):
        self.calls = []
        self.runs, self.histories = {}, {}
        for row in document["rows"]:
            if row["granularity"] != "matrix_case":
                continue
            self.runs[row["mlflow_parent_run_id"]] = SimpleNamespace(
                info=SimpleNamespace(status="FINISHED"),
                data=SimpleNamespace(
                    tags={
                        "benchmark_complete": "true",
                        "cb-report-sha256": row["source_artifact_sha256"],
                        "cb-local-run-key": row["job_id"].removeprefix("matrix-"),
                    }
                ),
            )

    def get_run(self, run_id):
        return self.runs[run_id]

    def get_metric_history(self, run_id, key):
        assert key == times.METRIC  # No access to a legacy timer or another metric.
        return self.histories.get(run_id, [])

    def log_batch(self, run_id, *, metrics, tags, synchronous):
        assert synchronous is True and len(metrics) <= 1
        self.calls.append({"run_id": run_id, "metrics": list(metrics), "tags": list(tags)})
        if metrics:
            self.histories.setdefault(run_id, []).extend(metrics)
        self.runs[run_id].data.tags.update({tag.key: tag.value for tag in tags})


def test_tracking_is_scoped_to_matrix_parents_and_idempotent(campaign):
    pytest.importorskip("mlflow")
    document = times.build_execution_times(campaign)
    client = FakeClient(document)
    published = times.publish_execution_times(document, client)
    assert published["complete"] and len(client.calls) == 60
    for call in client.calls:
        metric = call["metrics"][0]
        assert metric.key == times.METRIC and metric.value == 10.25 and metric.step == 0
        assert metric.timestamp == document["rows"][0]["finished_at_timestamp_ms"]
        assert {tag.key for tag in call["tags"]} >= {
            "corrective_campaign/time_source_sha256",
            "corrective_campaign/time_scope",
        }
    repeated = times.publish_execution_times(document, client)
    assert len(client.calls) == 60 and not any(row["written"] for row in repeated["results"])
    assert all(run.info.status == "FINISHED" for run in client.runs.values())


@pytest.mark.parametrize("conflict", ["value", "timestamp", "origin", "status", "report_hash", "duplicate"])
def test_tracking_preflights_every_parent_and_rejects_conflicts_before_any_write(campaign, conflict):
    pytest.importorskip("mlflow")
    document = times.build_execution_times(campaign)
    client = FakeClient(document)
    row = document["rows"][59]
    run_id = row["mlflow_parent_run_id"]
    if conflict in {"value", "timestamp", "duplicate"}:
        point = SimpleNamespace(
            step=0, value=row["job_body_wall_seconds"], timestamp=row["finished_at_timestamp_ms"]
        )
        if conflict == "value":
            point.value += 1
        elif conflict == "timestamp":
            point.timestamp += 1
        client.histories[run_id] = [point, copy.copy(point)] if conflict == "duplicate" else [point]
    elif conflict == "origin":
        client.runs[run_id].data.tags["corrective_campaign/time_source_sha256"] = "f" * 64
    elif conflict == "status":
        client.runs[run_id].info.status = "RUNNING"
    else:
        client.runs[run_id].data.tags["cb-report-sha256"] = "f" * 64
    with pytest.raises(ValueError):
        times.publish_execution_times(document, client)
    assert not client.calls
