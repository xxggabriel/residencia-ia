"""Fargate Spot worker recovery against an object store with the S3 conditional-create contract."""

import json
import os
import signal

import pytest

from continual_bench.application.cloud import layout, planning, worker
from continual_bench.application.cloud.checkpoints import SpotCheckpoint
from continual_bench.application.cloud.contracts import CampaignRequest, RunUnit, run_identity
from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json, put_json, split_uri


@pytest.fixture
def spot_case(tmp_path):
    store = LocalObjectStore(tmp_path / "durable")
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id="spot-worker-test",
        matrix={
            "kind": "experiment_matrix",
            "path": "configs/matrices/binary-comparison.yaml",
        },
    )
    plan = planning.build_plan(request, release)
    unit = RunUnit(**plan["units"][0])
    put_json(store, unit.unit_key, unit.model_dump(mode="json"))
    return store, unit, release


def fargate_env(release, job_id, batch_attempt=1):
    """The environment AWS Batch gives a Fargate Spot benchmark task."""
    return {
        "AWS_BATCH_JOB_ID": job_id,
        "AWS_BATCH_JOB_ATTEMPT": str(batch_attempt),
        "BENCH_FARGATE_SPOT": "true",
        "BENCH_GIT_REVISION": release.git_sha,
        "BENCH_IMAGE_DIGEST": release.images["runtime"].digest,
        "BENCH_IMAGE_URI": release.images["runtime"].uri,
    }


def spot_notice():
    """Fargate reclaims Spot capacity with SIGTERM, then SIGKILL after the stop timeout."""
    os.kill(os.getpid(), signal.SIGTERM)


def test_worker_restores_the_last_complete_and_valid_checkpoint(spot_case, tmp_path, monkeypatch):
    store, unit, _ = spot_case
    output = tmp_path / "first" / "run"
    output.mkdir(parents=True)
    predictions = output / "predictions"
    predictions.mkdir()
    first_part = predictions / "part-1"
    first_part.write_bytes(b"predictions 1")
    first = SpotCheckpoint(store, unit, output, 1, requested=lambda: False)
    confirmed = first.save({"cursor": 2})
    assert confirmed and not first.due()
    second_part = predictions / "part-2"
    second_part.write_bytes(b"unconfirmed predictions 2")
    original_put = store.put_bytes

    def fail_manifest(key, data, **kwargs):
        if key.startswith(first.manifests):
            raise OSError("instance disappeared after object uploads")
        return original_put(key, data, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store, "put_bytes", fail_manifest)
        with pytest.raises(OSError):
            first.save({"cursor": 3}, force=True)
    second = SpotCheckpoint(store, unit, tmp_path / "second" / "run", 2, requested=lambda: False)
    assert second.load() == {"cursor": 2}
    assert second.resumed == confirmed
    assert sorted(p.name for p in (second.output / "predictions").iterdir()) == ["part-1"]

    (second.output / "predictions" / second_part.name).write_bytes(b"corrupt newer checkpoint")
    bad_checkpoint = second.save({"cursor": 3}, force=True)
    manifest = get_json(store, split_uri(bad_checkpoint)[1])
    digest = manifest["files"]["predictions/part-2"]["sha256"]
    (store.root / second.root / "blobs" / digest).write_bytes(b"corrupted transfer")
    third = SpotCheckpoint(store, unit, tmp_path / "third" / "run", 3, requested=lambda: False)
    assert third.load() == {"cursor": 2}
    assert third.resumed == confirmed
    assert (third.output / "predictions" / "part-1").read_bytes() == b"predictions 1"
    assert not (third.output / "predictions" / "part-2").exists()


class HostLost(BaseException):
    """Simulate a SIGKILL that cannot run exception/finalization handlers."""


def evaluate_result(root, values):
    directory = root / "run" / "binary" / "spot-test"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    report = {"run_key": "spot-test", "predictions": values}
    report_path.write_text(json.dumps(report))
    (directory / "model.bin").write_bytes(bytes(values))
    return {
        "report": report,
        "report_path": report_path,
        "output": root / "run",
        "headline": {"accuracy": 0.5},
    }


@pytest.mark.parametrize("fault", ["notice", "no_notice", "publication"])
def test_spot_recovery_keeps_results_and_original_publication_path(
    spot_case,
    tmp_path,
    monkeypatch,
    fault,
):
    store, unit, release = spot_case
    first = fargate_env(release, "job-first")
    first_attempt = layout.attempt_key_for("job-first", "1")
    first_status = layout.attempt_status_key(unit.campaign_id, unit.run_id, first_attempt)
    visits = []
    calls = 0

    def evaluate(_store, _unit, workdir, *, checkpoint):
        nonlocal calls
        calls += 1
        restored = checkpoint.load()
        state = restored or {"predictions": []}
        for n in range(len(state["predictions"]), 4):
            visits.append(n)
            state["predictions"].append(n)
            if n == 1:
                checkpoint.save(state, force=True)
            if calls == 1 and n == 2:
                if fault == "notice":
                    spot_notice()
                    assert checkpoint.requested()
                    checkpoint.save(state, force=True)
                    raise InterruptedError("interruption notice")
                if fault == "no_notice":
                    raise HostLost()
        return evaluate_result(workdir, state["predictions"])

    original_put = store.put_file
    canonical_writes = []

    def flaky_put(key, path, **kwargs):
        if "/attempts/" in key:
            canonical_writes.append(key)
            if fault == "publication" and len(canonical_writes) == 2:
                raise OSError("partial S3 publication")
        return original_put(key, path, **kwargs)

    monkeypatch.setattr(worker, "execute_matrix_case", evaluate)
    monkeypatch.setattr(store, "put_file", flaky_put)
    if fault == "no_notice":
        with pytest.raises(HostLost):
            worker.run_unit(store, unit.unit_key, tmp_path / "first", env=first)
        assert get_json(store, first_status)["state"] == "RUNNING"
    else:
        exit_code = worker.run_unit(store, unit.unit_key, tmp_path / "first", env=first)
        status = get_json(store, first_status)
        if fault == "notice":
            assert exit_code == worker.EXIT_INTERRUPTED
            assert status["interruption_reason"] == "spot_interruption"
            assert status["checkpoint"].endswith("/000001/00000002.json")
        else:
            assert exit_code == worker.EXIT_FAILED
            assert status["state"] == "PUBLICATION_FAILED"
    final_key = layout.final_key(unit.campaign_id, unit.run_id)
    assert store.head(final_key) is None

    # Batch retries the same job (attempt 2); the claim numbers it as the run's second attempt.
    second = fargate_env(release, "job-first", batch_attempt=2)
    assert worker.run_unit(store, unit.unit_key, tmp_path / "second", env=second) == worker.EXIT_OK
    final = get_json(store, final_key)
    assert final["publication_attempt"] == {
        "attempt_key": layout.attempt_key_for("job-first", "2"),
        "worker_id": "job-first",
        "attempt": 2,
    }
    if fault != "publication":  # A resumed publication keeps the evaluating attempt's record.
        assert final["attempt"]["recovery_reason"] == {"notice": "INTERRUPTED", "no_notice": "LOST"}[fault]
        assert final["attempt"]["previous_attempt_key"] == first_attempt
    assert store.head(layout.analytics_marker_key(unit.campaign_id, unit.run_id))
    report = get_json(store, split_uri(final["report_uri"])[1])
    assert report["predictions"] == [0, 1, 2, 3]
    if fault == "publication":
        original_prefix = layout.attempt_prefix(unit.campaign_id, unit.run_id, first_attempt)
        assert final["published"]["prefix"] == original_prefix
        assert calls == 1
        assert len(set(canonical_writes)) == final["published"]["files"]
        assert len(list(store.list(original_prefix + "/"))) == final["published"]["files"] + 1
    else:
        assert calls == 2
        assert visits == ([0, 1, 2, 3] if fault == "notice" else [0, 1, 2, 2, 3])
        assert final["attempt"]["resumed_checkpoint"] is not None
        assert final["attempt"]["resumed_checkpoint"] != store.uri(final_key)
    reused = worker.run_unit(store, unit.unit_key, tmp_path / "duplicate", env=second)
    assert reused == worker.EXIT_CONTRACT


@pytest.mark.parametrize("blocker", ["cancelled", "replaced"])
def test_ineligible_attempt_cannot_start_evaluation(spot_case, tmp_path, monkeypatch, blocker):
    store, unit, release = spot_case
    env = fargate_env(release, "job-blocked")
    if blocker == "cancelled":
        put_json(store, layout.cancelled_key(unit.campaign_id), {"state": "CANCELLED"}, if_absent=True)
        expected = worker.EXIT_CANCELLED
    else:
        # A duplicate execution of the same Batch attempt must not evaluate a second time.
        attempt_key = layout.attempt_key_for("job-blocked", "1")
        put_json(
            store,
            layout.spot_claim_key(unit.campaign_id, unit.run_id, 1),
            {"attempt": 1, "attempt_key": attempt_key},
            if_absent=True,
        )
        expected = worker.EXIT_CONTRACT

    def forbidden(*args, **kwargs):
        pytest.fail("ineligible attempt entered evaluator")

    monkeypatch.setattr(worker, "execute_matrix_case", forbidden)
    assert worker.run_unit(store, unit.unit_key, tmp_path / "blocked", env=env) == expected
    assert store.head(layout.final_key(unit.campaign_id, unit.run_id)) is None


def test_zia_spot_worker_recovers_real_predictions_and_fit_lineage(tmp_path, monkeypatch):
    store = LocalObjectStore(tmp_path / "durable")
    snapshot = publish_fixture_snapshot(store, tmp_path)
    release = publish_fixture_release(store)
    snapshot_manifest = verify_snapshot(store, snapshot)

    def unit_for(campaign_id):
        request = CampaignRequest(
            campaign_id=campaign_id,
            snapshot=snapshot.model_dump(),
            matrix={
                "kind": "zia",
                "policies": ["incremental"],
            },
        )
        plan = planning.build_plan(request, release, snapshot_manifest=snapshot_manifest)
        unit = RunUnit(**plan["units"][0])
        put_json(store, unit.unit_key, unit.model_dump(mode="json"))
        return unit

    reference = unit_for("spot-reference")
    ref_env = fargate_env(release, "job-reference")
    assert worker.run_unit(store, reference.unit_key, tmp_path / "reference", env=ref_env) == worker.EXIT_OK

    candidate = unit_for("spot-resumed")
    first = fargate_env(release, "job-candidate")
    original = SpotCheckpoint.save

    def crash_at_confirmed_checkpoint(self, state, **kwargs):
        confirmed = original(self, state, **kwargs)
        if confirmed:
            raise HostLost()
        return confirmed

    with monkeypatch.context() as patch:
        patch.setattr(SpotCheckpoint, "save", crash_at_confirmed_checkpoint)
        with pytest.raises(HostLost):
            worker.run_unit(store, candidate.unit_key, tmp_path / "first", env=first)
    second = fargate_env(release, "job-candidate-resubmitted")
    assert worker.run_unit(store, candidate.unit_key, tmp_path / "resumed", env=second) == worker.EXIT_OK

    def documents(unit):
        final = get_json(store, layout.final_key(unit.campaign_id, unit.run_id))
        report = get_json(store, split_uri(final["report_uri"])[1])
        predictions = get_json(store, split_uri(final["predictions_manifest_uri"])[1])
        return final, report, predictions

    _, expected_report, expected_predictions = documents(reference)
    final, actual_report, actual_predictions = documents(candidate)
    assert final["attempt"]["resumed_checkpoint"].endswith("/000001/00000001.json")
    assert [part["sha256"] for part in actual_predictions["parts"]] == [
        part["sha256"] for part in expected_predictions["parts"]
    ]
    assert [(fit["fit_id"], fit["checkpoint"]["sha256"]) for fit in actual_report["training_history"]] == [
        (fit["fit_id"], fit["checkpoint"]["sha256"]) for fit in expected_report["training_history"]
    ]
    assert actual_report["holdout_confusion"] == expected_report["holdout_confusion"]
    assert not any("/.zia-resume/" in key for key in store.list(final["published"]["prefix"] + "/"))


def test_spot_notice_flushes_mlflow_async_queue_before_the_task_stops(spot_case, tmp_path, monkeypatch):
    """SIGTERM: checkpoint at the next safe point, drain queued MLflow writes, exit 75 for a retry."""
    store, unit, release = spot_case
    events = []

    class Tracker:
        mlflow_run_id = "run"

        def __init__(self, *args, **kwargs):
            pass

        def bind_spot_attempt(self, *args, **kwargs):
            events.append("bind")

        def start_live(self):
            return True

        def metrics(self, step, values, prefix="progress"):
            events.append(("metric", step))

        def record_spot_attempt(self, state, reason=None, checkpoint=None):
            events.append(("state", state, bool(checkpoint)))
            return True

        def flush(self):
            events.append("flush")
            return True

    def evaluate(_store, _unit, workdir, *, checkpoint, progress=None):
        progress(1, {"rows": 1})
        spot_notice()
        checkpoint.save({"cursor": 1}, force=True)
        raise InterruptedError("Spot interruption")

    from continual_bench.infrastructure.aws import matrix_tracking

    monkeypatch.setattr(matrix_tracking, "MatrixMlflowRun", Tracker)
    monkeypatch.setattr(worker, "execute_matrix_case", evaluate)
    env = {**fargate_env(release, "job-notice"), "MLFLOW_TRACKING_URI": "http://mlflow.example"}
    assert worker.run_unit(store, unit.unit_key, tmp_path / "run", env=env) == worker.EXIT_INTERRUPTED
    assert ("state", "INTERRUPTED", True) in events
    assert events.index("flush") > events.index(("state", "INTERRUPTED", True))
    status = get_json(
        store,
        layout.attempt_status_key(
            unit.campaign_id,
            unit.run_id,
            layout.attempt_key_for("job-notice", "1"),
        ),
    )
    assert status["state"] == "INTERRUPTED" and status["checkpoint"].endswith("/000001/00000001.json")


@pytest.mark.parametrize("fault", ["interruption", "publication"])
def test_metrics_only_spot_restarts_training_but_recovers_completed_metric_publication(
    spot_case, tmp_path, monkeypatch, fault
):
    store, original, release = spot_case
    params = {**original.params, "config": {**original.params["config"], "artifact_mode": "metrics_only"}}
    contract = {**original.contract, "params": params}
    run_id, contract_sha256 = run_identity(original.campaign_id, contract)
    unit = original.model_copy(
        update={
            "params": params,
            "contract": contract,
            "run_id": run_id,
            "contract_sha256": contract_sha256,
            "unit_key": layout.unit_key(original.campaign_id, run_id),
        }
    )
    put_json(store, unit.unit_key, unit.model_dump(mode="json"))
    calls = 0
    visits = []

    def forbidden_checkpoint(*args, **kwargs):
        pytest.fail("metrics-only Spot must never construct a training checkpoint")

    def evaluate(_store, _unit, workdir, *, stop_requested, checkpoint=None):
        nonlocal calls
        calls += 1
        assert checkpoint is None
        for index in range(4):
            visits.append(index)
            if fault == "interruption" and calls == 1 and index == 1:
                spot_notice()
                assert stop_requested()
                raise InterruptedError("no model state retained")
        directory = workdir / "run" / "binary" / "metrics-only-test"
        directory.mkdir(parents=True)
        report = {
            "complete": True,
            "run_key": "metrics-only-test",
            "config": {"artifact_mode": "metrics_only"},
            "metrics": {"accuracy": 0.5},
            "windows": [],
        }
        report_path = directory / "report.json"
        report_path.write_text(json.dumps(report))
        (directory / "metrics.json").write_text(json.dumps({"points": [0, 1, 2, 3]}))
        return {
            "report": report,
            "report_path": report_path,
            "output": workdir / "run",
            "headline": {"accuracy": 0.5},
        }

    put_file = store.put_file
    canonical_writes = 0

    def fail_publication(key, path, **kwargs):
        nonlocal canonical_writes
        if "/attempts/" in key:
            canonical_writes += 1
            if fault == "publication" and canonical_writes == 2:
                raise OSError("metric publication interrupted")
        return put_file(key, path, **kwargs)

    monkeypatch.setattr(worker, "SpotCheckpoint", forbidden_checkpoint)
    monkeypatch.setattr(worker, "execute_matrix_case", evaluate)
    monkeypatch.setattr(store, "put_file", fail_publication)
    first = fargate_env(release, "metrics-only-job")
    expected = worker.EXIT_INTERRUPTED if fault == "interruption" else worker.EXIT_FAILED
    assert worker.run_unit(store, unit.unit_key, tmp_path / "first", env=first) == expected
    second = fargate_env(release, "metrics-only-job", 2)
    assert worker.run_unit(store, unit.unit_key, tmp_path / "second", env=second) == worker.EXIT_OK
    final = get_json(store, layout.final_key(unit.campaign_id, unit.run_id))
    assert final["artifact_mode"] == "metrics_only"
    assert final["attempt"]["checkpoint"] is None
    assert final["publication_attempt"]["attempt"] == 2
    keys = list(store.list(layout.run_prefix(unit.campaign_id, unit.run_id)))
    assert all("/spot/checkpoints/" not in key for key in keys)
    assert all(not key.endswith((".pkl", ".pickle", ".bin", ".ubj")) for key in keys)
    assert calls == (2 if fault == "interruption" else 1)
    assert visits == ([0, 1, 0, 1, 2, 3] if fault == "interruption" else [0, 1, 2, 3])
    if fault == "publication":
        first_attempt = layout.attempt_key_for("metrics-only-job", "1")
        assert final["published"]["prefix"] == layout.attempt_prefix(
            unit.campaign_id, unit.run_id, first_attempt
        )


def test_metrics_only_spot_live_series_are_independent_between_restarted_attempts(spot_case, monkeypatch):
    from continual_bench.infrastructure.aws.matrix_tracking import MatrixMlflowRun

    _, original, _ = spot_case
    params = {**original.params, "config": {**original.params["config"], "artifact_mode": "metrics_only"}}
    unit = original.model_copy(update={"params": params})
    monkeypatch.setattr(MatrixMlflowRun, "native_logger", property(lambda self: object()))
    points = []
    for number in (1, 2):
        tracker = MatrixMlflowRun(unit, f"attempt-{number}", "job", "http://example", None)
        tracker.mlflow_run_id = "logical-run"
        monkeypatch.setattr(tracker, "log_progress_metrics", lambda items: points.extend(items))
        tracker.metrics(1, {"training_seconds": float(number)})
    assert [(point["key"], point["step"], point["value"]) for point in points] == [
        ("attempts/attempt-1/progress/training_seconds", 1, 1.0),
        ("attempts/attempt-2/progress/training_seconds", 1, 2.0),
    ]
