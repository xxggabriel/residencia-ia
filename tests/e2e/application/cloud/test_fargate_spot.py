"""Fargate Spot campaigns: all units at once, numbered resumable attempts, MLflow async logging."""

import pytest

from continual_bench.application.cloud import layout, planning
from continual_bench.application.cloud.checkpoints import SpotAttempt, SpotGrantError
from continual_bench.application.cloud.contracts import SPOT_BACKEND, CampaignRequest, RunUnit
from continual_bench.application.cloud.fixtures import publish_fixture_release
from continual_bench.infrastructure.object_store import LocalObjectStore, put_json

QUEUE = {"queue_state": "ENABLED", "queue_status": "VALID", "max_vcpus": 64}


def test_spot_admission_starts_what_the_64_vcpu_quota_allows_and_queues_the_rest():
    evidence = planning.admit_capacity(40, 2, quota_vcpus=64, queue=QUEUE, queue_excess=True)
    assert evidence["max_parallel_admitted"] == 32
    assert (evidence["running_in_parallel"], evidence["queued"]) == (32, 8)
    small = planning.admit_capacity(12, 2, quota_vcpus=64, queue=QUEUE, queue_excess=True)
    assert (small["running_in_parallel"], small["queued"]) == (12, 0)
    with pytest.raises(planning.CapacityError, match="exceeds the usable limit"):
        planning.admit_capacity(1, 2, quota_vcpus=1, queue=QUEUE, queue_excess=True)
    # On-demand campaigns still refuse parallelism beyond the quota.
    with pytest.raises(planning.CapacityError, match="regional Fargate quota"):
        planning.admit_capacity(40, 2, quota_vcpus=64, queue={**QUEUE, "max_vcpus": 256})


def spot_release(store):
    release = publish_fixture_release(store)
    return release.model_copy(update={"benchmark_compute": SPOT_BACKEND})


def test_spot_plan_submits_every_unit_in_parallel(tmp_path):
    release = spot_release(LocalObjectStore(tmp_path))
    request = CampaignRequest(
        campaign_id="fargate-all-at-once",
        matrix={
            "kind": "experiment_matrix",
            "path": "configs/matrices/binary-comparison.yaml",
        },
    )
    plan = planning.build_plan(request, release)
    assert plan["concurrency"] == min(32, len(plan["units"]))
    assert all(unit["contract"]["release"]["compute"] == SPOT_BACKEND for unit in plan["units"])
    limited = planning.build_plan(request.model_copy(update={"max_parallel": 3}), release)
    assert limited["concurrency"] == 3


def test_attempts_are_numbered_across_batch_retries_and_resubmissions(tmp_path):
    store = LocalObjectStore(tmp_path)
    release = spot_release(store)
    request = CampaignRequest(
        campaign_id="fargate-claims",
        matrix={
            "kind": "experiment_matrix",
            "path": "configs/matrices/binary-comparison.yaml",
        },
    )
    unit = RunUnit(**planning.build_plan(request, release)["units"][0])
    received = []

    def attempt(job, batch_attempt):
        return SpotAttempt(
            store,
            unit,
            job,
            layout.attempt_key_for(job, batch_attempt),
            stop_requested=lambda: bool(received),
        )

    first = attempt("job-a", 1)
    assert first.acquire()["attempt"] == 1 and first.reason() is None
    put_json(
        store,
        layout.attempt_status_key(unit.campaign_id, unit.run_id, first.attempt_key),
        {"state": "INTERRUPTED"},
    )
    retry = attempt("job-a", 2)
    grant = retry.acquire()
    assert grant["attempt"] == 2
    assert grant["previous_attempt_key"] == first.attempt_key and grant["recovery_reason"] == "INTERRUPTED"
    resubmitted = attempt("job-b", 1)
    assert resubmitted.acquire()["recovery_reason"] == "LOST"  # job-a-2 never wrote its status
    with pytest.raises(SpotGrantError, match="newer attempt"):
        retry.check_publication(strict=True)
    with pytest.raises(SpotGrantError, match="already claimed"):
        attempt("job-b", 1).acquire()
    received.append("SIGTERM")
    assert resubmitted.reason() == "spot_interruption"
    with pytest.raises(InterruptedError):
        resubmitted.check_publication()


def test_live_metrics_use_mlflow_native_async_logging_and_flush(tmp_path, monkeypatch):
    from mlflow import MlflowClient

    from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker
    from continual_bench.infrastructure.tracking.mlflow_async import NativeAsyncLogger

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")
    uri = (tmp_path / "mlruns").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("async")
    run_id = client.create_run(experiment).info.run_id
    # FileStore's async writers race when creating the shared metric namespace.
    (tmp_path / "mlruns" / experiment / run_id / "metrics" / "progress").mkdir(exist_ok=True)

    tracker = SageMakerMlflowTracker(
        tracking_uri=uri, dataset="sea", policy="periodic", model_name="xgboost", run_id="unit"
    )
    tracker.mlflow_run_id = run_id
    rest_calls = []
    monkeypatch.setattr(tracker, "_request", lambda *a, **k: rest_calls.append(a) or {})
    assert isinstance(tracker.native_logger, NativeAsyncLogger)
    for step in range(1, 51):
        tracker.progress(step, {"rows": step, "accuracy": step / 100})
    assert tracker.log_epoch_metrics({"loss": 0.25}, step=1)
    assert tracker.flush()
    assert rest_calls == []  # Live points never took the synchronous REST path.
    rows = client.get_metric_history(run_id, "progress/rows")
    assert sorted((m.step, m.value) for m in rows) == [(step, float(step)) for step in range(1, 51)]
    accuracy = client.get_metric_history(run_id, "progress/accuracy")
    assert sorted((m.step, m.value) for m in accuracy) == [(step, step / 100) for step in range(1, 51)]
    assert [(m.step, m.value) for m in client.get_metric_history(run_id, "loss")] == [(1, 0.25)]


def test_without_async_logging_the_rest_transport_is_kept(monkeypatch):
    from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker

    monkeypatch.delenv("MLFLOW_ENABLE_ASYNC_LOGGING", raising=False)
    tracker = SageMakerMlflowTracker(
        tracking_uri="http://mlflow.example",
        dataset="sea",
        policy="periodic",
        model_name="xgboost",
        run_id="unit",
    )
    assert tracker.native_logger is None
