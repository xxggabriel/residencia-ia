import json
import signal
from pathlib import Path

import pytest

from continual_bench.application.cloud import layout, planning, worker
from continual_bench.application.cloud.contracts import CampaignRequest
from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json
from continual_bench.infrastructure.zia_training_tracking import publish_sdk_training_rounds


def fit(fit_id, kind, first, values, deployed_version):
    return {
        "fit_id": fit_id,
        "action": "online" if kind == "minibatch" else "initial",
        "start_seq": 0,
        "end_seq": 12,
        "block_index": 0,
        "deployed_version": deployed_version,
        "train_samples": 10,
        "validation_samples": 2,
        "training_presentations": 20,
        "steps": [
            {"step": step, "global_step": first + step - 1, "kind": kind, "metrics": {"loss": loss}}
            for step, loss in enumerate(values, 1)
        ],
        "summary": {"best_step": 2},
        "checkpoint": {"file": "models/0000.npy", "sha256": "a" * 64},
    }


def test_each_fit_has_its_own_steps_and_retry_preserves_the_child_runs(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    experiment = client.create_experiment("zia-test")
    parent = client.create_run(experiment).info.run_id
    report = {
        "training_history": [
            fit("fit-000001", "epoch", 1, [0.8, 0.6], 0),
            fit("fit-000002", "minibatch", 3, [0.4, 0.25], 1),
        ],
    }
    receipt = publish_sdk_training_rounds(client, experiment, parent, report)
    assert publish_sdk_training_rounds(client, experiment, parent, report) == receipt
    for child, losses, global_steps, kind in zip(
        receipt, ([0.8, 0.6], [0.4, 0.25]), ([1, 2], [3, 4]), ("epoch", "minibatch"), strict=True
    ):
        child_id = child["run_id"]
        run = client.get_run(child_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["mlflow.parentRunId"] == parent and run.data.tags["cb-step-kind"] == kind
        assert run.data.params["checkpoint_sha256"] == "a" * 64
        assert [(p.step, p.value) for p in client.get_metric_history(child_id, "loss")] == [
            (1, losses[0]), (2, losses[1])
        ]
        assert [p.value for p in client.get_metric_history(child_id, "global_step")] == global_steps


def test_rest_publisher_marks_only_verified_evidence_complete(tmp_path, monkeypatch):
    tracker = SageMakerMlflowTracker(
        tracking_uri="http://mlflow.example", dataset="social", policy="online", model_name="logistic",
        run_id="my-unit", attempt_key="job-1",
    )
    tracker.mlflow_run_id = "parent"
    tracker.experiment_id = "experiment"
    tracker.artifact_uri = "s3://artifact-store/mlflow/parent"
    store = LocalObjectStore(tmp_path / "store")
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.json").write_text("{}")
    events = []

    def remote(method, url, *, payload=None, **_kwargs):
        events.append((method, url, payload))
        if url.endswith("/runs/create"):
            return {"run": {"info": {"run_id": "child", "status": "RUNNING"}}}
        return {}

    monkeypatch.setattr(tracker, "_request", remote)
    monkeypatch.setattr(tracker, "register_run_models", lambda **_: {})
    report = {"training_history": [fit("fit-000001", "epoch", 1, [0.8, 0.6], 0)]}
    adapted = {
        "tracking": {
            "version": "residency-metrics-v1", "metrics": {}, "availability": {}, "settings": {},
            "series": {"fading_accuracy": [{"step": 18, "value": 0.7}]},
        }
    }
    result = tracker.publish_run_report(report, adapted=adapted, output_dir=output, store=store)
    assert result["child_runs"][0]["fit_id"] == "fit-000001" and result["published"]
    child_log = next(event[2] for event in events if (event[2] or {}).get("metrics")
                     and event[2]["run_id"] == "child")
    assert [(p["step"], p["value"]) for p in child_log["metrics"] if p["key"] == "loss"] == [
        (1, 0.8), (2, 0.6),
    ]
    assert any(
        p["key"] == "tracking_v1/fading_accuracy" and p["step"] == 18
        for _, _, payload in events for p in (payload or {}).get("metrics", [])
    )
    assert events[-2][2]["status"] == "FINISHED"
    assert {"key": "benchmark_complete", "value": "true"} in events[-1][2]["tags"]
    assert tracker.run_key == "my-unit-job-1"
    with pytest.raises(ValueError, match="belongs to another attempt"):
        tracker.bind_attempt("job-2")

    events.clear()
    monkeypatch.setattr(tracker, "finish", lambda *_: False)
    result = tracker.publish_run_report(report, adapted=adapted, output_dir=output, store=store)
    assert not result["published"]
    assert all(
        tag != {"key": "benchmark_complete", "value": "true"}
        for _, _, payload in events for tag in (payload or {}).get("tags", [])
    )


def test_worker_retries_failed_tracking_from_archived_output(tmp_path, monkeypatch):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path, "social")
    request = CampaignRequest(
        campaign_id="zia-publish-retry-test",
        snapshot=snapshot.model_dump(),
        matrix={"kind": "zia", "policies": ["frozen"]},
    )
    plan = planning.build_plan(
        request, publish_fixture_release(store), snapshot_manifest=verify_snapshot(store, snapshot),
    )
    planning.publish_plan(store, plan)
    unit = plan["units"][0]
    events = []

    class Tracker:
        def __init__(self, failing):
            self.failing = failing

        def bind_attempt(self, attempt_key):
            assert attempt_key == "initial-1"

        def start_live(self):
            events.append("started")
            raise OSError("MLflow unavailable at startup")

        def progress(self, step, _values):
            events.append(step)
            if step == 0:
                raise OSError("MLflow unavailable for the first progress point")

        def publish_run_report(self, report, *, adapted, output_dir, **_kwargs):
            if self.failing:
                return {"published": False, "errors": ["stream endpoint offline"]}
            assert adapted["tracking"]["series"]
            assert (output_dir / "training-history.json").is_file()
            assert (output_dir / "model/model.tar.gz").is_file()
            child_runs = [{"fit_id": round_["fit_id"], "run_id": "child"}
                          for round_ in report["training_history"]]
            return {
                "published": True,
                "mlflow_run_id": "parent",
                "child_runs": child_runs,
                "artifacts_uploaded": {"files": sum(path.is_file() for path in output_dir.rglob("*"))},
            }

    final_key = layout.final_key(request.campaign_id, unit["run_id"])

    def registry_error(*_args, **_kwargs):
        raise RuntimeError("registry unavailable")

    materialize = worker.materialize

    def read_data(*args, **kwargs):
        assert events == ["started"]
        events.append("downloading")
        return materialize(*args, **kwargs)

    monkeypatch.setattr(worker, "materialize", read_data)
    assert worker.run_unit(
        store, unit["unit_key"], tmp_path / "first", tracker=Tracker(True),
        registry=registry_error,
        env={"AWS_BATCH_JOB_ID": "initial", "AWS_BATCH_JOB_ATTEMPT": "1"},
    ) == worker.EXIT_FAILED
    assert events[:3] == ["started", "downloading", 0]
    assert events[-1] > 0
    assert store.head(final_key) is None
    failed = get_json(store, layout.attempt_status_key(request.campaign_id, unit["run_id"], "initial-1"))
    assert failed["state"] == "PUBLICATION_FAILED"
    assert "registry unavailable" in failed["diagnostic"]["message"]

    def unexpected_training(*_args, **_kwargs):
        pytest.fail("tracking retry must reuse the original training evidence")

    def registry(_unit, _report, metadata, archive_uri, verified_snapshot):
        assert metadata["state_sha256"] and archive_uri.endswith("/model/model.tar.gz")
        assert verified_snapshot["snapshot_id"] == snapshot.snapshot_id
        return {"model_package_arn": "arn:batch-package"}

    restore = worker.restore_benchmark_evidence

    def restore_after_live_start(*args, **kwargs):
        assert events[-1] == "started"
        events.append("restoring")
        return restore(*args, **kwargs)

    monkeypatch.setattr(worker, "execute_zia", unexpected_training)
    monkeypatch.setattr(worker, "restore_benchmark_evidence", restore_after_live_start)
    assert worker.run_unit(
        store, unit["unit_key"], tmp_path / "second", tracker=Tracker(True), registry=registry,
        env={"AWS_BATCH_JOB_ID": "retry-tracking", "AWS_BATCH_JOB_ATTEMPT": "1"},
    ) == worker.EXIT_FAILED
    assert store.head(final_key) is None
    retry_key = layout.attempt_status_key(request.campaign_id, unit["run_id"], "retry-tracking-1")
    failed = get_json(store, retry_key)
    assert failed["state"] == "TRACKING_FAILED"
    assert "stream endpoint offline" in failed["diagnostic"]["message"]
    assert worker.run_unit(
        store, unit["unit_key"], tmp_path / "third", tracker=Tracker(False), registry=registry,
        env={"AWS_BATCH_JOB_ID": "retry-completed", "AWS_BATCH_JOB_ATTEMPT": "1"},
    ) == worker.EXIT_OK
    final = get_json(store, final_key)
    assert final["attempt"]["attempt_key"] == "initial-1"
    assert final["model"]["model_package_arn"] == "arn:batch-package"
    assert final["tracking"]["complete"] and final["tracking"]["child_runs"][0]["fit_id"] == "fit-000001"
    assert events.count("started") == 3
    assert events[-4:] == ["started", "restoring", "started", "restoring"]
    archived = Path(final["training_history_uri"].removeprefix("file://"))
    assert json.loads(archived.read_text())["fits"][0]["fit_id"] == "fit-000001"


@pytest.mark.parametrize("interrupted", [False, True], ids=["failure", "sigterm"])
def test_worker_closes_live_zia_run_on_failure(tmp_path, monkeypatch, interrupted):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path, "social")
    request = CampaignRequest(
        campaign_id="zia-live-failure-test",
        snapshot=snapshot.model_dump(),
        matrix={"kind": "zia", "policies": ["frozen"]},
    )
    plan = planning.build_plan(
        request, publish_fixture_release(store), snapshot_manifest=verify_snapshot(store, snapshot),
    )
    planning.publish_plan(store, plan)
    [unit] = plan["units"]
    events = []

    class Tracker:
        def bind_attempt(self, key):
            events.append(key)

        def start_live(self):
            events.append("RUNNING")

        def progress(self, step, values):
            assert step == 10 and values == {"rows": 10}
            events.append("progress")

        def finish(self, status):
            events.append(status)

        def mark_failed(self, message):
            assert message == "benchmark error"
            events.append("FAILED")

        def publish_run_report(self, *_args, **_kwargs):
            pytest.fail("failed evaluation cannot publish a final report")

    def abort(_unit, _store, _path, _log, *, stop_requested, progress):
        assert not stop_requested()
        progress(10, {"rows": 10})
        if interrupted:
            signal.raise_signal(signal.SIGTERM)
            assert stop_requested()
            raise InterruptedError("benchmark was interrupted")
        raise RuntimeError("benchmark error")

    flush = worker.flush_partial_state

    def verify_before_flush(*args, **kwargs):
        assert events[-1] == "KILLED"
        return flush(*args, **kwargs)

    monkeypatch.setattr(worker, "execute_zia", abort)
    monkeypatch.setattr(worker, "flush_partial_state", verify_before_flush)
    assert worker.run_unit(
        store, unit["unit_key"], tmp_path / "work", tracker=Tracker(),
        env={"AWS_BATCH_JOB_ID": "live", "AWS_BATCH_JOB_ATTEMPT": "3"},
    ) == worker.EXIT_FAILED
    assert events == ["live-3", "RUNNING", "progress", "KILLED" if interrupted else "FAILED"]
    status = get_json(store, layout.attempt_status_key(request.campaign_id, unit["run_id"], "live-3"))
    assert status["state"] == ("INTERRUPTED" if interrupted else "FAILED")
    assert store.head(layout.final_key(request.campaign_id, unit["run_id"])) is None


def test_worker_stops_after_sigterm_during_zia_publication(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path, "social")
    request = CampaignRequest(
        campaign_id="zia-live-publication-interrupted",
        snapshot=snapshot.model_dump(),
        matrix={"kind": "zia", "policies": ["frozen"]},
    )
    plan = planning.build_plan(
        request, publish_fixture_release(store), snapshot_manifest=verify_snapshot(store, snapshot),
    )
    planning.publish_plan(store, plan)
    [unit] = plan["units"]
    events = []

    class Tracker:
        def start_live(self):
            events.append("RUNNING")

        def mark_killed(self):
            events.append("KILLED")

        def publish_run_report(self, report, *, output_dir, **_kwargs):
            events.append("publishing")
            signal.raise_signal(signal.SIGTERM)
            children = [{"fit_id": fit["fit_id"], "run_id": "child"} for fit in report["training_history"]]
            return {
                "published": True,
                "mlflow_run_id": "parent",
                "child_runs": children,
                "artifacts_uploaded": {"files": sum(path.is_file() for path in output_dir.rglob("*"))},
            }

    assert worker.run_unit(
        store, unit["unit_key"], tmp_path / "work", tracker=Tracker(),
        env={"AWS_BATCH_JOB_ID": "live", "AWS_BATCH_JOB_ATTEMPT": "1"},
    ) == worker.EXIT_FAILED
    assert events == ["RUNNING", "publishing", "KILLED"]
    status = get_json(store, layout.attempt_status_key(request.campaign_id, unit["run_id"], "live-1"))
    assert status["state"] == "INTERRUPTED"
    assert store.head(layout.final_key(request.campaign_id, unit["run_id"])) is None
    pending = f"{layout.run_prefix(request.campaign_id, unit['run_id'])}/pending-publication.json"
    assert get_json(store, pending)["attempt"]["attempt_key"] == "live-1"
