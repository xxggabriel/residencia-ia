from continual_bench.infrastructure.aws import mlflow_tracking
from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker


def test_live_progress_and_final_evidence_share_the_parent(monkeypatch):
    tracker = SageMakerMlflowTracker(
        tracking_uri="http://mlflow.example", dataset="ecommerce", policy="online",
        model_name="return-booster", run_id="benchmark", attempt_key="batch-1",
    )
    requests = []

    def request(method, path, *, payload=None, params=None):
        requests.append((path, payload))
        if path.endswith("/get-by-name"):
            return {"experiment": {"experiment_id": "1"}}
        if path.endswith("/runs/create"):
            return {"run": {"info": {"run_id": "parent", "status": "RUNNING"}}}
        return {}

    clock = [100.0]
    monkeypatch.setattr(mlflow_tracking.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tracker, "_request", request)
    monkeypatch.setattr(tracker, "register_run_models", lambda **_: {})

    assert tracker.start_live()
    assert requests[-2][0].endswith("/runs/create")
    assert {"key": "zia_run_key", "value": "benchmark-batch-1"} in requests[-2][1]["tags"]
    assert {"key": "benchmark_complete", "value": "false"} in requests[-1][1]["tags"]

    tracker.progress(10, {"rows_processed": 10, "prequential/accuracy": float("nan")})
    tracker.progress(11, {"rows_processed": 11, "matrix/blocks_completed": 2})
    tracker.progress(12, {"rows_processed": 12})
    tracker.progress(9, {"rows_processed": 9})
    clock[0] += mlflow_tracking.PROGRESS_INTERVAL_SECONDS
    tracker.progress(13, {"rows_processed": 13, "prequential/accuracy": 0.7})
    tracker.progress(14, {"rows_processed": 14})
    assert all(not (data or {}).get("status") for _, data in requests)
    result = tracker.publish_run_report({"metrics": {"accuracy": 0.8}})
    assert result["published"] and result["mlflow_run_id"] == "parent"
    assert sum(path.endswith("/runs/create") for path, _ in requests) == 1
    assert all(data["run_id"] == "parent" for _, data in requests if data and "run_id" in data)
    points = [(p["key"], p["step"], p["value"]) for _, data in requests
              for p in (data or {}).get("metrics", [])]
    assert [point for point in points if point[0].startswith("progress/")] == [
        ("progress/rows_processed", 10, 10.0),
        ("progress/rows_processed", 13, 13.0),
        ("progress/matrix/blocks_completed", 11, 2.0),
        ("progress/prequential/accuracy", 13, 0.7),
        ("progress/rows_processed", 14, 14.0),
    ]
    assert ("summary/accuracy", 0, 0.8) in points
    assert requests[-2][1]["status"] == "FINISHED"


def test_progress_transport_failure_uses_the_next_snapshot(monkeypatch):
    tracker = SageMakerMlflowTracker(
        tracking_uri="http://mlflow.example", dataset="ecommerce", policy="online",
        model_name="return-booster", run_id="benchmark", attempt_key="batch-1",
    )
    tracker.mlflow_run_id = "parent"
    steps = []
    clock = [100.0]

    def request(_method, _path, *, payload):
        steps.append(payload["metrics"][0]["step"])
        if steps == [1]:
            raise TimeoutError("temporary tracking failure")
        return {}

    monkeypatch.setattr(mlflow_tracking.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tracker, "_request", request)
    tracker.progress(1, {"rows_processed": 1})
    tracker.progress(1, {"rows_processed": 1})
    tracker.progress(2, {"rows_processed": 2})
    clock[0] += mlflow_tracking.PROGRESS_INTERVAL_SECONDS
    tracker.progress(3, {"rows_processed": 3})

    assert steps == [1, 3]
    assert "temporary tracking failure" in tracker.api_errors[0]
