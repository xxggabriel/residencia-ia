"""Native child payloads are generated only as bounded publication slots open."""

import threading
import time
from types import SimpleNamespace

import pytest

from continual_bench.infrastructure import training_tracking
from continual_bench.infrastructure.tracking.backends import MlflowClientBackend
from continual_bench.infrastructure.tracking.rounds import publish_training_rounds


class RecordingClient:
    def __init__(self, concurrency):
        self.runs = {}
        self.points = {}
        self.lock = threading.Lock()
        self.started = set()
        self.completed = []
        self.first_batch = threading.Barrier(concurrency)

    def search_runs(self, *_args, **_kwargs):
        return list(self.runs.values())

    def create_run(self, experiment, *, tags):
        key = tags["cb-fit-id"]
        with self.lock:
            run = SimpleNamespace(
                info=SimpleNamespace(
                    run_id=key,
                    status="RUNNING",
                    artifact_uri=f"s3://fixture-artifacts/{key}",
                ),
                data=SimpleNamespace(tags=dict(tags), metrics={}),
            )
            self.runs[key] = run
            self.points[key] = []
        return run

    def get_metric_history(self, run_id, key):
        return [
            SimpleNamespace(step=step, value=value)
            for name, step, value in self.points[run_id]
            if name == key
        ]

    def log_batch(self, run_id, *, metrics=(), params=(), tags=(), **_kwargs):
        with self.lock:
            first = run_id not in self.started
            self.started.add(run_id)
        if first and len(self.started) <= self.first_batch.parties:
            self.first_batch.wait(timeout=5)
        if first and run_id == "fit-000001":
            # Later fits can finish first, but receipt order must remain stable.
            time.sleep(0.02)
        with self.lock:
            self.points[run_id].extend((point.key, point.step, point.value) for point in metrics)
            self.runs[run_id].data.tags.update({tag.key: tag.value for tag in tags})
            self.runs[run_id].data.metrics.update({point.key: point.value for point in metrics})

    def set_terminated(self, run_id, status):
        with self.lock:
            self.runs[run_id].info.status = status
            self.completed.append(run_id)


def report(seed, *, fits=12, steps=1001):
    return {
        "tracking_observations": {"version": "bounded-publication-fixture"},
        "training_history": [
            {
                "fit_id": f"fit-{index + 1:06d}",
                "action": "candidate",
                "seed": seed,
                "start_seq": 0,
                "end_seq": 10,
                "steps": [
                    {
                        "step": step,
                        "global_step": index * steps + step,
                        "kind": "epoch",
                        "metrics": {"loss": 1 / step, "accuracy": seed / 100},
                    }
                    for step in range(1, steps + 1)
                ],
                "summary": {},
            }
            for index in range(fits)
        ],
    }


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("concurrency", [1, 2, 4])
def test_payload_generation_is_bounded_and_all_metrics_are_published(monkeypatch, seed, concurrency):
    client = RecordingClient(concurrency)
    backend = MlflowClientBackend(client, concurrency=concurrency)
    source = report(seed)
    original = training_tracking.training_round_mlflow
    generated, outstanding_peak = [], 0

    def bounded(*args, **kwargs):
        nonlocal outstanding_peak
        for payload in original(*args, **kwargs):
            with client.lock:
                generated.append(payload[0]["cb-fit-id"])
                outstanding = len(generated) - len(client.completed)
            outstanding_peak = max(outstanding_peak, outstanding)
            assert outstanding <= concurrency, "generated native metrics exceed publication capacity"
            yield payload

    monkeypatch.setattr(training_tracking, "training_round_mlflow", bounded)
    receipts = publish_training_rounds(
        backend, "experiment", "parent", source, now_ms=7, include_artifact_uri=True
    )
    expected = [fit["fit_id"] for fit in source["training_history"]]
    assert [receipt["fit_id"] for receipt in receipts] == generated == expected
    assert outstanding_peak == concurrency
    assert sorted(client.completed) == expected
    for index, receipt in enumerate(receipts):
        run_id = receipt["run_id"]
        points = client.points[run_id]
        assert receipt["metric_points"] == len(points) == 3003
        assert len({(key, step) for key, step, _ in points}) == len(points)
        assert receipt["artifact_uri"] == f"s3://fixture-artifacts/{run_id}"
        assert client.runs[run_id].info.status == "FINISHED"
        assert client.runs[run_id].data.tags["cb-seed"] == str(seed)
        assert sorted((step, value) for key, step, value in points if key == "global_step") == [
            (step, index * 1001 + step) for step in range(1, 1002)
        ]

    # Reuse finished children without adding any metric points or changing receipts.
    monkeypatch.setattr(training_tracking, "training_round_mlflow", original)
    counts = {key: len(values) for key, values in client.points.items()}
    assert (
        publish_training_rounds(backend, "experiment", "parent", source, now_ms=7, include_artifact_uri=True)
        == receipts
    )
    assert {key: len(values) for key, values in client.points.items()} == counts


@pytest.mark.parametrize("concurrency", [1, 4])
def test_bounded_publication_preserves_transport_exception(monkeypatch, concurrency):
    client = RecordingClient(concurrency)
    backend = MlflowClientBackend(client, concurrency=concurrency)
    original = RuntimeError("transport failed")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(client, "log_batch", fail)
    with pytest.raises(RuntimeError) as caught:
        publish_training_rounds(backend, "experiment", "parent", report(42, fits=6, steps=2))
    assert caught.value is original
    assert len(client.runs) <= concurrency


def test_invalid_later_fit_leaves_earlier_child_reusable_without_duplicate_metrics():
    client = RecordingClient(1)
    backend = MlflowClientBackend(client, concurrency=1)
    source = report(42, fits=2, steps=2)
    source["training_history"][1]["fit_id"] = "fit-000001"
    with pytest.raises(ValueError, match="invalid or duplicate fit"):
        publish_training_rounds(backend, "experiment", "parent", source, now_ms=7)
    assert client.completed == ["fit-000001"]
    original_points = list(client.points["fit-000001"])
    source["training_history"][1]["fit_id"] = "fit-000002"
    receipts = publish_training_rounds(backend, "experiment", "parent", source, now_ms=7)
    assert [receipt["fit_id"] for receipt in receipts] == ["fit-000001", "fit-000002"]
    assert client.points["fit-000001"] == original_points
    assert len(client.points["fit-000002"]) == len(original_points) == 6
