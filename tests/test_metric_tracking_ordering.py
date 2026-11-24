"""A buffered initial status must never overwrite verified publication completion."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from continual_bench.infrastructure import metric_tracking


def _report(directory, seed):
    report = {
        "environment": {"code_sha256": "c" * 64},
        "dataset": {"sha256": "d" * 64},
        "tracking": {
            "version": metric_tracking.VERSION,
            "metrics": {"accuracy": 0.75, "roc_auc": None},
            "series": {
                "windows/accuracy": [{"step": i, "value": ((i + seed) % 7) / 6} for i in range(600)],
                "windows/n": [{"step": i, "value": i + 1} for i in range(600)],
            },
            "availability": {"roc_auc": "single_class"},
            "settings": {"seed": seed},
        },
    }
    (directory / "report.json").write_text(json.dumps(report, sort_keys=True))
    (directory / "predictions.jsonl").write_text('{"seq":0,"label":1,"p":0.75}\n')
    return report


def _expected_points(report):
    return {
        (key, step, float(value))
        for key, step, value in metric_tracking.metric_points(report["tracking"])
        if value is not None
    }


class BufferedTagClient:
    """Model global async logging: default set_tag waits until after sync batches."""

    def __init__(self):
        self.tags, self.histories, self.artifacts = {}, {}, {}
        self.pending, self.status_events, self.sync_batches = [], [], []
        self.tag_events = []

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, artifact_uri="file:///unused"),
            data=SimpleNamespace(
                metrics={k: rows[-1].value for k, rows in self.histories.items()},
                tags=self.tags,
            ),
        )

    def get_metric_history(self, run_id, key):
        return self.histories.get(key, [])

    def set_tag(self, run_id, key, value, synchronous=None):
        if synchronous is True:
            self._tag(key, value)
        else:
            self.pending.append((key, value))

    def _tag(self, key, value):
        self.tags[key] = value
        self.tag_events.append((key, value))
        if key == "tracking_v1_status":
            self.status_events.append(value)

    def log_batch(self, run_id, metrics=(), tags=(), synchronous=None):
        assert synchronous is True
        self.sync_batches.append((len(metrics), len(tags)))
        for point in metrics:
            self.histories.setdefault(point.key, []).append(point)
        for tag in tags:
            self._tag(tag.key, tag.value)

    def log_artifact(self, run_id, path, artifact_path=None):
        key = f"{artifact_path}/{Path(path).name}"
        body = Path(path).read_bytes()
        assert key not in self.artifacts or self.artifacts[key] == body
        self.artifacts[key] = body

    def flush(self):
        pending, self.pending = self.pending, []
        for key, value in pending:
            self._tag(key, value)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_buffered_tags_cannot_revert_completion_after_flush(tmp_path, monkeypatch, seed):
    monkeypatch.setattr(metric_tracking, "_source_sha256", lambda: "evaluation-code")
    report = _report(tmp_path, seed)
    report_bytes = (tmp_path / "report.json").read_bytes()
    client = BufferedTagClient()
    first = metric_tracking.publish_tracking(client, "run", report, tmp_path, max_points_per_key=None)
    client.flush()  # The old initial async set_tag arrived here, after synchronous completion.
    assert client.tags["tracking_v1_status"] == "complete"
    assert client.status_events == ["publishing", "complete"]
    assert not client.pending
    observed = {(key, p.step, p.value) for key, rows in client.histories.items() for p in rows}
    assert observed == _expected_points(report)
    counts = {key: len(rows) for key, rows in client.histories.items()}
    artifacts = dict(client.artifacts)
    second = metric_tracking.publish_tracking(client, "run", report, tmp_path, max_points_per_key=None)
    client.flush()
    assert second == first
    assert client.tags["tracking_v1_status"] == "complete"
    assert client.status_events == ["publishing", "complete", "publishing", "complete"]
    assert {key: len(rows) for key, rows in client.histories.items()} == counts
    assert client.artifacts == artifacts
    payload = Path(first["sidecar_path"]).read_bytes()
    assert first["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert json.loads(payload)["tracking"] == report["tracking"]
    assert (tmp_path / "report.json").read_bytes() == report_bytes


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_sdk_global_async_logging_keeps_status_complete_after_buffer_flush(tmp_path, monkeypatch, seed):
    from mlflow import MlflowClient

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")
    monkeypatch.setenv("MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS", "5")
    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    experiment = client.create_experiment("publication-order")
    run_id = client.create_run(experiment).info.run_id
    output = tmp_path / "report"
    output.mkdir()
    report = _report(output, seed)
    real_batch, synchronous = client.log_batch, []

    def log_batch(*args, **kwargs):
        synchronous.append(kwargs.get("synchronous"))
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(client, "log_batch", log_batch)
    # Exercise a real buffered queue without changing any tracking status outside the publisher.
    operation = client.set_tag(run_id, "queued_seed", str(seed), synchronous=False)
    receipt = metric_tracking.publish_tracking(client, run_id, report, output, max_points_per_key=None)
    assert client.get_run(run_id).data.tags["tracking_v1_status"] == "complete"
    client._tracking_client.store.flush_async_logging()
    operation.wait()
    run = client.get_run(run_id)
    assert run.data.tags["queued_seed"] == str(seed)
    assert run.data.tags["tracking_v1_status"] == "complete"
    assert synchronous and all(synchronous)
    observed = {
        (key, point.step, point.value)
        for key in run.data.metrics
        for point in client.get_metric_history(run_id, key)
    }
    assert observed == _expected_points(report)
    sidecar = Path(receipt["sidecar_path"])
    assert json.loads(sidecar.read_bytes())["tracking"] == report["tracking"]
    assert run.data.tags["tracking_v1_payload_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_zia_artifact_status_is_acknowledged_synchronously_before_and_after_publication(
    tmp_path, monkeypatch, seed
):
    from continual_bench.evaluation import zia_publication

    client = BufferedTagClient()
    client.tags["zia_run_key"] = tmp_path.name
    report = {
        "artifact_mode": "metrics_only",
        "model_versions": [{"file": None}],
        "metrics": {"training_used_post_warmup": seed},
    }

    def predictions(*args, **kwargs):
        assert client.tags["zia_artifacts_status"] == "publishing"

    monkeypatch.setattr(zia_publication, "publish_predictions", predictions)
    monkeypatch.setattr(zia_publication, "tracking_report", lambda *args: {})
    monkeypatch.setattr(zia_publication, "write_zia_artifacts", lambda *args: None)
    monkeypatch.setattr(zia_publication, "publish_run_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr(zia_publication, "run_documents", lambda *args: (None, {}, {}))
    monkeypatch.setattr(zia_publication, "publish_tracking", lambda *args, **kwargs: {})
    receipt = zia_publication.publish_artifacts(client, tmp_path, "run", report, root=None)
    client.flush()
    assert client.tags["zia_artifacts_status"] == "complete"
    assert [event for event in client.tag_events if event[0] == "zia_artifacts_status"] == [
        ("zia_artifacts_status", "publishing"),
        ("zia_artifacts_status", "complete"),
    ]
    assert receipt["registered_model"] is None
    assert [(point.step, point.value) for point in client.histories["training_used"]] == [(0, seed)]


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_zia_receipt_follows_acknowledged_terminal_tag_and_run_status(tmp_path, monkeypatch, seed):
    from continual_bench.evaluation import zia_publication

    client = BufferedTagClient()
    terminated = []

    def run_metrics(*args, **kwargs):
        assert client.tags["benchmark_complete"] == "false"

    def terminate(run_id, status):
        assert run_id == "run" and status == "FINISHED"
        assert client.tags["benchmark_complete"] == "true"
        assert not (tmp_path / "receipt.json").exists()
        terminated.append(status)

    monkeypatch.setattr(zia_publication, "publish_run_metrics", run_metrics)
    monkeypatch.setattr(zia_publication, "_publish_artifacts", lambda *args: {"verified_seed": seed})
    client.set_terminated = terminate
    receipt = zia_publication.publish(client, tmp_path, "run", "experiment", {}, root=None)
    client.flush()
    assert terminated == ["FINISHED"] and receipt["complete"] is True
    assert client.tags["benchmark_complete"] == "true"
    assert json.loads((tmp_path / "receipt.json").read_text()) == receipt
    assert [event for event in client.tag_events if event[0] == "benchmark_complete"] == [
        ("benchmark_complete", "false"),
        ("benchmark_complete", "true"),
    ]
