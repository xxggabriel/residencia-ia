import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from continual_bench.adapters import metric_tracking, services


@dataclass
class Metric:
    key: str
    value: float
    timestamp: int
    step: int


@dataclass
class NamedValue:
    key: str
    value: str


class Client:
    def __init__(self):
        self.histories = {}
        self.artifacts = {}
        self.tags = {}
        self.params = {}
        self.operations = []
        self.fail_after = None
        self.run = SimpleNamespace(info=SimpleNamespace(run_id="existing-run"),
                                   inputs=SimpleNamespace(dataset_inputs=["already-present"]))

    def get_metric_history(self, run_id, key):
        assert run_id == "existing-run"
        return self.histories.get(key, [])

    def log_batch(self, run_id, metrics=(), params=(), tags=(), synchronous=None):
        assert run_id == "existing-run"
        assert synchronous is True
        self.operations.append("batch")
        for param in params:
            self.params[param.key] = param.value
        for metric in metrics:
            self.histories.setdefault(metric.key, []).append(metric)
            if self.fail_after is not None:
                self.fail_after -= 1
                if self.fail_after == 0:
                    self.fail_after = None
                    raise RuntimeError("partial metric publication")
        self.tags.update({tag.key: tag.value for tag in tags})

    def log_artifact(self, run_id, path, artifact_path=None):
        assert run_id == "existing-run"
        self.operations.append("artifact")
        key = f"{artifact_path}/{Path(path).name}" if artifact_path else Path(path).name
        content = Path(path).read_bytes()
        assert key not in self.artifacts or self.artifacts[key] == content
        self.artifacts[key] = content

    def set_tag(self, run_id, key, value):
        assert run_id == "existing-run"
        self.operations.append((key, value))
        self.tags[key] = value

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="experiment")

    def search_runs(self, experiment_ids, query):
        return [self.run]

    def set_terminated(self, run_id, status):
        assert run_id == "existing-run"
        self.operations.append(status)


@pytest.fixture(autouse=True)
def fake_mlflow(monkeypatch):
    # No MLflow, pandas, botocore or running services are needed by these tests.
    entities = ModuleType("mlflow.entities")
    entities.Metric = Metric
    entities.Param = entities.RunTag = NamedValue
    entities.Dataset = entities.DatasetInput = entities.InputTag = SimpleNamespace
    module = ModuleType("mlflow")
    module.set_tracking_uri = lambda uri: None
    tracking = ModuleType("mlflow.tracking")
    tracking.MlflowClient = lambda: None
    for name, fake in [("mlflow", module), ("mlflow.entities", entities), ("mlflow.tracking", tracking)]:
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fixture.invalid")


def report_fixture(tmp_path):
    report = {
        "environment": {"code_sha256": "original-code"},
        "dataset": {"sha256": "dataset-version", "source": "fixture", "version": "1", "license": "CC0"},
        "tracking": {
            "version": metric_tracking.VERSION,
            "metrics": {"accuracy": 0.75, "roc_auc": None},
            "series": {"windows/accuracy": [{"step": 5, "value": 0.5}, {"step": 10, "value": 1}],
                       "windows/roc_auc": [{"step": 5, "value": None}]},
            "availability": {"roc_auc": "single_class", "windows/roc_auc": "single_class"},
            "settings": {"calibration_bins": 10, "threshold": 0.5},
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2))
    return report


def test_tracking_artifacts_lineage_nulls_and_idempotence(tmp_path, monkeypatch):
    report = report_fixture(tmp_path)
    original = (tmp_path / "report.json").read_bytes()
    (tmp_path / "predictions.jsonl").write_text('{"y":true,"p":0.75}\n')
    (tmp_path / "metric-definitions.json").write_text('{"definitions":{}}')
    (tmp_path / "continual-probes.json").write_text('{"references":[]}')
    report["tracking"]["definitions_sha256"] = "definitions-version"
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "reference.json").write_text('{"score":0.2}')
    stored = {}

    def immutable(objects, key, path):
        blob = Path(path).read_bytes()
        assert key not in stored or stored[key] == blob
        stored[key] = blob
        return "s3://fixture/" + key

    monkeypatch.setattr(metric_tracking, "put_immutable", immutable)
    monkeypatch.setattr(metric_tracking, "_source_sha256", lambda: "evaluation-code")
    client = Client()
    first = metric_tracking.publish_tracking(client, "existing-run", report, tmp_path, objects=object())
    history = copy.deepcopy(client.histories)
    second = metric_tracking.publish_tracking(client, "existing-run", report, tmp_path, objects=object())
    assert second == first
    assert client.histories == history
    assert len(client.artifacts) == len(stored) == 5
    assert len(history) == 2
    assert "tracking_v1/summary/roc_auc" not in history
    assert "tracking_v1/windows/roc_auc" not in history
    assert len(history["tracking_v1/windows/accuracy"]) == 2
    envelope = json.loads(Path(first["sidecar_path"]).read_bytes())
    assert envelope["training_code_sha256"] == "original-code"
    assert envelope["evaluation_code_sha256"] == "evaluation-code"
    assert envelope["dataset_sha256"] == "dataset-version"
    assert envelope["report_sha256"] == hashlib.sha256(original).hexdigest()
    assert envelope["tracking"] == report["tracking"]
    assert envelope["tracking"]["metrics"]["roc_auc"] is None
    assert envelope["derived"] is False
    settings = json.dumps(report["tracking"]["settings"], sort_keys=True, separators=(",", ":")).encode()
    assert envelope["settings_sha256"] == hashlib.sha256(settings).hexdigest()
    payload = Path(first["sidecar_path"]).read_bytes()
    assert first["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    for item in envelope["files"]:
        assert item["sha256"] == hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest()
    assert (tmp_path / "report.json").read_bytes() == original
    assert client.tags["tracking_v1_settings_sha256"] == envelope["settings_sha256"]
    assert client.tags["tracking_v1_definitions_sha256"] == "definitions-version"
    assert client.tags["tracking_v1_origin"] == "native"
    assert client.tags["tracking_v1_unavailable_count"] == "1"
    assert client.operations[-1] == ("tracking_v1_status", "complete")


def test_partial_metric_write_resumes_without_duplicates_or_early_complete(tmp_path):
    report = report_fixture(tmp_path)
    client = Client()
    client.fail_after = 1
    with pytest.raises(RuntimeError, match="partial"):
        metric_tracking.publish_tracking(client, "existing-run", report, tmp_path, derived=True)
    assert client.tags["tracking_v1_status"] == "publishing"
    assert "tracking_v1_payload_sha256" not in client.tags
    assert sum(map(len, client.histories.values())) == 1
    result = metric_tracking.publish_tracking(client, "existing-run", report, tmp_path, derived=True)
    assert sum(map(len, client.histories.values())) == 3
    assert json.loads(Path(result["sidecar_path"]).read_text())["derived"] is True
    assert client.tags["tracking_v1_origin"] == "derived"
    assert client.tags["tracking_v1_status"] == "complete"


def test_artifact_failure_never_marks_complete_and_retry_recovers(tmp_path, monkeypatch):
    report = report_fixture(tmp_path)
    client = Client()
    real_log = client.log_artifact

    def fail(*args, **kwargs):
        raise RuntimeError("artifact publication failed")

    monkeypatch.setattr(client, "log_artifact", fail)
    with pytest.raises(RuntimeError, match="artifact publication"):
        metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert client.tags["tracking_v1_status"] == "publishing"
    assert not client.histories
    monkeypatch.setattr(client, "log_artifact", real_log)
    metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert client.tags["tracking_v1_status"] == "complete"


def test_report_hash_fallback_excludes_new_tracking_to_avoid_circular_lineage(tmp_path):
    report = report_fixture(tmp_path)
    (tmp_path / "report.json").unlink()
    first = metric_tracking.publish_tracking(Client(), "existing-run", report, tmp_path, derived=True)
    original_hash = json.loads(Path(first["sidecar_path"]).read_text())["report_sha256"]
    report["tracking"]["settings"]["calibration_bins"] = 20
    second = metric_tracking.publish_tracking(Client(), "existing-run", report, tmp_path, derived=True)
    changed = json.loads(Path(second["sidecar_path"]).read_text())
    assert changed["report_sha256"] == original_hash
    assert first["payload_sha256"] != second["payload_sha256"]
    assert not (tmp_path / "report.json").exists()


def test_conflicting_remote_history_rejected_before_side_effects(tmp_path):
    report = report_fixture(tmp_path)
    client = Client()
    key = "tracking_v1/windows/accuracy"
    client.histories[key] = [Metric(key, 0.5, 1234, 5), Metric(key, 0.7, 9999, 5)]
    with pytest.raises(ValueError, match="remote metric conflicts"):
        metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert client.operations == []
    assert not (tmp_path / "tracking").exists()


def test_unavailable_value_conflicts_with_already_published_finite_value(tmp_path):
    report = report_fixture(tmp_path)
    client = Client()
    key = "tracking_v1/summary/roc_auc"
    client.histories[key] = [Metric(key, 0.9, 0, 0)]
    with pytest.raises(ValueError, match="remote metric conflicts"):
        metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert not client.operations


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "0.5", True])
def test_nonfinite_or_nonnumeric_payload_rejected(tmp_path, value):
    report = report_fixture(tmp_path)
    report["tracking"]["metrics"]["accuracy"] = value
    client = Client()
    with pytest.raises(ValueError):
        metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert not client.operations


def test_conflicting_payload_series_rejected(tmp_path):
    report = report_fixture(tmp_path)
    report["tracking"]["series"]["windows/accuracy"].append({"step": 5, "value": 0.2})
    with pytest.raises(ValueError, match="conflicting tracking"):
        metric_tracking.publish_tracking(Client(), "existing-run", report, tmp_path)


def test_no_tracking_is_a_noop_without_files_or_dependencies(tmp_path):
    client = Client()
    assert metric_tracking.publish_tracking(client, "existing-run", {}, tmp_path) is None
    assert not client.operations
    assert not list(tmp_path.iterdir())


def test_all_null_metrics_are_preserved_without_fake_logs(tmp_path):
    report = report_fixture(tmp_path)
    report["tracking"]["metrics"] = {"roc_auc": None}
    report["tracking"]["series"] = {"roc_auc": [{"step": 3, "value": None}]}
    client = Client()
    result = metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert not client.histories
    assert result
    assert client.tags["tracking_v1_status"] == "complete"


def test_native_publication_all_window_keys_none_and_retry(tmp_path, monkeypatch):
    report = report_fixture(tmp_path)
    report.update({"protocol": "fixture", "metrics": {"n": 2, "accuracy": 0.5, "roc_auc": None},
                   "windows": [{"end_seq": 5, "n": 2, "accuracy": 0.5, "roc_auc": None, "errors": 1}],
                   "updates": [{"train_samples": 3}], "alarms": []})
    config = {"dataset": {"name": "sea", "seed": 42, "samples": 4},
              "model": {"name": "logistic", "params": {}}, "mode": "incremental", "name": "fixture",
              "protocol_version": "1", "policy": {"name": "online", "every": 1, "cooldown": 0},
              "detector": {"name": "none"}}
    catalog = {"dataset_version": "dataset-version", "model_version_id": "version",
               "model_state_sha256": "model-sha", "model_versions_count": 1}
    client = Client()
    monkeypatch.setattr(sys.modules["mlflow.tracking"], "MlflowClient", lambda: client)
    monkeypatch.setattr(services, "register_final_model", lambda *args: {
        "registered_model": "fixture-model", "registered_model_version": "1"})
    path = str(tmp_path / "report.json")
    first = services.MLflowTracker().publish("key", config, report, [path], catalog)
    history = copy.deepcopy(client.histories)
    second = services.MLflowTracker().publish("key", config, report, [path], catalog)
    assert first == second
    assert client.histories == history
    assert set(client.histories) >= {"accuracy", "n", "errors", "summary_n", "summary_accuracy",
                                     "training_used", "tracking_v1/summary/accuracy"}
    assert "roc_auc" not in client.histories
    assert "summary_roc_auc" not in client.histories
    assert "summary_roc_auc" not in client.params
    assert client.operations[-2:] == [("tracking_v1_status", "complete"), "FINISHED"]
    client.histories["errors"].append(Metric("errors", 9, 25, 5))
    with pytest.raises(ValueError, match="remote metric conflicts"):
        services.MLflowTracker().publish("key", config, report, [path], catalog)
    report.pop("tracking")
    legacy_client = Client()
    monkeypatch.setattr(sys.modules["mlflow.tracking"], "MlflowClient", lambda: legacy_client)
    legacy = services.MLflowTracker().publish("key", config, report, [path], catalog)
    assert "tracking" not in legacy
    assert "tracking_v1_status" not in legacy_client.tags


def test_known_keys_skip_absent_histories_and_present_keys_are_checked_once(tmp_path):
    report = report_fixture(tmp_path)
    client = Client()
    key = "tracking_v1/summary/accuracy"
    client.histories[key] = [Metric(key, 0.75, 9999, 0)]
    calls = []

    def get_run(run_id):
        assert run_id == "existing-run"
        calls.append("run")
        return SimpleNamespace(data=SimpleNamespace(metrics={k: rows[-1].value
                                                            for k, rows in client.histories.items()}))

    def get_history(run_id, name):
        calls.append(name)
        return client.histories.get(name, [])

    client.get_run = get_run
    client.get_metric_history = get_history
    metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert calls == ["run", key]
    assert sum(map(len, client.histories.values())) == 3
    calls.clear()
    metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert calls[0] == "run"
    assert sorted(calls[1:]) == sorted(client.histories)
    assert len(calls) == 3


def test_known_keys_still_reject_unavailable_value_with_finite_history(tmp_path):
    report = report_fixture(tmp_path)
    client = Client()
    key = "tracking_v1/summary/roc_auc"
    client.histories[key] = [Metric(key, 0.9, 0, 0)]
    client.get_run = lambda _: SimpleNamespace(data=SimpleNamespace(metrics={key: 0.9}))
    with pytest.raises(ValueError, match="remote metric conflicts"):
        metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    assert not client.operations


def test_explicit_known_keys_avoids_get_run_and_absent_history():
    client = Client()

    def unexpected(*args):
        raise AssertionError("unnecessary remote GET")

    client.get_run = client.get_metric_history = unexpected
    points = [("new-key", 0, 0.3), ("unavailable-key", 0, None)]
    assert metric_tracking.missing_metric_points(client, "existing-run", points, known_keys=set()) == [
        ("new-key", 0, 0.3)]


def test_execution_first_success_cache_isolated_and_native_republication_safe(tmp_path):
    report = report_fixture(tmp_path)
    report["tracking"]["metrics"]["pipeline_elapsed_seconds"] = None
    report["tracking"]["availability"]["pipeline_elapsed_seconds"] = {"status": "unavailable"}
    original_report = copy.deepcopy(report)
    original_bytes = (tmp_path / "report.json").read_bytes()
    (tmp_path / "receipt.json").write_text('{"mlflow_run_id":"existing-run"}')
    (tmp_path / "checkpoint.bin").write_bytes(b"original-checkpoint")
    client = Client()
    metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    first = metric_tracking.publish_execution(client, "existing-run", report, tmp_path, 12.5)
    history = copy.deepcopy(client.histories)
    metric_tracking.publish_tracking(client, "existing-run", report, tmp_path)
    second = metric_tracking.publish_execution(client, "existing-run", report, tmp_path, 0.25)
    assert first == second
    assert client.histories == history
    key = "tracking_v1/summary/pipeline_elapsed_seconds"
    assert client.histories[key] == [Metric(key, 12.5, 0, 0)]
    assert report == original_report
    assert (tmp_path / "report.json").read_bytes() == original_bytes
    assert (tmp_path / "receipt.json").read_text() == '{"mlflow_run_id":"existing-run"}'
    assert (tmp_path / "checkpoint.bin").read_bytes() == b"original-checkpoint"
    cache = json.loads((tmp_path / "tracking/execution-v1.json").read_text())
    assert cache["elapsed_seconds"] == 12.5
    assert cache["source_report_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    payload = json.loads(Path(first["sidecar_path"]).read_text())
    assert payload["tracking"]["execution_scope"] == metric_tracking.EXECUTION_SCOPE
    assert payload["tracking"]["availability"]["pipeline_elapsed_seconds"] == {
        "status": "available", "reason": None, "scope": metric_tracking.EXECUTION_SCOPE}
    assert any(item["path"] == "tracking/execution-v1.json" for item in payload["files"])
    assert client.tags["tracking_v1_origin"] == "native"
    assert client.tags["tracking_v1_unavailable_count"] == "1"


def test_execution_cache_preserves_first_elapsed_after_publication_failure(tmp_path, monkeypatch):
    report = report_fixture(tmp_path)
    client = Client()
    original_log = client.log_artifact

    def fail(*args, **kwargs):
        raise RuntimeError("artifact unavailable")

    monkeypatch.setattr(client, "log_artifact", fail)
    with pytest.raises(RuntimeError, match="artifact unavailable"):
        metric_tracking.publish_execution(client, "existing-run", report, tmp_path, 30)
    monkeypatch.setattr(client, "log_artifact", original_log)
    metric_tracking.publish_execution(client, "existing-run", report, tmp_path, 1)
    assert client.histories["tracking_v1/summary/pipeline_elapsed_seconds"][0].value == 30


@pytest.mark.parametrize("elapsed", [math.nan, math.inf, -math.inf, -0.1, True, "1"])
def test_execution_invalid_elapsed_never_creates_cache(tmp_path, elapsed):
    report = report_fixture(tmp_path)
    client = Client()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        metric_tracking.publish_execution(client, "existing-run", report, tmp_path, elapsed)
    assert not client.operations
    assert not (tmp_path / "tracking").exists()


def test_execution_without_tracking_is_legacy_noop(tmp_path):
    client = Client()
    assert metric_tracking.publish_execution(client, "existing-run", {}, tmp_path, 2) is None
    assert not client.operations
    assert not list(tmp_path.iterdir())


def test_execution_cache_rejects_changed_original_report(tmp_path):
    report = report_fixture(tmp_path)
    client = Client()
    metric_tracking.publish_execution(client, "existing-run", report, tmp_path, 2)
    (tmp_path / "report.json").write_bytes(b"changed-original")
    with pytest.raises(ValueError, match="provenance"):
        metric_tracking.publish_execution(client, "existing-run", report, tmp_path, 1)
