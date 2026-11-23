"""Metrics-only execution keeps measurements and lineage without persisted model state."""

import hashlib
import json
import time

import numpy as np
import pytest

from continual_bench.application.runner import run
from continual_bench.config import Config
from continual_bench.evaluation.retraining_observations import candidate_report
from continual_bench.evaluation.system_metrics import SystemMetricsSampler
from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files
from continual_bench.infrastructure.storage import LocalCheckpoints
from continual_bench.infrastructure.versioning import ModelVersions, load_version
from continual_bench.models.linear import RiverModel


def no_checkpoint(*args, **kwargs):
    pytest.fail("metrics-only execution called a training checkpoint operation")


def test_artifact_mode_preserves_default_identity_and_separates_retention():
    default = Config()
    assert "artifact_mode" not in default.model_dump(mode="json")
    assert Config(artifact_mode="full").identity() == default.identity()
    metrics = Config(artifact_mode="metrics_only")
    assert metrics.identity() != default.identity()
    assert metrics.model_dump(mode="json")["artifact_mode"] == "metrics_only"


def test_version_metadata_preserves_digests_and_sizes_without_model_files(tmp_path):
    model = RiverModel()
    model.learn({"x": 1.0}, True)
    versions = ModelVersions(tmp_path, "d" * 64, "c" * 64, artifact_mode="metrics_only")
    versions.record(model, {"seq_range": [0, 1], "action": "initial_fit"})
    rejected = versions.record_candidate(model, 4)
    metadata = versions.finish(model, {"x": 2.0})
    assert metadata["count"] == 1 and metadata["storage_bytes"] == 0
    assert metadata["serialized_state_bytes"] > 0
    assert metadata["retained"] is False and metadata["journal_sha256"] is None
    assert rejected["file"] is None and rejected["bytes"] > 0
    assert rejected["retained"] is False and rejected["storage_bytes"] == 0
    index = (tmp_path / "model-versions.json").read_bytes()
    assert metadata["index_sha256"] == hashlib.sha256(index).hexdigest()
    assert metadata["state_sha256"] == json.loads(index)["versions"][0]["state_sha256"]
    assert {p.name for p in tmp_path.iterdir()} == {"model-versions.json", "model-metadata.json"}
    verify_metrics_only_files(tmp_path)


def test_metrics_only_samples_state_without_skipping_update_lineage(tmp_path, monkeypatch):
    import pickle

    model = RiverModel()
    versions = ModelVersions(tmp_path, "d" * 64, "c" * 64, artifact_mode="metrics_only", state_sample_every=4)
    serialize = pickle.dumps
    calls = []

    def measured(value, **kwargs):
        if value is model:
            calls.append(1)
        return serialize(value, **kwargs)

    monkeypatch.setattr("continual_bench.infrastructure.versioning.pickle.dumps", measured)
    for index in range(11):
        model.learn({"x": float(index)}, bool(index % 2))
        versions.record(
            model, {"seq_range": [index, index + 1], "action": "initial_fit" if index == 0 else "learn_one"}
        )
    assert len(calls) == 3  # Initial model, fourth update, eighth update.
    identities = [entry["version_id"] for entry in versions.entries]
    assert len(identities) == 11 and len(set(identities)) == 11
    assert versions.entries[1]["size"] is None and versions.entries[1]["state_sha256"] is None
    assert versions.entries[1]["state_measurement"] == "not_measured_between_state_sampling_cuts"
    metadata = versions.finish(model, {"x": 1.0})
    assert [entry["version_id"] for entry in versions.entries] == identities
    assert metadata["state_measurement"]["every_updates"] == 4
    assert [point["ordinal"] for point in metadata["state_measurements"]] == [0, 4, 8, 10]
    assert metadata["state_measurements"][-1]["sha256"] == metadata["state_sha256"]
    with pytest.raises(ValueError, match="weights were not retained"):
        load_version(tmp_path, identities[-1])
    verify_metrics_only_files(tmp_path)


@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_metrics_only_binary_keeps_quality_and_never_loads_or_saves_checkpoint(tmp_path, monkeypatch, mode):
    base = {
        "name": "metrics-only-test",
        "mode": mode,
        "dataset": {"samples": 120, "label_delay": 3},
        "warmup": 20,
        "metric_window": 10,
        "batch_size": 8,
        "policy": {"name": "none"} if mode == "offline" else {"name": "online", "action": "learn_one"},
    }
    expected = run(Config.model_validate(base), tmp_path / "full")
    config = Config.model_validate({**base, "artifact_mode": "metrics_only"})
    monkeypatch.setattr(LocalCheckpoints, "load", no_checkpoint)
    monkeypatch.setattr(LocalCheckpoints, "save", no_checkpoint)
    actual = run(config, tmp_path / "metrics")
    assert expected["metrics"] == actual["metrics"]
    assert expected.get("prediction_digest") == actual.get("prediction_digest")
    assert actual["model_versions"]["storage_bytes"] == 0
    assert actual["system_metrics"]["max_retained_samples"] is None
    assert actual["tracking"]["series"]
    directory = tmp_path / "metrics" / config.identity()
    verify_metrics_only_files(directory)
    assert run(config, tmp_path / "metrics") == actual


def test_metrics_only_refuses_external_training_checkpoint_before_read(tmp_path):
    class Checkpoint:
        load = no_checkpoint

    with pytest.raises(ValueError, match="cannot restore or save"):
        run(Config(artifact_mode="metrics_only"), tmp_path, checkpoint=Checkpoint())


def test_complete_system_sampling_keeps_every_point_past_default_bound(tmp_path, monkeypatch):
    sampler = SystemMetricsSampler(tmp_path, max_samples=None)
    sampler._started = time.monotonic()
    monkeypatch.setattr(sampler, "_measure", lambda cpu: {"metric": 1.0})
    for _ in range(3601):
        sampler.sample()
    report = sampler.report()
    assert len(report["samples"]) == 3601
    assert report["interval_seconds"] == sampler.interval
    assert report["max_retained_samples"] is None


def test_candidate_tracking_retains_metrics_without_write_state(tmp_path, monkeypatch):
    class Learner:
        observation_model = "fixture"
        observation_threshold = 0.5
        observation_step_kind = "fixture_fit"

        def observation_predict(self, state, x, positive):
            return x[:, 0].astype(int), x[:, 0]

        def observation_checkpoint(self, state):
            return bytes([state]), "pkl"

        def observation_durations(self, index, elapsed):
            return [elapsed]

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "stub")
    monkeypatch.setenv("BENCH_STATE", str(tmp_path))
    monkeypatch.setattr("continual_bench.evaluation.retraining_observations.write_state", no_checkpoint)
    monkeypatch.setattr(
        "continual_bench.evaluation.training_evidence.write_training_artifacts", lambda directory, report: {}
    )
    monkeypatch.setattr(
        "continual_bench.infrastructure.benchmark_tracking.publish_local_benchmark",
        lambda directory, report, **kwargs: {"complete": True},
    )
    x = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    report = candidate_report(
        "fixture",
        1,
        [(x, y, x, y)] * 2,
        [1, 2],
        Learner(),
        [0.1, 0.2],
        protocol="fixture",
        initial=0,
        artifact_mode="metrics_only",
    )
    assert report["contract"]["artifact_mode"] == "metrics_only"
    assert len(report["training_history"]) == 2
    assert all(not (fit["checkpoint"] or {}).get("file") for fit in report["training_history"])
    assert report["tracking_observations"]["windows"]
    verify_metrics_only_files(tmp_path)


def test_metrics_only_publication_rejects_accidental_model_state(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"must not publish")
    with pytest.raises(ValueError, match="contains model or checkpoint"):
        verify_metrics_only_files(tmp_path)


def test_metrics_only_mlflow_keeps_complete_quality_and_system_series(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from continual_bench.evaluation.tracking_observations import observations
    from continual_bench.infrastructure.benchmark_tracking import publish_local_benchmark
    from continual_bench.infrastructure.observation_tracking import publish_sdk_observations

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr("continual_bench.infrastructure.run_artifacts.log_detailed_figures", no_checkpoint)
    client = MlflowClient(uri)
    directory = tmp_path / "evidence"
    directory.mkdir()
    report = {
        "complete": True,
        "run_key": "complete-metric-series",
        "artifact_mode": "metrics_only",
        "dataset": "fixture",
        "config": {
            "model": {"name": "hoeffding_tree"},
            "policy": {"name": "none"},
            "detector": {"name": "none"},
            "mode": "incremental",
        },
        "contract": {"seed": 7, "protocol": "fixture-causal"},
        "windows": [{"end_seq": i, "accuracy": i / 600} for i in range(600)],
        "system_metrics": {
            "samples": [{"timestamp_ms": 1000 + i, "values": {"cpu": float(i)}} for i in range(600)]
        },
    }
    report["tracking_observations"] = observations(
        "binary",
        {},
        windows=[{"step": i, "scope": "fixture", "quality": {"accuracy": i / 600}} for i in range(600)],
    )
    plot_root = directory / "observation-fixture"
    plot_root.mkdir()
    (plot_root / "plots-manifest.json").write_text("{}")
    monkeypatch.setattr(
        "continual_bench.infrastructure.observation_tracking.write_observation_evidence",
        lambda *args: plot_root,
    )

    def complete_observations(*args, **kwargs):
        assert kwargs["max_points_per_key"] is None
        return publish_sdk_observations(*args, **kwargs)

    monkeypatch.setattr(
        "continual_bench.infrastructure.run_artifacts.publish_sdk_observations", complete_observations
    )
    (directory / "report.json").write_text(json.dumps(report))
    receipt = publish_local_benchmark(directory, report, experiment_name="metrics-only-series", client=client)
    run_id = receipt["mlflow_run_id"]
    assert len(client.get_metric_history(run_id, "prequential/windows/accuracy")) == 600
    assert len(client.get_metric_history(run_id, "system/cpu")) == 600
    assert len(client.get_metric_history(run_id, "observations_v1/windows/fixture/accuracy")) == 600
    assert client.get_run(run_id).data.tags["cb-artifact-mode"] == "metrics_only"
    tags = client.get_run(run_id).data.tags
    assert tags["cb-model"] == "hoeffding_tree" and tags["cb-policy"] == "none"
    assert tags["cb-mode"] == "incremental" and tags["cb-detector"] == "none"
    assert tags["cb-run-role"] == "matrix_case"
    assert "hoeffding_tree-none" in tags["mlflow.runName"]
    again = publish_local_benchmark(directory, report, experiment_name="metrics-only-series", client=client)
    assert again["mlflow_run_id"] == run_id
    assert len(client.get_metric_history(run_id, "system/cpu")) == 600


def test_metrics_only_defers_static_rendering_preserving_json_html_and_cache_mode(tmp_path, monkeypatch):
    import re

    from continual_bench.evaluation.tracking_observations import batch_quality, observations
    from continual_bench.infrastructure.observation_tracking import write_observation_evidence

    quality = batch_quality([0, 1], [0, 1], classes=[0, 1], positive_class=1, scores=[0.1, 0.9])
    report = {
        "training_history": [],
        "tracking_observations": observations(
            "binary", {"prequential": quality},
            windows=[{"step": 2, "scope": "prequential", "quality": quality}],
        ),
    }
    availability_calls = []

    def unavailable_matplotlib():
        availability_calls.append(1)
        return False

    monkeypatch.setattr(
        "continual_bench.infrastructure.training_figures.matplotlib_available", unavailable_matplotlib
    )
    monkeypatch.setattr("continual_bench.infrastructure.training_figures._pyplot", no_checkpoint)
    root = write_observation_evidence(tmp_path, report)
    assert availability_calls == [1]  # The full default retains the original rendering path.
    full_manifest = json.loads((root / "plots-manifest.json").read_text())
    assert full_manifest["static_rendering"]["enabled"] is True
    original = (root / "observations.json").read_bytes()
    html = (root / "retraining-explorer.html").read_text()
    original_data = json.loads(re.search(r'id="page-data">(.*?)</script>', html, re.S).group(1))
    availability_calls.clear()
    report["artifact_mode"] = "metrics_only"
    # A prior cached static artifact must not survive the render-mode change.
    stale = root / "overview" / "overview.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"obsolete-render")
    write_observation_evidence(tmp_path, report)
    manifest = json.loads((root / "plots-manifest.json").read_text())
    assert availability_calls == []
    assert manifest["source_checksums"]["render_mode"] == "deferred_static"
    assert manifest["static_rendering"] == {"enabled": False, "reason": "rendering_deferred"}
    assert not list(root.rglob("*.png"))
    assert any(entry["availability"]["reason"] == "rendering_deferred" for entry in manifest["figures"])
    assert all(
        entry["availability"]["reason"] != "Matplotlib indisponível"
        for entry in manifest["figures"]
    )
    assert (root / "observations.json").read_bytes() == original
    current_html = (root / "retraining-explorer.html").read_text()
    current_data = json.loads(re.search(r'id="page-data">(.*?)</script>', current_html, re.S).group(1))
    assert current_data == original_data
    # Reusing the same mode signature should not even enter the renderer again.
    monkeypatch.setattr(
        "continual_bench.infrastructure.training_figures.write_observation_figures", no_checkpoint
    )
    write_observation_evidence(tmp_path, report)
