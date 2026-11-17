"""Tests verifying forgetting and historical retention metrics are recorded across benchmark modes."""

import json

import pytest

from continual_bench.application.components import default_registry, make_source
from continual_bench.application.runner import evaluate, prepare
from continual_bench.config import Config, DatasetConfig
from continual_bench.evaluation.typed import run_typed
from continual_bench.infrastructure.benchmark_tracking import publish_local_benchmark


@pytest.mark.parametrize(
    "dataset,mode,policy",
    [
        ("sea", "incremental", {"name": "cara"}),
        ("hyperplane", "streaming", {"name": "drift"}),
        ("hyperplane", "incremental", {"name": "cara"}),
    ],
)
def test_binary_benchmarks_record_forgetting_and_retention_metrics(dataset, mode, policy, tmp_path):
    config = Config.model_validate(
        {
            "name": f"test-{dataset}-{mode}",
            "mode": mode,
            "dataset": {"name": dataset, "samples": 240, "label_delay": 2},
            "warmup": 40,
            "metric_window": 40,
            "batch_size": 40,
            "policy": policy,
        }
    )
    manifest = prepare(config, str(tmp_path))
    report = evaluate(config, manifest, str(tmp_path))

    tracking_metrics = report["tracking"]["metrics"]
    for key in (
        "cl_average_accuracy",
        "cl_average_incremental_accuracy",
        "cl_forgetting",
        "cl_backward_transfer",
        "retention_final_mean_accuracy",
        "retention_final_mean_drop",
    ):
        assert tracking_metrics[key] is not None, f"missing {key} for {dataset}/{mode}/{policy['name']}"

    retention = report["tracking_observations"]["retention"]
    for key in ("absolute_retention", "gem_bwt", "post_learning_fm", "matrix", "series"):
        assert retention[key] is not None, f"missing retention.{key} for {dataset}/{mode}/{policy['name']}"
    assert retention["availability"] == {}


@pytest.mark.parametrize("dataset", ["sea", "hyperplane"])
def test_typed_classification_records_historical_retention_and_forgetting(dataset, tmp_path):
    source = make_source(default_registry(), DatasetConfig(name=dataset, samples=240, label_delay=2))
    report = run_typed(
        source,
        tmp_path / f"typed-{dataset}",
        model={"name": "logistic"},
        policy="online",
        warmup=40,
        window=40,
    )
    retention = report["tracking_observations"]["retention"]
    for key in ("absolute_retention", "gem_bwt", "post_learning_fm", "matrix", "series"):
        assert retention[key] is not None, f"missing typed retention.{key} for {dataset}"
    assert retention["availability"] == {}


def test_local_benchmark_publication_logs_retention_and_forgetting_to_mlflow(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = MlflowClient(uri)

    config = Config.model_validate(
        {
            "name": "mlflow-retention-check",
            "mode": "streaming",
            "dataset": {"name": "hyperplane", "samples": 240, "label_delay": 2},
            "warmup": 40,
            "metric_window": 40,
            "policy": {"name": "drift"},
        }
    )
    manifest = prepare(config, str(tmp_path / "state"))
    report = evaluate(config, manifest, str(tmp_path / "state"))
    directory = tmp_path / "state" / config.identity()
    (directory / "report.json").write_text(json.dumps(report))

    receipt = publish_local_benchmark(directory, report, experiment_name="local-retention", client=client)
    run_metrics = client.get_run(receipt["mlflow_run_id"]).data.metrics
    assert "observations_v1/retention/absolute_retention" in run_metrics
    assert "observations_v1/retention/gem_bwt" in run_metrics
    assert "observations_v1/retention/post_learning_fm" in run_metrics
    assert "tracking_v1/cl_forgetting" in run_metrics
    assert "tracking_v1/summary/cl_forgetting" in run_metrics
    assert "tracking_v1/summary/retention_final_mean_accuracy" in run_metrics

    dashboard_path = directory / "dashboard.html"
    assert dashboard_path.is_file()
    html = dashboard_path.read_text(encoding="utf-8")
    assert 'id="timeline-slider"' in html
    assert 'id="smooth-strategy"' in html
    assert 'id="mode-global"' in html
    assert 'id="mode-live"' in html
    assert 'id="benchmark-data"' in html
    assert "cl_forgetting" in html
    assert "absolute_retention" in html
    artifact_paths = {item.path for item in client.list_artifacts(receipt["mlflow_run_id"])}
    assert "dashboard.html" in artifact_paths


def test_probes_are_independent_of_training_samples_and_unavailable_without_independent_source(tmp_path):
    from continual_bench.core.contracts import TaskSample
    from continual_bench.datasets.synthetic import MovingSyntheticSource

    for name in ("hyperplane", "rbf_drift"):
        src = MovingSyntheticSource(name, 30, 42, 0, 0)
        train_x = [e.x for e in src.events()]
        probes = src.episode_probes([10, 20, 30], 8)
        probe_x = [x for ep in probes.values() for x, _ in ep]
        assert probe_x
        assert not any(px in train_x for px in probe_x)

    class _TrainOnlySource:
        def samples_iter(self):
            for i in range(12):
                yield TaskSample(f"s{i}", i, i, i, {"a": float(i)}, "0" if i % 2 == 0 else "1", split="train")

        def manifest(self):
            return {"name": "train-only", "task": "multiclass", "labels": ["0", "1"]}

    report = run_typed(
        _TrainOnlySource(),
        tmp_path / "train-only",
        model={"name": "logistic"},
        policy="online",
        warmup=2,
        window=4,
    )
    assert report["tracking_observations"]["retention"]["absolute_retention"] is None


def test_typed_records_retention_even_when_training_budget_blocks_updates(tmp_path):
    from continual_bench.datasets.synthetic import SyntheticSource

    src = SyntheticSource("sea", 12, 42, 0, 0)
    report = run_typed(
        src,
        tmp_path / "budget-zero",
        model={"name": "logistic"},
        policy="online",
        warmup=2,
        window=4,
        training_budget=0,
    )
    retention = report["tracking_observations"]["retention"]
    assert len(retention["matrix"]) == 3
    assert retention["absolute_retention"] is not None
    assert retention["gem_bwt"] is not None
    assert retention["post_learning_fm"] is not None
