"""System metrics, segmented plots and plot data for every evaluator."""

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from continual_bench.application.components import default_registry, make_source
from continual_bench.application.runner import run
from continual_bench.config import DEFAULT_TRACKING, Config, DatasetConfig, EpochConfig
from continual_bench.evaluation import system_metrics
from continual_bench.evaluation.artifacts import svg
from continual_bench.evaluation.epochs import run_epochs
from continual_bench.evaluation.system_metrics import SystemMetricsSampler
from continual_bench.evaluation.typed import run_typed
from continual_bench.infrastructure.run_artifacts import publish_run_artifacts, system_metric_points


def small(**changes):
    return Config.model_validate(
        {
            "name": "artifacts",
            "dataset": {"name": "sea", "samples": 300, "label_delay": 2},
            "warmup": 30,
            "metric_window": 30,
            "replay_window": 60,
            "detector": {"name": "adwin"},
            "policy": {"name": "drift", "cooldown": 10, "min_samples": 30, "min_window": 15},
            **changes,
        }
    )


def assert_artifacts(directory, groups):
    root = directory / "artifacts"
    index = json.loads((root / "data/index.json").read_text())
    assert set(groups) <= set(index["groups"])
    for figure in index["figures"]:
        ET.fromstring((root / figure["plot"]).read_text())  # Every plot is well-formed SVG.
        data = json.loads((root / figure["data"]).read_text())
        assert data["title"] == figure["title"] and data["group"] == figure["group"]
        assert figure["plot"].startswith(f"plots/{figure['group']}/")
    for path in index["data"]:
        json.loads((root / path).read_text())
    return index


def test_svg_charts_are_well_formed_and_handle_gaps_and_empty_data():
    line = svg.line_chart(
        "t",
        [{"name": "a", "points": [[0, 0.1], [1, None], [2, 0.5], [3, 0.4]]}],
        markers=[{"x": 1, "kind": "alarm", "label": "x"}],
    )
    assert line.count("<polyline") == 1 and line.count("<circle") == 1 and "stroke-dasharray" in line
    for document in (
        line,
        svg.bar_chart("b", ["x", "y"], [1.5, None]),
        svg.heatmap("h", ["r"], ["c1", "c2"], [[0.2, None]]),
        svg.line_chart("e", []),
    ):
        ET.fromstring(document)
    assert svg.nice_ticks(0, 1) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_sampler_records_mlflow_named_series_at_start_interval_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(system_metrics, "_current_cgroup", lambda: None)
    with SystemMetricsSampler(tmp_path, interval=0.05) as sampler:
        sum(i * i for i in range(200_000))
    report = sampler.report()
    assert len(report["samples"]) >= 2 and report["version"] == "system-metrics-v1"
    assert "process_memory_rss_megabytes" in report["metrics"] or report["collector"] == "procfs_rss_only"
    if report["collector"] == "psutil":
        assert {
            "cpu_utilization_percentage",
            "system_memory_usage_megabytes",
            "disk_usage_percentage",
            "network_receive_megabytes",
        } <= set(report["metrics"])
    points, stamps = system_metric_points(report)
    assert points and all(key.startswith("system/") for key, _, _ in points)
    assert all(isinstance(stamps[key, step], int) for key, step, _ in points)


def test_sampler_publishes_container_usage_relative_to_cgroup_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(system_metrics, "_current_cgroup", lambda: tmp_path)
    clock = iter((1_000_000_000, 1_500_000_000))
    monkeypatch.setattr(system_metrics.time, "monotonic_ns", lambda: next(clock))
    (tmp_path / "cpu.max").write_text("100000 100000\n")
    usage = tmp_path / "cpu.stat"
    usage.write_text("usage_usec 1000000\nuser_usec 900000\n")
    (tmp_path / "memory.max").write_text(str(2 * 1024**3))
    (tmp_path / "memory.current").write_text(str(512 * 1024**2))
    sampler = SystemMetricsSampler(tmp_path)
    sampler._started = system_metrics.time.monotonic()
    sampler.sample(cpu=False)
    usage.write_text("usage_usec 1250000\nuser_usec 1100000\n")
    sampler.sample()
    points, _ = system_metric_points(sampler.report())
    latest = {name: value for name, _, value in points}
    assert latest["system/container_cpu_utilization_percentage"] == 50
    assert latest["system/container_cpu_usage_vcpus"] == 0.5
    assert latest["system/container_cpu_limit_vcpus"] == 1
    assert latest["system/container_memory_usage_megabytes"] == 512
    assert latest["system/container_memory_limit_megabytes"] == 2048
    assert latest["system/container_memory_usage_percentage"] == 25
    assert "system/cpu_utilization_percentage" not in latest
    assert "system/system_memory_usage_percentage" not in latest
    if sampler.report()["collector"] == "psutil":
        assert latest["system/process_threads"] >= 1


def test_runs_without_tracking_collect_the_full_default_profile(tmp_path):
    report = run(small(), tmp_path)
    assert small().tracking is None and report["tracking"]["settings"]["adaptation_shadows"] is True
    assert report["tracking"]["settings"]["task_evaluation"] == DEFAULT_TRACKING.task_evaluation.model_dump()
    assert report["task_evaluation"]["near_future"]["scheduled_windows"] > 0
    index = assert_artifacts(
        tmp_path / report["run_key"],
        {"performance", "probabilistic", "drift", "adaptation", "summary", "system"},
    )
    names = {f"{f['group']}/{f['name']}" for f in index["figures"]}
    assert {
        "performance/accuracy",
        "drift/timeline",
        "continual/accuracy_matrix",
        "system/cpu",
    } <= names or "system/memory_megabytes" in names
    assert report["artifacts"]["figures"] == len(index["figures"])
    assert "data/summary/metrics.json" in index["data"] and "data/system/samples.json" in index["data"]
    explicit = run(small(tracking={"probe_samples": 4}), tmp_path / "explicit")
    assert explicit["tracking"]["settings"]["adaptation_shadows"] is False  # Explicit settings win.


@pytest.mark.parametrize("mode", ["incremental", "offline"])
def test_other_binary_protocols_write_artifacts(tmp_path, mode):
    changes = {"mode": mode} if mode == "incremental" else {"mode": mode, "policy": {"name": "none"}}
    report = run(small(**changes), tmp_path)
    assert_artifacts(tmp_path / report["run_key"], {"performance", "summary"})


def test_typed_and_epoch_runs_write_artifacts_and_system_metrics(tmp_path):
    source = make_source(default_registry(), DatasetConfig(name="friedman_drift", samples=200, label_delay=2))
    typed = run_typed(source, tmp_path / "typed", policy="periodic", every=40, max_samples=200)
    assert typed["system_metrics"]["samples"]
    assert_artifacts(tmp_path / "typed" / typed["run_key"], {"performance", "adaptation", "costs"})
    assert run_typed(source, tmp_path / "typed", policy="periodic", every=40, max_samples=200) == typed
    epochs = run_epochs(EpochConfig(dataset="sea", examples=60), tmp_path / "epochs", tmp_path / "cache")
    assert epochs["system_metrics"]["samples"]
    assert_artifacts(tmp_path / "epochs" / epochs["run_key"], {"performance", "costs", "compute"})


class ArtifactClient:
    def __init__(self):
        self.metrics, self.artifacts, self.tags = {}, {}, {}

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics=dict.fromkeys(self.metrics)))

    def get_metric_history(self, run_id, key):
        return self.metrics.get(key, [])

    def log_batch(self, run_id, metrics=(), synchronous=None):
        for metric in metrics:
            self.metrics.setdefault(metric.key, []).append(metric)

    def log_artifacts(self, run_id, local_dir, artifact_path=None):
        self.artifacts[artifact_path] = sorted(
            p.relative_to(local_dir).as_posix() for p in __import__("pathlib").Path(local_dir).rglob("*.*")
        )

    def set_tag(self, run_id, key, value):
        self.tags[key] = value


def test_publication_logs_timestamped_system_metrics_and_segmented_artifacts_idempotently(tmp_path):
    pytest.importorskip("mlflow")
    report = run(small(), tmp_path)
    client = ArtifactClient()
    first = publish_run_artifacts(client, "run", tmp_path / report["run_key"], report)
    assert first["system_metric_points"] > 0 and set(client.artifacts) == {"plots", "data"}
    assert any(path.startswith("performance/") for path in client.artifacts["plots"])
    assert "index.json" in client.artifacts["data"]
    sample = next(iter(client.metrics.values()))[0]
    assert sample.key.startswith("system/") and sample.timestamp > 0
    counts = {key: len(values) for key, values in client.metrics.items()}
    publish_run_artifacts(client, "run", tmp_path / report["run_key"], report)
    assert {key: len(values) for key, values in client.metrics.items()} == counts
