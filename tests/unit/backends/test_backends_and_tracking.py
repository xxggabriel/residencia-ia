"""Unit tests for the Backends and Tracking layers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from continual_bench.backends import (
    AwsBackend,
    ExecutionBackend,
    HomelabBackend,
    LocalBackend,
    RuntimeSettings,
    StorageBackend,
    resolve_backend,
)
from continual_bench.tracking import BenchmarkTracker


class _FakeS3Client:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, Any]:
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body})
        return {"ETag": "fake-etag"}


def test_runtime_settings_defaults_from_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "BENCH_ENV",
        "BENCH_ENV_CONFIG",
        "BENCH_PLATFORM",
        "BENCH_SAVE_MODEL_CHECKPOINTS",
        "BENCH_TRACKING_ENABLED",
        "BENCH_OUTPUT_DIR",
        "BENCH_TRACKING_URI",
        "MLFLOW_TRACKING_URI",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = RuntimeSettings.from_env(config_path=Path("configs/base/benchmark-defaults.yaml"))
    assert settings.platform == "local"
    assert settings.save_model_checkpoints is False
    assert settings.tracking_enabled is True
    assert settings.output_dir == Path("state/benchmarks")
    assert settings.tracking_uri == ""


def test_runtime_settings_env_and_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BENCH_PLATFORM", "homelab")
    monkeypatch.setenv("BENCH_SAVE_MODEL_CHECKPOINTS", "true")
    monkeypatch.setenv("BENCH_TRACKING_ENABLED", "false")
    monkeypatch.setenv("BENCH_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("BENCH_TRACKING_URI", "http://localhost:5000")

    settings = RuntimeSettings.from_env(
        config_path=Path("configs/base/benchmark-defaults.yaml"),
        overrides={"platform": "aws", "tracking_enabled": True},
    )
    assert settings.platform == "aws"
    assert settings.save_model_checkpoints is True
    assert settings.tracking_enabled is True
    assert settings.output_dir == tmp_path / "out"
    assert settings.tracking_uri == "http://localhost:5000"


def test_resolve_backend_platforms(tmp_path: Path) -> None:
    settings = RuntimeSettings(output_dir=tmp_path)
    assert isinstance(resolve_backend("local", settings), LocalBackend)
    assert isinstance(resolve_backend("homelab", settings), HomelabBackend)
    assert isinstance(resolve_backend("aws", settings), AwsBackend)
    for platform in ("local", "homelab", "aws"):
        backend = resolve_backend(platform, settings)
        assert isinstance(backend, ExecutionBackend)
        assert isinstance(backend, StorageBackend)
    with pytest.raises(ValueError, match="Unsupported backend platform"):
        resolve_backend("unknown", settings)


@pytest.mark.parametrize("platform", ["local", "homelab", "aws"])
def test_zero_checkpoint_enforcement_across_backends(platform: str, tmp_path: Path) -> None:
    out_dir = tmp_path / platform
    settings = RuntimeSettings(platform=platform, save_model_checkpoints=False, output_dir=out_dir)
    backend = resolve_backend(platform, settings)

    result = backend.save_model_checkpoint("models/model.pkl", {"weights": [1, 2, 3]})
    assert result is None
    assert backend.verify_no_checkpoints() == []

    # Stray checkpoint file should be caught by verify_no_checkpoints & assert_no_checkpoints
    backend.write_bytes("stray/checkpoint.ubj", b"bad")
    violations = backend.verify_no_checkpoints()
    assert violations == ["stray/checkpoint.ubj"]

    tracker = BenchmarkTracker(backend=backend, run_id="run-1", settings=settings)
    with pytest.raises(RuntimeError, match="Model checkpoints are forbidden"):
        tracker.assert_no_checkpoints()


def test_save_model_checkpoint_allowed_when_enabled(tmp_path: Path) -> None:
    settings = RuntimeSettings(platform="local", save_model_checkpoints=True, output_dir=tmp_path)
    backend = resolve_backend("local", settings)
    written = backend.save_model_checkpoint("models/model.pkl", {"a": 1})
    assert written is not None
    assert Path(written).exists()
    assert backend.verify_no_checkpoints() == ["models/model.pkl"]


@pytest.mark.parametrize("platform", ["local", "homelab", "aws"])
def test_run_tasks_preserves_order(platform: str, tmp_path: Path) -> None:
    settings = RuntimeSettings(platform=platform, output_dir=tmp_path / platform)
    backend = resolve_backend(platform, settings)
    tasks = [lambda i=i: i * 10 for i in range(6)]
    assert backend.run_tasks(tasks, max_workers=1) == [0, 10, 20, 30, 40, 50]
    assert backend.run_tasks(tasks, max_workers=3) == [0, 10, 20, 30, 40, 50]
    assert backend.run_tasks([], max_workers=2) == []


def test_aws_backend_s3_and_staging_mirror(tmp_path: Path) -> None:
    fake_s3 = _FakeS3Client()
    settings = RuntimeSettings(
        platform="aws",
        save_model_checkpoints=False,
        output_dir="s3://my-bench-bucket/exp/run-42",
    )
    backend = AwsBackend(settings, s3_client=fake_s3, staging_dir=tmp_path / "staging")

    uri = backend.write_json("metrics/summary.json", {"accuracy": 0.95})
    assert uri == "s3://my-bench-bucket/exp/run-42/metrics/summary.json"
    assert len(fake_s3.puts) == 1
    assert fake_s3.puts[0]["Bucket"] == "my-bench-bucket"
    assert fake_s3.puts[0]["Key"] == "exp/run-42/metrics/summary.json"
    assert (tmp_path / "staging" / "metrics" / "summary.json").exists()
    assert backend.save_model_checkpoint("model.pt", b"weights") is None
    assert len(fake_s3.puts) == 1


def test_tracker_flush_writes_json_csv_and_parquet(tmp_path: Path) -> None:
    settings = RuntimeSettings(platform="local", tracking_enabled=True, output_dir=tmp_path)
    backend = resolve_backend("local", settings)
    tracker = BenchmarkTracker(backend=backend, run_id="run-abc", settings=settings)

    tracker.log_metadata({"dataset": "sea", "n_seeds": 1})
    tracker.log_batch(
        seed=0,
        policy="periodic",
        regime=0.5,
        batch_record={
            "batch_index": 1,
            "accuracy": 0.91,
            "error": 0.09,
            "staleness": 1,
            "retrain_decision": False,
            "model_age": 1,
            "fit_time_s": 0.0,
            "eval_time_s": 0.01,
        },
    )
    tracker.log_regime(
        seed=0,
        policy="periodic",
        regime_record={
            "alpha": 0.5,
            "cost": 1.2,
            "strategy_cost": 1.2,
            "scpe": 0.1,
            "retrains": 2,
            "average_accuracy": 0.91,
        },
    )
    tracker.log_seed_summary(seed=0, policy="periodic", summary={"average_accuracy": 0.91})
    tracker.log_aggregate_summary([{"policy": "periodic", "mean_accuracy": 0.91}])

    written = tracker.flush()
    expected_keys = {
        "metadata.json",
        "batch_metrics.csv",
        "batch_metrics.parquet",
        "regime_metrics.csv",
        "regime_metrics.parquet",
        "seed_summaries.csv",
        "seed_summaries.json",
        "aggregate_summary.csv",
        "aggregate_summary.json",
    }
    assert expected_keys.issubset(written.keys())
    for path_str in written.values():
        assert Path(path_str).exists()

    meta = json.loads(Path(written["metadata.json"]).read_text(encoding="utf-8"))
    assert meta["dataset"] == "sea"
    assert meta["run_id"] == "run-abc"

    with Path(written["batch_metrics.csv"]).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["policy"] == "periodic"

    tracker.assert_no_checkpoints()


def test_tracker_noop_when_disabled_still_checks_checkpoints(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        platform="local",
        tracking_enabled=False,
        save_model_checkpoints=False,
        output_dir=tmp_path,
    )
    backend = resolve_backend("local", settings)
    tracker = BenchmarkTracker(backend=backend, run_id="run-disabled", settings=settings)

    tracker.log_metadata({"dataset": "sea"})
    tracker.log_batch(0, "fixed", 1.0, {"batch_index": 0, "accuracy": 0.8})
    tracker.log_regime(0, "fixed", {"alpha": 1.0, "cost": 0.5})
    tracker.log_seed_summary(0, "fixed", {"average_accuracy": 0.8})
    tracker.log_aggregate_summary([{"policy": "fixed", "mean_accuracy": 0.8}])

    assert tracker.flush() == {}
    assert not (tmp_path / "run-disabled").exists()

    backend.write_bytes("bad.pt", b"123")
    with pytest.raises(RuntimeError, match="Model checkpoints are forbidden"):
        tracker.assert_no_checkpoints()


def test_tracker_mlflow_graceful_when_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    settings = RuntimeSettings(
        platform="local",
        tracking_enabled=True,
        tracking_uri="http://127.0.0.1:59999",
        output_dir=tmp_path,
    )
    backend = resolve_backend("local", settings)
    tracker = BenchmarkTracker(backend=backend, run_id="run-mlflow", settings=settings)
    tracker.log_aggregate_summary([{"policy": "periodic", "mean_accuracy": 0.9}])
    written = tracker.flush()
    assert "metadata.json" in written
