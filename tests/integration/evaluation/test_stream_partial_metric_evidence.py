"""Failed metrics-only attempts retain diagnostics without becoming resumable results."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from continual_bench.config import TrackingConfig
from continual_bench.datasets import pinned_stream as zia
from continual_bench.evaluation import stream_benchmark as stream
from continual_bench.evaluation import stream_predictions as predictions
from continual_bench.evaluation.system_metrics import MAX_SAMPLES, SystemMetricsSampler
from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files


class AttemptFailure(RuntimeError):
    pass


@pytest.fixture
def failed_stream(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    monkeypatch.setattr(zia, "SMOKE_ROWS", 2000)
    path = tmp_path / "social" / "smoke.parquet"
    path.parent.mkdir()
    ids = np.arange(2000)
    labels = (ids % 4).astype(np.uint8)
    pq.write_table(pa.table({"event_time": ids, "label": labels, "feature": labels.astype(np.float32)}), path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": "social",
                "scope": "smoke",
                "source": zia.DATASETS["social"],
                "format": zia.FORMAT,
                "features": ["feature"],
                "rows": len(ids),
                "bytes": path.stat().st_size,
                "sha256": zia.sha256_file(path),
            }
        )
    )
    tracking = TrackingConfig.model_validate(
        yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    ).model_dump()

    class FastSampler(SystemMetricsSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.counter = 0

        def _measure(self, cpu=True):
            point = self.counter
            self.counter += 1
            return {"test_counter": point, "process_memory_rss_megabytes": 1.0}

        def __enter__(self):
            self._started = time.monotonic()
            for _ in range(MAX_SAMPLES + 7):
                self.sample(cpu=False)
            return self

        def __exit__(self, *exc):
            self.sample(cpu=False)
            return False

    monkeypatch.setattr(stream, "SystemMetricsSampler", FastSampler)
    return tmp_path, tracking


def run_fixture(fixture, seed, *, mode="metrics_only", progress=None):
    root, tracking = fixture
    return stream.run(
        root,
        "social",
        "smoke",
        "frozen",
        root / "output",
        tracking,
        artifact_mode=mode,
        iterative_config={"seed": seed, "max_epochs": 1},
        progress=progress,
    )


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("exception", [AttemptFailure, InterruptedError])
def test_failed_attempt_retains_all_system_samples_and_sealed_prediction_parts(
    failed_stream, seed, exception
):
    import pyarrow.parquet as pq

    root, _ = failed_stream
    failure = exception("sensitive details must not enter diagnostics")

    def stop(step, metrics):
        if metrics.get("rows_processed", 0) >= 1800:
            raise failure

    with pytest.raises(exception) as caught:
        run_fixture(failed_stream, seed, progress=stop)
    assert caught.value is failure
    output = root / "output"
    partial = json.loads((output / stream.PARTIAL_SYSTEM_METRICS).read_text())
    assert partial["schema"] == "partial-system-metrics-v1"
    assert partial["complete"] is False and partial["status"] == "incomplete"
    assert partial["seed"] == seed and partial["exception_type"] == exception.__name__
    assert partial["usable_for_resume"] is False and partial["usable_for_final_result"] is False
    assert partial["dataset_sha256"] == zia.sha256_file(root / "social/smoke.parquet")
    system = partial["system_metrics"]
    assert system["max_retained_samples"] is None
    assert [p["values"]["test_counter"] for p in system["samples"]] == list(range(MAX_SAMPLES + 8))
    assert "sensitive details" not in (output / stream.PARTIAL_SYSTEM_METRICS).read_text()
    assert not (output / "report.json").exists()
    assert not (output / predictions.INDEX).exists()
    assert not (output / "predictions").exists()
    manifest = json.loads((output / predictions.PARTIAL_INDEX).read_text())
    assert manifest["complete"] is False and manifest["closed_parts_only"] is True
    assert manifest["usable_for_resume"] is False and manifest["usable_for_final_result"] is False
    initial = json.loads(next((output / "training-fits-partial").glob("*.json")).read_text())
    assert manifest["rows"] == manifest["observed_prediction_rows"] == 1800 - initial["end_seq"]
    assert manifest["parts"]
    for part in manifest["parts"]:
        path = output / predictions.PARTIAL_DIRECTORY / part["file"]
        assert zia.sha256_file(path) == part["sha256"]
        assert pq.ParquetFile(path).metadata.num_rows == part["rows"]
    assert not list(output.glob(".predictions-*"))
    verify_metrics_only_files(output)
    # A partial manifest never satisfies the final prediction-manifest lookup.
    assert predictions.prediction_manifest(output, {}) is None


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_post_loop_failure_also_retains_system_evidence(failed_stream, monkeypatch, seed):
    failure = AttemptFailure("postprocessing failed")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(stream, "run_observations", fail)
    with pytest.raises(AttemptFailure) as caught:
        run_fixture(failed_stream, seed)
    assert caught.value is failure
    root, _ = failed_stream
    partial = json.loads((root / "output" / stream.PARTIAL_SYSTEM_METRICS).read_text())
    assert partial["seed"] == seed
    assert len(partial["system_metrics"]["samples"]) == MAX_SAMPLES + 8
    assert not (root / "output/report.json").exists()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_success_creates_only_complete_prediction_manifest(failed_stream, seed):
    root, _ = failed_stream
    report = run_fixture(failed_stream, seed)
    output = root / "output"
    assert report["complete"]
    assert predictions.prediction_manifest(output, report)["rows"] == (
        report["metrics"]["n"] + report["holdout"]["n"]
    )
    assert not (output / stream.PARTIAL_SYSTEM_METRICS).exists()
    assert not (output / predictions.PARTIAL_INDEX).exists()
    assert not (output / predictions.PARTIAL_DIRECTORY).exists()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_full_retention_keeps_legacy_exception_cleanup(failed_stream, seed):
    root, _ = failed_stream

    def stop(step, metrics):
        if metrics.get("rows_processed", 0) >= 1800:
            raise AttemptFailure("legacy failure")

    with pytest.raises(AttemptFailure):
        run_fixture(failed_stream, seed, mode="full", progress=stop)
    output = root / "output"
    assert not (output / stream.PARTIAL_SYSTEM_METRICS).exists()
    assert not (output / predictions.PARTIAL_INDEX).exists()
    assert not (output / predictions.PARTIAL_DIRECTORY).exists()
    assert not list(output.glob(".predictions-*"))


def test_partial_prediction_persistence_failure_preserves_original_exception(tmp_path, monkeypatch):
    original = AttemptFailure("original exception")

    def reject_write(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(predictions, "atomic_write", reject_write)
    with pytest.raises(AttemptFailure) as caught:
        with predictions.PredictionArchive(
            tmp_path, ["0", "1"], "a" * 64, "native", preserve_partial=True
        ) as archive:
            archive.add(
                0,
                np.array([0, 1]),
                np.array([0, 1], dtype=np.uint8),
                np.array([[0.9, 0.1], [0.1, 0.9]]),
                False,
                0,
            )
            raise original
    assert caught.value is original
    assert any("partial prediction evidence" in note for note in original.__notes__)
    assert not (tmp_path / predictions.INDEX).exists()


def test_partial_system_persistence_failure_preserves_original_exception(failed_stream, monkeypatch):
    original_write = stream.atomic_write
    original = AttemptFailure("original evaluation exception")

    def write(path, payload):
        if Path(path).name == stream.PARTIAL_SYSTEM_METRICS:
            raise OSError("storage unavailable")
        return original_write(path, payload)

    def stop(step, metrics):
        if metrics.get("rows_processed", 0) >= 1800:
            raise original

    monkeypatch.setattr(stream, "atomic_write", write)
    with pytest.raises(AttemptFailure) as caught:
        run_fixture(failed_stream, 42, progress=stop)
    assert caught.value is original
    assert any("Partial system-metric evidence" in note for note in original.__notes__)
