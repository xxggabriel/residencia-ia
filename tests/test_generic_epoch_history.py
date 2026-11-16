import hashlib
import json
import pickle
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from continual_bench.application.cloud.fixtures import fixture_parquet
from continual_bench.config import EpochConfig
from continual_bench.core.contracts import TaskSample
from continual_bench.datasets.arrays import ListDataset
from continual_bench.evaluation.epochs import run_epochs
from continual_bench.evaluation.iterative import run_iterative
from continual_bench.orchestration import sagemaker


class PartitionedSamples:
    def manifest(self):
        return {"task": "multiclass", "version": "fixture-v1"}

    def samples_iter(self):
        for index in range(20):
            yield TaskSample(str(index), index, index, index, {"x": float(index)}, str(index % 2),
                             split="train" if index < 16 else "test")


def check_artifacts(directory, report):
    history_path = directory / "training-history.json"
    assert json.loads(history_path.read_text())["fits"] == report["training_history"]
    index_path = directory / "artifacts" / "data" / "index.json"
    index = json.loads(index_path.read_text())
    for fit in report["training_history"]:
        checkpoint = fit["checkpoint"]
        path = directory / checkpoint["file"]
        assert path.stat().st_size == checkpoint["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
        group = f"training/{fit['fit_id']}"
        figures = [figure for figure in index["figures"] if figure["group"] == group]
        assert figures
        for figure in figures:
            assert (directory / "artifacts" / figure["plot"]).is_file()
            assert (directory / "artifacts" / figure["data"]).is_file()


def test_epochs_preserve_holdout_scope_and_publishable_checkpoint(tmp_path):
    config = EpochConfig(dataset="sea", examples=20, epochs=2, model={"name": "logistic"})
    report = run_epochs(config, tmp_path, tmp_path, source=PartitionedSamples())
    directory = tmp_path / report["run_key"]
    [fit] = report["training_history"]
    assert [point["step"] for point in fit["steps"]] == [1, 2]
    assert fit["training_presentations"] == 32 and fit["validation_samples"] == 0
    for step, epoch in zip(fit["steps"], report["epochs"], strict=True):
        assert step["metrics"]["holdout:accuracy"] == epoch["metrics"]["accuracy"]
        assert all(not key.startswith("validation:") for key in step["metrics"])
    model = pickle.loads((directory / "model.pkl").read_bytes())
    assert model.predict_one({"x": 18.0}) in ("0", "1")
    check_artifacts(directory, report)
    (directory / "model.pkl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="training checkpoint checksum"):
        run_epochs(config, tmp_path, tmp_path, source=PartitionedSamples())


def test_iterative_groups_regression_updates_and_only_scores_disjoint_validation(tmp_path):
    values = [float(index) for index in range(7)]
    data = ListDataset("fixture", "regression", [{"x": x} for x in values], values, [0] * 6 + [1], {})
    report = run_iterative(data, tmp_path, max_epochs=2, patience=2, log=lambda _: None)
    first, last = report["training_history"]
    assert [step["step"] for step in first["steps"]] == [1, 2]
    assert [step["step"] for step in last["steps"]] == [1, 2]
    assert [step["global_step"] for step in last["steps"]] == [3, 4]
    assert first["summary"]["selection_scope"] == "validation"
    assert first["summary"]["selection_metric"] == "mae"
    assert first["steps"][0]["metrics"]["validation:mae"] >= 0
    assert "validation:log_loss" not in first["steps"][0]["metrics"]
    assert last["validation_samples"] == 0
    assert last["summary"]["selection_scope"] == "train"
    assert last["summary"]["best_validation_loss"] is None
    assert last["checkpoint"]["file"] == "model.pkl"
    check_artifacts(tmp_path, report)


@pytest.mark.parametrize("dataset", ["social", "fraud", "ecommerce"])
def test_native_training_records_epochs_and_preserves_model_package(tmp_path, dataset):
    pytest.importorskip("pyarrow")
    if dataset != "social":
        pytest.importorskip("xgboost")
    fixture_parquet(tmp_path / "train" / "smoke.parquet", dataset=dataset)
    model_dir = tmp_path / "model"
    report = sagemaker.train(tmp_path / "train", model_dir, {"max_epochs": 2})
    assert report["metadata"]["dataset"] == dataset
    assert report["metadata"]["splits"]["holdout_start_excluded"] == 9000
    [fit] = report["training_history"]
    assert (fit["start_seq"], fit["end_seq"]) == (0, 9000)
    assert [step["metrics"] for step in fit["steps"]] == report["training"]["history"]
    assert json.loads((model_dir / "report.json").read_text()) == report
    assert sagemaker.load_predictor(model_dir).features == ["feature"]
    check_artifacts(model_dir, report)


def test_direct_native_training_publishes_grouped_evidence_when_configured(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    from continual_bench.infrastructure.aws import native_tracking

    fixture_parquet(tmp_path / "input" / "data" / "train" / "smoke.parquet")
    config = tmp_path / "input" / "config"
    config.mkdir()
    (config / "hyperparameters.json").write_text(json.dumps({"max_epochs": "2"}))
    monkeypatch.setattr(sagemaker, "PREFIX", tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "app")
    monkeypatch.setenv("SM_TRAINING_ENV", json.dumps({"job_name": "direct-job"}))
    monkeypatch.delenv("TRAINING_JOB_NAME", raising=False)
    monkeypatch.delenv("BENCH_NATIVE_EXECUTION_ID", raising=False)
    tracker = Mock()

    @contextmanager
    def tracked(uri, execution, stage, job, **metadata):
        assert (uri, execution, stage, job) == ("app", "direct-job", "training", "direct-job")
        assert metadata == {"scope": "smoke", "dataset": "social", "model_name": "multinomial_logistic"}
        yield tracker

    monkeypatch.setattr(native_tracking, "track_native_job", tracked)
    sagemaker.main(["train"])
    report, directory = tracker.publish_training_evidence.call_args.args
    assert directory == tmp_path / "model"
    assert len(report["training_history"][0]["steps"]) == 2
    check_artifacts(directory, report)
