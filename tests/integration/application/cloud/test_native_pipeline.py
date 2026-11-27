"""Check native pipeline inputs, tuning parameters and SageMaker job tracking."""

import signal
from importlib import import_module
from pathlib import Path
from runpy import run_path
from unittest.mock import Mock

import pytest

from continual_bench.application.cloud.fixtures import fixture_parquet
from continual_bench.infrastructure.aws import native_tracking
from continual_bench.models.training import loop
from continual_bench.orchestration import sagemaker


def test_native_pipeline_uses_processing_output_and_tracks_each_job(monkeypatch):
    scripts = Path(__file__).parents[4] / "scripts" / "aws"
    monkeypatch.syspath_prepend(str(scripts))
    make = run_path(str(scripts / "native_pipeline.py"))["pipeline_definition"]
    definition = make(
        bucket="b",
        role="role",
        runtime_image="runtime",
        ml_image="ml",
        git_sha="a" * 40,
        key_arn="key",
        mlflow_arn="app",
    )
    steps = {step["Type"]: step for step in definition["Steps"]}
    processing = steps["Processing"]["Arguments"]
    training = steps["Training"]["Arguments"]
    tuning = steps["Tuning"]["Arguments"]
    trial = tuning["TrainingJobDefinition"]
    assert processing["AppSpecification"]["ImageUri"] == "runtime"
    assert "TrainingChannel" in {o["OutputName"] for o in processing["ProcessingOutputConfig"]["Outputs"]}
    for job in (training, trial):
        assert job["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]["Get"].startswith(
            "Steps.ProfileSnapshot."
        )
    assert steps["Tuning"]["DependsOn"] == ["TrainBaseline"]
    assert "max_epochs" not in trial["StaticHyperParameters"]
    integer_ranges = tuning["HyperParameterTuningJobConfig"]["ParameterRanges"]["IntegerParameterRanges"]
    assert integer_ranges[0]["Name"] == "max_epochs"
    assert integer_ranges[0]["MaxValue"] == {
        "Std:Join": {"On": "", "Values": [{"Get": "Parameters.TuningMaxEpochs"}]}
    }
    for stage, job in (("processing", processing), ("training", training), ("tuning", trial)):
        assert job["Environment"]["MLFLOW_TRACKING_URI"] == "app"
        assert job["Environment"]["BENCH_NATIVE_STAGE"] == stage
        assert job["Environment"]["BENCH_NATIVE_EXECUTION_ID"] == {"Get": "Execution.PipelineExecutionId"}


def test_interrupted_native_training_is_killed_before_publishing_results(monkeypatch, tmp_path, capsys):
    fixture_parquet(tmp_path / "input" / "data" / "train" / "smoke.parquet")
    monkeypatch.setattr(sagemaker, "PREFIX", tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "app")
    monkeypatch.setenv("BENCH_NATIVE_EXECUTION_ID", "execution")
    monkeypatch.setenv("BENCH_NATIVE_STAGE", "tuning")
    monkeypatch.setenv("TRAINING_JOB_NAME", "trial")
    monkeypatch.setattr(signal, "signal", Mock())
    # Bind learner imports before substituting the loop.
    import_module("continual_bench.models.stream_models")
    monkeypatch.setattr(loop, "train_iterative", Mock(return_value={"stopped_by_signal": True}))
    tracker = Mock()
    tracker.start.return_value = {"run_id": "run"}
    monkeypatch.setattr(native_tracking, "NativeJobRun", Mock(return_value=tracker))

    with pytest.raises(SystemExit) as stopped:
        sagemaker.native_pipeline_job("train")

    assert stopped.value.code == 128 + signal.SIGTERM
    tracker.finish.assert_called_once_with("KILLED")
    tracker.log_epoch_metrics.assert_not_called()
    assert not (tmp_path / "model").exists()
    assert "validation:log_loss=" not in capsys.readouterr().out
