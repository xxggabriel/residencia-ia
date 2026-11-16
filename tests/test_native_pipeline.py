"""Check that the native DAG shares a verified channel and tunes effective parameters."""

from pathlib import Path
from runpy import run_path


def test_native_pipeline_uses_processing_output_and_tracks_each_job(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts" / "aws"
    monkeypatch.syspath_prepend(str(scripts))
    make = run_path(str(scripts / "native_pipeline.py"))["pipeline_definition"]
    definition = make(
        bucket="b", role="role", runtime_image="runtime", ml_image="ml",
        git_sha="a" * 40, key_arn="key", mlflow_arn="app",
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
