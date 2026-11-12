import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("mlflow")
pytest.importorskip("sagemaker_mlflow")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from continual_bench.infrastructure.aws import sagemaker_mlflow as sm  # noqa: E402
from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker  # noqa: E402
from tests.test_mlflow_figures import sample_report  # noqa: E402

ARN = "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-TEST"


def report():
    data = sample_report()
    data.update(
        dataset="airlines",
        scope="smoke",
        policy="adaptive",
        model="zia-xgb",
        protocol="prequential-v1",
        contract={
            "input": {
                "sha256": "a" * 64,
                "rows": 2000,
                "source": {"repo": "airlines", "revision": "r1", "file": "data.parquet"},
            },
            "environment": {"code_sha256": "c" * 64, "git_revision": "abc1234"},
        },
        model_versions=[
            {
                "action": "initial",
                "end_seq": 0,
                "file": "model/model.tar.gz",
                "sha256": "1" * 64,
                "train_samples": 500,
            },
            {
                "action": "update",
                "end_seq": 800,
                "file": "model/model.tar.gz",
                "sha256": "2" * 64,
                "train_samples": 400,
            },
        ],
    )
    return data


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # skinny build has no SQL backend
    monkeypatch.setattr(sm, "is_sagemaker_arn", lambda uri: True)  # file store stands in for the ARN endpoint
    return SageMakerMlflowTracker(
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        dataset="airlines",
        policy="adaptive",
        model_name="zia-xgb",
        run_id="run-0001",
        scope="smoke",
        git_sha="abc1234",
    )


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "out"
    (out / "model").mkdir(parents=True)
    (out / "model" / "model.tar.gz").write_bytes(b"not-a-real-model")
    (out / "report.json").write_text("{}")
    (out / ".pending-skip").write_text("x")
    return out


def test_library_publication_tracks_figures_dataset_models_and_artifacts(tracker, output_dir):
    result = sm.publish_with_library(tracker, report(), output_dir=output_dir, outputs_uri="s3://b/out")
    assert result["publisher"] == "sagemaker-mlflow" and result["dataset_logged"] is True
    assert len(result["figures"]) == 13

    client = MlflowClient()
    run = client.get_run(result["mlflow_run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["benchmark_complete"] == "true" and run.data.tags["zia_run_key"] == "run-0001"
    assert run.data.params["dataset_sha256"] == "a" * 64
    assert run.data.metrics["summary/accuracy"] == pytest.approx(0.8)
    assert run.inputs.dataset_inputs[0].dataset.digest == ("a" * 32)

    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"plots", "model", "report.json"} <= artifacts
    assert ".pending-skip" not in artifacts
    plots = {a.path for a in client.list_artifacts(run.info.run_id, "plots/performance")}
    assert "plots/performance/dashboard.png" in plots

    name = result["registered_model"]
    versions = client.search_model_versions(f"name='{name}'")
    assert sorted(int(v.version) for v in versions) == [1, 2]
    final = client.get_model_version_by_alias(name, "final")
    assert int(final.version) == 2 and final.tags["is_final"] == "true"
    assert int(client.get_model_version_by_alias(name, "initial").version) == 1
    assert client.get_model_version_by_alias(result["registered_family_model"], "current")


def test_republishing_resumes_the_same_run_without_duplicate_versions(tracker, output_dir):
    first = sm.publish_with_library(tracker, report(), output_dir=output_dir)
    second = sm.publish_with_library(tracker, report(), output_dir=output_dir)
    assert second["mlflow_run_id"] == first["mlflow_run_id"]
    versions = MlflowClient().search_model_versions(f"name='{first['registered_model']}'")
    assert len(versions) == 2


def test_tracker_falls_back_to_rest_when_the_library_path_fails(tracker, monkeypatch):
    monkeypatch.setattr(
        sm, "publish_with_library", lambda *a, **k: (_ for _ in ()).throw(sm.LibraryPublishError("boom"))
    )
    assert (
        tracker._publish_with_library(report(), output_dir=None, outputs_uri=None, model_package_arn=None)
        is None
    )
    assert any("boom" in error for error in tracker.api_errors)


def test_non_arn_uri_is_rejected_so_http_servers_keep_the_rest_path(monkeypatch):
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: pytest.fail("must not configure MLflow"))
    tracker = SageMakerMlflowTracker(
        tracking_uri="https://mlflow.example", dataset="d", policy="p", model_name="m", run_id="r"
    )
    with pytest.raises(sm.LibraryPublishError):
        sm.publish_with_library(tracker, {})
