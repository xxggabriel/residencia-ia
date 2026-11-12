"""Unit tests for SageMakerExperimentsTracker using botocore.stub.Stubber."""

import boto3
import pytest
from botocore.stub import ANY, Stubber

from continual_bench.config.iterative import IterativeTrainingConfig
from continual_bench.infrastructure.aws.experiments import (
    REQUIRED_EPOCH_METRICS,
    SageMakerExperimentsTracker,
)


def _stubbed_clients():
    sm = boto3.client(
        "sagemaker",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    metrics = boto3.client(
        "sagemaker-metrics",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    return sm, metrics


def test_sagemaker_experiments_hierarchy_tags_parameters_and_epoch_metrics():
    sm_client, metrics_client = _stubbed_clients()
    tracker = SageMakerExperimentsTracker(
        project="continual-bench",
        dataset="social",
        policy="drift",
        model_name="multinomial-logistic",
        run_id="a1b2c3d4e5",
        job_name="cb-social-drift-a1b2",
        git_sha="1" * 40,
        stack="continual-bench",
        client=sm_client,
        metrics_client=metrics_client,
    )
    assert tracker.experiment_name == "continual-bench-zia-experiments"
    assert tracker.trial_name == "social-drift-multinomial-logistic-a1b2c3d4e5"
    assert tracker.trial_component_name == "cb-social-drift-a1b2-training"
    assert tracker.operational_tags() == [
        {"Key": "cb-dataset", "Value": "social"},
        {"Key": "cb-policy", "Value": "drift"},
        {"Key": "cb-stack", "Value": "continual-bench"},
        {"Key": "cb-git-sha", "Value": "1" * 40},
    ]

    exp_arn = "arn:aws:sagemaker:us-east-1:123456789012:experiment/continual-bench-zia-experiments"
    tc_arn = "arn:aws:sagemaker:us-east-1:123456789012:experiment-trial-component/tc1"
    trial_arn = "arn:aws:sagemaker:us-east-1:123456789012:experiment-trial/t1"

    with Stubber(sm_client) as sm_stub, Stubber(metrics_client) as metrics_stub:
        sm_stub.add_client_error("describe_experiment", "ResourceNotFound")
        sm_stub.add_response(
            "create_experiment",
            {"ExperimentArn": exp_arn},
            {
                "ExperimentName": "continual-bench-zia-experiments",
                "Description": ANY,
                "Tags": tracker.operational_tags(),
            },
        )
        sm_stub.add_client_error("describe_trial", "ResourceNotFound")
        sm_stub.add_response(
            "create_trial",
            {"TrialArn": trial_arn},
            {
                "TrialName": "social-drift-multinomial-logistic-a1b2c3d4e5",
                "ExperimentName": "continual-bench-zia-experiments",
                "Tags": tracker.operational_tags(),
            },
        )
        sm_stub.add_client_error("describe_trial_component", "ResourceNotFound")
        sm_stub.add_response(
            "create_trial_component",
            {"TrialComponentArn": tc_arn},
            {
                "TrialComponentName": "cb-social-drift-a1b2-training",
                "DisplayName": "cb-social-drift-a1b2-training",
                "Status": {"PrimaryStatus": "InProgress"},
                "Parameters": ANY,
                "Tags": tracker.operational_tags(),
            },
        )
        sm_stub.add_response(
            "associate_trial_component",
            {
                "TrialComponentArn": tc_arn,
                "TrialArn": trial_arn,
            },
            {
                "TrialComponentName": "cb-social-drift-a1b2-training",
                "TrialName": "social-drift-multinomial-logistic-a1b2c3d4e5",
            },
        )
        # 13 epoch metrics -> chunked into 2 batch_put_metrics calls (10 + 3)
        metrics_stub.add_response(
            "batch_put_metrics",
            {},
            {"TrialComponentName": "cb-social-drift-a1b2-training", "MetricData": ANY},
        )
        metrics_stub.add_response(
            "batch_put_metrics",
            {},
            {"TrialComponentName": "cb-social-drift-a1b2-training", "MetricData": ANY},
        )
        sm_stub.add_response(
            "update_trial_component",
            {"TrialComponentArn": tc_arn},
            {
                "TrialComponentName": "cb-social-drift-a1b2-training",
                "Status": {"PrimaryStatus": "Completed", "Message": ANY},
                "Parameters": ANY,
            },
        )

        config = IterativeTrainingConfig()
        assert tracker.initialize(config.to_hyperparameters_dict()) is True
        epoch_payload = {
            "train:log_loss": 0.45,
            "train:accuracy": 0.82,
            "validation:log_loss": 0.48,
            "validation:accuracy": 0.80,
            "validation:f1_macro": 0.78,
            "epoch": 1,
            "early_stopped": 0,
            "best_epoch": 1,
            "patience_consumed": 0,
            "training_duration_seconds": 0.12,
            "validation_duration_seconds": 0.03,
            "peak_rss_mb": 128.5,
            "best_validation_loss": 0.48,
        }
        assert set(epoch_payload) == set(REQUIRED_EPOCH_METRICS)
        tracker.log_epoch(epoch_payload)
        tracker.finalize(
            {"epochs_trained": 1, "best_epoch": 1, "early_stopped": False, "best_validation_loss": 0.48}
        )

        sm_stub.assert_no_pending_responses()
        metrics_stub.assert_no_pending_responses()

    assert len(tracker.history) == 1
    assert tracker.history[0]["validation:log_loss"] == pytest.approx(0.48)


def test_sagemaker_experiments_tracker_is_fail_safe_on_transient_errors(capsys):
    sm_client, metrics_client = _stubbed_clients()
    tracker = SageMakerExperimentsTracker(
        project="continual-bench",
        dataset="fraud",
        policy="periodic",
        model_name="fraud-booster",
        run_id="run999",
        job_name="cb-fraud-periodic-run999",
        git_sha="2" * 40,
        client=sm_client,
        metrics_client=metrics_client,
    )

    with Stubber(sm_client) as sm_stub:
        sm_stub.add_client_error("describe_experiment", "ThrottlingException", "Rate exceeded")
        # Initialization fails gracefully without raising
        assert tracker.initialize({"max_epochs": 100}) is False
        # Subsequent log_epoch and finalize emit to stdout and never abort the benchmark
        tracker.log_epoch({"epoch": 1, "validation:log_loss": 0.2345, "validation:accuracy": 0.91})
        tracker.finalize({"epochs_trained": 1, "best_epoch": 1, "early_stopped": False})

    captured = capsys.readouterr().out
    assert "validation:log_loss=0.2345;" in captured
    assert "validation:accuracy=0.91;" in captured
    assert len(tracker.history) == 1


def test_sagemaker_mlflow_tracker_logs_epochs_report_and_uploads_artifacts(tmp_path):
    import io
    import json
    import urllib.error
    from pathlib import Path

    from continual_bench.infrastructure.aws.mlflow_tracking import (
        SageMakerMlflowTracker,
        parse_sagemaker_mlflow_arn,
        presigned_mlflow_url,
        sanitize_mlflow_name,
    )

    app_arn = "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-ABCDEF123456"
    parsed_app = parse_sagemaker_mlflow_arn(app_arn)
    assert parsed_app["kind"] == "mlflow-app"
    assert parsed_app["service_name"] == "sagemaker"
    assert parsed_app["header_name"] == "x-sm-mlflow-app-arn"
    assert parsed_app["endpoints"] == [
        "https://mlflow.sagemaker.us-east-1.app.aws",
        "https://api.mlflow.sagemaker.us-east-1.app.aws",
    ]

    ts_arn = "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server"
    parsed_ts = parse_sagemaker_mlflow_arn(ts_arn)
    assert parsed_ts["kind"] == "mlflow-tracking-server"
    assert parsed_ts["service_name"] == "sagemaker-mlflow"
    assert parsed_ts["header_name"] == "x-mlflow-sm-tracking-server-arn"
    assert parsed_ts["endpoints"] == [
        "https://us-east-1.experiments.sagemaker.aws",
        "https://api.us-east-1.experiments.sagemaker.aws",
    ]

    assert sanitize_mlflow_name("train:log_loss") == "train/log_loss"

    requests_seen: list[tuple[str, str, dict, dict | None]] = []

    class FakeHttpResponse:
        def __init__(self, body: dict):
            self._bytes = json.dumps(body).encode("utf-8")

        def read(self) -> bytes:
            return self._bytes

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_opener(req, timeout=10.0):
        body = json.loads(req.data.decode("utf-8")) if req.data else None
        headers = {k.lower(): v for k, v in req.header_items()}
        requests_seen.append((req.get_method(), req.full_url, headers, body))
        # Verify SigV4 headers and x-sm-mlflow-app-arn are signed and attached
        assert headers.get("x-sm-mlflow-app-arn") == app_arn
        assert "authorization" in headers
        assert "x-amz-content-sha256" in headers

        if "/experiments/get-by-name" in req.full_url:
            err_bytes = json.dumps(
                {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "not found"}
            ).encode("utf-8")
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, io.BytesIO(err_bytes)
            )
        if req.full_url.endswith("/experiments/create"):
            return FakeHttpResponse({"experiment_id": "7"})
        if req.full_url.endswith("/runs/create"):
            return FakeHttpResponse(
                {
                    "run": {
                        "info": {
                            "run_id": "run-xyz-123",
                            "experiment_id": "7",
                            "artifact_uri": "s3://bench-bucket/mlflow/7/run-xyz-123/artifacts",
                        }
                    }
                }
            )
        if "/registered-models/get" in req.full_url:
            err_bytes = json.dumps(
                {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "not found"}
            ).encode("utf-8")
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, io.BytesIO(err_bytes)
            )
        if req.full_url.endswith("/model-versions/create"):
            created_count = sum(1 for c in requests_seen if c[1].endswith("/model-versions/create"))
            return FakeHttpResponse(
                {"model_version": {"name": body["name"], "version": str(created_count)}}
            )
        return FakeHttpResponse({})

    uploaded: list[tuple[str, Path]] = []

    class FakeStore:
        bucket = "bench-bucket"

        def put_file(self, key: str, local_path: Path):
            uploaded.append((key, Path(local_path)))

    test_session = boto3.Session(
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region_name="us-east-1",
    )

    mlflow_tracker = SageMakerMlflowTracker(
        project="continual-bench",
        dataset="social",
        policy="drift",
        model_name="multinomial-logistic",
        run_id="a1b2c3d4e5",
        job_name="cb-social-drift-a1b2",
        git_sha="1" * 40,
        stack="continual-bench",
        scope="smoke",
        campaign_id="camp-01",
        tracking_uri=app_arn,
        boto3_session=test_session,
        http_opener=fake_opener,
    )

    assert mlflow_tracker.initialize({"max_epochs": 5, "learning_rate": 0.01}) is True
    assert mlflow_tracker.experiment_id == "7"
    assert mlflow_tracker.mlflow_run_id == "run-xyz-123"

    mlflow_tracker.log_epoch({"epoch": 1, "train:log_loss": 0.5, "validation:accuracy": 0.8})
    mlflow_tracker.log_epoch({"epoch": 2, "train:log_loss": 0.4, "validation:accuracy": 0.85})
    assert mlflow_tracker.global_epoch_step == 2
    mlflow_tracker.finalize({"epochs_trained": 2, "best_epoch": 2, "best_validation_loss": 0.39})

    # Prepare local output directory and report
    (tmp_path / "artifacts" / "plots").mkdir(parents=True)
    (tmp_path / "artifacts" / "plots" / "accuracy.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "artifacts" / "data").mkdir(parents=True)
    (tmp_path / "artifacts" / "data" / "windows.json").write_text('{"windows": []}', encoding="utf-8")
    (tmp_path / "report.json").write_text('{"status": "ok"}', encoding="utf-8")
    (tmp_path / "model").mkdir(parents=True)
    (tmp_path / "model" / "model.tar.gz").write_bytes(b"tar-bytes")

    report = {
        "dataset": "social",
        "scope": "smoke",
        "policy": "drift",
        "model": "multinomial-logistic",
        "protocol": "streaming_prequential",
        "contract": {
            "holdout_start": 400,
            "block_rows": 100,
            "input": {
                "rows": 600,
                "sha256": "deadbeef" * 8,
                "features": ["f1", "f2"],
                "source": {"repo": "social-stream", "revision": "main", "file": "social.parquet"},
            },
            "environment": {"code_sha256": "abc123", "git_revision": "1" * 40},
        },
        "metrics": {"accuracy": 0.84, "f1_macro": 0.81, "update_count": 3, "retrain_count": 2},
        "holdout": {"accuracy": 0.83, "f1_macro": 0.80},
        "windows": [
            {"scope": "prequential", "end_seq": 100, "accuracy": 0.82, "f1_macro": 0.79},
            {"scope": "prequential", "end_seq": 200, "accuracy": 0.86, "f1_macro": 0.83},
        ],
        "class_metrics": {"0": {"f1": 0.85, "precision": 0.84, "recall": 0.86}},
        "model_versions": [
            {
                "action": "initial",
                "file": "models/0000.npy",
                "end_seq": 100,
                "train_samples": 100,
                "train_seconds": 0.42,
            },
            {
                "action": "drift_retrain",
                "file": "models/0001.npy",
                "end_seq": 200,
                "train_samples": 100,
                "train_seconds": 0.38,
            },
        ],
        "system_metrics": {"samples": [{"timestamp_ms": 1700000000000, "values": {"rss_mb": 140.0}}]},
    }

    res = mlflow_tracker.publish_run_report(
        report=report,
        output_dir=tmp_path,
        store=FakeStore(),
        outputs_uri="s3://bench-bucket/campaigns/camp-01/runs/a1b2c3d4e5",
        model_package_arn="arn:aws:sagemaker:us-east-1:123456789012:model-package/cb-social-drift/1",
    )
    assert res["published"] is True
    assert res["mlflow_run_id"] == "run-xyz-123"
    assert res["mlflow_experiment_id"] == "7"
    assert res["artifact_uri"] == "s3://bench-bucket/mlflow/7/run-xyz-123/artifacts"
    assert res["registered_model"] == "continual-bench-zia-social-drift-smoke"
    assert res["registered_model_versions"] == ["1", "2"]
    assert res["registered_model_version"] == "2"
    assert res["registered_family_model"] == "continual-bench-zia-social-smoke"
    assert res["registered_family_version"] == "3"

    uploaded_keys = {key for key, _ in uploaded}
    assert "mlflow/7/run-xyz-123/artifacts/plots/accuracy.svg" in uploaded_keys
    assert "mlflow/7/run-xyz-123/artifacts/data/windows.json" in uploaded_keys
    assert "mlflow/7/run-xyz-123/artifacts/report.json" in uploaded_keys
    assert "mlflow/7/run-xyz-123/artifacts/model/model.tar.gz" in uploaded_keys

    update_calls = [c for c in requests_seen if c[1].endswith("/runs/update")]
    assert update_calls and update_calls[-1][3]["status"] == "FINISHED"

    # Also verify presigned_mlflow_url with Stubber
    sm_client, _ = _stubbed_clients()
    with Stubber(sm_client) as sm_stub:
        sm_stub.add_response(
            "create_presigned_mlflow_app_url",
            {"AuthorizedUrl": "https://mlflow.sagemaker.us-east-1.app.aws/auth?token=xyz"},
            {
                "Arn": app_arn,
                "ExpiresInSeconds": 300,
                "SessionExpirationDurationInSeconds": 43200,
            },
        )
        presigned = presigned_mlflow_url(app_arn, sagemaker_client=sm_client)
        assert presigned["kind"] == "mlflow-app"
        assert presigned["authorized_url"].startswith("https://mlflow.sagemaker.us-east-1.app.aws/")


