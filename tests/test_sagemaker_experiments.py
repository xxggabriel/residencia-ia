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
