"""The Studio callback must mirror the downstream execution, including stops."""

import json
import runpy
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

from botocore.exceptions import ClientError

CALLBACK = runpy.run_path(str(Path(__file__).parents[1] / "infra" / "sagemaker" / "callback" / "index.py"))
MESSAGE = {
    "token": "Abcdef1234",
    "pipelineExecutionArn": "arn:studio-execution",
    "arguments": {
        "stage": "campaign",
        "request_uri": "s3://data/studio/requests/request.json",
        "request_version": "version-1",
    },
}


def clients(monkeypatch):
    monkeypatch.setenv("DATA_BUCKET", "data")
    monkeypatch.setenv("CAMPAIGN_ARN", "arn:aws:states:us-east-1:123456789012:stateMachine:benchmark")
    sfn, sm, s3 = Mock(), Mock(), Mock()
    body = json.dumps({"request": {"campaign_id": "matrix-campaign"}}).encode()
    s3.get_object.side_effect = lambda **_: {"Body": BytesIO(body), "ContentLength": len(body)}
    return sfn, sm, s3


def test_callback_reports_success_only_after_versioned_campaign_succeeds(monkeypatch):
    sfn, sm, s3 = clients(monkeypatch)
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Executing"}
    sfn.describe_execution.side_effect = [
        ClientError({"Error": {"Code": "ExecutionDoesNotExist"}}, "DescribeExecution"),
        {"status": "RUNNING"},
        {"status": "SUCCEEDED"},
    ]
    assert CALLBACK["tick"](MESSAGE, sfn, sm, s3) == {"done": False}
    assert CALLBACK["tick"](MESSAGE, sfn, sm, s3) == {"done": False}
    sm.send_pipeline_execution_step_success.assert_not_called()
    assert CALLBACK["tick"](MESSAGE, sfn, sm, s3) == {"done": True}
    sfn.start_execution.assert_called_once()
    assert sfn.start_execution.call_args.kwargs["name"].startswith("matrix-campaign--studio-")
    assert s3.get_object.call_args.kwargs["VersionId"] == "version-1"
    outputs = sm.send_pipeline_execution_step_success.call_args.kwargs["OutputParameters"]
    assert {"Name": "SummaryUri", "Value": "s3://data/campaigns/matrix-campaign/summary.json"} in outputs


def test_cancel_blocks_dependent_steps_after_abort(monkeypatch):
    sfn, sm, s3 = clients(monkeypatch)
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Stopping"}
    sfn.describe_execution.side_effect = [{"status": "RUNNING"}, {"status": "ABORTED"}]
    assert CALLBACK["tick"](MESSAGE, sfn, sm, s3) == {"done": False}
    sfn.stop_execution.assert_called_once()
    sm.send_pipeline_execution_step_failure.assert_not_called()
    assert CALLBACK["tick"](MESSAGE, sfn, sm, s3) == {"done": True}
    sm.send_pipeline_execution_step_success.assert_not_called()
    reason = sm.send_pipeline_execution_step_failure.call_args.kwargs["FailureReason"]
    assert reason.startswith("Cancelled from SageMaker Studio: workflow ABORTED")


def test_optional_preparation_reports_a_nonempty_result_without_submitting_work(monkeypatch):
    sfn, sm, s3 = clients(monkeypatch)
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Executing"}
    message = {**MESSAGE, "arguments": {"stage": "dataset", "request_uri": "", "request_version": ""}}
    assert CALLBACK["tick"](message, sfn, sm, s3) == {"done": True}
    sfn.start_execution.assert_not_called()
    s3.get_object.assert_not_called()
    assert sm.send_pipeline_execution_step_success.call_args.kwargs["OutputParameters"] == [
        {"Name": "DatasetResult", "Value": "NotRequested"}
    ]


def test_verify_requires_the_tracking_receipt_and_reads_the_campaign_artifact_index(monkeypatch):
    sfn, sm, s3 = clients(monkeypatch)
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Executing"}
    message = {
        **MESSAGE,
        "arguments": {"stage": "verify", "summary_uri": "s3://data/campaigns/zia-test/summary.json"},
    }
    summary = {
        "complete": True,
        "tracking": {"required": True, "complete": False},
        "artifacts_index_uri": "s3://data/campaigns/zia-test/artifacts/data/index.json",
    }

    def fetch(**args):
        doc = summary if args["Key"].endswith("/summary.json") else {"figures": [{"plot": "test.svg"}]}
        return {"Body": BytesIO(json.dumps(doc).encode())}

    s3.get_object.side_effect = fetch
    assert CALLBACK["tick"](message, sfn, sm, s3) == {"done": True}
    sm.send_pipeline_execution_step_success.assert_not_called()
    assert sm.send_pipeline_execution_step_failure.call_args.kwargs["FailureReason"].startswith(
        "Incomplete MLflow publication"
    )
    summary["tracking"]["complete"] = True
    assert CALLBACK["tick"](message, sfn, sm, s3) == {"done": True}
    assert s3.get_object.call_args.kwargs["Key"] == "campaigns/zia-test/artifacts/data/index.json"
    sm.send_pipeline_execution_step_success.assert_called_once()
