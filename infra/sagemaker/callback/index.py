"""Run the existing workflows behind SageMaker CallbackSteps."""

import hashlib
import json
import os
import re

import boto3
from botocore.exceptions import ClientError


def code(error):
    return error.response["Error"]["Code"]


def execution_name(message):
    return hashlib.sha256(message["token"].encode()).hexdigest()


def read_json(s3, uri, prefix, *, version=None):
    bucket = os.environ["DATA_BUCKET"]
    location = f"s3://{bucket}/{prefix}"
    if not uri.startswith(location) or len(uri) == len(location):
        raise ValueError(f"Expected an S3 object under {location}")
    args = {"Bucket": bucket, "Key": uri[len(f"s3://{bucket}/"):]}
    if version:
        args["VersionId"] = version
    obj = s3.get_object(**args)
    with obj["Body"] as body:
        if version and obj["ContentLength"] > 250 * 1024:
            raise ValueError("Workflow request exceeds the Step Functions input budget")
        return json.load(body)


def respond(sm, message, *, outputs=None, failure=None):
    args = {"CallbackToken": message["token"], "ClientRequestToken": execution_name(message)}
    if failure is None:
        sm.send_pipeline_execution_step_success(
            **args, OutputParameters=[{"Name": k, "Value": v} for k, v in (outputs or {}).items()]
        )
    else:
        sm.send_pipeline_execution_step_failure(**args, FailureReason=failure[:256])
    return {"done": True}


def enqueue(sfn, message):
    # Keep the SageMaker status outside the bridge input so a Stopping delivery
    # uses the same execution name and input as the normal SQS delivery.
    payload = {key: message[key] for key in ("token", "pipelineExecutionArn", "arguments")}
    try:
        sfn.start_execution(
            stateMachineArn=os.environ["BRIDGE_ARN"],
            name=execution_name(message),
            input=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    except ClientError as error:
        if code(error) != "ExecutionAlreadyExists":
            raise


def tick(message, sfn, sm, s3):
    args = message["arguments"]
    stage = args["stage"]
    progress = sm.describe_pipeline_execution(PipelineExecutionArn=message["pipelineExecutionArn"])
    stopping = progress["PipelineExecutionStatus"] == "Stopping"
    if stage == "verify":
        if stopping:
            return respond(sm, message, failure="Cancelled from SageMaker Studio")
        uri = args["summary_uri"]
        summary = read_json(s3, uri, "campaigns/")
        if summary.get("complete") is not True:
            return respond(sm, message, failure=f"Incomplete campaign summary at {uri}")
        tracking = summary.get("tracking") or {}
        if tracking.get("required") is True:
            if tracking.get("complete") is not True:
                return respond(sm, message, failure=f"Incomplete MLflow publication at {uri}")
            index_uri = summary.get("artifacts_index_uri")
            prefix = uri.rsplit("/", 1)[0] + "/"
            if index_uri or summary.get("snapshot") is not None:
                if not index_uri or not index_uri.startswith(prefix):
                    return respond(sm, message, failure=f"Campaign artifact index is missing from {uri}")
                if not read_json(s3, index_uri, "campaigns/").get("figures"):
                    return respond(
                        sm, message, failure=f"Campaign artifact index has no figures at {index_uri}"
                    )
            else:
                units = summary.get("units") or []
                if not units or any(
                    not (unit.get("artifacts_index_uri") or "").startswith(prefix + "runs/")
                    or not (unit.get("tracking") or {}).get("mlflow_run_id")
                    for unit in units
                ):
                    return respond(sm, message, failure=f"Matrix unit evidence is missing from {uri}")
        return respond(sm, message, outputs={"SummaryUri": uri})

    uri = args["request_uri"]
    if stage == "dataset" and not uri:
        if stopping:
            return respond(sm, message, failure="Cancelled from SageMaker Studio")
        return respond(sm, message, outputs={"DatasetResult": "NotRequested"})
    if stage not in {"dataset", "campaign"} or not args["request_version"]:
        raise ValueError("Invalid stage or missing request object version")

    request = read_json(s3, uri, "studio/requests/", version=args["request_version"])
    child_name = f"studio-{execution_name(message)[:24]}"
    if stage == "campaign":
        campaign = request["request"]["campaign_id"]
        if not re.fullmatch("[a-z0-9][a-z0-9-]{0,39}", campaign):
            raise ValueError("Invalid campaign ID")
        # Preserve the existing abort observer's convention for campaign retries.
        child_name = f"{campaign}--{child_name}"
    machine = os.environ["DATASET_ARN" if stage == "dataset" else "CAMPAIGN_ARN"]
    execution = machine.replace(":stateMachine:", ":execution:", 1) + ":" + child_name

    try:
        child = sfn.describe_execution(executionArn=execution)
    except ClientError as error:
        if code(error) != "ExecutionDoesNotExist":
            raise
        if stopping:
            return respond(sm, message, failure="Cancelled from SageMaker Studio")
        try:
            sfn.start_execution(
                stateMachineArn=machine, name=child_name,
                input=json.dumps(request, sort_keys=True, separators=(",", ":")),
            )
        except ClientError as start_error:
            if code(start_error) != "ExecutionAlreadyExists":
                raise
        return {"done": False}

    status = child["status"]
    if status == "RUNNING":
        if stopping:
            sfn.stop_execution(executionArn=execution, cause="Stopped from SageMaker Pipelines")
        return {"done": False}
    if stopping:
        # A success callback can start dependent steps even after StopPipelineExecution.
        # Fail this step with an explicit cancellation reason once the child is terminal.
        reason = f"Cancelled from SageMaker Studio: workflow {status}: {execution}"
        return respond(sm, message, failure=reason)
    if status != "SUCCEEDED":
        return respond(sm, message, failure=f"Workflow {status}: {execution}")

    if stage == "dataset":
        return respond(sm, message, outputs={"DatasetResult": execution})
    outputs = {
        "WorkflowExecutionArn": execution,
        "SummaryUri": f"s3://{os.environ['DATA_BUCKET']}/campaigns/{campaign}/summary.json",
    }
    return respond(sm, message, outputs=outputs)


def handler(event, _context):
    sfn = boto3.client("stepfunctions")
    if "Records" in event:
        for record in event["Records"]:
            enqueue(sfn, json.loads(record["body"]))
        return
    sm = boto3.client("sagemaker")
    if "failure" in event:
        return respond(sm, event, failure="Callback bridge failed. Inspect the callback workflow execution.")
    return tick(event, sfn, sm, boto3.client("s3"))
