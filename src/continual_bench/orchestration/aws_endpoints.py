"""Provision the shared interface endpoints before private workers can start."""

import json
import os
import time
from pathlib import Path
from uuid import uuid4

from continual_bench.infrastructure.aws.batch import ACTIVE
from continual_bench.infrastructure.aws.session import client, error_code
from continual_bench.infrastructure.aws.spot import ACTIVE_STATES, COMPONENT

TEMPLATE = Path("infra/cloudformation/modules/interface-endpoints.yaml")
STATE_KEY = "runtime/interface-endpoints.json"
LOCK_SECONDS = 180  # Longer than the Lambda timeout, including SDK retries.
DRAIN_SECONDS = 120  # Let in-flight worker submissions/cancellations become visible.
SPOT_DRAIN_SECONDS = 360  # Cover the control Lambda deadline and EC2 launch visibility after a stop.
STACK_READY = {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
STACK_BUSY = {"CREATE_IN_PROGRESS", "ROLLBACK_IN_PROGRESS", "DELETE_IN_PROGRESS"}


class EndpointsPending(RuntimeError):
    """The workflow must wait before submitting a private job."""


class EndpointLifecycle:
    def __init__(self, env):
        self.s3 = client("s3", max_attempts=2, read_timeout=5)
        self.ec2 = client("ec2", max_attempts=2, read_timeout=5)
        self.cfn = client("cloudformation", max_attempts=2, read_timeout=5)
        self.states = client("stepfunctions", max_attempts=2, read_timeout=5)
        self.batch = client("batch", max_attempts=2, read_timeout=5)
        self.bucket = env["ENDPOINT_STATE_BUCKET"]
        self.kms_key = env["ENDPOINT_STATE_KMS_KEY"]
        self.queue = env["ENDPOINT_JOB_QUEUE"]
        self.benchmark_name = env["ENDPOINT_BENCHMARK_WORKFLOW"]
        self.name = env["ENDPOINT_STACK_NAME"]
        self.parameters = json.loads(env["ENDPOINT_PARAMETERS"])
        self.project = env.get("BENCH_STACK")

    def stack(self):
        try:
            return self.cfn.describe_stacks(StackName=self.name)["Stacks"][0]
        except Exception as error:
            if error_code(error) == "ValidationError" and "does not exist" in str(error):
                return None
            raise

    def read(self):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=STATE_KEY)
        except Exception as error:
            if error_code(error) == "NoSuchKey":
                return {"leases": {}, "locked_until": 0}, None
            raise
        with response["Body"] as body:
            return json.load(body), response["ETag"]

    def write(self, state, etag):
        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        return self.s3.put_object(
            Bucket=self.bucket,
            Key=STATE_KEY,
            Body=json.dumps({**state, "revision": uuid4().hex}).encode(),
            ContentType="application/json",
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=self.kms_key,
            **condition,
        )["ETag"]

    def acquire(self, deleting_stack):
        stack = self.stack()
        if not stack:
            template = Path(os.environ["LAMBDA_TASK_ROOT"]) / TEMPLATE
            try:
                self.cfn.create_stack(
                    StackName=self.name,
                    TemplateBody=template.read_text(),
                    Parameters=[
                        {"ParameterKey": key, "ParameterValue": value}
                        for key, value in self.parameters.items()
                    ],
                    OnFailure="ROLLBACK",
                )
            except Exception as error:
                if error_code(error) != "AlreadyExistsException":
                    raise
            return False

        status = stack["StackStatus"]
        if status in STACK_BUSY or deleting_stack == stack["StackId"]:
            return False
        if status not in STACK_READY:
            raise RuntimeError(f"Endpoint stack {self.name} is {status}")
        deployed = {p["ParameterKey"]: p["ParameterValue"] for p in stack["Parameters"]}
        if deployed != self.parameters:
            raise RuntimeError("Endpoint network settings changed; retry after the current jobs drain.")
        resources = self.cfn.list_stack_resources(StackName=stack["StackId"])["StackResourceSummaries"]
        ids = [r["PhysicalResourceId"] for r in resources if r["ResourceType"] == "AWS::EC2::VPCEndpoint"]
        endpoints = self.ec2.describe_vpc_endpoints(VpcEndpointIds=ids)["VpcEndpoints"] if ids else []
        return bool(ids) and len(endpoints) == len(ids) and all(e["State"] == "available" for e in endpoints)

    def execution_needs_network(self, arn, admitted_at, now):
        try:
            execution = self.states.describe_execution(executionArn=arn)
        except Exception as error:
            if error_code(error) == "ExecutionDoesNotExist":
                return now < admitted_at + DRAIN_SECONDS
            raise
        stop = execution.get("stopDate")
        benchmark = arn.split(":")[-2] == self.benchmark_name
        spot_drain = self.project and benchmark and execution["status"] != "SUCCEEDED"
        drain = SPOT_DRAIN_SECONDS if spot_drain else DRAIN_SECONDS
        if execution["status"] == "RUNNING" or not stop or now < stop.timestamp() + drain:
            return True
        if execution["status"] == "SUCCEEDED" or not benchmark:
            return False

        # A stopped distributed parent can leave children draining independently.
        for page in self.states.get_paginator("list_map_runs").paginate(executionArn=arn):
            for run in page["mapRuns"]:
                map_run = self.states.describe_map_run(mapRunArn=run["mapRunArn"])
                if map_run["status"] == "RUNNING" or map_run["executionCounts"]["running"]:
                    return True
        return False

    def cleanup(self, leases, now, deleting_stack):
        active = {
            arn: admitted
            for arn, admitted in leases.items()
            if self.execution_needs_network(arn, admitted, now)
        }
        stack = self.stack()
        if not stack or stack["StackStatus"] in STACK_BUSY:
            return active, None
        if active and deleting_stack != stack["StackId"]:
            return active, None
        for status in ACTIVE:
            if self.batch.list_jobs(jobQueue=self.queue, jobStatus=status, maxResults=1)["jobSummaryList"]:
                return active, None
        if self.project and self.ec2.describe_instances(Filters=[
            {"Name": "tag:cb-stack", "Values": [self.project]},
            {"Name": "tag:cb-component", "Values": [COMPONENT]},
            {"Name": "instance-state-name", "Values": list(ACTIVE_STATES)},
        ], MaxResults=5)["Reservations"]:
            return active, None
        return active, stack["StackId"]

    def reconcile(self, execution_arn=None):
        now = time.time()
        state, etag = self.read()
        if not execution_arn and not state["leases"] and not self.stack():
            return False
        if state["locked_until"] > now:
            return False
        state["locked_until"] = now + LOCK_SECONDS
        if execution_arn:
            state["leases"].setdefault(execution_arn, now)
        try:
            etag = self.write(state, etag)
        except Exception as error:
            if error_code(error) in {"ConditionalRequestConflict", "PreconditionFailed"}:
                return False
            raise
        try:
            if execution_arn:
                return self.acquire(state.get("deleting_stack"))
            state["leases"], idle_stack = self.cleanup(state["leases"], now, state.get("deleting_stack"))
            if idle_stack:
                # Fence admissions before DeleteStack so a stale DescribeStacks cannot admit a job.
                state["deleting_stack"] = idle_stack
                etag = self.write(state, etag)
                self.cfn.delete_stack(StackName=idle_stack)
            return False
        finally:
            state["locked_until"] = 0
            self.write(state, etag)


def handler(event, _context):
    execution = event.get("execution_arn")
    ready = EndpointLifecycle(os.environ).reconcile(execution)
    if execution and not ready:
        raise EndpointsPending("Waiting for the interface endpoint stack")
    return {"ready": ready}
