from datetime import datetime, timezone
from unittest.mock import Mock

from botocore.exceptions import ClientError

from continual_bench.orchestration.aws_endpoints import EndpointLifecycle

EXECUTION = "arn:aws:states:us-east-1:123456789012:execution:continual-bench-benchmark:campaign"
STACK = "arn:aws:cloudformation:us-east-1:123456789012:stack/continual-bench-interface-endpoints/id"


def lifecycle(monkeypatch):
    clients = {service: Mock() for service in ("s3", "ec2", "cloudformation", "stepfunctions", "batch")}
    monkeypatch.setattr(
        "continual_bench.orchestration.aws_endpoints.client", lambda service, **_: clients[service]
    )
    control = EndpointLifecycle(
        {
            "ENDPOINT_STATE_BUCKET": "data",
            "ENDPOINT_STATE_KMS_KEY": "key",
            "ENDPOINT_JOB_QUEUE": "queue",
            "ENDPOINT_BENCHMARK_WORKFLOW": "continual-bench-benchmark",
            "ENDPOINT_STACK_NAME": "continual-bench-interface-endpoints",
            "ENDPOINT_PARAMETERS": "{}",
        }
    )
    control.cfn.describe_stacks.return_value = {
        "Stacks": [{"StackStatus": "CREATE_COMPLETE", "StackId": STACK, "Parameters": []}]
    }
    control.batch.list_jobs.return_value = {"jobSummaryList": []}
    return control


def test_acquire_creates_stack_and_waits_for_endpoint_availability(monkeypatch):
    control = lifecycle(monkeypatch)
    monkeypatch.setenv("LAMBDA_TASK_ROOT", ".")
    with monkeypatch.context() as patch:
        patch.setattr(control, "stack", lambda: None)
        assert control.acquire(None) is False
    assert control.cfn.create_stack.call_args.kwargs["StackName"] == control.name
    assert "AWS::EC2::VPCEndpoint" in control.cfn.create_stack.call_args.kwargs["TemplateBody"]

    control.cfn.list_stack_resources.return_value = {
        "StackResourceSummaries": [{"PhysicalResourceId": "vpce-1", "ResourceType": "AWS::EC2::VPCEndpoint"}]
    }
    endpoint = {"State": "pending"}
    control.ec2.describe_vpc_endpoints.return_value = {"VpcEndpoints": [endpoint]}
    assert control.acquire(None) is False
    endpoint["State"] = "available"
    assert control.acquire(None) is True
    assert control.acquire(STACK) is False


def test_cleanup_waits_for_each_execution_and_in_flight_batch_jobs(monkeypatch):
    control = lifecycle(monkeypatch)
    second = EXECUTION.replace("benchmark", "dataset")
    stopped = {"status": "SUCCEEDED", "stopDate": datetime.fromtimestamp(900, timezone.utc)}
    control.states.describe_execution.side_effect = lambda executionArn: (
        stopped if executionArn == EXECUTION else {"status": "RUNNING"}
    )
    assert control.cleanup({EXECUTION: 800, second: 800}, 1021, None) == ({second: 800}, None)
    assert control.cleanup({EXECUTION: 800}, 1000, None) == ({EXECUTION: 800}, None)

    control.batch.list_jobs.side_effect = lambda **kw: {
        "jobSummaryList": [{"jobId": "active"}] if kw["jobStatus"] == "RUNNABLE" else []
    }
    assert control.cleanup({EXECUTION: 800}, 1021, None) == ({}, None)
    control.batch.list_jobs.side_effect = None
    assert control.cleanup({EXECUTION: 800}, 1021, None) == ({}, STACK)


def test_admission_retries_after_a_concurrent_state_write(monkeypatch):
    control = lifecycle(monkeypatch)
    monkeypatch.setattr(control, "read", lambda: ({"leases": {}, "locked_until": 0}, "old"))
    control.s3.put_object.side_effect = ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
    assert control.reconcile(EXECUTION) is False
    control.cfn.describe_stacks.assert_not_called()
    assert control.s3.put_object.call_args.kwargs["SSEKMSKeyId"] == "key"


def test_cleanup_stores_delete_fence_before_requesting_deletion(monkeypatch):
    control = lifecycle(monkeypatch)
    monkeypatch.setattr(control, "read", lambda: ({"leases": {}, "locked_until": 0}, "old"))
    writes = []

    def write(state, etag):
        writes.append((dict(state), etag))
        return str(len(writes))

    def delete(StackName):
        assert StackName == STACK
        assert writes[-1][0]["deleting_stack"] == STACK
        assert writes[-1][0]["locked_until"] > 0

    monkeypatch.setattr(control, "write", write)
    control.cfn.delete_stack.side_effect = delete
    assert control.reconcile() is False
    assert writes[-1] == ({"leases": {}, "locked_until": 0, "deleting_stack": STACK}, "2")


def test_cleanup_waits_for_distributed_children_after_parent_abort(monkeypatch):
    control = lifecycle(monkeypatch)
    control.states.describe_execution.return_value = {
        "status": "ABORTED",
        "stopDate": datetime.fromtimestamp(900, timezone.utc),
    }
    control.states.get_paginator.return_value.paginate.return_value = [{"mapRuns": [{"mapRunArn": "map"}]}]
    counts = {"running": 1}
    control.states.describe_map_run.return_value = {"status": "ABORTED", "executionCounts": counts}
    assert control.cleanup({EXECUTION: 800}, 1021, None) == ({EXECUTION: 800}, None)

    counts["running"] = 0
    assert control.cleanup({EXECUTION: 800}, 1021, None) == ({}, STACK)


def test_stop_idle_services_tears_down_flink_and_ephemeral_endpoints(monkeypatch):
    control = lifecycle(monkeypatch)
    control.auto_stop_idle = True
    control.project = "continual-bench"
    control.flink_application = "continual-bench-event-time-windows"

    ka = Mock()
    ka.describe_application.return_value = {"ApplicationDetail": {"ApplicationStatus": "RUNNING"}}
    sm = Mock()
    sm.get_paginator.return_value.paginate.return_value = [
        {
            "Endpoints": [
                {
                    "EndpointName": "cb-ephemeral",
                    "EndpointArn": "arn:aws:sagemaker:us-east-1:1:endpoint/cb-ephemeral",
                },
                {
                    "EndpointName": "cb-persistent",
                    "EndpointArn": "arn:aws:sagemaker:us-east-1:1:endpoint/cb-persistent",
                },
            ]
        }
    ]
    sm.list_tags.side_effect = lambda ResourceArn: (
        {"Tags": [{"Key": "cb-stack", "Value": "continual-bench"}, {"Key": "cb-ephemeral", "Value": "true"}]}
        if ResourceArn.endswith("cb-ephemeral")
        else {"Tags": [{"Key": "cb-stack", "Value": "continual-bench"}]}
    )

    monkeypatch.setattr(
        "continual_bench.orchestration.aws_endpoints.client",
        lambda name, **_: {"kinesisanalyticsv2": ka, "sagemaker": sm}.get(name, Mock()),
    )
    stopped = control.stop_idle_services()
    assert stopped == {
        "flink": ["continual-bench-event-time-windows"],
        "sagemaker_endpoints": ["cb-ephemeral"],
    }
    ka.stop_application.assert_called_once_with(
        ApplicationName="continual-bench-event-time-windows", Force=False
    )
    sm.delete_endpoint.assert_called_once_with(EndpointName="cb-ephemeral")
