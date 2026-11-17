"""EC2 Spot request identity, AZ placement and cleanup queries."""

import subprocess
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from continual_bench.infrastructure.aws.spot import SpotJobs

_CAMPAIGN = "spot-acceptance"
_ENV = {
    "BENCH_STACK": "continual-bench",
    "BENCH_BUCKET": "bench-test",
    "BENCH_KMS_KEY_ID": "arn:aws:kms:us-east-1:123456789012:key/test",
    "BENCH_SPOT_LAUNCH_TEMPLATE_ID": "lt-abc123",
    "BENCH_SPOT_LAUNCH_TEMPLATE_VERSION": "3",
    "BENCH_SPOT_LOG_GROUP": "/continual-bench/spot",
    "BENCH_SPOT_SUBNETS": "subnet-2,subnet-1",
    "BENCH_SPOT_SECURITY_GROUP_ID": "sg-abc123",
    "AWS_REGION": "us-east-1",
}
_ITEM = {
    "run_id": "run-1",
    "unit_uri": f"s3://bench-test/campaigns/{_CAMPAIGN}/units/run-1.json",
    "runtime_image": f"123456789012.dkr.ecr.us-east-1.amazonaws.com/bench@sha256:{'a' * 64}",
    "inference_image": f"123456789012.dkr.ecr.us-east-1.amazonaws.com/bench-ml@sha256:{'b' * 64}",
    "git_sha": "f" * 40,
    "run_timeout_seconds": 900,
    "execution_key": "e" * 64,
    "placement_index": 0,
}


def _client_error(code, *, operation="RunInstances"):
    return ClientError({"Error": {"Code": code, "Message": "test error"}}, operation)


def test_launch_tokens_replay_ambiguous_requests_and_rotate_definitive_capacity_rejections():
    ec2 = Mock()
    accepted = {
        "Instances": [
            {
                "InstanceId": "i-12345",
                "InstanceType": "c6a.large",
                "InstanceLifecycle": "spot",
                "LaunchTime": datetime(2026, 11, 17, tzinfo=UTC),
                "Placement": {"AvailabilityZone": "us-east-1b"},
                "SubnetId": "subnet-2",
                "BlockDeviceMappings": [{"Ebs": {"VolumeId": "vol-12345", "DeleteOnTermination": True}}],
            }
        ]
    }
    ec2.run_instances.side_effect = [
        ReadTimeoutError(endpoint_url="https://ec2.us-east-1.amazonaws.com"),
        _client_error("InsufficientInstanceCapacity"),
        accepted,
    ]
    jobs = SpotJobs(ec2, _ENV)
    with pytest.raises(ReadTimeoutError):
        jobs.launch(_ITEM, _CAMPAIGN, attempt=1, provision_round=1)
    waiting = jobs.launch(_ITEM, _CAMPAIGN, attempt=1, provision_round=1)
    assert waiting == {
        "instance_id": None,
        "status": "WAITING_FOR_SPOT",
        "reason": "subnet-1: InsufficientInstanceCapacity",
    }
    launched = jobs.launch(_ITEM, _CAMPAIGN, attempt=1, provision_round=2)
    assert launched["instance_type"] == "c6a.large"
    assert launched["lifecycle"] == "spot"
    assert launched["volume_ids"] == ["vol-12345"]
    assert launched["log_stream"] == f"{_CAMPAIGN}/run-1/i-12345-1"

    first, replay, rotated = [call.kwargs for call in ec2.run_instances.call_args_list]
    assert first == replay
    assert rotated["ClientToken"] != first["ClientToken"]
    assert len(first["ClientToken"]) == 64
    assert first["MinCount"] == first["MaxCount"] == 1
    assert first["LaunchTemplate"] == {"LaunchTemplateId": "lt-abc123", "Version": "3"}
    assert rotated["NetworkInterfaces"] == [
        {
            "DeviceIndex": 0,
            "SubnetId": "subnet-2",
            "Groups": ["sg-abc123"],
            "AssociatePublicIpAddress": False,
            "DeleteOnTermination": True,
        }
    ]
    for tags in first["TagSpecifications"]:
        tag_values = {item["Key"]: item["Value"] for item in tags["Tags"]}
        assert tag_values["cb-run-id"] == "run-1"
        assert tag_values["cb-execution"] == _ITEM["execution_key"]
    user_data = first["UserData"]
    assert _ITEM["runtime_image"] in user_data
    assert "awslogs-stream=$BENCH_EC2_LOG_STREAM" in user_data
    assert "shutdown -P now" in user_data
    assert len(user_data.encode()) <= 16 * 1024
    subprocess.run(["bash", "-n"], input=user_data, text=True, check=True)


def test_provisioning_rejects_unpinned_releases_and_surfaces_permission_failures():
    ec2 = Mock()
    jobs = SpotJobs(ec2, _ENV)
    with pytest.raises(ValueError, match="digest references"):
        jobs.launch({**_ITEM, "runtime_image": "bench:latest"}, _CAMPAIGN, attempt=1, provision_round=1)
    ec2.run_instances.assert_not_called()
    denied = _client_error("UnauthorizedOperation")
    ec2.run_instances.side_effect = denied
    with pytest.raises(ClientError) as caught:
        jobs.launch(_ITEM, _CAMPAIGN, attempt=1, provision_round=1)
    assert caught.value is denied
    ec2.run_instances.assert_called_once()


def test_list_volumes_uses_execution_run_and_attempt_tags_to_verify_cleanup():
    ec2 = Mock()
    paginator = ec2.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Volumes": [{"VolumeId": "vol-111", "State": "deleting"}]},
        {"Volumes": [{"VolumeId": "vol-222", "State": "deleted"}]},
    ]
    jobs = SpotJobs(ec2, _ENV)
    assert jobs.list_volumes(_CAMPAIGN, execution_key="execution", run_id="run-1", attempt=1) == ["vol-111"]
    assert {f["Name"]: f["Values"] for f in paginator.paginate.call_args.kwargs["Filters"]} == {
        "tag:cb-stack": ["continual-bench"],
        "tag:cb-component": ["benchmark-spot"],
        "tag:cb-campaign": [_CAMPAIGN],
        "tag:cb-execution": ["execution"],
        "tag:cb-run-id": ["run-1"],
        "tag:cb-attempt": ["1"],
    }
    assert not jobs.volumes_deleted(["vol-111"])
    paginator.paginate.return_value = [{"Volumes": []}]
    assert jobs.volumes_deleted(["vol-111"])


def test_stale_instance_cleanup_handles_ec2_not_found():
    ec2 = Mock()
    ec2.describe_instances.side_effect = _client_error(
        "InvalidInstanceID.NotFound", operation="DescribeInstances"
    )
    ec2.terminate_instances.side_effect = _client_error(
        "InvalidInstanceID.NotFound", operation="TerminateInstances"
    )
    jobs = SpotJobs(ec2, _ENV)
    assert jobs.describe("i-123")["state"] == "missing"
    jobs.terminate("i-123")
    ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-123"])
