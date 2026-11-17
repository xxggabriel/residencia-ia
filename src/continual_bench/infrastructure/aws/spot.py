"""One disposable EC2 Spot instance per benchmark attempt."""

import hashlib
import os
import re
import shlex

from continual_bench.infrastructure.aws.session import client, error_code

INSTANCE_TYPE = "c6a.large"
VCPUS = 2
COMPONENT = "benchmark-spot"
BACKEND = f"ec2-spot-{INSTANCE_TYPE}"
SPOT_QUOTA_CODE = "L-34B43A08"
ACTIVE_STATES = ("pending", "running", "stopping", "stopped", "shutting-down")
_CAPACITY_ERRORS = {
    "InsufficientInstanceCapacity",
    "InsufficientFreeAddressesInSubnet",
    "MaxSpotInstanceCountExceeded",
    "SpotMaxPriceTooLow",
    "Unsupported",
}
_ECR_IMAGE = re.compile(r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9._/-]+@sha256:[0-9a-f]{64}")


def _instance_record(instance):
    tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
    reason = instance.get("StateReason", {})
    launched = instance.get("LaunchTime")
    return {
        "instance_id": instance["InstanceId"],
        "state": instance.get("State", {}).get("Name", "pending"),
        "reason": reason.get("Message") or instance.get("StateTransitionReason") or "",
        "reason_code": reason.get("Code"),
        "instance_type": instance.get("InstanceType"),
        "lifecycle": instance.get("InstanceLifecycle"),
        "spot_request_id": instance.get("SpotInstanceRequestId"),
        "availability_zone": instance.get("Placement", {}).get("AvailabilityZone"),
        "subnet_id": instance.get("SubnetId"),
        "launch_at": launched.timestamp() if launched else None,
        "volume_ids": sorted(
            mapping["Ebs"]["VolumeId"]
            for mapping in instance.get("BlockDeviceMappings", [])
            if mapping.get("Ebs", {}).get("VolumeId")
        ),
        "run_id": tags.get("cb-run-id"),
        "attempt": int(tags["cb-attempt"]) if tags.get("cb-attempt", "").isdigit() else None,
    }


class SpotJobs:
    def __init__(self, ec2, env):
        self.ec2 = ec2
        self.env = env
        self.project = env["BENCH_STACK"]

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        aws_region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
        return cls(client("ec2", region_name=aws_region, max_attempts=3, read_timeout=20), env)

    def launch(self, item, campaign_id, *, attempt, provision_round):
        run_id = item["run_id"]
        image = item["runtime_image"]
        inference_image = item["inference_image"]
        if not _ECR_IMAGE.fullmatch(image) or not _ECR_IMAGE.fullmatch(inference_image):
            raise ValueError("Spot worker images must be ECR digest references")
        unit_uri = f"s3://{self.env['BENCH_BUCKET']}/campaigns/{campaign_id}/units/{run_id}.json"
        if item["unit_uri"] != unit_uri:
            raise ValueError("Spot work item points outside its published campaign unit")
        version = str(self.env["BENCH_SPOT_LAUNCH_TEMPLATE_VERSION"])
        if not version.isdecimal() or int(version) <= 0:
            raise ValueError("Spot launch template must be pinned to a positive numeric version")
        subnets = tuple(sorted(set(self.env["BENCH_SPOT_SUBNETS"].split(",")) - {""}))
        if not subnets:
            raise ValueError("no private subnets are configured for the Spot benchmark")

        log_group = self.env["BENCH_SPOT_LOG_GROUP"]
        tags = {
            "Name": f"cb-{campaign_id}-{run_id}-{attempt}",
            "cb-stack": self.project,
            "cb-component": COMPONENT,
            "cb-campaign": campaign_id,
            "cb-run-id": run_id,
            "cb-attempt": str(attempt),
        }
        if execution_key := item.get("execution_key"):
            tags["cb-execution"] = execution_key
        data = _user_data(
            {
                "BENCH_BUCKET": self.env["BENCH_BUCKET"],
                "BENCH_KMS_KEY_ID": self.env["BENCH_KMS_KEY_ID"],
                "BENCH_STACK": self.project,
                "BENCH_CAMPAIGN_ID": campaign_id,
                "BENCH_RUN_ID": run_id,
                "BENCH_EC2_ATTEMPT": attempt,
                "BENCH_EC2_LOG_GROUP": log_group,
                "BENCH_UNIT_URI": unit_uri,
                "BENCH_IMAGE_URI": image,
                "BENCH_IMAGE_DIGEST": image.rsplit("@", 1)[1],
                "BENCH_GIT_REVISION": item["git_sha"],
                "BENCH_INFERENCE_IMAGE": inference_image,
                "BENCH_MODEL_REGISTRY": self.env.get("BENCH_MODEL_REGISTRY", "false"),
                "MLFLOW_TRACKING_URI": self.env.get("MLFLOW_TRACKING_URI", ""),
                "AWS_DEFAULT_REGION": (
                    self.env.get("AWS_REGION")
                    or self.env.get("AWS_DEFAULT_REGION")
                    or self.ec2.meta.region_name
                ),
                # The campaign controller owns the normal deadline; the host also enforces a limit
                # if the workflow is stopped or loses access to the EC2 control plane.
                "BENCH_INSTANCE_LIFETIME_MINUTES": (int(item["run_timeout_seconds"]) + 59) // 60 + 25,
            }
        )
        request = {
            "MinCount": 1,
            "MaxCount": 1,
            "LaunchTemplate": {
                "LaunchTemplateId": self.env["BENCH_SPOT_LAUNCH_TEMPLATE_ID"],
                "Version": version,
            },
            "TagSpecifications": [
                {
                    "ResourceType": resource,
                    "Tags": [{"Key": key, "Value": value} for key, value in tags.items()],
                }
                for resource in ("instance", "volume")
            ],
            "UserData": data,
        }
        launch_id = f"{self.project}/{campaign_id}/{run_id}/{attempt}/{provision_round}"
        initial = item.get("placement_index")
        if initial is None:
            initial = int(hashlib.sha256(run_id.encode()).hexdigest(), 16)
        subnet = subnets[(initial + attempt + provision_round) % len(subnets)]
        # EC2 tokens are zonal. Keep the same token and subnet on ambiguous errors, and
        # rotate only when the controller durably advances a definitively rejected round.
        token = hashlib.sha256(f"{launch_id}/{subnet}".encode()).hexdigest()
        try:
            response = self.ec2.run_instances(
                **request,
                ClientToken=token,
                NetworkInterfaces=[
                    {
                        "DeviceIndex": 0,
                        "SubnetId": subnet,
                        "Groups": [self.env["BENCH_SPOT_SECURITY_GROUP_ID"]],
                        "AssociatePublicIpAddress": False,
                        "DeleteOnTermination": True,
                    }
                ],
            )
        except Exception as error:
            code = error_code(error)
            if code not in _CAPACITY_ERRORS:
                raise
            return {"instance_id": None, "status": "WAITING_FOR_SPOT", "reason": f"{subnet}: {code}"}
        (instance,) = response["Instances"]
        record = _instance_record(instance)
        return dict(
            record,
            status="LAUNCHED",
            log_group=log_group,
            log_stream=f"{campaign_id}/{run_id}/{record['instance_id']}-{attempt}",
        )

    def describe(self, instance_id):
        try:
            result = self.ec2.describe_instances(InstanceIds=[instance_id])
        except Exception as error:
            if error_code(error) != "InvalidInstanceID.NotFound":
                raise
            result = {}
        for reservation in result.get("Reservations", []):
            for instance in reservation["Instances"]:
                return _instance_record(instance)
        return {
            "instance_id": instance_id,
            "state": "missing",
            "reason": "instance not visible",
            "volume_ids": [],
        }

    def terminate(self, instance_id):
        try:
            self.ec2.terminate_instances(InstanceIds=[instance_id])
        except Exception as error:
            if error_code(error) != "InvalidInstanceID.NotFound":
                raise

    def _campaign_filters(self, campaign_id, execution_key, *, run_id=None, attempt=None):
        tags = {"cb-stack": self.project, "cb-component": COMPONENT, "cb-campaign": campaign_id}
        if execution_key:
            tags["cb-execution"] = execution_key
        if run_id:
            tags["cb-run-id"] = run_id
        if attempt is not None:
            tags["cb-attempt"] = str(attempt)
        return [{"Name": f"tag:{key}", "Values": [value]} for key, value in tags.items()]

    def list_active(self, campaign_id, *, execution_key=None):
        filters = self._campaign_filters(campaign_id, execution_key)
        filters.append({"Name": "instance-state-name", "Values": list(ACTIVE_STATES)})
        pages = self.ec2.get_paginator("describe_instances").paginate(Filters=filters)
        return [
            _instance_record(instance)
            for page in pages
            for reservation in page["Reservations"]
            for instance in reservation["Instances"]
        ]

    def list_volumes(self, campaign_id, *, execution_key=None, run_id=None, attempt=None):
        pages = self.ec2.get_paginator("describe_volumes").paginate(
            Filters=self._campaign_filters(campaign_id, execution_key, run_id=run_id, attempt=attempt),
        )
        return [
            volume["VolumeId"] for page in pages for volume in page["Volumes"] if volume["State"] != "deleted"
        ]

    def volumes_deleted(self, volume_ids):
        if not volume_ids:
            return True
        pages = self.ec2.get_paginator("describe_volumes").paginate(
            Filters=[{"Name": "volume-id", "Values": sorted(set(volume_ids))}]
        )
        return not any(volume["State"] != "deleted" for page in pages for volume in page["Volumes"])


def _user_data(values):
    settings = "".join(f"export {name}={shlex.quote(str(value))}\n" for name, value in values.items())
    script = f"#!/bin/bash\nset -Eeuo pipefail\n{settings}{_BOOTSTRAP}"
    if len(script.encode()) > 16384:
        raise ValueError("Spot cloud-init user data exceeds the EC2 limit")
    # Botocore base64-encodes RunInstances UserData during request serialization.
    return script


# This runs on the AL2023 ECS-optimized AMI, whose packages already provide Docker,
# AWS CLI and Python. The private worker has no package repository or NAT access.
_BOOTSTRAP = r"""
export AWS_PAGER="" AWS_RETRY_MODE=standard AWS_MAX_ATTEMPTS=3
state_dir=/var/lib/continual-bench
notice_dir="$state_dir/notice"
mkdir -pm 755 "$notice_dir"
host_log="$state_dir/host.log"
exec > >(tee -a "$host_log") 2>&1
metadata=http://169.254.169.254/latest
container=continual-bench
stage=metadata
host_state=BOOTSTRAP_FAILED
instance_id=""
watcher=""
container_created=false
notice_file="$notice_dir/spot-notice.json"

imds_token() {
    curl -fsS --max-time 3 -X PUT "$metadata/api/token" \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 120'
}

append_host_log() {
    local message=$1
    local event
    event=$(python3 -c '
import json, sys, time
print(json.dumps([{"timestamp": int(time.time() * 1000), "message": sys.argv[1]}]))
' "$message")
    aws logs put-log-events --cli-connect-timeout 3 --cli-read-timeout 5 \
        --log-group-name "$BENCH_EC2_LOG_GROUP" --log-stream-name "$BENCH_EC2_LOG_STREAM/host" \
        --log-events "$event"
}

put_marker() {
    aws s3api put-object --cli-connect-timeout 3 --cli-read-timeout 5 \
        --bucket "$BENCH_BUCKET" --key "$output_key/$1" --body "$2" \
        --server-side-encryption aws:kms --ssekms-key-id "$BENCH_KMS_KEY_ID" \
        --content-type "$3" --if-none-match '*' >/dev/null
}

finish() {
    local rc=$?
    trap - EXIT
    set +e
    if [[ -n "$watcher" ]]; then
        kill "$watcher" 2>/dev/null
        wait "$watcher" 2>/dev/null
    fi
    if $container_created; then
        docker rm --force "$container"
    fi
    local reason="bootstrap failed during $stage"
    if [[ "$host_state" == WORKER_EXIT ]]; then
        reason="benchmark container exited with code $rc"
    fi
    if [[ -f "$notice_file" ]]; then
        host_state=SPOT_NOTICE
        reason="EC2 Spot interruption notice at $stage"
        put_marker spot-notice.json "$notice_file" application/json
    fi
    echo "$reason"
    if [[ -n "$instance_id" ]]; then
        append_host_log "$reason"
        python3 -c '
import json, sys, time
instance, attempt, state, code, reason = sys.argv[1:]
print(json.dumps({
    "instance_id": instance, "attempt": int(attempt), "state": state,
    "exit_code": int(code), "reason": reason, "finished_at": time.time(),
}))
' \
            "$instance_id" "$BENCH_EC2_ATTEMPT" "$host_state" "$rc" "$reason" > "$state_dir/host-exit.json"
        put_marker host.log "$host_log" text/plain
        put_marker host-exit.json "$state_dir/host-exit.json" application/json
    fi
    shutdown -P now
    exit "$rc"
}
trap finish EXIT

shutdown -P "+$BENCH_INSTANCE_LIFETIME_MINUTES"
token=$(imds_token)
instance_id=$(curl -fsS --max-time 3 -H "X-aws-ec2-metadata-token: $token" "$metadata/meta-data/instance-id")
export BENCH_EC2_INSTANCE_ID="$instance_id"
export BENCH_EC2_LOG_STREAM="$BENCH_CAMPAIGN_ID/$BENCH_RUN_ID/$instance_id-$BENCH_EC2_ATTEMPT"
output_key="campaigns/$BENCH_CAMPAIGN_ID/runs/$BENCH_RUN_ID/attempts/$instance_id-$BENCH_EC2_ATTEMPT"
aws logs create-log-stream --cli-connect-timeout 3 --cli-read-timeout 5 \
    --log-group-name "$BENCH_EC2_LOG_GROUP" --log-stream-name "$BENCH_EC2_LOG_STREAM/host"
append_host_log "Bootstrapping $BENCH_IMAGE_URI on $instance_id"

# ECS is baked into the AMI for Docker; this host runs one container directly.
# --no-block avoids waiting for a service that is ordered after cloud-final.
stage=docker
systemctl mask --now --no-block ecs.service
systemctl start docker.service

watch_spot_notice() {
    local token action
    while true; do
        if token=$(imds_token) && action=$(curl -fsS --max-time 3 \
            -H "X-aws-ec2-metadata-token: $token" \
            "$metadata/meta-data/spot/instance-action" 2>/dev/null); then
            printf '%s\n' "$action" > "$notice_file.tmp"
            mv "$notice_file.tmp" "$notice_file"
            # The mounted file records the cause before the worker handles SIGTERM.
            docker kill -s TERM "$container" >/dev/null 2>&1 || true
            put_marker spot-notice.json "$notice_file" application/json || true
            append_host_log "EC2 Spot interruption notice: $action" || true
            return
        fi
        sleep 5
    done
}
watch_spot_notice &
watcher=$!

stage=pull
registry=${BENCH_IMAGE_URI%%/*}
aws ecr get-login-password --cli-connect-timeout 3 --cli-read-timeout 10 \
    | docker login --username AWS --password-stdin "$registry"
docker pull "$BENCH_IMAGE_URI"
docker logout "$registry"
append_host_log "Pulled pinned benchmark image"
[[ ! -f "$notice_file" ]]

stage=start
docker create --name "$container" --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges \
    --cpus 2 --log-driver awslogs --log-opt "awslogs-group=$BENCH_EC2_LOG_GROUP" \
    --log-opt "awslogs-region=$AWS_DEFAULT_REGION" --log-opt "awslogs-stream=$BENCH_EC2_LOG_STREAM" \
    --mount "type=bind,src=$notice_dir,dst=/run/continual-bench,readonly" \
    -e BENCH_BUCKET -e BENCH_KMS_KEY_ID -e BENCH_STACK -e BENCH_EC2_ATTEMPT -e BENCH_EC2_INSTANCE_ID \
    -e BENCH_EC2_LOG_GROUP -e BENCH_EC2_LOG_STREAM -e BENCH_IMAGE_URI -e BENCH_IMAGE_DIGEST \
    -e BENCH_GIT_REVISION -e BENCH_INFERENCE_IMAGE -e BENCH_MODEL_REGISTRY -e MLFLOW_TRACKING_URI \
    -e AWS_DEFAULT_REGION -e BENCH_PROJECT_ROOT=/app -e BENCH_BENCHMARK_VCPUS=2 \
    -e PYTHONHASHSEED=0 -e BENCH_EXECUTION_ENV=aws_ec2_spot \
    -e 'BENCH_SPOT_NOTICE_FILE=/run/continual-bench/spot-notice.json' \
    -e "AWS_BATCH_JOB_ID=$instance_id" -e "AWS_BATCH_JOB_ATTEMPT=$BENCH_EC2_ATTEMPT" \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e NUMEXPR_NUM_THREADS=2 \
    -e OMP_DYNAMIC=FALSE -e POLARS_MAX_THREADS=2 -e MALLOC_ARENA_MAX=2 -e MPLCONFIGDIR=/tmp/matplotlib \
    "$BENCH_IMAGE_URI" continual-bench aws run-unit --unit-uri "$BENCH_UNIT_URI" --workdir /tmp/cb/run
container_created=true
docker start "$container"
if [[ -f "$notice_file" ]]; then
    docker kill -s TERM "$container" >/dev/null 2>&1 || true
fi
stage=benchmark
host_state=WORKER_EXIT
code=$(docker wait "$container")
exit "$code"
"""
