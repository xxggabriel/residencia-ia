"""Capacity evidence for campaign admission: regional Fargate quota, compute environment and queue."""

from .session import client as make_client

# Service Quotas code for "Fargate On-Demand vCPU resource count" (us-east-1, account level).
FARGATE_ON_DEMAND_VCPU = ("fargate", "L-3032A538")
FARGATE_SPOT_VCPU = ("fargate", "L-36FBB829")


def fargate_vcpu_quota(*, spot=False, client=None) -> float:
    client = client or make_client("service-quotas")
    service, code = FARGATE_SPOT_VCPU if spot else FARGATE_ON_DEMAND_VCPU
    try:
        return float(client.get_service_quota(ServiceCode=service, QuotaCode=code)["Quota"]["Value"])
    except client.exceptions.NoSuchResourceException:
        return float(
            client.get_aws_default_service_quota(ServiceCode=service, QuotaCode=code)["Quota"]["Value"]
        )


def queue_capacity(queue: str, *, client=None) -> dict:
    """Max vCPUs across enabled, valid compute environments attached to an enabled queue."""
    client = client or make_client("batch")
    queues = client.describe_job_queues(jobQueues=[queue])["jobQueues"]
    if not queues:
        raise ValueError(f"job queue not found: {queue}")
    detail = queues[0]
    environments = [entry["computeEnvironment"] for entry in detail["computeEnvironmentOrder"]]
    described = client.describe_compute_environments(computeEnvironments=environments)["computeEnvironments"]
    usable = [env for env in described if env.get("state") == "ENABLED" and env.get("status") == "VALID"]
    return {
        "queue_state": detail.get("state"),
        "queue_status": detail.get("status"),
        "max_vcpus": sum(env.get("computeResources", {}).get("maxvCpus", 0) for env in usable),
        "environments": [
            {
                "name": env["computeEnvironmentName"],
                "state": env.get("state"),
                "status": env.get("status"),
                "type": env.get("computeResources", {}).get("type"),
                "max_vcpus": env.get("computeResources", {}).get("maxvCpus"),
            }
            for env in described
        ],
    }
