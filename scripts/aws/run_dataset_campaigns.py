"""Start and inspect the full ordered-stream suite through authenticated AWS CLI calls.

This command never deploys infrastructure. ``start-base`` requires completed smoke and
deployment receipts; ``advance`` submits the remaining cost policies only after the
three CARA matrix-metric artifacts of a dataset have been verified. Online is excluded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

DATASET_ROWS = {
    "electricity": 45_312,
    "airlines": 539_383,
    "insects_abrupt_imbalanced": 355_275,
    "insects_incremental_abrupt_reoccurring_imbalanced": 452_044,
    "fraud": 50_000_000,
    "ecommerce": 50_000_000,
    "social": 100_000_000,
}
SEEDS = (42, 43, 44)
BASE_POLICIES = ("frozen", "drift", "drift_adwin", "drift_ddm", "incremental", "periodic", "cara")
CACHED_POLICIES = ("cara_cumulative", "cara_periodic", "upf", "oracle")
POLICIES = BASE_POLICIES + CACHED_POLICIES
ACTIVE = {"SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"}
PROTOCOL = "ordered-stream-batch-prequential-metrics-v2"


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def state_lock(directory):
    """One local coordinator at a time; final publication remains conditional in S3."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".coordinator.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class AwsCli:
    def __init__(self, region, *, runner=subprocess.run):
        self.region, self.runner = region, runner

    def call(self, service, operation, params=None):
        command = ["aws", service, operation, "--region", self.region, "--output", "json", "--no-paginate"]
        if params:
            command += ["--cli-input-json", canonical(params)]
        result = self.runner(command, capture_output=True, text=True, timeout=180, check=False)
        if result.returncode:
            raise RuntimeError(f"AWS CLI {service}/{operation} failed: {result.stderr[-1000:]}")
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def optional_json(self, bucket, key):
        with tempfile.TemporaryDirectory(prefix="dataset-suite-read-") as directory:
            path = Path(directory) / "object.json"
            command = [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--region",
                self.region,
                "--output",
                "json",
                str(path),
            ]
            result = self.runner(command, capture_output=True, text=True, timeout=180, check=False)
            if result.returncode:
                if "NoSuchKey" in result.stderr or "(404)" in result.stderr:
                    return None
                raise RuntimeError(f"S3 read {key} failed: {result.stderr[-1000:]}")
            return read_json(path)

    def json_object(self, bucket, key):
        value = self.optional_json(bucket, key)
        if value is None:
            raise ValueError(f"required S3 evidence is absent: {key}")
        return value

    def verified_json(self, bucket, key, sha256):
        with tempfile.TemporaryDirectory(prefix="dataset-suite-report-") as directory:
            path = Path(directory) / "report.json"
            self.download(bucket, key, path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
                raise ValueError(f"S3 evidence checksum mismatch: {key}")
            return read_json(path)

    def download(self, bucket, key, destination):
        command = [
            "aws",
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--region",
            self.region,
            "--output",
            "json",
            str(destination),
        ]
        result = self.runner(command, capture_output=True, text=True, timeout=180, check=False)
        if result.returncode:
            raise RuntimeError(f"S3 read {key} failed: {result.stderr[-1000:]}")


def object_key(uri, bucket):
    prefix = f"s3://{bucket}/"
    if not uri.startswith(prefix):
        raise ValueError("suite evidence must remain in the configured S3 bucket")
    key = uri[len(prefix) :]
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("invalid evidence object key")
    return key


def all_items(
    client, service, operation, params, field, *, input_token="nextToken", output_token="nextToken"
):
    items, tokens = [], set()
    params = dict(params)
    while True:
        page = client.call(service, operation, params)
        items.extend(page.get(field, []))
        token = page.get(output_token)
        if not token:
            return items
        if token in tokens:
            raise ValueError("AWS pagination repeated a continuation token")
        tokens.add(token)
        params[input_token] = token


def validate_smoke(smoke, deployment, release):
    coverage, checks = smoke.get("coverage", {}), smoke.get("checks", {})
    required = (
        "prediction_integrity",
        "metrics_reconstructed",
        "no_model_files",
        "tracking_receipts_complete",
        "native_tracking_catalog_verified",
    )
    if not (
        smoke.get("schema") == "ordered-dataset-suite-qa-v1"
        and smoke.get("complete") is True
        and smoke.get("scope") == "smoke"
        and coverage.get("expected_cases") == coverage.get("passed_cases") == 231
        and set(coverage.get("datasets", [])) == set(DATASET_ROWS)
        and set(coverage.get("policies", [])) == set(POLICIES)
        and coverage.get("seeds") == list(SEEDS)
        and all(checks.get(key) is True for key in required)
    ):
        raise ValueError("all 231 smoke cases and native metric checks must be complete")
    workflow = deployment.get("workflow", {})
    if not (
        re.fullmatch(r"[a-f0-9]{40}", release)
        and deployment.get("complete") is True
        and deployment.get("git_sha") == release
        and workflow.get("headSha") == release
        and workflow.get("status") == "completed"
        and workflow.get("conclusion") == "success"
        and type(workflow.get("databaseId")) is int
        and re.fullmatch(r"[a-f0-9]{64}", smoke.get("code", {}).get("code_sha256", ""))
        and smoke.get("code", {}).get("code_sha256") == deployment.get("smoke_code_sha256")
    ):
        raise ValueError("deployment receipt must bind the successful release to the smoke source hash")


def read_requests(directory, release):
    directory = Path(directory).resolve()
    index = read_json(directory / "index.json")
    if index.get("release") != release or index.get("stage") != "base":
        raise ValueError("start-base requires base requests pinned to the deployed release")
    requests = {}
    for entry in index["requests"]:
        path = (directory / entry["file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("request file escapes its directory")
        request = read_json(path)["request"]
        dataset = entry["dataset"]
        matrix = request.get("matrix", {})
        if not (
            dataset in DATASET_ROWS
            and dataset not in requests
            and request.get("dataset", request.get("snapshot", {}).get("dataset")) == dataset
            and request.get("release") == release
            and request.get("scope", request.get("snapshot", {}).get("scope")) == "full"
            and matrix.get("artifact_mode") == "metrics_only"
            and matrix.get("seeds") == list(SEEDS)
            and matrix.get("iterative_training", {}).get("max_epochs") == 100
            and set(matrix.get("policies", [])) == set(BASE_POLICIES)
            and len(matrix.get("policies", [])) == len(BASE_POLICIES)
            and request.get("max_parallel") == 32
            and request.get("queue_timeout_seconds") == 86_400
            and request["campaign_id"] == entry["campaign_id"]
            and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", request["campaign_id"])
            and not matrix.get("metric_cache_sources")
        ):
            raise ValueError(f"invalid full base request for {dataset}")
        requests[dataset] = request
    if set(requests) != set(DATASET_ROWS) or len({r["campaign_id"] for r in requests.values()}) != 7:
        raise ValueError("exactly seven distinct complete stream requests are required")
    return requests


def verify_workflow(deployment, *, runner=subprocess.run):
    expected = deployment["workflow"]
    result = runner(
        [
            "gh",
            "run",
            "view",
            str(expected["databaseId"]),
            "--repo",
            "xxggabriel/residencia-ia",
            "--json",
            "databaseId,headSha,status,conclusion,url,workflowName",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    actual = json.loads(result.stdout)
    if any(actual.get(k) != expected[k] for k in ("databaseId", "headSha", "status", "conclusion")):
        raise ValueError("live GitHub workflow does not match the successful deployment receipt")
    if actual.get("workflowName") != "Build and deploy":
        raise ValueError("the receipt must identify the completed deployment workflow")
    return actual


def preflight(client, bucket, project, release):
    stacks = client.call("cloudformation", "describe-stacks", {"StackName": f"{project}-runtime"})["Stacks"]
    stack = stacks[0]
    outputs = {r["OutputKey"]: r["OutputValue"] for r in stack["Outputs"]}
    if stack["StackStatus"] not in ("CREATE_COMPLETE", "UPDATE_COMPLETE") or outputs.get("GitSha") != release:
        raise ValueError("runtime stack is not complete at the selected release")
    published = client.json_object(bucket, f"releases/{release}/release.json")
    if published.get("git_sha") != release or published.get("benchmark_compute") != "batch-fargate-spot":
        raise ValueError("release is not the requested immutable Fargate Spot build")
    definition = published["job_definitions"]["benchmark"]
    if definition != outputs["BenchmarkJobDefinitionArn"]:
        raise ValueError("release and runtime job definitions differ")
    job = client.call("batch", "describe-job-definitions", {"jobDefinitions": [definition]})[
        "jobDefinitions"
    ][0]
    image = published["images"]["runtime"]
    if job["containerProperties"]["image"] != image["repository"] + "@" + image["digest"]:
        raise ValueError("job definition image differs from the immutable release digest")
    resources = {r["type"]: r["value"] for r in job["containerProperties"]["resourceRequirements"]}
    if (
        job.get("status") != "ACTIVE"
        or job.get("platformCapabilities") != ["FARGATE"]
        or resources
        != {
            "VCPU": "2",
            "MEMORY": "4096",
        }
    ):
        raise ValueError("benchmark job definition must request exactly 2 vCPU and 4096 MiB on Fargate")
    queue = client.call("batch", "describe-job-queues", {"jobQueues": [outputs["JobQueueArn"]]})["jobQueues"][
        0
    ]
    environments = client.call(
        "batch",
        "describe-compute-environments",
        {
            "computeEnvironments": [e["computeEnvironment"] for e in queue["computeEnvironmentOrder"]],
        },
    )["computeEnvironments"]
    if queue.get("state") != "ENABLED" or queue.get("status") != "VALID" or not environments:
        raise ValueError("benchmark queue is not usable")
    if (
        any(
            e.get("state") != "ENABLED"
            or e.get("status") != "VALID"
            or e["computeResources"].get("type") != "FARGATE_SPOT"
            or e["computeResources"].get("maxvCpus") != 64
            for e in environments
        )
        or len(environments) != 1
    ):
        raise ValueError("suite must use one valid Spot compute environment with a global 64-vCPU ceiling")
    quotas = all_items(
        client,
        "service-quotas",
        "list-service-quotas",
        {"ServiceCode": "fargate"},
        "Quotas",
        input_token="NextToken",
        output_token="NextToken",
    )
    quota = next(q for q in quotas if q["QuotaCode"] == "L-36FBB829")
    if quota["Value"] < 64:
        raise ValueError("regional Fargate Spot quota is below the requested 64 vCPU")
    regional_tasks = []
    clusters = all_items(client, "ecs", "list-clusters", {}, "clusterArns")
    own_clusters = {e["ecsClusterArn"] for e in environments}
    for cluster in clusters:
        tasks = all_items(
            client, "ecs", "list-tasks", {"cluster": cluster, "desiredStatus": "RUNNING"}, "taskArns"
        )
        for offset in range(0, len(tasks), 100):
            described = client.call(
                "ecs", "describe-tasks", {"cluster": cluster, "tasks": tasks[offset : offset + 100]}
            )
            if described.get("failures"):
                raise ValueError("regional Fargate usage could not be fully inspected")
            for task in described["tasks"]:
                if task.get("launchType") == "FARGATE" and task.get("lastStatus") != "STOPPED":
                    regional_tasks.append(
                        {
                            "task_arn": task["taskArn"],
                            "cluster": cluster,
                            "vcpus": int(task["cpu"]) / 1024,
                            "capacity_provider": task.get("capacityProviderName"),
                            "suite_compute_environment": cluster in own_clusters,
                        }
                    )
    external_vcpus = sum(t["vcpus"] for t in regional_tasks if not t["suite_compute_environment"])
    if external_vcpus > 0:
        raise ValueError(
            "external regional Fargate workloads are active; reconcile their resource budget first"
        )
    return {
        "stack": stack["StackId"],
        "release": published,
        "job_definition": definition,
        "resources": resources,
        "queue": outputs["JobQueueArn"],
        "quota": quota,
        "compute_environments": environments,
        "regional_fargate_tasks": regional_tasks,
        "external_fargate_vcpus": external_vcpus,
        "state_machine": outputs["BenchmarkStateMachineArn"],
    }


def execution_arn(state_machine, name):
    if ":stateMachine:" not in state_machine:
        raise ValueError("invalid state machine ARN")
    return state_machine.replace(":stateMachine:", ":execution:") + ":" + name


def start_request(client, request, state_machine):
    """Re-read on uncertain submissions; never invent a second execution name."""
    name = request["campaign_id"]
    expected_input = {"request": request}
    arn = execution_arn(state_machine, name)
    try:
        actual = client.call("stepfunctions", "describe-execution", {"executionArn": arn})
    except RuntimeError as error:
        if "ExecutionDoesNotExist" not in str(error):
            raise
    else:
        if json.loads(actual["input"]) != expected_input:
            raise ValueError(f"existing execution {name} has a different request")
        return {"executionArn": arn, "reused": True, "status": actual["status"]}
    try:
        result = client.call(
            "stepfunctions",
            "start-execution",
            {
                "stateMachineArn": state_machine,
                "name": name,
                "input": canonical(expected_input),
            },
        )
    except RuntimeError:
        # A transport error may have occurred after AWS accepted the request.
        actual = client.call("stepfunctions", "describe-execution", {"executionArn": arn})
        if json.loads(actual["input"]) != expected_input:
            raise ValueError(f"uncertain execution {name} does not match the submitted request") from None
        return {"executionArn": arn, "reused": True, "status": actual["status"]}
    if result["executionArn"] != arn:
        raise ValueError("AWS returned an unexpected execution identity")
    return {"executionArn": arn, "reused": False, "status": "RUNNING"}


def verify_final(client, bucket, unit, final, dataset):
    params = unit["params"]
    source_sha = (unit["contract"].get("stream_source") or unit["contract"].get("snapshot") or {}).get(
        "prepared_sha256"
    )
    if not (
        final.get("contract_sha256") == unit["contract_sha256"]
        and final.get("artifact_mode") == "metrics_only"
        and final.get("model") is None
        and final.get("tracking", {}).get("complete") is True
        and final.get("tracking", {}).get("required") is True
        and final.get("labels", {}).get("seed") == str(params["seed"])
        and not (final.get("attempt") or {}).get("checkpoint")
    ):
        raise ValueError(f"invalid metric-only final manifest for {unit['run_id']}")
    report = client.verified_json(bucket, object_key(final["report_uri"], bucket), final["report_sha256"])
    metrics, contract = report["metrics"], report["contract"]
    if not (
        report.get("dataset") == dataset
        and report.get("scope") == "full"
        and report.get("policy") == params["policy"]
        and contract["input"]["rows"] == metrics["dataset_rows"] == DATASET_ROWS[dataset]
        and source_sha is not None
        and contract["input"]["sha256"] == source_sha
        and metrics["initial_train_rows"] + metrics["n"] + report["holdout"]["n"] == DATASET_ROWS[dataset]
        and contract["iterative_training"]["seed"] == params["seed"]
        and contract.get("artifact_mode") == "metrics_only"
        and metrics["model_storage_bytes"] == 0
        and all(v.get("file") is None for v in report["model_versions"])
        and all((f.get("checkpoint") or {}).get("file") is None for f in report["training_history"])
    ):
        raise ValueError(f"coverage, randomness or no-weight contract failed for {unit['run_id']}")
    files = all_items(
        client,
        "s3api",
        "list-objects-v2",
        {
            "Bucket": bucket,
            "Prefix": object_key(final["outputs_uri"], bucket) + "/",
        },
        "Contents",
        input_token="ContinuationToken",
        output_token="NextContinuationToken",
    )
    forbidden = re.compile(
        r"(?:^|/)(?:models|checkpoints|\.zia-resume)(?:/|$)"
        r"|\.(?:pkl|pickle|ubj|npy|npz|pt|pth|ckpt|bin)$|/(?:model|checkpoint)\.tar\.gz$"
    )
    if any(forbidden.search(row["Key"]) for row in files):
        raise ValueError("published metric-only result contains a model/checkpoint artifact")
    return {
        "run_id": unit["run_id"],
        "dataset": dataset,
        "seed": params["seed"],
        "policy": params["policy"],
        "rows": metrics["dataset_rows"],
        "report_sha256": final["report_sha256"],
        "outputs_uri": final["outputs_uri"],
        "tracking": final["tracking"],
        "artifact_objects": len(files),
    }


def matrix_cache(client, bucket, final):
    uri = final["outputs_uri"] + "/matrix-metrics.json.gz"
    key = object_key(uri, bucket)
    head = client.call("s3api", "head-object", {"Bucket": bucket, "Key": key})
    sha256 = head.get("Metadata", {}).get("sha256")
    if not sha256 or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("matrix cache lacks a content hash")
    with tempfile.TemporaryDirectory(prefix="dataset-suite-cache-") as directory:
        path = Path(directory) / "matrix-metrics.json.gz"
        client.download(bucket, key, path)
        if (
            hashlib.sha256(path.read_bytes()).hexdigest() != sha256
            or path.stat().st_size != head["ContentLength"]
        ):
            raise ValueError("matrix metric cache failed its byte/hash verification")
    return {"uri": uri, "sha256": sha256}


def inspect_campaign(client, bucket, record, dataset, verified):
    execution = client.call("stepfunctions", "describe-execution", {"executionArn": record["executionArn"]})
    if json.loads(execution["input"]) != {"request": record["request"]}:
        raise ValueError("execution input differs from the recorded campaign")
    campaign = record["request"]["campaign_id"]
    plan = client.optional_json(bucket, f"campaigns/{campaign}/plan.json")
    if plan is None and execution["status"] == "SUCCEEDED":
        raise ValueError("Step Functions finished without a published campaign plan")
    policies = tuple(record["request"]["matrix"]["policies"])
    caches, complete, count = {}, False, 0
    if plan is not None:
        expected = {(policy, seed) for policy in policies for seed in SEEDS}
        actual = {(u["params"].get("policy"), u["params"].get("seed")) for u in plan["units"]}
        if expected != actual or len(plan["units"]) != len(expected):
            raise ValueError("the published plan does not contain the exact policy/seed matrix")
        for unit in plan["units"]:
            if unit["contract"].get("protocol") != PROTOCOL:
                raise ValueError("published unit uses a different evaluation protocol")
            final = client.optional_json(bucket, f"campaigns/{campaign}/runs/{unit['run_id']}/final.json")
            if final is None:
                continue
            previous = verified.get(unit["run_id"])
            final_hash = digest(final)
            if previous is None:
                previous = verify_final(client, bucket, unit, final, dataset)
                previous["final_sha256"] = final_hash
                verified[unit["run_id"]] = previous
            elif previous["final_sha256"] != final_hash:
                raise ValueError("a previously verified final manifest changed")
            count += 1
            if unit["params"]["policy"] == "cara":
                if "matrix_cache" not in previous:
                    previous["matrix_cache"] = matrix_cache(client, bucket, final)
                caches[str(unit["params"]["seed"])] = previous["matrix_cache"]
        summary = client.optional_json(bucket, f"campaigns/{campaign}/summary.json")
        complete = (
            execution["status"] == "SUCCEEDED"
            and count == len(expected)
            and summary is not None
            and summary.get("complete") is True
            and summary.get("total") == summary.get("succeeded") == len(expected)
            and summary.get("tracking", {}).get("complete") is True
        )
        if execution["status"] == "SUCCEEDED" and not complete:
            raise ValueError("Step Functions finished without verified complete metric evidence")
    if execution["status"] not in ("RUNNING", "SUCCEEDED"):
        raise ValueError(
            f"campaign {campaign} ended as {execution['status']}: {execution.get('cause', '')[:400]}"
        )
    return {
        "execution_status": execution["status"],
        "verified_units": count,
        "expected_units": len(policies) * len(SEEDS),
        "complete": complete,
        "matrix_caches": caches,
    }


def cached_request(base, caches):
    if set(caches) != {str(seed) for seed in SEEDS}:
        raise ValueError("all three independently seeded CARA matrix caches are required")
    request = copy.deepcopy(base)
    suffix = digest({"base": base["campaign_id"], "stage": "cached"})[:6]
    request["campaign_id"] = base["campaign_id"][:30].rstrip("-") + "-c-" + suffix
    request["matrix"]["policies"] = list(CACHED_POLICIES)
    request["matrix"]["metric_cache_sources"] = caches
    return request


def batch_snapshot(client, queue, campaign_ids):
    summaries = []
    for campaign in campaign_ids:
        found = all_items(
            client,
            "batch",
            "list-jobs",
            {
                "jobQueue": queue,
                "filters": [{"name": "JOB_NAME", "values": [f"cb-{campaign}-*"]}],
            },
            "jobSummaryList",
        )
        summaries.extend(
            r for r in found if re.fullmatch(rf"cb-{re.escape(campaign)}-[a-f0-9]{{20}}", r["jobName"])
        )
    details = []
    for offset in range(0, len(summaries), 100):
        details += client.call(
            "batch", "describe-jobs", {"jobs": [r["jobId"] for r in summaries[offset : offset + 100]]}
        )["jobs"]
    records = []
    for job in details:
        container = job.get("container", {})
        resources = {r["type"]: r["value"] for r in container.get("resourceRequirements", [])}
        if resources != {"VCPU": "2", "MEMORY": "4096"}:
            raise ValueError("a submitted suite job does not have the requested 2-vCPU/4-GiB reservation")
        records.append(
            {
                "job_id": job["jobId"],
                "job_name": job["jobName"],
                "status": job["status"],
                "reason": job.get("statusReason"),
                "task_arn": container.get("taskArn"),
                "exit_code": container.get("exitCode"),
                "resources": resources,
                "attempts": len(job.get("attempts", [])),
            }
        )
    active = sum(r["status"] in ACTIVE for r in records)
    running = sum(r["status"] in ("STARTING", "RUNNING") for r in records)
    if running > 32:
        raise ValueError("suite running task count exceeds the 64-vCPU ceiling")
    return {
        "counts": dict(Counter(r["status"] for r in records)),
        "active": active,
        "running_or_starting": running,
        "reserved_vcpus_running_or_starting": running * 2,
        "jobs": records,
    }


def status(client, state, *, advance=False):
    result = {"at_epoch_seconds": time.time(), "campaigns": {}}
    verified = state.setdefault("verified_finals", {})
    for dataset, stages in state["datasets"].items():
        for stage, record in list(stages.items()):
            current = inspect_campaign(client, state["bucket"], record, dataset, verified)
            result["campaigns"][record["request"]["campaign_id"]] = current
            if (
                stage == "base"
                and advance
                and "cached" not in stages
                and set(current["matrix_caches"]) == {str(s) for s in SEEDS}
            ):
                request = cached_request(record["request"], current["matrix_caches"])
                stages["cached"] = {
                    "request": request,
                    **start_request(client, request, state["state_machine"]),
                }
    ids = [r["request"]["campaign_id"] for stages in state["datasets"].values() for r in stages.values()]
    result["batch"] = batch_snapshot(client, state["queue"], ids)
    result["verified_units"] = len(verified)
    result["expected_units"] = 231
    result["complete"] = (
        len(verified) == 231
        and len(result["campaigns"]) == 14
        and all(r["complete"] for r in result["campaigns"].values())
        and result["batch"]["active"] == 0
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start-base", "status", "advance"))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket")
    parser.add_argument("--project", default="continual-bench")
    parser.add_argument("--requests-dir", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--deployment-receipt", type=Path)
    parser.add_argument("--release")
    args = parser.parse_args(argv)
    client = AwsCli(args.region)
    with state_lock(args.state_dir):
        state_path = args.state_dir / "suite-state.json"
        if args.command == "start-base":
            if not all(
                (args.requests_dir, args.smoke_receipt, args.deployment_receipt, args.release, args.bucket)
            ):
                parser.error("start-base requires requests, smoke/deployment receipts, release and bucket")
            smoke, deployment = read_json(args.smoke_receipt), read_json(args.deployment_receipt)
            validate_smoke(smoke, deployment, args.release)
            requests = read_requests(args.requests_dir, args.release)
            workflow = verify_workflow(deployment)
            runtime = preflight(client, args.bucket, args.project, args.release)
            contract = {
                "requests": requests,
                "release": args.release,
                "bucket": args.bucket,
                "region": args.region,
                "smoke_sha256": digest(smoke),
                "deployment_sha256": digest(deployment),
            }
            if state_path.exists():
                state = read_json(state_path)
                if state["contract"] != contract:
                    raise ValueError("the state directory already describes a different suite")
            else:
                state = {
                    "schema": "aws-ordered-dataset-suite-v1",
                    "contract": contract,
                    "bucket": args.bucket,
                    "region": args.region,
                    "release": args.release,
                    "queue": runtime["queue"],
                    "state_machine": runtime["state_machine"],
                    "datasets": {},
                    "verified_finals": {},
                }
                save_json(state_path, state)
            save_json(args.state_dir / "preflight.json", {"runtime": runtime, "workflow": workflow})
            for dataset, request in requests.items():
                stages = state["datasets"].setdefault(dataset, {})
                if "base" not in stages:
                    stages["base"] = {
                        "request": request,
                        **start_request(client, request, runtime["state_machine"]),
                    }
                save_json(state_path, state)
            result = {
                "submitted_base_campaigns": len(state["datasets"]),
                "base_units": 147,
                "cached_units_deferred": 84,
                "complete": False,
            }
        else:
            state = read_json(state_path)
            if state["region"] != args.region:
                raise ValueError("selected region differs from the suite state")
            result = status(client, state, advance=args.command == "advance")
            save_json(state_path, state)
            save_json(args.state_dir / "latest-status.json", result)
        display = result
        if "batch" in result:
            display = {
                **{key: result[key] for key in ("complete", "verified_units", "expected_units")},
                "campaigns": {
                    key: {k: v for k, v in value.items() if k != "matrix_caches"}
                    for key, value in result["campaigns"].items()
                },
                "batch": {k: v for k, v in result["batch"].items() if k != "jobs"},
                "receipt": str(args.state_dir / "latest-status.json"),
            }
        print(json.dumps(display, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
