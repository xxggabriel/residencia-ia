"""Reconciliation before submission, attempt records, consolidation, cancellation and the control Lambda."""

import json
import subprocess
import sys

import pytest

from continual_bench.application.cloud import layout, lifecycle, planning
from continual_bench.application.cloud.contracts import CampaignRequest
from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.infrastructure.aws.batch import task_cluster
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json, put_json
from continual_bench.orchestration import aws_control


class FakeJobs:
    def __init__(self, jobs=None, tasks=None):
        self.jobs = jobs or {}
        self.tasks = tasks or {}
        self.terminated, self.submitted = [], []

    def find(self, pattern):
        prefix = pattern.rstrip("*")
        match = (lambda n: n.startswith(prefix)) if pattern.endswith("*") else (lambda n: n == pattern)
        return [
            {"jobId": i, "jobName": j["jobName"], "status": j["status"], "createdAt": j.get("createdAt", 0)}
            for i, j in sorted(self.jobs.items(), key=lambda kv: kv[1].get("createdAt", 0))
            if match(j["jobName"])
        ]

    def describe(self, ids):
        return [{"jobId": i, **self.jobs[i]} for i in ids]

    def terminate(self, job_id, reason):
        self.terminated.append(job_id)
        self.jobs[job_id]["status"] = "FAILED"
        self.jobs[job_id]["statusReason"] = reason

    def task_states(self, jobs):
        return {
            j["container"]["taskArn"]: self.tasks.get(j["container"]["taskArn"], "STOPPED")
            for j in jobs
            if j.get("container", {}).get("taskArn")
        }


def job(name, status, created=0, task="arn:aws:ecs:us-east-1:1:task/cluster/t", **extra):
    return {
        "jobName": name,
        "status": status,
        "createdAt": created,
        "startedAt": created + 1000,
        "stoppedAt": created + 61_000 if status in ("SUCCEEDED", "FAILED") else None,
        "container": {
            "taskArn": task,
            "exitCode": 0 if status == "SUCCEEDED" else 1,
            "resourceRequirements": [{"type": "VCPU", "value": "1"}, {"type": "MEMORY", "value": "2048"}],
        },
        "jobDefinition": "arn:aws:batch:us-east-1:1:job-definition/cb-benchmark:1",
        **extra,
    }


@pytest.fixture
def planned(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path)
    release = publish_fixture_release(store)
    request = CampaignRequest(campaign_id="c1", snapshot=snapshot.model_dump())
    plan = planning.build_plan(request, release, snapshot_manifest=verify_snapshot(store, snapshot))
    published = planning.publish_plan(store, plan)
    return store, plan, published["items"]


def test_reconcile_decides_from_s3_and_batch_facts(planned):
    store, _, items = planned
    item = items[0]
    assert lifecycle.reconcile_unit(store, FakeJobs(), item)["decision"] == "submit"
    jobs = FakeJobs({"a": job(item["job_name"], "RUNNING", 1), "b": job(item["job_name"], "RUNNABLE", 2)})
    decision = lifecycle.reconcile_unit(store, jobs, item)
    assert decision["decision"] == "wait" and decision["job_id"] == "a" and jobs.terminated == ["b"]
    failed = FakeJobs({"a": job(item["job_name"], "FAILED", 1), "b": job(item["job_name"], "FAILED", 2)})
    assert lifecycle.reconcile_unit(store, failed, item)["decision"] == "exhausted"
    one = FakeJobs({"a": job(item["job_name"], "FAILED", 1)})
    assert lifecycle.reconcile_unit(store, one, item) | {} == {
        **item,
        "decision": "submit",
        "attempt": 2,
        "job_id": None,
    }
    put_json(store, layout.final_key("c1", item["run_id"]), {"done": True})
    assert lifecycle.reconcile_unit(store, failed, item)["decision"] == "skip"


def test_record_attempt_classifies_and_terminates_stragglers(planned):
    store, _, items = planned
    item = items[1]
    jobs = FakeJobs({"x": job(item["job_name"], "SUCCEEDED", 1)})
    prices = {
        "retrieved": "test",
        "prices": {
            "fargate_vcpu_hour": {"usd": 0.04},
            "fargate_memory_gb_hour": {"usd": 0.004},
            "fargate_ephemeral_storage_gb_hour": {"usd": 0.0001},
        },
    }
    outcome = lifecycle.record_attempt(store, jobs, item, prices=prices)
    assert outcome["status"] == "FAILED_PUBLICATION"  # exit 0 without a final manifest
    estimate = json.loads(store.get_bytes(layout.analytics_key("cost_estimates", "c1", item["run_id"], "x")))
    assert estimate["billed_seconds"] == 60 and estimate["queue_seconds"] == 1.0
    put_json(store, layout.final_key("c1", item["run_id"]), {"done": True})
    assert lifecycle.record_attempt(store, jobs, item)["status"] == "SUCCEEDED"
    running = FakeJobs({"y": job(items[2]["job_name"], "RUNNING", 5)})
    outcome = lifecycle.record_attempt(store, running, items[2], error={"Error": "States.Timeout"})
    assert running.terminated == ["y"] and outcome["status"] == "FAILED"
    events = list(store.list(f"{layout.run_prefix('c1', items[2]['run_id'])}/events/"))
    assert events and get_json(store, events[0])["workflow_error"]["Error"] == "States.Timeout"


def test_consolidate_points_to_final_manifests(planned):
    store, plan, items = planned
    for unit in plan["units"][:3]:
        put_json(
            store,
            layout.final_key("c1", unit["run_id"]),
            {"attempt": {"job_id": "j"}, "metrics": {"accuracy": 0.5}, "holdout": {"accuracy": 0.4}},
        )
    result = lifecycle.consolidate(store, "c1", [{"run_id": plan["units"][3]["run_id"], "status": "FAILED"}])
    assert (result["complete"], result["succeeded"], result["total"]) == (False, 3, 4)
    summary = get_json(store, layout.summary_key("c1"))
    assert len(summary["final_manifests"]) == 3 and summary["release"] == plan["release"]["git_sha"]
    row = json.loads(store.get_bytes(layout.analytics_key("campaigns", "c1")))
    assert row["complete"] is False and len(row["final_manifests"]) == 3


def test_campaign_reconciliation_leaves_no_active_task(planned):
    store, _, items = planned
    jobs = FakeJobs(
        {
            "a": job(items[0]["job_name"], "RUNNING", task="arn:aws:ecs:us-east-1:1:task/c/a"),
            "b": job(items[1]["job_name"], "SUCCEEDED", task="arn:aws:ecs:us-east-1:1:task/c/b"),
            "z": job("cb-other-campaign-x", "RUNNING"),
        },
        tasks={"arn:aws:ecs:us-east-1:1:task/c/a": "STOPPED"},
    )
    report = lifecycle.reconcile_campaign(store, jobs, "c1", "execution ABORTED")
    assert report["clean"] and report["terminated"] == ["a"] and jobs.jobs["z"]["status"] == "RUNNING"
    stuck = FakeJobs(
        {"a": job(items[0]["job_name"], "SUCCEEDED", task="arn:aws:ecs:us-east-1:1:task/c/a")},
        tasks={"arn:aws:ecs:us-east-1:1:task/c/a": "RUNNING"},
    )
    assert lifecycle.reconcile_campaign(store, stuck, "c1", "check")["running_tasks"] == [
        "arn:aws:ecs:us-east-1:1:task/c/a"
    ]
    assert list(store.list("campaigns/c1/reconciliation/"))
    assert task_cluster("arn:aws:ecs:us-east-1:1:task/cluster-x/abc") == "cluster-x"


class FakeContext(aws_control.Context):
    def __init__(self, store, jobs, quota=8):
        super().__init__(
            {
                "BENCH_BUCKET": "bucket",
                "BENCH_JOB_QUEUE": "q",
                "BENCH_BENCHMARK_VCPUS": "1",
                "BENCH_PROJECT_ROOT": ".",
            }
        )
        self._store, self._jobs, self.quota, self.notifications = store, jobs, quota, []

    def capacity(self, concurrency):
        return planning.admit_capacity(
            concurrency,
            1,
            quota_vcpus=self.quota,
            queue={"queue_state": "ENABLED", "queue_status": "VALID", "max_vcpus": 8},
        )

    def notify(self, subject, message):
        self.notifications.append(subject)


def test_control_lambda_plans_and_routes(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path)
    release = publish_fixture_release(store)
    ctx = FakeContext(store, FakeJobs())
    request = {"campaign_id": "c2", "snapshot": snapshot.model_dump(), "release": "latest"}
    plan = aws_control.handler({"action": "plan_campaign", "request": request}, ctx=ctx)
    assert (
        plan["concurrency"] == 4
        and len(plan["items"]) == 4
        and plan["control_function"] == release.control_function
    )
    assert list(store.list("campaigns/c2/preflight/"))
    with pytest.raises(planning.CapacityError):
        aws_control.handler(
            {"action": "plan_campaign", "request": {**request, "campaign_id": "c3"}},
            ctx=FakeContext(store, FakeJobs(), quota=2),
        )
    with pytest.raises(Exception) as missing:
        aws_control.handler(
            {
                "action": "plan_campaign",
                "request": {
                    **request,
                    "campaign_id": "c4",
                    "snapshot": {**snapshot.model_dump(), "scope": "full"},
                },
            },
            ctx=ctx,
        )
    assert type(missing.value).__name__ == "SnapshotNotReady"
    decision = aws_control.handler({"action": "reconcile_unit", "item": plan["items"][0]}, ctx=ctx)
    assert decision["decision"] == "submit"
    with pytest.raises(ValueError):
        aws_control.handler({"action": "start_benchmark"}, ctx=ctx)


def test_resolve_dataset_decisions(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    publish_fixture_release(store)
    ctx = FakeContext(store, FakeJobs())
    resolved = aws_control.handler(
        {"action": "resolve_dataset", "release": "latest", "request": {"dataset": "social", "scope": "full"}},
        ctx=ctx,
    )
    assert resolved["decision"] == "acquire" and resolved["import_uri"] == ""
    assert resolved["prepare_job_definition"].endswith(":1") and resolved["control_function"].endswith(":1")
    with pytest.raises(Exception, match="missing"):
        aws_control.handler(
            {
                "action": "resolve_dataset",
                "release": "latest",
                "request": {"dataset": "social", "scope": "full", "mode": "prepare"},
            },
            ctx=ctx,
        )
    publish_fixture_snapshot(store, tmp_path)
    ready = aws_control.handler(
        {
            "action": "resolve_dataset",
            "release": "latest",
            "request": {"dataset": "social", "scope": "smoke"},
        },
        ctx=ctx,
    )
    assert ready["decision"] == "ready"


def test_observe_records_alerts_and_reconciles_but_never_starts(planned):
    store, _, items = planned
    jobs = FakeJobs({"a": job(items[0]["job_name"], "RUNNING")})
    ctx = FakeContext(store, jobs)
    event = {
        "detail-type": "Batch Job State Change",
        "detail": {
            "jobName": items[0]["job_name"],
            "jobId": "a",
            "status": "FAILED",
            "statusReason": "Essential container exited",
            "tags": {"cb-campaign": "c1", "cb-run-id": items[0]["run_id"]},
            "container": {"taskArn": "t"},
        },
    }
    observed = aws_control.handler(event, ctx=ctx)
    assert observed["run_id"] == items[0]["run_id"] and ctx.notifications
    assert list(store.list(f"{layout.run_prefix('c1', items[0]['run_id'])}/events/"))
    aborted = {
        "detail-type": "Step Functions Execution Status Change",
        "detail": {
            "status": "ABORTED",
            "name": "c1--retry2",
            "stateMachineArn": "arn:aws:states:us-east-1:1:stateMachine:continual-bench-benchmark",
        },
    }
    ctx.env = {**ctx.env, "BENCH_BENCHMARK_STATE_MACHINE_SUFFIX": ":continual-bench-benchmark"}
    result = aws_control.handler(aborted, ctx=ctx)
    assert result["reconciliation"]["terminated"] == ["a"] and jobs.submitted == []


def test_control_plane_imports_stay_light():
    """The Lambda zip ships pydantic and PyYAML only; heavy or forbidden modules must not load."""
    code = (
        "import sys; import continual_bench.orchestration.aws_control; "
        "import continual_bench.application.matrix; "
        "print(sorted(m for m in ('numpy','pyarrow','river','scipy','duckdb','mlflow','zenml','pandas',"
        "'confluent_kafka','boto3') if m in sys.modules))"
    )
    loaded = json.loads(
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.replace("'", '"')
    )
    assert loaded == []


def test_reconciliation_ignores_campaigns_sharing_a_prefix(planned):
    store, _, items = planned
    sibling = "cb-c1-b-" + "f" * 20  # campaign "c1-b" also matches the Batch filter cb-c1-*
    jobs = FakeJobs(
        {
            "own": job(items[0]["job_name"], "RUNNING", task="arn:aws:ecs:us-east-1:1:task/c/own"),
            "sibling": job(sibling, "RUNNING", task="arn:aws:ecs:us-east-1:1:task/c/sib"),
        }
    )
    report = lifecycle.reconcile_campaign(store, jobs, "c1", "post-campaign verification")
    assert report["terminated"] == ["own"] and jobs.jobs["sibling"]["status"] == "RUNNING"
    assert lifecycle.campaign_job_name("c1").match(items[0]["job_name"])
    assert not lifecycle.campaign_job_name("c1").match(sibling)


def test_consolidation_accepts_the_distributed_map_descriptor(planned):
    store, plan, _ = planned
    descriptor = {
        "MapRunArn": "arn:aws:states:us-east-1:1:mapRun:x/y:z",
        "ResultWriterDetails": {"Bucket": "b", "Key": "campaigns/c1/map-results/manifest.json"},
    }
    result = lifecycle.consolidate(store, "c1", descriptor)
    assert (result["complete"], result["succeeded"], result["total"]) == (False, 0, 4)
    assert {u["status"] for u in get_json(store, layout.summary_key("c1"))["units"]} == {"FAILED"}
