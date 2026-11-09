"""Control-plane Lambda: one versioned function, dispatched by `action`.

Step Functions pins the version-qualified ARN recorded in the campaign's release, so a merge
during a campaign does not change the control code it runs. Exceptions keep their class name
(`SnapshotNotReady`, `CampaignConflict`, `CapacityError`, `ValidationError`) so the state
machines can catch them by name. EventBridge invokes `observe`, which records lifecycle events,
alerts through SNS and reconciles stopped executions; it never starts a benchmark.
"""

import json
import os
import time

from continual_bench.application.cloud import layout, lifecycle, planning, snapshots
from continual_bench.application.cloud.contracts import CampaignRequest, DatasetRequest, SnapshotRef

TERMINAL_EXECUTION = ("FAILED", "TIMED_OUT", "ABORTED")


class Context:
    """Lazily built adapters; tests inject fakes through the same attributes."""

    def __init__(self, env=None):
        self.env = os.environ if env is None else env
        self._store = self._jobs = self._sns = None

    @property
    def store(self):
        if self._store is None:
            from continual_bench.infrastructure.aws.s3 import S3ObjectStore

            self._store = S3ObjectStore(self.env["BENCH_BUCKET"], kms_key_id=self.env.get("BENCH_KMS_KEY_ID"))
        return self._store

    @property
    def jobs(self):
        if self._jobs is None:
            from continual_bench.infrastructure.aws.batch import BatchJobs

            self._jobs = BatchJobs(self.env["BENCH_JOB_QUEUE"])
        return self._jobs

    def notify(self, subject, message):
        topic = self.env.get("BENCH_SNS_TOPIC")
        if not topic:
            return None
        if self._sns is None:
            from continual_bench.infrastructure.aws.session import client

            self._sns = client("sns")
        return self._sns.publish(
            TopicArn=topic, Subject=subject[:100], Message=json.dumps(message, default=str)
        )

    def capacity(self, concurrency):
        from continual_bench.infrastructure.aws.quotas import fargate_vcpu_quota, queue_capacity

        spot = self.env.get("BENCH_FARGATE_SPOT", "false") == "true"
        return planning.admit_capacity(
            concurrency,
            float(self.env.get("BENCH_BENCHMARK_VCPUS", "1")),
            quota_vcpus=fargate_vcpu_quota(spot=spot),
            queue=queue_capacity(self.env["BENCH_JOB_QUEUE"]),
        )

    @property
    def project_root(self):
        return self.env.get("BENCH_PROJECT_ROOT") or self.env.get("LAMBDA_TASK_ROOT") or "."

    def prices(self):
        from continual_bench.application.cloud.costs import load_prices

        try:
            return load_prices(project_root=self.project_root)
        except FileNotFoundError:
            return None


def _stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# Benchmark campaign ---------------------------------------------------------------------------------
def plan_campaign(ctx, event):
    request = CampaignRequest(**event["request"])
    release = planning.resolve_release(ctx.store, request.release)
    snapshot = snapshots.verify_snapshot(ctx.store, request.snapshot) if request.snapshot else None
    plan = planning.build_plan(request, release, snapshot_manifest=snapshot, project_root=ctx.project_root)
    evidence = ctx.capacity(plan["concurrency"])
    from continual_bench.infrastructure.object_store import put_json

    put_json(
        ctx.store,
        f"{layout.campaign_prefix(request.campaign_id)}/preflight/{_stamp()}.json",
        {"capacity": evidence, "snapshot": plan["snapshot"], "release": release.git_sha},
    )
    published = planning.publish_plan(ctx.store, plan)
    return {
        **published,
        "timeouts": plan["timeouts"],
        "max_attempts": plan["max_attempts"],
        "bucket": ctx.env["BENCH_BUCKET"],
        "control_function": release.control_function,
        "capacity": {
            k: evidence[k] for k in ("required_vcpus", "fargate_vcpu_quota", "max_parallel_admitted")
        },
    }


def reconcile_unit(ctx, event):
    return lifecycle.reconcile_unit(ctx.store, ctx.jobs, event["item"])


def describe_job(ctx, event):
    return lifecycle.describe_job(ctx.jobs, event["job_id"])


def record_attempt(ctx, event):
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("Cause"), str) and len(error["Cause"]) > 4000:
        error = {**error, "Cause": error["Cause"][:4000]}
    return lifecycle.record_attempt(ctx.store, ctx.jobs, event["item"], error=error, prices=ctx.prices())


def consolidate(ctx, event):
    result = lifecycle.consolidate(ctx.store, event["campaign_id"], event.get("results"))
    if not result["complete"] and event.get("notify", True):
        ctx.notify(f"continual-bench campaign {event['campaign_id']} incomplete", result)
    return result


def reconcile_campaign(ctx, event):
    return lifecycle.reconcile_campaign(
        ctx.store, ctx.jobs, event["campaign_id"], event.get("reason", "requested")
    )


# Dataset pipeline -----------------------------------------------------------------------------------
def resolve_dataset(ctx, event):
    request = DatasetRequest(**event["request"])
    snapshot = request.snapshot()
    release = planning.resolve_release(ctx.store, event.get("release", "latest"))
    base = {
        "snapshot": snapshot.model_dump(),
        "snapshot_id": snapshot.snapshot_id,
        "mode": request.mode,
        "import_uri": request.import_uri or "",
        "release": release.git_sha,
        "control_function": release.control_function,
        "runtime_image": release.images["runtime"].uri,
        "prepare_job_definition": release.job_definitions["prepare"],
        "job_name": f"cb-prepare-{snapshot.dataset}-{snapshot.scope}-{snapshot.revision[:12]}",
    }
    try:
        manifest = snapshots.verify_snapshot(ctx.store, snapshot)
        return {**base, "decision": "ready", "manifest_uri": manifest["manifest_uri"]}
    except snapshots.SnapshotNotReady as error:
        if error.reason == "incompatible":
            raise
    if snapshots.prepared_status(ctx.store, snapshot) is not None:
        return {**base, "decision": "publish"}
    try:
        snapshots.verify_raw(ctx.store, snapshot)
        return {**base, "decision": "prepare"}
    except snapshots.SnapshotNotReady as error:
        if error.reason == "incompatible":
            raise
    if request.mode == "prepare":
        raise snapshots.SnapshotNotReady("missing", "raw files absent; use mode=download or mode=import")
    return {**base, "decision": "acquire"}


def verify_raw(ctx, event):
    manifest = snapshots.verify_raw(ctx.store, SnapshotRef(**event["snapshot"]))
    return {"files": len(manifest["files"]), "verified": True}


def publish_snapshot(ctx, event):
    published = snapshots.publish_snapshot(ctx.store, SnapshotRef(**event["snapshot"]))
    return {k: published[k] for k in ("manifest_uri", "marker_uri", "created", "snapshot_id")}


# Lifecycle observation ------------------------------------------------------------------------------
def observe(ctx, event):
    detail, kind = event.get("detail", {}), event.get("detail-type")
    if kind == "Batch Job State Change":
        tags = detail.get("tags") or {}
        name = detail.get("jobName", "")
        campaign, run_id = tags.get("cb-campaign"), tags.get("cb-run-id")
        if not (campaign and run_id) and name.startswith("cb-") and not name.startswith("cb-prepare-"):
            campaign, _, run_id = name[3:].rpartition("-")
        status = detail.get("status")
        if campaign and run_id:
            from continual_bench.infrastructure.object_store import put_json

            put_json(
                ctx.store,
                layout.event_key(campaign, run_id, _stamp(), detail.get("jobId", "unknown"), status),
                {
                    "source": "eventbridge",
                    "status": status,
                    "reason": detail.get("statusReason"),
                    "job_id": detail.get("jobId"),
                    "attempts": len(detail.get("attempts", [])),
                    "task_arn": detail.get("container", {}).get("taskArn"),
                },
            )
        if status == "FAILED":
            ctx.notify(
                f"continual-bench job failed: {name}",
                {"job": name, "reason": detail.get("statusReason"), "campaign": campaign, "run_id": run_id},
            )
        return {"observed": kind, "status": status, "campaign_id": campaign, "run_id": run_id}
    if kind == "Step Functions Execution Status Change":
        status, name = detail.get("status"), detail.get("name", "")
        result = {"observed": kind, "status": status, "execution": name}
        if status in TERMINAL_EXECUTION and detail.get("stateMachineArn", "").endswith(
            ctx.env.get("BENCH_BENCHMARK_STATE_MACHINE_SUFFIX", ":continual-bench-benchmark")
        ):
            # Execution names are `<campaign_id>` or `<campaign_id>--<retry-suffix>`.
            campaign = name.split("--", 1)[0]
            result["reconciliation"] = lifecycle.reconcile_campaign(
                ctx.store, ctx.jobs, campaign, f"execution {status}"
            )
        if status in TERMINAL_EXECUTION:
            ctx.notify(f"continual-bench execution {status}: {name}", result)
        return result
    raise ValueError(f"unsupported event: {kind}")


ACTIONS = {
    f.__name__: f
    for f in (
        plan_campaign,
        reconcile_unit,
        describe_job,
        record_attempt,
        consolidate,
        reconcile_campaign,
        resolve_dataset,
        verify_raw,
        publish_snapshot,
        observe,
    )
}


def handler(event, _context=None, ctx=None):
    ctx = ctx or Context()
    action = event.get("action") or ("observe" if "detail-type" in event else None)
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    result = ACTIONS[action](ctx, event)
    print(json.dumps({"action": action, "result_keys": sorted(result)}, default=str))
    return result
