"""Job lifecycle around Step Functions: reconcile before (re)submitting, record, consolidate, stop.

`batch:submitJob.sync` cancellation is best effort, and a lost submission response can leave
a job the workflow never saw. Every decision therefore starts from the facts in S3 (final
manifests, attempt records) and in Batch (jobs found by their deterministic name).
"""

import json
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from continual_bench.infrastructure.object_store import get_json, put_json

from . import layout
from .contracts import SPOT_BACKEND
from .costs import attempt_estimate
from .records import ensure_rows, put_row_once

ACTIVE = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")


def _stamp(clock=time.time) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(clock()))


def reconcile_unit(store, jobs, item: dict) -> dict:
    """Decide skip / wait / submit / exhausted for one Map item."""
    campaign, run_id, name = item["campaign_id"], item["run_id"], item["job_name"]
    final = layout.final_key(campaign, run_id)
    if store.head(final) is not None:
        return {**item, "decision": "skip", "status": "REUSED", "final_uri": store.uri(final), "job_id": None}
    found = jobs.find(name)
    active = [job for job in found if job["status"] in ACTIVE]
    if len(active) > 1:
        # Lost submission responses produced duplicates: keep the oldest, terminate the rest.
        for duplicate in active[1:]:
            jobs.terminate(duplicate["jobId"], f"duplicate submission for run {run_id} reconciled")
    if active:
        return {**item, "decision": "wait", "job_id": active[0]["jobId"], "status": active[0]["status"]}
    finished = [job for job in found if job["status"] in ("FAILED", "SUCCEEDED")]
    attempts = len(finished)
    if attempts >= item["max_attempts"]:
        return {**item, "decision": "exhausted", "status": "FAILED", "attempts": attempts, "job_id": None}
    return {**item, "decision": "submit", "attempt": attempts + 1, "job_id": None}


def describe_job(jobs, job_id: str) -> dict:
    job = jobs.describe([job_id])[0]
    container = job.get("container", {})
    return {
        "job_id": job_id,
        "status": job["status"],
        "terminal": job["status"] in ("SUCCEEDED", "FAILED"),
        "exit_code": container.get("exitCode"),
        "reason": job.get("statusReason"),
        "task_arn": container.get("taskArn"),
    }


def record_attempt(
    store, jobs, item: dict, *, error: dict | None = None, prices: dict | None = None, clock=time.time
) -> dict:
    """After `.sync` returns or fails: terminate stragglers, persist Batch facts, classify outcome."""
    campaign, run_id = item["campaign_id"], item["run_id"]
    found = jobs.find(item["job_name"])
    for job in found:
        if job["status"] in ACTIVE:
            jobs.terminate(job["jobId"], f"workflow {error.get('Error') if error else 'ended'}; reconciling")
    described = jobs.describe([job["jobId"] for job in found]) if found else []
    records = []
    for job in described:
        container = job.get("container", {})
        record = {
            "campaign_id": campaign,
            "run_id": run_id,
            "job_id": job["jobId"],
            "job_name": job["jobName"],
            "status": job["status"],
            "status_reason": job.get("statusReason"),
            "created_at": job.get("createdAt"),
            "started_at": job.get("startedAt"),
            "stopped_at": job.get("stoppedAt"),
            "exit_code": container.get("exitCode"),
            "task_arn": container.get("taskArn"),
            "attempts": [
                {
                    "task_arn": a.get("container", {}).get("taskArn"),
                    "started_at": a.get("startedAt"),
                    "stopped_at": a.get("stoppedAt"),
                    "exit_code": a.get("container", {}).get("exitCode"),
                    "reason": a.get("statusReason"),
                }
                for a in job.get("attempts", [])
            ],
            "job_definition": job.get("jobDefinition"),
            "resources": {r["type"]: r["value"] for r in container.get("resourceRequirements", [])},
            "ephemeral_gib": (job.get("ephemeralStorage") or container.get("ephemeralStorage") or {}).get(
                "sizeInGiB"
            ),
        }
        records.append(record)
        if job["status"] in ("SUCCEEDED", "FAILED"):  # Terminal Batch facts never change.
            put_row_once(store, layout.analytics_key("attempts", campaign, run_id, job["jobId"]), record)
            if prices and record["started_at"] and record["stopped_at"]:
                estimate = attempt_estimate(
                    started_ms=record["started_at"],
                    stopped_ms=record["stopped_at"],
                    vcpus=float(record["resources"].get("VCPU", 0)),
                    memory_gib=float(record["resources"].get("MEMORY", 0)) / 1024,
                    ephemeral_gib=float(record["ephemeral_gib"] or 20),
                    prices=prices,
                )
                put_row_once(
                    store,
                    layout.analytics_key("cost_estimates", campaign, run_id, job["jobId"]),
                    {
                        "campaign_id": campaign,
                        "run_id": run_id,
                        "job_id": job["jobId"],
                        "status": job["status"],
                        "vcpu_seconds": estimate["reserved"]["vcpus"] * estimate["seconds"],
                        **{
                            k: estimate[k]
                            for k in ("seconds", "billed_seconds", "total_usd", "price_snapshot")
                        },
                        **{f"{k}": v for k, v in estimate["components"].items()},
                        "queue_seconds": (record["started_at"] - record["created_at"]) / 1000
                        if record["created_at"]
                        else None,
                    },
                )
    final = layout.final_key(campaign, run_id)
    published = store.head(final) is not None
    latest = records[-1] if records else None
    if published:
        status = "SUCCEEDED"
    elif latest and latest["status"] == "SUCCEEDED":
        status = "FAILED_PUBLICATION"  # Exit 0 without a final manifest is not a success.
    elif latest and latest["status"] in ACTIVE:
        status = "TERMINATING"
    else:
        status = "FAILED"
    event = {
        "campaign_id": campaign,
        "run_id": run_id,
        "status": status,
        "jobs": records,
        "workflow_error": error,
        "recorded_at": clock(),
    }
    put_json(
        store,
        layout.event_key(campaign, run_id, _stamp(clock), latest["job_id"] if latest else "none", status),
        event,
    )
    return {
        **{k: item[k] for k in ("campaign_id", "run_id", "job_name")},
        "status": status,
        "final_uri": store.uri(final) if published else None,
        "job_id": latest["job_id"] if latest else None,
    }


def reported_outcomes(results) -> dict:
    """Inline Map results are a list of unit outcomes; a Distributed Map with ResultWriter passes
    a descriptor (MapRunArn, ResultWriterDetails) instead. Final manifests remain the source of
    truth either way, so the descriptor contributes no per-unit status."""
    if not isinstance(results, list):
        return {}
    return {r["run_id"]: r for r in results if isinstance(r, dict) and "run_id" in r}


def _publish_zia_comparisons(store, campaign_id, units):
    from continual_bench.evaluation.zia_artifacts import write_zia_campaign_artifacts

    prefix = f"{layout.campaign_prefix(campaign_id)}/artifacts"
    with TemporaryDirectory(prefix="zia-campaign-") as temp:
        write_zia_campaign_artifacts(temp, units)
        root = Path(temp) / "artifacts"
        index = root / "data/index.json"
        for artifact in sorted(root.rglob("*")):
            if artifact.is_file() and artifact != index:
                relative = artifact.relative_to(root).as_posix()
                content_type = "image/svg+xml" if artifact.suffix == ".svg" else "application/json"
                store.put_bytes(f"{prefix}/{relative}", artifact.read_bytes(), content_type=content_type)
        index_key = f"{prefix}/data/index.json"
        store.put_bytes(index_key, index.read_bytes(), content_type="application/json")
    return store.uri(index_key)


def consolidate(store, campaign_id: str, results=None, *, clock=time.time) -> dict:
    plan = get_json(store, layout.plan_key(campaign_id))
    spot = plan["release"].get("benchmark_compute") == SPOT_BACKEND
    reported_by_run = reported_outcomes(results)
    units, analytics_errors = [], []
    for unit in plan["units"]:
        final_key = layout.final_key(campaign_id, unit["run_id"])
        if store.head(final_key) is not None:
            final = get_json(store, final_key)
            try:
                if spot:
                    from .spot_lifecycle import verify_publication

                    analytics = verify_publication(store, final, unit["contract_sha256"])
                else:
                    analytics = ensure_rows(store, final)
            except Exception as error:  # Keep consolidating; the campaign is reported incomplete.
                analytics = "error"
                analytics_errors.append({"run_id": unit["run_id"], "error_type": type(error).__name__,
                                         "message": str(error)[:500]})
            units.append(
                {
                    "run_id": unit["run_id"],
                    "labels": unit["labels"],
                    "status": "PUBLICATION_FAILED" if spot and analytics == "error" else "SUCCEEDED",
                    "final_uri": store.uri(final_key),
                    "metrics": final.get("metrics"),
                    "holdout": final.get("holdout"),
                    "decision_threshold": final.get("decision_threshold"),
                    "attempt": final["attempt"],
                    "artifacts_index_uri": final.get("artifacts_index_uri"),
                    "model_package_arn": (final.get("model") or {}).get("model_package_arn"),
                    "analytics": analytics,
                    **({"tracking": final["tracking"]} if "tracking" in final else {}),
                }
            )
        else:
            reported = reported_by_run.get(unit["run_id"], {})
            if spot:
                state_key = f"{layout.run_prefix(campaign_id, unit['run_id'])}/spot/state.json"
                try:
                    reported = get_json(store, state_key)
                except FileNotFoundError:
                    pass
            units.append(
                {
                    "run_id": unit["run_id"],
                    "labels": unit["labels"],
                    "status": reported.get("status", "FAILED"),
                    "final_uri": None,
                    **({"status_reason": reported.get("reason")} if spot else {}),
                }
            )
    succeeded = sum(u["status"] == "SUCCEEDED" for u in units)
    summary = {
        "version": "campaign-summary-v1",
        "campaign_id": campaign_id,
        "release": plan["release"]["git_sha"],
        "snapshot": plan["snapshot"],
        "units": units,
        "succeeded": succeeded,
        "total": len(units),
        "complete": succeeded == len(units) and not analytics_errors,
        "analytics_errors": analytics_errors,
        "consolidated_at": clock(),
        "final_manifests": [u["final_uri"] for u in units if u["final_uri"]],
    }
    receipts = [unit.get("tracking") for unit in units]
    if any(receipt is not None for receipt in receipts):
        required = any(receipt and receipt.get("required") for receipt in receipts)
        complete = not required or all(receipt and receipt.get("complete") for receipt in receipts)
        summary["tracking"] = {"required": required, "complete": complete}
        summary["complete"] = summary["complete"] and complete
    if plan["snapshot"] is not None:
        summary["artifacts_index_uri"] = _publish_zia_comparisons(store, campaign_id, units)
    put_json(store, layout.summary_key(campaign_id), summary)
    row = {k: summary[k] for k in ("campaign_id", "release", "succeeded", "total", "complete")}
    row["snapshot_id"] = (plan["snapshot"] or {}).get("id")
    row["final_manifests"] = summary["final_manifests"]
    store.put_bytes(layout.analytics_key("campaigns", campaign_id), (json.dumps(row) + "\n").encode())
    result = {
        "campaign_id": campaign_id,
        "complete": summary["complete"],
        "succeeded": succeeded,
        "total": len(units),
        "summary_uri": store.uri(layout.summary_key(campaign_id)),
    }
    if "artifacts_index_uri" in summary:
        result["artifacts_index_uri"] = summary["artifacts_index_uri"]
    return result


def campaign_job_name(campaign_id: str):
    """Exact job-name pattern of one campaign: `cb-<campaign_id>-<20-hex run_id>`.

    The Batch name filter only supports a prefix wildcard, so `cb-a-*` also matches campaign
    `a-b`; this pattern keeps reconciliation from touching another campaign's jobs."""
    return re.compile(rf"^cb-{re.escape(campaign_id)}-[0-9a-f]{{20}}$")


def reconcile_campaign(store, jobs, campaign_id: str, reason: str, *, clock=time.time) -> dict:
    """Terminate every non-terminal job of the campaign and report the ECS task states."""
    own = campaign_job_name(campaign_id)
    found = [job for job in jobs.find(f"cb-{campaign_id}-*") if own.match(job["jobName"])]
    terminated = []
    for job in found:
        if job["status"] in ACTIVE:
            jobs.terminate(job["jobId"], f"campaign reconciliation: {reason}")
            terminated.append(job["jobId"])
    described = jobs.describe([job["jobId"] for job in found]) if found else []
    tasks = jobs.task_states(described) if described else {}
    active_jobs = [job["jobId"] for job in described if job["status"] in ACTIVE]
    running_tasks = [
        arn
        for arn, state in tasks.items()
        if state != "STOPPED" and not state.startswith("MISSING")
    ]
    report = {
        "campaign_id": campaign_id,
        "reason": reason,
        "terminated": terminated,
        "jobs": {job["jobId"]: job["status"] for job in described},
        "tasks": tasks,
        "active_jobs": active_jobs,
        "running_tasks": running_tasks,
        "clean": not active_jobs and not running_tasks,
        "checked_at": clock(),
    }
    put_json(store, layout.reconciliation_key(campaign_id, _stamp(clock)), report)
    return {k: report[k] for k in ("campaign_id", "clean", "terminated", "active_jobs", "running_tasks")}
