"""Reconcile a logical benchmark against its immutable EC2 attempt ledger."""

import hashlib
import json
import time
from contextlib import contextmanager
from uuid import uuid4

from botocore.exceptions import ClientError, ConnectionError, HTTPClientError

from continual_bench.infrastructure.aws.session import error_code
from continual_bench.infrastructure.object_store import (
    ObjectExists,
    get_json,
    json_bytes,
    put_json,
    split_uri,
)

from . import layout
from .records import ensure_rows, put_row_once

TERMINATED = {"terminated", "missing"}
BOOT_TIMEOUT_SECONDS = 15 * 60
VISIBILITY_SECONDS = 5 * 60
LAUNCH_LEASE_SECONDS = 330  # Longer than the control Lambda's 300-second timeout.
RETRYABLE = {"SPOT_INTERRUPTED", "INSTANCE_LOST", "BOOTSTRAP_FAILED", "PUBLICATION_FAILED"}
RETRYABLE_LAUNCH_ERRORS = {"InternalError", "RequestLimitExceeded", "ServiceUnavailable", "RequestTimeout"}


def _optional(store, key):
    try:
        return get_json(store, key)
    except FileNotFoundError:
        return None


def _once(store, key, value):
    try:
        put_json(store, key, value, if_absent=True)
    except ObjectExists:
        return get_json(store, key)
    return value


def execution_key(execution_arn):
    return hashlib.sha256(execution_arn.encode()).hexdigest()


def _stop_key(campaign, owner):
    return f"{layout.campaign_prefix(campaign)}/stopped-executions/{owner}.json"


def _closed_key(campaign, owner):
    return f"{layout.campaign_prefix(campaign)}/closed-executions/{owner}.json"


def _launches_prefix(campaign, owner):
    return f"{layout.campaign_prefix(campaign)}/executions/{owner}/launches/"


def _get_versioned(store, key):
    try:
        data, revision = store.read_versioned(key)
    except FileNotFoundError:
        return {}, None
    return json.loads(data), revision


def admit_execution(store, campaign, execution_arn):
    """Fence stale or concurrent parent executions before any child can request EC2."""
    owner = execution_key(execution_arn)
    key = f"{layout.campaign_prefix(campaign)}/active-execution.json"
    while True:
        active, revision = _get_versioned(store, key)
        previous = active.get("execution_key")
        if previous == owner:
            return True
        if previous and store.head(_closed_key(campaign, previous)) is None:
            return False
        try:
            store.put_versioned(
                key, json_bytes({"execution_key": owner, "execution_arn": execution_arn}), revision
            )
        except ObjectExists:
            continue
        return True


def owns_execution(store, campaign, owner):
    key = f"{layout.campaign_prefix(campaign)}/active-execution.json"
    return (_optional(store, key) or {}).get("execution_key") == owner


@contextmanager
def _launch_lease(store, key, clock):
    lock, revision = _get_versioned(store, key)
    if lock.get("expires", 0) > clock():
        yield False
        return
    token = uuid4().hex
    try:
        revision = store.put_versioned(
            key, json_bytes({"token": token, "expires": clock() + LAUNCH_LEASE_SECONDS}), revision
        )
    except ObjectExists:
        yield False
        return
    # If EC2's response was lost, retain the lease so a stop cannot race delayed instance visibility.
    yield True
    try:
        store.put_versioned(key, json_bytes({"token": token, "expires": 0}), revision)
    except ObjectExists:
        pass


def stop_execution(store, campaign, execution_arn, reason, *, clock=time.time):
    owner = execution_key(execution_arn)
    _once(store, _stop_key(campaign, owner), {"reason": reason, "stopped_at": clock()})
    return owner


def cancel_campaign(store, campaign, *, clock=time.time):
    return _once(store, layout.cancelled_key(campaign), {"state": "CANCELLED", "cancelled_at": clock()})


def verify_publication(store, final, contract_sha256):
    run_id = final["run_id"]
    tracking = final.get("tracking") or {}
    if final.get("contract_sha256") != contract_sha256 or (
        tracking.get("required") and not tracking.get("complete")
    ):
        raise ValueError(f"Invalid final manifest for {run_id}")
    report = store.head(split_uri(final["report_uri"])[1])
    if report is None or report.sha256 != final["report_sha256"]:
        raise ValueError(f"Published report failed its checksum for {run_id}")
    if store.head(split_uri(final["artifacts_index_uri"])[1]) is None:
        raise ValueError(f"Artifact index is missing for {run_id}")
    status = ensure_rows(store, final)
    if status != "present" and status != "repaired":
        raise ValueError(f"Analytics publication incomplete for {run_id}")
    return status


class SpotRun:
    """Advance by one poll. A transport retry can replay the same provisioning request."""

    def __init__(self, store, jobs, item, execution_arn, *, clock=time.time):
        self.store, self.jobs, self.item, self.clock = store, jobs, item, clock
        self.campaign, self.run_id = item["campaign_id"], item["run_id"]
        self.owner = execution_key(execution_arn)
        self.final = layout.final_key(self.campaign, self.run_id)
        self.cancel = layout.cancelled_key(self.campaign)
        self.state_key = f"{layout.run_prefix(self.campaign, self.run_id)}/spot/state.json"

    def result(self, attempt, status, *, reason=None, instance_id=None, terminal=False):
        result = {
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "instance_id": instance_id,
            "attempt": attempt,
            "terminal": terminal,
            "wait_seconds": self.item["poll_seconds"],
            "final_uri": self.store.uri(self.final) if status == "SUCCEEDED" else None,
        }
        put_json(self.store, self.state_key, {**result, "updated_at": self.clock()})
        return result

    def record(self, attempt):
        instance = attempt.get("instance_id")
        if instance:
            # Keep the existing Athena attempt table and include EC2 coordinates in the raw record.
            put_row_once(
                self.store, layout.analytics_key("attempts", self.campaign, self.run_id, instance), attempt
            )

    def resolved(self, attempt, status, reason, *, grant=None, instance=None, worker=None, host=None):
        prefix = layout.spot_attempt_prefix(self.campaign, self.run_id, attempt)
        grant, instance, worker, host = grant or {}, instance or {}, worker or {}, host or {}
        decision = _once(
            self.store,
            f"{prefix}/outcome.json",
            {
                "campaign_id": self.campaign,
                "run_id": self.run_id,
                "job_name": self.item["job_name"],
                "attempt": attempt,
                "attempt_key": grant.get("attempt_key"),
                "status": status,
                "status_reason": reason,
                "finished_at": self.clock(),
                "stopped_at": round(self.clock() * 1000),
                "started_at": round(grant["launch_at"] * 1000) if grant.get("launch_at") else None,
                "exit_code": host.get("exit_code", (worker.get("diagnostic") or {}).get("exit_code")),
                "worker_status": worker.get("state"),
                "resumed_checkpoint": worker.get("resumed_checkpoint"),
                "checkpoint": worker.get("checkpoint"),
                "interruption_reason": worker.get("interruption_reason"),
                "host_status": host.get("state"),
                "ec2_state": instance.get("state"),
                "ec2_reason": instance.get("reason"),
                "ec2_reason_code": instance.get("reason_code"),
                "volume_cleanup_verified": bool(grant),
                "resources": {"VCPU": "2", "MEMORY": "4096"},
                "image": self.item["runtime_image"],
                **{
                    k: grant.get(k)
                    for k in (
                        "instance_id",
                        "instance_type",
                        "instance_lifecycle",
                        "availability_zone",
                        "log_group",
                        "log_stream",
                        "job_id",
                        "volume_ids",
                    )
                },
            },
        )
        self.record(decision)
        return self.previous_result(decision)

    def previous_result(self, previous):
        status, attempt = previous["status"], previous["attempt"]
        terminal = status not in RETRYABLE or attempt >= self.item["max_attempts"]
        return self.result(
            attempt,
            status,
            reason=previous["status_reason"],
            instance_id=previous["instance_id"],
            terminal=terminal,
        )

    def published(self):
        final = _optional(self.store, self.final)
        if final is None:
            return False
        verify_publication(self.store, final, self.item["contract_sha256"])
        return True

    def advance(self):
        if not owns_execution(self.store, self.campaign, self.owner):
            return {
                "run_id": self.run_id,
                "status": "EXECUTION_REPLACED",
                "terminal": True,
                "reason": "Another workflow owns this campaign",
            }
        previous = None
        for attempt in range(1, self.item["max_attempts"] + 1):
            prefix = layout.spot_attempt_prefix(self.campaign, self.run_id, attempt)
            decision = _optional(self.store, f"{prefix}/outcome.json")
            if decision is not None:
                self.record(decision)
                if decision["status"] not in RETRYABLE or attempt == self.item["max_attempts"]:
                    return self.previous_result(decision)
                previous = decision
                continue
            grant = _optional(self.store, layout.spot_instance_key(self.campaign, self.run_id, attempt))
            if grant:
                return self.poll(attempt, prefix, grant)
            if self.store.head(self.cancel) is not None:
                return self.result(attempt, "CANCELLED", terminal=True)
            if self.store.head(_stop_key(self.campaign, self.owner)) is not None:
                return self.result(attempt, "EXECUTION_STOPPED", terminal=True)
            if self.store.head(self.final) is not None and self.published():
                return self.result(attempt, "SUCCEEDED", terminal=True)
            return self.provision(attempt, prefix, previous)
        raise AssertionError("max_attempts must be positive")

    def provision(self, attempt, prefix, previous):
        queue = _once(self.store, f"{prefix}/queue.json", {"requested_at": self.clock()})
        rounds_prefix = f"{prefix}/rounds/"
        requests = sorted(key for key in self.store.list(rounds_prefix) if key.endswith("-request.json"))
        if requests:
            round_number = int(requests[-1].rsplit("/", 1)[1].split("-", 1)[0])
            last_outcome = _optional(self.store, f"{rounds_prefix}{round_number:05d}-outcome.json")
            if last_outcome is not None:
                round_number += 1
            elapsed = self.clock() - queue["requested_at"]
            if last_outcome is not None and elapsed >= self.item["queue_timeout_seconds"]:
                return self.resolved(attempt, "CAPACITY_TIMEOUT", last_outcome["reason"])
            if elapsed >= self.item["queue_timeout_seconds"] + LAUNCH_LEASE_SECONDS:
                return self.result(
                    attempt, "LAUNCH_UNCONFIRMED", reason="EC2 request could not be confirmed", terminal=True
                )
        else:
            round_number = 1
        request_prefix = f"{rounds_prefix}{round_number:05d}"
        request = _once(self.store, f"{request_prefix}-request.json", {"execution_key": self.owner})
        requested_owner = request["execution_key"]
        if self.store.head(_stop_key(self.campaign, requested_owner)) is not None:
            outcome = {"reason": "previous workflow stopped before the launch was confirmed"}
            if self.store.head(_closed_key(self.campaign, requested_owner)) is not None:
                _once(self.store, f"{request_prefix}-outcome.json", outcome)
            return self.result(attempt, "WAITING_FOR_CLEANUP", **outcome)
        lock = f"{_launches_prefix(self.campaign, requested_owner)}{self.run_id}.json"
        try:
            with _launch_lease(self.store, lock, self.clock) as leased:
                if not leased:
                    return self.result(
                        attempt, "WAITING_FOR_LAUNCH", reason="waiting for the same EC2 request"
                    )
                return self.launch(attempt, round_number, request_prefix, requested_owner, previous)
        except (ConnectionError, HTTPClientError, ClientError) as error:
            if isinstance(error, ClientError) and error_code(error) not in RETRYABLE_LAUNCH_ERRORS:
                raise
            return self.result(
                attempt, "WAITING_FOR_LAUNCH", reason=error_code(error) or type(error).__name__
            )

    def launch(self, attempt, round_number, request_prefix, requested_owner, previous):
        if self.store.head(self.cancel) is not None:
            return self.result(attempt, "CANCELLED", terminal=True)
        if self.store.head(_stop_key(self.campaign, requested_owner)) is not None:
            return self.result(attempt, "WAITING_FOR_CLEANUP", reason="launching workflow stopped")
        grant_key = layout.spot_instance_key(self.campaign, self.run_id, attempt)
        existing = _optional(self.store, grant_key)
        if existing:
            return self.result(attempt, "LAUNCHED", instance_id=existing["instance_id"])
        previous_answer = _optional(self.store, f"{request_prefix}-outcome.json")
        if previous_answer:
            return self.result(attempt, "WAITING_FOR_SPOT", reason=previous_answer["reason"])
        launched = self.jobs.launch(
            {**self.item, "execution_key": requested_owner},
            self.campaign,
            attempt=attempt,
            provision_round=round_number,
        )
        instance = launched.get("instance_id")
        if not instance:
            reason = launched["reason"]
            _once(self.store, f"{request_prefix}-outcome.json", {"reason": reason})
            return self.result(attempt, "WAITING_FOR_SPOT", reason=reason)

        if launched.get("instance_type") not in (None, "c6a.large") or launched.get("lifecycle") not in (
            None,
            "spot",
        ):
            self.jobs.terminate(instance)
            raise ValueError(f"Unexpected instance configuration for {instance}")

        grant = _once(
            self.store,
            grant_key,
            {
                **launched,
                "instance_id": instance,
                "job_id": instance,
                "instance_lifecycle": launched.get("lifecycle"),
                "attempt": attempt,
                "attempt_key": layout.attempt_key_for(instance, attempt),
                "launch_at": launched.get("launch_at") or self.clock(),
                "previous_attempt_key": (previous or {}).get("attempt_key"),
                "recovery_reason": (previous or {}).get("status"),
                "execution_key": requested_owner,
            },
        )
        if grant["instance_id"] != instance:
            self.jobs.terminate(instance)
            raise RuntimeError(f"Conflicting instance launches for {self.run_id} attempt {attempt}")
        if (
            self.store.head(self.cancel) is not None
            or self.store.head(_stop_key(self.campaign, requested_owner)) is not None
        ):
            self.jobs.terminate(instance)
        return self.result(attempt, "LAUNCHED", instance_id=instance)

    def poll(self, attempt, prefix, grant):
        instance_id = grant["instance_id"]
        attempt_key = grant["attempt_key"]
        worker = (
            _optional(self.store, layout.attempt_status_key(self.campaign, self.run_id, attempt_key)) or {}
        )
        host = _optional(
            self.store, f"{layout.attempt_prefix(self.campaign, self.run_id, attempt_key)}/host-exit.json"
        )
        instance = self.jobs.describe(instance_id)
        state = instance["state"]
        # EC2 eventually clears BlockDeviceMappings on terminated instances. Save IDs when visible.
        volume_key = f"{prefix}/volumes.json"
        observed = _optional(self.store, volume_key)
        tagged = self.jobs.list_volumes(self.campaign, run_id=self.run_id, attempt=attempt)
        volumes = sorted(set(grant.get("volume_ids", []) + instance.get("volume_ids", []) + tagged))
        if volumes and observed is None:
            observed = _once(self.store, volume_key, {"volume_ids": volumes, "observed_at": self.clock()})
        cancelled = self.store.head(self.cancel) is not None
        stopped = self.store.head(_stop_key(self.campaign, grant["execution_key"])) is not None
        elapsed = self.clock() - grant["launch_at"]
        if state == "missing" and elapsed < VISIBILITY_SECONDS:
            return self.result(attempt, "WAITING_FOR_INSTANCE_VISIBILITY", instance_id=instance_id)
        boot_timed_out = not worker and not host and elapsed >= BOOT_TIMEOUT_SECONDS
        timed_out = elapsed >= self.item["run_timeout_seconds"]
        if state not in TERMINATED:
            if state != "shutting-down" and (
                host
                or cancelled
                or stopped
                or timed_out
                or boot_timed_out
                or state in {"stopped", "stopping"}
            ):
                # Remember the controller's intent before EC2 changes the reported reason code.
                cause = (
                    "CANCELLED"
                    if cancelled
                    else "INSTANCE_LOST"
                    if stopped
                    else "BOOTSTRAP_FAILED"
                    if boot_timed_out
                    else "RUN_TIMEOUT"
                    if timed_out
                    else None
                )
                if cause:
                    _once(self.store, f"{prefix}/terminate.json", {"cause": cause})
                self.jobs.terminate(instance_id)
                state = "shutting-down"
            return self.result(
                attempt, state.upper(), reason=(host or {}).get("reason"), instance_id=instance_id
            )

        volumes = sorted(set(volumes + (observed or {}).get("volume_ids", [])))
        if tagged or not self.jobs.volumes_deleted(volumes):
            return self.result(attempt, "WAITING_FOR_VOLUME_DELETION", instance_id=instance_id)

        notice_key = f"{layout.attempt_prefix(self.campaign, self.run_id, attempt_key)}/spot-notice.json"
        failure = self.classify(
            prefix, instance, worker, host or {}, cancelled, self.store.head(notice_key) is not None
        )
        status, reason = failure if failure else ("SUCCEEDED", "final publication and analytics verified")
        verified = {**grant, "volume_ids": volumes}
        if instance.get("instance_type"):
            verified["instance_type"] = instance["instance_type"]
        if instance.get("lifecycle"):
            verified["instance_lifecycle"] = instance["lifecycle"]
        return self.resolved(
            attempt,
            status,
            reason,
            grant=verified,
            instance=instance,
            worker=worker,
            host=host,
        )

    def classify(self, prefix, instance, worker, host, cancelled, notice):
        if self.published():
            return None
        worker_state, host_state = worker.get("state"), host.get("state")
        diagnostic = worker.get("diagnostic") or {}
        if cancelled or worker_state == "CANCELLED":
            return "CANCELLED", "campaign explicitly cancelled"
        if (
            worker_state == "INTERRUPTED"
            and worker.get("interruption_reason") == "spot_interruption_notice"
            or notice
            or host_state == "SPOT_NOTICE"
            or instance.get("reason_code") == "Server.SpotInstanceTermination"
        ):
            return "SPOT_INTERRUPTED", "EC2 Spot interruption"
        termination = _optional(self.store, f"{prefix}/terminate.json")
        if termination:
            return termination["cause"], "controller requested EC2 termination"
        if host_state == "BOOTSTRAP_FAILED":
            return "BOOTSTRAP_FAILED", host.get("reason") or "worker setup failed"
        if worker_state in {"TRACKING_FAILED", "PUBLICATION_FAILED", "REUSED_ANALYTICS_FAILED"}:
            return "PUBLICATION_FAILED", diagnostic.get("message") or "final publication incomplete"
        if worker_state == "FAILED" or host.get("exit_code") not in (None, 0, 75):
            return "BENCHMARK_FAILED", diagnostic.get("message") or host.get(
                "reason"
            ) or "worker returned an error"
        if host.get("exit_code") == 0:
            return "PUBLICATION_FAILED", "worker exited without a complete final manifest"
        return "INSTANCE_LOST", "EC2 ended without a confirmed worker outcome"


def reconcile_campaign(store, jobs, campaign, reason, *, owner=None, clock=time.time):
    active = jobs.list_active(campaign, execution_key=owner)
    terminated = []
    for instance in active:
        if instance["state"] not in ("shutting-down", "terminated"):
            if instance.get("run_id") and instance.get("attempt"):
                prefix = layout.spot_attempt_prefix(campaign, instance["run_id"], instance["attempt"])
                cause = "CANCELLED" if store.head(layout.cancelled_key(campaign)) else "INSTANCE_LOST"
                _once(store, f"{prefix}/terminate.json", {"cause": cause, "reason": reason})
            jobs.terminate(instance["instance_id"])
            terminated.append(instance["instance_id"])
    remaining = jobs.list_active(campaign, execution_key=owner)
    volumes = jobs.list_volumes(campaign, execution_key=owner)
    active_launches = (
        [
            key
            for key in store.list(_launches_prefix(campaign, owner))
            if (_optional(store, key) or {}).get("expires", 0) > clock()
        ]
        if owner
        else []
    )
    evidence = {
        "campaign_id": campaign,
        "reason": reason,
        "execution_key": owner,
        "terminated": terminated,
        "instances": {instance["instance_id"]: instance["state"] for instance in remaining},
        "active_instances": [instance["instance_id"] for instance in remaining],
        "remaining_volumes": volumes,
        "active_launches": active_launches,
        "clean": not remaining and not volumes and not active_launches,
        "checked_at": clock(),
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(clock()))
    put_json(
        store, layout.reconciliation_key(campaign, f"{stamp}-spot-{(owner or 'campaign')[:12]}"), evidence
    )
    if owner and evidence["clean"] and store.head(_stop_key(campaign, owner)) is not None:
        _once(store, _closed_key(campaign, owner), {"verified_at": evidence["checked_at"]})
    return evidence
