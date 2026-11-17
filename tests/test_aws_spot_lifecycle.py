from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from botocore.exceptions import ReadTimeoutError

from continual_bench.application.cloud import layout, planning
from continual_bench.application.cloud.contracts import SPOT_BACKEND, CampaignRequest
from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.application.cloud.spot_lifecycle import (
    LAUNCH_LEASE_SECONDS,
    SpotRun,
    admit_execution,
    cancel_campaign,
    execution_key,
    reconcile_campaign,
    stop_execution,
)
from continual_bench.core.policies import ZIA_POLICIES
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json, put_json

CAMPAIGN = "spot-smoke"
WORKFLOW = "arn:aws:states:us-east-1:123456789012:execution:continual-bench-benchmark:spot-smoke"
RUN = "abcd" * 5


class FakeSpotJobs:
    def __init__(self):
        self.calls = []
        self.instances = {}
        self.volumes = set()
        self.wait_for_capacity = False
        self.launch_started = self.continue_launch = None
        self.lose_response = False

    def launch(self, item, campaign_id, *, attempt, provision_round):
        self.calls.append((attempt, provision_round))
        if self.launch_started:
            self.launch_started.set()
            assert self.continue_launch.wait(10)
        if self.wait_for_capacity:
            return {"reason": "InsufficientInstanceCapacity"}
        instance_id, volume_id = f"i-{attempt}", f"vol-{attempt}"
        instance = self.instances.setdefault(
            instance_id,
            {
                "instance_id": instance_id,
                "volume_ids": [volume_id],
                "state": "running",
                "availability_zone": "us-east-1a",
                "execution_key": item["execution_key"],
            },
        )
        self.volumes.add(volume_id)
        if self.lose_response:
            self.lose_response = False
            raise ReadTimeoutError(endpoint_url="https://ec2.us-east-1.amazonaws.com")
        return {**instance, "volume_ids": []}  # RunInstances can return before EBS attachment.

    def describe(self, instance_id):
        return self.instances[instance_id]

    def terminate(self, instance_id):
        self.instances[instance_id]["state"] = "shutting-down"

    def list_active(self, _, *, execution_key=None):
        return [
            instance
            for instance in self.instances.values()
            if instance["state"] not in ("terminated", "missing")
            and (execution_key is None or execution_key == instance["execution_key"])
        ]

    def list_volumes(self, _, *, execution_key=None, run_id=None, attempt=None):
        candidates = self.volumes if attempt is None else {f"vol-{attempt}"} & self.volumes
        return sorted(candidates)

    def volumes_deleted(self, volume_ids):
        return not self.volumes.intersection(volume_ids)


@pytest.fixture
def run(tmp_path):
    store = LocalObjectStore(tmp_path)
    jobs = FakeSpotJobs()
    item = {
        "campaign_id": CAMPAIGN,
        "run_id": RUN,
        "job_name": "job",
        "poll_seconds": 30,
        "max_attempts": 3,
        "queue_timeout_seconds": 600,
        "run_timeout_seconds": 900,
        "runtime_image": "example@sha256:test",
        "contract_sha256": "test",
    }
    now = [1_700_000_000]
    assert admit_execution(store, CAMPAIGN, WORKFLOW)
    return SpotRun(store, jobs, item, WORKFLOW, clock=lambda: now[0]), jobs, now


def terminal_worker(run, attempt, worker_state, *, host_state="WORKER_EXIT", code=1):
    instance_id = f"i-{attempt}"
    attempt_key = layout.attempt_key_for(instance_id, attempt)
    put_json(run.store, layout.attempt_status_key(CAMPAIGN, RUN, attempt_key), {"state": worker_state})
    host_key = f"{layout.attempt_prefix(CAMPAIGN, RUN, attempt_key)}/host-exit.json"
    put_json(run.store, host_key, {"state": host_state, "exit_code": code, "reason": worker_state})


def test_retry_waits_for_termination_and_volume_deletion(run):
    unit, jobs, _ = run
    jobs.wait_for_capacity = True
    assert unit.advance()["reason"] == "InsufficientInstanceCapacity"
    jobs.wait_for_capacity = False
    assert unit.advance()["instance_id"] == "i-1"
    assert unit.advance()["status"] == "RUNNING"
    terminal_worker(unit, 1, "INTERRUPTED", host_state="SPOT_NOTICE", code=75)
    assert unit.advance()["status"] == "SHUTTING-DOWN"
    jobs.instances["i-1"]["state"] = "terminated"
    jobs.instances["i-1"]["volume_ids"] = []
    assert unit.advance()["status"] == "WAITING_FOR_VOLUME_DELETION"
    assert jobs.calls == [(1, 1), (1, 2)]
    jobs.volumes.clear()
    first = unit.advance()
    assert first["status"] == "SPOT_INTERRUPTED" and not first["terminal"]
    first_attempt = layout.spot_attempt_prefix(CAMPAIGN, RUN, 1)
    outcome = get_json(unit.store, f"{first_attempt}/outcome.json")
    assert outcome["volume_ids"] == ["vol-1"] and outcome["volume_cleanup_verified"]
    assert unit.advance()["instance_id"] == "i-2"
    grant = get_json(unit.store, layout.spot_instance_key(CAMPAIGN, RUN, 2))
    assert grant["previous_attempt_key"] == "i-1-1" and grant["recovery_reason"] == "SPOT_INTERRUPTED"


def test_loss_and_failed_publication_are_retried_with_one_final_manifest(run):
    unit, jobs, now = run
    assert unit.advance()["instance_id"] == "i-1"
    jobs.instances["i-1"]["state"] = "missing"
    assert unit.advance()["status"] == "WAITING_FOR_INSTANCE_VISIBILITY"
    now[0] += 301
    jobs.volumes.clear()
    assert unit.advance()["status"] == "INSTANCE_LOST"
    assert unit.advance()["instance_id"] == "i-2"
    terminal_worker(unit, 2, "TRACKING_FAILED")
    jobs.instances["i-2"]["state"] = "terminated"
    jobs.volumes.clear()
    assert unit.advance()["status"] == "PUBLICATION_FAILED"
    assert unit.advance()["instance_id"] == "i-3"
    terminal_worker(unit, 3, "SUCCEEDED", code=0)
    report = put_json(unit.store, "report.json", {"metrics": {}})
    artifact = put_json(unit.store, "artifacts/data/index.json", {"files": []})
    put_json(
        unit.store,
        unit.final,
        {
            "contract_sha256": "test",
            "campaign_id": CAMPAIGN,
            "run_id": RUN,
            "report_uri": report.uri,
            "report_sha256": report.sha256,
            "artifacts_index_uri": artifact.uri,
        },
    )
    put_json(unit.store, layout.analytics_marker_key(CAMPAIGN, RUN), {"attempt_key": "i-3-3"})
    jobs.instances["i-3"]["state"] = "terminated"
    jobs.volumes.clear()
    done = unit.advance()
    assert done["status"] == "SUCCEEDED" and done["terminal"]
    assert unit.advance() == done
    assert len(list(unit.store.list("analytics/attempts/"))) == 3


def test_lost_launch_response_and_concurrent_delivery_keep_one_instance(run):
    unit, jobs, now = run
    jobs.lose_response = True
    assert unit.advance()["status"] == "WAITING_FOR_LAUNCH"
    assert unit.advance()["status"] == "WAITING_FOR_LAUNCH"
    now[0] += LAUNCH_LEASE_SECONDS + 1
    jobs.launch_started, jobs.continue_launch = Event(), Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        replay = pool.submit(unit.advance)
        assert jobs.launch_started.wait(10)
        assert pool.submit(unit.advance).result(timeout=10)["status"] == "WAITING_FOR_LAUNCH"
        jobs.continue_launch.set()
        assert replay.result(timeout=10)["instance_id"] == "i-1"
    assert jobs.calls == [(1, 1), (1, 1)]
    assert list(jobs.instances) == ["i-1"]


def test_cancel_fences_new_attempts_and_cleans_boot_failures(run):
    unit, jobs, _ = run
    assert not admit_execution(unit.store, CAMPAIGN, WORKFLOW + "--other")
    unit.advance()
    terminal_worker(unit, 1, "BOOTSTRAP_FAILED", host_state="BOOTSTRAP_FAILED")
    assert unit.advance()["status"] == "SHUTTING-DOWN"
    jobs.instances["i-1"]["state"] = "terminated"
    jobs.volumes.clear()
    assert unit.advance()["status"] == "BOOTSTRAP_FAILED"
    cancel_campaign(unit.store, CAMPAIGN)
    owner = stop_execution(unit.store, CAMPAIGN, WORKFLOW, "operator cancellation")
    assert owner == execution_key(WORKFLOW)
    assert unit.advance()["status"] == "CANCELLED"
    assert jobs.calls == [(1, 1)]
    cleanup = reconcile_campaign(unit.store, jobs, CAMPAIGN, "operator cancellation", owner=owner)
    assert cleanup["clean"] and cleanup["remaining_volumes"] == []


def test_spot_plan_fans_out_every_policy_with_one_digest(tmp_path):
    store = LocalObjectStore(tmp_path / "data")
    snapshot = publish_fixture_snapshot(store, tmp_path)
    release = publish_fixture_release(store).model_copy(update={"benchmark_compute": SPOT_BACKEND})
    request = CampaignRequest(
        campaign_id=CAMPAIGN,
        snapshot=snapshot.model_dump(),
        matrix={"kind": "zia", "policies": ZIA_POLICIES},
    )
    plan = planning.build_plan(request, release, snapshot_manifest=verify_snapshot(store, snapshot))
    items = planning.map_items(plan, store)
    assert plan["concurrency"] == len(items) == 12
    assert plan["mode"] == "distributed"
    assert {item["runtime_image"] for item in items} == {release.images["runtime"].uri}
    assert [item["placement_index"] for item in items] == list(range(12))
