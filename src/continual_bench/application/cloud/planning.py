"""Campaign planning: resolve the release and snapshot, expand the matrix, publish the plan first.

Each matrix combination becomes an independent `RunUnit` with a deterministic `run_id`
derived from the campaign and its execution contract (protocol, parameters, snapshot,
release image and job definition, tracking profile). The plan is written once; re-planning
the same campaign must reproduce it byte for byte, otherwise the request is refused.
"""

import hashlib
from pathlib import Path

from continual_bench.infrastructure.object_store import ObjectExists, get_json, put_json, put_json_once

from . import layout
from .contracts import (
    EVALUATION_PROTOCOL,
    OFFLINE_MATRIX_FAMILIES,
    CampaignRequest,
    ReleaseManifest,
    RunUnit,
    run_identity,
)

PLAN_VERSION = "campaign-plan-v1"
INLINE_LIMIT = 40  # Larger fan-outs use a Distributed Map reading items from S3.


class CampaignConflict(ValueError):
    """The campaign already has a different plan (another release, snapshot or matrix)."""


class CapacityError(ValueError):
    """Quota, compute environment or queue cannot run the requested parallelism."""


def resolve_release(store, release: str) -> ReleaseManifest:
    if release == "latest":
        pointer = get_json(store, layout.LATEST_RELEASE)
        release = pointer["git_sha"]
    return ReleaseManifest(**get_json(store, layout.release_key(release)))


def file_sha256(project_root, relative) -> str:
    path = Path(project_root) / relative
    if not path.resolve().is_relative_to(Path(project_root).resolve()):
        raise ValueError("configuration path escapes the project root")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit_capacity(concurrency: int, vcpus_per_job: float, *, quota_vcpus: float, queue: dict) -> dict:
    """Fail the preflight when the regional quota or the queue cannot hold the parallel tasks."""
    required = concurrency * vcpus_per_job
    evidence = {
        "concurrency": concurrency,
        "vcpus_per_job": vcpus_per_job,
        "required_vcpus": required,
        "fargate_vcpu_quota": quota_vcpus,
        "queue": queue,
    }
    if queue.get("queue_state") != "ENABLED" or queue.get("queue_status") != "VALID":
        raise CapacityError(f"job queue not usable: {queue.get('queue_state')}/{queue.get('queue_status')}")
    if required > queue["max_vcpus"]:
        raise CapacityError(f"{required} vCPUs exceed the compute environment limit {queue['max_vcpus']}")
    if required > quota_vcpus:
        raise CapacityError(f"{required} vCPUs exceed the regional Fargate quota {quota_vcpus}")
    evidence["max_parallel_admitted"] = int(min(queue["max_vcpus"], quota_vcpus) // vcpus_per_job)
    return evidence


def _zia_units(request, release, snapshot_manifest, tracking_sha256):
    snapshot = request.snapshot
    for policy in request.matrix.policies:
        params = {
            "dataset": snapshot.dataset,
            "scope": snapshot.scope,
            "policy": policy,
            "tracking": request.tracking,
            "snapshot": snapshot.model_dump(),
        }
        yield (
            "zia",
            params,
            {"policy": policy, "dataset": snapshot.dataset, "scope": snapshot.scope},
            {
                "snapshot": {
                    "id": snapshot.snapshot_id,
                    "manifest_sha256": snapshot_manifest["manifest_sha256"],
                    "prepared_sha256": snapshot_manifest["prepared"]["sha256"],
                },
            },
        )


def _matrix_units(request, project_root):
    from continual_bench.application.matrix import ExperimentMatrix, case_key, describe
    from continual_bench.config import Config, EpochConfig, TypedConfig, load_model
    from continual_bench.datasets.catalog import CATALOG

    matrix = load_model(ExperimentMatrix, str(Path(project_root) / request.matrix.path))
    plan = matrix.plan()
    for config in plan["cases"]:
        coordinates = describe(config)
        family = CATALOG.get(coordinates["dataset"]).family
        if family not in OFFLINE_MATRIX_FAMILIES:
            raise ValueError(
                f"dataset {coordinates['dataset']} needs network acquisition; private workers "
                f"accept only offline families {OFFLINE_MATRIX_FAMILIES}"
            )
        evaluator = {Config: "binary", TypedConfig: "typed", EpochConfig: "epochs"}[type(config)]
        params = {
            "evaluator": evaluator,
            "config": config.model_dump(mode="json"),
            "case_key": case_key(config),
            "matrix": plan["matrix"],
        }
        labels = {k: str(v) for k, v in coordinates.items() if v is not None}
        yield "matrix_case", params, labels, {"matrix": plan["matrix"]}


def build_plan(
    request: CampaignRequest, release: ReleaseManifest, *, snapshot_manifest=None, project_root="."
) -> dict:
    tracking_sha256 = file_sha256(project_root, request.tracking)
    if request.matrix.kind == "zia":
        expansions = _zia_units(request, release, snapshot_manifest, tracking_sha256)
    else:
        expansions = _matrix_units(request, project_root)
    units = []
    for kind, params, labels, extra in expansions:
        contract = {
            "protocol": EVALUATION_PROTOCOL if kind == "zia" else f"matrix-{params['evaluator']}",
            "kind": kind,
            "params": params,
            "tracking_sha256": tracking_sha256,
            "release": {
                "git_sha": release.git_sha,
                "runtime_image": release.images["runtime"].digest,
                "job_definition": release.job_definitions["benchmark"],
            },
            **extra,
        }
        run_id, contract_sha256 = run_identity(request.campaign_id, contract)
        units.append(
            RunUnit(
                run_id=run_id,
                campaign_id=request.campaign_id,
                kind=kind,
                params=params,
                contract=contract,
                contract_sha256=contract_sha256,
                job_name=f"cb-{request.campaign_id}-{run_id}",
                unit_key=layout.unit_key(request.campaign_id, run_id),
                labels=labels,
            )
        )
    if len({u.run_id for u in units}) != len(units):
        raise ValueError("matrix produced duplicate execution contracts")
    if not units:
        raise ValueError("campaign matrix expanded to no runnable units")
    concurrency = min(request.max_parallel, len(units))
    return {
        "version": PLAN_VERSION,
        "campaign_id": request.campaign_id,
        "request": request.model_dump(mode="json"),
        "release": release.model_dump(mode="json"),
        "snapshot": (
            {
                "id": request.snapshot.snapshot_id,
                "manifest_uri": snapshot_manifest["manifest_uri"],
                "manifest_sha256": snapshot_manifest["manifest_sha256"],
            }
            if request.snapshot
            else None
        ),
        "units": [u.model_dump(mode="json") for u in units],
        "concurrency": concurrency,
        "mode": "inline" if len(units) <= INLINE_LIMIT else "distributed",
        "timeouts": {
            "queue_seconds": request.queue_timeout_seconds,
            "run_seconds": request.run_timeout_seconds,
            "task_seconds": request.queue_timeout_seconds + request.run_timeout_seconds + 300,
        },
        "max_attempts": request.max_attempts,
    }


def map_items(plan: dict, store) -> list[dict]:
    """Only URIs and identifiers travel through Step Functions state; documents stay in S3."""
    return [
        {
            "campaign_id": plan["campaign_id"],
            "run_id": u["run_id"],
            "job_name": u["job_name"],
            "unit_uri": store.uri(u["unit_key"]),
            "job_definition": plan["release"]["job_definitions"]["benchmark"],
            "max_attempts": plan["max_attempts"],
            "task_timeout_seconds": plan["timeouts"]["task_seconds"],
            "labels": u["labels"],
        }
        for u in plan["units"]
    ]


def publish_plan(store, plan: dict) -> dict:
    """Write request, units and plan once; an identical re-plan is accepted, a different one refused."""
    campaign = plan["campaign_id"]
    try:
        put_json_once(store, layout.plan_key(campaign), plan)
    except ObjectExists as error:
        existing = get_json(store, layout.plan_key(campaign))
        changed = sorted(k for k in plan if existing.get(k) != plan[k])
        raise CampaignConflict(
            f"campaign {campaign} was planned with different {changed}; use a new campaign_id"
        ) from error
    put_json_once(store, layout.request_key(campaign), plan["request"])
    for unit in plan["units"]:
        put_json_once(store, unit["unit_key"], unit)
    items = map_items(plan, store)
    put_json(store, layout.plan_items_key(campaign), items)
    return {
        "plan_uri": store.uri(layout.plan_key(campaign)),
        "items_uri": store.uri(layout.plan_items_key(campaign)),
        "items_key": layout.plan_items_key(campaign),
        "items": items if plan["mode"] == "inline" else [],
        "mode": plan["mode"],
        "concurrency": plan["concurrency"],
        "campaign_id": campaign,
        "release": plan["release"]["git_sha"],
        "units": len(items),
    }
