"""USD accounting for AWS attempts: estimate, CUR-observed cost and allocation, kept apart.

- *estimated*: reserved vCPU, memory and ephemeral storage × billed seconds × the versioned
  Price List snapshot (Fargate bills per second from image pull to task stop, minimum 60 s);
- *observed*: CUR 2.0 split cost allocation lines of the task (tag `cb-run-id`/`cb-campaign`);
- *allocated*: shared CI/CD, network, S3, CloudWatch, streaming and Feature Store lines in the
  period, split by each campaign task's share of reserved vCPU-seconds.

Scenario costs computed by the evaluator (`cost_*` in scenario units) are never summed here.
"""

import json
import math
from pathlib import Path

PRICE_SNAPSHOT = Path("configs/contracts/aws-prices.us-east-1.json")
SHARED_CATEGORIES = {
    "ci_cd": ("AWSCodePipeline", "CodeBuild", "AmazonECR"),
    "network": ("AmazonVPC",),
    "storage": ("AmazonS3",),
    "observability": ("AmazonCloudWatch", "AWSEvents", "AWSLambda", "AmazonSNS", "AWSStepFunctions"),
    "streaming": ("AmazonKinesis", "AmazonKinesisFirehose", "AmazonKinesisAnalytics"),
    "feature_store": ("AmazonSageMaker", "AWSGlue"),
    "analytics": ("AmazonAthena",),
}


def load_prices(path=PRICE_SNAPSHOT, project_root=".") -> dict:
    return json.loads((Path(project_root) / path).read_text())


def price(prices: dict, name: str) -> float:
    entry = prices["prices"][name]
    if "usd" not in entry:
        raise ValueError(f"price {name} is unavailable in snapshot {prices.get('retrieved')}")
    return float(entry["usd"])


def attempt_estimate(
    *,
    started_ms: int,
    stopped_ms: int,
    vcpus: float,
    memory_gib: float,
    ephemeral_gib: float,
    prices: dict,
    free_ephemeral_gib: float = 20,
) -> dict:
    if stopped_ms < started_ms:
        raise ValueError("attempt stopped before it started")
    seconds = (stopped_ms - started_ms) / 1000
    billed = max(60, math.ceil(seconds))
    hours = billed / 3600
    extra_storage = max(0.0, ephemeral_gib - free_ephemeral_gib)
    components = {
        "vcpu_usd": vcpus * hours * price(prices, "fargate_vcpu_hour"),
        "memory_usd": memory_gib * hours * price(prices, "fargate_memory_gb_hour"),
        "ephemeral_storage_usd": extra_storage * hours * price(prices, "fargate_ephemeral_storage_gb_hour"),
    }
    return {
        "kind": "estimated",
        "unit": "USD",
        "price_snapshot": prices.get("retrieved"),
        "reserved": {"vcpus": vcpus, "memory_gib": memory_gib, "ephemeral_gib": ephemeral_gib},
        "seconds": seconds,
        "billed_seconds": billed,
        "components": components,
        "total_usd": sum(components.values()),
        "scope": "Fargate task reservation only; excludes S3, logs, endpoints and shared services",
    }


def classify(product_code: str) -> str | None:
    for category, codes in SHARED_CATEGORIES.items():
        if product_code in codes:
            return category
    return None


def reconcile(
    campaign_id: str, attempts: list[dict], cur_rows: list[dict], *, period_vcpu_seconds: float
) -> dict:
    """Join estimates with CUR rows; shared lines are allocated by reserved vCPU-second share.

    `attempts` carry `run_id`, `estimate.total_usd` and `vcpu_seconds`. `cur_rows` carry
    `product_code`, `cost`, `split_cost` (task lines), `run_id`/`campaign` tag values.
    `period_vcpu_seconds` is every Fargate vCPU-second billed in the period (all campaigns).
    """
    campaign_vcpu_seconds = sum(a["vcpu_seconds"] for a in attempts)
    share = campaign_vcpu_seconds / period_vcpu_seconds if period_vcpu_seconds else 0.0
    per_run = {}
    for attempt in attempts:
        row = per_run.setdefault(
            attempt["run_id"],
            {"run_id": attempt["run_id"], "estimated_usd": 0.0, "observed_usd": 0.0, "attempts": 0},
        )
        row["estimated_usd"] += attempt["estimate"]["total_usd"]
        row["attempts"] += 1
    shared, unattributed = {}, 0.0
    for line in cur_rows:
        if line.get("campaign") == campaign_id and line.get("run_id") in per_run:
            per_run[line["run_id"]]["observed_usd"] += float(line.get("split_cost") or line.get("cost") or 0)
            continue
        category = classify(line.get("product_code", ""))
        if category is None:
            unattributed += float(line.get("cost") or 0)
        else:
            shared[category] = shared.get(category, 0.0) + float(line.get("cost") or 0)
    allocated = {category: total * share for category, total in shared.items()}
    observed = sum(r["observed_usd"] for r in per_run.values())
    return {
        "campaign_id": campaign_id,
        "unit": "USD",
        "runs": sorted(per_run.values(), key=lambda r: r["run_id"]),
        "estimated_usd": sum(r["estimated_usd"] for r in per_run.values()),
        "observed_task_usd": observed,
        "shared_period_usd": shared,
        "allocation_share": share,
        "allocated_shared_usd": allocated,
        "observed_plus_allocated_usd": observed + sum(allocated.values()),
        "unattributed_period_usd": unattributed,
        "method": "split cost allocation for tasks; shared services by reserved vCPU-second share",
    }
