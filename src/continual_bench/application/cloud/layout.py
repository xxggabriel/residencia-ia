"""S3 key layout: raw, prepared and manifests per dataset/revision/preparation; campaigns per run.

Every key is derived here so producers, verifiers, IAM prefixes and Athena locations agree.
"""

DATASETS = "datasets"
RELEASES = "releases"
CAMPAIGNS = "campaigns"
ANALYTICS = "analytics"
ANALYTICS_TABLES = ("runs", "metrics", "windows", "model_versions", "attempts", "campaigns", "cost_estimates")


def dataset_root(dataset, revision):
    return f"{DATASETS}/{dataset}/{revision}"


def raw_key(dataset, revision, filename):
    return f"{dataset_root(dataset, revision)}/raw/{filename}"


def raw_manifest_key(dataset, revision):
    return f"{dataset_root(dataset, revision)}/raw/_manifest.json"


def matrix_source_key(dataset, sha256):
    return f"datasets/matrix/{dataset}/{sha256}/source"


def prepared_key(snapshot, suffix):
    """Parquet sits alone under `<scope>/` so Glue/Athena tables read no JSON sidecars."""
    base = f"{dataset_root(snapshot.dataset, snapshot.revision)}/prepared/{snapshot.preparation_version}"
    if suffix == "parquet":
        return f"{base}/{snapshot.scope}/data.parquet"
    return f"{base}/{snapshot.scope}.{suffix}"


def snapshot_prefix(snapshot):
    return (
        f"{dataset_root(snapshot.dataset, snapshot.revision)}/manifests/"
        f"{snapshot.preparation_version}/{snapshot.scope}"
    )


def snapshot_manifest_key(snapshot):
    return f"{snapshot_prefix(snapshot)}/snapshot.json"


def snapshot_marker_key(snapshot):
    return f"{snapshot_prefix(snapshot)}/_SUCCESS.json"


def release_key(git_sha):
    return f"{RELEASES}/{git_sha}/release.json"


LATEST_RELEASE = f"{RELEASES}/latest.json"


def campaign_prefix(campaign_id):
    return f"{CAMPAIGNS}/{campaign_id}"


def request_key(campaign_id):
    return f"{campaign_prefix(campaign_id)}/request.json"


def plan_key(campaign_id):
    return f"{campaign_prefix(campaign_id)}/plan.json"


def plan_items_key(campaign_id):
    return f"{campaign_prefix(campaign_id)}/plan-items.json"


def unit_key(campaign_id, run_id):
    return f"{campaign_prefix(campaign_id)}/units/{run_id}.json"


def run_prefix(campaign_id, run_id):
    return f"{campaign_prefix(campaign_id)}/runs/{run_id}"


def attempt_prefix(campaign_id, run_id, attempt_key):
    return f"{run_prefix(campaign_id, run_id)}/attempts/{attempt_key}"


def attempt_status_key(campaign_id, run_id, attempt_key):
    return f"{attempt_prefix(campaign_id, run_id, attempt_key)}/status.json"


def final_key(campaign_id, run_id):
    return f"{run_prefix(campaign_id, run_id)}/final.json"


def analytics_marker_key(campaign_id, run_id):
    """Written after the winning attempt's Athena rows are published; absent means repair needed."""
    return f"{run_prefix(campaign_id, run_id)}/analytics.json"


def event_key(campaign_id, run_id, stamp, job_id, status):
    return f"{run_prefix(campaign_id, run_id)}/events/{stamp}-{job_id}-{status}.json"


def summary_key(campaign_id):
    return f"{campaign_prefix(campaign_id)}/summary.json"


def reconciliation_key(campaign_id, stamp):
    return f"{campaign_prefix(campaign_id)}/reconciliation/{stamp}.json"


def analytics_key(table, *parts):
    if table not in ANALYTICS_TABLES:
        raise ValueError(f"unknown analytics table: {table}")
    # Flat keys: Athena tables read one prefix without partition discovery.
    return f"{ANALYTICS}/{table}/{'__'.join(parts)}.json"


def attempt_key_for(job_id: str | None, attempt: int | str | None) -> str:
    """Batch job ID + attempt number; local runs use a deterministic placeholder."""
    return f"{job_id or 'local'}-{attempt if attempt is not None else 0}"
