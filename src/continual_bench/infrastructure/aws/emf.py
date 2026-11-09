"""CloudWatch Embedded Metric Format: metrics travel inside log lines, no PutMetricData call.

Dimensions are deliberately low cardinality (dataset, scope, policy, outcome). Run, job and
campaign identifiers are log properties for search, never metric dimensions.
"""

import json
import sys
import time

NAMESPACE = "ContinualBench"
ALLOWED_DIMENSIONS = {"Dataset", "Scope", "Policy", "Outcome", "Stage"}
HIGH_CARDINALITY = {"run_id", "job_id", "campaign_id", "attempt", "task_arn"}


def emf_record(
    metrics: dict,
    dimensions: dict,
    properties: dict | None = None,
    *,
    namespace=NAMESPACE,
    units: dict | None = None,
    timestamp_ms: int | None = None,
) -> dict:
    unknown = set(dimensions) - ALLOWED_DIMENSIONS
    if unknown:
        raise ValueError(f"metric dimensions must be low cardinality; refused: {sorted(unknown)}")
    clean = {k: float(v) for k, v in metrics.items() if v is not None}
    record = {
        "_aws": {
            "Timestamp": timestamp_ms or int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [sorted(dimensions)],
                    "Metrics": [{"Name": k, "Unit": (units or {}).get(k, "None")} for k in sorted(clean)],
                }
            ],
        },
        **{k: str(v) for k, v in dimensions.items()},
        **clean,
    }
    for key, value in (properties or {}).items():
        if key in record:
            raise ValueError(f"property collides with metric or dimension: {key}")
        record[key] = value
    return record


def emit(metrics: dict, dimensions: dict, properties: dict | None = None, *, stream=None, **options) -> dict:
    record = emf_record(metrics, dimensions, properties, **options)
    print(json.dumps(record, sort_keys=True, allow_nan=False), file=stream or sys.stdout, flush=True)
    return record
