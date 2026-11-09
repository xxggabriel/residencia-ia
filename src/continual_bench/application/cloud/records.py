"""Athena-friendly JSON Lines records derived from completed reports (one row per fact)."""

import json
import math

from . import layout

ROWS_FILE = "analytics-rows.json"


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def jsonl(rows) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()


def zia_rows(base: dict, report: dict, final_uri: str, outputs_uri: str) -> dict[str, list[dict]]:
    run = {
        **base,
        "model": report["model"],
        "protocol": report["protocol"],
        "rows": report["metrics"]["dataset_rows"],
        "holdout_rows": report["holdout"]["n"],
        "holdout_start": report["contract"]["holdout_start"],
        "block_rows": report["contract"]["block_rows"],
        "dataset_sha256": report["contract"]["input"]["sha256"],
        "final_uri": final_uri,
        "outputs_uri": outputs_uri,
        "report_uri": f"{outputs_uri}/report.json",
        "predictions_uri": f"{outputs_uri}/predictions/",
        "model_versions_count": len(report["model_versions"]),
    }
    metrics = [
        {**base, "partition": "prequential", "metric": k, "value": float(v), "step": 0}
        for k, v in report["metrics"].items()
        if _numeric(v)
    ]
    metrics += [
        {**base, "partition": "holdout", "metric": k, "value": float(v), "step": 0}
        for k, v in report["holdout"].items()
        if _numeric(v)
    ]
    for group, partition in (("class_metrics", "prequential"), ("holdout_class_metrics", "holdout")):
        metrics += [
            {**base, "partition": f"{partition}_class:{label}", "metric": k, "value": float(v), "step": 0}
            for label, row in report[group].items()
            for k, v in row.items()
            if _numeric(v)
        ]
    windows = [
        {
            **base,
            "partition": w["scope"],
            "step": w["end_seq"],
            "end_seq": w["end_seq"],
            "start_time_ms": w["start_time_ms"],
            "end_time_ms": w["end_time_ms"],
            "metric": k,
            "value": float(v),
        }
        for w in report["windows"]
        for k, v in w.items()
        if k not in {"scope", "end_seq", "start_time_ms", "end_time_ms"} and _numeric(v)
    ]
    versions = [
        {
            **base,
            "version": index,
            **{
                k: v[k]
                for k in (
                    "end_seq",
                    "start_seq",
                    "action",
                    "sha256",
                    "bytes",
                    "train_samples",
                    "train_seconds",
                    "iterations",
                )
            },
            "uri": f"{outputs_uri}/{v['file']}",
        }
        for index, v in enumerate(report["model_versions"])
    ]
    return {"runs": [run], "metrics": metrics, "windows": windows, "model_versions": versions}


def matrix_rows(base: dict, headline: dict, report: dict, final_uri: str, outputs_uri: str) -> dict:
    run = {
        **base,
        "model": base.get("model"),
        "protocol": report.get("protocol"),
        "final_uri": final_uri,
        "outputs_uri": outputs_uri,
        "run_key": report.get("run_key"),
    }
    metrics = [
        {**base, "partition": "final", "metric": k, "value": float(v), "step": 0}
        for k, v in headline.items()
        if _numeric(v)
    ]
    windows = [
        {
            **base,
            "partition": "prequential",
            "step": w["end_seq"],
            "end_seq": w["end_seq"],
            "start_time_ms": None,
            "end_time_ms": None,
            "metric": k,
            "value": float(v),
        }
        for w in report.get("windows", [])
        if "end_seq" in w
        for k, v in w.items()
        if k != "end_seq" and _numeric(v)
    ]
    return {"runs": [run], "metrics": metrics, "windows": windows, "model_versions": []}


def publish_rows(store, campaign_id, run_id, tables: dict[str, list[dict]]):
    written = {}
    for table, rows in tables.items():
        key = layout.analytics_key(table, campaign_id, run_id)
        store.put_bytes(key, jsonl(rows), content_type="application/x-ndjson")
        written[table] = store.uri(key)
    return written


def put_row_once(store, key, row) -> bool:
    """Single-line JSON row (Athena JSON SerDe), created once; identical retries are accepted."""
    from continual_bench.infrastructure.object_store import ObjectExists

    data = jsonl([row])
    try:
        store.put_bytes(key, data, if_absent=True, content_type="application/x-ndjson")
        return True
    except ObjectExists:
        if store.get_bytes(key) != data:
            raise
        return False


def ensure_rows(store, final: dict) -> str:
    """Publish the winning attempt's Athena rows unless its completion marker exists.

    Rows are archived with the attempt evidence (`analytics-rows.json`) before the final
    manifest is created, so an interruption between finalization and publication is repaired
    by any later reader of the final manifest (a reused job or the campaign consolidation).
    """
    from continual_bench.infrastructure.object_store import put_json

    campaign, run_id = final["campaign_id"], final["run_id"]
    marker = layout.analytics_marker_key(campaign, run_id)
    if store.head(marker) is not None:
        return "present"
    prefix = (final.get("published") or {}).get("prefix")
    if not prefix:
        return "unavailable"
    tables = json.loads(store.get_bytes(f"{prefix}/{ROWS_FILE}"))
    written = publish_rows(store, campaign, run_id, tables)
    put_json(store, marker, {"attempt_key": final["attempt"]["attempt_key"], "tables": written})
    return "repaired"
