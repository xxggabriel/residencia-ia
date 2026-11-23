"""Retain and publish per-seed policy analyses through the existing benchmark tracker."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from continual_bench.infrastructure.metric_tracking import VERSION
from continual_bench.infrastructure.storage import atomic_write, run_lock

from .environment import encoded


def repetition_seeds(values):
    """Keep integer repetition counts compatible while accepting explicit unique seed identities."""
    if type(values) is int:
        values = list(range(values))
    else:
        try:
            values = list(values)
        except TypeError:
            raise ValueError("seeds must be a repetition count or a sequence of integer seeds") from None
    if (
        not values
        or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("seeds must be unique integers in [0, 2**31), with at least one repetition")
    return values


def _scalar_items(values, prefix=""):
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            yield from _scalar_items(value, name)
        elif value is None or type(value) in (int, float):
            if value is None or math.isfinite(value):
                yield name, value


def policy_tracking(analysis):
    """Preserve alpha/kappa coordinates, all scalar policy metrics and decision schedules."""
    is_upf = "rules" in analysis
    axis_name = "alpha" if is_upf else "kappa"
    grid = analysis["alphas" if is_upf else "kappa"]
    rows = (
        {rule: info["curve"] for rule, info in analysis["rules"].items()}
        if is_upf
        else analysis["curves_by_policy"]
    )
    summary = (
        {
            rule: {key: value for key, value in info.items() if key != "curve"}
            for rule, info in analysis["rules"].items()
        }
        if is_upf
        else analysis["summary"]
    )
    metrics = dict(_scalar_items(summary))
    series = {f"axis/{axis_name}": [{"step": step, "value": value} for step, value in enumerate(grid)]}
    availability = {}
    for rule, curve in rows.items():
        for step, row in enumerate(curve):
            for key, value in _scalar_items(row):
                series.setdefault(f"policy/{rule}/{key}", []).append({"step": step, "value": value})
                if value is None:
                    availability[f"policy/{rule}/{key}"] = "undefined_for_this_protocol_or_reference"
            schedule = row.get("retrain_points", [])
            if is_upf:
                start, end = analysis["setup"]["offline_batches"], analysis["setup"]["batches"]
            else:
                start, end = 0, analysis["online_batches"]
            series[f"policy/{rule}/schedule/{axis_name}_{step}"] = [
                {"step": batch, "value": int(batch in schedule)} for batch in range(start, end)
            ]
            if "served_by" in row:
                series[f"policy/{rule}/served_by/{axis_name}_{step}"] = [
                    {"step": batch, "value": model} for batch, model in enumerate(row["served_by"])
                ]
    return {
        "version": VERSION,
        "metrics": metrics,
        "series": series,
        "availability": availability,
        "settings": {
            "axis": axis_name,
            "step": "index_into_exact_cost_grid_except_explicit_schedule_series",
            "max_points_per_key": None,
            "schedule": "batch_index_within_protocol_phase",
        },
    }


def policy_report(name, seed, *, protocol, analysis, candidate_reports, settings, artifact_mode="full"):
    """Publish after policy evaluation; no candidate state or model bytes are accepted here."""
    tracking = policy_tracking(analysis)
    contract = {
        "dataset": name,
        "seed": seed,
        "protocol": protocol,
        "candidate_run_keys": [report["run_key"] for report in candidate_reports],
        "environment": candidate_reports[0]["contract"]["environment"],
        "settings": settings,
        "artifact_mode": artifact_mode,
    }
    if candidate_model := candidate_reports[0]["contract"].get("model"):
        contract["model"] = candidate_model
    run_key = hashlib.sha256(encoded(contract)).hexdigest()[:24]
    report = {
        "run_key": run_key,
        "complete": True,
        "dataset": name,
        "seed": seed,
        "artifact_mode": artifact_mode,
        "contract": contract,
        "metrics": tracking["metrics"],
        "tracking": tracking,
        "analysis": {key: value for key, value in analysis.items() if key != "candidate_tracking"},
    }
    receipt = None
    if "MLFLOW_TRACKING_URI" in os.environ:
        root = Path(os.environ.get("BENCH_STATE", "state")) / "retraining-policy-results"
        with run_lock(root, run_key):
            directory = root / run_key
            path = directory / "report.json"
            if path.exists():
                retained = json.loads(path.read_bytes())
                if retained.get("contract") != contract:
                    raise ValueError("retained policy report has a different execution contract")
                report = retained
            else:
                atomic_write(path, encoded(report))
            from continual_bench.infrastructure.benchmark_tracking import publish_local_benchmark

            try:
                receipt = publish_local_benchmark(
                    directory, report, experiment_name="continual-bench/retraining-policy-results"
                )
                if artifact_mode == "metrics_only" and receipt.get("complete") is not True:
                    raise RuntimeError("policy publication receipt is incomplete")
            except Exception as error:
                receipt = {"complete": False, "error_type": type(error).__name__}
                if artifact_mode == "metrics_only":
                    raise RuntimeError(
                        f"policy publication incomplete ({type(error).__name__}); retry retained report"
                    ) from None
    return {
        "run_key": run_key,
        "contract": contract,
        "recorded_metric_points": sum(len(points) for points in tracking["series"].values())
        + len(tracking["metrics"]),
        "mlflow": receipt,
    }
