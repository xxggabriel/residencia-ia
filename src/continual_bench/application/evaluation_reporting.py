"""Semana 09 report assembly; domain evaluation remains free of I/O dependencies."""

import json
from pathlib import Path

from continual_bench.domain.evaluation_metrics import _integer, task_matrix_metrics


def validated_detection(report, tolerance):
    """Validate annotations then reuse the existing censored one-to-one matcher."""
    from continual_bench.application.tracking import detector_metrics

    _integer(tolerance, "tolerance", 1)
    observed = _integer(report["observed"], "observed")
    n = _integer(report["metrics"]["n"], "evaluated samples")
    if n > observed:
        raise ValueError("evaluated sample count exceeds observations")
    points = report["dataset"].get("changepoints")
    if points is not None:
        for point in points:
            _integer(point, "changepoint")
        if list(points) != sorted(set(points)):
            raise ValueError("changepoints must be unique and increasing")
        if any(point >= observed for point in points):
            raise ValueError("changepoints must be within the observed source")
    seen = set()
    for alarm in report["alarms"]:
        seq = _integer(alarm["seq"], "alarm seq")
        if seq >= observed or seq in seen:
            raise ValueError("alarm offsets must be observed and unique")
        seen.add(seq)
    values, details, reason = detector_metrics(report, {"detection_window": tolerance})
    return {"metrics": values, "details": details, "reason": reason}


def supplemental_report(report, directory=None, *, near_future=None):
    """Build/write an independently versioned supplement to an existing run report."""
    settings = report.get("tracking", {}).get("settings", {})
    continual = report.get("tracking", {}).get("continual") or {}
    rows = continual.get("matrix", [])
    matrix = [[row["accuracies"].get(str(task)) for task in range(len(rows[0]["accuracies"]))]
              for row in rows] if rows else []
    candidates = report.get("candidates", report.get("updates", []))
    retrains = [row for row in candidates if row["action"] == "reset_retrain"]
    rejected = [row for row in candidates if not row.get("promoted", True)]
    candidate_resources = {
        "training_actions": len(candidates),
        "candidate_retrains": len(retrains),
        "promoted_actions": sum(row.get("promoted", True) for row in candidates),
        "rejected_actions": len(rejected),
        "train_samples": sum(row["train_samples"] for row in candidates),
        "training_seconds": sum(row["seconds"] for row in candidates),
        "rejected_training_seconds": sum(row["seconds"] for row in rejected),
        "retrain_duration_seconds": [row["seconds"] for row in retrains],
        "scope": "postwarmup training includes rejected candidates; promotion is a separate decision",
    }
    supplement = {
        "version": "semana09-metrics-v1",
        "detection": validated_detection(report, settings.get("detection_window", 500)),
        "task_matrix": task_matrix_metrics(matrix),
        "near_future": near_future.report() if near_future is not None else {
            "near_future_accuracy": None, "reason": "frozen_future_predictions_not_captured"},
        "resources": {
            "source": "residency-metrics-v1",
            "metrics": {name: value
                        for name, value in report.get("tracking", {}).get("metrics", {}).items()
                        if name in {"train_seconds", "prediction_seconds", "process_peak_rss_bytes",
                                    "sampled_rss_byte_seconds", "retrain_count",
                                    "update_count", "cost_total"}},
            "candidates": candidate_resources,
            "unit": settings.get("costs", {}).get("unit") if settings.get("costs") else None,
            "interpretation": "scenario cost is not measured currency, energy or carbon",
        },
    }
    if directory is not None:
        from continual_bench.adapters.local import atomic_write

        atomic_write(Path(directory) / "task_evaluation-metrics.json",
                     json.dumps(supplement, sort_keys=True, indent=2, allow_nan=False).encode())
    return supplement
