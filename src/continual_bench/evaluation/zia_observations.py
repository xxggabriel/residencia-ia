"""Comparable, bounded diagnostics on Zia's never-trained temporal holdout."""

import time
from datetime import datetime, timezone

import numpy as np

from continual_bench.evaluation.tracking_observations import observations, process_peak_rss_bytes
from continual_bench.evaluation.zia_training_history import unique_train_rows
from continual_bench.models.classification_metrics import decide

PERIOD_MS = 30 * 24 * 60 * 60 * 1000
MAX_PERIODS = 12
MAX_PERIOD_ROWS = 256
MAX_CHECKPOINTS = 16


class TemporalHoldoutProbes:
    """Retain the first bounded prefix from each fixed UTC period in the holdout."""

    def __init__(self):
        self.data = {}

    def add(self, start, timestamps, features, labels):
        offset = 0
        while offset < len(timestamps):
            period = int(timestamps[offset]) // PERIOD_MS * PERIOD_MS
            if period not in self.data:
                if len(self.data) >= MAX_PERIODS:
                    break
                self.data[period] = (start + offset, [], [], 0)
            end = int(np.searchsorted(timestamps, period + PERIOD_MS, side="left"))
            first, x_parts, y_parts, count = self.data[period]
            take = min(end - offset, MAX_PERIOD_ROWS - count)
            if take:
                x_parts.append(features[offset:offset + take].copy())
                y_parts.append(labels[offset:offset + take].copy())
                self.data[period] = (first, x_parts, y_parts, count + take)
            offset = end

    def evaluate(self, model, output, versions, threshold):
        periods, parts = [], []
        for start_ms, (first, x_parts, y_parts, count) in self.data.items():
            label = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date().isoformat()
            periods.append({
                "label": label, "start_ms": start_ms, "end_ms_exclusive": start_ms + PERIOD_MS,
                "start_seq": first, "end_seq": first + count, "support": count,
            })
            parts.append((np.concatenate(x_parts), np.concatenate(y_parts)))
        selected = set(np.linspace(0, len(versions) - 1, min(MAX_CHECKPOINTS, len(versions)), dtype=int))
        matrix, checkpoints, series = [], [], []
        for index, version in enumerate(versions):
            checkpoints.append({
                "label": f"v{index}", "version": index, "fit_id": version["fit_id"],
                "sha256": version["sha256"], "end_seq": version["end_seq"],
            })
            if index not in selected:
                matrix.append([None] * len(periods))
                series.append({"step": index, "absolute_retention": None,
                               "reason": "checkpoint_outside_fixed_diagnostic_budget"})
                continue
            model.load((output / version["file"]).read_bytes())
            row = [float(np.mean(decide(model.predict(x), threshold) == y)) for x, y in parts]
            matrix.append(row)
            series.append({"step": index, "absolute_retention": sum(row) / len(row)})
        return {
            "scope": "fixed_period_temporal_holdout", "metric": "accuracy",
            "checkpoints": [row["label"] for row in checkpoints], "checkpoint_metadata": checkpoints,
            "periods": [row["label"] for row in periods], "period_metadata": periods,
            "matrix": matrix, "series": series, "absolute_retention": series[-1]["absolute_retention"],
            "sampling_method": "UTC 30-day periods anchored at Unix epoch; chronological period prefix",
            "max_checkpoints": MAX_CHECKPOINTS, "max_periods": MAX_PERIODS,
            "max_rows_per_period": MAX_PERIOD_ROWS,
        }


def run_observations(
    *, report, history, windows, prequential, holdout, retention, has_detector, overhead, started,
):
    """Total counts include candidates; the legacy operational estimates cover deployed models."""
    fits = history.fits
    initial = next(fit for fit in fits if fit["deployed_version"] == 0)
    promoted = [fit for fit in fits if fit["deployed_version"] is not None]
    rejected = [fit for fit in fits if fit["promotion_status"] == "rejected"]
    deployed_updates = promoted[1:]
    metrics = report["metrics"]
    evaluated = prequential["evaluated_examples"] + holdout["evaluated_examples"]

    def total(collection, key):
        return sum(fit["observation_effort"].get(key) or 0 for fit in collection)

    def fit_seconds(key):
        return sum(fit["observation_resources"].get(key) or 0 for fit in fits)

    update_epochs = total(deployed_updates, "epochs")
    if deployed_updates and all(fit["observation_effort"]["epochs"] is None for fit in deployed_updates):
        update_epochs = None
    aws_cost = metrics.get("aws_cost_total_usd")
    scenario_cost = metrics["scenario_total_cost"]
    cost = aws_cost if aws_cost is not None else scenario_cost
    cost_scope = ("deployed_training_serving_and_model_storage" if aws_cost is not None
                  else "deployed_training_serving_and_prequential_prediction_errors")
    served = {fit["fit_id"]: [0, 0.0] for fit in promoted}
    for window in windows:
        usage = served[window["fit_id"]]
        usage[0] += window["quality"]["evaluated_examples"]
        usage[1] += window["inference_seconds"]
    for fit in promoted:
        count, seconds = served[fit["fit_id"]]
        fit["observation_effort"]["served_evaluated_examples"] = count
        fit["observation_resources"].update(
            served_inference_seconds=seconds,
            served_predictions_per_second=count / seconds if seconds else None,
        )
    outcome = observations(
        "binary" if prequential["positive_class"] is not None else "multiclass",
        {"prequential": prequential, "holdout": holdout}, windows=windows,
        effort={
            "examples_observed": metrics["dataset_rows"], "evaluated_examples": evaluated,
            "unique_training_examples": unique_train_rows(fits),
            "training_presentations": total(fits, "training_presentations"),
            "replay_presentations": total(fits, "replay_presentations"),
            "steps": total(fits, "steps"), "epochs": total(fits, "epochs"),
            "updates": len(deployed_updates), "promoted_versions": len(promoted), "fit_count": len(fits),
            "rejected_candidate_count": len(rejected),
            "rejected_training_presentations": total(rejected, "training_presentations"),
            "deployed_training_presentations": total(promoted, "training_presentations"),
            "initial_unique_training_examples": initial["train_samples"],
            "initial_training_presentations": initial["training_presentations"],
            "initial_steps": initial["observation_effort"]["steps"],
            "initial_epochs": initial["observation_effort"]["epochs"],
            "update_training_presentations": total(deployed_updates, "training_presentations"),
            "update_steps": total(deployed_updates, "steps"),
            "update_epochs": update_epochs,
        },
        resources={
            "duration_seconds": time.monotonic() - started,
            "training_seconds": fit_seconds("training_seconds"),
            "inference_seconds": metrics["prediction_seconds"],
            "predictions_per_second": metrics["prediction_throughput_per_second"],
            "peak_rss_bytes": process_peak_rss_bytes(), "rss_scope": "process_lifetime_high_water_mark",
            "cost_total": cost, "cost_per_1000_evaluated": 1000 * cost / evaluated,
            "cost_currency": "USD" if aws_cost is not None else report["costs"]["settings"]["unit"],
            "cost_kind": "estimated", "cost_scope": cost_scope,
            "tracking_overhead_seconds": overhead + fit_seconds("tracking_overhead_seconds"),
        },
        retention=retention,
        drift={"alarms": report["alarms"], "alarm_count": len(report["alarms"]), "delay_unit": "examples",
               "axis": "evaluated_end_seq"} if has_detector else None,
        lineage={
            "dataset": report["dataset"], "dataset_version": report["contract"]["input"]["sha256"],
            "source_revision": report["contract"]["input"]["source"]["revision"],
            "model": report["model"], "policy": report["policy"],
            "seed": report["contract"]["iterative_training"]["seed"],
            "environment": report["contract"]["environment"]["execution"],
            "code_revision": report["contract"]["environment"]["git_revision"],
            "protocol": report["protocol"],
        },
    )
    for key in ("gem_bwt", "post_learning_fm"):
        outcome["retention"]["availability"][key]["reason"] = "no_historical_post_learning_baselines"
    if update_epochs is None:
        outcome["effort"]["availability"]["update_epochs"]["reason"] = "online_minibatches_have_no_epoch"
    for key, item in outcome["drift"]["availability"].items():
        item["reason"] = ("detector_not_enabled" if not has_detector and key in ("alarms", "alarm_count")
                          else "no_independently_annotated_changes")
    return outcome
