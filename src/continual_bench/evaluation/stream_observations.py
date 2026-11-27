"""Observations from released labels and measured learner activity in stream protocols."""

import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import batched, chain

import numpy as np

from continual_bench.core.evaluation_metrics import task_matrix_metrics
from continual_bench.core.tasks import BINARY, REGRESSION
from continual_bench.evaluation.stream_training_history import unique_train_rows
from continual_bench.models.classification_metrics import decide

from .tracking_observations import (
    batch_quality,
    binary_ranking,
    classification_count_quality,
    classification_quality,
    observations,
)


@dataclass(frozen=True)
class StreamQuality:
    task: str
    classes: tuple | None = None
    positive_class: bool | str | None = None
    decision_threshold: float | None = None
    target_unit: str | None = None
    confusion_classes: tuple = ()

    def collect(self, rows, *, curves=True, dense_confusion=True):
        if not dense_confusion and self.classes is not None:
            quality = self._class_window(rows)
        else:
            quality = batch_quality(
                (row["label"] for row in rows),
                (self.prediction(row) for row in rows),
                classes=self.classes,
                positive_class=self.positive_class,
                confusion_classes=self.confusion_classes,
                decision_threshold=self.decision_threshold,
                target_unit=self.target_unit,
                curves=curves,
            )
        if self.positive_class is not None:
            ranked = [row for row in rows if row.get("probability") is not None]
            quality.update(
                binary_ranking(
                    (row["label"] == self.positive_class for row in ranked),
                    (row["probability"] for row in ranked),
                    curves=curves,
                )
            )
            positive_count = sum(row["label"] == self.positive_class for row in ranked)
            quality.update(
                ranking_evaluated_examples=len(ranked),
                ranking_positive_examples=positive_count,
                ranking_scope="examples_with_observed_positive_probability",
            )
        return quality

    def _class_window(self, rows):
        pairs = Counter((row["label"], self.prediction(row)) for row in rows)
        support, predicted = Counter(), Counter()
        labels = list(
            dict.fromkeys(
                chain(self.classes, self.confusion_classes, (label for pair in pairs for label in pair))
            )
        )
        for (actual, prediction), count in pairs.items():
            support[actual] += count
            predicted[prediction] += count
        result = classification_count_quality(
            [support[label] for label in labels],
            [predicted[label] for label in labels],
            [pairs[label, label] for label in labels],
            labels,
            target_classes=self.classes,
            positive_class=self.positive_class,
            decision_threshold=self.decision_threshold,
        )
        indices = {label: index for index, label in enumerate(labels)}
        result["confusion_counts_sparse"] = [[indices[a], indices[p], n] for (a, p), n in pairs.items()]
        return result

    def prediction(self, row):
        if self.decision_threshold is not None:
            return row["probability"] >= self.decision_threshold
        return row["prediction"]


def single_prediction_metrics(row, quality):
    """Sufficient statistics for a batch of one, preserving the prediction's prequential scope."""
    truth = row["label"]
    prediction = quality.prediction(row)
    scores = {"prequential/evaluated_examples": 1}
    if quality.task == REGRESSION:
        error = abs(truth - prediction)
        scores.update({"prequential/mae": error, "prequential/rmse": error})
        return scores
    correct = int(truth == prediction)
    scores.update({"prequential/accuracy": correct, "prequential/macro_f1": correct / len(quality.classes)})
    if quality.positive_class is not None:
        actual = truth == quality.positive_class
        predicted = prediction == quality.positive_class
        scores.update(
            {
                "prequential/positive_precision": correct if predicted else None,
                "prequential/positive_recall": correct if actual else None,
            }
        )
        if (probability := row.get("probability")) is not None:
            scores["prequential/average_precision"] = 1.0 if actual else None
            true_probability = probability if actual else 1 - probability
            scores["prequential/log_loss"] = -math.log(max(true_probability, 1e-15))
    return scores


def binary_summary_quality(metrics):
    return classification_quality(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        [False, True],
        positive_class=True,
        decision_threshold=0.5,
        average_precision=metrics.get("average_precision"),
        ranking_method="exact_ties_grouped" if metrics.get("average_precision") is not None else None,
    )


BINARY_QUALITY = StreamQuality(BINARY, (False, True), True, 0.5)


def declared_class_labels(manifest):
    order = manifest.get("class_order") or manifest.get("labels") or ()
    return tuple(dict.fromkeys(str(label) for label in order))


def adapted_binary_positive_class(manifest):
    """Identify the explicit EventSamples adapter independently of an observed class subset."""
    if manifest.get("labels") == ["0", "1"] and (
        manifest.get("target_conversion") == "explicit binary boolean to class strings, no information loss"
    ):
        return "1"
    return None


def fit_effort(fits, *, observed, evaluated, unique_training, updates):
    total = sum(fit["training_presentations"] for fit in fits)
    measured = all("observation_steps" in fit for fit in fits)
    steps = [len(fit["observation_steps"]) for fit in fits] if measured else []
    initial = fits[0] if fits else {}
    return {
        "examples_observed": observed,
        "evaluated_examples": evaluated,
        "unique_training_examples": unique_training,
        "training_presentations": total,
        "replay_presentations": total - unique_training,
        "steps": sum(steps) if measured else None,
        "updates": updates,
        "initial_unique_training_examples": initial.get("train_samples"),
        "initial_training_presentations": initial.get("training_presentations"),
        "initial_steps": steps[0] if steps else None,
        "update_steps": sum(steps[1:]) if measured else None,
        "update_training_presentations": total - (initial.get("training_presentations") or 0),
        "fit_count": len(fits),
        "promoted_fits": sum(fit["summary"]["promoted"] for fit in fits),
        "rejected_fits": sum(not fit["summary"]["promoted"] for fit in fits),
    }


def cost_resources(costs, evaluated):
    aws = costs.get("aws") or {}
    usd = aws.get("metrics", {}).get("cost_total_usd")
    if usd is not None:
        total, currency, kind = usd, aws["unit"], aws["kind"]
        scope = aws["scope"]
    else:
        total = costs.get("metrics", {}).get("cost_total")
        currency, kind = costs.get("settings", {}).get("unit"), costs.get("kind")
        scope = costs.get("scope")
    return {
        "cost_total": total,
        "cost_per_1000_evaluated": (1000 * total / evaluated if total is not None and evaluated else None),
        "cost_currency": currency,
        "cost_kind": kind,
        "cost_scope": scope,
    }


def quality_window(rows, quality, scope):
    versions = list(dict.fromkeys(row.get("model_version") for row in rows))
    return {
        "step": rows[-1]["seq"],
        "scope": scope,
        "start_seq": rows[0]["seq"],
        "quality": quality.collect(rows, curves=False, dense_confusion=False),
        "model_version": versions[0] if len(versions) == 1 else None,
        "model_versions": versions,
    }


def typed_observations(report, quality, scored, *, unique_training, tracking_seconds, retention=None):
    tick = time.perf_counter()
    manifest, config = report["config"]["source"], report["config"]
    count = sum(len(rows) for rows in scored.values())
    scopes = {split: quality.collect(rows) for split, rows in scored.items()}
    windows = [
        quality_window(chunk, quality, split)
        for split, rows in scored.items()
        for chunk in batched(rows, config["window"])
    ]
    compute = report["compute"]
    effort = fit_effort(
        report["training_history"],
        observed=report["observed"],
        evaluated=count,
        unique_training=unique_training,
        updates=len(report["updates"]),
    )
    effort["promoted_versions"] = report["model_versions"]["count"]
    resources = {
        "duration_seconds": compute["elapsed_seconds"],
        "training_seconds": compute["training_seconds"],
        "inference_seconds": compute["prediction_seconds"],
        "inference_examples": report["observed"],
        "predictions_per_second": report["observed"] / compute["prediction_seconds"]
        if compute["prediction_seconds"]
        else None,
        "peak_rss_bytes": compute["process_peak_rss_bytes"],
        "rss_scope": compute["memory_scope"],
        **cost_resources(report["costs"], count),
    }
    result = observations(
        quality.task,
        scopes,
        windows=windows,
        effort=effort,
        resources=resources,
        retention=retention,
        lineage={
            "dataset": manifest.get("name"),
            "dataset_version": manifest.get("version") or config["samples_sha256"],
            "materialized_sha256": config["samples_sha256"],
            "promoted_model_versions_artifact": "model-versions.json"
            if effort["promoted_versions"]
            else None,
            "model": config["model"],
            "policy": config["policy"],
            "seed": manifest.get("seed"),
            "environment": config["environment"],
            "code_revision": config["environment"]["code_sha256"],
            "protocol": report["protocol"],
            "scope_notes": {
                "prequential": "postwarmup training-source predictions, scored on label release",
                "validation": "source validation partition; evolving model, labels excluded from training",
                "test": "source test partition; evolving model, labels excluded from training",
            },
        },
    )
    result["resources"]["tracking_overhead_seconds"] = tracking_seconds + time.perf_counter() - tick
    result["resources"]["availability"].pop("tracking_overhead_seconds", None)
    return result


def _binary_retention(tracking, versions):
    rows = (tracking.get("continual") or {}).get("matrix", [])
    if not rows:
        return None
    periods = list(rows[0]["accuracies"])
    matrix = [[row["accuracies"][period] for period in periods] for row in rows]
    series = []
    for index, row in enumerate(rows):
        result = task_matrix_metrics(matrix[: index + 1])["metrics"]
        series.append(
            {
                "step": row["end_seq"],
                "model_version": versions[index] if index < len(versions) else None,
                "absolute_retention": result["past_task_retention_accuracy"],
                "gem_bwt": result["gem_backward_transfer"],
                "post_learning_fm": result["postlearning_forgetting"],
            }
        )
    return {
        "matrix": matrix,
        "periods": periods,
        "checkpoints": [row["model_version"] or f"released-through-{row['step']}" for row in series],
        "metric": "accuracy",
        **{key: series[-1][key] for key in ("absolute_retention", "gem_bwt", "post_learning_fm")},
        "series": series,
    }


def binary_observations(report, observer, *, duration_seconds, tracking_seconds=0.0):
    started = time.perf_counter()
    tracking, config, dataset = report["tracking"], report["config"], report["dataset"]
    metrics = tracking["metrics"]
    scope = "holdout" if "split" in report else "prequential"
    final_id = report["model_versions"].get("final_version_id")
    rows = (
        [{**row, "model_version": final_id} for row in observer.predictions]
        if scope == "holdout"
        else observer.predictions
    )
    windows, offset = [], 0
    for window in report["windows"]:
        stop = offset + window["n"]
        selected = rows[offset:stop]
        item = quality_window(selected, BINARY_QUALITY, scope)
        if item["step"] != window["end_seq"] or len(selected) != window["n"]:
            raise ValueError("binary tracking window differs from its released-label observations")
        windows.append(item)
        offset = stop
    if offset != len(rows):
        raise ValueError("binary tracking windows do not cover all evaluated predictions")
    count = len(rows)
    effort = fit_effort(
        report["training_history"],
        observed=report["observed"],
        evaluated=count,
        unique_training=metrics["training_unique_examples"],
        updates=len(report["updates"]),
    )
    effort["promoted_versions"] = report["model_versions"]["count"]
    enabled = scope == "prequential" and config["detector"]["name"] != "none"
    detection = tracking["detection"]
    changes = dataset.get("changepoints")
    matches = detection["matches"]
    drift = {
        "alarms": [a["seq"] for a in report["alarms"]] if enabled else None,
        "changes": changes,
        "precision": metrics.get("drift_precision") if enabled else None,
        "recall": metrics.get("drift_recall") if enabled else None,
        "f1": metrics.get("drift_f1") if enabled else None,
        "delays": [match["delay"] for match in matches if "delay" in match] if matches else None,
        "delay_unit": "examples" if matches else None,
        "matching": matches if changes is not None else None,
        "alarms_outside_annotated_windows": detection.get("alarms_outside_annotated_windows")
        if enabled and changes is not None
        else None,
        "alarms_outside_annotated_windows_count": detection.get("alarms_outside_annotated_windows_count")
        if enabled and changes is not None
        else None,
        "recovery": tracking.get("recovery") if changes is not None else None,
        "regimes": tracking.get("regimes"),
    }
    release_delays = [
        match["release_delay_source_clock"] for match in matches if "release_delay_source_clock" in match
    ]
    drift.update(
        release_delays=release_delays or None,
        release_delay_mean=sum(release_delays) / len(release_delays) if release_delays else None,
        release_delay_unit=(
            "seconds" if "seconds" in dataset.get("timestamp_kind", "") else "source_clock_unit"
        )
        if release_delays
        else None,
    )
    fargate_retraining = tracking.get("fargate_retraining") or {}
    oracle_reference = tracking.get("oracle_reference") or {}
    prediction_seconds = observer.inference_seconds
    result = observations(
        BINARY,
        {scope: BINARY_QUALITY.collect(rows)},
        windows=windows,
        effort=effort,
        resources={
            "duration_seconds": duration_seconds,
            "training_seconds": metrics["train_seconds"],
            "inference_seconds": prediction_seconds,
            "inference_examples": observer.prediction_count,
            "predictions_per_second": observer.prediction_count / prediction_seconds
            if prediction_seconds
            else None,
            "peak_rss_bytes": metrics["process_peak_rss_bytes"],
            "rss_scope": "process lifetime high-water mark (finalizing process)",
            "rejected_checkpoint_storage_bytes": sum(
                (fit.get("checkpoint") or {}).get("bytes", 0)
                for fit in report["training_history"]
                if fit.get("promotion_status") == "rejected"
                and (fit.get("checkpoint") or {}).get("retained", True)
            ),
            "fargate_retrain_alpha": fargate_retraining.get("alpha"),
            "fargate_retrain_lambda": fargate_retraining.get("lambda"),
            "fargate_retrain_reference_cost_usd": fargate_retraining.get("reference_cost_per_action_usd"),
            "fargate_retrain_estimated_cost_usd": fargate_retraining.get("estimated_retraining_cost_usd"),
            "fargate_retrain_count": fargate_retraining.get("retrain_count"),
            "fargate_retrain_duration_seconds": fargate_retraining.get("duration_seconds"),
            "fargate_retrain_billed_seconds": fargate_retraining.get("billed_seconds"),
            "fargate_retrain_hourly_rate_usd": fargate_retraining.get("hourly_rate_usd"),
            "fargate_retraining": fargate_retraining,
            **cost_resources(tracking["costs"], count),
        },
        retention=_binary_retention(tracking, observer.continual_versions),
        drift=drift,
        lineage={
            "dataset": dataset.get("name") or config["dataset"]["name"],
            "dataset_version": dataset.get("dataset_version", dataset.get("sha256")),
            "model": config["model"],
            "policy": config["policy"],
            "seed": config["dataset"]["seed"],
            "environment": report["environment"],
            "code_revision": report["environment"]["code_sha256"],
            "protocol": report["protocol"],
            "final_version": final_id,
            "promoted_model_versions_artifact": "model-versions.json",
            "oracle_reference": oracle_reference,
        },
    )
    result["resources"]["tracking_overhead_seconds"] = (
        metrics["instrumentation_seconds"] + tracking_seconds + time.perf_counter() - started
    )
    result["resources"]["availability"].pop("tracking_overhead_seconds", None)
    return result


# --- Temporal holdout observations (migrated from zia_observations) ---


PERIOD_MS = 30 * 24 * 60 * 60 * 1000
MAX_PERIODS = 12
MAX_PERIOD_ROWS = 256
MAX_CHECKPOINTS = 16


class TemporalHoldoutProbes:
    """Retain the first bounded prefix from each fixed UTC period in the holdout."""

    def __init__(self):
        self.data = {}
        self.captured = []

    def _parts(self):
        return [(np.concatenate(x), np.concatenate(y)) for _, x, y, _ in self.data.values()]

    def capture(self, model, threshold):
        """Score fixed diagnostic probes while the version is live, without retaining weights."""
        self.captured.append(
            [float(np.mean(decide(model.predict(x), threshold) == y)) for x, y in self._parts()]
        )

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
                x_parts.append(features[offset : offset + take].copy())
                y_parts.append(labels[offset : offset + take].copy())
                self.data[period] = (first, x_parts, y_parts, count + take)
            offset = end

    def evaluate(self, model, output, versions, threshold, *, artifact_mode="full"):
        periods, parts = [], []
        for start_ms, (first, x_parts, y_parts, count) in self.data.items():
            label = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date().isoformat()
            periods.append(
                {
                    "label": label,
                    "start_ms": start_ms,
                    "end_ms_exclusive": start_ms + PERIOD_MS,
                    "start_seq": first,
                    "end_seq": first + count,
                    "support": count,
                }
            )
            parts.append((np.concatenate(x_parts), np.concatenate(y_parts)))
        selected = set(np.linspace(0, len(versions) - 1, min(MAX_CHECKPOINTS, len(versions)), dtype=int))
        matrix, checkpoints, series = [], [], []
        for index, version in enumerate(versions):
            checkpoints.append(
                {
                    "label": f"v{index}",
                    "version": index,
                    "fit_id": version["fit_id"],
                    "sha256": version["sha256"],
                    "end_seq": version["end_seq"],
                }
            )
            if index not in selected:
                matrix.append([None] * len(periods))
                series.append(
                    {
                        "step": index,
                        "absolute_retention": None,
                        "reason": "checkpoint_outside_fixed_diagnostic_budget",
                    }
                )
                continue
            if artifact_mode == "metrics_only":
                if len(self.captured) != len(versions):
                    raise ValueError("missing live-version holdout diagnostics")
                row = self.captured[index]
            else:
                model.load((output / version["file"]).read_bytes())
                row = [float(np.mean(decide(model.predict(x), threshold) == y)) for x, y in parts]
            matrix.append(row)
            series.append({"step": index, "absolute_retention": sum(row) / len(row)})
        return {
            "scope": "fixed_period_temporal_holdout",
            "metric": "accuracy",
            "checkpoints": [row["label"] for row in checkpoints],
            "checkpoint_metadata": checkpoints,
            "periods": [row["label"] for row in periods],
            "period_metadata": periods,
            "matrix": matrix,
            "series": series,
            "absolute_retention": series[-1]["absolute_retention"],
            "sampling_method": "UTC 30-day periods anchored at Unix epoch; chronological period prefix",
            "max_checkpoints": MAX_CHECKPOINTS,
            "max_periods": MAX_PERIODS,
            "max_rows_per_period": MAX_PERIOD_ROWS,
            **(
                {"evaluation_method": "live_version_on_fixed_read_only_holdout_probes"}
                if artifact_mode == "metrics_only"
                else {}
            ),
        }


def run_observations(
    *,
    report,
    history,
    windows,
    prequential,
    holdout,
    retention,
    has_detector,
    overhead,
    started,
):
    """Total counts include candidates; the legacy operational estimates cover deployed models."""
    fits = history.fits
    executed = [fit for fit in fits if fit.get("execution_origin") != "reused_metric_evidence"]
    initial = next(fit for fit in fits if fit["deployed_version"] == 0)
    promoted = [fit for fit in fits if fit["deployed_version"] is not None]
    rejected = [fit for fit in executed if fit["promotion_status"] == "rejected"]
    deployed_updates = promoted[1:]
    metrics = report["metrics"]
    source = report["contract"]["input"]["source"]
    evaluated = prequential["evaluated_examples"] + holdout["evaluated_examples"]

    def total(collection, key):
        return sum(fit["observation_effort"].get(key) or 0 for fit in collection)

    def fit_seconds(key):
        return sum(fit["observation_resources"].get(key) or 0 for fit in executed)

    update_epochs = total(deployed_updates, "epochs")
    if deployed_updates and all(fit["observation_effort"]["epochs"] is None for fit in deployed_updates):
        update_epochs = None
    aws_cost = metrics.get("aws_cost_total_usd")
    scenario_cost = metrics["scenario_total_cost"]
    cost = aws_cost if aws_cost is not None else scenario_cost
    cost_scope = (
        "deployed_training_serving_and_model_storage"
        if aws_cost is not None
        else "deployed_training_serving_and_prequential_prediction_errors"
    )
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
        {"prequential": prequential, "holdout": holdout},
        windows=windows,
        effort={
            "examples_observed": metrics["dataset_rows"],
            "evaluated_examples": evaluated,
            "unique_training_examples": unique_train_rows(executed),
            "training_presentations": total(executed, "training_presentations"),
            "replay_presentations": total(executed, "replay_presentations"),
            "steps": total(executed, "steps"),
            "epochs": total(executed, "epochs"),
            "updates": len(deployed_updates),
            "promoted_versions": len(promoted),
            "fit_count": len(executed),
            "reused_matrix_candidate_fit_count": len(fits) - len(executed),
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
            "peak_rss_bytes": metrics["process_peak_rss_bytes"],
            "rss_scope": "process_lifetime_high_water_mark",
            "cost_total": cost,
            "cost_per_1000_evaluated": 1000 * cost / evaluated,
            "cost_currency": "USD" if aws_cost is not None else report["costs"]["settings"]["unit"],
            "cost_kind": "estimated",
            "cost_scope": cost_scope,
            "tracking_overhead_seconds": overhead + fit_seconds("tracking_overhead_seconds"),
        },
        retention=retention,
        drift={
            "alarms": report["alarms"],
            "alarm_count": len(report["alarms"]),
            "delay_unit": "examples",
            "axis": "evaluated_end_seq",
        }
        if has_detector
        else None,
        lineage={
            "dataset": report["dataset"],
            "dataset_version": report["contract"]["input"]["sha256"],
            "source_revision": (
                source.get("revision") or source.get("raw_sha256") or source.get("source_version")
            ),
            "model": report["model"],
            "policy": report["policy"],
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
        item["reason"] = (
            "detector_not_enabled"
            if not has_detector and key in ("alarms", "alarm_count")
            else "no_independently_annotated_changes"
        )
    return outcome
