"""Versioned metric derivation, including explicit missing-evidence states."""

import copy
import hashlib
import json
import math
from pathlib import Path

from continual_bench.domain.metrics import Metrics
from continual_bench.domain.observability import first_target, learning_speed_ratio, relative_cost_error

VERSION = "residency-metrics-v1"


def metric_catalog():
    candidates = (
        Path(__file__).resolve().parents[1] / "metric-definitions.json",
        Path(__file__).resolve().parents[3] / "configs/metric-definitions.json",
        Path("configs/metric-definitions.json"),
        Path("/app/configs/metric-definitions.json"),
    )
    for path in candidates:
        if path.is_file():
            raw = path.read_bytes()
            return json.loads(raw), raw
    raise FileNotFoundError("metric-definitions.json must accompany the benchmark deployment")


def confusion_metrics(summary, positives):
    """Recover integer confusion counts, never scores, from exact binary summaries."""
    n = summary["n"]
    if (
        type(n) is not int
        or type(positives) is not int
        or not 0 <= positives <= n
        or any(
            not isinstance(summary[k], (int, float))
            or not math.isfinite(summary[k])
            or not 0 <= summary[k] <= 1
            for k in ("accuracy", "f1")
        )
    ):
        raise ValueError("invalid historical metric counts or ratios")
    correct = round(summary["accuracy"] * n)
    if not math.isclose(correct, summary["accuracy"] * n, abs_tol=1e-6):
        raise ValueError("noninteger accuracy count")
    errors = n - correct
    tp_float = summary["f1"] * errors / (2 * (1 - summary["f1"])) if summary["f1"] < 1 else positives
    tp = round(tp_float)
    fn = positives - tp
    fp = errors - fn
    tn = correct - tp
    if min(tp, fn, fp, tn) < 0 or not math.isclose(tp_float, tp, abs_tol=1e-6):
        raise ValueError("inconsistent historical confusion matrix")
    accumulator = Metrics(n=n, correct=correct, tp=tp, fp=fp, fn=fn)
    values = accumulator.values()
    if not math.isclose(values["f1"], summary["f1"], abs_tol=1e-12):
        raise ValueError("historical F1 does not match reconstructed counts")
    values.update(summary)
    for key in ("roc_auc", "average_precision", "ece", "mce"):
        values[key] = None
    for key in ("brier", "log_loss"):
        if key not in summary:
            values[key] = None
    return values


def detector_metrics(report, settings):
    points = report["dataset"].get("changepoints")
    enabled = report["config"]["detector"]["name"] != "none"
    names = (
        "drift_tp",
        "drift_fp",
        "drift_fn",
        "drift_precision",
        "drift_recall",
        "drift_f1",
        "drift_delay_mean_samples",
        "drift_false_alarm_rate_per_1000",
        "drift_delay_release_mean",
        "drift_false_alarm_rate",
        "drift_stable_false_alarms",
        "drift_stable_exposure_samples",
        "drift_unmatched_alarms_per_1000_evaluated",
        "drift_complete_followup_changes",
        "drift_censored_changes",
        "drift_observed_detections",
    )
    metrics = dict.fromkeys(names)
    details = {
        "matching": "earliest alarm in [change,min(change+tolerance,next_change)); complete followup only",
        "tolerance_samples": settings["detection_window"],
        "matches": [],
    }
    reason = "detector_disabled" if not enabled else "no_ground_truth_changepoints"
    if points is not None and enabled:
        n = report["metrics"]["n"]
        observed = report["observed"]
        evaluation_start = observed - n
        alarms = [a for a in report["alarms"] if evaluation_start <= a["seq"] < observed]
        used = set()
        excluded = set()
        intervals = []
        complete = observed_detections = censored = 0
        delays, released = [], []
        for index, point in enumerate(points):
            deadline = min(
                point + settings["detection_window"],
                points[index + 1] if index + 1 < len(points) else point + settings["detection_window"],
            )
            start, end = max(point, evaluation_start), min(deadline, observed)
            if start < end:
                intervals.append((start, end))
            candidates = [
                (a["seq"], i) for i, a in enumerate(alarms) if start <= a["seq"] < end
            ]
            observed_detections += bool(candidates)
            followup_complete = point >= evaluation_start and deadline <= observed
            row = {"change_seq": point, "deadline": deadline, "observed_end": end,
                   "censored": not followup_complete, "alarm_seq": min(candidates)[0] if candidates else None}
            if not followup_complete:
                censored += 1
                excluded.update(i for _, i in candidates)
                row["reason"] = "incomplete_followup" if deadline > observed else "change_before_evaluation"
                details["matches"].append(row)
                continue
            complete += 1
            if candidates:
                seq, i = min(candidates)
                used.add(i)
                delays.append(seq - point)
                if "released_at" in alarms[i]:
                    released.append(alarms[i]["released_at"] - report["config"]["dataset"]["epoch"] - point)
                row["delay"] = seq - point
            details["matches"].append(row)
        considered = len(alarms) - len(excluded)
        tp, fp, fn = len(used), considered - len(used), complete - len(used)
        stable_exposure = n - sum(end - start for start, end in intervals)
        stable_alarms = sum(not any(start <= a["seq"] < end for start, end in intervals) for a in alarms)
        unmatched_rate = 1000 * fp / n if n else None
        metrics.update(
            drift_tp=tp,
            drift_fp=fp,
            drift_fn=fn,
            drift_precision=tp / considered if considered else None,
            drift_recall=tp / complete if complete else None,
            drift_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            drift_delay_mean_samples=sum(delays) / len(delays) if delays else None,
            drift_false_alarm_rate_per_1000=unmatched_rate,
            drift_unmatched_alarms_per_1000_evaluated=unmatched_rate,
            drift_delay_release_mean=sum(released) / len(released) if released else None,
            drift_false_alarm_rate=stable_alarms / stable_exposure if stable_exposure else None,
            drift_stable_false_alarms=stable_alarms,
            drift_stable_exposure_samples=stable_exposure,
            drift_complete_followup_changes=complete,
            drift_censored_changes=censored,
            drift_observed_detections=observed_detections,
        )
        details.update(evaluation_range=[evaluation_start, observed],
                       censored_alarm_count=len(excluded), stable_exposure_samples=stable_exposure,
                       stable_false_alarms=stable_alarms,
                       unmatched_rate_denominator="all evaluated examples; censored-window alarms excluded")
        details["unavailable_reasons"] = {
            "drift_precision": "no_uncensored_alarms",
            "drift_recall": "no_complete_followup_changes",
            "drift_f1": "no_complete_change_or_uncensored_alarm",
            "drift_false_alarm_rate": "no_stable_evaluated_exposure",
            "drift_delay_mean_samples": "no_complete_followup_detections",
            "drift_delay_release_mean": "no_complete_detections_with_release_timestamps",
        }
        reason = None
    return metrics, details, reason


def recovery_metrics(report, settings):
    points = report["dataset"].get("changepoints") or []
    rows = []
    windows = report["windows"]
    for index, point in enumerate(points):
        previous = [w for w in windows if w["end_seq"] < point
                    and w["n"] == report["config"]["metric_window"]]
        if not previous:
            rows.append({"change_seq": point, "delay": None, "available": False,
                         "censored": False, "reason": "missing_prechange_window"})
            continue
        target = previous[-1]["accuracy"] - settings["recovery_tolerance"]
        end = points[index + 1] if index + 1 < len(points) else report["observed"]
        consecutive = 0
        recovered = None
        for window in windows:
            if window["n"] != report["config"]["metric_window"]:
                continue
            if window["end_seq"] - window["n"] + 1 < point or window["end_seq"] >= end:
                continue
            consecutive = consecutive + 1 if window["accuracy"] >= target else 0
            if consecutive >= settings["recovery_windows"]:
                recovered = window["end_seq"] - point
                break
        rows.append(
            {
                "change_seq": point,
                "available": True,
                "target_accuracy": target,
                "delay": recovered,
                "censored": recovered is None,
                "horizon_end": end,
                "prechange_accuracy": previous[-1]["accuracy"],
                "first_postchange_accuracy": next(
                    (
                        w["accuracy"]
                        for w in windows
                        if w["n"] == report["config"]["metric_window"]
                        and w["end_seq"] - w["n"] + 1 >= point
                        and w["end_seq"] < end
                    ),
                    None,
                ),
            }
        )
    delays = [r["delay"] for r in rows if r["delay"] is not None]
    return {
        "recovery_delay_mean_samples": sum(delays) / len(delays) if delays else None,
        "recovery_recovered_changes": len(delays) if points else None,
        "recovery_censored_changes": sum(r["censored"] for r in rows) if points else None,
        "recovery_unavailable_changes": sum(not r["available"] for r in rows) if points else None,
    }, rows


def validate_learning_curve(curve):
    """Validate externally supplied curves before deriving any target ratios."""
    if not isinstance(curve, list) or not curve:
        raise ValueError("learning curve must be a nonempty list")
    previous = -1
    for point in curve:
        if not isinstance(point, dict) or not {"step", "value"} <= point.keys():
            raise ValueError("learning curve points require explicit step and value")
        step, value = point["step"], point["value"]
        if type(step) is not int or step <= previous:
            raise ValueError("learning curve steps must be nonnegative and strictly increasing")
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("learning curve accuracies must be finite in [0,1] or null")
        previous = step
    return curve


def build_tracking(report, settings, observation=None, continual=None, predictions=None):
    settings = copy.deepcopy(settings)
    observation = observation or {}
    metrics = dict(report["metrics"])
    series = {}
    availability = {}

    def available(name, value, reason="missing_evidence", scope=None):
        metrics[name] = value
        availability[name] = {
            "status": "available" if value is not None else "unavailable",
            "reason": None if value is not None else reason,
            "scope": scope,
        }

    for key, value in metrics.copy().items():
        available(key, value, "undefined_for_class_support_or_missing_prediction_scores")
    for window in report["windows"]:
        for key, value in window.items():
            if key != "end_seq":
                series.setdefault(key, []).append({"step": window["end_seq"], "value": value})
    if predictions is not None:
        probability_mae = sum(abs(int(r["label"]) - r["probability"]) for r in predictions)
        available("mae_probability", probability_mae / len(predictions) if predictions else None)
    else:
        available("mae_probability", None, "historical_prediction_scores_not_recorded")
    available("mae", None, "binary_probability_error_is_not_a_regression_task_metric")
    available("mse", None, "binary_probability_error_is_not_a_regression_task_metric")
    available(
        "mse_probability", report["metrics"].get("brier"), scope="binary_probability_error_equals_brier"
    )
    curve = [{"step": w["end_seq"] + 1, "value": w["accuracy"]} for w in report["windows"]]
    available(
        "samples_to_target_accuracy",
        first_target(curve, settings["target_accuracy"]),
        "target_not_reached_at_recorded_windows",
        "window_end_source_offset_including_warmup",
    )
    available("iterations_to_target", None, "requires_training_iteration_curve_on_fixed_test")
    available(
        "mean_window_accuracy",
        sum(w["accuracy"] for w in report["windows"]) / len(report["windows"]) if report["windows"] else None,
    )
    for key in (
        "prediction_seconds",
        "prediction_latency_mean_seconds",
        "prediction_latency_p50_seconds",
        "prediction_latency_p95_seconds",
        "prediction_latency_p99_seconds",
        "prediction_throughput_per_second",
        "fading_accuracy",
        "sampled_rss_byte_seconds",
        "instrumentation_seconds",
        "monitor_seconds",
    ):
        available(
            key, observation.get(key), "not_in_historical_capture", "evaluation_process_not_service_SLA"
        )
    if observation.get("fading_series"):
        series["fading_accuracy"] = observation["fading_series"]
    config = report["config"]
    updates = report["updates"]
    used = sum(u["train_samples"] for u in updates)
    initial = report["split"]["train"][1] if "split" in report else config["warmup"]
    trained = set(range(initial))
    for u in updates:
        trained.update(range(u["seq"] - u["train_samples"] + 1, u["seq"] + 1))
    available("training_presentations", initial + used, scope="includes_initial_fit_and_replays")
    available("training_used_post_warmup", used)
    available("training_unique_examples", len(trained))
    available("training_repeated_presentations", initial + used - len(trained))
    available("update_count", len(updates))
    available("retrain_count", sum(u["action"] == "reset_retrain" for u in updates))
    available("alarm_count", len(report["alarms"]))
    available("updates_per_1000_evaluated", 1000 * len(updates) / metrics["n"] if metrics["n"] else None)
    available(
        "training_budget_blocked_decisions",
        report.get("training", {}).get("blocked_decisions"),
        "offline_has_no_postwarmup_budget",
    )
    budget = config.get("training_budget")
    available("training_budget_utilization", used / budget if budget else None, "no_positive_budget")
    train_seconds = sum(u["seconds"] for u in updates)
    available("update_train_seconds", train_seconds)
    initial_seconds = observation.get("warmup_train_seconds")
    available("initial_train_seconds", initial_seconds, "initial_fit_timer_not_captured")
    available(
        "train_seconds",
        train_seconds + initial_seconds if initial_seconds is not None else None,
        "initial_fit_timer_not_captured",
    )
    available(
        "postwarmup_train_samples_per_second",
        used / train_seconds if train_seconds else None,
        "no_timed_postwarmup_training",
    )
    for name, field in (
        ("evaluation_compute_seconds", "compute_seconds"),
        ("evaluation_attempt_wall_seconds", "wall_seconds_this_attempt"),
        ("process_peak_rss_bytes", "peak_rss_bytes_process"),
    ):
        available(name, report.get(field), scope="timer_scope_differs_between_protocols")
    retention = report.get("retention_probes", [])
    final = {}
    retention_series = {}
    for point in retention:
        concept = point["concept"]
        final[concept] = point
        for name, field in (("retention_accuracy", "accuracy"), ("retention_drop", "drop_from_best_probe")):
            # The final label drain can change the model without advancing the source watermark.
            # Keep every raw probe below; a published curve has one latest value per source step.
            retention_series.setdefault(f"{name}/concept_{concept}", {})[point["observed_seq"]] = point[field]
    for name, values in retention_series.items():
        series[name] = [{"step": step, "value": value} for step, value in values.items()]
    available(
        "retention_final_mean_accuracy",
        sum(p["accuracy"] for p in final.values()) / len(final) if final else None,
        "no_synthetic_retention_probes",
        "sampled_concept_probes_not_task_matrix",
    )
    available(
        "retention_final_mean_drop",
        sum(p["drop_from_best_probe"] for p in final.values()) / len(final) if final else None,
        "no_synthetic_retention_probes",
        "nonnegative_sampled_probe_drop",
    )
    if continual is not None:
        for key, value in continual["metrics"].items():
            metrics[key] = value
            availability[key] = continual["availability"][key]
        series.update(continual["series"])
        for key in (
            "probe_seconds",
            "reference_train_seconds",
            "reference_train_samples",
            "overhead_seconds",
        ):
            available("cl_" + key, continual[key], scope="reference_evaluation_separate_from_policy_budget")
    else:
        for key in (
            "cl_average_accuracy",
            "cl_average_incremental_accuracy",
            "cl_forgetting",
            "cl_backward_transfer",
            "cl_intransigence",
            "cl_forward_transfer",
        ):
            available(key, None, "task_boundary_matrix_and_references_not_captured")
    detection, detection_details, reason = detector_metrics(report, settings)
    for key, value in detection.items():
        available(key, value, reason or detection_details.get("unavailable_reasons", {}).get(
            key, "no_evaluated_exposure"))
    recovery, recovery_details = recovery_metrics(report, settings)
    for key, value in recovery.items():
        available(key, value, "no_known_changes_or_recovery_not_observed")
    error_sum = sum(1 - w["accuracy"] for w in report["windows"])
    retrains = sum(u["action"] == "reset_retrain" for u in updates)
    reset_only = (config.get("policy", {}).get("name") != "online"
                  and config.get("policy", {}).get("action", "reset_retrain") == "reset_retrain"
                  and all(u["action"] == "reset_retrain" for u in updates))
    available(
        "upf_total_cost",
        settings.get("upf_alpha", 1) * retrains + error_sum if reset_only else None,
        "incremental_updates_do_not_follow_the_UPF_retraining_contract",
        "reset_retrain_only_plus_sum_window_error_rates_fixed_policy_not_paper_reproduction",
    )
    available("posthoc_update_event_cost", settings.get("upf_alpha", 1) * len(updates) + error_sum,
              scope="all_update_events_including_learn_one_fixed_trajectory_scenario")
    sensitivity = [
        {"alpha": alpha, "value": alpha * len(updates) + error_sum} for alpha in settings["upf_alphas"]
    ]
    series["posthoc_cost_sensitivity"] = [
        {"step": index, "value": row["value"]} for index, row in enumerate(sensitivity)
    ]
    availability["posthoc_cost_sensitivity"] = {
        "status": "available", "reason": None,
        "scope": "series indexed by alpha grid position; see cost_sensitivity metadata; all update events",
    }
    area = sum(
        (b["alpha"] - a["alpha"]) * (a["value"] + b["value"]) / 2
        for a, b in zip(sensitivity, sensitivity[1:])
    )
    available("cost_sensitivity_auc", area, scope="configured_alpha_grid_not_UPF_oracle_campaign_AUC")
    costs = settings.get("costs")
    cost_total = None
    if costs is not None:
        error_cost = metrics["fp"] * costs["false_positive"] + metrics["fn"] * costs["false_negative"]
        available("cost_prediction_errors", error_cost, scope=costs["unit"])
        available(
            "cost_training_presentations", (initial + used) * costs["train_sample"], scope=costs["unit"]
        )
        available("cost_update_events", len(updates) * costs["update"], scope=costs["unit"])
        timed_train = metrics["train_seconds"]
        timed_predict = metrics["prediction_seconds"]
        if (not costs["train_second"] or timed_train is not None) and (
            not costs["predict_second"] or timed_predict is not None
        ):
            cost_total = (
                error_cost
                + (initial + used) * costs["train_sample"]
                + len(updates) * costs["update"]
                + (timed_train or 0) * costs["train_second"]
                + (timed_predict or 0) * costs["predict_second"]
            )
    for key in ("cost_prediction_errors", "cost_training_presentations", "cost_update_events"):
        if key not in metrics:
            available(key, None, "cost_coefficients_not_configured")
    available(
        "cost_total", cost_total, "cost_profile_or_required_timers_missing", costs["unit"] if costs else None
    )
    def adaptation_cost(update):
        if costs is None or costs["predict_second"]:
            return None
        return (update["train_samples"] * costs["train_sample"] + costs["update"]
                + update["seconds"] * costs["train_second"])

    gain_rows = []
    for comparison in observation.get("comparisons", []):
        samples = comparison["evaluated"]
        gain = comparison["old_weighted_loss"] - comparison["new_weighted_loss"] if samples else None
        # The prediction-time shadow trace keeps delayed labels attached to their original interval.
        incremental_cost = adaptation_cost(comparison)
        roi = gain / incremental_cost if gain is not None and incremental_cost else None
        gain_rows.append(
            {**comparison, "realized_gain": gain, "adaptation_cost": incremental_cost, "roi": roi}
        )
    gains = [row["realized_gain"] for row in gain_rows if row["realized_gain"] is not None]
    available(
        "adaptation_realized_gain",
        sum(gains) if gains else None,
        "requires_shadow_predictions_on_same_future_examples",
    )
    eligible = [
        row for row in gain_rows if row["realized_gain"] is not None and row["adaptation_cost"] is not None
    ]
    total_cost = sum(row["adaptation_cost"] for row in eligible)
    available(
        "adaptation_roi_aggregate",
        sum(row["realized_gain"] for row in eligible) / total_cost if total_cost else None,
        "no_evaluated_shadows_with_positive_priced_adaptation_cost",
        "eligible_evaluated_priced_shadow_intervals_only_not_all_updates",
    )
    eligible_seqs = {row["seq"] for row in eligible}
    omitted = [u for u in updates if u["seq"] not in eligible_seqs]
    omitted_costs = [adaptation_cost(u) for u in omitted]
    all_comparable = (bool(updates) and len(gain_rows) == len(updates) and not omitted
                      and all(row["seq"] == update["seq"]
                              and row["train_samples"] == update["train_samples"]
                              for row, update in zip(gain_rows, updates)))
    available("adaptation_roi_global",
              sum(row["realized_gain"] for row in eligible) / total_cost
              if all_comparable and total_cost else None,
              "requires_all_updates_priced_with_evaluated_future_shadow_horizons",
              "all_postwarmup_updates_excluding_initial_fit_and_evaluation_overhead")
    available("adaptation_roi_eligible_updates", len(eligible))
    available("adaptation_roi_omitted_updates", len(omitted))
    available("adaptation_roi_eligible_cost", total_cost if costs else None, "cost_profile_not_configured")
    available("adaptation_roi_omitted_priced_cost",
              sum(omitted_costs) if all(c is not None for c in omitted_costs) and costs else None,
              "omitted_update_costs_not_fully_priced")
    rois = [row for row in gain_rows if row["roi"] is not None]
    impacted = sum(row["evaluated"] for row in rois)
    available(
        "adaptation_roi_mean",
        sum(row["roi"] for row in rois) / len(rois) if rois else None,
        "no_evaluated_shadows_with_positive_priced_adaptation_cost",
    )
    available(
        "adaptation_roi_prediction_weighted",
        sum(row["roi"] * row["evaluated"] for row in rois) / impacted if impacted else None,
        "no_evaluated_shadows_with_positive_priced_adaptation_cost",
    )
    drops = [
        row["prechange_accuracy"] - row["first_postchange_accuracy"]
        for row in recovery_details
        if row.get("first_postchange_accuracy") is not None
    ]
    available(
        "degradation",
        sum(drops) / len(drops) if drops else None,
        "no_complete_pre_and_postchange_windows",
        "prechange_accuracy_minus_first_full_postchange_window",
    )
    reference = settings.get("reference")
    if reference is not None:
        required = {"dataset_sha256", "protocol", "evidence_uri", "evidence_sha256"}
        if not required <= reference.keys() or reference["dataset_sha256"] != report["dataset"]["sha256"]:
            raise ValueError("metric reference must identify matching dataset and evidence")
        evidence_sha = reference["evidence_sha256"]
        if (reference["protocol"] != report["protocol"] or not isinstance(evidence_sha, str)
                or len(evidence_sha) != 64 or any(c.lower() not in "0123456789abcdef" for c in evidence_sha)):
            raise ValueError("metric reference protocol/evidence hash mismatch")
        if not isinstance(reference["evidence_uri"], str) or not reference["evidence_uri"].strip():
            raise ValueError("metric reference requires a nonempty evidence URI")
        scope = {
            "evaluation_n": report["metrics"]["n"],
            "warmup": config["warmup"],
            "metric_window": config["metric_window"],
            "batch_size": config["batch_size"],
            "split": report.get("split"),
            "training_budget": config.get("training_budget"),
        }
        if reference.get("evaluation_scope") != scope:
            raise ValueError("metric reference evaluation scope mismatch")
        if "oracle_cost" in reference:
            if costs is None or reference.get("cost_settings") != costs:
                raise ValueError("Oracle cost profile differs")
            available(
                "cost_scpe",
                relative_cost_error(cost_total, reference["oracle_cost"]) if cost_total is not None else None,
                "missing_total_cost_or_zero_oracle",
            )
        if "baseline_curve" in reference:
            target = reference["baseline_target"]
            if type(target) not in (float, int) or not math.isfinite(target) or not 0 <= target <= 1:
                raise ValueError("baseline target must be finite in [0,1]")
            baseline = validate_learning_curve(reference["baseline_curve"])
            curve_scope = reference.get("curve_scope")
            if curve_scope == "window_end_source_offset_including_warmup":
                method_curve = curve
                names = ((1, "prequential_samples_ratio_l100"), (0.99, "prequential_samples_ratio_l99"))
            elif curve_scope == "training_iteration_fixed_test":
                test_sha = reference.get("test_dataset_sha256", "")
                if (not isinstance(test_sha, str) or len(test_sha) != 64
                        or any(c.lower() not in "0123456789abcdef" for c in test_sha)):
                    raise ValueError("fixed-test curves require a test dataset SHA256")
                method_curve = validate_learning_curve(reference.get("method_curve"))
                names = ((1, "learning_speed_l100"), (0.99, "learning_speed_l99"))
                available("iterations_to_target", first_target(method_curve, settings["target_accuracy"]),
                          "target_not_reached_at_recorded_fixed_test_iterations",
                          "externally_provided_training_iterations_same_fixed_test:" + test_sha)
            else:
                raise ValueError("metric reference curve scope mismatch")
            for ratio, name in names:
                available(
                    name,
                    learning_speed_ratio(method_curve, baseline, target, ratio),
                    "target_not_reached_or_zero_iteration",
                    curve_scope,
                )
    for key, reason in {
        "cost_scpe": "requires_compatible_oracle_cost_and_provenance",
        "learning_speed_l99": "requires_compatible_baseline_curve_and_target",
        "learning_speed_l100": "requires_compatible_baseline_curve_and_target",
        "prequential_samples_ratio_l99": "requires_prequential_reference_curve",
        "prequential_samples_ratio_l100": "requires_prequential_reference_curve",
        "drift_roc_auc": "requires_continuous_drift_scores_and_labeled_drift_trials",
        "drift_false_positive_rate": "requires_defined_negative_drift_trials_use_unmatched_alarm_rate",
        "upf_paper_auc": "requires_Oracle_defined_alpha_max_and_policy_campaign_across_alpha",
        "performance_forecast_mae": "no_performance_forecaster_or_forecast_targets",
        "performance_forecast_bias": "no_performance_forecaster_or_forecast_targets",
        "platform_time_overhead_ratio": "requires_equivalent_direct_execution_baseline",
        "platform_disk_overhead_ratio": "requires_equivalent_direct_execution_baseline",
        "financial_ros": "requires_real_revenue_and_operating_costs",
        "financial_infrastructure_cost": "no_metered_resource_prices",
        "label_acquisition_cost": "no_observed_label_acquisition_prices",
        "communication_cost": "no_metered_communication_volume_and_prices",
        "energy_kwh": "no_energy_meter",
        "carbon_emissions": "no_energy_meter_and_carbon_intensity",
        "ci_delivery_seconds": "not_a_CI_delivery_execution",
        "ci_build_seconds": "not_a_CI_build_execution",
        "ci_test_seconds": "not_a_CI_test_execution",
        "ci_test_coverage": "no_coverage_measurement_for_this_run",
        "ci_standard_violations": "no_static_analysis_measurement_for_this_run",
    }.items():
        if key not in metrics:
            available(key, None, reason)
    aliases = {
        "drift_alarm_count": "alarm_count",
        "drift_true_positives": "drift_tp",
        "drift_false_positives": "drift_fp",
        "drift_false_negatives": "drift_fn",
        "drift_detection_delay": "drift_delay_mean_samples",
        "drift_score_roc_auc": "drift_roc_auc",
        "recovery_time": "recovery_delay_mean_samples",
        "adaptation_count": "update_count",
        "realized_adaptation_gain": "adaptation_realized_gain",
        "adaptation_roi": "adaptation_roi_mean",
        "adaptation_roi_weighted": "adaptation_roi_prediction_weighted",
        "scenario_total_cost": "cost_total",
        "scpe": "cost_scpe",
        "upf_paper_cost_auc": "upf_paper_auc",
        "prediction_latency": "prediction_latency_mean_seconds",
        "prediction_throughput": "prediction_throughput_per_second",
        "train_samples": "training_presentations",
        "peak_rss_bytes": "process_peak_rss_bytes",
        "ram_time_bytes_seconds": "sampled_rss_byte_seconds",
        "reference_train_seconds": "cl_reference_train_seconds",
        "reference_train_samples": "cl_reference_train_samples",
        "evaluation_overhead_seconds": "instrumentation_seconds",
        "l99": "learning_speed_l99",
        "l100": "learning_speed_l100",
        "platform_runtime_overhead": "platform_time_overhead_ratio",
        "platform_disk_overhead": "platform_disk_overhead_ratio",
        "build_seconds": "ci_build_seconds",
        "test_seconds": "ci_test_seconds",
        "test_coverage": "ci_test_coverage",
        "code_standard_violations": "ci_standard_violations",
        "return_on_sales": "financial_ros",
        "infrastructure_cost_brl": "financial_infrastructure_cost",
        "labeling_cost_brl": "label_acquisition_cost",
        "communication_cost_brl": "communication_cost",
    }
    for alias, source in aliases.items():
        if source in metrics:
            metrics[alias] = metrics[source]
            availability[alias] = {**availability[source], "alias_of": source}
    catalog, catalog_raw = metric_catalog()
    for name, definition in catalog["definitions"].items():
        if name not in metrics and name not in series:
            available(name, None, "; ".join(definition["na_reasons"]))
    return {
        "version": VERSION,
        "settings": settings,
        "metrics": metrics,
        "series": series,
        "availability": availability,
        "continual": continual,
        "retention_probes": copy.deepcopy(retention),
        "detection": detection_details,
        "recovery": recovery_details,
        "cost_sensitivity": sensitivity,
        "series_metadata": {
            **{name: {
                "step": "observed_seq",
                "duplicate_step_policy": "latest probe at equal source sequence (final label drain wins)",
                "raw_probes": "retention_probes",
            } for name in retention_series},
            "posthoc_cost_sensitivity": {
                "step": "zero-based alpha grid index", "alpha_grid": [r["alpha"] for r in sensitivity],
                "cost_basis": "all update events including learn_one; fixed policy trajectory",
                "unit": "normalized_scenario_cost", "paper_UPF_AUC": False,
            },
        },
        "adaptation": gain_rows,
        "adaptation_coverage": {"updates": len(updates), "eligible_updates": len(eligible),
                                "omitted_update_seqs": [u["seq"] for u in omitted],
                                "all_updates_comparable": all_comparable},
        "definitions_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "aliases": aliases,
        "scopes": {
            "prediction_latency": "model.predict elapsed seconds, evaluated examples only",
            "memory": observation.get("memory_sampling", "current RSS integral not captured"),
            "reference_models": "one-pass diagnostics; independent budget and overhead ledger",
            "cost": "configured scenario coefficients, not measured money unless externally priced",
        },
    }


def save_tracking_files(directory, report, settings, observer=None, events=None):
    from continual_bench.adapters.local import atomic_write

    directory = Path(directory)
    observation = observer.report() if observer is not None else None
    continual = (
        observer.continual.report() if observer is not None and observer.continual is not None else None
    )
    predictions = observer.predictions if observer is not None else None
    tracking = build_tracking(report, settings, observation, continual, predictions)
    _, definitions_raw = metric_catalog()
    atomic_write(directory / "metric-definitions.json", definitions_raw)
    report["tracking"] = tracking
    if predictions is not None:
        raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions).encode()
        atomic_write(directory / "predictions.jsonl", raw)
        tracking["prediction_ledger"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(predictions),
            "selection": "postwarmup_or_temporal_holdout_released_labels",
        }
    if observer is not None and observer.continual is not None:
        from continual_bench.adapters.versioning import ModelVersions

        probes = {
            "schema_version": 1,
            "dataset_sha256": report["dataset"]["sha256"],
            "code_sha256": report["environment"]["code_sha256"],
            "task_ends": observer.continual.task_ends,
            "episodes": [{"episode": episode, "rows": [{"features": x, "label": bool(y)} for x, y in rows]}
                         for episode, rows in sorted(observer.continual.probes.items())],
            "protocol": "heldout_episode_probes_never_trained_independent_of_recurrent_concept_identity",
        }
        probes_raw = json.dumps(probes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        atomic_write(directory / "continual-probes.json", probes_raw)
        tracking["probe_dataset"] = {
            "sha256": hashlib.sha256(probes_raw).hexdigest(),
            "rows": sum(len(rows) for rows in observer.continual.probes.values()),
            "episodes": len(observer.continual.probes),
            "path": "continual-probes.json", "protocol": probes["protocol"],
            "source_dataset_sha256": probes["dataset_sha256"], "code_sha256": probes["code_sha256"],
        }
        references = []
        for item in observer.continual.reference_models:
            path = directory / "references" / f"{item['name']}-{item['task']}"
            versions = ModelVersions(path, report["dataset"]["sha256"], report["environment"]["code_sha256"])
            versions.record(
                item["model"], {"action": "reference_initial_fit", "seq_range": item["seq_range"]}
            )
            probe = observer.continual.probes[item["task"]][0][0]
            metadata = versions.finish(item["model"], probe)
            references.append({k: v for k, v in item.items() if k != "model"} | {"metadata": metadata})
        tracking["reference_models"] = references
    return tracking
