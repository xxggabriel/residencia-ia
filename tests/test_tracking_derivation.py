"""Metric derivation must distinguish measured zero, missing evidence and scoped references."""

import json

import pytest

from continual_bench.adapters.metric_tracking import metric_points
from continual_bench.application.config import TrackingConfig
from continual_bench.application.tracking import (
    build_tracking,
    confusion_metrics,
    detector_metrics,
    recovery_metrics,
)
from continual_bench.domain.contracts import Event
from continual_bench.domain.metrics import Metrics
from continual_bench.domain.observability import (
    OnlineObservability,
    first_target,
    learning_speed_ratio,
    quantile,
    relative_cost_error,
)


def settings(**changes):
    return TrackingConfig.model_validate(changes).model_dump(mode="json")


def evidence():
    accumulator = Metrics()
    for y, probability in [(True, 0.8), (False, 0.8), (True, 0.2), (False, 0.2)] * 2:
        accumulator.add(y, probability)
    return {
        "metrics": accumulator.values(),
        "observed": 10,
        "config": {"warmup": 2, "metric_window": 2, "batch_size": 1, "training_budget": 8,
                   "dataset": {"epoch": 1000}, "detector": {"name": "adwin"}},
        "dataset": {"sha256": "a" * 64, "changepoints": [4]},
        "protocol": "event_prequential",
        "windows": [{"end_seq": seq, "n": 2, "accuracy": accuracy}
                    for seq, accuracy in [(3, 0.4), (5, 0.6), (7, 0.8), (9, 0.9)]],
        "updates": [{"seq": 3, "action": "reset_retrain", "train_samples": 4, "seconds": 2},
                    {"seq": 5, "action": "reset_retrain", "train_samples": 4, "seconds": 3}],
        "alarms": [],
        "training": {"blocked_decisions": 0},
        "retention_probes": [],
        "compute_seconds": 9,
        "wall_seconds_this_attempt": 10,
        "peak_rss_bytes_process": 4096,
    }


def reference(report, **changes):
    value = {
        "dataset_sha256": report["dataset"]["sha256"],
        "protocol": report["protocol"],
        "evidence_uri": "fixture://external-reference",
        "evidence_sha256": "b" * 64,
        "evaluation_scope": {"evaluation_n": report["metrics"]["n"],
                             "warmup": report["config"]["warmup"],
                             "metric_window": report["config"]["metric_window"],
                             "batch_size": report["config"]["batch_size"],
                             "split": report.get("split"),
                             "training_budget": report["config"].get("training_budget")},
    }
    value.update(changes)
    return value


def test_historical_confusion_recovers_counts_without_inventing_scores():
    report = evidence()
    original = {key: report["metrics"][key] for key in ("n", "accuracy", "f1", "brier", "log_loss")}
    derived = confusion_metrics(original, positives=4)
    assert {key: derived[key] for key in ("tp", "fp", "tn", "fn")} == dict.fromkeys(
        ("tp", "fp", "tn", "fn"), 2
    )
    assert all(derived[key] is None for key in ("roc_auc", "average_precision", "ece", "mce"))
    assert {key: derived[key] for key in original} == original
    assert derived["balanced_accuracy"] == 0.5


@pytest.mark.parametrize("changes,positives", [({"accuracy": 0.123}, 4), ({"f1": 0.9}, 0),
                                             ({"f1": 0}, 8)])
def test_impossible_historical_confusion_rejected(changes, positives):
    summary = {"n": 8, "accuracy": 0.5, "f1": 0.5}
    summary.update(changes)
    with pytest.raises(ValueError):
        confusion_metrics(summary, positives)


def test_cost_prices_presentations_including_warmup_and_replay():
    costs = {"false_positive": 2, "false_negative": 3, "train_sample": 0.5, "update": 7}
    tracking = build_tracking(evidence(), settings(costs=costs))
    metrics = tracking["metrics"]
    assert metrics["training_presentations"] == 10
    assert metrics["training_unique_examples"] == 6
    assert metrics["training_repeated_presentations"] == 4
    assert metrics["training_used_post_warmup"] == 8
    assert metrics["training_budget_utilization"] == 1
    assert metrics["cost_prediction_errors"] == 10
    assert metrics["cost_training_presentations"] == 5
    assert metrics["cost_update_events"] == 14
    assert metrics["cost_total"] == 29
    assert metrics["initial_train_seconds"] is metrics["train_seconds"] is None
    assert metrics["update_train_seconds"] == 5
    assert metrics["training_budget_blocked_decisions"] == 0
    assert tracking["availability"]["training_budget_blocked_decisions"]["status"] == "available"


def test_offline_cost_uses_purged_initial_fit_not_warmup_or_zero_updates():
    report = evidence()
    report.update(split={"train": [0, 69], "purged_label_delay": [69, 72], "test": [72, 120]},
                  updates=[], training={})
    report["config"]["training_budget"] = None
    tracking = build_tracking(report, settings(costs={"train_sample": 2}))
    metrics = tracking["metrics"]
    assert metrics["training_presentations"] == metrics["training_unique_examples"] == 69
    assert metrics["training_repeated_presentations"] == metrics["training_used_post_warmup"] == 0
    assert metrics["cost_training_presentations"] == 138
    assert metrics["training_budget_blocked_decisions"] is None


def test_nonzero_time_prices_require_timers_but_measured_zero_is_available():
    configured = settings(costs={"train_second": 2, "predict_second": 3})
    absent = build_tracking(evidence(), configured)
    assert absent["metrics"]["cost_total"] is None
    assert absent["metrics"]["prediction_seconds"] is None
    measured = build_tracking(evidence(), configured,
                              observation={"warmup_train_seconds": 0, "prediction_seconds": 0})
    assert measured["metrics"]["initial_train_seconds"] == 0
    assert measured["metrics"]["train_seconds"] == 5
    assert measured["metrics"]["cost_total"] == 14  # Four errors and 5 measured train seconds.
    assert measured["availability"]["prediction_seconds"]["status"] == "available"


def test_missing_cost_and_unmeasured_quantities_are_null_not_zero():
    report = build_tracking(evidence(), settings())
    for key in ("cost_total", "cost_scpe", "energy_kwh", "financial_ros", "ci_test_coverage",
                "performance_forecast_mae", "drift_roc_auc", "platform_time_overhead_ratio"):
        assert report["metrics"][key] is None
        assert report["availability"][key]["status"] == "unavailable"
        assert report["availability"][key]["reason"]
    assert report["metrics"]["recovery_recovered_changes"] == 1
    assert report["metrics"]["mse_probability"] == evidence()["metrics"]["brier"]
    assert report["metrics"]["mae_probability"] is None
    assert report["availability"]["cost_sensitivity_auc"]["scope"] != "UPF_oracle_campaign_AUC"


def test_detectors_match_earliest_unused_alarm_and_count_duplicates_as_false_alarms():
    report = evidence()
    report["observed"] = 120
    report["metrics"]["n"] = 118  # Evaluated offsets [2,120), consistent with the alarm at 12.
    report["dataset"]["changepoints"] = [40, 80]
    report["alarms"] = [{"seq": seq, "released_at": 1000 + seq + 3}
                        for seq in (45, 40, 40, 12, 80, 90)]
    metrics, details, reason = detector_metrics(report, settings(detection_window=10))
    assert reason is None
    assert [item["alarm_seq"] for item in details["matches"]] == [40, 80]
    assert (metrics["drift_tp"], metrics["drift_fp"], metrics["drift_fn"]) == (2, 4, 0)
    assert metrics["drift_precision"] == pytest.approx(1 / 3)
    assert metrics["drift_recall"] == 1
    assert metrics["drift_f1"] == 0.5
    assert metrics["drift_delay_mean_samples"] == 0
    assert metrics["drift_delay_release_mean"] == 3
    assert metrics["drift_unmatched_alarms_per_1000_evaluated"] == pytest.approx(4000 / 118)
    assert metrics["drift_false_alarm_rate"] == pytest.approx(2 / 98)
    assert metrics["drift_stable_exposure_samples"] == 98


def test_detection_windows_are_exclusive_and_do_not_cross_next_change_or_end():
    report = evidence()
    report["observed"] = 20
    report["metrics"]["n"] = 18
    report["dataset"]["changepoints"] = [5, 10, 19]
    report["alarms"] = [{"seq": 10}, {"seq": 20}]
    metrics, details, _ = detector_metrics(report, settings(detection_window=100))
    assert [item["alarm_seq"] for item in details["matches"]] == [None, 10, None]
    assert details["matches"][0]["censored"] is False
    assert details["matches"][2]["censored"] is True
    assert metrics["drift_tp"] == 1 and metrics["drift_fn"] == 1
    assert metrics["drift_censored_changes"] == 1
    assert metrics["drift_delay_release_mean"] is None


@pytest.mark.parametrize("disabled", [False, True])
def test_real_drift_ground_truth_missing_and_disabled_detector_are_explicit(disabled):
    report = evidence()
    report["dataset"]["changepoints"] = None
    if disabled:
        report["config"]["detector"]["name"] = "none"
    metrics, _, reason = detector_metrics(report, settings())
    assert all(value is None for value in metrics.values())
    assert reason == ("detector_disabled" if disabled else "no_ground_truth_changepoints")


def test_no_alarms_means_zero_recall_and_undefined_precision_not_perfect_precision():
    metrics, _, _ = detector_metrics(evidence(), settings(detection_window=6))
    assert metrics["drift_precision"] is metrics["drift_delay_mean_samples"] is None
    assert metrics["drift_recall"] == metrics["drift_f1"] == 0
    assert metrics["drift_false_alarm_rate_per_1000"] == 0


def test_recovery_requires_two_complete_consecutive_postchange_windows():
    report = evidence()
    report["dataset"]["changepoints"] = [4]
    report["observed"] = 13
    report["windows"] = [{"end_seq": 3, "n": 2, "accuracy": 0.8},
                         {"end_seq": 4, "n": 2, "accuracy": 1},  # Straddles the boundary.
                         {"end_seq": 6, "n": 2, "accuracy": 0.9},
                         {"end_seq": 8, "n": 2, "accuracy": 0.5},  # Resets consecutive streak.
                         {"end_seq": 10, "n": 2, "accuracy": 0.9},
                         {"end_seq": 11, "n": 1, "accuracy": 1}]  # Partial is insufficient.
    metrics, rows = recovery_metrics(report, settings(recovery_tolerance=0, recovery_windows=2))
    assert metrics["recovery_delay_mean_samples"] is None
    assert metrics["recovery_recovered_changes"] == 0
    assert metrics["recovery_censored_changes"] == 1
    assert rows[0]["censored"] is True
    report["windows"][-1] = {"end_seq": 12, "n": 2, "accuracy": 0.9}
    metrics, _ = recovery_metrics(report, settings(recovery_tolerance=0, recovery_windows=2))
    assert metrics["recovery_delay_mean_samples"] == 8


def test_recovery_is_censored_at_next_change_and_needs_prechange_reference():
    report = evidence()
    report["dataset"]["changepoints"] = [0, 4, 6]
    metrics, rows = recovery_metrics(report, settings(recovery_windows=2))
    assert rows[0]["reason"] == "missing_prechange_window"
    assert rows[1]["delay"] is None and rows[1]["horizon_end"] == 6
    assert metrics["recovery_censored_changes"] == 1
    assert metrics["recovery_unavailable_changes"] == 1
    assert rows[0]["available"] is rows[0]["censored"] is False


def test_scpe_and_learning_speed_require_comparable_reference_scope():
    report = evidence()
    configured = settings(costs={"train_sample": 0.5})
    costs = configured["costs"]
    configured["reference"] = reference(
        report, oracle_cost=20, cost_settings=costs,
        baseline_curve=[{"step": 4, "value": 0.6}, {"step": 8, "value": 0.8}],
        baseline_target=0.8, curve_scope="window_end_source_offset_including_warmup",
    )
    result = build_tracking(report, configured)
    assert result["metrics"]["cost_scpe"] == -55  # Cost4(errors)+5(training)=9 versus20.
    assert result["metrics"]["prequential_samples_ratio_l100"] == 1
    assert result["metrics"]["prequential_samples_ratio_l99"] == 1
    assert result["metrics"]["l99"] is result["metrics"]["l100"] is None


@pytest.mark.parametrize("field,value", [
    ("dataset_sha256", "c" * 64), ("protocol", "temporal_holdout"),
    ("evidence_sha256", "bad"), ("evaluation_scope", {}),
    ("curve_scope", "labeled_window_index"), ("cost_settings", {}),
])
def test_incompatible_reference_is_rejected(field, value):
    report = evidence()
    configured = settings(costs={})
    configured["reference"] = reference(
        report, oracle_cost=1, cost_settings=configured["costs"],
        baseline_curve=[{"step": 8, "value": 0.8}], baseline_target=0.8,
        curve_scope="window_end_source_offset_including_warmup",
    )
    configured["reference"][field] = value
    with pytest.raises(ValueError):
        build_tracking(report, configured)


def test_metric_points_namespace_is_versioned_and_unavailable_values_stay_null():
    tracking = build_tracking(evidence(), settings())
    points = metric_points(tracking)
    assert all(key.startswith("tracking_v1/") for key, step, value in points)
    assert ("tracking_v1/summary/cost_total", 0, None) in points
    assert ("tracking_v1/summary/training_budget_blocked_decisions", 0, 0) in points
    assert ("tracking_v1/accuracy", 3, 0.4) in points
    json.dumps(tracking, allow_nan=False)


def test_observability_shadow_compares_old_and_new_on_identical_future_events():
    class ConstantModel:
        def __init__(self, probability):
            self.probability = probability
            self.calls = []

        def predict(self, x):
            self.calls.append(x["row"])
            return self.probability

    configured = settings(adaptation_shadows=True, costs={"false_negative": 3, "false_positive": 2})
    observer = OnlineObservability(configured)
    old, new = ConstantModel(0.1), ConstantModel(0.9)
    prior = Event("prior", 1, 1, 4, {"row": 1}, True)
    prior_trace = observer.before_prediction(prior, old, 0, 0.01)
    previous = observer.before_update(old)
    observer.after_update(previous, {"seq": 0, "train_samples": 1, "seconds": 1})
    # This prediction was made before the update; a late label cannot enter its comparison.
    observer.after_label(prior, 0.1, prior_trace, 4, new, True)
    future = Event("future", 2, 2, 5, {"row": 2}, True)
    trace = observer.before_prediction(future, new, 1, 0.02)
    original_shadow = observer.shadow
    observer.after_update(observer.before_update(new), {"seq": 1, "train_samples": 1, "seconds": 1})
    # A second update before this label arrives must not reassign its comparison.
    observer.after_label(future, 0.9, trace, 5, new, True)
    rows = observer.report()["comparisons"]
    assert rows[0]["evaluated"] == 1
    assert rows[1]["evaluated"] == 0
    assert rows[0]["old_errors"] == 1 and rows[0]["new_errors"] == 0
    assert rows[0]["old_weighted_loss"] == 3 and rows[0]["new_weighted_loss"] == 0
    assert old.calls == new.calls == []
    assert original_shadow.calls == [2]
    assert observer.shadow.calls == []
    derived = build_tracking(evidence(), configured, observation=observer.report())
    assert derived["metrics"]["adaptation_realized_gain"] == 3
    assert derived["metrics"]["adaptation_roi_aggregate"] is None  # Zero priced adaptation.


def test_runtime_empty_zero_and_missing_memory_are_distinct():
    observer = OnlineObservability(settings())
    empty = observer.report()
    assert empty["prediction_seconds"] is empty["prediction_latency_mean_seconds"] is None
    assert empty["sampled_rss_byte_seconds"] is None
    event = Event("x", 0, 0, 0, {}, True)
    trace = observer.before_prediction(event, None, 0, 0)
    observer.after_label(event, 0.9, trace, 0, None, True)
    measured = observer.report()
    assert measured["prediction_seconds"] == measured["prediction_latency_mean_seconds"] == 0
    assert measured["prediction_throughput_per_second"] is None
    assert measured["fading_accuracy"] == 1
    assert quantile([], 0.5) is None
    assert quantile([1, 4, 2, 3], 0.5) == 2.5


@pytest.mark.parametrize("samples,expected", [([0, 0], 0), ([100, 300], 400)])
def test_sampled_rss_integral_preserves_zero_and_trapezoidal_units(monkeypatch, samples, expected):
    readings = iter(samples)
    times = iter([1, 3])
    observer = OnlineObservability(settings(), memory_sampler=lambda: next(readings))
    monkeypatch.setattr("continual_bench.domain.observability.time.perf_counter", lambda: next(times))
    observer.sample_memory()
    observer.sample_memory()
    assert observer.report()["sampled_rss_byte_seconds"] == expected


def test_observability_helpers_have_explicit_zero_reference_and_target_censoring():
    assert relative_cost_error(15, 10) == 50
    assert relative_cost_error(0, 0) is None
    curve = [{"step": 3, "value": None}, {"step": 5, "value": 0.8}]
    assert first_target(curve, 0.9) is None
    assert learning_speed_ratio(curve, [{"step": 10, "value": 0.8}], 0.8) == 2
    assert learning_speed_ratio(curve, [{"step": 10, "value": 0.8}], 0.9) is None
    for cost, oracle in [(-1, 1), (1, -1), (float("nan"), 1), (1, float("inf"))]:
        with pytest.raises(ValueError):
            relative_cost_error(cost, oracle)


def test_censored_tail_detections_and_alarms_do_not_enter_complete_followup_counts():
    report = evidence()
    report.update(observed=20)
    report['metrics']['n'] = 18
    report['dataset']['changepoints'] = [5, 15]
    report['alarms'] = [{'seq': value} for value in (3, 6, 16, 17)]
    values, details, _ = detector_metrics(report, settings(detection_window=10))
    assert (values['drift_tp'], values['drift_fp'], values['drift_fn']) == (1, 1, 0)
    assert values['drift_precision'] == 0.5
    assert values['drift_recall'] == 1
    assert values['drift_complete_followup_changes'] == values['drift_censored_changes'] == 1
    assert values['drift_observed_detections'] == 2
    assert details['matches'][1]['alarm_seq'] == 16
    assert details['matches'][1]['censored'] is True
    assert details['censored_alarm_count'] == 2
    assert values['drift_stable_exposure_samples'] == 3
    assert values['drift_false_alarm_rate'] == pytest.approx(1 / 3)


def test_known_stable_ground_truth_is_distinct_from_unknown_real_ground_truth():
    report = evidence()
    report['dataset']['changepoints'] = []
    report['alarms'] = [{'seq': 3}]
    values, _, reason = detector_metrics(report, settings())
    assert reason is None
    assert values['drift_stable_exposure_samples'] == 8
    assert values['drift_false_alarm_rate'] == 1 / 8
    assert values['drift_recall'] is None
    report['dataset']['changepoints'] = None
    values, _, reason = detector_metrics(report, settings())
    assert values['drift_false_alarm_rate'] is None
    assert reason == 'no_ground_truth_changepoints'


def test_binary_probability_extensions_are_not_regression_aliases():
    report = evidence()
    predictions = [{'label': True, 'probability': 0.8}, {'label': False, 'probability': 0.3}]
    tracking = build_tracking(report, settings(), predictions=predictions)
    assert tracking['metrics']['mae_probability'] == pytest.approx(0.25)
    assert tracking['metrics']['mse_probability'] == report['metrics']['brier']
    assert tracking['metrics']['mae'] is tracking['metrics']['mse'] is None
    assert 'mae' not in tracking['aliases'] and 'mse' not in tracking['aliases']


def test_upf_retraining_cost_and_posthoc_update_grid_have_separate_contracts():
    report = evidence()
    configured = settings(upf_alpha=2, upf_alphas=[0, 1, 2])
    frozen = build_tracking(report, configured)
    errors = sum(1 - row['accuracy'] for row in report['windows'])
    assert frozen['metrics']['upf_total_cost'] == pytest.approx(4 + errors)
    report['updates'][0]['action'] = 'learn_one'
    incremental = build_tracking(report, configured)
    assert incremental['metrics']['upf_total_cost'] is None
    assert incremental['metrics']['posthoc_update_event_cost'] == pytest.approx(4 + errors)
    assert 'posthoc_cost_sensitivity' not in incremental['metrics']
    assert incremental['series']['posthoc_cost_sensitivity'] == [
        {'step': i, 'value': 2 * alpha + errors} for i, alpha in enumerate([0, 1, 2])
    ]
    metadata = incremental['series_metadata']['posthoc_cost_sensitivity']
    assert metadata['alpha_grid'] == [0, 1, 2]
    assert metadata['paper_UPF_AUC'] is False
    assert incremental['metrics']['upf_paper_cost_auc'] is None
    report['updates'] = []
    report['config']['policy'] = {'name': 'online'}
    assert build_tracking(report, configured)['metrics']['upf_total_cost'] is None


def test_global_roi_requires_every_update_to_have_priced_evaluated_shadow_horizon():
    report = evidence()
    configured = settings(costs={'train_sample': 1}, adaptation_shadows=True)
    comparison = {**report['updates'][0], 'evaluated': 2, 'old_weighted_loss': 2, 'new_weighted_loss': 1}
    partial = build_tracking(report, configured, observation={'comparisons': [comparison]})
    values = partial['metrics']
    assert values['adaptation_roi_aggregate'] == 1 / 4
    assert values['adaptation_roi_global'] is None
    assert values['adaptation_roi_omitted_updates'] == 1
    assert values['adaptation_roi_omitted_priced_cost'] == 4
    assert partial['adaptation_coverage']['omitted_update_seqs'] == [5]
    unseen = {**report['updates'][1], 'evaluated': 0, 'old_weighted_loss': 0, 'new_weighted_loss': 0}
    tail = build_tracking(report, configured, observation={'comparisons': [comparison, unseen]})
    assert tail['metrics']['adaptation_roi_global'] is None
    unseen.update(evaluated=2, old_weighted_loss=3, new_weighted_loss=1)
    complete = build_tracking(report, configured, observation={'comparisons': [comparison, unseen]})
    assert complete['metrics']['adaptation_roi_global'] == 3 / 8
    assert complete['metrics']['adaptation_roi_omitted_priced_cost'] == 0
    assert complete['adaptation_coverage']['all_updates_comparable'] is True


def test_canonical_speed_requires_fixed_test_training_iterations_not_stream_offsets():
    report = evidence()
    configured = settings(target_accuracy=0.8)
    configured['reference'] = reference(
        report, curve_scope='training_iteration_fixed_test', test_dataset_sha256='d' * 64,
        baseline_target=0.8, baseline_curve=[{'step': 10, 'value': 0.7}, {'step': 20, 'value': 0.8}],
        method_curve=[{'step': 5, 'value': 0.8}, {'step': 8, 'value': 0.9}],
    )
    tracking = build_tracking(report, configured)
    assert tracking['metrics']['iterations_to_target'] == 5
    assert tracking['metrics']['l99'] == tracking['metrics']['l100'] == 4
    assert tracking['metrics']['prequential_samples_ratio_l100'] is None
    configured['reference']['test_dataset_sha256'] = 'not-a-version'
    with pytest.raises(ValueError, match='test dataset SHA256'):
        build_tracking(report, configured)


@pytest.mark.parametrize('curve', [[], [{'step': -1, 'value': 0.8}],
                                   [{'step': 2, 'value': float('nan')}],
                                   [{'step': 2, 'value': 0.8}, {'step': 1, 'value': 0.9}]])
def test_reference_learning_curves_reject_invalid_order_or_scores(curve):
    report = evidence()
    configured = settings()
    configured['reference'] = reference(report, curve_scope='window_end_source_offset_including_warmup',
                                        baseline_target=0.8, baseline_curve=curve)
    with pytest.raises(ValueError, match='learning curve'):
        build_tracking(report, configured)


def test_episode_probe_dataset_is_saved_with_independent_version_and_source_lineage(tmp_path):
    import hashlib

    from continual_bench.application.config import Config
    from continual_bench.application.runner import run

    config = Config.model_validate({
        'name': 'probe-evidence', 'mode': 'streaming',
        'dataset': {'name': 'sea', 'samples': 120, 'label_delay': 2},
        'warmup': 20, 'metric_window': 20, 'replay_window': 30,
        'policy': {'name': 'none', 'min_window': 10}, 'tracking': {'probe_samples': 3},
    })
    report = run(config, tmp_path)
    metadata = report['tracking']['probe_dataset']
    raw = (tmp_path / config.identity() / 'continual-probes.json').read_bytes()
    probes = json.loads(raw)
    assert metadata['sha256'] == hashlib.sha256(raw).hexdigest()
    assert metadata['rows'] == sum(len(row['rows']) for row in probes['episodes']) == 9
    assert probes['task_ends'] == [40, 80, 120]
    assert [row['episode'] for row in probes['episodes']] == [0, 1, 2]
    assert probes['dataset_sha256'] == metadata['source_dataset_sha256'] == report['dataset']['sha256']
    assert probes['code_sha256'] == metadata['code_sha256'] == report['environment']['code_sha256']


@pytest.mark.parametrize('f1', [1.01, -0.01, float('nan'), float('inf')])
def test_historical_f1_out_of_range_is_rejected(f1):
    with pytest.raises(ValueError, match='historical'):
        confusion_metrics({'n': 8, 'accuracy': 0.5, 'f1': f1}, positives=4)


@pytest.mark.parametrize('field,value', [('evidence_sha256', 'z' * 64), ('evidence_sha256', None),
                                        ('evidence_uri', ''), ('evidence_uri', 3),
                                        ('baseline_target', float('nan')), ('baseline_target', -0.1),
                                        ('baseline_target', 1.1),
                                        ('baseline_curve', [{'step': True, 'value': 0.8}]),
                                        ('baseline_curve', [{'step': 2, 'value': True}]),
                                        ('baseline_curve', [{'step': 2}])])
def test_invalid_reference_provenance_and_curves_fail_before_derivation(field, value):
    report = evidence()
    configured = settings()
    configured['reference'] = reference(report, baseline_target=0.8,
                                        baseline_curve=[{'step': 2, 'value': 0.8}],
                                        curve_scope='window_end_source_offset_including_warmup')
    configured['reference'][field] = value
    with pytest.raises(ValueError):
        build_tracking(report, configured)


def test_fixed_test_speed_cannot_be_activated_without_method_curve():
    report = evidence()
    configured = settings()
    configured['reference'] = reference(report, baseline_target=0.8,
                                        baseline_curve=[{'step': 2, 'value': 0.8}],
                                        curve_scope='training_iteration_fixed_test',
                                        test_dataset_sha256='d' * 64)
    with pytest.raises(ValueError, match='learning curve'):
        build_tracking(report, configured)


def test_every_static_tracking_output_has_catalog_definition_and_no_semantic_alias():
    from continual_bench.application.tracking import metric_catalog

    catalog, _ = metric_catalog()
    tracking = build_tracking(evidence(), settings())
    assert (tracking['metrics'].keys() | tracking['series'].keys()) <= catalog['definitions'].keys()
    for alias in ('mae', 'mse', 'iterations_to_target', 'adaptation_roi_global',
                  'posthoc_cost_sensitivity', 'drift_false_alarm_rate'):
        assert alias not in tracking['aliases']
    assert catalog['definitions']['posthoc_cost_sensitivity']['representation'] == 'series'
    assert catalog['definitions']['drift_false_alarm_rate']['unit'] == 'alarms/example'


def test_historical_summary_without_probability_losses_does_not_invent_zero():
    derived = confusion_metrics({'n': 8, 'accuracy': 0.5, 'f1': 0.5}, positives=4)
    assert derived['brier'] is derived['log_loss'] is None
    assert derived['tp'] == derived['fp'] == derived['tn'] == derived['fn'] == 2


def test_metric_catalog_prefers_installed_package_asset_over_repository_fallback(tmp_path, monkeypatch):
    import continual_bench.application.tracking as module

    package = tmp_path / 'project' / 'src' / 'continual_bench'
    package.mkdir(parents=True)
    (package / 'metric-definitions.json').write_text('{"version":"package-asset"}')
    monkeypatch.setattr(module, '__file__', str(package / 'application' / 'tracking.py'))
    catalog, raw = module.metric_catalog()
    assert catalog == {'version': 'package-asset'}
    assert json.loads(raw) == catalog


def test_retention_curves_keep_latest_equal_step_but_preserve_every_raw_probe():
    import copy

    report = evidence()
    probes = [
        {'concept': 0, 'observed_seq': 5, 'accuracy': 0.4, 'drop_from_best_probe': 0.2},
        {'concept': 0, 'observed_seq': 9, 'accuracy': 0.6, 'drop_from_best_probe': 0.1},
        {'concept': 1, 'observed_seq': 9, 'accuracy': 0.5, 'drop_from_best_probe': 0.3},
        {'concept': 0, 'observed_seq': 9, 'accuracy': 0.9, 'drop_from_best_probe': 0.0},
        {'concept': 1, 'observed_seq': 9, 'accuracy': 0.7, 'drop_from_best_probe': 0.1},
    ]
    report['retention_probes'] = copy.deepcopy(probes)
    tracking = build_tracking(report, settings())
    assert tracking['retention_probes'] == report['retention_probes'] == probes
    assert tracking['retention_probes'] is not report['retention_probes']
    assert tracking['series']['retention_accuracy/concept_0'] == [
        {'step': 5, 'value': 0.4}, {'step': 9, 'value': 0.9},
    ]
    assert tracking['series']['retention_drop/concept_0'] == [
        {'step': 5, 'value': 0.2}, {'step': 9, 'value': 0.0},
    ]
    assert tracking['series']['retention_accuracy/concept_1'] == [{'step': 9, 'value': 0.7}]
    assert tracking['metrics']['retention_final_mean_accuracy'] == pytest.approx(0.8)
    metadata = tracking['series_metadata']['retention_accuracy/concept_0']
    assert metadata['duplicate_step_policy'] == (
        'latest probe at equal source sequence (final label drain wins)'
    )
    assert metadata['raw_probes'] == 'retention_probes'
    points = metric_points(tracking)
    assert [(step, value) for key, step, value in points
            if key == 'tracking_v1/retention_accuracy/concept_0'] == [(5, 0.4), (9, 0.9)]
    # Nonduplicate curves retain exactly their original values and source steps.
    report['retention_probes'] = probes[:3]
    unique = build_tracking(report, settings())
    assert unique['series']['retention_accuracy/concept_0'] == [
        {'step': 5, 'value': 0.4}, {'step': 9, 'value': 0.6},
    ]


def test_incremental_stagger_knn_final_drain_repeated_probe_step_can_publish(tmp_path):
    from collections import Counter

    from continual_bench.application.config import Config
    from continual_bench.application.runner import run

    config = Config.model_validate({
        'name': 'retention-final-drain', 'mode': 'incremental',
        'dataset': {'name': 'stagger', 'samples': 315, 'seed': 42, 'label_delay': 3},
        'model': {'name': 'knn', 'params': {'n_neighbors': 3, 'window_size': 32}},
        'warmup': 100, 'metric_window': 100, 'batch_size': 32, 'replay_window': 100,
        'training_budget': 215,
        'detector': {'name': 'none'},
        'policy': {'name': 'online', 'action': 'learn_one', 'min_samples': 100,
                   'min_window': 20, 'cooldown': 0},
        'tracking': {'continual_references': False, 'probe_samples': 4},
    })
    report = run(config, tmp_path)
    tracking = report['tracking']
    assert report['observed'] == report['labeled'] == 315
    assert tracking['retention_probes'] == report['retention_probes']
    counts = Counter((row['concept'], row['observed_seq']) for row in report['retention_probes'])
    assert any(count > 1 and seq == 314 for (_, seq), count in counts.items())
    for name, entries in tracking['series'].items():
        if name.startswith(('retention_accuracy/', 'retention_drop/')):
            assert len(entries) == len({row['step'] for row in entries})
    assert metric_points(tracking)
