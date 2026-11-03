import copy
import json
import pickle
from dataclasses import replace

import pytest

from continual_bench.application.week09_reporting import supplemental_report, validated_detection
from continual_bench.domain.contracts import Event
from continual_bench.domain.week09_metrics import (
    NearFutureEvaluation,
    task_matrix_metrics,
    validate_temporal_holdout,
)


class MutableModel:
    def __init__(self, probability=0.9):
        self.probability = probability
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        x["mutation"] = True
        return self.probability


def event(seq, y=True, delay=2):
    return Event(f"e-{seq}", seq, 100 + seq, 100 + seq + delay, {"index": seq}, y)


def release(collector, item, now=None):
    collector.after_label(item, 0.5, {}, item.available_at if now is None else now, None, True)


def detector_report(points=None, alarms=(11, 12, 29), observed=40, warmup=5):
    return {
        "dataset": {"changepoints": points}, "observed": observed,
        "metrics": {"n": observed - warmup},
        "config": {"detector": {"name": "adwin"}, "dataset": {"epoch": 100}},
        "alarms": [{"seq": seq, "released_at": 100 + seq + 2} for seq in alarms],
    }


def test_gem_matrix_equations_and_postlearning_forgetting_are_distinct():
    matrix = [[0.8, 0.9, 0.2], [0.6, 0.7, 0.4], [0.5, 0.6, 0.9]]
    result = task_matrix_metrics(matrix, initial=[0.1, 0.2, 0.3])
    values = result["metrics"]
    assert values["gem_average_accuracy"] == pytest.approx(2 / 3)
    assert values["gem_backward_transfer"] == pytest.approx(-0.2)
    assert values["gem_forward_transfer"] == pytest.approx(0.4)
    # Task 1's diagnostic prelearning .9 is excluded from postlearning max.
    assert values["postlearning_forgetting"] == pytest.approx(0.2)
    assert values["past_task_retention_accuracy"] == pytest.approx(0.55)
    assert matrix[0][1] == 0.9
    json.dumps(result, allow_nan=False)


def test_task_matrix_missing_cells_propagate_and_positive_bwt_is_retained():
    values = task_matrix_metrics([[0.4, None], [0.6, 0.8]])["metrics"]
    assert values["postlearning_forgetting"] == pytest.approx(-0.2)
    assert values["gem_backward_transfer"] == pytest.approx(0.2)
    assert values["gem_forward_transfer"] is None
    missing = task_matrix_metrics([[None, None], [0.6, None]], initial=[0.1, 0.1])
    assert missing["metrics"]["gem_average_accuracy"] is None
    assert missing["metrics"]["gem_backward_transfer"] is None
    assert missing["metrics"]["gem_forward_transfer"] is None
    assert task_matrix_metrics([])["reason"] == "no_completed_tasks"
    assert task_matrix_metrics([[0.4]])["metrics"]["gem_backward_transfer"] is None


@pytest.mark.parametrize("matrix,initial", [
    ([[0.5], [0.3]], None), ([[0.5, 0.3], [0.4]], None),
    ([[float("nan")]], None), ([[1.01]], None), ([[True]], None),
    ([[0.3]], []), ([[0.5]], [float("inf")]),
])
def test_invalid_task_matrices_are_rejected(matrix, initial):
    with pytest.raises(ValueError):
        task_matrix_metrics(matrix, initial)


def test_detector_matching_reuses_one_to_one_censored_implementation():
    report = detector_report([10, 25, 38], alarms=[11, 12, 29, 39])
    result = validated_detection(report, 5)
    metrics = result["metrics"]
    assert metrics["drift_tp"] == 2
    assert metrics["drift_fp"] == 1  # repeated alarm at 12 cannot redetect change 10.
    assert metrics["drift_fn"] == 0
    assert metrics["drift_delay_mean_samples"] == 2.5
    assert metrics["drift_censored_changes"] == 1
    assert metrics["drift_f1"] == pytest.approx(0.8)
    assert result["details"]["censored_alarm_count"] == 1
    assert validated_detection(detector_report(), 5)["metrics"]["drift_f1"] is None


@pytest.mark.parametrize("points,alarms,tolerance", [
    ([20, 10], [21], 5), ([10, 10], [11], 5), ([True], [11], 5),
    ([10], [11, 11], 5), ([40], [], 5), ([10], [40], 5), ([10], [11], 0),
])
def test_invalid_annotation_contract_rejected(points, alarms, tolerance):
    with pytest.raises(ValueError):
        validated_detection(detector_report(points, alarms), tolerance)


def test_explicit_temporal_split_checks_availability_and_identity_leakage():
    train, test = [event(0), event(1)], [event(4), event(5)]
    assert validate_temporal_holdout(train, test, cutoff=104)["train_samples"] == 2
    with pytest.raises(ValueError, match="available before cutoff"):
        validate_temporal_holdout([event(0, delay=10)], test, cutoff=104)
    with pytest.raises(ValueError, match="duplicate event ID"):
        validate_temporal_holdout(train, [replace(event(4), event_id="e-0")], cutoff=104)
    with pytest.raises(ValueError, match="chronological"):
        validate_temporal_holdout(train[::-1], test, cutoff=104)


def test_frozen_future_window_ignores_intervening_learning_and_waits_for_label_release():
    model = MutableModel()
    collector = NearFutureEvaluation(warmup=2, horizon=2, gap=1, max_windows=1)
    before = pickle.dumps(model)
    collector.before_prediction(event(2), model, 2)
    model.probability = 0.1
    for seq in (3, 4):
        item = event(seq)
        original = copy.deepcopy(item.x)
        collector.before_prediction(item, model, 2)
        assert item.x == original
    partial = collector.report()
    assert partial["near_future_accuracy"] is None
    assert partial["windows"][0]["predictions"][0]["probability"] == 0.9
    assert model.calls == 0
    with pytest.raises(ValueError, match="not available"):
        release(collector, event(3), 104)
    release(collector, event(3))
    assert collector.report()["near_future_accuracy"] is None
    release(collector, event(4))
    result = collector.report()
    assert result["near_future_accuracy"] == 1
    assert result["completed_windows"] == 1
    assert result["windows"][0]["anchor_seq"] == 1
    assert result["windows"][0]["start_seq"] == 3
    assert before != pickle.dumps(model)  # Only deliberate caller mutation occurred.
    result["windows"][0]["predictions"][0]["probability"] = 0
    assert collector.report()["near_future_accuracy"] == 1


def test_near_future_replay_pickle_resume_and_failure_retry():
    collector = NearFutureEvaluation(warmup=1, horizon=2)
    invalid = MutableModel(float("nan"))
    with pytest.raises(ValueError):
        collector.before_prediction(event(1), invalid, 1)
    assert not collector.windows  # Failed calls cannot commit partial schedules.
    model = MutableModel()
    collector.before_prediction(event(1), model, 1)
    collector.before_prediction(event(1), model, 1)
    release(collector, event(1))
    release(collector, event(1))
    restored = pickle.loads(pickle.dumps(collector))
    for subject in (collector, restored):
        subject.before_prediction(event(2, False), model, 1)
        release(subject, event(2, False))
    assert collector.report() == restored.report()
    assert collector.report()["near_future_accuracy"] == 0.5
    with pytest.raises(ValueError, match="conflicting replayed"):
        collector.before_prediction(replace(event(1), x={"changed": True}), model, 1)
    with pytest.raises(ValueError, match="conflicting replayed"):
        release(collector, event(1, False))


def test_future_label_leakage_order_and_bounded_windows():
    collector = NearFutureEvaluation(warmup=1, horizon=1, max_windows=2)
    model = MutableModel()
    with pytest.raises(ValueError, match="future"):
        collector.before_prediction(event(1), model, 2)
    with pytest.raises(ValueError, match="contiguous"):
        collector.before_prediction(event(2), model, 1)
    for seq in range(1, 8):
        collector.before_prediction(event(seq), model, 1)
        release(collector, event(seq))
    assert len(collector.windows) == 2
    assert sum(len(row["predictions"]) for row in collector.windows) == 2
    for kwargs in ({"horizon": 1025}, {"max_windows": 129}, {"horizon": True}, {"gap": -1}):
        with pytest.raises(ValueError):
            NearFutureEvaluation(warmup=1, **kwargs)


def test_supplement_is_versioned_atomic_and_does_not_fabricate_future_accuracy(tmp_path):
    report = detector_report()
    report["tracking"] = {"metrics": {"train_seconds": 1.5, "cost_total": 5}}
    before = copy.deepcopy(report)
    result = supplemental_report(report, tmp_path)
    assert result["version"] == "semana09-metrics-v1"
    assert result["near_future"]["near_future_accuracy"] is None
    assert result["resources"]["metrics"]["train_seconds"] == 1.5
    assert json.loads((tmp_path / "week09-metrics.json").read_text()) == result
    assert report == before


def test_supplement_counts_rejected_candidates_training_separately_from_promotions():
    report = detector_report()
    report["candidates"] = [
        {"action": "reset_retrain", "train_samples": 10, "seconds": 2, "promoted": False},
        {"action": "reset_retrain", "train_samples": 10, "seconds": 3, "promoted": True},
        {"action": "learn_one", "train_samples": 1, "seconds": 0.1, "promoted": True},
    ]
    resources = supplemental_report(report)["resources"]["candidates"]
    assert resources["training_actions"] == 3
    assert resources["candidate_retrains"] == 2
    assert resources["promoted_actions"] == 2
    assert resources["rejected_actions"] == 1
    assert resources["train_samples"] == 21
    assert resources["training_seconds"] == pytest.approx(5.1)
    assert resources["rejected_training_seconds"] == 2
    assert resources["retrain_duration_seconds"] == [2, 3]
