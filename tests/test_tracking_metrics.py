import json
import math
import pickle
from itertools import permutations

import pytest

from continual_bench.domain.metrics import Metrics


def accumulate(examples):
    metrics = Metrics()
    for y, p in examples:
        metrics.add(y, p)
    return metrics


def test_known_confusion_matrix_and_calibration():
    examples = [(True, 0.8)] * 3 + [(False, 0.7)] + [(False, 0.2)] * 4 + [(True, 0.4)] * 2
    values = accumulate(examples).values()
    expected = {
        "n": 10, "tp": 3, "fp": 1, "tn": 4, "fn": 2, "errors": 3,
        "accuracy": 0.7, "error_rate": 0.3, "precision": 0.75, "recall": 0.6,
        "specificity": 0.8, "false_positive_rate": 0.2, "false_negative_rate": 0.4,
        "negative_predictive_value": 2 / 3, "balanced_accuracy": 0.7,
        "f1": 2 / 3, "f1_macro": (2 / 3 + 8 / 11) / 2,
        "f1_weighted": (5 * 2 / 3 + 5 * 8 / 11) / 10,
        "mcc": 10 / math.sqrt(600), "cohen_kappa": 0.4,
        "positive_rate": 0.5, "predicted_positive_rate": 0.4,
        "roc_auc": 23 / 25, "average_precision": 14 / 15, "ece": 0.33, "mce": 0.7,
    }
    for key, value in expected.items():
        assert values[key] == pytest.approx(value), key


def test_original_five_metrics_are_bit_for_bit_unchanged():
    examples = [(True, 0.8), (False, 0.2), (True, 0.1), (False, 0.5), (True, 0), (False, 1)]
    n = correct = tp = fp = fn = 0
    brier = loss = 0.0
    for y, p in examples:
        pred = p >= 0.5
        n += 1
        correct += int(pred == y)
        tp += int(pred and y)
        fp += int(pred and not y)
        fn += int(not pred and y)
        brier += (int(y) - p) ** 2
        q = min(max(p, 1e-15), 1 - 1e-15)
        loss -= math.log(q if y else 1 - q)
    expected = {
        "n": n, "accuracy": correct / n, "f1": 2 * tp / (2 * tp + fp + fn),
        "brier": brier / n, "log_loss": loss / n,
    }
    values = accumulate(examples).values()
    assert {key: values[key] for key in expected} == expected


@pytest.mark.parametrize("examples,auc,ap", [
    ([(True, 0.9), (True, 0.8), (False, 0.2), (False, 0.1)], 1, 1),
    ([(False, 0.9), (False, 0.8), (True, 0.2), (True, 0.1)], 0, 5 / 12),
    ([(True, 0.5), (False, 0.5), (True, 0.5), (False, 0.5)], 0.5, 0.5),
    ([(True, 0.9), (False, 0.8), (True, 0.7), (False, 0.6)], 0.75, 5 / 6),
    ([(True, 0.9), (False, 0.9), (True, 0.1), (False, 0.1)], 0.5, 0.5),
    ([(True, 0.8), (True, 0.8), (False, 0.8), (True, 0.2), (False, 0.2)], 7 / 12, 29 / 45),
])
def test_ranking_known_perfect_reversed_ties_and_ap_not_roc(examples, auc, ap):
    # All orderings must agree, including different orderings inside tie groups.
    for ordering in permutations(examples):
        values = accumulate(ordering).values()
        assert values["roc_auc"] == pytest.approx(auc)
        assert values["average_precision"] == pytest.approx(ap)


def test_roc_matches_pairwise_wins_and_half_ties():
    examples = [(True, 0), (True, 0.2), (False, 0.2), (False, 0.6), (True, 1)]
    positive = [p for y, p in examples if y]
    negative = [p for y, p in examples if not y]
    wins = sum(1 if p > q else 0.5 if p == q else 0 for p in positive for q in negative)
    assert accumulate(examples).values()["roc_auc"] == wins / (len(positive) * len(negative))


def test_calibration_boundary_zero_one_and_internal_bin_edges():
    # 0.1 must be in a different bin from 0.09. Scores 0.9 and 1 share the last.
    # Gaps: bin0=.045; bin1=.35; bin9=.45. Equal bin populations give ECE=.28166...
    examples = [(False, 0), (False, 0.09), (True, 0.1), (False, 0.2 - 0.0000000001),
                (False, 0.9), (True, 1)]
    values = accumulate(examples).values()
    middle_gap = abs((0.1 + 0.2 - 0.0000000001) / 2 - 0.5)
    assert values["ece"] == pytest.approx((0.045 + middle_gap + 0.45) / 3)
    assert values["mce"] == pytest.approx(0.45)


def test_calibration_uses_positive_probability_not_confidence():
    # Mean p and frequency are both .5: perfect positive-probability calibration
    # within the sole occupied bin, even though individual predictions can err.
    values = accumulate([(True, 0.5), (False, 0.5)]).values()
    assert values["ece"] == values["mce"] == 0


def test_empty_values_are_explicit_and_json_safe():
    values = Metrics().values()
    undefined = {"roc_auc", "average_precision", "balanced_accuracy", "mcc", "cohen_kappa"}
    assert {key for key, value in values.items() if value is None} == undefined
    assert all(value == 0 for key, value in values.items() if key not in undefined)
    json.dumps(values, allow_nan=False)


@pytest.mark.parametrize("label,p,ap", [(True, 1, 1), (False, 0, None)])
def test_single_class_perfect_prediction_has_defined_and_undefined_metrics(label, p, ap):
    values = accumulate([(label, p)] * 3).values()
    assert values["roc_auc"] is values["balanced_accuracy"] is values["mcc"] is None
    assert values["cohen_kappa"] is None
    assert values["average_precision"] == ap
    assert values["f1_macro"] == 0.5  # Fixed label universe {False, True}.
    assert values["f1_weighted"] == values["accuracy"] == 1
    assert values["ece"] == values["mce"] == 0
    assert values["precision"] == values["recall"] == float(label)
    json.dumps(values, allow_nan=False)


def test_all_negative_predictions_do_not_hide_errors_or_imbalance():
    values = accumulate([(True, 0.2)] + [(False, 0.2)] * 9).values()
    assert values["accuracy"] == 0.9
    assert values["f1"] == values["precision"] == values["recall"] == 0
    assert values["specificity"] == 1
    assert values["balanced_accuracy"] == 0.5
    assert values["mcc"] is None
    assert values["cohen_kappa"] == 0
    assert values["average_precision"] == 0.1
    assert values["roc_auc"] == 0.5


def test_macro_and_weighted_f1_have_different_weights():
    examples = [(True, 0.8), (False, 0.8)] + [(False, 0.2)] * 6 + [(True, 0.2)] * 2
    values = accumulate(examples).values()
    assert values["f1"] == pytest.approx(0.4)
    assert values["f1_macro"] == pytest.approx(0.6)
    assert values["f1_weighted"] == pytest.approx(0.68)


def test_reversed_classification_has_negative_mcc_and_kappa():
    values = accumulate([(True, 0), (False, 1)]).values()
    assert values["mcc"] == values["cohen_kappa"] == -1
    assert values["errors"] == 2


def test_pickle_resume_matches_uninterrupted_and_values_are_read_only():
    examples = [(bool(i % 3), (i % 11) / 10) for i in range(100)]
    uninterrupted = accumulate(examples)
    resumed = pickle.loads(pickle.dumps(accumulate(examples[:37])))
    for y, p in examples[37:]:
        resumed.add(y, p)
    before = pickle.dumps(resumed)
    assert resumed.values() == uninterrupted.values()
    assert pickle.dumps(resumed) == before
    assert len(resumed._scores) == resumed.n == len(examples)


@pytest.mark.parametrize("p", [math.nan, math.inf, -math.inf, -0.01, 1.01])
def test_invalid_probability_does_not_mutate_state(p):
    metrics = accumulate([(True, 0.8)])
    before = pickle.dumps(metrics)
    with pytest.raises(ValueError):
        metrics.add(True, p)
    assert pickle.dumps(metrics) == before
