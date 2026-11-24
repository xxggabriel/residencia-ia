import random

import pytest
from sklearn.metrics import roc_auc_score

from continual_bench.evaluation.tracking import regime_metrics


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_regime_auc_matches_ranked_probabilities_with_ties(seed):
    rng = random.Random(seed)
    rows = [
        {"seq": index, "label": bool(index % 2), "probability": rng.choice([0.1, 0.4, 0.7, 0.9])}
        for index in range(40)
    ]
    result = regime_metrics(rows, [20], 40)
    assert result["available"]
    for group, selected in zip(result["regimes"], [rows[:20], rows[20:]], strict=True):
        expected = roc_auc_score([r["label"] for r in selected], [r["probability"] for r in selected])
        assert group["roc_auc"] == pytest.approx(expected, abs=1e-15)
        assert "roc_auc" not in group["unavailable_reasons"]


@pytest.mark.parametrize("label", [False, True])
def test_regime_auc_stays_unavailable_for_one_class(label):
    result = regime_metrics([{"seq": 0, "label": label, "probability": 0.7}], [], 1)
    group = result["regimes"][0]
    assert group["roc_auc"] is None
    assert group["unavailable_reasons"]["roc_auc"] == "single_class_or_undefined_in_regime"
