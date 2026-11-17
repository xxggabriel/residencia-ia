import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_curve

from continual_bench.evaluation.zia_metrics import ClassificationMetrics
from continual_bench.evaluation.zia_observations import (
    MAX_CHECKPOINTS,
    MAX_PERIOD_ROWS,
    MAX_PERIODS,
    PERIOD_MS,
    TemporalHoldoutProbes,
)
from continual_bench.models.classification_metrics import SCORE_BINS
from continual_bench.models.zia import make_zia_model


def test_binary_macro_f1_uses_fixed_catalog_classes_with_explicit_legacy_variant():
    labels = np.array([0, 0, 0])
    probabilities = np.tile([0.99, 0.01], (3, 1))
    stream = ClassificationMetrics(2, threshold=0.025)
    stream.add(labels, probabilities)
    legacy, _ = stream.metrics()
    quality = stream.quality(["legitimate", "fraud"], curves=True)
    expected = f1_score(labels, [0, 0, 0], labels=[0, 1], average="macro", zero_division=0)
    assert quality["macro_f1"] == expected == 0.5
    assert legacy["f1_macro"] == legacy["macro_f1"] == 1.0
    assert quality["positive_class"] == "fraud"
    assert quality["decision_threshold"] == 0.025
    assert quality["class_support"] == [3, 0]
    assert quality["class_recall"] == [1, None]
    assert quality["positive_precision"] is quality["positive_recall"] is None
    assert quality["average_precision"] is quality["pr_curve"] is quality["roc_curve"] is None


def test_histogram_ranking_matches_exact_reference_for_bin_ties_and_low_fpr():
    labels = np.array([1, 0, 1, 0, 0, 1, 0])
    positive = np.array([1, 0.75, 0.75, 0.5, 0, 0.5, 0.5])
    probabilities = np.column_stack([1 - positive, positive])
    stream = ClassificationMetrics(2, threshold=0.025)
    for start, stop in ((0, 3), (3, 6), (6, 7)):
        partial = ClassificationMetrics(2, threshold=0.025)
        partial.add(labels[start:stop], probabilities[start:stop])
        stream.merge(partial)
    quality = stream.quality(["legitimate", "fraud"], curves=True)
    fpr, tpr, _ = roc_curve(labels, positive, drop_intermediate=False)
    assert quality["average_precision"] == pytest.approx(average_precision_score(labels, positive))
    assert quality["recall_at_fpr_0_01"] == max(tpr[fpr <= 0.01]) == pytest.approx(1 / 3)
    assert quality["roc_curve"]["tpr"] == pytest.approx(tpr.tolist())
    assert quality["roc_curve"]["fpr"] == pytest.approx(fpr.tolist())
    assert quality["ranking_method"] == f"uniform_histogram_{SCORE_BINS}_bins"


def test_histogram_ranking_coarsens_distinct_scores_in_the_same_bin():
    positive = np.array([0.25 + 0.7 / SCORE_BINS, 0.25 + 0.3 / SCORE_BINS])
    labels = np.array([1, 0])
    stream = ClassificationMetrics(2)
    stream.add(labels, np.column_stack([1 - positive, positive]))
    assert average_precision_score(labels, positive) == 1
    assert stream.quality([0, 1])["average_precision"] == 0.5


@pytest.mark.parametrize(("learner", "labels"), [
    ("xgboost", ["other", "returned"]), ("multinomial_logistic", ["follow", "friend", "block", "mute"]),
])
def test_online_aggregate_is_computed_from_every_preupdate_prediction(monkeypatch, learner, labels):
    if learner == "xgboost":
        pytest.importorskip("xgboost")
    model = make_zia_model(learner, labels, decision_threshold=0.025)
    x = np.arange(24, dtype=np.float32).reshape(-1, 1)
    y = np.arange(len(x)) % len(labels)
    model.fit(x, y, config={"max_epochs": 2}, timestamps=np.arange(len(y)))
    offline = model.training_summary
    assert len(offline["observation_steps"]) == len(offline["observation_epochs"]) == 2
    assert offline["observation_scopes"]["train"]["classes"] == labels
    if learner == "xgboost":
        checkpoint = ClassificationMetrics(len(labels), threshold=0.025)
        trained = offline["train_rows"]
        checkpoint.add(y[:trained], model.predict(x[:trained]))
        expected = checkpoint.quality(labels, curves=True)
        assert offline["observation_scopes"]["train"]["pr_curve"] == expected["pr_curve"]
        assert offline["observation_scopes"]["train"]["roc_curve"] == expected["roc_curve"]
        assert not offline["observation_epochs"][0]["scopes"]["train"]["pr_curve"]
    predict = model.predict
    scores = []

    def capture(features):
        result = predict(features)
        scores.append(result.copy())
        return result

    monkeypatch.setattr(model, "predict", capture)
    model.online_params = {**model.online_params, "minibatch_rows": 4}
    model.fit_online(x[:11], y[:11])
    summary = model.training_summary
    expected = ClassificationMetrics(len(labels), threshold=0.025)
    expected.add(y[:11], np.concatenate(scores))
    quality = summary["observation_scopes"]["training_pre_update"]
    reference = expected.quality(labels)
    for key in ("macro_f1", "accuracy", "evaluated_examples", "average_precision", "confusion_matrix"):
        assert quality.get(key) == reference.get(key)
    if learner == "xgboost":
        assert quality["pr_curve"] and quality["roc_curve"]
        assert all(
            "pr_curve" not in row["scopes"]["training_pre_update"] for row in summary["observation_steps"]
        )
    assert summary["observation_epochs"] == []
    assert summary["observation_effort"]["epochs"] is None
    assert [row["metrics"]["training/batch_size"] for row in summary["observation_steps"]] == [4, 4, 3]
    assert [row["metrics"]["training/presentations_cumulative"] for row in summary["observation_steps"]] == [
        4, 8, 11,
    ]


def test_temporal_diagnostic_compares_identical_reserved_period_prefixes_within_budget(tmp_path):
    n = MAX_PERIOD_ROWS + 3
    timestamps = np.repeat(np.arange(MAX_PERIODS + 2, dtype=np.int64) * PERIOD_MS, n)
    labels = np.tile(np.arange(n) % 2, MAX_PERIODS + 2)
    x = labels[:, None].astype(np.float32)
    probes = TemporalHoldoutProbes()
    probes.add(100, timestamps[:3], x[:3], labels[:3])
    probes.add(103, timestamps[3:], x[3:], labels[3:])
    versions = []
    for index in range(MAX_CHECKPOINTS + 3):
        versions.append({
            "file": str(index), "fit_id": f"fit-{index:06d}", "sha256": str(index), "end_seq": index,
        })
        (tmp_path / str(index)).write_bytes(bytes([index]))

    class Model:
        loaded = []

        def load(self, payload):
            self.loaded.append(payload[0])

        def predict(self, features):
            positive = features[:, 0] if not (self.loaded[-1] % 2) else 1 - features[:, 0]
            return np.column_stack([1 - positive, positive])

    model = Model()
    result = probes.evaluate(model, tmp_path, versions, threshold=0.5)
    assert len(model.loaded) == MAX_CHECKPOINTS
    assert (model.loaded[0], model.loaded[-1]) == (0, len(versions) - 1)
    assert [p["support"] for p in result["period_metadata"]] == [MAX_PERIOD_ROWS] * MAX_PERIODS
    assert result["period_metadata"][0]["start_seq"] == 100
    assert result["period_metadata"][1]["start_seq"] == 100 + n
    for i, row in enumerate(result["matrix"]):
        expected = float(1 - i % 2) if i in model.loaded else None
        assert row == [expected] * MAX_PERIODS
