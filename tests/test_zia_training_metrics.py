import numpy as np
import pytest

from continual_bench.evaluation.zia_metrics import ClassificationMetrics
from continual_bench.models.zia import compute_classification_metrics, make_zia_model


def test_training_average_precision_groups_tied_scores_and_matches_stream():
    probabilities = np.full((4, 2), 0.5)
    for labels in (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])):
        training = compute_classification_metrics(labels, probabilities, 2)
        assert training["pr_auc"] == training["macro_pr_auc"] == 0.5

    labels = np.arange(4)
    probabilities = np.eye(4)
    training = compute_classification_metrics(labels, probabilities, 4)
    stream = ClassificationMetrics(4)
    stream.add(labels, probabilities)
    assert "pr_auc" not in training
    assert training["macro_pr_auc"] == stream.metrics()[0]["macro_pr_auc"] == 1.0


def test_training_uses_the_campaign_decision_threshold():
    labels = np.array([1, 1, 1, 0])
    probabilities = np.tile([0.9, 0.1], (4, 1))
    threshold = 0.025
    training = compute_classification_metrics(labels, probabilities, 2, threshold)
    stream = ClassificationMetrics(2, threshold)
    stream.add(labels, probabilities)
    assert training["accuracy"] == stream.metrics()[0]["accuracy"] == 0.75
    assert compute_classification_metrics(labels, probabilities, 2)["accuracy"] == 0.25


class MinibatchTracker:
    def __init__(self):
        self.steps = []
        self.summary = None

    def log_minibatch_metrics(self, metrics, *, step):
        self.steps.append((step, dict(metrics)))

    def finalize(self, summary):
        self.summary = dict(summary)


@pytest.mark.parametrize(("learner", "size"), [("xgboost", 2), ("multinomial_logistic", 4)])
def test_online_reports_each_minibatch_and_its_actual_example_count(learner, size):
    if learner == "xgboost":
        pytest.importorskip("xgboost")
    model = make_zia_model(learner, size, decision_threshold=0.025)
    x = np.arange(24, dtype=np.float32).reshape(-1, 1)
    y = np.arange(len(x)) % size
    model.fit(x, y, config={"max_epochs": 2}, timestamps=np.arange(len(y)))
    initial = compute_classification_metrics(y[:4], model.predict(x[:4]), size, 0.025)
    model.online_params = {**model.online_params, "minibatch_rows": 4}
    tracker = MinibatchTracker()
    model.fit_online(x[:11], y[:11], tracker=tracker)

    assert [step for step, _ in tracker.steps] == [1, 2, 3]
    assert [metrics["train_samples"] for _, metrics in tracker.steps] == [4, 4, 3]
    assert tracker.steps[0][1]["train:log_loss"] == initial["log_loss"]
    assert tracker.summary["epochs_run"] == 0
    assert tracker.summary["iterations"] == 3
    assert tracker.summary["training_presentations"] == 11
    assert model.training_summary["history"] == [metrics for _, metrics in tracker.steps]


@pytest.mark.parametrize(("learner", "size"), [("xgboost", 2), ("multinomial_logistic", 4)])
def test_both_zia_learners_fit_singleton_temporal_windows(learner, size):
    if learner == "xgboost":
        pytest.importorskip("xgboost")
    model = make_zia_model(learner, size)
    model.fit(
        np.array([[0.0]], dtype=np.float32),
        np.array([1]),
        config={"max_epochs": 2},
        timestamps=np.array([20]),
    )
    assert model.training_summary["validation_rows"] == 0
    assert model.training_summary["training_presentations"] == 2
    assert np.isfinite(model.predict(np.array([[0.0]], dtype=np.float32))).all()
