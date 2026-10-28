import ast
import math
import pickle
from pathlib import Path

import pytest

from continual_bench.domain.contracts import DecisionContext, Event
from continual_bench.domain.engine import Engine
from continual_bench.domain.metrics import Metrics
from continual_bench.domain.policies import PolicyStrategy


class SpyModel:
    def __init__(self):
        self.learned = []
        self.predictions = []

    def predict(self, x):
        self.predictions.append(len(self.learned))
        return 0.75 if self.learned else 0.5

    def learn(self, x, y):
        self.learned.append((x, y))


class SpyDetector:
    def __init__(self):
        self.errors = []

    def update(self, error):
        self.errors.append(error)
        return bool(error)


def event(seq, delay=0, y=True):
    return Event(f"event-{seq}", seq, seq, seq + delay, {"x": seq}, y)


def engine(**kwargs):
    return Engine(
        SpyModel, SpyDetector, PolicyStrategy("none"), warmup=1, window=5, metric_window=2, **kwargs
    )


def test_delayed_labels_are_invisible_to_training_and_detector():
    subject = engine()
    subject.observe_batch([event(0, delay=3)])
    subject.observe_batch([event(1, delay=3, y=False)])
    subject.observe_batch([event(2, delay=3)])
    assert subject.model.predictions == [0, 0, 0]
    assert not subject.model.learned
    assert not subject.detector.errors
    subject.observe_batch([event(3, delay=3)])
    assert len(subject.model.learned) == 1  # Only warmup label now available.
    assert not subject.detector.errors
    subject.observe_batch([event(4, delay=3)])
    assert subject.detector.errors == [1]


def test_whole_incremental_batch_is_predicted_before_any_learning():
    subject = engine()
    subject.observe_batch([event(0), event(1), event(2)])
    assert subject.model.predictions == [0, 0, 0]
    assert len(subject.model.learned) == 1
    assert subject.total.n == 2


def test_duplicate_replay_is_idempotent_and_gap_is_rejected():
    subject = engine()
    subject.observe_batch([event(0), event(1)])
    digest = subject.audit
    subject.observe_batch([event(0), event(1)])
    assert subject.audit == digest
    assert subject.labeled == 2
    with pytest.raises(ValueError, match="out-of-order"):
        subject.observe_batch([event(3)])


@pytest.mark.parametrize("kind", ["none", "periodic", "drift"])
def test_policy_contract_and_cooldown(kind):
    subject = PolicyStrategy(kind, every=5, cooldown=3, min_samples=5, min_window=2)
    assert subject.decide(DecisionContext(4, 2, True)).action == "none"
    assert subject.decide(DecisionContext(5, 1, True)).action == "none"
    decision = subject.decide(DecisionContext(5, 2, True))
    assert decision.action == ("none" if kind == "none" else "reset_retrain")
    subject.updated(5)
    assert subject.decide(DecisionContext(6, 2, True)).action == "none"
    if kind == "drift":
        assert subject.decide(DecisionContext(9, 2, False)).action == "none"


@pytest.mark.parametrize("action,expected", [("learn_one", 2), ("reset_retrain", 3)])
def test_reset_window_retraining_and_learn_one_are_different(action, expected):
    subject = Engine(
        SpyModel,
        SpyDetector,
        PolicyStrategy("periodic", every=2, cooldown=0, min_samples=1, min_window=1, action=action),
        warmup=1,
        window=5,
        metric_window=2,
    )
    for i in range(3):
        subject.observe_batch([event(i)])
    assert len(subject.model.learned) == expected
    assert subject.updates[0]["action"] == action
    restored = pickle.loads(pickle.dumps(subject))
    assert restored.model.learned == subject.model.learned


def test_metrics_and_probabilities():
    metrics = Metrics()
    for y, p in [(True, 0.8), (False, 0.2), (True, 0.1)]:
        metrics.add(y, p)
    assert metrics.values()["accuracy"] == pytest.approx(2 / 3)
    assert metrics.values()["f1"] == pytest.approx(2 / 3)
    assert metrics.values()["brier"] == pytest.approx((0.04 + 0.04 + 0.81) / 3)
    for p in [math.nan, math.inf, -1, 2]:
        with pytest.raises(ValueError):
            metrics.add(True, p)


@pytest.mark.parametrize("size", [1, 7])
def test_online_updates_every_released_label_and_drains_delays(size):
    subject = Engine(SpyModel, SpyDetector,
                     PolicyStrategy("online", cooldown=100, min_samples=100, min_window=50),
                     warmup=3, window=5, metric_window=4, training_budget=17)
    for start in range(0, 20, size):
        subject.observe_batch([event(i, delay=3) for i in range(start, min(start + size, 20))])
        assert len(subject.model.learned) == max(0, subject.last_timestamp - 2)
    subject.finish()
    assert len(subject.model.learned) == 20
    assert [u["seq"] for u in subject.updates] == list(range(3, 20))
    assert subject.training_used == 17


def test_budget_caps_replayed_examples_and_resume_accounting():
    subject = Engine(SpyModel, SpyDetector,
                     PolicyStrategy("periodic", every=2, cooldown=0, min_samples=1, min_window=1),
                     warmup=1, window=5, metric_window=2, training_budget=7, min_retrain_window=1)
    for i in range(7):
        subject.observe_batch([event(i)])
    subject = pickle.loads(pickle.dumps(subject))
    for i in range(7, 12):
        subject.observe_batch([event(i)])
    subject.finish()
    assert sum(u["train_samples"] for u in subject.updates) == subject.training_used == 7
    assert [u["train_samples"] for u in subject.updates] == [3, 4]
    assert subject.budget_blocked > 0
    assert subject.total.n == 11


def test_domain_has_no_service_or_framework_imports():
    allowed = {"dataclasses", "typing", "math", "copy", "hashlib", "json", "time", "collections"}
    domain = Path(__file__).parents[1] / "src/continual_bench/domain"
    for file in domain.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split(".")[0] in allowed
