"""Bounded policy/detector and temporal promotion contracts, run on homelab only."""

import pickle

import pytest

from continual_bench.adapters.river import RiverDetector
from continual_bench.application.config import Config, TrackingConfig
from continual_bench.domain.contracts import DecisionContext, Event
from continual_bench.domain.engine import Engine
from continual_bench.domain.policies import PolicyStrategy


class MajorityModel:
    def __init__(self):
        self.labels = []

    def predict(self, x):
        return 0.9 if not self.labels or sum(self.labels) * 2 >= len(self.labels) else 0.1

    def learn(self, x, y):
        self.labels.append(y)


class NeverAlarm:
    def update(self, error):
        return False


def event(i, y=True, delay=0):
    return Event(f"id-{i}", i, i, i + delay, {"x": i}, y)


def policy(kind, **params):
    return PolicyStrategy(kind, cooldown=0, min_samples=1, min_window=1, params=params)


def test_performance_requires_released_monitoring_window_and_threshold():
    subject = policy("performance", monitor_window=4, error_threshold=0.5)
    assert subject.decide(DecisionContext(10, 5, False, 1.0, 3)).action == "none"
    assert subject.decide(DecisionContext(10, 5, False, 0.49, 4)).action == "none"
    assert subject.decide(DecisionContext(10, 5, False, 0.5, 4)).action == "reset_retrain"


def test_cost_heuristic_tradeoff_and_strict_break_even():
    subject = policy("cost_heuristic", target_error=0.1, horizon=10,
                     error_cost=2, fixed_train_cost=1, sample_train_cost=1)
    # Expected avoided scenario cost = (0.6 - 0.1) * 10 * 2 = 10.
    assert subject.decide(DecisionContext(100, 9, False, 0.6, 100)).action == "none"
    assert subject.decide(DecisionContext(100, 8, False, 0.6, 100)).action == "reset_retrain"
    assert subject.decide(DecisionContext(100, 8, False, 0.05, 100)).action == "none"


@pytest.mark.parametrize("params", [
    {"error_threshold": 2}, {"horizon": 0}, {"monitor_window": 1.5},
    {"fixed_train_cost": -1}, {"error_cost": float("nan")}, {"unknown": 1},
])
def test_policy_rejects_invalid_configuration(params):
    with pytest.raises(ValueError):
        policy("cost_heuristic", **params)


def test_threshold_policy_obeys_label_availability_before_deciding():
    subject = Engine(MajorityModel, NeverAlarm,
                     policy("performance", monitor_window=2, error_threshold=0.5),
                     warmup=1, window=8, metric_window=2)
    for i in range(4):
        subject.observe_batch([event(i, y=False, delay=3)])
    assert subject.labeled == 1
    assert not subject.updates
    subject.observe_batch([event(4, y=False, delay=3)])
    assert subject.labeled == 2
    assert len(subject.recent_errors) == 1
    assert not subject.updates


def promotion_engine(budget=20):
    return Engine(MajorityModel, NeverAlarm,
                  PolicyStrategy("periodic", every=3, cooldown=0, min_samples=1, min_window=1),
                  warmup=1, window=8, metric_window=2, training_budget=budget,
                  promotion={"validation_samples": 1, "min_improvement": 0})


def test_promotion_candidate_uses_disjoint_past_training_and_holdout():
    subject = promotion_engine()
    for i, label in enumerate([True, False, False, False]):
        subject.observe_batch([event(i, label)])
    assert len(subject.candidates) == len(subject.updates) == 1
    assert subject.model.labels == [True, False, False]  # Never learned holdout event 3.
    candidate = subject.candidates[0]
    assert candidate["validation_seq_range"] == [3, 4]
    assert candidate["train_seq_range"] == [0, 3]
    assert candidate["candidate_accuracy"] == 1
    assert candidate["incumbent_accuracy"] == 0
    assert candidate["promoted"]
    assert subject.training_used == 3


def test_rejected_candidate_consumes_training_budget_and_cooldown():
    subject = promotion_engine(budget=3)
    incumbent = subject.model
    for i, label in enumerate([True, True, True, False]):
        subject.observe_batch([event(i, label)])
    assert subject.model is incumbent
    assert not subject.updates
    assert subject.candidates[0]["promoted"] is False
    assert subject.policy.last_update == 4
    assert subject.report()["promotion"]["rejected"] == 1
    assert subject.report()["promotion"]["training_seconds"] >= 0
    for i in range(4, 8):
        subject.observe_batch([event(i)])
    assert subject.training_used == 3
    assert subject.budget_blocked > 0


def test_promotion_resume_and_duplicate_replay_are_idempotent():
    subject = promotion_engine()
    for i in range(4):
        subject.observe_batch([event(i, y=i == 0)])
    restored = pickle.loads(pickle.dumps(subject))
    digest = restored.audit
    restored.observe_batch([event(i, y=i == 0) for i in range(4)])
    assert restored.audit == digest
    assert restored.candidates == subject.candidates
    for i in range(4, 9):
        restored.observe_batch([event(i, y=False)])
        subject.observe_batch([event(i, y=False)])
    assert restored.audit == subject.audit
    assert restored.model.labels == subject.model.labels
    assert restored.training_used == subject.training_used
    assert [c["promoted"] for c in restored.candidates] == [c["promoted"] for c in subject.candidates]


def test_failed_candidate_does_not_replace_incumbent():
    class FailingModel(MajorityModel):
        def learn(self, x, y):
            if x["x"] == 1:
                raise RuntimeError("training failed")
            super().learn(x, y)

    subject = Engine(FailingModel, NeverAlarm,
                     PolicyStrategy("periodic", every=1, cooldown=0, min_samples=1, min_window=1),
                     warmup=1, window=4, metric_window=2)
    subject.observe_batch([event(0)])
    incumbent = subject.model
    with pytest.raises(RuntimeError, match="training failed"):
        subject.observe_batch([event(1)])
    assert subject.model is incumbent
    assert subject.training_used == 0
    assert not subject.candidates


@pytest.mark.parametrize("name", ["adwin", "ddm", "eddm", "hddm_a", "hddm_w", "page_hinkley", "kswin"])
def test_detector_binary_error_contract_and_pickle(name):
    subject = RiverDetector(name)
    for i in range(80):
        assert isinstance(subject.update(i % 2), bool)
    restored = pickle.loads(pickle.dumps(subject))
    assert [subject.update(i % 3 == 0) for i in range(80)] == [
        restored.update(i % 3 == 0) for i in range(80)
    ]
    assert subject.signal == "released_label_binary_prediction_error"
    with pytest.raises(ValueError, match="binary"):
        subject.update(0.2)


def test_page_hinkley_detects_controlled_error_increase():
    subject = RiverDetector("page_hinkley", min_instances=10, delta=0.005, threshold=5, mode="up")
    assert not any(subject.update(0) for _ in range(100))
    assert any(subject.update(1) for _ in range(100))


def test_kswin_has_reproducible_default_and_valid_sampling_window():
    assert RiverDetector("kswin").inner.seed == 42
    with pytest.raises(ValueError, match="stat_size"):
        RiverDetector("kswin", window_size=20, stat_size=15)


def test_new_options_do_not_change_historical_config_serialization():
    config = Config()
    assert "promotion" not in config.model_dump()
    assert "params" not in config.policy.model_dump()
    assert "week09" not in TrackingConfig().model_dump()
    with pytest.raises(ValueError, match="promotion gate"):
        Config.model_validate({"policy": {"name": "online", "action": "learn_one"},
                               "promotion": {"validation_samples": 8}})
    with pytest.raises(ValueError, match="leave min_window"):
        Config.model_validate({"replay_window": 64, "promotion": {"validation_samples": 32}})


def test_promotion_rejects_nonfinite_validation_predictions():
    class InvalidModel:
        def predict(self, x):
            return float("nan")

    with pytest.raises(ValueError, match="promotion validation"):
        Engine.holdout_accuracy(InvalidModel(), [({"x": 1}, True)])


def test_promotion_insufficient_prefix_still_records_every_released_label():
    class Observer:
        def __init__(self):
            self.labels = []
            self.warmup_train_seconds = 0
            self.warmup_train_samples = 0
            self.monitor_seconds = 0

        def before_prediction(self, *args):
            return {}

        def after_label(self, event, *args):
            self.labels.append(event.seq)

        def sample_memory(self):
            pass

    observer = Observer()
    subject = Engine(MajorityModel, NeverAlarm,
                     PolicyStrategy("periodic", every=1, cooldown=0, min_samples=1, min_window=1),
                     warmup=1, window=8, metric_window=1, min_retrain_window=3,
                     observer=observer, promotion={"validation_samples": 2, "min_improvement": 0})
    for i in range(4):
        subject.observe_batch([event(i)])
    assert observer.labels == [0, 1, 2, 3]
    assert len(subject.windows) == 3
    assert not subject.candidates


def test_default_monitor_can_reach_minimum_when_metric_window_is_smaller():
    subject = Engine(MajorityModel, NeverAlarm,
                     PolicyStrategy("performance", cooldown=0, min_samples=1, min_window=3,
                                    params={"error_threshold": 0}),
                     warmup=1, window=8, metric_window=1)
    for i in range(4):
        subject.observe_batch([event(i)])
    assert subject.recent_errors.maxlen == 3
    assert len(subject.updates) == 1


def test_cost_heuristic_rejects_derived_nonfinite_scenario_costs():
    subject = policy("cost_heuristic", error_cost=1e308, horizon=100)
    with pytest.raises(ValueError, match="overflow"):
        subject.decide(DecisionContext(10, 5, False, 1.0, 10))
