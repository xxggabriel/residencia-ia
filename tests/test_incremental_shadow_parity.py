"""Idle shadow copies may be omitted only without changing their realized population."""

import copy

import pytest

from continual_bench.config import TrackingConfig
from continual_bench.core.contracts import Decision, Event
from continual_bench.core.engine import Engine
from continual_bench.core.observability import OnlineObservability
from continual_bench.core.policies import OnlineUpdate
from continual_bench.detectors.river import NullDetector
from continual_bench.models.linear import RiverModel
from continual_bench.models.tree import HoeffdingTreeModel


class CountingCopies:
    copies = 0

    def __deepcopy__(self, memo):
        type(self).copies += 1
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return copied


class CountingLogistic(CountingCopies, RiverModel):
    pass


class CountingTree(CountingCopies, HoeffdingTreeModel):
    pass


class LogicalVersions:
    """Identical logical IDs isolate shadow parity from model-state sampling cadence."""

    def __init__(self, artifact_mode="metrics_only"):
        self.artifact_mode = artifact_mode
        self.records = []

    def record(self, model, training):
        self.records.append(copy.deepcopy(training))
        return f"version-{len(self.records)}"


class Reservations:
    def __init__(self, reserved):
        self.reserved_seqs = frozenset(reserved)
        self.matrix = []
        self.labels = []

    def after_label(self, seq, x, y, model):
        self.labels.append((seq, y))


class AlwaysCopyEngine(Engine):
    """Reference implementation retains the pre-optimization copy at every update."""

    def _can_skip_online_shadow(self, now):
        return False


def untimed(value):
    if isinstance(value, dict):
        return {
            key: untimed(item)
            for key, item in value.items()
            if "seconds" not in key and key != "shadow_unavailable_reason"
        }
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


def run_engine(
    engine_type,
    *,
    budget=None,
    delay=3,
    batch_size=10,
    reserved=(),
    factory=CountingLogistic,
    policy_type=OnlineUpdate,
    observer_type=OnlineObservability,
    mode="metrics_only",
    shadows=True,
    tied_release=False,
):
    settings = TrackingConfig(
        adaptation_shadows=shadows,
        continual_references=False,
        task_evaluation={"horizon": 2, "gap": 1, "max_windows": 4},
    ).model_dump(mode="json")
    observer = observer_type(settings, continual=Reservations(reserved), warmup=7)
    versions = LogicalVersions(mode)
    engine = engine_type(
        factory,
        NullDetector,
        # Built-in OnlineUpdate intentionally ignores the guarded-retraining minima.
        policy_type(min_samples=1000, min_window=1000, cooldown=1000),
        warmup=7,
        window=15,
        metric_window=5,
        training_budget=budget,
        observer=observer,
        versions=versions,
    )
    events = [
        Event(
            f"event-{seq}",
            seq,
            1000 + seq,
            1100 if tied_release else 1000 + seq + delay,
            {"a": (seq % 9) / 9, "b": float(seq % 3)},
            seq % 4 in (0, 1),
        )
        for seq in range(43)
    ]
    factory.copies = 0
    for start in range(0, len(events), batch_size):
        engine.observe_batch(events[start : start + batch_size])
    engine.finish()
    return engine, factory.copies


def assert_parity(before, after):
    for field in ("metrics", "windows", "training", "prediction_digest", "observed", "labeled"):
        assert before.report()[field] == after.report()[field]
    assert untimed(before.updates) == untimed(after.updates)
    assert untimed(before.candidates) == untimed(after.candidates)
    assert before.versions.records == after.versions.records
    assert untimed(before.observer.predictions) == untimed(after.observer.predictions)
    assert untimed(before.observer.comparisons) == untimed(after.observer.comparisons)
    assert before.observer.continual.labels == after.observer.continual.labels
    assert untimed(before.observer.near_future.report()) == untimed(after.observer.near_future.report())
    for expected, actual in zip(before.observer.comparisons, after.observer.comparisons, strict=True):
        if actual.get("shadow_unavailable_reason"):
            assert expected["evaluated"] == actual["evaluated"] == 0
            assert expected["old_errors"] == expected["new_errors"] == 0
            assert expected["old_weighted_loss"] == expected["new_weighted_loss"] == 0
    for event in ({"a": 0.1, "b": 0.0}, {"a": 0.8, "b": 2.0}):
        assert before.model.predict(event) == after.model.predict(event)
        if before.observer.shadow is not None:
            assert before.observer.shadow.predict(event) == after.observer.shadow.predict(event)


@pytest.mark.parametrize("budget", [None, 0, 1, 6, 12])
@pytest.mark.parametrize("delay", [0, 3, 55])
def test_incremental_shadows_preserve_ledgers_budget_reservations_and_drain(budget, delay):
    options = {"budget": budget, "delay": delay, "reserved": {2, 4, 6, 8, 9, 15, 28, 40, 42}}
    before, copies_before = run_engine(AlwaysCopyEngine, **options)
    after, copies_after = run_engine(Engine, **options)
    assert_parity(before, after)
    assert copies_after <= copies_before
    if delay < 43 and budget is None:
        assert copies_after < copies_before


@pytest.mark.parametrize("delay,budget", [(0, None), (3, None), (0, 9), (3, 9)])
def test_real_hoeffding_predictions_and_shadow_losses_match_reference(delay, budget):
    options = {"factory": CountingTree, "delay": delay, "budget": budget}
    before, copies_before = run_engine(AlwaysCopyEngine, **options)
    after, copies_after = run_engine(Engine, **options)
    assert_parity(before, after)
    assert copies_after < copies_before


@pytest.mark.parametrize("tied_release", [False, True])
def test_streaming_and_explicit_final_drain_keep_original_copy_behavior(tied_release):
    options = {"batch_size": 1, "tied_release": tied_release}
    before, copies_before = run_engine(AlwaysCopyEngine, **options)
    after, copies_after = run_engine(Engine, **options)
    assert_parity(before, after)
    assert copies_after == copies_before
    assert not any(point.get("shadow_unavailable_reason") for point in after.observer.comparisons)


class IntermittentOnline(OnlineUpdate):
    def decide(self, context):
        return Decision("learn_one", "custom") if context.labeled % 3 == 0 else Decision("none", "custom")


class CustomObserver(OnlineObservability):
    pass


@pytest.mark.parametrize(
    "options",
    [{"policy_type": IntermittentOnline}, {"observer_type": CustomObserver}, {"mode": "full"}],
)
def test_custom_policies_observers_and_full_retention_are_not_optimized(options):
    before, copies_before = run_engine(AlwaysCopyEngine, **options)
    after, copies_after = run_engine(Engine, **options)
    assert_parity(before, after)
    assert copies_after == copies_before
    assert not any(point.get("shadow_unavailable_reason") for point in after.observer.comparisons)


def test_disabled_shadows_do_not_create_comparison_populations():
    before, copies_before = run_engine(AlwaysCopyEngine, shadows=False)
    after, copies_after = run_engine(Engine, shadows=False)
    assert_parity(before, after)
    assert copies_after == copies_before
    assert after.observer.comparisons == []
