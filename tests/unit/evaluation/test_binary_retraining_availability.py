"""Released-label invariants for the event-time batch retraining adaptation."""

import itertools
import json
import math
import time
from dataclasses import replace

import numpy as np
import pytest

from continual_bench.config import Config
from continual_bench.core.contracts import Event, event_json
from continual_bench.evaluation.binary import BinaryRun
from continual_bench.evaluation.binary_retraining import compute_binary_retraining, run_batch_retraining
from continual_bench.retraining import UPF, CaraCT, CaraP, CaraT, Oracle, PerformanceMatrix, simulate
from continual_bench.retraining.analysis import analyze, deployed_metrics
from continual_bench.retraining.policies import CaraThreshold
from continual_bench.retraining.schedule import (
    alpha_max,
    alpha_max_same_batch,
    min_error_by_retrains,
    optimal_schedule,
    optimal_schedule_same_batch,
    schedule_cost,
)


class LabelMean:
    def __init__(self):
        self.rows = []

    def learn(self, x, y):
        self.rows.append((int(x["seq"]), bool(y)))

    def predict(self, x):
        return sum(y for _, y in self.rows) / len(self.rows) if self.rows else 0.5


def events_with_delay(delay, count=120):
    return [
        Event(str(i), i, 1_000 + i, 1_000 + i + delay, {"seq": float(i)}, bool(i % 3)) for i in range(count)
    ]


def build(events, *, replay=11):
    return compute_binary_retraining(
        events, LabelMean, batch_size=10, replay_window=replay, n_alphas=2, mc_samples=4, seed=7
    )


@pytest.mark.parametrize("delay", [0, 3, 40, 1_000])
def test_candidate_fit_uses_only_released_rows_with_exact_replay_limit(delay):
    events = events_with_delay(delay)
    blocks, matrix, states, analysis, _ = build(events)
    for index, state in enumerate(states):
        cutoff = blocks[index][-1].timestamp
        eligible = [e for e in events if e.timestamp <= cutoff and e.available_at <= cutoff][-11:]
        assert state.rows == [(e.seq, e.y) for e in eligible]
        assert len({seq for seq, _ in state.rows}) == len(state.rows)
        record = matrix.training_records[index]
        assert record["cutoff_timestamp"] == cutoff
        assert record["train_samples"] == len(eligible) <= 11
        assert record["pending_samples"] == sum(e.timestamp <= cutoff < e.available_at for e in events)
        if eligible:
            assert record["train_available_at_range"][1] <= cutoff
        else:
            assert record["reason"] == "no_released_training_labels"
    assert analysis["temporal_contract"]["version"] == "released-labels-v1"
    assert analysis["temporal_contract"]["empty_label_information"].startswith("unavailable")
    assert analysis["deployed"]["model_rows"] == [len(state.rows) for state in states]
    assert analysis["deployed"]["source_block_rows"] == [len(block) for block in blocks]


def test_labels_pending_at_one_boundary_are_admitted_later_without_duplication():
    _, matrix, states, _, _ = build(events_with_delay(15))
    assert states[0].rows == []
    assert [seq for seq, _ in states[1].rows] == list(range(5))
    assert [seq for seq, _ in states[2].rows] == list(range(4, 15))
    assert [seq for seq, _ in states[3].rows] == list(range(14, 25))
    assert matrix.training_records[1]["newly_released_samples"] == 5
    assert matrix.training_records[2]["newly_released_samples"] == 10


@pytest.mark.parametrize("make_rule", [CaraT, CaraCT, CaraP, lambda: UPF(samples=4, seed=5)])
def test_unavailable_labels_cannot_change_fit_observations_or_earlier_decisions(make_rule):
    events = events_with_delay(3)
    blocks, matrix, states, _, _ = build(events)
    horizon = 5
    cutoff = blocks[horizon][-1].timestamp
    altered = [replace(e, y=not e.y) if e.available_at > cutoff else e for e in events]
    _, other, other_states, _, _ = build(altered)
    for index in range(horizon + 1):
        assert states[index].rows == other_states[index].rows
        assert states[index].predict({"seq": 0.0}) == other_states[index].predict({"seq": 0.0})
    view = matrix.view(horizon, range(horizon))
    other_view = other.view(horizon, range(horizon))
    assert view.observations() == other_view.observations()
    for model in view.models:
        assert view.staleness(horizon, model) == pytest.approx(other_view.staleness(horizon, model))
    # This is a real counterfactual: retrospective errors do change while the
    # available-information errors and policy decisions do not.
    assert matrix.err[0, horizon] != other.err[0, horizon]
    real = simulate(make_rule(), matrix, 0.05)
    changed = simulate(make_rule(), other, 0.05)
    assert [t for t in real.retrains if t <= horizon] == [t for t in changed.retrains if t <= horizon]
    assert real.deployed[: horizon - matrix.offline + 2] == changed.deployed[: horizon - matrix.offline + 2]


def test_missing_labels_are_unavailable_instead_of_zero_error_or_invented_retrains():
    _, matrix, states, _, _ = build(events_with_delay(1_000))
    view = matrix.offline_view()
    assert view.observations() == []
    assert math.isnan(view.staleness(1, 0))
    with pytest.raises(ValueError, match="no labels observable"):
        view.err(0, 1)
    for rule in [CaraT(), CaraCT(), CaraP(), UPF(samples=4)]:
        result = simulate(rule, matrix, 0.05)
        assert result.retrains == []
    assert all(state.rows == [] for state in states)
    metadata = matrix.to_dict()["temporal_contract"]
    assert all(not r["values"] for r in metadata["observed_errors"])
    assert all(n in (None, 0) for row in metadata["observed_label_counts"] for n in row)


@pytest.mark.parametrize("make_rule", [CaraT, CaraCT, CaraP])
def test_offline_calibration_ignores_labels_still_pending_at_phase_boundary(make_rule):
    events = events_with_delay(3)
    blocks, matrix, _, _, _ = build(events)
    cutoff = blocks[matrix.offline - 1][-1].timestamp
    # Only flip the tail of the offline phase. Those labels are retrospective
    # offline outcomes but were still unknown when parameters were selected.
    altered = [replace(e, y=not e.y) if e.timestamp <= cutoff < e.available_at else e for e in events]
    _, other, _, _, _ = build(altered)
    real, changed = make_rule(), make_rule()
    real.fit_offline(matrix.offline_view(), 0.05)
    changed.fit_offline(other.offline_view(), 0.05)
    assert real.parameters() == changed.parameters()


def test_observed_error_uses_only_eligible_labels_as_its_denominator():
    events = events_with_delay(3)
    blocks, matrix, states, _, _ = build(events)
    t, model = 5, 2
    cutoff = blocks[t][-1].timestamp
    eligible = [e for e in blocks[t] if e.available_at <= cutoff]
    assert len(eligible) == 7
    expected = np.mean([(states[model].predict(e.x) >= 0.5) != e.y for e in eligible])
    assert matrix.view(t, {model}).err(model, t) == pytest.approx(expected)
    assert matrix.to_dict()["temporal_contract"]["observed_label_counts"][t][t] == len(eligible)


def test_batch_observability_releases_labels_at_causal_boundaries_and_drains_tail(tmp_path):
    events = events_with_delay(13)

    class Source:
        def manifest(self):
            return {"changepoints": None}

    class Versions:
        entries = []

        def record(self, model, metadata):
            self.entries.append(metadata)
            return len(self.entries)

    config = Config.model_validate(
        {
            "mode": "incremental",
            "warmup": 10,
            "batch_size": 10,
            "replay_window": 11,
            "dataset": {"name": "sea", "samples": len(events), "label_delay": 13},
            "detector": {"name": "none"},
            "policy": {
                "name": "cara",
                "params": {"alpha": 0.0, "tau": 0.0},
                "min_samples": 1,
                "min_window": 1,
            },
        }
    )
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(event_json(e) for e in events) + "\n")
    run = BinaryRun(
        config=config,
        source=Source(),
        events_path=path,
        model_factory=LabelMean,
        detector_factory=lambda: None,
        policy_factory=lambda: CaraThreshold(params={"alpha": 0.0, "tau": 0.0}),
        versions=Versions(),
        settings=config.tracking_settings().model_dump(mode="json"),
        started=time.monotonic(),
        checkpoint=lambda state: None,
    )
    outcome = run_batch_retraining(run)
    assert outcome.report["labeled"] == len(events)
    assert outcome.observer.continual.next_seq == len(events)
    for point in outcome.observer.predictions:
        assert point["released_at"] >= point["available_at"]
        assert point["trained_labels_at_prediction"] <= sum(
            e.available_at < point["timestamp"] for e in events
        )
    for record in outcome.report["training"]["candidate_training"]:
        if record["train_samples"]:
            assert record["train_available_at_range"][1] <= record["cutoff_timestamp"]
    assert outcome.report["retraining"]["deployed"]["model_rows"] == [
        record["train_samples"] for record in outcome.report["training"]["candidate_training"]
    ]
    assert (
        outcome.report["retraining"]["deployed"]["source_block_rows"]
        == (outcome.report["retraining"]["performance_matrix"]["rows"])
    )
    json.dumps(outcome.report, allow_nan=False)


def test_binary_retraining_rejects_nonmonotone_label_availability():
    events = events_with_delay(3)
    events[1] = replace(events[1], available_at=events[0].available_at - 1)
    with pytest.raises(ValueError, match="ordered timestamps and label availability"):
        build(events)


def unavailable_candidate_matrix(*, paper=False):
    size = 5
    error = np.full((size, size), np.nan)
    error[np.triu_indices(size, 1)] = 1.0
    error[2, 3:] = 0.0  # An attractive but inadmissible candidate.
    error[3, 4] = 0.25
    if paper:
        error[np.arange(2, 5), np.arange(2, 5)] = [0.4, 0.2, 0.3]
    return PerformanceMatrix(
        err=error,
        offline=2,
        points=[np.array([[float(i)]]) for i in range(size)],
        queries=[np.array([[float(i)]]) for i in range(size)],
        sub_loss={
            (i, j): np.array([0.0 if i == j else error[i, j]]) for i in range(size) for j in range(i, size)
        },
        feature_means=np.arange(size, dtype=float)[:, None],
        fit_seconds=np.zeros(size),
        rows=np.full(size, 10),
        paper_timing=paper,
        training_records=[{"train_samples": 0 if i == 2 else 10} for i in range(size if paper else size - 1)],
    )


def schedules(points):
    return [list(s) for count in range(len(points) + 1) for s in itertools.combinations(points, count)]


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.75, 1.0, 10.0])
def test_oracle_dp_equals_enumeration_over_admissible_candidates(alpha):
    matrix = unavailable_candidate_matrix()
    legal = schedules([3])
    minimum = min(schedule_cost(matrix.err, 1, 2, 4, schedule, alpha) for schedule in legal)
    result = simulate(Oracle(), matrix, alpha)
    oracle = Oracle()
    oracle.fit_future(matrix, alpha)
    points, value = optimal_schedule(matrix.err, 1, 2, 4, alpha, allowed_retrains={3})
    assert points in legal and result.retrains in legal
    assert oracle.schedule == result.retrains == points
    assert oracle.cost == pytest.approx(result.cost)
    assert value == pytest.approx(minimum)
    assert result.cost == pytest.approx(minimum)
    if alpha == 0.1:
        assert result.retrains == [3]
        assert result.cost == pytest.approx(2.35)


def test_oracle_derived_alpha_grid_and_exact_retrain_errors_use_legal_support():
    matrix = unavailable_candidate_matrix()
    errors = min_error_by_retrains(matrix.err, 1, 2, 4, 2, allowed_retrains={3})
    assert errors == [3.0, 2.25, math.inf]
    assert alpha_max(matrix.err, 1, 2, 4, allowed_retrains={3}) == pytest.approx(0.75)
    analysis = analyze(matrix, rules=("oracle",), n_alphas=3, focus_alpha=0.1)
    assert analysis["oracle_schedule_at_focus"] == [3]
    assert analysis["setup"]["admissible_retrain_points"] == [3]
    assert analysis["setup"]["oracle_alpha_max"] == pytest.approx(0.75)
    assert analysis["alphas"][-1] == pytest.approx(0.75)
    # Proposition 3.1's unrestricted action-space bound is not claimed for a
    # restricted candidate space. The admissible Oracle itself is still exact.
    assert analysis["setup"]["retrain_bound_available"] is False
    assert all(value is None for value in analysis["setup"]["bound_alphas"].values())
    result = simulate(Oracle(), matrix, 0.1)
    assert deployed_metrics(analysis, "oracle", result, result)["retraining_retrain_upper_bound"] is None


@pytest.mark.parametrize("paper", [False, True])
def test_same_batch_oracle_and_alpha_limit_respect_candidate_support(paper):
    matrix = unavailable_candidate_matrix(paper=paper)
    allowed = sorted(matrix.admissible_retrains)
    stale = np.nan_to_num(matrix.err.T, nan=0.0)
    diagonal = matrix.err.diagonal() if paper else None

    def cost(schedule, alpha):
        deployed, total = matrix.initial, 0.0
        for t in range(matrix.offline, matrix.batches):
            if t in schedule:
                deployed = t
                total += alpha + (diagonal[t] if paper else 0.0)
            else:
                total += stale[t, deployed]
        return total

    legal = schedules(allowed)
    for alpha in [0.0, 0.1, 1.0, 5.0]:
        selected, value = optimal_schedule_same_batch(
            stale, 1, 2, 4, alpha, retrain_errors=diagonal, allowed_retrains=allowed
        )
        assert selected in legal
        assert value == pytest.approx(min(cost(schedule, alpha) for schedule in legal))
        if paper:
            result = simulate(Oracle(), matrix, alpha)
            assert result.retrains == selected
            assert result.cost == pytest.approx(value)
    expected_limit = max([(cost([], 0.0) - cost(s, 0.0)) / len(s) for s in legal if s] or [0.0])
    assert alpha_max_same_batch(
        stale, 1, 2, 4, retrain_errors=diagonal, allowed_retrains=allowed
    ) == pytest.approx(expected_limit)
