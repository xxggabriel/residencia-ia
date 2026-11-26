"""Regression checks for the CARA authors' experiment protocol."""

import math

import numpy as np
import pytest

from continual_bench.evaluation.cara_replication import (
    Phase,
    build_phase,
    evaluate_kappa,
    prequential_accuracy,
)
from continual_bench.retraining.cara_reference import (
    fit_period,
    fit_threshold,
    oracle_schedule,
    periodic_schedule,
    strategy_cost,
    threshold_schedule,
)
from continual_bench.retraining.rules import MAX_CANDIDATES, _candidates
from continual_bench.retraining.schedule import alpha_max_same_batch


def test_cara_phase_holds_out_queries_and_uses_summed_qdm():
    pytest.importorskip("sklearn")
    from sklearn.model_selection import train_test_split

    class ConstantLearner:
        def fit(self, index, x, y):
            assert len(y) == 18
            return index

        def losses(self, model, x, y):
            return model != y

    _, held_out = train_test_split(np.arange(20), test_size=0.1, random_state=0)
    first = np.zeros(20, dtype=int)
    first[held_out] = 1
    second = np.ones(20, dtype=int)
    second[held_out] = 0
    x = np.zeros((20, 1))

    phase = build_phase([(x, first), (x, second)], ConstantLearner(), gamma=20)
    assert phase.stale[0, 1] == pytest.approx(2.0)
    assert phase.accuracy[0, 1] == 1.0
    assert phase.accuracy[1, 1] == 0.0
    assert phase.stale[1, 1] == 0.0


def test_cara_reference_strict_thresholds_and_period_phase():
    cost = np.array([[0, 1, 1, -1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    assert threshold_schedule(cost, 1) == [0]
    assert threshold_schedule(cost, 1, cumulative=True) == [0, 2]
    period, offset = fit_period(cost, 100)
    assert (period, offset) == (4, 0)
    assert periodic_schedule(10, period, offset) == [0, 4, 8]
    assert periodic_schedule(10, 3, 2) == [0, 2, 5, 8]


def test_cara_reference_signed_scpe_and_initial_cost():
    pytest.importorskip("scipy")
    cost = np.array([[0.0, -0.5, -0.4], [0, 0, -0.9], [0, 0, 0]])
    accuracy = np.triu(np.full_like(cost, 0.75))
    phase = Phase(cost, accuracy, [])
    report = evaluate_kappa(phase, phase, kappa=0.2, seed=0)
    assert oracle_schedule(cost, 0.2) == [0]
    assert strategy_cost(cost, [0], 0.2) == pytest.approx(-0.7)
    assert report["oracle_staleness"]["strategy_scpe"] == 0.0
    assert report["always_retrain"]["strategy_scpe"] == pytest.approx(100 * 1.3 / 0.7)


def test_oracle_boundary_brackets_even_when_frozen_cost_is_negative():
    stale = np.zeros((5, 5))
    stale[2:, 1] = [0.1, -0.5, -0.4]
    stale[3:, 2] = [-0.9, -0.9]
    assert alpha_max_same_batch(stale, 1, 2, 4) == pytest.approx(1.0)


def test_cara_candidate_cap_keeps_never_retrain_sentinel():
    candidates = _candidates(np.linspace(0.0, 1.0, MAX_CANDIDATES + 10), [math.inf])
    assert candidates[-1] == math.inf
    assert all(not math.isnan(value) for value in candidates)


def test_cara_reference_final_accuracy_and_degenerate_annealing():
    pytest.importorskip("scipy")
    cost = np.zeros((3, 3))
    for cumulative in (False, True):
        assert fit_threshold(cost, 0, cumulative=cumulative, seed=0) == 0
    accuracy = np.array([[0.9, 0.5, 0.0], [0, 0.8, 0.6], [0, 0, 0.3]])
    assert prequential_accuracy(accuracy, [0, 1, 2]) == pytest.approx((0.9 + 0.5 + 0.3) / 3)
    assert prequential_accuracy(accuracy, [2, 1, 0]) == pytest.approx((0.9 + 0.5 + 0.6) / 3)
