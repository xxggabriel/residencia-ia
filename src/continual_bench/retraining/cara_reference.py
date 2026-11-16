"""CARA decisions on a phase-local, query-summed cost matrix from the authors' protocol."""

from __future__ import annotations

import numpy as np

from .schedule import optimal_schedule_same_batch


def threshold_schedule(cost: np.ndarray, threshold: float, *, cumulative: bool = False) -> list[int]:
    """Use the reference's strict comparison, including the final decision."""
    deployed, accumulated = 0, 0.0
    retrains = [0]
    for t in range(1, len(cost)):
        stale = cost[deployed, t]
        accumulated = accumulated + stale if cumulative else stale
        if accumulated > threshold:
            retrains.append(t)
            deployed, accumulated = t, 0.0
    return retrains


def strategy_cost(cost: np.ndarray, retrains: list[int], kappa: float) -> float:
    """The initial fit is charged like subsequent retrains but excluded from their count."""
    batches = np.arange(len(cost))
    deployed = np.zeros(len(cost), dtype=int)
    deployed[retrains] = retrains
    deployed = np.maximum.accumulate(deployed)
    return float(np.where(deployed == batches, kappa, cost[deployed, batches]).sum())


def strategy_scpe(cost: float, optimum: float) -> float | None:
    """Percentage error relative to the Oracle, including signed strategy costs."""
    return 100.0 * abs((cost - optimum) / optimum) if optimum != 0 else None


def periodic_schedule(batches: int, period: int, offset: int = 0) -> list[int]:
    return [0, *range(offset or period, batches, period)]


def fit_threshold(cost: np.ndarray, kappa: float, *, cumulative: bool, seed: int) -> float:
    from scipy.optimize import dual_annealing

    indices = np.triu_indices(len(cost), k=1)
    upper = max(kappa, float(cost[indices].max(initial=0.0)))
    if cumulative:
        upper *= len(cost) - 1
    if upper <= 0:
        return 0.0

    def objective(parameters):
        retrains = threshold_schedule(cost, parameters[0], cumulative=cumulative)
        return strategy_cost(cost, retrains, kappa)

    result = dual_annealing(objective, [(0.0, upper)], seed=seed)
    if not result.success:
        raise RuntimeError(f"CARA threshold optimization failed: {result.message}")
    return float(result.x[0])


def oracle_schedule(cost: np.ndarray, kappa: float) -> list[int]:
    schedule, _ = optimal_schedule_same_batch(cost.T, 0, 1, len(cost) - 1, kappa)
    return [0, *schedule]


def fit_period(cost: np.ndarray, kappa: float) -> tuple[int, int]:
    period = min(range(1, len(cost) + 1),
                 key=lambda p: strategy_cost(cost, periodic_schedule(len(cost), p), kappa))
    return period, (-len(cost)) % period
