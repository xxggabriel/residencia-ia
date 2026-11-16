"""Exact retraining schedules by dynamic programming (the Oracle of both papers).

A schedule is the list of batches ``t`` after which the model is retrained; the new model serves
batches ``t + 1`` onward. ``cost[i, j]`` is the cost of serving batch ``j`` with the model trained at
``i`` (error rate for UPF's C_hat, relative staleness for CARA's strategy cost). A retrain costs
``alpha``. Complexity ``O(B^2)`` time and memory, matching CARA's Oracle (Algorithms 2-3).
"""

from __future__ import annotations

import numpy as np


def schedule_cost(cost: np.ndarray, initial: int, first: int, last: int,
                  retrains: list[int], alpha: float) -> float:
    """``alpha * |schedule| + sum_j cost[deployed(j), j]`` for batches ``first..last``."""
    deployed, total, points = initial, 0.0, set(retrains)
    for j in range(first, last + 1):
        total += float(cost[deployed, j])
        if j in points:
            total += alpha
            deployed = j
    return total


def optimal_schedule(cost: np.ndarray, initial: int, first: int, last: int,
                     alpha: float) -> tuple[list[int], float]:
    """Minimum-cost retraining points in ``[first, last - 1]`` for batches ``first..last``."""
    if not first <= last or alpha < 0:
        raise ValueError("invalid phase or retraining cost")
    prefix = np.zeros((last + 2, last + 2))  # prefix[m, c] = sum of cost[m, first..c-1] (c > first)
    for m in range(min(initial, first - 1), last):
        start = max(first, m + 1)
        for c in range(start, last + 1):
            prefix[m, c + 1] = prefix[m, c] + cost[m, c]
    models = sorted({initial, *range(first, last)})
    best: dict[int, tuple[float, int | None]] = {}
    for m in reversed(models):
        start = max(first, m + 1)
        stay = prefix[m, last + 1] - prefix[m, start]
        choice, value = None, stay
        for nxt in range(start, last):
            candidate = prefix[m, nxt + 1] - prefix[m, start] + alpha + best[nxt][0]
            if candidate < value - 1e-12:
                choice, value = nxt, candidate
        best[m] = (value, choice)
    schedule, node = [], initial
    while best[node][1] is not None:
        node = best[node][1]
        schedule.append(node)
    return schedule, best[initial][0]


def min_error_by_retrains(cost: np.ndarray, initial: int, first: int, last: int,
                          max_retrains: int) -> list[float]:
    """``E_r``: minimum summed cost with exactly ``r`` retrains, ``r = 0..max_retrains`` (inf: impossible)."""
    models = sorted({initial, *range(first, last)})
    prefix = {}
    for m in models:
        start = max(first, m + 1)
        sums = np.concatenate([[0.0], np.cumsum(cost[m, start:last + 1])])
        prefix[m] = (start, sums)

    def segment(m, upto):  # serve batches start..upto with model m
        start, sums = prefix[m]
        return float(sums[upto - start + 1])

    inf = float("inf")
    table = {m: [inf] * (max_retrains + 1) for m in models}
    for m in reversed(models):
        table[m][0] = segment(m, last)
        for r in range(1, max_retrains + 1):
            start = max(first, m + 1)
            options = [segment(m, nxt) + table[nxt][r - 1] for nxt in range(start, last)]
            table[m][r] = min(options, default=inf)
    return table[initial]


def alpha_max(cost: np.ndarray, initial: int, first: int, last: int) -> float:
    """Smallest ``alpha`` at which the Oracle no longer retrains: ``max_r (E_0 - E_r) / r``."""
    horizon = last - first
    errors = min_error_by_retrains(cost, initial, first, last, horizon)
    gains = [(errors[0] - errors[r]) / r for r in range(1, horizon + 1) if np.isfinite(errors[r])]
    return max(0.0, max(gains, default=0.0))


def lipschitz_bound(err: np.ndarray, first: int) -> float:
    """Empirical ``L >= |pe[i, j] - pe[i+1, j]|`` over observed online pairs (UPF Proposition 3.1)."""
    gaps = []
    size = err.shape[0]
    for j in range(first, size):
        for i in range(j):
            if np.isfinite(err[i, j]) and np.isfinite(err[i + 1, j]):
                gaps.append(abs(err[i, j] - err[i + 1, j]))
    return float(max(gaps, default=0.0))


def retrain_upper_bound(alpha: float, lipschitz: float, horizon: int) -> float:
    """Proposition 3.1: the optimal schedule has at most ``T - sqrt(alpha / L)`` retrains."""
    if lipschitz <= 0:
        return 0.0
    return max(0.0, horizon - (alpha / lipschitz) ** 0.5)


def alpha_for_retrain_bound(retrains: int, lipschitz: float, horizon: int) -> float:
    """``alpha`` above which Proposition 3.1 guarantees at most ``retrains`` retrains: ``L (T - r)^2``."""
    return lipschitz * max(0, horizon - retrains) ** 2


def optimal_schedule_same_batch(stale: np.ndarray, initial: int, first: int, last: int,
                                alpha: float, *, retrain_errors: np.ndarray | None = None
                                ) -> tuple[list[int], float]:
    """CARA's accounting (Eq. 4-6, 13-14): a retrain at ``t`` costs ``alpha`` *instead of* the staleness of
    batch ``t`` (the new model is trained on it and serves it with staleness 0) and is allowed at ``last``.

    Pass held-out diagonal errors as ``retrain_errors`` for the UPF error objective.
    Returns ``(retrain batches, cost)``.
    """
    if not first <= last or alpha < 0:
        raise ValueError("invalid phase or retraining cost")
    value = {initial: 0.0}
    parent: list[dict[int, int | None]] = []  # parent[t - first][m] = model deployed before batch t
    for t in range(first, last + 1):
        step = {m: c + float(stale[t, m]) for m, c in value.items()}
        origin: dict[int, int | None] = {m: m for m in value}
        cheapest = min(value, key=value.get)
        fresh = 0.0 if retrain_errors is None else float(retrain_errors[t])
        step[t], origin[t] = value[cheapest] + alpha + fresh, cheapest
        value = step
        parent.append(origin)
    node = min(value, key=value.get)
    total, retrains = value[node], []
    for t in range(last, first - 1, -1):
        previous = parent[t - first][node]
        if node == t:
            retrains.append(t)
        node = previous
    return sorted(retrains), total


def alpha_max_same_batch(stale: np.ndarray, initial: int, first: int, last: int,
                         *, retrain_errors: np.ndarray | None = None) -> float:
    """Smallest ``alpha`` at which :func:`optimal_schedule_same_batch` stops retraining (bisection)."""
    def switches(alpha):
        schedule, _ = optimal_schedule_same_batch(stale, initial, first, last, alpha,
                                                  retrain_errors=retrain_errors)
        return schedule

    high = 1.0
    if not switches(0.0):
        return 0.0
    while switches(high):
        high *= 2.0
    low = 0.0
    for _ in range(50):
        mid = (low + high) / 2
        if switches(mid):
            low = mid
        else:
            high = mid
    return high
