"""Replay a rule over a performance matrix (the evaluation protocol of the UPF paper)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .matrix import PerformanceMatrix
from .rules import UPF, Rule


@dataclass
class SimResult:
    rule: str
    alpha: float
    retrains: list[int]
    deployed: list[int]  # model serving each online batch
    errors: list[float]
    parameters: dict = field(default_factory=dict)
    forecasts: list[dict] = field(default_factory=list)
    offline_seconds: float = 0.0
    decision_seconds: list[float] = field(default_factory=list)
    same_batch: bool = False

    @property
    def cost(self) -> float:
        """``C_hat_alpha(theta) = alpha * ||theta||_1 + sum_t pe_{r(t), t}`` (eN = 1)."""
        return self.alpha * len(self.retrains) + float(sum(self.errors))

    @property
    def average_accuracy(self) -> float:
        return 1.0 - float(np.mean(self.errors))

    def strategy_cost(self, matrix: PerformanceMatrix) -> float:
        """CARA's strategy cost: summed relative staleness of the deployed models plus retrain costs."""
        # Same-batch accounting: a retrain at ``j`` replaces the staleness of ``j`` (model ``j`` serves it).
        stale = sum(
            0.0 if m == j else matrix.staleness(j, m)
            for j, m in zip(range(matrix.offline, matrix.batches), self.deployed, strict=True)
        )
        return float(stale + self.alpha * len(self.retrains))


def simulate(rule: Rule, matrix: PerformanceMatrix, alpha: float, same_batch: bool = False) -> SimResult:
    """Replay ``rule``. By default a retrain decided after batch ``j`` serves ``j + 1`` onward.

    ``same_batch=True`` is CARA's accounting: the decision for batch ``j`` is taken from its queries and the
    retrained model serves ``j`` itself (staleness 0, cost ``alpha``). Without a held-out diagonal,
    ``errors`` keeps the previous model's error there; use :meth:`SimResult.strategy_cost`.

    A paper matrix has a held-out diagonal. UPF decides with the history through ``j - 1``
    and the selected model serves the held-out split of ``j``.
    """
    import time

    if same_batch and isinstance(rule, UPF) and not matrix.paper_timing:
        raise ValueError("UPF forecasts the next batches; it has no same-batch accounting")

    started = time.perf_counter()
    rule.fit_offline(matrix.offline_view(), alpha)
    if isinstance(rule, UPF):
        rule.bind(matrix)
    if rule.needs_future:
        rule.fit_future(matrix, alpha)
    offline_seconds = time.perf_counter() - started
    rule.begin_online()
    deployed, models = matrix.initial, set(range(matrix.offline))
    retrains, served, errors, timings = [], [], [], []
    for j in range(matrix.offline, matrix.batches):
        if matrix.paper_timing:
            # UPF commits before seeing j. Query-based CARA baselines can observe its queries.
            upto = j - 1 if isinstance(rule, UPF) else j
            began = time.perf_counter()
            if matrix.candidate_available(j) and rule.decide(j, deployed, matrix.view(upto, models)):
                retrains.append(j)
                models.add(j)
                deployed = j
            timings.append(time.perf_counter() - began)
            served.append(deployed)
            errors.append(float(matrix.err[deployed, j]))
            continue
        if same_batch:
            errors.append(float(matrix.err[deployed, j]))
            began = time.perf_counter()
            if matrix.candidate_available(j) and rule.decide(j, deployed, matrix.view(j, models)):
                retrains.append(j)
                models.add(j)
                deployed = j
            timings.append(time.perf_counter() - began)
            served.append(deployed)
            continue
        served.append(deployed)
        errors.append(float(matrix.err[deployed, j]))
        if j == matrix.batches - 1:
            break
        began = time.perf_counter()
        if matrix.candidate_available(j) and rule.decide(j, deployed, matrix.view(j, models)):
            retrains.append(j)
            models.add(j)
            deployed = j
        timings.append(time.perf_counter() - began)
    return SimResult(
        rule.name,
        float(alpha),
        retrains,
        served,
        errors,
        rule.parameters(),
        list(getattr(rule, "forecasts", [])),
        offline_seconds,
        timings,
        same_batch or matrix.paper_timing,
    )
