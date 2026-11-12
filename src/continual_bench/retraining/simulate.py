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

    @property
    def cost(self) -> float:
        """``C_hat_alpha(theta) = alpha * ||theta||_1 + sum_t pe_{r(t), t}`` (eN = 1)."""
        return self.alpha * len(self.retrains) + float(sum(self.errors))

    @property
    def average_accuracy(self) -> float:
        return 1.0 - float(np.mean(self.errors))

    def strategy_cost(self, matrix: PerformanceMatrix) -> float:
        """CARA's strategy cost: summed relative staleness of the deployed models plus retrain costs."""
        stale = sum(matrix.staleness(j, m) for j, m in zip(
            range(matrix.offline, matrix.batches), self.deployed, strict=True))
        return float(stale + self.alpha * len(self.retrains))


def simulate(rule: Rule, matrix: PerformanceMatrix, alpha: float) -> SimResult:
    import time

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
        served.append(deployed)
        errors.append(float(matrix.err[deployed, j]))
        if j == matrix.batches - 1:
            break
        began = time.perf_counter()
        if rule.decide(j, deployed, matrix.view(j, models)):
            retrains.append(j)
            models.add(j)
            deployed = j
        timings.append(time.perf_counter() - began)
    return SimResult(rule.name, float(alpha), retrains, served, errors, rule.parameters(),
                     list(getattr(rule, "forecasts", [])), offline_seconds, timings)
