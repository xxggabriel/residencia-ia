"""CARA, UPF and Oracle as model-update policies.

They plug into the same ``PolicyStrategy`` factory as the other policies but decide per *batch*:
``prepare`` receives the performance matrix, ``decide`` is called once per observed batch ``t``
(``DecisionContext.batch``) and answers from a masked view of the matrix. The Oracle is the only one
allowed to read future entries, which makes it a lower bound rather than a deployable policy.
"""

from __future__ import annotations

from typing import ClassVar

from continual_bench.core.contracts import Decision, DecisionContext
from continual_bench.core.policies import RetrainingPolicy, validate_batch_policy_params

from .matrix import PerformanceMatrix
from .rules import UPF, CaraCT, CaraP, CaraT, Oracle, Rule


class BatchPolicy(RetrainingPolicy):
    requires_batches: ClassVar[bool] = True
    permitted: ClassVar[frozenset[str]] = frozenset({"alpha"})

    def __init__(
        self,
        params: dict | None = None,
        objective: str = "error",
        action: str = "reset_retrain",
        **guards,
    ):
        self.objective = objective
        super().__init__(params=params, action=action, **guards)
        self.alpha = float(self.params.get("alpha", 1.0))
        self.rule: Rule | None = None
        self.matrix: PerformanceMatrix | None = None
        self.deployed = -1
        self.models: set[int] = set()
        self.retrains: list[int] = []

    def _validate(self):
        validate_batch_policy_params(self.kind, self.action, self.params)

    def make_rule(self) -> Rule:
        raise NotImplementedError

    def prepare(self, matrix: PerformanceMatrix) -> None:
        """Offline phase: fit rule parameters on the offline batches (Oracle: solve the DP)."""
        self.matrix, self.rule = matrix, self.make_rule()
        self.rule.fit_offline(matrix.offline_view(), self.alpha)
        if isinstance(self.rule, UPF):
            self.rule.bind(matrix)
        if self.rule.needs_future:
            self.rule.fit_future(matrix, self.alpha)
        self.rule.begin_online()
        self.deployed, self.models, self.retrains = matrix.initial, set(range(matrix.offline)), []

    def decide(self, context: DecisionContext) -> Decision:
        if self.rule is None or self.matrix is None:
            raise ValueError("batch policy needs prepare(matrix) before the online phase")
        t = context.labeled if context.batch is None else context.batch
        if not self.matrix.offline <= t <= self.matrix.batches - 2:
            return Decision("none", "outside_decision_horizon")
        if self.rule.decide(t, self.deployed, self.matrix.view(t, self.models)):
            self.retrains.append(t)
            self.models.add(t)
            self.deployed = t
            return Decision("reset_retrain", self.kind)
        return Decision("none", "no_trigger")

    def updated(self, labeled: int) -> None:
        self.last_update = labeled


class CaraThreshold(BatchPolicy):
    kind = "cara"
    permitted = frozenset({"alpha", "tau"})

    def make_rule(self):
        return CaraT(self.params.get("tau"))


class CaraCumulative(BatchPolicy):
    kind = "cara_cumulative"
    permitted = frozenset({"alpha", "tau_cum"})

    def make_rule(self):
        return CaraCT(self.params.get("tau_cum"))


class CaraPeriodic(BatchPolicy):
    kind = "cara_periodic"
    permitted = frozenset({"alpha", "period"})

    def make_rule(self):
        return CaraP(self.params.get("period"))


class UncertaintyPerformanceForecaster(BatchPolicy):
    kind = "upf"
    permitted = frozenset({"alpha", "delta", "samples", "seed"})

    def make_rule(self):
        return UPF(
            delta=self.params.get("delta", 0.95),
            samples=int(self.params.get("samples", 100)),
            seed=int(self.params.get("seed", 0)),
        )


class RetrospectiveOracle(BatchPolicy):
    kind = "oracle"

    def make_rule(self):
        return Oracle(self.objective)


BATCH_POLICIES: dict[str, type[BatchPolicy]] = {
    cls.kind: cls
    for cls in (
        CaraThreshold,
        CaraCumulative,
        CaraPeriodic,
        UncertaintyPerformanceForecaster,
        RetrospectiveOracle,
    )
}
