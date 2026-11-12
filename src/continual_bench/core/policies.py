"""Retraining policies: when to update, never how a detector signals or a model learns.

Strategy pattern: one class per policy kind. `GuardedPolicy.decide` is a template
method that applies the shared sample/cooldown guards and delegates only the
trigger to subclasses. `PolicyStrategy(kind, ...)` remains the factory entry point.
"""

import math
from typing import ClassVar, Literal

from .contracts import Decision, DecisionContext

ACTIONS = frozenset({"reset_retrain", "learn_one"})
SCENARIO_PARAMS = frozenset({"error_threshold"})
PERMITTED_PARAMS = SCENARIO_PARAMS | {"monitor_window"}
# Cost-aware model-update policies (CARA, UPF) and the retrospective Oracle decide per *batch* from a
# performance matrix; the per-sample Engine cannot host them. They live in `continual_bench.retraining`.
BATCH_POLICY_KINDS = ("cara", "cara_cumulative", "cara_periodic", "upf", "oracle")
ZiaPolicy = Literal["frozen", "online", "drift", "drift_adwin", "drift_ddm",
                    "incremental", "periodic", "cara", "cara_cumulative", "cara_periodic", "upf", "oracle"]


class RetrainingPolicy:
    kind: ClassVar[str]
    accepts_scenario_params: ClassVar[bool] = False
    requires_batches: ClassVar[bool] = False

    def __init__(self, every: int = 100, cooldown: int = 100, min_samples: int = 100, min_window: int = 50,
                 action: str = "reset_retrain", last_update: int = 0, params: dict | None = None):
        self.every, self.cooldown = every, cooldown
        self.min_samples, self.min_window = min_samples, min_window
        self.action, self.last_update = action, last_update
        self.params = {} if params is None else params
        self._validate()

    def _validate(self):
        if self.action not in ACTIONS:
            raise ValueError("invalid policy action")
        if min(self.every, self.min_samples, self.min_window) < 1 or self.cooldown < 0:
            raise ValueError("invalid policy sample guards")
        if self.params.keys() - PERMITTED_PARAMS:
            raise ValueError("unknown policy params")
        for key, value in self.params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"policy {key} must be finite numeric")
            if value < 0:
                raise ValueError(f"policy {key} must be nonnegative")
        if "monitor_window" in self.params and (
                self.params["monitor_window"] < 1
                or int(self.params["monitor_window"]) != self.params["monitor_window"]):
            raise ValueError("policy monitor_window must be a positive integer")
        if self.params.get("error_threshold", 0) > 1:
            raise ValueError("policy error_threshold must be in [0,1]")
        if self.params.get("monitor_window", 1) > 10000:
            raise ValueError("policy monitor_window exceeds 10000")
        if not self.accepts_scenario_params and self.params.keys() & SCENARIO_PARAMS:
            raise ValueError("performance params require a matching policy")

    def decide(self, context: DecisionContext) -> Decision:
        raise NotImplementedError

    def updated(self, labeled: int) -> None:
        self.last_update = labeled

    def __repr__(self):
        return (f"{type(self).__name__}(kind={self.kind!r}, every={self.every}, cooldown={self.cooldown}, "
                f"min_samples={self.min_samples}, min_window={self.min_window}, action={self.action!r}, "
                f"last_update={self.last_update}, params={self.params!r})")


class NoRetraining(RetrainingPolicy):
    """Frozen reference: model and preprocessing stay as fitted during warmup."""

    kind = "none"

    def decide(self, context):
        return Decision("none", "frozen_after_warmup")


class OnlineUpdate(RetrainingPolicy):
    kind = "online"

    def decide(self, context):
        # Warmup and label availability are enforced by the engine. Every subsequent
        # released label must update the model, regardless of retraining guards.
        return Decision("learn_one", "online")


class GuardedPolicy(RetrainingPolicy):
    """Template method: shared min_samples/min_window/cooldown guards, then a trigger."""

    def decide(self, context):
        if context.labeled < self.min_samples or context.window_size < self.min_window:
            return Decision("none", "insufficient_samples")
        if context.labeled - self.last_update < self.cooldown:
            return Decision("none", "cooldown")
        refusal = self.refusal(context)
        if refusal is not None:
            return refusal
        return Decision(self.action, self.kind) if self.triggered(context) else Decision("none", "no_trigger")

    def refusal(self, context: DecisionContext) -> Decision | None:
        return None

    def triggered(self, context: DecisionContext) -> bool:
        raise NotImplementedError


class PeriodicRetraining(GuardedPolicy):
    kind = "periodic"

    def triggered(self, context):
        return context.labeled - self.last_update >= self.every


class DriftTriggeredRetraining(GuardedPolicy):
    """Acts on a detector alarm; the detector itself never decides to retrain."""

    kind = "drift"

    def triggered(self, context):
        return context.drift


class MonitoredPolicy(GuardedPolicy):
    """Policies driven by the recent released-label error rate."""

    accepts_scenario_params = True

    def refusal(self, context):
        required = self.params.get("monitor_window", self.min_window)
        if context.recent_error is None or context.monitored < required:
            return Decision("none", "insufficient_monitoring_labels")
        if not math.isfinite(context.recent_error) or not 0 <= context.recent_error <= 1:
            raise ValueError("recent error must be a finite rate in [0,1]")
        return None


class PerformanceTriggeredRetraining(MonitoredPolicy):
    kind = "performance"

    def triggered(self, context):
        return context.recent_error >= self.params.get("error_threshold", 0.3)


POLICIES: dict[str, type[RetrainingPolicy]] = {
    cls.kind: cls for cls in (NoRetraining, OnlineUpdate, PeriodicRetraining, DriftTriggeredRetraining,
                              PerformanceTriggeredRetraining)
}


def PolicyStrategy(kind: str, **settings) -> RetrainingPolicy:  # noqa: N802 - historical public factory
    if kind in BATCH_POLICY_KINDS:
        from continual_bench.retraining.policies import BATCH_POLICIES  # numpy stays out of the core import

        return BATCH_POLICIES[kind](**settings)
    try:
        policy = POLICIES[kind]
    except KeyError:
        raise ValueError(f"unknown policy: {kind}") from None
    return policy(**settings)
