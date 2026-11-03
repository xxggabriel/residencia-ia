import math
from dataclasses import dataclass, field

from .contracts import Decision, DecisionContext


@dataclass
class PolicyStrategy:
    kind: str
    every: int = 100
    cooldown: int = 100
    min_samples: int = 100
    min_window: int = 50
    action: str = "reset_retrain"
    last_update: int = 0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        allowed = {"none", "online", "periodic", "drift", "performance", "cost_heuristic"}
        if self.kind not in allowed:
            raise ValueError(f"unknown policy: {self.kind}")
        if self.action not in {"reset_retrain", "learn_one"}:
            raise ValueError("invalid policy action")
        if min(self.every, self.min_samples, self.min_window) < 1 or self.cooldown < 0:
            raise ValueError("invalid policy sample guards")
        permitted = {"monitor_window", "error_threshold", "target_error", "horizon",
                     "error_cost", "fixed_train_cost", "sample_train_cost"}
        if self.params.keys() - permitted:
            raise ValueError("unknown policy params")
        for key, value in self.params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"policy {key} must be finite numeric")
            if value < 0:
                raise ValueError(f"policy {key} must be nonnegative")
        for key in ("monitor_window", "horizon"):
            if key in self.params and (self.params[key] < 1 or int(self.params[key]) != self.params[key]):
                raise ValueError(f"policy {key} must be a positive integer")
        for key in ("error_threshold", "target_error"):
            if self.params.get(key, 0) > 1:
                raise ValueError(f"policy {key} must be in [0,1]")
        if self.params.get("monitor_window", 1) > 10000:
            raise ValueError("policy monitor_window exceeds 10000")
        specific = permitted - {"monitor_window"}
        if self.kind not in {"performance", "cost_heuristic"} and self.params.keys() & specific:
            raise ValueError("cost/performance params require a matching policy")


    def decide(self, context: DecisionContext) -> Decision:
        if self.kind == "none":
            return Decision("none", "frozen_after_warmup")
        if self.kind == "online":
            # Warmup and label availability are enforced by the engine. Every subsequent
            # released label must update the model, regardless of retraining guards.
            return Decision("learn_one", "online")
        if context.labeled < self.min_samples or context.window_size < self.min_window:
            return Decision("none", "insufficient_samples")
        if context.labeled - self.last_update < self.cooldown:
            return Decision("none", "cooldown")
        triggered = (
            (self.kind == "periodic" and context.labeled - self.last_update >= self.every)
            or (self.kind == "drift" and context.drift)
        )
        if self.kind in {"performance", "cost_heuristic"}:
            required = self.params.get("monitor_window", self.min_window)
            if context.recent_error is None or context.monitored < required:
                return Decision("none", "insufficient_monitoring_labels")
            if not math.isfinite(context.recent_error) or not 0 <= context.recent_error <= 1:
                raise ValueError("recent error must be a finite rate in [0,1]")
            if self.kind == "performance":
                triggered = context.recent_error >= self.params.get("error_threshold", 0.3)
            else:
                # Explicit scenario-cost heuristic, NOT CARA or UPF reproduction.
                # Assumes a retrain could recover target_error over the configured horizon.
                excess = max(0.0, context.recent_error - self.params.get("target_error", 0.1))
                avoided = excess * self.params.get("horizon", 100) * self.params.get("error_cost", 1.0)
                training = (self.params.get("fixed_train_cost", 1.0)
                            + context.window_size * self.params.get("sample_train_cost", 0.0))
                if not math.isfinite(avoided) or not math.isfinite(training):
                    raise ValueError("scenario cost arithmetic overflow")
                triggered = avoided > training
        return Decision(self.action, self.kind) if triggered else Decision("none", "no_trigger")

    def updated(self, labeled: int) -> None:
        self.last_update = labeled
