from dataclasses import dataclass

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
        return Decision(self.action, self.kind) if triggered else Decision("none", "no_trigger")

    def updated(self, labeled: int) -> None:
        self.last_update = labeled
