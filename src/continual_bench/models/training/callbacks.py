"""Early stopping strategies, model training protocols, and Fargate SIGTERM callbacks."""

from __future__ import annotations

import math
import signal
import threading
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class IterativeTrainableModel(Protocol):
    """Protocol for iterative/batch models trained over chronological partitions."""

    def train_epoch(self, train_data: Any) -> dict[str, float]: ...

    def evaluate_split(self, val_data: Any) -> dict[str, float]: ...

    def get_state(self) -> bytes: ...

    def restore_state(self, state: bytes) -> None: ...


@runtime_checkable
class EpochMetricsTracker(Protocol):
    """Protocol for decoupled experiment trackers (e.g., SageMaker Experiments)."""

    def log_epoch_metrics(self, metrics: dict[str, float | int], *, step: int | None = None) -> Any: ...


class EarlyStoppingStrategy:
    """Encapsulates best-score tracking, patience counter, checkpoint bytes, and stopping rule."""

    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 0.0,
        *,
        mode: Literal["min", "max"] = "min",
        restore_best_weights: bool = True,
    ):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0 or not math.isfinite(min_delta):
            raise ValueError("min_delta must be finite and >= 0")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode: Literal["min", "max"] = mode
        self.restore_best_weights = bool(restore_best_weights)
        self.best_score: float = math.inf if mode == "min" else -math.inf
        self.best_epoch: int = 0
        self.patience_counter: int = 0
        self.best_state: bytes | None = None
        self.early_stopped: bool = False
        self.stopped_epoch: int | None = None

    @property
    def best_loss(self) -> float:
        return self.best_score

    @best_loss.setter
    def best_loss(self, value: float) -> None:
        self.best_score = float(value)

    @property
    def best_validation_loss(self) -> float:
        return self.best_score

    @property
    def patience_consumed(self) -> int:
        return self.patience_counter

    @property
    def should_stop(self) -> bool:
        return self.early_stopped

    def is_improvement(self, score: float) -> bool:
        if not math.isfinite(score):
            return False
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        return score > (self.best_score + self.min_delta)

    def step(
        self,
        epoch: int,
        score: float,
        state: bytes | Callable[[], bytes] | None = None,
    ) -> bool:
        """Evaluate epoch score against best_score and return True if training should stop."""
        if self.is_improvement(score):
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.patience_counter = 0
            if state is not None:
                raw = state() if callable(state) else state
                self.best_state = bytes(raw)
            return False

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            self.early_stopped = True
            self.stopped_epoch = int(epoch)
            return True
        return False

    check = step
    on_epoch_end = step
    __call__ = step

    def restore(self, model: IterativeTrainableModel) -> bool:
        """Restore the best checkpoint bytes into the model if available and enabled."""
        if self.restore_best_weights and self.best_state is not None:
            model.restore_state(self.best_state)
            return True
        return False

    def reset(self) -> None:
        self.best_score = math.inf if self.mode == "min" else -math.inf
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_state = None
        self.early_stopped = False
        self.stopped_epoch = None


class MinDeltaEarlyStopping(EarlyStoppingStrategy):
    """Concrete alias for EarlyStoppingStrategy using min_delta and patience."""


class GracefulTerminationHandler:
    """Trap Fargate SIGTERM to allow saving the best/partial state before container exit."""

    def __init__(self, on_terminate: Callable[..., Any] | None = None):
        self.on_terminate = on_terminate
        self.received = False
        self.signum: int | None = None
        self._previous_handler: Any = None
        self._installed = False

    @property
    def terminated(self) -> bool:
        return self.received

    def trigger(self, signum: int = signal.SIGTERM, _frame: Any = None) -> None:
        self.received = True
        self.signum = signum
        if self.on_terminate is not None:
            try:
                self.on_terminate(signum)
            except TypeError:
                self.on_terminate()

    def __enter__(self) -> GracefulTerminationHandler:
        if threading.current_thread() is threading.main_thread():
            try:
                self._previous_handler = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, self.trigger)
                self._installed = True
            except (ValueError, OSError):
                self._installed = False
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._installed and self._previous_handler is not None:
            try:
                signal.signal(signal.SIGTERM, self._previous_handler)
            except (ValueError, OSError):
                pass
            self._installed = False


__all__ = [
    "EarlyStoppingStrategy",
    "EpochMetricsTracker",
    "GracefulTerminationHandler",
    "IterativeTrainableModel",
    "MinDeltaEarlyStopping",
]
