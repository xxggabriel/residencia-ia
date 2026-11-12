"""Iterative training loop (up to 100 epochs) with Early Stopping and Fargate hardening."""

from __future__ import annotations

import gc
import os
import platform
import resource
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from continual_bench.config.iterative import IterativeTrainingConfig

from .callbacks import (
    EarlyStoppingStrategy,
    EpochMetricsTracker,
    GracefulTerminationHandler,
    IterativeTrainableModel,
)


def enforce_single_thread_env(vcpus: int | float | None = None) -> None:
    """Enforce BLAS/OpenMP thread limits (supports 2 threads when vcpus >= 2)
    to prevent CPU throttling in AWS Fargate.
    """
    if vcpus is None:
        raw = os.environ.get("BENCH_BENCHMARK_VCPUS") or os.environ.get("BENCH_VCPUS")
        if raw is not None:
            try:
                vcpus = float(raw)
            except ValueError:
                vcpus = None
    if vcpus is not None:
        threads = "2" if vcpus >= 2 else "1"
    else:
        threads = os.environ.get("OMP_NUM_THREADS", "2")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


enforce_single_thread_env()


def current_peak_rss_mb() -> float:
    """Process peak RSS in MiB (ru_maxrss is bytes on Darwin, KiB on Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return float(usage / divisor)


def unpack_split(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) arrays without copying."""
    if isinstance(data, tuple) and len(data) == 2:
        return data[0], data[1]
    if isinstance(data, list) and len(data) == 2:
        return data[0], data[1]
    if isinstance(data, dict) and "x" in data and "y" in data:
        return data["x"], data["y"]
    raise ValueError("split data must be a (x, y) tuple or {'x': ..., 'y': ...} mapping")


def chronological_train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.1,
    *,
    holdout_start: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Slice chronological train and validation views strictly before the frozen temporal holdout."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows")
    if holdout_start is not None:
        if holdout_start <= 0 or holdout_start > len(y):
            raise ValueError("invalid holdout_start boundary")
        x = x[:holdout_start]
        y = y[:holdout_start]
    total = len(y)
    if total < 2:
        raise ValueError("at least 2 samples are required for chronological train/validation split")
    if not (0.0 < validation_fraction < 0.5):
        raise ValueError("validation_fraction must be between 0 and 0.5")
    val_rows = max(1, int(total * validation_fraction))
    split_idx = max(1, total - val_rows)
    if not (0 < split_idx < total):
        raise ValueError("dataset too small for chronological train/validation split")
    # Zero-copy NumPy slice views: never clone arrays in memory
    train_x, train_y = x[:split_idx], y[:split_idx]
    val_x, val_y = x[split_idx:], y[split_idx:]
    return (train_x, train_y), (val_x, val_y)


def emit_cloudwatch_metrics(metrics: dict[str, float | int]) -> None:
    """Emit regex-formatted metrics to stdout for CloudWatch / SageMaker AMT."""
    keys = (
        "validation:log_loss",
        "validation:accuracy",
        "validation:balanced_accuracy",
        "validation:f1_macro",
        "validation:pr_auc",
        "train:log_loss",
        "train:accuracy",
        "train:balanced_accuracy",
        "train:f1_macro",
        "train:pr_auc",
        "best_validation_loss",
        "epoch",
        "best_epoch",
        "early_stopped",
        "patience_consumed",
    )
    parts = [f"{k}={metrics[k]};" for k in keys if k in metrics and metrics[k] is not None]
    if parts:
        print(" ".join(parts), flush=True)


class TrainingResult(dict):
    """Dictionary of iterative training outcomes supporting attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _resolve_eval_metric(
    eval_metric: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> float:
    if eval_metric.startswith("validation:"):
        key = eval_metric.split(":", 1)[1]
        if key in val_metrics:
            return float(val_metrics[key])
    if eval_metric.startswith("train:"):
        key = eval_metric.split(":", 1)[1]
        if key in train_metrics:
            return float(train_metrics[key])
    if eval_metric in val_metrics:
        return float(val_metrics[eval_metric])
    return float(val_metrics["log_loss"])


class IterativeTrainingLoop:
    """Orchestrates up to max_epochs of iterative training with Early Stopping and tracking."""

    def __init__(
        self,
        config: IterativeTrainingConfig | dict[str, Any] | None = None,
        *,
        strategy: EarlyStoppingStrategy | None = None,
        tracker: EpochMetricsTracker | Any | None = None,
        emit_stdout: bool = False,
        on_sigterm_checkpoint: Callable[[bytes, dict[str, Any]], Any] | None = None,
    ):
        self.config = IterativeTrainingConfig.from_params(config)
        self.strategy = strategy or EarlyStoppingStrategy(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="min",
            restore_best_weights=self.config.restore_best_weights,
        )
        self.tracker = tracker
        self.emit_stdout = emit_stdout
        self.on_sigterm_checkpoint = on_sigterm_checkpoint

    def run(
        self,
        model: IterativeTrainableModel,
        train_data: Any,
        val_data: Any | None = None,
        *,
        holdout_start: int | None = None,
    ) -> TrainingResult:
        enforce_single_thread_env()
        self.strategy.reset()

        if val_data is None:
            x, y = unpack_split(train_data)
            train_split, val_split = chronological_train_val_split(
                x,
                y,
                self.config.validation_fraction,
                holdout_start=holdout_start,
            )
        else:
            if holdout_start is not None:
                tx, ty = unpack_split(train_data)
                vx, vy = unpack_split(val_data)
                if len(ty) + len(vy) > holdout_start:
                    raise ValueError("train/validation splits overlap with the frozen temporal holdout")
                train_split, val_split = (tx, ty), (vx, vy)
            else:
                train_split, val_split = train_data, val_data

        history: list[dict[str, float | int]] = []
        total_train_seconds = 0.0
        total_val_seconds = 0.0
        best_train_metrics: dict[str, float] = {}
        best_val_metrics: dict[str, float] = {}

        with GracefulTerminationHandler() as sigterm:
            for epoch in range(1, self.config.max_epochs + 1):
                if sigterm.received:
                    break

                t0 = time.perf_counter()
                train_metrics = model.train_epoch(train_split)
                train_duration = time.perf_counter() - t0
                total_train_seconds += train_duration

                v0 = time.perf_counter()
                val_metrics = model.evaluate_split(val_split)
                val_duration = time.perf_counter() - v0
                total_val_seconds += val_duration

                target_loss = _resolve_eval_metric(self.config.eval_metric, train_metrics, val_metrics)
                prev_best_epoch = self.strategy.best_epoch
                should_stop = self.strategy.step(epoch, target_loss, state=model.get_state)
                if self.strategy.best_epoch != prev_best_epoch:
                    best_train_metrics = dict(train_metrics)
                    best_val_metrics = dict(val_metrics)

                rss_mb = current_peak_rss_mb()
                epoch_metrics: dict[str, float | int] = {
                    "epoch": int(epoch),
                    "train:log_loss": float(train_metrics["log_loss"]),
                    "train:accuracy": float(train_metrics["accuracy"]),
                    "validation:log_loss": float(val_metrics["log_loss"]),
                    "validation:accuracy": float(val_metrics["accuracy"]),
                    "validation:f1_macro": float(
                        val_metrics.get("f1_macro", val_metrics.get("macro_f1", 0.0))
                    ),
                    "early_stopped": 1 if should_stop else 0,
                    "best_epoch": int(self.strategy.best_epoch),
                    "patience_consumed": int(self.strategy.patience_counter),
                    "training_duration_seconds": float(train_duration),
                    "validation_duration_seconds": float(val_duration),
                    "peak_rss_mb": float(rss_mb),
                    "best_validation_loss": float(self.strategy.best_loss),
                }
                if "balanced_accuracy" in train_metrics:
                    epoch_metrics["train:balanced_accuracy"] = float(train_metrics["balanced_accuracy"])
                if "pr_auc" in train_metrics:
                    epoch_metrics["train:pr_auc"] = float(train_metrics["pr_auc"])
                if "f1_macro" in train_metrics:
                    epoch_metrics["train:f1_macro"] = float(train_metrics["f1_macro"])
                if "balanced_accuracy" in val_metrics:
                    epoch_metrics["validation:balanced_accuracy"] = float(val_metrics["balanced_accuracy"])
                if "pr_auc" in val_metrics:
                    epoch_metrics["validation:pr_auc"] = float(val_metrics["pr_auc"])
                history.append(epoch_metrics)

                if self.emit_stdout:
                    emit_cloudwatch_metrics(epoch_metrics)

                if self.tracker is not None:
                    try:
                        if hasattr(self.tracker, "log_epoch_metrics"):
                            self.tracker.log_epoch_metrics(epoch_metrics, step=epoch)
                        elif hasattr(self.tracker, "log_epoch"):
                            self.tracker.log_epoch(epoch_metrics)
                    except Exception:
                        # Fail-safe: transient tracker failures never abort benchmark execution
                        if not self.emit_stdout:
                            emit_cloudwatch_metrics(epoch_metrics)

                if should_stop or sigterm.received:
                    break

            restored = False
            if self.config.restore_best_weights and self.strategy.best_state is not None:
                restored = self.strategy.restore(model)

            if sigterm.received and self.on_sigterm_checkpoint is not None:
                state_bytes = (
                    self.strategy.best_state
                    if self.strategy.best_state is not None
                    else model.get_state()
                )
                self.on_sigterm_checkpoint(
                    state_bytes,
                    {
                        "interrupted": True,
                        "signal": sigterm.signum,
                        "epochs_completed": len(history),
                        "best_epoch": self.strategy.best_epoch,
                        "best_validation_loss": self.strategy.best_loss,
                    },
                )

        if hasattr(model, "release_training_cache"):
            model.release_training_cache()
        gc.collect()

        epochs_run = len(history)
        result = TrainingResult(
            epochs_run=epochs_run,
            epochs_trained=epochs_run,
            iterations=epochs_run,
            best_epoch=self.strategy.best_epoch,
            best_loss=self.strategy.best_loss,
            best_validation_loss=self.strategy.best_loss,
            early_stopped=self.strategy.early_stopped,
            patience_consumed=self.strategy.patience_counter,
            restored_best=restored,
            stopped_by_signal=sigterm.received,
            training_duration_seconds=total_train_seconds,
            validation_duration_seconds=total_val_seconds,
            peak_rss_mb=current_peak_rss_mb(),
            train_metrics=best_train_metrics,
            validation_metrics=best_val_metrics,
            validation=best_val_metrics,
            history=history,
        )
        if self.tracker is not None and hasattr(self.tracker, "finalize"):
            try:
                self.tracker.finalize(
                    {
                        "epochs_trained": epochs_run,
                        "best_epoch": self.strategy.best_epoch,
                        "early_stopped": self.strategy.early_stopped,
                        "best_validation_loss": self.strategy.best_loss,
                    }
                )
            except Exception:
                pass
        return result

    __call__ = run


def train_iterative(
    model: IterativeTrainableModel,
    train_data: Any,
    val_data: Any | None = None,
    config: IterativeTrainingConfig | dict[str, Any] | None = None,
    *,
    strategy: EarlyStoppingStrategy | None = None,
    tracker: EpochMetricsTracker | Any | None = None,
    emit_stdout: bool = False,
    holdout_start: int | None = None,
    on_sigterm_checkpoint: Callable[[bytes, dict[str, Any]], Any] | None = None,
) -> TrainingResult:
    """Convenience entry point to run the iterative training loop with early stopping."""
    loop = IterativeTrainingLoop(
        config=config,
        strategy=strategy,
        tracker=tracker,
        emit_stdout=emit_stdout,
        on_sigterm_checkpoint=on_sigterm_checkpoint,
    )
    return loop.run(model, train_data, val_data, holdout_start=holdout_start)


run_training_loop = train_iterative

__all__ = [
    "EarlyStoppingStrategy",
    "IterativeTrainableModel",
    "IterativeTrainingLoop",
    "TrainingResult",
    "chronological_train_val_split",
    "current_peak_rss_mb",
    "emit_cloudwatch_metrics",
    "enforce_single_thread_env",
    "run_training_loop",
    "train_iterative",
    "unpack_split",
]
