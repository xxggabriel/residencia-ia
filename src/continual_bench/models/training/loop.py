"""Iterative training loop (up to 100 epochs) with Early Stopping and Fargate hardening."""

from __future__ import annotations

import gc
import os
import platform
import resource
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import numpy as np

from continual_bench.config.iterative import METRIC_DIRECTIONS, IterativeTrainingConfig

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
    timestamps: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Slice train and validation before holdout, keeping each timestamp in one partition."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows")
    if timestamps is not None and len(timestamps) != len(y):
        raise ValueError("timestamps and labels must have the same number of rows")
    if holdout_start is not None:
        if holdout_start <= 0 or holdout_start > len(y):
            raise ValueError("invalid holdout_start boundary")
        x = x[:holdout_start]
        y = y[:holdout_start]
        if timestamps is not None:
            timestamps = timestamps[:holdout_start]
    total = len(y)
    if not total or (timestamps is None and total < 2):
        raise ValueError("at least 2 samples are required for chronological train/validation split")
    if not (0.0 < validation_fraction < 0.5):
        raise ValueError("validation_fraction must be between 0 and 0.5")
    val_rows = max(1, int(total * validation_fraction))
    split_idx = total - val_rows
    if timestamps is not None:
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("timestamps must be sorted")
        timestamp = timestamps[split_idx]
        before = int(np.searchsorted(timestamps, timestamp, side="left"))
        after = int(np.searchsorted(timestamps, timestamp, side="right"))
        boundaries = [boundary for boundary in (before, after) if 0 < boundary < total]
        # A homogeneous window can train the model but cannot provide temporal validation.
        split_idx = min(boundaries, key=lambda boundary: abs(split_idx - boundary), default=total)
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
        "validation:macro_pr_auc",
        "train:log_loss",
        "train:accuracy",
        "train:balanced_accuracy",
        "train:f1_macro",
        "train:pr_auc",
        "train:macro_pr_auc",
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


def _row_count(data: Any) -> int | None:
    try:
        _, y = unpack_split(data)
    except ValueError:
        return None
    return len(y)


def _selection_coordinates(eval_metric: str, val_rows: int | None) -> tuple[str, str]:
    scope, separator, name = eval_metric.partition(":")
    if not separator:
        scope, name = "validation", scope
    return ("train" if val_rows == 0 else scope), name


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
        capture_sigterm: bool = True,
        stop_requested: Callable[[], bool] | None = None,
    ):
        self.config = IterativeTrainingConfig.from_params(config)
        self.strategy = strategy or EarlyStoppingStrategy(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode=METRIC_DIRECTIONS[self.config.eval_metric.split(":")[-1]],
            restore_best_weights=self.config.restore_best_weights,
        )
        self.tracker = tracker
        self.emit_stdout = emit_stdout
        self.on_sigterm_checkpoint = on_sigterm_checkpoint
        self.capture_sigterm = capture_sigterm
        self.stop_requested = stop_requested

    def run(
        self,
        model: IterativeTrainableModel,
        train_data: Any,
        val_data: Any | None = None,
        *,
        holdout_start: int | None = None,
        timestamps: np.ndarray | None = None,
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
                timestamps=timestamps,
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

        train_rows, val_rows = _row_count(train_split), _row_count(val_split)
        selection_scope, selection_metric = _selection_coordinates(self.config.eval_metric, val_rows)
        history: list[dict[str, float | int | None]] = []
        total_train_seconds = 0.0
        total_val_seconds = 0.0
        best_train_metrics: dict[str, float | None] = {}
        best_val_metrics: dict[str, float | None] = {}

        sigterm = GracefulTerminationHandler()

        def interrupted() -> bool:
            return sigterm.received or (self.stop_requested is not None and self.stop_requested())

        with sigterm if self.capture_sigterm else nullcontext():
            for epoch in range(1, self.config.max_epochs + 1):
                if interrupted():
                    break

                t0 = time.perf_counter()
                train_metrics = model.train_epoch(train_split)
                train_duration = time.perf_counter() - t0
                total_train_seconds += train_duration

                v0 = time.perf_counter()
                val_metrics = model.evaluate_split(val_split) if val_rows != 0 else {}
                val_duration = time.perf_counter() - v0 if val_rows != 0 else 0.0
                total_val_seconds += val_duration

                selection_metrics = train_metrics if selection_scope == "train" else val_metrics
                selection_key = selection_metric
                if selection_key not in selection_metrics:
                    selection_key = {"macro_f1": "f1_macro", "f1_macro": "macro_f1"}.get(
                        selection_metric, selection_metric,
                    )
                if selection_key not in selection_metrics:
                    name = f"{selection_scope}:{selection_metric}"
                    raise ValueError(f"early stopping metric {name} is unsupported")
                target = selection_metrics[selection_key]
                prev_best_epoch = self.strategy.best_epoch
                should_stop = target is not None and self.strategy.step(
                    epoch, float(target), state=model.get_state,
                )
                if self.strategy.best_epoch != prev_best_epoch:
                    best_train_metrics = dict(train_metrics)
                    best_val_metrics = dict(val_metrics)

                rss_mb = current_peak_rss_mb()
                epoch_metrics: dict[str, float | int | None] = {
                    "epoch": int(epoch),
                    "early_stopped": 1 if should_stop else 0,
                    "best_epoch": int(self.strategy.best_epoch),
                    "patience_consumed": int(self.strategy.patience_counter),
                    "training_duration_seconds": float(train_duration),
                    "validation_duration_seconds": float(val_duration),
                    "peak_rss_mb": float(rss_mb),
                    "best_validation_loss": best_val_metrics.get("log_loss"),
                    "best_score": self.strategy.best_score if self.strategy.best_epoch else None,
                    "train_samples": train_rows,
                    "validation_samples": val_rows,
                }
                for scope, metrics in (("train", train_metrics), ("validation", val_metrics)):
                    epoch_metrics.update({f"{scope}:{key}": value for key, value in metrics.items()})
                    if "macro_f1" in metrics and "f1_macro" not in metrics:
                        epoch_metrics[f"{scope}:f1_macro"] = metrics["macro_f1"]
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

                if should_stop or interrupted():
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
                        "best_validation_loss": best_val_metrics.get("log_loss"),
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
            best_loss=self.strategy.best_score if self.strategy.best_epoch else None,
            best_score=self.strategy.best_score if self.strategy.best_epoch else None,
            best_validation_loss=best_val_metrics.get("log_loss"),
            early_stopped=self.strategy.early_stopped,
            patience_consumed=self.strategy.patience_counter,
            restored_best=restored,
            stopped_by_signal=interrupted(),
            selection_scope=selection_scope,
            selection_metric=selection_metric,
            train_rows=train_rows,
            validation_rows=val_rows,
            training_presentations=epochs_run * train_rows if train_rows is not None else None,
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
                        "best_validation_loss": result.best_validation_loss,
                        "best_score": result.best_score,
                        "stopped_by_signal": result.stopped_by_signal,
                        "train_rows": train_rows,
                        "validation_rows": val_rows,
                        "training_presentations": result.training_presentations,
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
    timestamps: np.ndarray | None = None,
    capture_sigterm: bool = True,
    stop_requested: Callable[[], bool] | None = None,
) -> TrainingResult:
    """Convenience entry point to run the iterative training loop with early stopping."""
    loop = IterativeTrainingLoop(
        config=config,
        strategy=strategy,
        tracker=tracker,
        emit_stdout=emit_stdout,
        on_sigterm_checkpoint=on_sigterm_checkpoint,
        capture_sigterm=capture_sigterm,
        stop_requested=stop_requested,
    )
    return loop.run(model, train_data, val_data, holdout_start=holdout_start, timestamps=timestamps)


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
