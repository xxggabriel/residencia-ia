"""Iterative training abstractions, callbacks, and loop orchestrator."""

from .callbacks import (
    EarlyStoppingStrategy,
    EpochMetricsTracker,
    GracefulTerminationHandler,
    IterativeTrainableModel,
    MinDeltaEarlyStopping,
)
from .loop import (
    IterativeTrainingLoop,
    TrainingResult,
    chronological_train_val_split,
    current_peak_rss_mb,
    emit_cloudwatch_metrics,
    enforce_single_thread_env,
    run_training_loop,
    train_iterative,
    unpack_split,
)

__all__ = [
    "EarlyStoppingStrategy",
    "EpochMetricsTracker",
    "GracefulTerminationHandler",
    "IterativeTrainableModel",
    "IterativeTrainingLoop",
    "MinDeltaEarlyStopping",
    "TrainingResult",
    "chronological_train_val_split",
    "current_peak_rss_mb",
    "emit_cloudwatch_metrics",
    "enforce_single_thread_env",
    "run_training_loop",
    "train_iterative",
    "unpack_split",
]
