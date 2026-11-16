"""Fit coordinates for evaluators that measure real, repeated training passes."""

import hashlib
import pickle
from pathlib import Path

from continual_bench.infrastructure.storage import atomic_write


def holdout_epoch_rows(epochs):
    """Keep frozen test scores distinct from training and validation measurements."""
    return [{
        "epoch": row["epoch"],
        "train_presentations": row["train_presentations"],
        "holdout_examples": row["test_examples"],
        "training_duration_seconds": row["training_seconds"],
        "prediction_duration_seconds": row["prediction_seconds"],
        **{f"holdout:{key}": value for key, value in row["metrics"].items()},
        **{f"cost:{key}": value for key, value in row["costs"]["metrics"].items()},
    } for row in epochs]


def epoch_fit(rows, *, number=1, action="fit", previous_steps=0, start_seq=None, end_seq=None,
              block_index=None, checkpoint=None, deployed_version=None, train_samples=None,
              validation_samples=None, train_seconds=None, summary=None):
    steps = []
    for epoch, metrics in enumerate(rows, 1):
        if metrics["epoch"] != epoch:
            raise ValueError("training epochs must start at 1 and increase by one within a fit")
        steps.append({
            "step": epoch, "global_step": previous_steps + epoch, "kind": "epoch", "metrics": metrics,
        })
    if not steps:
        raise ValueError("a completed training fit must contain an epoch")
    return {
        "fit_id": f"fit-{number:06d}",
        "block_index": block_index,
        "action": action,
        "status": "complete",
        "start_seq": start_seq,
        "end_seq": end_seq,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "training_presentations": (train_samples * len(steps) if train_samples is not None else None),
        "train_seconds": train_seconds,
        "steps": steps,
        "summary": {"epochs_trained": len(steps), **(summary or {})},
        "checkpoint": checkpoint,
        "deployed_version": deployed_version,
    }


def write_state(directory, file, state, *, format):
    atomic_write(Path(directory) / file, state)
    return {"file": file, "sha256": hashlib.sha256(state).hexdigest(), "bytes": len(state), "format": format}


def pickle_state(model, *, optional=False):
    try:
        return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, AttributeError, TypeError):
        if not optional:
            raise
        return None
