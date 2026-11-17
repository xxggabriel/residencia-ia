"""Fit coordinates for evaluators that measure real, repeated training passes."""

import hashlib
import pickle
from pathlib import Path

from continual_bench.infrastructure.storage import atomic_write
from continual_bench.models.observations import training_step as training_step


def holdout_epoch_rows(epochs):
    """Keep frozen test scores distinct from training and validation measurements."""
    return [
        {
            "epoch": row["epoch"],
            "train_presentations": row["train_presentations"],
            "holdout_examples": row["test_examples"],
            "training_duration_seconds": row["training_seconds"],
            "prediction_duration_seconds": row["prediction_seconds"],
            **{f"holdout:{key}": value for key, value in row["metrics"].items()},
            **{f"cost:{key}": value for key, value in row["costs"]["metrics"].items()},
        }
        for row in epochs
    ]


def epoch_fit(
    rows,
    *,
    number=1,
    action="fit",
    previous_steps=0,
    start_seq=None,
    end_seq=None,
    block_index=None,
    checkpoint=None,
    deployed_version=None,
    train_samples=None,
    validation_samples=None,
    train_seconds=None,
    summary=None,
    observation_steps=None,
    observation_epochs=None,
    observed_step_kind=None,
    source_model_version=None,
    promotion_status=None,
    observation_effort=None,
    observation_resources=None,
    observation_scopes=None,
    data_scope=None,
):
    steps = []
    for epoch, metrics in enumerate(rows, 1):
        if metrics["epoch"] != epoch:
            raise ValueError("training epochs must start at 1 and increase by one within a fit")
        steps.append(
            {
                "step": epoch,
                "global_step": previous_steps + epoch,
                "kind": "epoch",
                "metrics": metrics,
            }
        )
    if not steps:
        raise ValueError("a completed training fit must contain an epoch")
    result = {
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
    result.update(
        {
            key: value
            for key, value in {
                "observation_steps": observation_steps,
                "observation_epochs": observation_epochs,
                "observed_step_kind": observed_step_kind,
                "source_model_version": source_model_version,
                "promotion_status": promotion_status,
                "observation_effort": observation_effort,
                "observation_resources": observation_resources,
                "observation_scopes": observation_scopes,
                "data_scope": data_scope,
            }.items()
            if value is not None
        }
    )
    return result


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
