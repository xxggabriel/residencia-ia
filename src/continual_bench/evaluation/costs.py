"""Normalized scenario accounting, independent of learner decisions."""

from continual_bench.config import CostConfig
from continual_bench.core.tasks import task_kind


def profile(value=None):
    if value is None:
        value = CostConfig(train_sample=0.001, update=0.05)
    elif not isinstance(value, CostConfig):
        value = CostConfig.model_validate(value)
    return value.model_dump()


def error_units(task, target, prediction, costs):
    return task_kind(task).error_units(target, prediction, costs)


def account(costs, *, errors, presentations, updates, train_seconds, predict_seconds):
    """Timers may be absent; a required absent timer makes the total unavailable."""
    components = {
        "cost_prediction_errors": errors,
        "cost_training_presentations": presentations * costs["train_sample"],
        "cost_update_events": updates * costs["update"],
        "cost_training_time": (train_seconds * costs["train_second"]
                               if train_seconds is not None else (0 if not costs["train_second"] else None)),
        "cost_prediction_time": (predict_seconds * costs["predict_second"]
                                 if predict_seconds is not None
                                 else (0 if not costs["predict_second"] else None)),
    }
    components["cost_total"] = (sum(components.values())
                                if all(v is not None for v in components.values()) else None)
    return {"version": "normalized-scenario-v1", "kind": "scenario_estimate", "settings": costs,
            "metrics": components, "consumption": {
                "training_presentations": presentations, "update_events": updates,
                "train_seconds": train_seconds, "prediction_seconds": predict_seconds},
            "scope": ("learner training and predictions; excludes acquisition, publication "
                      "and evaluation references")}
