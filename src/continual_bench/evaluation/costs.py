"""Normalized scenario accounting, independent of learner decisions."""

import math

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


def account(costs, *, errors, presentations, updates, train_seconds, predict_seconds,
            model_storage_bytes=None, include_aws=True):
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
    result = {"version": "normalized-scenario-v1", "kind": "scenario_estimate", "settings": costs,
            "metrics": components, "consumption": {
                "training_presentations": presentations, "update_events": updates,
                "train_seconds": train_seconds, "prediction_seconds": predict_seconds},
            "scope": ("learner training and predictions; excludes acquisition, publication "
                      "and evaluation references")}

    if include_aws and costs.get("aws") is not None:
        result["aws"] = aws_account(costs["aws"], train_seconds=train_seconds,
                                    predict_seconds=predict_seconds, model_storage_bytes=model_storage_bytes)
    return result


def aws_account(settings, *, train_seconds, predict_seconds, model_storage_bytes):
    """One provisioned task per run; learner timers proxy task duration, never an AWS invoice."""
    for value in (train_seconds, predict_seconds, model_storage_bytes):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("AWS consumption must be finite and nonnegative or unavailable")
    seconds = (train_seconds + predict_seconds
               if train_seconds is not None and predict_seconds is not None else None)
    billed = max(settings["minimum_compute_seconds"], math.ceil(seconds)) if seconds is not None else None
    cpu_hours = settings["vcpus"] * billed / 3600 if billed is not None else None
    memory_hours = settings["memory_gb"] * billed / 3600 if billed is not None else None
    gb = model_storage_bytes / 1024**3 if model_storage_bytes is not None else None
    gb_months = gb * settings["retention_days"] / settings["billing_month_days"] if gb is not None else None
    metrics = {
        "cost_cpu_usd": cpu_hours * settings["vcpu_hour_usd"] if cpu_hours is not None else None,
        "cost_ram_usd": memory_hours * settings["memory_gb_hour_usd"] if memory_hours is not None else None,
        "cost_model_storage_usd": (gb_months * settings["storage_gb_month_usd"]
                                   if gb_months is not None else None),
    }
    metrics["cost_total_usd"] = (sum(metrics.values())
                                 if all(v is not None for v in metrics.values()) else None)
    return {
        "version": "aws-resource-estimate-v1", "kind": "infrastructure_estimate", "unit": "USD",
        "settings": settings, "metrics": metrics,
        "consumption": {"processing_seconds": seconds, "billed_compute_seconds": billed,
                        "vcpu_hours": cpu_hours, "ram_gb_hours": memory_hours,
                        "model_storage_bytes": model_storage_bytes, "model_storage_gb": gb,
                        "model_storage_gb_months": gb_months},
        "scope": ("one provisioned Fargate task using learner training + prediction wall time; "
                  "all retained local model payloads projected to S3 for configured retention; "
                  "GB = 1024^3 bytes; month length is a scenario assumption"),
        "excludes": ["container startup", "idle time", "data acquisition", "evaluation references",
                     "publication", "tracking services", "checkpoints", "model metadata",
                     "S3 requests", "network transfer", "taxes", "free tier", "discounts"],
    }
