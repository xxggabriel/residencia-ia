"""Versioned observations shared by the batch, stream and temporal protocols."""

from continual_bench.models.observations import (
    batch_quality,
    binary_ranking,
    classification_count_quality,
    classification_quality,
    quality_scope,
    with_availability,
)

from .environment import peak_rss_bytes as process_peak_rss_bytes

__all__ = [
    "VERSION",
    "batch_quality",
    "binary_ranking",
    "classification_count_quality",
    "classification_quality",
    "observations",
    "process_peak_rss_bytes",
    "quality_scope",
]

VERSION = "benchmark-observations-v1"
EFFORT = (
    "examples_observed",
    "evaluated_examples",
    "unique_training_examples",
    "training_presentations",
    "replay_presentations",
    "steps",
    "epochs",
    "updates",
    "initial_unique_training_examples",
    "initial_training_presentations",
    "initial_steps",
    "initial_epochs",
    "update_training_presentations",
    "update_steps",
    "update_epochs",
    "promoted_versions",
)
RESOURCES = (
    "duration_seconds",
    "training_seconds",
    "inference_seconds",
    "predictions_per_second",
    "peak_rss_bytes",
    "rss_scope",
    "cost_total",
    "cost_per_1000_evaluated",
    "cost_currency",
    "cost_kind",
    "tracking_overhead_seconds",
)
RETENTION = (
    "matrix",
    "checkpoints",
    "periods",
    "absolute_retention",
    "gem_bwt",
    "post_learning_fm",
    "series",
)
DRIFT = (
    "alarm_count",
    "alarms",
    "changes",
    "precision",
    "recall",
    "f1",
    "delays",
    "delay_unit",
    "release_delays",
    "release_delay_mean",
    "release_delay_unit",
)
LINEAGE = (
    "dataset",
    "dataset_version",
    "model",
    "policy",
    "seed",
    "environment",
    "code_revision",
    "protocol",
)


def observations(
    task,
    scopes,
    *,
    windows=None,
    effort=None,
    resources=None,
    retention=None,
    drift=None,
    lineage=None,
):
    """Publish explicit nullability and reasons independently of legacy values."""
    drift = dict(drift or {})
    if "alarm_count" not in drift and drift.get("alarms") is not None:
        drift["alarm_count"] = len(drift["alarms"])
    return {
        "version": VERSION,
        "task": task,
        "scopes": {key: quality_scope(task, value) for key, value in scopes.items()},
        "windows": windows or [],
        "effort": with_availability(EFFORT, effort, "counter_not_observed_at_this_scope"),
        "resources": with_availability(RESOURCES, resources, "measurement_or_attribution_basis_unavailable"),
        "retention": with_availability(
            RETENTION,
            retention,
            "matched_reserved_historical_evaluations_unavailable",
        ),
        "drift": with_availability(
            DRIFT,
            drift,
            "detector_or_independent_annotations_and_followup_unavailable",
        ),
        "lineage": with_availability(LINEAGE, lineage, "lineage_not_recorded"),
    }
