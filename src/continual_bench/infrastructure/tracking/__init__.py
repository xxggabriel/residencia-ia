"""Backend-agnostic MLflow tracking shared by the homelab and AWS publishers.

Ports & Adapters: publication algorithms (`rounds`, `reconcile`, `artifacts`)
depend only on the `TrackingBackend` port; `MlflowClientBackend` adapts the
official client (homelab, SageMaker SDK path) and `RestBackend` adapts the
SigV4 REST transport used by the AWS image without MLflow installed.
"""

from .async_writer import AsyncMetricWriter
from .backends import MlflowClientBackend, RestBackend, as_backend
from .batching import MAX_ENTITIES, MAX_METRICS, MAX_PARAMS, MAX_TAGS, batch_payloads
from .ports import MetricPoint, RunRecord, TrackingBackend
from .reconcile import missing_metric_points

__all__ = [
    "MAX_ENTITIES",
    "MAX_METRICS",
    "MAX_PARAMS",
    "MAX_TAGS",
    "AsyncMetricWriter",
    "MetricPoint",
    "MlflowClientBackend",
    "RestBackend",
    "RunRecord",
    "TrackingBackend",
    "as_backend",
    "batch_payloads",
    "missing_metric_points",
]
