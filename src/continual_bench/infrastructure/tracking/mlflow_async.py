"""Live metrics through MLflow's own asynchronous logging.

``mlflow.config.enable_async_logging`` (``MLFLOW_ENABLE_ASYNC_LOGGING``) makes the client
queue writes and send them from MLflow's thread pool (``MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE``),
batching for ``MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS``. Training calls return at once; a
Spot interruption or the final publication calls :meth:`NativeAsyncLogger.flush`, which
drains the queue and surfaces any failed write.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from .batching import batch_payloads

TRUE = {"1", "true", "yes"}


def async_logging_requested(env=None) -> bool:
    return str((env or os.environ).get("MLFLOW_ENABLE_ASYNC_LOGGING", "")).lower() in TRUE


def native_logging_available(tracking_uri: str | None) -> bool:
    """MLflow is installed and can reach ``tracking_uri`` (SageMaker ARNs need the plugin)."""
    if not tracking_uri:
        return False
    try:
        import mlflow  # noqa: F401

        if str(tracking_uri).startswith("arn:"):
            import sagemaker_mlflow  # noqa: F401
    except ImportError:
        return False
    return True


class NativeAsyncLogger:
    def __init__(self, tracking_uri: str, *, client: Any = None):
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.config.enable_async_logging(True)
        self.client = client or MlflowClient(tracking_uri=tracking_uri)
        self._operations: list = []
        self._lock = threading.Lock()

    def log_batch(self, run_id: str, metrics=(), params=(), tags=()) -> None:
        """Queue points ({key, value, step, timestamp}) and params/tags ({key, value})."""
        from mlflow.entities import Metric, Param, RunTag

        for chunk, chunk_params, chunk_tags in batch_payloads(metrics, params, tags):
            operation = self.client.log_batch(
                run_id,
                metrics=[
                    Metric(m["key"], float(m["value"]), int(m.get("timestamp", 0)), int(m.get("step", 0)))
                    for m in chunk
                ],
                params=[Param(p["key"], str(p["value"])) for p in chunk_params],
                tags=[RunTag(t["key"], str(t["value"])) for t in chunk_tags],
                synchronous=False,
            )
            if operation is not None:
                with self._lock:
                    self._operations.append(operation)

    def flush(self) -> None:
        """Block until every queued write is stored; raise if any of them failed."""
        store = getattr(getattr(self.client, "_tracking_client", None), "store", None)
        if store is not None and callable(getattr(store, "flush_async_logging", None)):
            store.flush_async_logging()
        with self._lock:
            operations, self._operations = self._operations, []
        for operation in operations:
            operation.wait()
