"""Adapters from the tracking port to the MLflow client and to the SigV4 REST API."""

import time
from typing import Any

from .batching import batch_payloads
from .ports import MetricPoint, RunRecord

REMOTE_SCHEMES = ("http://", "https://", "arn:", "databricks")
DEFAULT_CONCURRENCY = 4


def client_concurrency(client: Any) -> int:
    """Parallelize only remote stores; local file/SQLite stores stay serialized."""
    tracking = getattr(client, "_tracking_client", None)
    uri = str(getattr(tracking, "tracking_uri", "") or "")
    return DEFAULT_CONCURRENCY if uri.startswith(REMOTE_SCHEMES) else 1


def _point(point: MetricPoint | dict | tuple) -> MetricPoint:
    if isinstance(point, MetricPoint):
        return point
    if isinstance(point, dict):
        return MetricPoint(
            point["key"], float(point["value"]), int(point.get("step", 0)), int(point.get("timestamp", 0))
        )
    key, step, value = point
    return MetricPoint(key, float(value), int(step), 0)


class MlflowClientBackend:
    """Adapter over ``mlflow.MlflowClient`` (homelab server and the SageMaker plugin)."""

    def __init__(self, client: Any, *, concurrency: int | None = None):
        self.client = client
        self.concurrency = concurrency or client_concurrency(client)

    @staticmethod
    def _record(run: Any) -> RunRecord:
        return RunRecord(
            run.info.run_id,
            run.info.status,
            getattr(run.info, "artifact_uri", None),
            dict(run.data.tags),
            set(run.data.metrics),
        )

    def search_runs(self, experiment_id, filter_string):
        records, token = [], None
        while True:
            found = self.client.search_runs([experiment_id], filter_string, max_results=500, page_token=token)
            records.extend(self._record(run) for run in found)
            token = getattr(found, "token", None)
            if not token:
                return records

    def create_run(self, experiment_id, tags, run_name=None):
        return self._record(self.client.create_run(experiment_id, tags=tags))

    def get_run(self, run_id):
        return self._record(self.client.get_run(run_id))

    def get_metric_history(self, run_id, key):
        return [(int(row.step), float(row.value)) for row in self.client.get_metric_history(run_id, key)]

    def log_batch(self, run_id, metrics=(), params=None, tags=None):
        from mlflow.entities import Metric, Param, RunTag

        for chunk, chunk_params, chunk_tags in batch_payloads(
            [_point(p) for p in metrics],
            list((params or {}).items()),
            list((tags or {}).items()),
        ):
            entities = {
                "metrics": [Metric(p.key, p.value, p.timestamp, p.step) for p in chunk],
                "params": [Param(k, v) for k, v in chunk_params],
                "tags": [RunTag(k, v) for k, v in chunk_tags],
            }
            # Only non-empty groups: duck-typed clients may accept a subset of log_batch's arguments.
            self.client.log_batch(run_id, **{k: v for k, v in entities.items() if v}, synchronous=True)

    def set_terminated(self, run_id, status="FINISHED"):
        self.client.set_terminated(run_id, status)


class RestBackend:
    """Adapter over a ``request(method, path, *, payload=None, params=None)`` REST callable."""

    API = "/api/2.0/mlflow"

    def __init__(self, request, *, concurrency: int = DEFAULT_CONCURRENCY):
        self.request = request
        self.concurrency = concurrency

    @staticmethod
    def _record(run: dict) -> RunRecord:
        info, data = run.get("info") or {}, run.get("data") or {}
        return RunRecord(
            info.get("run_id"),
            info.get("status", "RUNNING"),
            info.get("artifact_uri"),
            {tag["key"]: tag["value"] for tag in data.get("tags") or ()},
            {metric["key"] for metric in data.get("metrics") or ()},
        )

    def search_runs(self, experiment_id, filter_string):
        payload = {"experiment_ids": [experiment_id], "filter": filter_string, "max_results": 200}
        records = []
        while True:
            found = self.request("POST", f"{self.API}/runs/search", payload=dict(payload))
            records.extend(self._record(run) for run in found.get("runs") or ())
            if not (token := found.get("next_page_token")):
                return records
            payload["page_token"] = token

    def create_run(self, experiment_id, tags, run_name=None):
        response = self.request(
            "POST",
            f"{self.API}/runs/create",
            payload={
                "experiment_id": experiment_id,
                "start_time": int(time.time() * 1000),
                "run_name": run_name or tags.get("mlflow.runName"),
                "tags": [{"key": key, "value": value} for key, value in tags.items()],
            },
        )
        return self._record(response["run"])

    def get_run(self, run_id):
        return self._record(self.request("GET", f"{self.API}/runs/get", params={"run_id": run_id})["run"])

    def get_metric_history(self, run_id, key):
        params = {"run_id": run_id, "metric_key": key, "max_results": 10_000}
        rows = []
        while True:
            result = self.request("GET", f"{self.API}/metrics/get-history", params=dict(params))
            rows.extend((int(p.get("step", 0)), float(p["value"])) for p in result.get("metrics") or ())
            if not (token := result.get("next_page_token")):
                return rows
            params["page_token"] = token

    def log_batch(self, run_id, metrics=(), params=None, tags=None):
        for chunk, chunk_params, chunk_tags in batch_payloads(
            [_point(p) for p in metrics],
            list((params or {}).items()),
            list((tags or {}).items()),
        ):
            payload: dict[str, Any] = {"run_id": run_id}
            if chunk:
                payload["metrics"] = [
                    {"key": p.key, "value": p.value, "timestamp": p.timestamp, "step": p.step} for p in chunk
                ]
            if chunk_params:
                payload["params"] = [{"key": k, "value": v} for k, v in chunk_params]
            if chunk_tags:
                payload["tags"] = [{"key": k, "value": v} for k, v in chunk_tags]
            self.request("POST", f"{self.API}/runs/log-batch", payload=payload)

    def set_terminated(self, run_id, status="FINISHED"):
        self.request(
            "POST",
            f"{self.API}/runs/update",
            payload={
                "run_id": run_id,
                "status": status,
                "end_time": int(time.time() * 1000),
            },
        )


def as_backend(client: Any):
    """Accept a backend or a raw ``MlflowClient`` (legacy call sites)."""
    if isinstance(client, (MlflowClientBackend, RestBackend)):
        return client
    return MlflowClientBackend(client)
