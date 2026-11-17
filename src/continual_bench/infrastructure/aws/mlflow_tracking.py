"""AWS SageMaker with MLflow (Serverless MLflow App & Tracking Server) SigV4 client and tracker."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from continual_bench.config.iterative import IterativeTrainingConfig
from continual_bench.infrastructure.metric_tracking import (
    DEFAULT_MAX_POINTS_PER_KEY,
    downsample_points,
    metric_points,
    missing_metric_points,
)
from continual_bench.infrastructure.object_store import ObjectExists, sha256_path, split_uri
from continual_bench.infrastructure.observation_tracking import (
    ARTIFACT_ROOT,
    observation_tags,
    parent_observation_points,
    write_observation_evidence,
)
from continual_bench.infrastructure.tracking import AsyncMetricWriter, RestBackend, batch_payloads
from continual_bench.infrastructure.tracking.artifacts import (
    DEFAULT_UPLOAD_WORKERS,
    SKIP_PREFIXES,
    SKIP_SUFFIXES,
    artifact_plan,
    upload_concurrently,
)
from continual_bench.infrastructure.tracking.http import shared_opener
from continual_bench.infrastructure.tracking.mlflow_async import (
    NativeAsyncLogger,
    async_logging_requested,
    native_logging_available,
)
from continual_bench.infrastructure.tracking.rounds import publish_training_rounds as publish_child_runs
from continual_bench.infrastructure.training_tracking import sanitize_mlflow_name

from .session import DEFAULT_REGION as REGION
from .session import client as make_client

EMPTY_SHA256_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
PROGRESS_INTERVAL_SECONDS = 30
#: Concurrent MLflow requests during publication (history reads, child runs).
PUBLISH_CONCURRENCY = int(os.environ.get("BENCH_MLFLOW_PUBLISH_CONCURRENCY", "4"))
# Backwards-compatible names; the layout itself lives in tracking.artifacts.
_SKIP_SUFFIXES = SKIP_SUFFIXES
_SKIP_PREFIXES = SKIP_PREFIXES


def parse_sagemaker_mlflow_arn(uri: str) -> dict[str, Any]:
    """Parse a SageMaker MLflow App or Tracking Server ARN into routing and SigV4 metadata."""
    raw = (uri or "").strip()
    if raw.startswith(("http://", "https://")):
        return {
            "kind": "http",
            "arn": None,
            "region": os.environ.get("AWS_DEFAULT_REGION", REGION),
            "service_name": None,
            "header_name": None,
            "endpoints": [raw.rstrip("/")],
            "resource_id": raw,
        }
    parts = raw.split(":")
    if len(parts) < 6 or parts[0] != "arn" or parts[2] != "sagemaker" or "/" not in parts[5]:
        raise ValueError(f"Unsupported SageMaker MLflow tracking URI: {uri!r}")
    partition = parts[1]
    region = parts[3] or os.environ.get("AWS_DEFAULT_REGION", REGION)
    account = parts[4]
    resource_type, resource_id = parts[5].split("/", 1)
    if partition not in {"aws", "aws-us-gov"}:
        raise ValueError(f"Unsupported AWS partition in MLflow ARN: {partition!r}")
    custom = os.environ.get("SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT", "").strip()
    if resource_type == "mlflow-app":
        endpoints = (
            [custom.rstrip("/")]
            if custom
            else [
                f"https://mlflow.sagemaker.{region}.app.aws",
                f"https://api.mlflow.sagemaker.{region}.app.aws",
            ]
        )
        return {
            "kind": "mlflow-app",
            "arn": raw,
            "partition": partition,
            "region": region,
            "account": account,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "service_name": "sagemaker",
            "header_name": "x-sm-mlflow-app-arn",
            "endpoints": endpoints,
        }
    if resource_type == "mlflow-tracking-server":
        endpoints = (
            [custom.rstrip("/")]
            if custom
            else [
                f"https://{region}.experiments.sagemaker.aws",
                f"https://api.{region}.experiments.sagemaker.aws",
            ]
        )
        return {
            "kind": "mlflow-tracking-server",
            "arn": raw,
            "partition": partition,
            "region": region,
            "account": account,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "service_name": "sagemaker-mlflow",
            "header_name": "x-mlflow-sm-tracking-server-arn",
            "endpoints": endpoints,
        }
    raise ValueError(f"Unsupported SageMaker MLflow resource type: {resource_type!r}")


def discover_mlflow_tracking_uri(
    project: str = "continual-bench",
    *,
    sagemaker_client: Any | None = None,
) -> str | None:
    """Resolve the active MLflow Tracking URI from environment or SageMaker API discovery."""
    for env_key in ("MLFLOW_TRACKING_URI", "BENCH_MLFLOW_ARN", "SAGEMAKER_MLFLOW_ARN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    sm = sagemaker_client or make_client("sagemaker")
    expected_name = f"{project}-mlflow"
    try:
        if hasattr(sm, "list_mlflow_apps"):
            apps = (sm.list_mlflow_apps() or {}).get("Summaries") or []
            for app in apps:
                if app.get("Name") == expected_name and app.get("Status") in {"Created", "Updated"}:
                    return app.get("Arn")
            for app in apps:
                if app.get("Status") in {"Created", "Updated"} and app.get("Arn"):
                    return app.get("Arn")
    except Exception:
        pass
    try:
        if hasattr(sm, "list_mlflow_tracking_servers"):
            servers = (sm.list_mlflow_tracking_servers() or {}).get("TrackingServerSummaries") or []
            for srv in servers:
                if srv.get("TrackingServerName") == expected_name and srv.get("IsActive") == "Active":
                    return srv.get("TrackingServerArn")
            for srv in servers:
                if srv.get("IsActive") == "Active" and srv.get("TrackingServerArn"):
                    return srv.get("TrackingServerArn")
    except Exception:
        pass
    return None


def presigned_mlflow_url(
    tracking_uri: str | None = None,
    *,
    project: str = "continual-bench",
    expires_in_seconds: int = 300,
    session_duration_seconds: int = 43200,
    sagemaker_client: Any | None = None,
) -> dict[str, str]:
    """Generate a presigned browser URL to open the SageMaker MLflow UI."""
    sm = sagemaker_client or make_client("sagemaker")
    uri = tracking_uri or discover_mlflow_tracking_uri(project, sagemaker_client=sm)
    if not uri:
        raise RuntimeError(
            f"No active SageMaker MLflow App or Tracking Server found for project '{project}'."
        )
    meta = parse_sagemaker_mlflow_arn(uri)
    if meta["kind"] == "mlflow-app":
        resp = sm.create_presigned_mlflow_app_url(
            Arn=meta["arn"],
            ExpiresInSeconds=expires_in_seconds,
            SessionExpirationDurationInSeconds=session_duration_seconds,
        )
        return {"arn": meta["arn"], "kind": "mlflow-app", "authorized_url": resp["AuthorizedUrl"]}
    if meta["kind"] == "mlflow-tracking-server":
        resp = sm.create_presigned_mlflow_tracking_server_url(
            TrackingServerName=meta["resource_id"],
            ExpiresInSeconds=expires_in_seconds,
            SessionExpirationDurationInSeconds=session_duration_seconds,
        )
        return {
            "arn": meta["arn"],
            "kind": "mlflow-tracking-server",
            "authorized_url": resp["AuthorizedUrl"],
        }
    return {"arn": uri, "kind": "http", "authorized_url": uri}


def upload_directory_to_mlflow_artifacts(
    store: Any,
    output_dir: Path,
    artifact_uri: str | None,
    *,
    immutable: bool = False,
    exclude: tuple[str, ...] = (),
    workers: int = DEFAULT_UPLOAD_WORKERS,
) -> dict[str, Any]:
    """Upload the run directory once, in parallel, to the MLflow run's S3 artifact store.

    The layout is ``tracking.artifacts.artifact_plan`` (``artifacts/plots`` ->
    ``plots/``, ``artifacts/data`` -> ``data/``), shared with the SDK publisher.
    """
    if not artifact_uri or not artifact_uri.startswith("s3://") or store is None:
        return {"files": 0, "bytes": 0, "artifact_uri": artifact_uri}
    bucket, base_prefix = split_uri(artifact_uri.rstrip("/"))
    if getattr(store, "bucket", bucket) != bucket:
        return {"files": 0, "bytes": 0, "artifact_uri": artifact_uri}

    def _put(local_path: Path, target: str) -> int:
        key = f"{base_prefix}/{target.lstrip('/')}"
        size = local_path.stat().st_size
        if immutable:
            digest = sha256_path(local_path)
            try:
                store.put_file(key, local_path, if_absent=True, sha256=digest)
            except ObjectExists:
                existing = store.head(key)
                if existing is None or existing.sha256 != digest or existing.size != size:
                    raise ValueError(f"MLflow artifact differs from archived evidence: {key}") from None
        else:
            store.put_file(key, local_path)
        return size

    sizes = upload_concurrently(_put, artifact_plan(output_dir, exclude=exclude), workers=workers)
    return {"files": len(sizes), "bytes": sum(sizes), "artifact_uri": artifact_uri}


def report_params(tracker: "SageMakerMlflowTracker", report: dict[str, Any]) -> dict[str, str]:
    """Sanitized run parameters shared by the REST and the `sagemaker-mlflow` publishers."""
    contract = report.get("contract") or {}
    inp = contract.get("input") or {}
    source = inp.get("source") or {}
    env = contract.get("environment") or {}
    param_dict = {
        "dataset": report.get("dataset", tracker.dataset),
        "scope": report.get("scope", tracker.scope),
        "policy": report.get("policy", tracker.policy),
        "model": report.get("model", tracker.model_name),
        "protocol": report.get("protocol", ""),
        "dataset_source": f"ziadatalabs/{source.get('repo', '')}" if source.get("repo") else "",
        "dataset_revision": source.get("revision", ""),
        "dataset_sha256": inp.get("sha256", ""),
        "samples": inp.get("rows", ""),
        "holdout_start": contract.get("holdout_start", ""),
        "block_rows": contract.get("block_rows", ""),
        "code_sha256": env.get("code_sha256", ""),
        "git_revision": env.get("git_revision", tracker.git_sha),
    }
    for mp_key, mp_val in (report.get("model_params") or {}).items():
        param_dict[f"model_{mp_key}"] = mp_val
    return {sanitize_mlflow_name(k): str(v)[:500] for k, v in param_dict.items() if v is not None and v != ""}


def report_metric_items(
    report: dict[str, Any],
    now_ms: int,
    adapted=None,
    *,
    include_observations=True,
    max_points_per_key: int | None = DEFAULT_MAX_POINTS_PER_KEY,
    max_system_points: int | None = 200,
) -> list[dict[str, Any]]:
    """Finite metric points (key, value, timestamp, step) shared by both publishers."""
    metric_items: list[dict[str, Any]] = []

    def _add_metric(key: str, val: Any, step: int = 0, ts: int = now_ms) -> None:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return
        fval = float(val)
        if not math.isfinite(fval):
            return
        metric_items.append(
            {"key": sanitize_mlflow_name(key), "value": fval, "timestamp": int(ts), "step": max(0, int(step))}
        )

    summary_metrics = report.get("metrics") or {}
    holdout_metrics = report.get("holdout") or {}
    for k, v in summary_metrics.items():
        _add_metric(f"summary/{k}", v, 0)
    for k, v in holdout_metrics.items():
        _add_metric(f"holdout/{k}", v, 0)
    for window in report.get("windows") or []:
        w_scope = window.get("scope", "prequential")
        end_seq = int(window.get("end_seq", 0))
        for k, v in window.items():
            if k not in {"scope", "end_seq", "start_time_ms", "end_time_ms"}:
                _add_metric(f"{w_scope}/windows/{k}", v, end_seq)
    for group in ("class_metrics", "holdout_class_metrics"):
        for label, row in (report.get(group) or {}).items():
            if isinstance(row, dict):
                for k, v in row.items():
                    _add_metric(f"{group}/{label}/{k}", v, 0)
    for version in report.get("model_versions") or []:
        v_step = int(version.get("end_seq", 0))
        for k in (
            "train_samples",
            "validation_samples",
            "training_presentations",
            "train_seconds",
            "epochs",
            "minibatches",
            "iterations",
            "bytes",
        ):
            if k in version:
                _add_metric(f"training/{k}", version[k], v_step)
    for idx, pair in enumerate(report.get("cost_sensitivity") or []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            _add_metric("cost_sensitivity/total_cost", pair[1], idx)
    raw_system: list[tuple[str, int, float]] = []
    system_stamps: dict[tuple[str, int], int] = {}
    for step_idx, sample in enumerate((report.get("system_metrics") or {}).get("samples") or []):
        sample_ts = int(sample.get("timestamp_ms", now_ms))
        for k, v in (sample.get("values") or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                continue
            key = f"system/{k}"
            raw_system.append((key, step_idx, float(v)))
            system_stamps[key, step_idx] = sample_ts
    for key, step, value in downsample_points(raw_system, max_points_per_key=max_system_points):
        _add_metric(key, value, step, system_stamps[key, step])
    if adapted and adapted.get("tracking"):
        for key, step, value in downsample_points(
            metric_points(adapted["tracking"]), max_points_per_key=max_points_per_key
        ):
            _add_metric(key, value, step)
    if include_observations:
        for key, step, value in downsample_points(
            parent_observation_points(report), max_points_per_key=max_points_per_key
        ):
            _add_metric(key, value, step)
    return metric_items


class SageMakerMlflowTracker:
    """Fail-safe SigV4 client for logging parameters, metrics, and artifacts to SageMaker MLflow."""

    run_key_tag = "zia_run_key"

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        dataset: str,
        policy: str,
        model_name: str,
        run_id: str,
        scope: str = "smoke",
        campaign_id: str | None = None,
        attempt_key: str | None = None,
        job_name: str | None = None,
        project: str = "continual-bench",
        stack: str | None = None,
        git_sha: str | None = None,
        boto3_session: boto3.Session | None = None,
        sagemaker_client: Any | None = None,
        http_opener: Any | None = None,
        logger: logging.Logger | None = None,
        timeout_seconds: float = 10.0,
        async_progress: bool = False,
    ):
        self.project = project
        self.stack = stack or project
        self.dataset = dataset
        self.policy = policy
        self.scope = scope
        self.campaign_id = campaign_id or os.environ.get("BENCH_CAMPAIGN_ID", "")
        self.model_name = model_name
        self.run_id = run_id
        self.attempt_key = attempt_key
        self.job_name = job_name or f"cb-{dataset}-{policy}-{run_id}"
        self.git_sha = git_sha or os.environ.get("BENCH_GIT_REVISION", "uncommitted")
        self._tracking_uri = (tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")).strip() or None
        self._session = boto3_session
        self._sagemaker_client = sagemaker_client
        # Keep-alive pool: urlopen paid a TCP+TLS handshake on every MLflow call.
        self._http_opener = http_opener or shared_opener()
        self._credentials = None
        self.concurrency = max(1, PUBLISH_CONCURRENCY)
        self.async_progress = async_progress
        self._live_writer: AsyncMetricWriter | None = None
        self._native: NativeAsyncLogger | None = None
        self._native_checked = False
        self.timeout_seconds = timeout_seconds
        self.logger = logger or logging.getLogger("continual_bench.aws.mlflow")

        self.experiment_id: str | None = None
        self.mlflow_run_id: str | None = None
        self.artifact_uri: str | None = None
        self.global_epoch_step: int = 0
        self.training_round: int = 0
        self.recorded_epochs: list[dict[str, Any]] = []
        self.api_errors: list[str] = []
        self._meta: dict[str, Any] | None = None
        self._progress_step = -1
        self._progress_attempted_at: float | None = None
        self._pending_progress: dict[str, dict[str, Any]] = {}
        self._spot_attempt_tags: dict[str, str] | None = None
        self._recorded_progress: dict[str, set[int]] = {}
        self._known_remote_metrics: set[str] | None = None
        self._rest_endpoint: str | None = None

    @property
    def tracking_uri(self) -> str | None:
        if self._tracking_uri is None:
            self._tracking_uri = discover_mlflow_tracking_uri(
                self.project, sagemaker_client=self._sagemaker_client
            )
        return self._tracking_uri

    @property
    def short_sha(self) -> str:
        if getattr(self, "_cached_short_sha", None) is not None:
            return self._cached_short_sha
        sha = (self.git_sha or os.environ.get("BENCH_GIT_REVISION") or "").strip()
        if sha and sha.lower() != "uncommitted":
            cleaned = re.sub(r"[^0-9a-zA-Z]", "", sha)
            if cleaned:
                self._cached_short_sha = cleaned[:7]
                return self._cached_short_sha
        try:
            import subprocess

            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
            if out:
                cleaned = re.sub(r"[^0-9a-zA-Z]", "", out)[:7]
                if cleaned:
                    self._cached_short_sha = cleaned
                    if self.git_sha == "uncommitted":
                        self.git_sha = cleaned
                    return cleaned
        except Exception:
            pass
        self._cached_short_sha = "uncommitted"
        return "uncommitted"

    @property
    def experiment_name(self) -> str:
        short = self.short_sha
        prefix = f"{short}-" if short else ""
        raw = f"{prefix}{self.project}-zia-experiments" if self.project else f"{prefix}zia-experiments"
        return sanitize_mlflow_name(raw)

    @property
    def run_name(self) -> str:
        short_id = self.run_id[:8] if self.run_id else "run"
        short = self.short_sha
        base = f"{self.dataset}-{self.policy}-{self.scope}-{short}-{short_id}"
        return f"{base}-{self.attempt_key}" if self.attempt_key and not self.spot else base

    @property
    def logical_run_id(self) -> str:
        return self.run_id

    @property
    def spot(self) -> bool:
        return self._spot_attempt_tags is not None

    @property
    def run_key(self) -> str:
        if self.spot:
            return f"spot-{self.logical_run_id}"
        return f"{self.run_id}-{self.attempt_key}" if self.attempt_key else self.run_id

    def bind_attempt(self, attempt_key: str) -> None:
        if self.mlflow_run_id is not None and self.attempt_key != attempt_key:
            raise ValueError("MLflow tracker already belongs to another attempt")
        self.attempt_key = attempt_key

    def bind_spot_attempt(
        self,
        attempt_key: str,
        *,
        instance_id: str,
        attempt: int | str,
        resumed_checkpoint: str | None = None,
        recovery_reason: str | None = None,
        previous_attempt_key: str | None = None,
        log_group: str | None = None,
        log_stream: str | None = None,
    ) -> None:
        """Bind attempt metadata before resolving the shared MLflow run."""
        if self.mlflow_run_id is not None:
            raise ValueError("Bind the Spot attempt before opening the MLflow run")
        self.bind_attempt(attempt_key)
        prefix = f"cb-spot-attempt-{attempt_key}"
        self._spot_attempt_tags = {
            "cb-compute": "batch-fargate-spot",
            "cb-spot-attempt": attempt_key,
            f"{prefix}-number": str(attempt),
            f"{prefix}-instance-id": instance_id,
        }
        if log_group:
            self._spot_attempt_tags[f"{prefix}-log-group"] = log_group
        if log_stream:
            self._spot_attempt_tags[f"{prefix}-log-stream"] = log_stream
        if resumed_checkpoint:
            self._spot_attempt_tags[f"{prefix}-resumed-checkpoint"] = resumed_checkpoint
        if recovery_reason:
            self._spot_attempt_tags[f"{prefix}-recovery-reason"] = recovery_reason
            if previous_attempt_key:
                previous = f"cb-spot-attempt-{previous_attempt_key}"
                self._spot_attempt_tags[f"{previous}-recovery-cause"] = recovery_reason
                previous_id, _, previous_number = previous_attempt_key.rpartition("-")
                if previous_id.startswith("i-") and previous_number.isdigit():
                    self._spot_attempt_tags[f"{previous}-instance-id"] = previous_id
                    self._spot_attempt_tags[f"{previous}-number"] = previous_number

    def record_spot_attempt(
        self,
        state: str,
        reason: str | None = None,
        checkpoint: str | None = None,
    ) -> bool:
        """Persist attempt-specific evidence without ending the logical run."""
        if self._spot_attempt_tags is None:
            return False
        prefix = f"cb-spot-attempt-{self.attempt_key}"
        tags = {**self._spot_attempt_tags, f"{prefix}-state": state}
        if reason:
            tags[f"{prefix}-reason"] = reason
        if checkpoint:
            tags[f"{prefix}-last-checkpoint"] = checkpoint
        # Queued live metrics land before the attempt state that summarizes them.
        self.flush()
        return self.log_batch(tags=[{"key": key, "value": value[:4500]} for key, value in tags.items()])

    @property
    def native_logger(self) -> NativeAsyncLogger | None:
        """MLflow's own async client when MLFLOW_ENABLE_ASYNC_LOGGING is on and MLflow is installed."""
        if not self._native_checked:
            self._native_checked = True
            if async_logging_requested() and native_logging_available(self.tracking_uri):
                try:
                    self._native = NativeAsyncLogger(self.tracking_uri)
                except Exception as error:  # Fall back to the REST transport.
                    self._record_error("mlflow_async_logging", error)
        return self._native

    def log_live(self, metrics: list[dict[str, Any]]) -> bool:
        """Send live points without blocking training: MLflow async queue, else the REST batch."""
        native = self.native_logger
        if native is None:
            return self.log_batch(metrics=metrics)
        run_id = self.ensure_run()
        if not run_id:
            return False
        try:
            native.log_batch(run_id, metrics=metrics)
            return True
        except Exception as error:
            self._record_error("mlflow_async_log_batch", error)
            return self.log_batch(metrics=metrics)

    def flush(self) -> bool:
        """Drain every asynchronous write (MLflow queue and the REST live writer)."""
        ok = True
        if self._live_writer is not None:
            self._live_writer.flush()
        if self._native is not None:
            try:
                self._native.flush()
            except Exception as error:
                self._record_error("mlflow_async_flush", error)
                ok = False
        return ok

    @property
    def mlflow_tags(self) -> list[dict[str, str]]:
        tags = [
            {"key": "cb-dataset", "value": str(self.dataset)},
            {"key": "cb-policy", "value": str(self.policy)},
            {"key": "cb-scope", "value": str(self.scope)},
            {"key": "cb-model", "value": str(self.model_name)},
            {"key": "cb-run-id", "value": str(self.logical_run_id if self.spot else self.run_id)},
            {"key": "cb-stack", "value": str(self.stack)},
            {"key": "cb-git-sha", "value": str(self.git_sha)},
            {"key": self.run_key_tag, "value": self.run_key},
            {"key": "cb-tracking-kind", "value": "benchmark"},
            {"key": "mlflow.runName", "value": self.run_name},
            {"key": "mlflow.user", "value": self.project},
            {"key": "mlflow.source.name", "value": "continual-bench aws run-unit"},
            {"key": "mlflow.source.type", "value": "JOB"},
            {"key": "mlflow.source.git.commit", "value": str(self.git_sha)},
        ]
        if self.campaign_id:
            tags.append({"key": "cb-campaign-id", "value": str(self.campaign_id)})
        if self.attempt_key and not self.spot:
            tags.append({"key": "cb-attempt-key", "value": self.attempt_key})
        if self._spot_attempt_tags:
            tags.extend({"key": k, "value": v[:4500]} for k, v in self._spot_attempt_tags.items())
        return tags

    def _record_error(self, action: str, error: Exception) -> None:
        msg = f"{action} failed ({type(error).__name__}): {error}"
        self.api_errors.append(msg)
        self.logger.warning("SageMaker MLflow %s", msg)

    def _host_meta(self) -> dict[str, Any] | None:
        if self._meta is not None:
            return self._meta
        uri = self.tracking_uri
        if not uri:
            return None
        try:
            self._meta = parse_sagemaker_mlflow_arn(uri)
            return self._meta
        except Exception as error:
            self._record_error("parse_tracking_uri", error)
            return None

    def _session_credentials(self, region: str):
        # One session per tracker: resolving the credential chain (IMDS on EC2) per
        # request was a measurable share of every call. Refreshable credentials
        # still rotate through get_frozen_credentials().
        if self._credentials is None:
            if self._session is None:
                self._session = boto3.Session(region_name=region)
            self._credentials = self._session.get_credentials()
        if self._credentials is None:
            raise RuntimeError("No AWS credentials available for SigV4 signing")
        return self._credentials.get_frozen_credentials()

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        meta = self._host_meta()
        if meta is None:
            raise RuntimeError("MLflow tracking URI is not configured")

        query = ""
        if params:
            encoded_query = urllib.parse.urlencode(params)
            if method.upper() in {"GET", "DELETE"}:
                encoded_query = encoded_query.replace("+", "%20")
            query = f"?{encoded_query}"

        body_bytes = (
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if payload is not None
            else b""
        )
        body_hash = sha256(body_bytes).hexdigest() if body_bytes else EMPTY_SHA256_HASH

        last_error: Exception | None = None
        endpoints = [self._rest_endpoint] if self.spot and self._rest_endpoint else meta["endpoints"]
        for endpoint in endpoints:
            url = f"{endpoint}{path}{query}"
            headers: dict[str, str] = {
                "Accept": "application/json",
                "X-Amz-Content-SHA256": body_hash,
            }
            if payload is not None:
                headers["Content-Type"] = "application/json"
            if meta.get("header_name") and meta.get("arn"):
                headers[meta["header_name"]] = meta["arn"]

            if meta.get("service_name"):
                creds = self._session_credentials(meta["region"])
                aws_req = AWSRequest(
                    method=method.upper(),
                    url=url,
                    data=body_bytes if body_bytes else None,
                    headers=headers,
                )
                SigV4Auth(creds, meta["service_name"], meta["region"]).add_auth(aws_req)
                prepared = aws_req.prepare()
                req_headers = dict(prepared.headers)
                req_url = prepared.url
            else:
                req_headers = headers
                req_url = url

            http_req = urllib.request.Request(
                req_url,
                data=body_bytes if body_bytes else None,
                headers=req_headers,
                method=method.upper(),
            )
            try:
                with self._http_opener(http_req, timeout=self.timeout_seconds) as resp:
                    raw_resp = resp.read()
                    if self.spot:
                        self._rest_endpoint = endpoint
                    return json.loads(raw_resp.decode("utf-8")) if raw_resp else {}
            except urllib.error.HTTPError as http_err:
                raw_body = ""
                try:
                    raw_body = http_err.read().decode("utf-8", errors="ignore")
                except Exception:
                    pass
                parsed_err = {}
                try:
                    parsed_err = json.loads(raw_body) if raw_body else {}
                except Exception:
                    pass
                error_code = parsed_err.get("error_code", f"HTTP_{http_err.code}")
                message = parsed_err.get("message", raw_body[:300] or str(http_err))
                err = MlflowRestError(http_err.code, error_code, message)
                if self.spot and parsed_err.get("error_code"):
                    self._rest_endpoint = endpoint
                raise err from http_err
            except urllib.error.URLError as url_err:
                last_error = url_err
                continue
            except OSError as os_err:
                last_error = os_err
                continue

        raise last_error or RuntimeError("All SageMaker MLflow endpoints failed")

    def ensure_experiment(self) -> str | None:
        if self.experiment_id is not None:
            return self.experiment_id
        if self._host_meta() is None:
            return None
        try:
            resp = self._request(
                "GET",
                "/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": self.experiment_name},
            )
            exp = resp.get("experiment") or {}
            exp_id = exp.get("experiment_id")
            if exp_id:
                if exp.get("lifecycle_stage") == "deleted":
                    self._request(
                        "POST",
                        "/api/2.0/mlflow/experiments/restore",
                        payload={"experiment_id": str(exp_id)},
                    )
                self.experiment_id = str(exp_id)
                return self.experiment_id
        except MlflowRestError as err:
            if err.error_code != "RESOURCE_DOES_NOT_EXIST" and err.status_code != 404:
                self._record_error("get_experiment_by_name", err)
                return None
        except Exception as err:
            self._record_error("get_experiment_by_name", err)
            return None

        try:
            resp = self._request(
                "POST",
                "/api/2.0/mlflow/experiments/create",
                payload={
                    "name": self.experiment_name,
                    "tags": [
                        {"key": "cb-stack", "value": str(self.stack)},
                        {"key": "cb-project", "value": str(self.project)},
                    ],
                },
            )
            self.experiment_id = str(resp["experiment_id"])
            return self.experiment_id
        except MlflowRestError as err:
            if err.error_code in {"RESOURCE_ALREADY_EXISTS", "INVALID_PARAMETER_VALUE"}:
                try:
                    resp = self._request(
                        "GET",
                        "/api/2.0/mlflow/experiments/get-by-name",
                        params={"experiment_name": self.experiment_name},
                    )
                    exp = resp.get("experiment") or {}
                    if exp.get("experiment_id"):
                        self.experiment_id = str(exp["experiment_id"])
                        return self.experiment_id
                except Exception as inner:
                    self._record_error("get_experiment_by_name", inner)
                    return None
            self._record_error("create_experiment", err)
            return None
        except Exception as err:
            self._record_error("create_experiment", err)
            return None

    def ensure_run(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig | None = None,
    ) -> str | None:
        if self.mlflow_run_id is not None:
            if hyperparameters is not None:
                self.log_hyperparameters(hyperparameters)
            return self.mlflow_run_id

        exp_id = self.ensure_experiment()
        if exp_id is None:
            return None

        # An EC2 worker retains the logical key across attempts. Batch retains the attempt key.
        try:
            search_resp = self._request(
                "POST",
                "/api/2.0/mlflow/runs/search",
                payload={
                    "experiment_ids": [exp_id],
                    "filter": f"tags.{self.run_key_tag} = '{self.run_key}'",
                    "max_results": 2,
                },
            )
            runs = search_resp.get("runs") or []
            if len(runs) > 1:
                raise ValueError(f"duplicate MLflow parent runs for {self.run_key}")
            if runs:
                info = runs[0].get("info") or {}
                self.mlflow_run_id = info.get("run_id")
                self.artifact_uri = info.get("artifact_uri")
                metrics = (runs[0].get("data") or {}).get("metrics")
                if self.spot and metrics is not None:
                    self._known_remote_metrics = {point["key"] for point in metrics}
                if info.get("status") in {"FAILED", "KILLED", "FINISHED"}:
                    try:
                        self._request(
                            "POST",
                            "/api/2.0/mlflow/runs/update",
                            payload={"run_id": self.mlflow_run_id, "status": "RUNNING"},
                        )
                    except Exception:
                        pass
                if hyperparameters is not None:
                    self.log_hyperparameters(hyperparameters)
                return self.mlflow_run_id
        except Exception as err:
            self._record_error("search_run", err)
            return None

        try:
            create_resp = self._request(
                "POST",
                "/api/2.0/mlflow/runs/create",
                payload={
                    "experiment_id": exp_id,
                    "run_name": self.run_name,
                    "start_time": int(time.time() * 1000),
                    "tags": self.mlflow_tags,
                },
            )
            run_info = (create_resp.get("run") or {}).get("info") or {}
            self.mlflow_run_id = run_info.get("run_id")
            self.artifact_uri = run_info.get("artifact_uri")
            if self.spot:
                self._known_remote_metrics = set()
            if hyperparameters is not None:
                self.log_hyperparameters(hyperparameters)
            return self.mlflow_run_id
        except Exception as err:
            self._record_error("create_run", err)
            return None

    def start(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig | None = None,
    ) -> dict[str, str | None]:
        exp_id = self.ensure_experiment()
        run_id = self.ensure_run(hyperparameters) if exp_id is not None else None
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": exp_id,
            "run_name": self.run_name,
            "run_id": run_id,
            "artifact_uri": self.artifact_uri,
        }

    def initialize(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig | None = None,
    ) -> bool:
        info = self.start(hyperparameters)
        return bool(info.get("run_id"))

    def start_live(self) -> bool:
        """Expose the Zia attempt before training or dataset materialization."""
        if self.spot and not self.record_spot_attempt("RUNNING"):
            return False
        return self.log_batch(
            tags=[
                {"key": "benchmark_complete", "value": "false"},
                {"key": "zia_artifacts_status", "value": "running"},
            ]
        )

    def progress(self, step: int, metrics: dict[str, float]) -> None:
        """Coalesce each live series between requests and keep final metric keys separate."""
        if step <= self._progress_step:
            return
        stamp = int(time.time() * 1000)
        points = [
            {
                "key": sanitize_mlflow_name(f"progress/{key}"),
                "value": float(value),
                "step": step,
                "timestamp": stamp,
            }
            for key, value in metrics.items()
            if type(value) in (int, float) and math.isfinite(value)
        ]
        if not points:
            return
        self._progress_step = step
        if self.native_logger is not None:
            # MLflow buffers and sends from its own thread pool; no extra coalescing needed.
            self.log_progress_metrics(points)
            return
        if self.async_progress and not self.spot:
            # Training never waits on the network; a daemon thread sends coalesced snapshots.
            if self._live_writer is None:
                self._live_writer = AsyncMetricWriter(self.log_progress_metrics, PROGRESS_INTERVAL_SECONDS)
            self._live_writer.submit(points)
            return
        self._pending_progress.update((point["key"], point) for point in points)
        if (
            self._progress_attempted_at is None
            or time.monotonic() - self._progress_attempted_at >= PROGRESS_INTERVAL_SECONDS
        ):
            self._flush_progress()

    def _flush_progress(self) -> None:
        self.flush()
        if self._pending_progress:
            points = list(self._pending_progress.values())
            self._pending_progress.clear()
            self.log_progress_metrics(points)
            self._progress_attempted_at = time.monotonic()

    def log_progress_metrics(self, points) -> bool:
        if not self.spot:
            return self.log_live(points)
        run_id = self.ensure_run()
        if run_id is None:
            return False
        written = {}
        try:
            for point in points:
                key, step = point["key"], point["step"]
                if key not in self._recorded_progress:
                    history = ()
                    if self._known_remote_metrics is None or key in self._known_remote_metrics:
                        history = self.get_metric_history(run_id, key)
                    self._recorded_progress[key] = {row.step for row in history}
                if step not in self._recorded_progress[key]:
                    written.setdefault((key, step), point)
        except Exception as err:
            self._record_error("resolve_live_progress", err)
            return False
        if not written:
            return True
        if self._known_remote_metrics is not None:
            self._known_remote_metrics.update(key for key, _ in written)
        # A resumed attempt skips steps its predecessor already stored, then queues the rest.
        if not self.log_live(list(written.values())):
            # A failed response can have persisted some points. Reconcile those series on the next flush.
            for key, _ in written:
                self._recorded_progress.pop(key, None)
            return False
        for key, step in written:
            self._recorded_progress[key].add(step)
        return True

    def log_batch(
        self,
        *,
        metrics: list[dict[str, Any]] | None = None,
        params: list[dict[str, str]] | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> bool:
        run_id = self.ensure_run()
        if not run_id:
            return False

        # Params, tags and metrics share requests up to MLflow's limits.
        chunks = batch_payloads(metrics or (), params or (), tags or ())
        for chunk_metrics, chunk_params, chunk_tags in chunks:
            payload: dict[str, Any] = {"run_id": run_id}
            if chunk_params:
                payload["params"] = chunk_params
            if chunk_tags:
                payload["tags"] = chunk_tags
            if chunk_metrics:
                payload["metrics"] = chunk_metrics
            try:
                self._request("POST", "/api/2.0/mlflow/runs/log-batch", payload=payload)
            except Exception as err:
                self._record_error("log_batch", err)
                return False
        return True

    def log_hyperparameters(
        self,
        hyperparameters: dict[str, Any] | IterativeTrainingConfig,
    ) -> bool:
        if isinstance(hyperparameters, IterativeTrainingConfig):
            raw = hyperparameters.to_hyperparameters_dict()
        elif isinstance(hyperparameters, dict):
            raw = dict(hyperparameters)
        else:
            return False
        params = [
            {"key": sanitize_mlflow_name(k), "value": str(v)[:500]} for k, v in raw.items() if v is not None
        ]
        return self.log_batch(params=params) if params else True

    def log_epoch_metrics(
        self,
        metrics: dict[str, float | int],
        *,
        step: int | None = None,
        timestamp: int | float | None = None,
    ) -> bool:
        self.global_epoch_step += 1
        mlflow_step = int(step if step is not None and self.training_round == 0 else self.global_epoch_step)
        now_ms = int((timestamp if timestamp is not None else time.time()) * 1000)
        record = {"epoch": mlflow_step, **metrics}
        self.recorded_epochs.append(record)

        mlflow_metrics = []
        for key, raw_val in metrics.items():
            if isinstance(raw_val, bool):
                val = float(int(raw_val))
            elif isinstance(raw_val, (int, float)) and math.isfinite(float(raw_val)):
                val = float(raw_val)
            else:
                continue
            mlflow_metrics.append(
                {
                    "key": sanitize_mlflow_name(key),
                    "value": val,
                    "timestamp": now_ms,
                    "step": mlflow_step,
                }
            )
        if not mlflow_metrics:
            return True
        return self.log_live(mlflow_metrics)

    log_epoch = log_epoch_metrics

    def finalize(self, summary: dict[str, Any] | None = None, **kwargs: Any) -> bool:
        """Record completion of a training round without closing the continual-learning run."""
        self.training_round += 1
        merged = {**(summary or {}), **kwargs}
        if not merged:
            return True
        now_ms = int(time.time() * 1000)
        round_metrics = []
        for key, raw_val in merged.items():
            if isinstance(raw_val, bool):
                val = float(int(raw_val))
            elif isinstance(raw_val, (int, float)) and math.isfinite(float(raw_val)):
                val = float(raw_val)
            else:
                continue
            round_metrics.append(
                {
                    "key": sanitize_mlflow_name(f"training_round/{key}"),
                    "value": val,
                    "timestamp": now_ms,
                    "step": self.training_round,
                }
            )
        return self.log_batch(metrics=round_metrics) if round_metrics else True

    def finish(self, status: str = "FINISHED") -> bool:
        if not self.mlflow_run_id:
            return False
        self.flush()
        if self._live_writer is not None:
            self._live_writer.close()
        mlflow_status = {
            "Completed": "FINISHED",
            "FINISHED": "FINISHED",
            "Failed": "FAILED",
            "FAILED": "FAILED",
            "Stopped": "KILLED",
            "KILLED": "KILLED",
        }.get(status, "FINISHED")
        try:
            self._request(
                "POST",
                "/api/2.0/mlflow/runs/update",
                payload={
                    "run_id": self.mlflow_run_id,
                    "status": mlflow_status,
                    "end_time": int(time.time() * 1000),
                },
            )
            return True
        except Exception as err:
            self._record_error("finish_run", err)
            return False

    def mark_failed(self, error_message: str | None = None) -> bool:
        if not self.mlflow_run_id:
            return False
        if error_message:
            self.log_batch(tags=[{"key": "cb-error", "value": str(error_message)[:1000]}])
        return self.finish("FAILED")

    def mark_killed(self) -> bool:
        if not self.finish("KILLED"):
            return False
        return self.log_batch(
            tags=[
                {"key": "benchmark_complete", "value": "false"},
                {"key": "zia_artifacts_status", "value": "interrupted"},
            ]
        )

    def get_metric_history(self, run_id, key):
        params = {"run_id": run_id, "metric_key": key, "max_results": 10_000}
        while True:
            result = self._request("GET", "/api/2.0/mlflow/metrics/get-history", params=params)
            for point in result.get("metrics") or ():
                yield SimpleNamespace(step=int(point.get("step", 0)), value=float(point["value"]))
            if not (token := result.get("next_page_token")):
                break
            params["page_token"] = token

    def _missing_completed_metrics(self, run_id, metrics, known_keys):
        desired = [(point["key"], point["step"], point["value"]) for point in metrics]
        missing = missing_metric_points(self, run_id, desired, known_keys=known_keys)
        missing_coordinates = {(key, step) for key, step, _ in missing}
        result = []
        for point in metrics:
            identity = point["key"], point["step"]
            if identity in missing_coordinates:
                result.append(point)
                missing_coordinates.remove(identity)
        return result

    def log_completed_metrics(self, metrics):
        """Preflight snapshots so retries also work on servers with an append-only metric store."""
        run_id = self.ensure_run()
        if run_id is None:
            return False
        run = self._request("GET", "/api/2.0/mlflow/runs/get", params={"run_id": run_id}).get("run") or {}
        known_keys = {metric["key"] for metric in run.get("data", {}).get("metrics") or ()}
        return self.log_batch(metrics=self._missing_completed_metrics(run_id, metrics, known_keys))

    def publish_training_rounds(self, report) -> list[dict]:
        """Use fit-local steps inside one child run per round of this parent run."""
        return publish_child_runs(
            RestBackend(self._request, concurrency=self.concurrency),
            self.experiment_id,
            self.mlflow_run_id,
            report,
            checkpoint_base_uri=self.artifact_uri if self.spot else None,
            include_artifact_uri=True,
        )

    def stage_observation_artifacts(self, directory, report, children, store):
        root = write_observation_evidence(directory, report, self.mlflow_run_id, children)
        if root is None:
            return
        for child in children:
            fit_root = root / "fits" / child["fit_id"]
            if not fit_root.is_dir():
                continue
            target = child.get("artifact_uri")
            if not target:
                raise RuntimeError("MLflow child did not provide an artifact URI")
            uploaded = upload_directory_to_mlflow_artifacts(
                store,
                fit_root,
                f"{target.rstrip('/')}/{ARTIFACT_ROOT}",
                immutable=self.spot,
            )
            if not uploaded["files"]:
                raise RuntimeError("MLflow did not publish fit observation artifacts")

    def publish_run_report(
        self,
        report: dict[str, Any],
        *,
        adapted: dict[str, Any] | None = None,
        output_dir: Path | None = None,
        store: Any | None = None,
        outputs_uri: str | None = None,
        model_package_arn: str | None = None,
    ) -> dict[str, Any]:
        """Publish full run parameters, window/holdout/system metrics, and S3 artifacts to MLflow.

        Uses the official `mlflow` + `sagemaker-mlflow` libraries when installed (opt-in
        `aws-mlflow` extra) and falls back to the REST/SigV4 client below otherwise or on failure.
        """
        self._flush_progress()
        library = (
            None
            if self.spot
            else self._publish_with_library(
                report,
                adapted=adapted,
                output_dir=output_dir,
                outputs_uri=outputs_uri,
                model_package_arn=model_package_arn,
            )
        )
        if library is not None:
            return library
        run_id = self.ensure_run()
        if not run_id:
            return {"published": False, "errors": list(self.api_errors)}

        contract = report.get("contract") or {}
        inp = contract.get("input") or {}
        source = inp.get("source") or {}

        # 1. Parameters
        params_payload = [{"key": key, "value": value} for key, value in report_params(self, report).items()]

        # 2. Tags and rich Markdown note
        summary_metrics = report.get("metrics") or {}
        holdout_metrics = report.get("holdout") or {}
        note_lines = [
            f"### Continual-Bench Run: `{self.dataset}` | `{self.policy}` | `{self.scope}`",
            f"- **Campaign ID:** `{self.campaign_id or 'n/a'}`",
            f"- **Run ID:** `{self.logical_run_id}`",
            f"- **Model:** `{report.get('model', self.model_name)}`",
            f"- **Holdout Accuracy:** `{holdout_metrics.get('accuracy', 'n/a')}`",
            f"- **Prequential Accuracy:** `{summary_metrics.get('accuracy', 'n/a')}`",
            f"- **Prequential F1 Macro:** `{summary_metrics.get('f1_macro', 'n/a')}`",
            (
                f"- **Updates / Retrains:** `{summary_metrics.get('update_count', 0)}` / "
                f"`{summary_metrics.get('retrain_count', 0)}`"
            ),
        ]
        if outputs_uri:
            note_lines.append(f"- **S3 Outputs URI:** `{outputs_uri}`")
        if model_package_arn:
            note_lines.append(f"- **SageMaker Model Package ARN:** `{model_package_arn}`")

        tags_payload = [
            {"key": "mlflow.note.content", "value": "\n".join(note_lines)[:4500]},
            {"key": "benchmark_complete", "value": "false"},
            {"key": "zia_artifacts_status", "value": "publishing"},
        ]
        if outputs_uri:
            tags_payload.append({"key": "cb-outputs-uri", "value": str(outputs_uri)[:500]})
        if model_package_arn:
            tags_payload.append({"key": "cb-model-package-arn", "value": str(model_package_arn)[:500]})

        tags_payload.extend({"key": k, "value": v} for k, v in observation_tags(report).items())
        if not self.log_batch(params=params_payload, tags=tags_payload):
            return {"published": False, "errors": list(self.api_errors)}

        # 3. Log Dataset Input Lineage
        if source.get("repo") and inp.get("sha256"):
            try:
                ds_name = f"ziadatalabs/{source['repo']}"
                ds_url = (
                    f"https://huggingface.co/datasets/{ds_name}"
                    f"/resolve/{source.get('revision', 'main')}/{source.get('file', '')}"
                )
                self._request(
                    "POST",
                    "/api/2.0/mlflow/runs/log-inputs",
                    payload={
                        "run_id": run_id,
                        "datasets": [
                            {
                                "tags": [
                                    {
                                        "key": "mlflow.data.context",
                                        "value": str(report.get("protocol", "training")),
                                    }
                                ],
                                "dataset": {
                                    "name": ds_name,
                                    "digest": str(inp["sha256"])[:32],
                                    "source_type": "http",
                                    "source": json.dumps({"url": ds_url}),
                                    "profile": json.dumps(
                                        {
                                            "num_rows": inp.get("rows"),
                                            "num_features": len(inp.get("features") or []),
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                )
            except Exception as err:
                self._record_error("log_inputs", err)
                return {"published": False, "errors": list(self.api_errors)}

        uploaded = {"files": 0, "bytes": 0, "artifact_uri": self.artifact_uri}
        try:
            # Deterministic timestamps keep replay of the same evidence idempotent.
            if not self.log_completed_metrics(report_metric_items(report, 0, adapted)):
                return {"published": False, "errors": list(self.api_errors)}
            child_runs = self.publish_training_rounds(report)
            if output_dir is not None:
                self.stage_observation_artifacts(output_dir, report, child_runs, store)
                uploaded = upload_directory_to_mlflow_artifacts(
                    store,
                    output_dir,
                    self.artifact_uri,
                    immutable=self.spot,
                    exclude=child_artifact_prefixes(child_runs),
                )
                if not uploaded["files"]:
                    raise ValueError("MLflow run has no uploaded artifacts")
        except Exception as err:
            self._record_error("publish_evidence", err)
            return {"published": False, "errors": list(self.api_errors)}

        # 6. Register and version models in MLflow Model Registry
        model_registry_info: dict[str, Any] = {}
        try:
            model_registry_info = self.register_run_models(
                report=report,
                model_package_arn=model_package_arn,
            )
        except Exception as err:
            self._record_error("register_run_models", err)
            if self.spot:
                return {"published": False, "errors": list(self.api_errors)}

        if not self.finish("FINISHED") or not self.log_batch(
            tags=[
                {"key": "benchmark_complete", "value": "true"},
                {"key": "zia_artifacts_status", "value": "complete"},
                {"key": "cb-training-rounds", "value": str(len(child_runs))},
            ]
        ):
            return {"published": False, "errors": list(self.api_errors)}
        return {
            "published": True,
            "mlflow_experiment_id": self.experiment_id,
            "mlflow_run_id": self.mlflow_run_id,
            "artifact_uri": self.artifact_uri,
            "artifacts_uploaded": uploaded,
            "child_runs": child_runs,
            **model_registry_info,
        }

    def _publish_with_library(self, report, *, output_dir, outputs_uri, model_package_arn, adapted=None):
        """Library publisher result, or None when unavailable/failed (REST path takes over)."""
        try:
            from .sagemaker_mlflow import (
                LibraryPublishError,
                is_sagemaker_arn,
                library_available,
                publish_with_library,
            )

            if not (is_sagemaker_arn(self.tracking_uri) and library_available()):
                return None
            try:
                return publish_with_library(
                    self,
                    report,
                    adapted=adapted,
                    output_dir=output_dir,
                    outputs_uri=outputs_uri,
                    model_package_arn=model_package_arn,
                )
            except LibraryPublishError as error:
                self._record_error("sagemaker_mlflow_library", error)
        except Exception as error:  # discovery itself failed: stay on the REST path
            self._record_error("sagemaker_mlflow_library", error)
        return None

    def prepare_artifacts(self, report, output_dir) -> None:
        try:
            from continual_bench.infrastructure.mlflow_figures import save_report_figures

            save_report_figures(report, Path(output_dir) / "artifacts")
        except Exception as error:
            self._record_error("save_figures", error)

    def _ensure_registered_model(
        self,
        name: str,
        description: str,
        tags: list[dict[str, str]],
    ) -> bool:
        try:
            self._request(
                "GET",
                "/api/2.0/mlflow/registered-models/get",
                params={"name": name},
            )
            return True
        except MlflowRestError as err:
            if err.error_code != "RESOURCE_DOES_NOT_EXIST" and err.status_code != 404:
                self._record_error("get_registered_model", err)
                return False
        except Exception as err:
            self._record_error("get_registered_model", err)
            return False

        try:
            self._request(
                "POST",
                "/api/2.0/mlflow/registered-models/create",
                payload={"name": name, "description": description[:1000], "tags": tags},
            )
            return True
        except MlflowRestError as err:
            if err.error_code == "RESOURCE_ALREADY_EXISTS":
                return True
            self._record_error("create_registered_model", err)
            return False
        except Exception as err:
            self._record_error("create_registered_model", err)
            return False

    def _existing_model_versions(self, name: str, run_id: str) -> dict[str, dict[str, Any]]:
        """Map existing MLflow model versions for (name, run_id) keyed by ordinal or source."""
        found: dict[str, dict[str, Any]] = {}
        try:
            params = {"filter": f"name='{name}'", "max_results": "200"}
            while True:
                resp = self._request("GET", "/api/2.0/mlflow/model-versions/search", params=params)
                for mv in resp.get("model_versions") or []:
                    if mv.get("run_id") == run_id:
                        mv_tags = {t["key"]: t["value"] for t in (mv.get("tags") or []) if "key" in t}
                        ordinal = mv_tags.get("ordinal", "final")
                        if self.spot and ordinal in found:
                            raise ValueError(f"duplicate MLflow model versions for {run_id}/{name}/{ordinal}")
                        found[ordinal] = mv
                page = resp.get("next_page_token")
                if not page:
                    break
                params["page_token"] = page
        except Exception:
            if self.spot:
                raise
        return found

    def _set_model_alias(self, name: str, alias: str, version: str, *, recovering: bool = False) -> None:
        clean_alias = re.sub(r"[^a-zA-Z0-9_\-]", "-", alias).strip("-")[:64]
        if not clean_alias:
            return
        if recovering:
            try:
                existing = (
                    self._request(
                        "GET",
                        "/api/2.0/mlflow/registered-models/alias",
                        params={"name": name, "alias": clean_alias},
                    ).get("model_version")
                    or {}
                )
            except MlflowRestError as err:
                if err.error_code not in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                    raise
                existing = {}
            if (current := existing.get("version")) and int(current) >= int(version):
                return
        try:
            self._request(
                "POST",
                "/api/2.0/mlflow/registered-models/alias",
                payload={"name": name, "alias": clean_alias, "version": str(version)},
            )
        except Exception:
            if self.spot:
                raise

    def _policy_aliases(self, name, version, *, idx, end_seq, is_final, recovering=False):
        if idx == 0:
            self._set_model_alias(name, "initial", version, recovering=recovering)
        self._set_model_alias(name, f"step-{end_seq}", version, recovering=recovering)
        if is_final:
            self._set_model_alias(name, "final", version, recovering=recovering)
            self._set_model_alias(name, "current", version, recovering=recovering)

    def register_run_models(
        self,
        *,
        report: dict[str, Any],
        model_package_arn: str | None = None,
    ) -> dict[str, Any]:
        """Log and version all continual-learning checkpoints and final candidate in MLflow Model Registry."""
        run_id = self.ensure_run()
        if not run_id:
            return {}

        dataset = str(report.get("dataset") or self.dataset)
        scope = str(report.get("scope") or self.scope)
        policy = str(report.get("policy") or self.policy)
        model_raw = str(report.get("model") or self.model_name)
        policy_slug = re.sub(r"[^a-zA-Z0-9]+", "-", policy).strip("-")
        dataset_slug = re.sub(r"[^a-zA-Z0-9]+", "-", dataset).strip("-")
        scope_suffix = "-smoke" if scope == "smoke" else ""

        policy_model_name = f"{self.project}-zia-{dataset_slug}-{policy_slug}{scope_suffix}"[:57].rstrip("-")
        family_model_name = f"{self.project}-zia-{dataset_slug}{scope_suffix}"[:57].rstrip("-")

        contract = report.get("contract") or {}
        inp = contract.get("input") or {}
        env = contract.get("environment") or {}
        summary_metrics = report.get("metrics") or {}
        holdout_metrics = report.get("holdout") or {}
        raw_versions = list(report.get("model_versions") or [])
        if not raw_versions:
            raw_versions = [
                {
                    "action": "final",
                    "start_seq": 0,
                    "end_seq": int(inp.get("rows") or 0),
                    "file": "model/model.tar.gz",
                    "sha256": "",
                    "train_samples": int(inp.get("rows") or 0),
                    "train_seconds": 0.0,
                    "iterations": 0,
                    "bytes": 0,
                }
            ]

        base_uri = (self.artifact_uri or f"runs:/{run_id}").rstrip("/")
        common_rm_tags = [
            {"key": "dataset", "value": dataset},
            {"key": "scope", "value": scope},
            {"key": "model", "value": model_raw},
            {"key": "project", "value": self.project},
        ]

        # 2. Ensure per-policy Registered Model and register every continual-learning checkpoint version
        ensured_policy = self._ensure_registered_model(
            policy_model_name,
            (
                f"Continual-Bench ZIA model versions for dataset={dataset}, "
                f"model={model_raw}, policy={policy} ({scope})"
            ),
            [*common_rm_tags, {"key": "policy", "value": policy}],
        )
        if self.spot and not ensured_policy:
            raise RuntimeError(f"could not verify MLflow registered model {policy_model_name}")
        existing_policy_versions = self._existing_model_versions(policy_model_name, run_id)
        created_versions: list[str] = []

        for idx, ver in enumerate(raw_versions):
            ord_key = str(idx)
            is_final = idx == len(raw_versions) - 1
            rel_file = "model/model.tar.gz" if is_final else str(ver["file"]).lstrip("/")
            source_uri = f"{base_uri}/{rel_file}"
            end_seq = int(ver.get("end_seq", 0))
            if ord_key in existing_policy_versions:
                existing_version = existing_policy_versions[ord_key]
                v_num = str(existing_version.get("version", ""))
                if self.spot:
                    if not v_num or existing_version.get("source") != source_uri:
                        raise ValueError(f"archive mismatch for MLflow model version {source_uri}")
                    self._policy_aliases(
                        policy_model_name,
                        v_num,
                        idx=idx,
                        end_seq=end_seq,
                        is_final=is_final,
                        recovering=True,
                    )
                if v_num:
                    created_versions.append(v_num)
                continue

            action = str(ver.get("action") or ("initial" if idx == 0 else "update"))
            start_seq = int(ver.get("start_seq", 0))
            train_samples = int(ver.get("train_samples", 0))
            sha_short = str(ver.get("sha256", ""))[:12]

            desc = (
                f"Checkpoint {idx} ({action}): seq=[{start_seq}..{end_seq}], "
                f"train_samples={train_samples}, sha256={sha_short or 'n/a'}"
            )
            if is_final:
                desc += (
                    f" |FINAL| preq_acc={summary_metrics.get('accuracy', 'n/a')}, "
                    f"holdout_acc={holdout_metrics.get('accuracy', 'n/a')}"
                )

            mv_tags = [
                {"key": "ordinal", "value": ord_key},
                {"key": "action", "value": action},
                {"key": "start_seq", "value": str(start_seq)},
                {"key": "end_seq", "value": str(end_seq)},
                {"key": "train_samples", "value": str(train_samples)},
                {"key": "train_seconds", "value": str(round(float(ver.get("train_seconds", 0.0)), 4))},
                {"key": "iterations", "value": str(ver.get("iterations", 0))},
                {"key": "bytes", "value": str(ver.get("bytes", 0))},
                {"key": "model_sha256", "value": str(ver.get("sha256", ""))},
                {"key": "dataset_version", "value": str(inp.get("sha256", ""))},
                {"key": "code_sha256", "value": str(env.get("code_sha256", ""))},
                {"key": "dataset", "value": dataset},
                {"key": "policy", "value": policy},
                {"key": "scope", "value": scope},
                {"key": "model", "value": model_raw},
                {"key": "is_final", "value": "true" if is_final else "false"},
                {
                    "key": "artifact_format",
                    "value": "sagemaker_tar_gz" if is_final else "zia_native_checkpoint",
                },
                {"key": "prequential_accuracy", "value": str(summary_metrics.get("accuracy", ""))},
                {"key": "holdout_accuracy", "value": str(holdout_metrics.get("accuracy", ""))},
            ]
            if is_final and model_package_arn:
                mv_tags.append({"key": "sagemaker_model_package_arn", "value": str(model_package_arn)})

            try:
                mv_resp = self._request(
                    "POST",
                    "/api/2.0/mlflow/model-versions/create",
                    payload={
                        "name": policy_model_name,
                        "source": source_uri,
                        "run_id": run_id,
                        "description": desc[:1000],
                        "tags": [t for t in mv_tags if t["value"] != ""],
                    },
                )
                v_num = str((mv_resp.get("model_version") or {}).get("version", ""))
                if v_num:
                    created_versions.append(v_num)
                    self._policy_aliases(
                        policy_model_name,
                        v_num,
                        idx=idx,
                        end_seq=end_seq,
                        is_final=is_final,
                    )
                elif self.spot:
                    raise RuntimeError("MLflow did not return a model version")
            except Exception as err:
                if self.spot:
                    raise
                self._record_error("create_model_version", err)

        # 3. Also register the final model in the cross-policy family registry for side-by-side comparison
        family_version = ""
        ensured_family = self._ensure_registered_model(
            family_model_name,
            (
                f"Unified Continual-Bench ZIA model registry comparing policies "
                f"for dataset={dataset}, model={model_raw} ({scope})"
            ),
            common_rm_tags,
        )
        if self.spot and not ensured_family:
            raise RuntimeError(f"could not verify MLflow registered model {family_model_name}")
        family_source = f"{base_uri}/model/model.tar.gz"
        existing_family = self._existing_model_versions(family_model_name, run_id)
        if existing_family:
            first_existing = next(iter(existing_family.values()))
            family_version = str(first_existing.get("version", ""))
            if self.spot:
                if not family_version or first_existing.get("source") != family_source:
                    raise ValueError(f"archive mismatch for MLflow model version {family_source}")
                self._set_model_alias(family_model_name, policy_slug, family_version, recovering=True)
                self._set_model_alias(family_model_name, "current", family_version, recovering=True)
        else:
            final_ver = raw_versions[-1]
            family_desc = (
                f"Policy={policy} ({len(raw_versions)} checkpoints, "
                f"final end_seq={final_ver.get('end_seq', 0)}): "
                f"preq_acc={summary_metrics.get('accuracy', 'n/a')}, "
                f"holdout_acc={holdout_metrics.get('accuracy', 'n/a')}"
            )
            family_tags = [
                {"key": "ordinal", "value": "final"},
                {"key": "artifact_format", "value": "sagemaker_tar_gz"},
                {"key": "policy", "value": policy},
                {"key": "dataset", "value": dataset},
                {"key": "scope", "value": scope},
                {"key": "model", "value": model_raw},
                {"key": "checkpoints_count", "value": str(len(raw_versions))},
                {"key": "model_sha256", "value": str(final_ver.get("sha256", ""))},
                {"key": "dataset_version", "value": str(inp.get("sha256", ""))},
                {"key": "prequential_accuracy", "value": str(summary_metrics.get("accuracy", ""))},
                {"key": "holdout_accuracy", "value": str(holdout_metrics.get("accuracy", ""))},
            ]
            if model_package_arn:
                family_tags.append({"key": "sagemaker_model_package_arn", "value": str(model_package_arn)})
            try:
                fam_resp = self._request(
                    "POST",
                    "/api/2.0/mlflow/model-versions/create",
                    payload={
                        "name": family_model_name,
                        "source": family_source,
                        "run_id": run_id,
                        "description": family_desc[:1000],
                        "tags": [t for t in family_tags if t["value"] != ""],
                    },
                )
                family_version = str((fam_resp.get("model_version") or {}).get("version", ""))
                if family_version:
                    self._set_model_alias(family_model_name, policy_slug, family_version)
                    self._set_model_alias(family_model_name, "current", family_version)
                elif self.spot:
                    raise RuntimeError("MLflow did not return a family model version")
            except Exception as err:
                if self.spot:
                    raise
                self._record_error("create_family_model_version", err)

        final_version = created_versions[-1] if created_versions else ""
        run_tags = [
            {"key": "registered_model", "value": policy_model_name},
            {"key": "registered_model_versions_count", "value": str(len(created_versions))},
            {"key": "registered_family_model", "value": family_model_name},
        ]
        if final_version:
            run_tags.append({"key": "registered_model_version", "value": final_version})
        if family_version:
            run_tags.append({"key": "registered_family_version", "value": family_version})
        if not self.log_batch(tags=run_tags) and self.spot:
            raise RuntimeError("MLflow did not persist registered model links")

        return {
            "registered_model": policy_model_name,
            "registered_model_version": final_version,
            "registered_model_versions": created_versions,
            "registered_family_model": family_model_name,
            "registered_family_version": family_version,
        }


def child_artifact_prefixes(child_runs) -> tuple[str, ...]:
    """Fit observations already published on their child runs (never re-sent to the parent)."""
    return tuple(
        f"{ARTIFACT_ROOT}/fits/{child['fit_id']}" for child in child_runs if child.get("artifact_uri")
    )


class MlflowRestError(RuntimeError):
    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(f"{error_code} (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


__all__ = [
    "MlflowRestError",
    "SageMakerMlflowTracker",
    "discover_mlflow_tracking_uri",
    "parse_sagemaker_mlflow_arn",
    "presigned_mlflow_url",
    "sanitize_mlflow_name",
    "upload_directory_to_mlflow_artifacts",
]
