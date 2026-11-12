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
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from continual_bench.config.iterative import IterativeTrainingConfig
from continual_bench.infrastructure.object_store import split_uri

from .session import DEFAULT_REGION as REGION
from .session import client as make_client

EMPTY_SHA256_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
MLFLOW_LOG_BATCH_MAX_METRICS = 500
MLFLOW_LOG_BATCH_MAX_PARAMS = 100
MLFLOW_LOG_BATCH_MAX_TAGS = 100
_VALID_MLFLOW_NAME = re.compile(r"[^/\w.\- ]")
_SKIP_SUFFIXES = (".lock",)
_SKIP_PREFIXES = (".pending-", ".predictions-", ".model-", ".prepare-")


def sanitize_mlflow_name(name: str) -> str:
    """Convert metric/param names into valid MLflow identifiers (`:` -> `/`)."""
    cleaned = str(name).replace(":", "/").strip()
    cleaned = _VALID_MLFLOW_NAME.sub("_", cleaned)
    return cleaned[:250] or "metric"


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
) -> dict[str, Any]:
    """Upload run files, plots/, and data/ directly to the MLflow run's S3 artifact store."""
    if not artifact_uri or not artifact_uri.startswith("s3://") or store is None:
        return {"files": 0, "bytes": 0, "artifact_uri": artifact_uri}
    bucket, base_prefix = split_uri(artifact_uri.rstrip("/"))
    if getattr(store, "bucket", bucket) != bucket:
        return {"files": 0, "bytes": 0, "artifact_uri": artifact_uri}

    files_uploaded = 0
    bytes_uploaded = 0
    seen_keys: set[str] = set()

    def _put(local_path: Path, rel_Target: str) -> None:
        nonlocal files_uploaded, bytes_uploaded
        key = f"{base_prefix}/{rel_Target.lstrip('/')}"
        if key in seen_keys:
            return
        seen_keys.add(key)
        store.put_file(key, local_path)
        files_uploaded += 1
        bytes_uploaded += local_path.stat().st_size

    # Mirror artifacts/plots -> plots/ and artifacts/data -> data/ at the root of the MLflow run
    artifacts_root = output_dir / "artifacts"
    for subfolder in ("plots", "data"):
        sub_dir = artifacts_root / subfolder
        if sub_dir.is_dir():
            for path in sorted(sub_dir.rglob("*")):
                if path.is_file() and not path.name.endswith(_SKIP_SUFFIXES):
                    rel = path.relative_to(sub_dir).as_posix()
                    _put(path, f"{subfolder}/{rel}")

    # Upload all standard run documents, reports, and model archive
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(_SKIP_SUFFIXES):
            continue
        relative = path.relative_to(output_dir)
        if any(part.startswith(_SKIP_PREFIXES) for part in relative.parts):
            continue
        _put(path, relative.as_posix())

    return {"files": files_uploaded, "bytes": bytes_uploaded, "artifact_uri": artifact_uri}


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
    return {
        sanitize_mlflow_name(k): str(v)[:500] for k, v in param_dict.items() if v is not None and v != ""
    }


def report_metric_items(report: dict[str, Any], now_ms: int) -> list[dict[str, Any]]:
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
        for k in ("train_samples", "train_seconds", "iterations", "bytes"):
            if k in version:
                _add_metric(f"training/{k}", version[k], v_step)
    for idx, pair in enumerate(report.get("cost_sensitivity") or []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            _add_metric("cost_sensitivity/total_cost", pair[1], idx)
    for step_idx, sample in enumerate((report.get("system_metrics") or {}).get("samples") or []):
        sample_ts = int(sample.get("timestamp_ms", now_ms))
        for k, v in (sample.get("values") or {}).items():
            _add_metric(f"system/{k}", v, step_idx, sample_ts)
    return metric_items


class SageMakerMlflowTracker:
    """Fail-safe SigV4 client for logging parameters, metrics, and artifacts to SageMaker MLflow."""

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
        job_name: str | None = None,
        project: str = "continual-bench",
        stack: str | None = None,
        git_sha: str | None = None,
        boto3_session: boto3.Session | None = None,
        sagemaker_client: Any | None = None,
        http_opener: Any | None = None,
        logger: logging.Logger | None = None,
        timeout_seconds: float = 10.0,
    ):
        self.project = project
        self.stack = stack or project
        self.dataset = dataset
        self.policy = policy
        self.scope = scope
        self.campaign_id = campaign_id or os.environ.get("BENCH_CAMPAIGN_ID", "")
        self.model_name = model_name
        self.run_id = run_id
        self.job_name = job_name or f"cb-{dataset}-{policy}-{run_id}"
        self.git_sha = git_sha or os.environ.get("BENCH_GIT_REVISION", "uncommitted")
        self._tracking_uri = (tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")).strip() or None
        self._session = boto3_session
        self._sagemaker_client = sagemaker_client
        self._http_opener = http_opener or urllib.request.urlopen
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
        return f"{self.dataset}-{self.policy}-{self.scope}-{short}-{short_id}"

    @property
    def mlflow_tags(self) -> list[dict[str, str]]:
        tags = [
            {"key": "cb-dataset", "value": str(self.dataset)},
            {"key": "cb-policy", "value": str(self.policy)},
            {"key": "cb-scope", "value": str(self.scope)},
            {"key": "cb-model", "value": str(self.model_name)},
            {"key": "cb-run-id", "value": str(self.run_id)},
            {"key": "cb-stack", "value": str(self.stack)},
            {"key": "cb-git-sha", "value": str(self.git_sha)},
            {"key": "zia_run_key", "value": str(self.run_id)},
            {"key": "mlflow.runName", "value": self.run_name},
            {"key": "mlflow.user", "value": self.project},
            {"key": "mlflow.source.name", "value": "continual-bench aws run-unit"},
            {"key": "mlflow.source.type", "value": "JOB"},
            {"key": "mlflow.source.git.commit", "value": str(self.git_sha)},
        ]
        if self.campaign_id:
            tags.append({"key": "cb-campaign-id", "value": str(self.campaign_id)})
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
        session = self._session or boto3.Session(region_name=region)
        creds = session.get_credentials()
        if creds is None:
            raise RuntimeError("No AWS credentials available for SigV4 signing")
        return creds.get_frozen_credentials()

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
        for endpoint in meta["endpoints"]:
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

        # Check if an active or existing run already exists for this run_id (idempotent retries)
        try:
            search_resp = self._request(
                "POST",
                "/api/2.0/mlflow/runs/search",
                payload={
                    "experiment_ids": [exp_id],
                    "filter": f"tags.`cb-run-id` = '{self.run_id}'",
                    "max_results": 1,
                },
            )
            runs = search_resp.get("runs") or []
            if runs:
                info = runs[0].get("info") or {}
                self.mlflow_run_id = info.get("run_id")
                self.artifact_uri = info.get("artifact_uri")
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
        except Exception:
            pass

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

        all_metrics = metrics or []
        all_params = params or []
        all_tags = tags or []
        success = True

        for offset in range(0, len(all_params), MLFLOW_LOG_BATCH_MAX_PARAMS):
            chunk = all_params[offset : offset + MLFLOW_LOG_BATCH_MAX_PARAMS]
            try:
                self._request(
                    "POST",
                    "/api/2.0/mlflow/runs/log-batch",
                    payload={"run_id": run_id, "params": chunk},
                )
            except Exception as err:
                success = False
                self._record_error("log_batch_params", err)
                break

        for offset in range(0, len(all_tags), MLFLOW_LOG_BATCH_MAX_TAGS):
            chunk = all_tags[offset : offset + MLFLOW_LOG_BATCH_MAX_TAGS]
            try:
                self._request(
                    "POST",
                    "/api/2.0/mlflow/runs/log-batch",
                    payload={"run_id": run_id, "tags": chunk},
                )
            except Exception as err:
                success = False
                self._record_error("log_batch_tags", err)
                break

        for offset in range(0, len(all_metrics), MLFLOW_LOG_BATCH_MAX_METRICS):
            chunk = all_metrics[offset : offset + MLFLOW_LOG_BATCH_MAX_METRICS]
            try:
                self._request(
                    "POST",
                    "/api/2.0/mlflow/runs/log-batch",
                    payload={"run_id": run_id, "metrics": chunk},
                )
            except Exception as err:
                success = False
                self._record_error("log_batch_metrics", err)
                break

        return success

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
            {"key": sanitize_mlflow_name(k), "value": str(v)[:500]}
            for k, v in raw.items()
            if v is not None
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
        return self.log_batch(metrics=mlflow_metrics)

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
        library = self._publish_with_library(
            report, output_dir=output_dir, outputs_uri=outputs_uri, model_package_arn=model_package_arn
        )
        if library is not None:
            return library
        if output_dir is not None:  # PNG dashboards ride along with the REST artifact upload
            self._save_figures(report, output_dir)
        run_id = self.ensure_run()
        if not run_id:
            return {"published": False, "errors": list(self.api_errors)}

        now_ms = int(time.time() * 1000)
        contract = report.get("contract") or {}
        inp = contract.get("input") or {}
        source = inp.get("source") or {}

        # 1. Parameters
        params_payload = [
            {"key": key, "value": value} for key, value in report_params(self, report).items()
        ]

        # 2. Tags and rich Markdown note
        summary_metrics = report.get("metrics") or {}
        holdout_metrics = report.get("holdout") or {}
        note_lines = [
            f"### Continual-Bench Run: `{self.dataset}` | `{self.policy}` | `{self.scope}`",
            f"- **Campaign ID:** `{self.campaign_id or 'n/a'}`",
            f"- **Run ID:** `{self.run_id}`",
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
            {"key": "benchmark_complete", "value": "true"},
            {"key": "zia_artifacts_status", "value": "complete"},
        ]
        if outputs_uri:
            tags_payload.append({"key": "cb-outputs-uri", "value": str(outputs_uri)[:500]})
        if model_package_arn:
            tags_payload.append({"key": "cb-model-package-arn", "value": str(model_package_arn)[:500]})

        self.log_batch(params=params_payload, tags=tags_payload)

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

        # 4. Collect all metric points (summary, holdout, windows, class_metrics, training versions, system)
        metric_items = report_metric_items(report, now_ms)

        self.log_batch(metrics=metric_items)

        # 5. Upload artifacts to MLflow's S3 artifact store URI
        uploaded = {"files": 0, "bytes": 0, "artifact_uri": self.artifact_uri}
        if output_dir is not None and store is not None and self.artifact_uri:
            try:
                uploaded = upload_directory_to_mlflow_artifacts(store, output_dir, self.artifact_uri)
            except Exception as err:
                self._record_error("upload_artifacts", err)

        # 6. Register and version models in MLflow Model Registry
        model_registry_info: dict[str, Any] = {}
        try:
            model_registry_info = self.register_run_models(
                report=report,
                model_package_arn=model_package_arn,
            )
        except Exception as err:
            self._record_error("register_run_models", err)

        self.finish("FINISHED")
        return {
            "published": True,
            "mlflow_experiment_id": self.experiment_id,
            "mlflow_run_id": self.mlflow_run_id,
            "artifact_uri": self.artifact_uri,
            "artifacts_uploaded": uploaded,
            **model_registry_info,
        }

    def _publish_with_library(self, report, *, output_dir, outputs_uri, model_package_arn):
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
                    output_dir=output_dir,
                    outputs_uri=outputs_uri,
                    model_package_arn=model_package_arn,
                )
            except LibraryPublishError as error:
                self._record_error("sagemaker_mlflow_library", error)
        except Exception as error:  # discovery itself failed: stay on the REST path
            self._record_error("sagemaker_mlflow_library", error)
        return None

    def _save_figures(self, report, output_dir) -> None:
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
            resp = self._request(
                "GET",
                "/api/2.0/mlflow/model-versions/search",
                params={"filter": f"name='{name}'", "max_results": "200"},
            )
            for mv in resp.get("model_versions") or []:
                if mv.get("run_id") != run_id:
                    continue
                mv_tags = {t["key"]: t["value"] for t in (mv.get("tags") or []) if "key" in t}
                ordinal = mv_tags.get("ordinal", "final")
                found[ordinal] = mv
        except Exception:
            pass
        return found

    def _set_model_alias(self, name: str, alias: str, version: str) -> None:
        clean_alias = re.sub(r"[^a-zA-Z0-9_\-]", "-", alias).strip("-")[:64]
        if not clean_alias:
            return
        try:
            self._request(
                "POST",
                "/api/2.0/mlflow/registered-models/alias",
                payload={"name": name, "alias": clean_alias, "version": str(version)},
            )
        except Exception:
            pass

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

        # 1. Log model on the Run so the Experiments table "Models" column shows the logged model
        try:
            model_json = {
                "artifact_path": "model",
                "flavors": {
                    "python_function": {
                        "loader_module": "continual_bench.models.zia",
                        "data": "model/model.tar.gz",
                        "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
                        "python_version": "3.12",
                    }
                },
                "run_id": run_id,
                "model_uuid": str(uuid.uuid4()),
                "utc_time_created": time.strftime("%Y-%m-%d %H:%M:%S.000000", time.gmtime()),
                "mlflow_version": "2.16.0",
            }
            self._request(
                "POST",
                "/api/2.0/mlflow/runs/log-model",
                payload={"run_id": run_id, "model_json": json.dumps(model_json)},
            )
        except Exception as err:
            self._record_error("log_model", err)

        base_uri = (self.artifact_uri or f"runs:/{run_id}").rstrip("/")
        common_rm_tags = [
            {"key": "dataset", "value": dataset},
            {"key": "scope", "value": scope},
            {"key": "model", "value": model_raw},
            {"key": "project", "value": self.project},
        ]

        # 2. Ensure per-policy Registered Model and register every continual-learning checkpoint version
        self._ensure_registered_model(
            policy_model_name,
            (
                f"Continual-Bench ZIA model versions for dataset={dataset}, "
                f"model={model_raw}, policy={policy} ({scope})"
            ),
            [*common_rm_tags, {"key": "policy", "value": policy}],
        )
        existing_policy_versions = self._existing_model_versions(policy_model_name, run_id)
        created_versions: list[str] = []

        for idx, ver in enumerate(raw_versions):
            ord_key = str(idx)
            is_final = idx == len(raw_versions) - 1
            if ord_key in existing_policy_versions:
                v_num = str(existing_policy_versions[ord_key].get("version", ""))
                if v_num:
                    created_versions.append(v_num)
                continue

            rel_file = str(ver.get("file") or "model/model.tar.gz").lstrip("/")
            source_uri = f"{base_uri}/{rel_file}"
            action = str(ver.get("action") or ("initial" if idx == 0 else "update"))
            start_seq = int(ver.get("start_seq", 0))
            end_seq = int(ver.get("end_seq", 0))
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
                    if idx == 0:
                        self._set_model_alias(policy_model_name, "initial", v_num)
                    self._set_model_alias(policy_model_name, f"step-{end_seq}", v_num)
                    if is_final:
                        self._set_model_alias(policy_model_name, "final", v_num)
                        self._set_model_alias(policy_model_name, "latest", v_num)
            except Exception as err:
                self._record_error("create_model_version", err)

        # 3. Also register the final model in the cross-policy family registry for side-by-side comparison
        family_version = ""
        self._ensure_registered_model(
            family_model_name,
            (
                f"Unified Continual-Bench ZIA model registry comparing policies "
                f"for dataset={dataset}, model={model_raw} ({scope})"
            ),
            common_rm_tags,
        )
        existing_family = self._existing_model_versions(family_model_name, run_id)
        if existing_family:
            first_existing = next(iter(existing_family.values()))
            family_version = str(first_existing.get("version", ""))
        else:
            final_ver = raw_versions[-1]
            rel_file = str(final_ver.get("file") or "model/model.tar.gz").lstrip("/")
            family_desc = (
                f"Policy={policy} ({len(raw_versions)} checkpoints, "
                f"final end_seq={final_ver.get('end_seq', 0)}): "
                f"preq_acc={summary_metrics.get('accuracy', 'n/a')}, "
                f"holdout_acc={holdout_metrics.get('accuracy', 'n/a')}"
            )
            family_tags = [
                {"key": "ordinal", "value": "final"},
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
                        "source": f"{base_uri}/{rel_file}",
                        "run_id": run_id,
                        "description": family_desc[:1000],
                        "tags": [t for t in family_tags if t["value"] != ""],
                    },
                )
                family_version = str((fam_resp.get("model_version") or {}).get("version", ""))
                if family_version:
                    self._set_model_alias(family_model_name, policy_slug, family_version)
                    self._set_model_alias(family_model_name, "latest", family_version)
            except Exception as err:
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
        self.log_batch(tags=run_tags)

        return {
            "registered_model": policy_model_name,
            "registered_model_version": final_version,
            "registered_model_versions": created_versions,
            "registered_family_model": family_model_name,
            "registered_family_version": family_version,
        }


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
