"""Shared MLflow campaign publisher using TrackingBackend (MlflowClientBackend / RestBackend).

Publishes the 4-level hierarchy:
1. Experiment: ``<campaign_name>/<dataset>`` (with optional ``experiment_prefix``)
2. Campaign root run -> Policy nested runs (``mlflow.parentRunId = campaign_run_id``)
3. Policy run -> Seed nested runs (``mlflow.parentRunId = policy_run_id``)
4. Seed run -> Retrain version nested runs (``mlflow.parentRunId = seed_run_id``)
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Lock
from typing import Any

from continual_bench.backends.base import CHECKPOINT_EXTENSIONS, RuntimeSettings
from continual_bench.infrastructure.tracking import (
    MetricPoint,
    MlflowClientBackend,
    RestBackend,
)

logger = logging.getLogger(__name__)

TRACKING_URI_PREFIXES = ("http://", "https://", "mlflow", "arn:", "sqlite:", "file:")
_PLOT_EXTENSIONS = frozenset({".png", ".svg", ".jpg", ".jpeg", ".pdf"})
_INVALID_MLFLOW_CHARS = re.compile(r"[^a-zA-Z0-9_.\-/ ]")


def _sanitize_mlflow_name(name: str) -> str:
    return _INVALID_MLFLOW_CHARS.sub("_", str(name))[:250]


def build_experiment_name(
    campaign_name: str,
    dataset: str,
    *,
    experiment_prefix: str = "",
) -> str:
    """Return isolated MLflow experiment name for a single campaign + dataset."""
    clean_campaign = str(campaign_name or "benchmark").strip("/")
    clean_dataset = str(dataset or "unknown").strip("/")
    base = f"{clean_campaign}/{clean_dataset}"
    prefix = str(experiment_prefix or "").strip("/")
    return f"{prefix}/{base}" if prefix else base


def _finite_float(val: Any) -> float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    fval = float(val)
    return fval if math.isfinite(fval) else None


def _sanitize_params(raw: dict[str, Any]) -> dict[str, str]:
    params: dict[str, str] = {}
    for k, v in raw.items():
        if v is None or v == "":
            continue
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_v is not None and sub_v != "":
                    params[_sanitize_mlflow_name(f"{k}_{sub_k}")] = str(sub_v)[:500]
        elif isinstance(v, (list, tuple)):
            params[_sanitize_mlflow_name(k)] = ",".join(str(x) for x in v)[:500]
        else:
            params[_sanitize_mlflow_name(k)] = str(v)[:500]
    return params


def _resolve_tracking_adapter(
    uri: str,
    *,
    client: Any | None = None,
) -> tuple[Any, Any]:
    """Return ``(raw_client_or_none, TrackingBackend)`` for local, Docker, or SageMaker MLflow."""
    if client is not None:
        if isinstance(client, (MlflowClientBackend, RestBackend)):
            return None, client
        return client, MlflowClientBackend(client)

    if uri.startswith("arn:aws") and ":sagemaker:" in uri:
        try:
            import mlflow  # type: ignore[import-untyped]
            import sagemaker_mlflow  # type: ignore[import-untyped]  # noqa: F401
            from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

            # SageMaker authentication and presigned artifact URLs read the global URI.
            mlflow.set_tracking_uri(uri)
            ml_client = MlflowClient(tracking_uri=uri)
            return ml_client, MlflowClientBackend(ml_client)
        except ImportError:
            from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker

            sm = SageMakerMlflowTracker(
                tracking_uri=uri,
                dataset="campaign",
                policy="campaign",
                model_name="campaign",
                run_id="campaign",
            )
            return None, RestBackend(sm._request, concurrency=sm.concurrency)

    from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

    ml_client = MlflowClient(tracking_uri=uri)
    # Keep concurrency bounded (2) on local/Docker HTTP servers to avoid SQLite lock contention.
    return ml_client, MlflowClientBackend(ml_client, concurrency=2)


def _ensure_experiment_id(
    exp_name: str,
    raw_client: Any | None,
    backend: Any,
) -> str:
    if raw_client is not None:
        exp = raw_client.get_experiment_by_name(exp_name)
        if exp is not None:
            if getattr(exp, "lifecycle_stage", "active") == "deleted":
                raw_client.restore_experiment(exp.experiment_id)
            return str(exp.experiment_id)
        try:
            return str(raw_client.create_experiment(exp_name))
        except Exception:
            exp = raw_client.get_experiment_by_name(exp_name)
            if exp is None:
                raise
            if getattr(exp, "lifecycle_stage", "active") == "deleted":
                raw_client.restore_experiment(exp.experiment_id)
            return str(exp.experiment_id)

    if isinstance(backend, RestBackend):
        try:
            found = backend.request(
                "GET",
                f"{backend.API}/experiments/get-by-name",
                params={"experiment_name": exp_name},
            )
            exp_obj = found["experiment"]
            exp_id = str(exp_obj["experiment_id"])
            if exp_obj.get("lifecycle_stage") == "deleted":
                backend.request(
                    "POST",
                    f"{backend.API}/experiments/restore",
                    payload={"experiment_id": exp_id},
                )
            return exp_id
        except Exception:
            created = backend.request(
                "POST",
                f"{backend.API}/experiments/create",
                payload={"name": exp_name},
            )
            return str(created["experiment_id"])
    raise RuntimeError("Cannot resolve MLflow experiment without raw client or RestBackend")


def _upload_selected_artifacts(
    raw_client: Any | None,
    run_id: str,
    artifact_paths: dict[str, str],
    *,
    settings: RuntimeSettings,
    backend: Any | None = None,
) -> list[str]:
    store = None
    base_prefix = ""
    if isinstance(backend, RestBackend):
        from continual_bench.infrastructure.aws.s3 import S3ObjectStore
        from continual_bench.infrastructure.object_store import split_uri

        if artifact_paths:
            artifact_uri = backend.get_run(run_id).artifact_uri
            if not artifact_uri or not artifact_uri.startswith("s3://"):
                raise RuntimeError(f"Missing S3 artifact URI on MLflow run {run_id}")
            bucket, base_prefix = split_uri(artifact_uri)
            store = S3ObjectStore(bucket)
    elif raw_client is None or not hasattr(raw_client, "log_artifact"):
        return []
    uploaded: list[str] = []
    for name, path_str in sorted(artifact_paths.items()):
        path = Path(path_str)
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if not settings.generate_plots and suffix in _PLOT_EXTENSIONS:
            continue
        if not settings.save_model_checkpoints and suffix in CHECKPOINT_EXTENSIONS:
            continue
        if store is not None:
            store.put_file("/".join(filter(None, (base_prefix, path.name))), path)
        else:
            raw_client.log_artifact(run_id, str(path))
        uploaded.append(name)
    return uploaded


def _campaign_params(
    settings: RuntimeSettings, metadata: dict[str, Any], campaign_name: str, dataset: str
) -> dict[str, Any]:
    base_params_raw = {
        "campaign_name": campaign_name,
        "protocol": metadata.get("protocol", ""),
        "dataset": dataset,
        "model": metadata.get("model", ""),
        "seeds": metadata.get("seeds", []),
        "policies": metadata.get("policies", []),
        "staleness_mode": metadata.get("staleness_mode", ""),
        "kernel_gamma": metadata.get("kernel_gamma") or metadata.get("gamma"),
        "alpha_max": metadata.get("alpha_max"),
        "kappa_min": metadata.get("kappa_min"),
        "kappa_max": metadata.get("kappa_max"),
        "regimes": metadata.get("regimes"),
        "git_revision": metadata.get("git_revision", ""),
        "code_sha256": metadata.get("code_sha256", ""),
        "dataset_sha256": metadata.get("dataset_sha256", ""),
        "platform": settings.platform,
        "env_name": settings.env_name,
        "save_model_checkpoints": settings.save_model_checkpoints,
        "generate_plots": settings.generate_plots,
    }
    if isinstance(metadata.get("model_params"), dict):
        for mp_k, mp_v in metadata["model_params"].items():
            base_params_raw[f"model_{mp_k}"] = mp_v
    return base_params_raw


def _publish_seed_results(
    *,
    settings: RuntimeSettings,
    raw_client: Any,
    backend: Any,
    experiment_id: str,
    campaign_name: str,
    dataset: str,
    metadata: dict[str, Any],
    pol_name: str,
    p_run_id: str,
    seed_val: int,
    seed_summaries: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    retrain_events: list[dict[str, Any]],
    all_artifacts: dict[str, str],
    existing_run_id: str | None = None,
) -> tuple[str, list[str]]:
    now_ms = int(time.time() * 1000)
    base_params_raw = _campaign_params(settings, metadata, campaign_name, dataset)
    seed_run_name = f"seed-{seed_val}"
    s_tags = _seed_tags(p_run_id, metadata, campaign_name, dataset, pol_name, seed_val)
    s_run_id = existing_run_id or backend.create_run(
        experiment_id, s_tags, run_name=seed_run_name
    ).run_id
    s_params = _sanitize_params({**base_params_raw, "policy": pol_name, "seed": seed_val})
    s_metrics: list[MetricPoint] = []

    for srow in seed_summaries:
        reg = str(srow.get("regime", "price_grid") or "price_grid")
        for k, v in srow.items():
            if k in {"run_id", "seed", "policy", "regime"}:
                continue
            fval = _finite_float(v)
            if fval is not None:
                s_metrics.append(
                    MetricPoint(_sanitize_mlflow_name(f"summary/{k}"), fval, 0, now_ms)
                )
                s_metrics.append(
                    MetricPoint(_sanitize_mlflow_name(f"regimes/{reg}/{k}"), fval, 0, now_ms)
                )

    grid_idx = 0
    for rrow in regime_rows:
        reg = str(rrow.get("regime", "default") or "default")
        if reg == "price_grid":
            cost_val = _finite_float(rrow.get("strategy_cost") or rrow.get("cost"))
            if cost_val is not None:
                s_metrics.append(
                    MetricPoint("cost_sensitivity/total_cost", cost_val, grid_idx, now_ms)
                )
            for extra_key in ("kappa", "alpha", "strategy_scpe", "average_accuracy", "retrains"):
                extra_val = _finite_float(rrow.get(extra_key))
                if extra_val is not None:
                    s_metrics.append(
                        MetricPoint(f"cost_sensitivity/{extra_key}", extra_val, grid_idx, now_ms)
                    )
            grid_idx += 1
            continue
        for k, v in rrow.items():
            if k in {"run_id", "seed", "policy", "regime"}:
                continue
            fval = _finite_float(v)
            if fval is not None:
                s_metrics.append(
                    MetricPoint(_sanitize_mlflow_name(f"regimes/{reg}/{k}"), fval, 0, now_ms)
                )

    # Timeline metrics per batch
    for brow in batch_rows:
        b_step = max(0, int(brow.get("batch_index", 0)))
        for k in (
            "accuracy",
            "error",
            "staleness",
            "model_age",
            "deployed_model",
            "cumulative_cost",
            "fit_time_s",
        ):
            fval = _finite_float(brow.get(k))
            if fval is not None:
                metric_key = (
                    "training/fit_seconds"
                    if k == "fit_time_s"
                    else f"prequential/batches/{k}"
                )
                s_metrics.append(
                    MetricPoint(_sanitize_mlflow_name(metric_key), fval, b_step, now_ms)
                )
        if "retrain_decision" in brow:
            s_metrics.append(
                MetricPoint(
                    "prequential/batches/retrain_decision",
                    1.0 if brow["retrain_decision"] else 0.0,
                    b_step,
                    now_ms,
                )
            )

    backend.log_batch(s_run_id, metrics=s_metrics, params=s_params, tags=s_tags)
    s_artifacts = {
        k: v
        for k, v in all_artifacts.items()
        if k.startswith(f"policies/{pol_name}/seeds/seed-{seed_val}/")
        and "/retrains/" not in k
    }
    _upload_selected_artifacts(raw_client, s_run_id, s_artifacts, settings=settings, backend=backend)

    # 4. Retrain Version Runs (nested under Seed)
    ret_ids: list[str] = []
    for idx, ev in enumerate(retrain_events, start=1):
        batch_idx = int(ev.get("batch_index", 0))
        step_val = int(ev.get("step", batch_idx))
        v_idx = int(ev.get("version_index", idx))
        v_name = str(ev.get("version_name") or f"{pol_name}-s{seed_val}-v{v_idx}-b{batch_idx}")
        c_tags = {
            "mlflow.parentRunId": s_run_id,
            "mlflow.runName": v_name,
            "cb-campaign-name": campaign_name,
            "cb-protocol": str(metadata.get("protocol", "")),
            "cb-dataset": dataset,
            "cb-policy": pol_name,
            "cb-retrain-seed": str(seed_val),
            "cb-retrain-batch": str(batch_idx),
            "cb-model-version": f"v{v_idx}",
            "cb-version-name": v_name,
            "cb-hierarchy-level": "retrain",
        }
        c_run = backend.create_run(experiment_id, c_tags, run_name=v_name)
        c_params = _sanitize_params(
            {
                "campaign_name": campaign_name,
                "protocol": metadata.get("protocol", ""),
                "dataset": dataset,
                "policy": pol_name,
                "model": metadata.get("model", ""),
                "seed": seed_val,
                "batch_index": batch_idx,
                "step": step_val,
                "version_index": v_idx,
                "version_name": v_name,
                "regime": ev.get("regime", ""),
                "train_samples": ev.get("train_samples"),
                "window_size": ev.get("window_size"),
                "model_params": metadata.get("model_params") or {},
            }
        )
        c_metrics: list[MetricPoint] = []
        for m_key in (
            "fit_seconds",
            "train_seconds",
            "train_samples",
            "window_size",
            "retrain_cost",
            "post_retrain_error",
            "post_retrain_accuracy",
            "staleness",
            "kappa",
            "alpha",
        ):
            fval = _finite_float(ev.get(m_key))
            if fval is not None:
                c_metrics.append(
                    MetricPoint(
                        _sanitize_mlflow_name(f"training/{m_key}"),
                        fval,
                        max(0, step_val),
                        now_ms,
                    )
                )
        try:
            backend.log_batch(c_run.run_id, metrics=c_metrics, params=c_params, tags=c_tags)
            r_key = f"policies/{pol_name}/seeds/seed-{seed_val}/retrains/{v_name}.json"
            if r_key in all_artifacts:
                _upload_selected_artifacts(
                    raw_client, c_run.run_id, {r_key: all_artifacts[r_key]},
                    settings=settings, backend=backend,
                )
            backend.set_terminated(c_run.run_id, "FINISHED")
        except Exception:
            try:
                backend.set_terminated(c_run.run_id, "FAILED")
            except Exception:
                logger.warning("Could not mark MLflow run %s as failed", c_run.run_id, exc_info=True)
            raise
        ret_ids.append(c_run.run_id)

    backend.set_terminated(s_run_id, "FINISHED")
    return s_run_id, ret_ids



def _run_tags(
    metadata: dict[str, Any], campaign_name: str, dataset: str, name: str, level: str
) -> dict[str, str]:
    return {
        "mlflow.runName": name,
        "cb-campaign-name": campaign_name,
        "cb-protocol": str(metadata.get("protocol", "")),
        "cb-dataset": dataset,
        "cb-hierarchy-level": level,
    }


def _campaign_tags(settings, run_id, metadata, campaign_name, dataset):
    return {
        **_run_tags(metadata, campaign_name, dataset, run_id, "campaign"),
        "cb-run-id": run_id,
        "cb-platform": settings.platform,
        "cb-env": settings.env_name,
    }


def _policy_tags(parent, metadata, campaign_name, dataset, policy):
    return {
        **_run_tags(metadata, campaign_name, dataset, policy, "policy"),
        "mlflow.parentRunId": parent,
        "cb-policy": policy,
    }


def _seed_tags(parent, metadata, campaign_name, dataset, policy, seed):
    return {
        **_run_tags(metadata, campaign_name, dataset, f"seed-{seed}", "seed"),
        "mlflow.parentRunId": parent,
        "cb-policy": policy,
        "cb-seed": str(seed),
    }


class CampaignRuns:
    """MLflow IDs for one campaign, shared by the execution and publication stages."""

    def __init__(self, settings, run_id, metadata, *, resume=False, recover_evaluations=False):
        self.settings = settings
        self.metadata = metadata
        self.campaign_name = str(metadata.get("campaign_name") or run_id)
        self.dataset = str(metadata.get("dataset") or "unknown")
        self.raw_client, self.backend = _resolve_tracking_adapter(settings.tracking_uri)
        self.experiment_id = _ensure_experiment_id(
            build_experiment_name(
                self.campaign_name, self.dataset, experiment_prefix=settings.experiment_prefix
            ),
            self.raw_client,
            self.backend,
        )
        tags = _campaign_tags(settings, run_id, metadata, self.campaign_name, self.dataset)
        matches = (
            self.backend.search_runs(self.experiment_id, f"tags.`cb-run-id` = '{run_id}'")
            if resume else []
        )
        self.campaign_run_id = (
            matches[0].run_id if matches else self._create(tags)
        )
        self.policy_runs: dict[str, str] = {}
        self.seed_runs: dict[tuple[str, int], str] = {}
        self.completed_seeds: dict[tuple[str, int], tuple[str, list[str]]] = {}
        if matches and recover_evaluations:
            self._recover_evaluations()
        self._lock = Lock()
        self.publication_slots = BoundedSemaphore(max(1, min(self.backend.concurrency, 4)))

    def _children(self, run_id, level):
        return self.backend.search_runs(
            self.experiment_id,
            f"tags.`mlflow.parentRunId` = '{run_id}' AND tags.`cb-hierarchy-level` = '{level}'",
        )

    def _recover_evaluations(self):
        for policy_run in self._children(self.campaign_run_id, "policy"):
            policy = policy_run.tags["cb-policy"]
            self.policy_runs[policy] = policy_run.run_id
            for seed_run in self._children(policy_run.run_id, "seed"):
                key = (policy, int(seed_run.tags["cb-seed"]))
                if seed_run.status == "FINISHED":
                    self.seed_runs[key] = seed_run.run_id
                    self.completed_seeds[key] = (
                        seed_run.run_id,
                        [run.run_id for run in self._children(seed_run.run_id, "retrain")],
                    )
                elif seed_run.status == "RUNNING":
                    self.fail(seed_run.run_id)

    def _create(self, tags: dict[str, str]) -> str:
        return self.backend.create_run(
            self.experiment_id, tags, run_name=tags["mlflow.runName"]
        ).run_id

    def start_policy(self, policy: str) -> str:
        with self._lock:
            if policy not in self.policy_runs:
                self.policy_runs[policy] = self._create(
                    _policy_tags(
                        self.campaign_run_id, self.metadata, self.campaign_name, self.dataset, policy
                    )
                )
            return self.policy_runs[policy]

    def start_seed(self, policy: str, seed: int) -> str:
        parent = self.start_policy(policy)
        key = (policy, seed)
        with self._lock:
            if key not in self.seed_runs:
                self.seed_runs[key] = self._create(
                    _seed_tags(parent, self.metadata, self.campaign_name, self.dataset, policy, seed)
                )
            return self.seed_runs[key]

    def start_matrix(self, seed: int) -> str:
        return self._create({
            **_run_tags(self.metadata, self.campaign_name, self.dataset, f"matrix-seed-{seed}", "matrix"),
            "mlflow.parentRunId": self.campaign_run_id,
            "cb-seed": str(seed),
        })

    def finish_matrix(self, run_id, seed, elapsed, fit_times):
        now_ms = int(time.time() * 1000)
        batches = [
            {"batch_index": i, "fit_seconds": float(seconds)}
            for i, seconds in enumerate(fit_times)
        ]
        metrics = [
            MetricPoint("matrix/duration_seconds", elapsed, 0, now_ms),
            MetricPoint("matrix/batches", float(len(batches)), 0, now_ms),
            *(
                MetricPoint("training/fit_seconds", batch["fit_seconds"], i, now_ms)
                for i, batch in enumerate(batches)
            ),
        ]
        self.backend.log_batch(run_id, metrics=metrics, params={"seed": str(seed)})
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "matrix_summary.json"
            path.write_text(
                json.dumps({"seed": seed, "duration_seconds": elapsed, "batches": batches}, indent=2),
                encoding="utf-8",
            )
            _upload_selected_artifacts(
                self.raw_client, run_id, {path.name: str(path)},
                settings=self.settings, backend=self.backend,
            )
        self.backend.set_terminated(run_id, "FINISHED")

    def finish_seed(self, policy, seed, rows, artifact_paths):
        regime_rows, seed_summaries, retrain_events, batch_rows = rows
        with self.publication_slots:
            self.completed_seeds[policy, seed] = _publish_seed_results(
                settings=self.settings,
                raw_client=self.raw_client,
                backend=self.backend,
                experiment_id=self.experiment_id,
                campaign_name=self.campaign_name,
                dataset=self.dataset,
                metadata=self.metadata,
                pol_name=policy,
                p_run_id=self.policy_runs[policy],
                seed_val=seed,
                seed_summaries=seed_summaries,
                regime_rows=regime_rows,
                batch_rows=batch_rows,
                retrain_events=retrain_events,
                all_artifacts=artifact_paths,
                existing_run_id=self.seed_runs[policy, seed],
            )

    def fail(self, run_id: str) -> None:
        try:
            self.backend.set_terminated(run_id, "FAILED")
        except Exception:
            logger.warning("Could not mark MLflow run %s as failed", run_id, exc_info=True)

    def fail_campaign(self) -> None:
        for key, run_id in self.seed_runs.items():
            if key not in self.completed_seeds:
                self.fail(run_id)
        for run_id in self.policy_runs.values():
            self.fail(run_id)
        self.fail(self.campaign_run_id)


def publish_campaign_to_mlflow(
    *,
    settings: RuntimeSettings,
    run_id: str,
    metadata: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    seed_summaries: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    retrain_events: list[dict[str, Any]],
    batch_rows: Sequence[dict[str, Any]] = (),
    paper_comparison_rows: Sequence[dict[str, Any]] = (),
    system_metrics: Sequence[dict[str, Any]] = (),
    artifact_paths: dict[str, str] | None = None,
    client: Any | None = None,
    active_runs: CampaignRuns | None = None,
) -> dict[str, Any] | None:
    """Publish campaign -> policy -> seed -> retrain hierarchy to MLflow via TrackingBackend."""
    uri = (settings.tracking_uri or "").strip()
    if not uri or not uri.startswith(TRACKING_URI_PREFIXES):
        return None

    now_ms = int(time.time() * 1000)
    raw_client, backend = (
        (active_runs.raw_client, active_runs.backend)
        if active_runs else _resolve_tracking_adapter(uri, client=client)
    )

    campaign_name = str(metadata.get("campaign_name") or run_id)
    dataset = str(metadata.get("dataset") or "unknown")
    exp_name = build_experiment_name(
        campaign_name,
        dataset,
        experiment_prefix=settings.experiment_prefix,
    )
    experiment_id = (
        active_runs.experiment_id if active_runs else _ensure_experiment_id(exp_name, raw_client, backend)
    )

    base_params_raw = _campaign_params(settings, metadata, campaign_name, dataset)

    all_artifacts = dict(artifact_paths or {})
    top_level_artifacts = {
        k: v for k, v in all_artifacts.items() if not k.startswith("policies/")
    }

    # 1. Campaign Root Run
    campaign_params = _sanitize_params(base_params_raw)
    campaign_tags = _campaign_tags(settings, run_id, metadata, campaign_name, dataset)
    campaign_run_id = (
        active_runs.campaign_run_id if active_runs
        else backend.create_run(experiment_id, campaign_tags, run_name=run_id).run_id
    )

    campaign_metrics: list[MetricPoint] = []
    for row in aggregate_rows:
        pol = str(row.get("policy", "aggregate"))
        reg = str(row.get("regime", "default") or "default")
        metric_name = str(row.get("metric", "value"))
        for stat_key in ("mean", "std", "n_seeds"):
            fval = _finite_float(row.get(stat_key))
            if fval is not None:
                campaign_metrics.append(
                    MetricPoint(
                        _sanitize_mlflow_name(f"aggregate/{pol}/{reg}/{metric_name}_{stat_key}"),
                        fval,
                        0,
                        now_ms,
                    )
                )

    for prow in paper_comparison_rows:
        pol = str(prow.get("policy", ""))
        reg = str(prow.get("regime", "default") or "default")
        metric_name = str(prow.get("metric", ""))
        for field_key in (
            "observed_mean",
            "observed_std",
            "published_value",
            "published_spread",
            "abs_diff",
            "rel_diff_pct",
        ):
            fval = _finite_float(prow.get(field_key))
            if fval is not None:
                campaign_metrics.append(
                    MetricPoint(
                        _sanitize_mlflow_name(f"paper_comparison/{pol}/{reg}/{metric_name}/{field_key}"),
                        fval,
                        0,
                        now_ms,
                    )
                )

    for step_idx, s_sample in enumerate(system_metrics):
        ts = int(s_sample.get("timestamp_ms", now_ms))
        for k, v in (s_sample.get("values") or s_sample).items():
            if k == "timestamp_ms":
                continue
            fval = _finite_float(v)
            if fval is not None:
                campaign_metrics.append(
                    MetricPoint(_sanitize_mlflow_name(f"system/{k}"), fval, step_idx, ts)
                )

    backend.log_batch(
        campaign_run_id, metrics=campaign_metrics, params=campaign_params, tags=campaign_tags
    )
    uploaded_campaign = _upload_selected_artifacts(
        raw_client, campaign_run_id, top_level_artifacts, settings=settings, backend=backend
    )

    # Index records by policy and (policy, seed)
    policies_order: list[str] = list(metadata.get("policies") or [])
    for row in aggregate_rows:
        p = str(row.get("policy", ""))
        if p and p not in policies_order:
            policies_order.append(p)

    agg_by_pol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        agg_by_pol[str(row.get("policy", ""))].append(row)

    seed_by_pol_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_summaries:
        seed_by_pol_seed[(str(row.get("policy", "")), int(row.get("seed", 0)))].append(row)

    reg_by_pol_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in regime_rows:
        reg_by_pol_seed[(str(row.get("policy", "")), int(row.get("seed", 0)))].append(row)

    batch_by_pol_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in batch_rows:
        batch_by_pol_seed[(str(row.get("policy", "")), int(row.get("seed", 0)))].append(row)

    retrain_by_pol_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in retrain_events:
        retrain_by_pol_seed[(str(row.get("policy", "")), int(row.get("seed", 0)))].append(row)

    paper_by_pol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prow in paper_comparison_rows:
        paper_by_pol[str(prow.get("policy", ""))].append(prow)

    policy_runs: dict[str, str] = {}
    seed_runs: dict[str, dict[int, str]] = {}
    retrain_runs: dict[str, list[str]] = {}
    workers = max(1, min(getattr(backend, "concurrency", 2) or 2, 4))

    for pol in policies_order:
        # 2. Policy Run (nested under Campaign)
        pol_tags = _policy_tags(campaign_run_id, metadata, campaign_name, dataset, pol)
        pol_run_id = (
            active_runs.start_policy(pol) if active_runs
            else backend.create_run(experiment_id, pol_tags, run_name=pol).run_id
        )
        policy_runs[pol] = pol_run_id
        seed_runs[pol] = {}
        retrain_runs[pol] = []

        pol_params = _sanitize_params({**base_params_raw, "policy": pol})
        pol_metrics: list[MetricPoint] = []
        for row in agg_by_pol[pol]:
            reg = str(row.get("regime", "default") or "default")
            metric_name = str(row.get("metric", "value"))
            mean_val = _finite_float(row.get("mean"))
            if mean_val is not None:
                summary_key = (
                    f"summary/{metric_name}"
                    if reg in {"", "default", "price_grid"}
                    else f"summary/{reg}/{metric_name}"
                )
                pol_metrics.append(
                    MetricPoint(_sanitize_mlflow_name(summary_key), mean_val, 0, now_ms)
                )
            for stat_key in ("mean", "std", "n_seeds"):
                fval = _finite_float(row.get(stat_key))
                if fval is not None:
                    pol_metrics.append(
                        MetricPoint(
                            _sanitize_mlflow_name(f"aggregate/{reg}/{metric_name}_{stat_key}"),
                            fval,
                            0,
                            now_ms,
                        )
                    )

        for prow in paper_by_pol[pol]:
            reg = str(prow.get("regime", "default") or "default")
            metric_name = str(prow.get("metric", ""))
            for field_key in (
                "observed_mean",
                "observed_std",
                "published_value",
                "published_spread",
                "abs_diff",
                "rel_diff_pct",
            ):
                fval = _finite_float(prow.get(field_key))
                if fval is not None:
                    pol_metrics.append(
                        MetricPoint(
                            _sanitize_mlflow_name(f"paper_comparison/{reg}/{metric_name}/{field_key}"),
                            fval,
                            0,
                            now_ms,
                        )
                    )

        backend.log_batch(pol_run_id, metrics=pol_metrics, params=pol_params, tags=pol_tags)
        pol_artifacts = {
            k: v
            for k, v in all_artifacts.items()
            if k.startswith(f"policies/{pol}/") and "/seeds/" not in k
        }
        _upload_selected_artifacts(raw_client, pol_run_id, pol_artifacts, settings=settings, backend=backend)

        # 3. Seed Runs (nested under Policy)
        seeds_for_pol = sorted(
            {
                s
                for (p, s) in (
                    list(seed_by_pol_seed.keys())
                    + list(reg_by_pol_seed.keys())
                    + list(batch_by_pol_seed.keys())
                    + list(retrain_by_pol_seed.keys())
                )
                if p == pol
            }
        )
        if not seeds_for_pol and isinstance(metadata.get("seeds"), list):
            seeds_for_pol = [int(s) for s in metadata["seeds"]]

        def _publish_seed(seed_val: int) -> tuple[int, str, list[str]]:
            key = (pol, seed_val)
            if active_runs is not None and key in active_runs.completed_seeds:
                s_rid, r_ids = active_runs.completed_seeds[key]
            else:
                s_rid, r_ids = _publish_seed_results(
                    settings=settings,
                    raw_client=raw_client,
                    backend=backend,
                    experiment_id=experiment_id,
                    campaign_name=campaign_name,
                    dataset=dataset,
                    metadata=metadata,
                    pol_name=pol,
                    p_run_id=pol_run_id,
                    seed_val=seed_val,
                    seed_summaries=seed_by_pol_seed[key],
                    regime_rows=reg_by_pol_seed[key],
                    batch_rows=batch_by_pol_seed[key],
                    retrain_events=retrain_by_pol_seed[key],
                    all_artifacts=all_artifacts,
                    existing_run_id=active_runs.seed_runs.get(key) if active_runs else None,
                )
            return seed_val, s_rid, r_ids

        if workers > 1 and len(seeds_for_pol) > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mlflow-seeds") as pool:
                seed_outputs = list(pool.map(_publish_seed, seeds_for_pol))
        else:
            seed_outputs = [_publish_seed(s) for s in seeds_for_pol]

        for s_val, s_rid, r_ids in seed_outputs:
            seed_runs[pol][s_val] = s_rid
            retrain_runs[pol].extend(r_ids)

        backend.set_terminated(pol_run_id, "FINISHED")

    backend.set_terminated(campaign_run_id, "FINISHED")

    return {
        "experiment_id": experiment_id,
        "experiment_name": exp_name,
        "campaign_run_id": campaign_run_id,
        "policy_runs": policy_runs,
        "seed_runs": seed_runs,
        "retrain_runs": retrain_runs,
        "uploaded_artifacts": uploaded_campaign,
    }
