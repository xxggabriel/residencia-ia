"""Airflow DAG for executing CARA and UPF benchmark campaigns in parallel across workers."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import timedelta

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, get_current_context, task
from bench_readiness import require_dependencies

WORKER_URL = os.environ.get("BENCH_WORKER_URL", "http://worker:8088").rstrip("/")


def _worker_request(path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{WORKER_URL}{path}",
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def build_campaign_payload(params: dict) -> dict:
    """Build normalized campaign request payload from Airflow DAG params."""
    config_path = str(params.get("config_path") or "").strip()
    protocol = str(params.get("protocol") or "cara").strip().lower()
    dataset = str(params.get("dataset") or "electricity").strip()
    campaign = str(params.get("campaign") or "").strip()
    model = str(params.get("model") or "logistic").strip()
    policies = [str(p).strip() for p in (params.get("policies") or []) if str(p).strip()]
    seeds = [int(s) for s in (params.get("seeds") or [])]
    env_name = str(params.get("env") or "docker").strip().lower()
    save_checkpoints = bool(params.get("save_checkpoints", False))

    return {
        "env": env_name,
        "config_path": config_path or None,
        "protocol": protocol,
        "dataset": dataset,
        "campaign": campaign or None,
        "model": model,
        "policies": policies,
        "seeds": seeds,
        "save_checkpoints": save_checkpoints,
    }


def _poll_job(status_path: str, *, deadline_seconds: int = 3600) -> dict:
    started = time.monotonic()
    while time.monotonic() - started < deadline_seconds:
        status = _worker_request(status_path)
        if status["status"] == "succeeded":
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"Job {status_path} failed: {status.get('error')}")
        time.sleep(3)
    raise TimeoutError(f"Job still incomplete after {deadline_seconds}s: {status_path}")


def execute_seed_job(seed_job: dict, airflow_task: dict | None = None) -> dict:
    """Submit a single seed matrix job to the worker pool and wait for completion."""
    payload = {**seed_job, "airflow_task": airflow_task} if airflow_task else seed_job
    submitted = _worker_request("/campaign-seeds", payload)
    if association := submitted.get("task_association"):
        logging.info("Worker job association: %s", association)
    seed_key = submitted["seed_key"]
    return _poll_job(f"/campaign-seeds/{seed_key}")


def execute_consolidate_job(seed_results: list[dict], seed_jobs: list[dict]) -> dict:
    """Score per-seed matrices, then publish campaign aggregates and paper comparisons."""
    first = seed_jobs[0]
    payload = {
        "campaign_key": first["campaign_key"],
        "config": first["config"],
        "env": first["env"],
        "save_checkpoints": first["save_checkpoints"],
        "seed_paths": [r["seed_path"] for r in seed_results],
    }
    submitted = _worker_request("/campaign-consolidate", payload)
    cons_key = submitted["consolidate_key"]
    return _poll_job(f"/campaign-consolidate/{cons_key}")


@dag(
    dag_id="benchmark_campaign",
    schedule=None,
    catchup=False,
    max_active_runs=8,
    max_active_tasks=32,
    tags=["benchmark", "cara", "upf", "mlflow"],
    params={
        "env": Param("docker", type="string", enum=["local", "docker", "aws"]),
        "config_path": Param(
            "",
            type="string",
            description="Optional path to campaign YAML (e.g. configs/campaigns/cara-electricity.yaml)",
        ),
        "protocol": Param("cara", type="string", enum=["cara", "upf"]),
        "dataset": Param("electricity", type="string"),
        "campaign": Param(
            "",
            type="string",
            description="Campaign name (defaults to config name or <protocol>-<dataset>)",
        ),
        "model": Param("logistic", type="string"),
        "policies": Param([], type="array", items={"type": "string"}),
        "seeds": Param([], type="array", items={"type": "integer"}),
        "save_checkpoints": Param(False, type="boolean"),
    },
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    description="Execute CARA/UPF benchmark & paper comparison campaigns in parallel across workers.",
)
def benchmark_campaign():
    @task(retries=1, retry_delay=timedelta(seconds=15), execution_timeout=timedelta(minutes=5))
    def services_ready():
        try:
            return require_dependencies()
        except RuntimeError as error:
            raise AirflowFailException(str(error)) from None

    @task
    def plan_seeds(params=None):
        payload = {
            **build_campaign_payload(params or {}),
            "dag_run_id": get_current_context()["ti"].run_id,
        }
        planned = _worker_request("/campaign-plan", payload)
        return planned["seed_jobs"]

    @task(max_active_tis_per_dagrun=8, execution_timeout=timedelta(minutes=45))
    def run_seed(seed_job):
        ti = get_current_context()["ti"]
        return execute_seed_job(seed_job, {
            "dag_id": ti.dag_id,
            "dag_run_id": ti.run_id,
            "task_id": ti.task_id,
            "map_index": ti.map_index,
            "try_number": ti.try_number,
        })

    @task(execution_timeout=timedelta(minutes=60))
    def consolidate_campaign(seed_results, seed_jobs):
        result = execute_consolidate_job(list(seed_results), list(seed_jobs))
        logging.info("Completed campaign: %s", result.get("mlflow_receipt"))
        return result

    ready = services_ready()
    seed_jobs = plan_seeds()
    ready >> seed_jobs
    seed_results = run_seed.expand(seed_job=seed_jobs)
    consolidate_campaign(seed_results, seed_jobs)


benchmark_campaign()
