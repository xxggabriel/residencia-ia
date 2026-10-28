"""Manual finite experiment jobs. No scheduling inside ZenML and no per-event Airflow tasks."""

import json
import time
import urllib.request
from datetime import timedelta

from airflow.exceptions import AirflowFailException
from airflow.sdk import dag, task
from bench_readiness import require_dependencies


@dag(
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["benchmark", "smoke"],
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
)
def benchmark_smoke():
    @task(task_id="services_ready", retries=0, execution_timeout=timedelta(seconds=30))
    def services_ready():
        try:
            return require_dependencies()
        except RuntimeError as error:
            raise AirflowFailException(str(error)) from None

    admission = services_ready()

    @task(execution_timeout=timedelta(minutes=20))
    def run_job(dataset: str, mode: str):
        config = {
            "name": "airflow-smoke",
            "mode": mode,
            "dataset": {"name": dataset, "samples": 600, "label_delay": 3},
            "warmup": 50,
            "metric_window": 50,
            "policy": {
                "name": "none" if mode == "offline" else "drift",
                "min_samples": 50,
                "min_window": 25,
                "cooldown": 50,
            },
        }
        request = urllib.request.Request(
            "http://worker:8088/runs",
            data=json.dumps(config).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            identity = json.load(response)["run_key"]
        started = time.monotonic()
        while time.monotonic() - started < 1000:
            with urllib.request.urlopen(f"http://worker:8088/runs/{identity}", timeout=15) as response:
                status = json.load(response)
            if status["status"] == "succeeded":
                return status
            if status["status"] == "failed":
                raise RuntimeError(status["error"])
            time.sleep(3)
        raise TimeoutError(f"job still incomplete: {identity}")

    for dataset in ("sea", "agrawal", "stagger"):
        for mode in ("streaming", "incremental", "offline"):
            admission >> run_job.override(task_id=f"{dataset}_{mode}")(dataset, mode)


benchmark_smoke()
