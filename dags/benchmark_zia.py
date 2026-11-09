"""Three manually triggered full or 10k Zia benchmark DAGs."""

import time
import urllib.error
from datetime import timedelta

from airflow.sdk import Param, dag, task
from bench_jobs import consolidate, request_json
from bench_readiness import require_dependencies

POLICIES = ("frozen", "periodic", "drift", "incremental")
WAIT_SECONDS = 2 * 24 * 3600 + 600


def build(dataset):
    @dag(
        dag_id=f"benchmark_zia_{dataset}", schedule=None, catchup=False,
        max_active_runs=1, max_active_tasks=4, tags=["benchmark", "zia", dataset, "mlflow"],
        params={
            "scope": Param("smoke", enum=["smoke", "full"]),
            "campaign": Param("v1", type="string", pattern="^[A-Za-z0-9_-]{1,64}$"),
        },
        default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    )
    def workflow():
        @task(priority_weight=10)
        def plan(params=None):
            require_dependencies()
            return [{"dataset": dataset, "scope": params["scope"], "policy": policy,
                     "campaign": params["campaign"]} for policy in POLICIES]

        @task(max_active_tis_per_dagrun=3, execution_timeout=timedelta(seconds=WAIT_SECONDS + 60))
        def evaluate(config):
            deadline = time.monotonic() + WAIT_SECONDS
            while True:
                try:
                    key = request_json("/zia-runs", config)["run_key"]
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 429 or time.monotonic() >= deadline:
                        raise
                    time.sleep(15)
            while time.monotonic() < deadline:
                status = request_json(f"/zia-runs/{key}")
                if status["status"] == "succeeded":
                    return status
                if status["status"] == "failed":
                    raise RuntimeError(f"{key}: {status['error']}")
                time.sleep(15)
            raise TimeoutError(f"Zia job still incomplete: {key}")

        @task
        def finish(results):
            return consolidate(results)

        finish(evaluate.expand(config=plan()))

    return workflow()


for _dataset in ("fraud", "social", "ecommerce"):
    globals()[f"benchmark_zia_{_dataset}"] = build(_dataset)
