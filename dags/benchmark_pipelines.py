"""Manual, parameterized production benchmark DAGs, independent of smoke checks."""

import logging
from datetime import timedelta

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, task
from bench_jobs import configurations, consolidate, execute_job, tracking_profile
from bench_readiness import require_dependencies


def build_dag(suite):
    @dag(
        dag_id=f"benchmark_{suite}", schedule=None, catchup=False,
        max_active_runs=1, max_active_tasks=8,
        tags=["benchmark", suite, "zenml", "mlflow"],
        params={
            "samples": Param(300 if suite == "metrics_smoke" else 3000,
                             type="integer", minimum=300, maximum=100000),
            "seeds": Param(
                [42, 43, 44, 45, 46]
                if suite in ("controlled", "versioned", "nonlinear", "metrics") else [42],
                type="array", minItems=1, maxItems=6 if suite in ("nonlinear", "metrics") else 10,
                uniqueItems=True,
                items={"type": "integer", "minimum": 0, "maximum": 2147483647}),
            "label_delay": Param(3, type="integer", minimum=0, maximum=100),
            **({"campaign": Param("metrics-v6" if suite == "metrics" else "metrics-v6-smoke", type="string",
                                  pattern="^[A-Za-z0-9_-]{1,64}$"),
                "tracking": Param(tracking_profile(), type="object")}
               if suite in ("metrics", "metrics_smoke") else {}),
        },
        default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
        description="Materialize data, evaluate drift and retraining policies, publish and consolidate.",
    )
    def workflow():
        @task(retries=0, execution_timeout=timedelta(seconds=30))
        def services_ready():
            try:
                return require_dependencies()
            except RuntimeError as error:
                raise AirflowFailException(str(error)) from None

        @task
        def plan(params=None):
            return configurations(params, suite)

        @task(execution_timeout=timedelta(minutes=40 if suite in ("metrics", "metrics_smoke") else 22))
        def evaluate_and_publish(config):
            return execute_job(config)

        @task
        def consolidate_results(results):
            summary = consolidate(results)
            logging.info("Completed benchmark matrix: %s", summary)
            return summary

        ready = services_ready()
        configs = plan()
        ready >> configs
        results = evaluate_and_publish.expand(config=configs)
        consolidate_results(results)

    return workflow()


for _suite in (
    "streaming", "incremental", "offline", "strategies", "full", "controlled", "versioned", "nonlinear",
    "metrics", "metrics_smoke",
):
    globals()[f"benchmark_{_suite}"] = build_dag(_suite)
