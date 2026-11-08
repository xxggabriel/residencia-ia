"""Manually started whole-project benchmark, gated by all 100-example/two-epoch smokes."""

from datetime import timedelta

from airflow.sdk import Param, dag, task
from task_jobs import configurations, execute, readiness


@dag(dag_id="benchmark_task_suite", schedule=None, catchup=False, max_active_runs=1,
     max_active_tasks=8, tags=["benchmark", "semana09", "all-datasets"],
     params={"samples": Param(3000, type="integer", minimum=300, maximum=100000),
             "dataset_examples": Param(100, type="integer", minimum=100, maximum=1000),
             "epochs": Param(2, type="integer", minimum=1, maximum=20),
             "label_delay": Param(3, type="integer", minimum=0, maximum=100),
             "seeds": Param([42], type="array", minItems=1, maxItems=1, uniqueItems=True,
                            items={"type": "integer", "minimum": 0, "maximum": 2147483647})},
     default_args={"retries": 1, "retry_delay": timedelta(seconds=30)})
def full_project():
    @task(retries=0, execution_timeout=timedelta(seconds=60))
    def admit_datasets():
        return readiness()

    @task
    def plan(admission, params=None):
        return configurations(params, admission["code_sha256"])

    @task(execution_timeout=timedelta(minutes=26))
    def benchmark_case(case):
        return execute(case)

    @task
    def consolidate(results):
        if not results or len({r["key"] for r in results}) != len(results):
            raise ValueError("expected nonempty, unique, published benchmark results")
        return {"complete": True, "runs": len(results)}

    cases = plan(admit_datasets())
    consolidate(benchmark_case.expand(case=cases))


benchmark_task_suite = full_project()
