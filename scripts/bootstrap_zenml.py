"""Idempotently register the benchmark project and a private MinIO artifact store.

Run in the worker with the Compose environment. Credentials never go to stdout.
The dedicated secret must already have been provisioned by the operator.
"""

import os

from zenml.client import Client
from zenml.enums import StackComponentType

from continual_bench.adapters.zenml_pipeline import configure_local_stack


def main():
    client = Client()
    project = os.environ.get("BENCH_ZENML_PROJECT", "default")
    try:
        client.get_project(project)
    except KeyError:
        client.create_project(project, description="Concept drift detection and retraining policy benchmarks")
    client.set_active_project(project)
    client.update_project(project, new_display_name="Continual Benchmark",
                          new_description="Benchmark de deteccao de drift e politicas de retreinamento")
    name = os.environ["BENCH_ZENML_ARTIFACT_STORE"]
    client.get_secret("benchmark-minio-artifacts")
    try:
        client.get_stack_component(StackComponentType.ARTIFACT_STORE, name)
    except KeyError:
        client.create_stack_component(
            name, "s3", StackComponentType.ARTIFACT_STORE,
            {"path": "s3://zenml-artifacts",
             "key": "{{benchmark-minio-artifacts.key}}",
             "secret": "{{benchmark-minio-artifacts.secret}}",
             "client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT_URL"], "region_name": "us-east-1"},
             "config_kwargs": {"s3": {"addressing_style": "path"}}},
        )
    configure_local_stack(os.environ.get("BENCH_STATE", "/state/runs"))
    print("Benchmark project and S3 stack configured")


if __name__ == "__main__":
    main()
