"""Finite ZenML steps run locally in the worker, without an independent schedule."""

import json
import os
from pathlib import Path


def configure_local_stack(root):
    from zenml.client import Client
    from zenml.enums import StackComponentType

    client = Client()
    project = os.environ.get("BENCH_ZENML_PROJECT", "continual-benchmark")
    try:
        client.get_project(project)
    except KeyError:
        client.create_project(project, description="Concept drift detection and retraining policy benchmarks")
    client.set_active_project(project)
    name = os.environ.get("BENCH_ZENML_STACK", "benchmark-local-v1")
    artifact_type = StackComponentType.ARTIFACT_STORE
    orchestrator_type = StackComponentType.ORCHESTRATOR
    try:
        client.get_stack(name)
    except KeyError:
        store = os.environ.get("BENCH_ZENML_ARTIFACT_STORE")
        if store:
            # Provisioned with a bucket-scoped credential by the deployment bootstrap.
            artifact = client.get_stack_component(artifact_type, store)
        else:
            artifact_path = str(Path(root).resolve().parent / "zen-artifacts")
            try:
                artifact = client.get_stack_component(artifact_type, name)
            except KeyError:
                artifact = client.create_stack_component(
                    name, "local", artifact_type, {"path": artifact_path}
                )
        orchestrator = client.get_stack_component(orchestrator_type, "default")
        client.create_stack(name, {artifact_type: artifact.id, orchestrator_type: orchestrator.id})
    client.activate_stack(name)


def execute_pipeline(config, root, services):
    if services:
        from continual_bench.application.readiness import ensure_tracking_ready

        ensure_tracking_ready()
    if services and not os.environ.get("ZENML_STORE_URL", "").startswith("http"):
        raise ValueError("service mode requires an explicit remote ZENML_STORE_URL")
    configure_local_stack(root)
    from continual_bench.adapters.zenml_steps import finite_benchmark

    finite_benchmark(config.model_dump(mode="json"), str(Path(root).resolve()), services)
    return json.loads((Path(root) / config.identity() / "report.json").read_text())
