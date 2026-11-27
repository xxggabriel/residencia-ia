"""Contracts, Strategy Registry, and Dependency Injection for CLI, Container, and AWS runtimes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

from continual_bench.application.components import ComponentRegistry, default_registry
from continual_bench.infrastructure.object_store import LocalObjectStore

from .contracts import RunUnit

ExecutionMode = Literal["cli", "container", "aws"]


class WorkloadExecutor(Protocol):
    """Strategy contract for executing one RunUnit (zia snapshot, matrix case, or custom)."""

    def execute(
        self,
        unit: RunUnit,
        store: Any,
        workdir: Path,
        log: logging.Logger,
        *,
        stop_requested: Callable[[], bool] | None = None,
        progress: Callable[[int, dict], None] | None = None,
        checkpoint: Any = None,
    ) -> dict[str, Any]: ...


class TrackerProtocol(Protocol):
    """Contract for experiment/run tracking across CLI, Container, and AWS."""

    def start(self, hyperparameters: Any = None) -> Any: ...
    def log_epoch_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> bool: ...
    def progress(self, step: int, metrics: dict[str, Any]) -> Any: ...
    def finish(self, status: str = "Completed", message: str | None = None) -> Any: ...
    def mark_failed(self, error_message: str | None = None) -> Any: ...


class ModelRegistrarProtocol(Protocol):
    """Contract for registering trained model artifacts and evaluation metrics."""

    def __call__(
        self,
        unit: RunUnit,
        report: dict[str, Any],
        metadata: dict[str, Any],
        archive_uri: str,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


class WorkloadRegistry:
    """Registry mapping RunUnit.kind to its WorkloadExecutor strategy."""

    def __init__(self) -> None:
        self._executors: dict[str, WorkloadExecutor] = {}

    def register(self, kind: str, executor: WorkloadExecutor) -> None:
        self._executors[kind] = executor

    def get(self, kind: str) -> WorkloadExecutor:
        if kind not in self._executors:
            raise ValueError(f"unknown workload kind {kind!r}; available={sorted(self._executors)}")
        return self._executors[kind]

    def kinds(self) -> tuple[str, ...]:
        return tuple(self._executors)


class ZiaWorkloadExecutor:
    """Strategy executing Zia snapshot-based evaluations."""

    def execute(
        self,
        unit: RunUnit,
        store: Any,
        workdir: Path,
        log: logging.Logger,
        *,
        stop_requested: Callable[[], bool] | None = None,
        progress: Callable[[int, dict], None] | None = None,
        checkpoint: Any = None,
    ) -> dict[str, Any]:
        from .worker import execute_zia

        return execute_zia(
            unit,
            store,
            workdir,
            log,
            stop_requested=stop_requested,
            progress=progress,
            checkpoint=checkpoint,
        )


class MatrixWorkloadExecutor:
    """Strategy executing generic matrix cases (any registered dataset, model, policy, metrics)."""

    def execute(
        self,
        unit: RunUnit,
        store: Any,
        workdir: Path,
        log: logging.Logger,
        *,
        stop_requested: Callable[[], bool] | None = None,
        progress: Callable[[int, dict], None] | None = None,
        checkpoint: Any = None,
    ) -> dict[str, Any]:
        from .worker import execute_matrix_case

        return execute_matrix_case(
            store,
            unit,
            workdir,
            progress=progress,
            checkpoint=checkpoint,
        )


def default_workloads() -> WorkloadRegistry:
    registry = WorkloadRegistry()
    registry.register("zia", ZiaWorkloadExecutor())
    registry.register("matrix_case", MatrixWorkloadExecutor())
    return registry


def resolve_unit_model_name(unit: RunUnit) -> str:
    """Resolve model name from a RunUnit without hardcoding Zia dataset names."""
    params = unit.params or {}
    if "model" in params:
        model = params["model"]
        return model["name"] if isinstance(model, dict) else str(model)
    config = params.get("config") or {}
    if isinstance(config, dict) and "model" in config:
        model = config["model"]
        return model["name"] if isinstance(model, dict) else str(model)
    if "model" in unit.labels:
        return str(unit.labels["model"])
    dataset = params.get("dataset") or unit.labels.get("dataset")
    if dataset:
        from continual_bench.datasets.pinned_stream import DATASETS as ZIA_DATASETS

        if dataset in ZIA_DATASETS:
            return ZIA_DATASETS[dataset]["model"]
    return "generic"


def resolve_unit_dataset(unit: RunUnit) -> str:
    params = unit.params or {}
    if "dataset" in params:
        ds = params["dataset"]
        return ds["name"] if isinstance(ds, dict) else str(ds)
    if "dataset" in unit.labels:
        return str(unit.labels["dataset"])
    config = params.get("config") or {}
    if isinstance(config, dict) and "dataset" in config:
        ds = config["dataset"]
        return ds["name"] if isinstance(ds, dict) else str(ds)
    return "unknown"


def resolve_unit_policy(unit: RunUnit) -> str:
    params = unit.params or {}
    if "policy" in params:
        pol = params["policy"]
        return pol["name"] if isinstance(pol, dict) else str(pol)
    if "policy" in unit.labels:
        return str(unit.labels["policy"])
    config = params.get("config") or {}
    if isinstance(config, dict) and "policy" in config:
        pol = config["policy"]
        return pol["name"] if isinstance(pol, dict) else str(pol)
    return "none"


@dataclass
class ExecutionRuntime:
    """Dependency Injection container binding storage, component registry, workloads, and tracking."""

    mode: ExecutionMode
    store: Any
    registry: ComponentRegistry = field(default_factory=default_registry)
    workloads: WorkloadRegistry = field(default_factory=default_workloads)
    tracker_factory: Callable[[RunUnit], Any] | None = None
    model_registrar: ModelRegistrarProtocol | None = None
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))


def build_cli_runtime(
    root: str | Path,
    *,
    plugins: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
) -> ExecutionRuntime:
    """Create a decoupled runtime for local CLI execution (no AWS/container dependencies)."""
    resolved_env = dict(os.environ if env is None else env)
    resolved_env.pop("MLFLOW_TRACKING_URI", None)
    return ExecutionRuntime(
        mode="cli",
        store=LocalObjectStore(Path(root)),
        registry=default_registry(plugins),
        workloads=default_workloads(),
        tracker_factory=None,
        model_registrar=None,
        env=resolved_env,
    )


def build_container_runtime(
    *,
    local_root: str | Path | None = None,
    bucket: str | None = None,
    endpoint_url: str | None = None,
    mlflow_tracking_uri: str | None = None,
    plugins: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
) -> ExecutionRuntime:
    """Create a decoupled runtime for containerized execution (MinIO/S3 or mounted volume)."""
    resolved_env = dict(os.environ if env is None else env)
    if local_root is not None:
        store = LocalObjectStore(Path(local_root))
    else:
        from continual_bench.infrastructure.aws.s3 import S3ObjectStore

        target_bucket = bucket or resolved_env.get("BENCH_BUCKET", "benchmarks")
        endpoint = endpoint_url or resolved_env.get("S3_ENDPOINT_URL")
        if endpoint:
            import boto3

            s3_client = boto3.client("s3", endpoint_url=endpoint)
            store = S3ObjectStore(target_bucket, client=s3_client)
        else:
            store = S3ObjectStore(target_bucket)

    uri = mlflow_tracking_uri or resolved_env.get("MLFLOW_TRACKING_URI")
    tracker_factory = None
    if uri:
        from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker

        def tracker_factory(unit: RunUnit) -> Any:
            return SageMakerMlflowTracker(
                project=resolved_env.get("BENCH_STACK", "continual-bench"),
                dataset=resolve_unit_dataset(unit),
                policy=resolve_unit_policy(unit),
                scope=str(unit.params.get("scope", "smoke")),
                campaign_id=unit.campaign_id,
                model_name=resolve_unit_model_name(unit),
                run_id=unit.run_id,
                job_name=unit.job_name,
                git_sha=resolved_env.get(
                    "BENCH_GIT_REVISION",
                    unit.contract.get("release", {}).get("git_sha", "local"),
                ),
                stack=resolved_env.get("BENCH_STACK", "continual-bench"),
                tracking_uri=uri,
                async_progress=True,
            )

    return ExecutionRuntime(
        mode="container",
        store=store,
        registry=default_registry(plugins),
        workloads=default_workloads(),
        tracker_factory=tracker_factory,
        model_registrar=None,
        env=resolved_env,
    )


def build_aws_runtime(
    bucket: str,
    *,
    kms_key_id: str | None = None,
    inference_image: str | None = None,
    mlflow_tracking_uri: str | None = None,
    enable_sagemaker_experiments: bool = True,
    enable_model_registry: bool = True,
    plugins: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
) -> ExecutionRuntime:
    """Create a cloud runtime for AWS Batch / SageMaker execution."""
    from continual_bench.infrastructure.aws.s3 import S3ObjectStore

    resolved_env = dict(os.environ if env is None else env)
    kms = kms_key_id or resolved_env.get("BENCH_KMS_KEY_ID")
    store = S3ObjectStore(bucket, kms_key_id=kms)

    registrar = None
    inf_img = inference_image or resolved_env.get("BENCH_INFERENCE_IMAGE")
    if enable_model_registry and inf_img:
        from continual_bench.infrastructure.aws.model_registry import SageMakerModelRegistry

        from .worker import sagemaker_registrar

        registrar = sagemaker_registrar(SageMakerModelRegistry(), inf_img)

    uri = mlflow_tracking_uri or resolved_env.get("MLFLOW_TRACKING_URI")
    tracker_factory = None
    if uri or enable_sagemaker_experiments:

        def tracker_factory(unit: RunUnit) -> Any:
            dataset = resolve_unit_dataset(unit)
            policy = resolve_unit_policy(unit)
            scope = str(unit.params.get("scope", "smoke"))
            model_name = resolve_unit_model_name(unit)
            git_sha = resolved_env.get(
                "BENCH_GIT_REVISION", unit.contract.get("release", {}).get("git_sha", "uncommitted")
            )
            stack = resolved_env.get("BENCH_STACK", "continual-bench")
            if enable_sagemaker_experiments:
                from continual_bench.infrastructure.aws.experiments import SageMakerExperimentsTracker

                return SageMakerExperimentsTracker(
                    project=stack,
                    dataset=dataset,
                    policy=policy,
                    scope=scope,
                    campaign_id=unit.campaign_id,
                    model_name=model_name,
                    run_id=unit.run_id,
                    job_name=unit.job_name,
                    git_sha=git_sha,
                    stack=stack,
                    mlflow_tracking_uri=uri,
                    emit_stdout=False,
                )
            from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker

            return SageMakerMlflowTracker(
                project=stack,
                dataset=dataset,
                policy=policy,
                scope=scope,
                campaign_id=unit.campaign_id,
                model_name=model_name,
                run_id=unit.run_id,
                job_name=unit.job_name,
                git_sha=git_sha,
                stack=stack,
                tracking_uri=uri,
                async_progress=True,
            )

    return ExecutionRuntime(
        mode="aws",
        store=store,
        registry=default_registry(plugins),
        workloads=default_workloads(),
        tracker_factory=tracker_factory,
        model_registrar=registrar,
        env=resolved_env,
    )


def resolve_runtime(
    env_name: str | None = None,
    *,
    settings: Any = None,
    plugins: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
) -> ExecutionRuntime:
    """Resolve an ``ExecutionRuntime`` from ``RuntimeSettings`` (``local``, ``docker``, or ``aws``)."""
    from continual_bench.backends.base import RuntimeSettings

    resolved_settings = settings or RuntimeSettings.from_env(env_name=env_name)
    merged_env = dict(os.environ if env is None else env)
    merged_env.setdefault("BENCH_ENV", resolved_settings.env_name)
    merged_env.setdefault("BENCH_PLATFORM", resolved_settings.platform)
    merged_env.setdefault(
        "BENCH_SAVE_MODEL_CHECKPOINTS",
        "true" if resolved_settings.save_model_checkpoints else "false",
    )
    merged_env.setdefault(
        "BENCH_GENERATE_PLOTS",
        "true" if resolved_settings.generate_plots else "false",
    )
    if resolved_settings.tracking_uri:
        merged_env.setdefault("MLFLOW_TRACKING_URI", resolved_settings.tracking_uri)
    if resolved_settings.s3_endpoint_url:
        merged_env.setdefault("S3_ENDPOINT_URL", resolved_settings.s3_endpoint_url)
    if resolved_settings.bucket:
        merged_env.setdefault("BENCH_BUCKET", resolved_settings.bucket)

    if resolved_settings.platform == "aws":
        bucket = resolved_settings.bucket or merged_env.get("BENCH_BUCKET", "benchmarks")
        return build_aws_runtime(
            bucket,
            mlflow_tracking_uri=resolved_settings.tracking_uri or None,
            enable_model_registry=resolved_settings.save_model_checkpoints,
            plugins=plugins,
            env=merged_env,
        )
    if resolved_settings.platform == "homelab":
        raw_out = str(resolved_settings.output_dir)
        local_root = None if raw_out.startswith("s3://") else raw_out
        return build_container_runtime(
            local_root=local_root,
            bucket=resolved_settings.bucket or None,
            endpoint_url=resolved_settings.s3_endpoint_url or None,
            mlflow_tracking_uri=resolved_settings.tracking_uri or None,
            plugins=plugins,
            env=merged_env,
        )
    return build_cli_runtime(
        str(resolved_settings.output_dir),
        plugins=plugins,
        env=merged_env,
    )
