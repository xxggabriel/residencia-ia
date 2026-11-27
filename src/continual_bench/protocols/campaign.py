"""Campaign configuration loading and end-to-end execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from continual_bench.backends import RuntimeSettings, resolve_backend
from continual_bench.protocols.base import BenchmarkProtocol, CampaignConfig, ProtocolResult
from continual_bench.protocols.cara import CaraProtocol
from continual_bench.protocols.report import (
    compare_with_published_reference,
    generate_campaign_report,
    generate_notebook_artifacts,
)
from continual_bench.protocols.upf import UpfProtocol
from continual_bench.tracking.tracker import BenchmarkTracker

_PROTOCOLS: dict[str, type[BenchmarkProtocol]] = {
    "cara": CaraProtocol,
    "upf": UpfProtocol,
}

_PAPER_IDS: dict[str, str] = {
    "cara": "cara_2024",
    "upf": "upf_2025",
}


def load_campaign_config(path: str | Path) -> CampaignConfig:
    """Load a ``CampaignConfig`` from a YAML or JSON file."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    return CampaignConfig(**payload)


def resolve_campaign_settings(
    config: CampaignConfig,
    *,
    env_name: str | None = None,
    platform: str | None = None,
    output_dir: str | None = None,
    tracking_uri: str | None = None,
    save_model_checkpoints: bool | None = None,
    overrides: dict[str, Any] | None = None,
) -> RuntimeSettings:
    """Resolve ``RuntimeSettings`` for a campaign from config, env, and explicit overrides."""
    merged_overrides: dict[str, Any] = {}
    if config.platform is not None:
        merged_overrides["platform"] = config.platform
    if config.output_dir is not None:
        merged_overrides["output_dir"] = config.output_dir
    if config.tracking_uri is not None:
        merged_overrides["tracking_uri"] = config.tracking_uri
    if config.save_model_checkpoints is not None:
        merged_overrides["save_model_checkpoints"] = config.save_model_checkpoints
    if overrides:
        merged_overrides.update({k: v for k, v in overrides.items() if v is not None})
    if platform is not None:
        merged_overrides["platform"] = platform
    if output_dir is not None:
        merged_overrides["output_dir"] = output_dir
    if tracking_uri is not None:
        merged_overrides["tracking_uri"] = tracking_uri
    if save_model_checkpoints is not None:
        merged_overrides["save_model_checkpoints"] = save_model_checkpoints

    return RuntimeSettings.from_env(
        env_name=env_name,
        overrides=merged_overrides or None,
    )


def _collect_system_sample() -> dict[str, Any]:
    """Collect a lightweight system resource sample for MLflow ``system/*`` metrics."""
    import resource
    import time

    values: dict[str, float] = {
        "process_cpu_user_s": float(resource.getrusage(resource.RUSAGE_SELF).ru_utime),
        "process_cpu_sys_s": float(resource.getrusage(resource.RUSAGE_SELF).ru_stime),
    }
    try:
        import psutil  # type: ignore[import-untyped]

        proc = psutil.Process()
        values["cpu_utilization_percentage"] = float(psutil.cpu_percent(interval=None))
        values["system_memory_percentage"] = float(psutil.virtual_memory().percent)
        values["process_memory_mb"] = float(proc.memory_info().rss / (1024 * 1024))
    except Exception:
        pass
    return {"timestamp_ms": int(time.time() * 1000), "values": values}


def create_campaign_tracker(
    config: CampaignConfig,
    settings: RuntimeSettings,
    *,
    run_id: str | None = None,
) -> tuple[Any, BenchmarkTracker, str]:
    """Create backend, run_id, and initialized ``BenchmarkTracker`` for a campaign."""
    backend = resolve_backend(platform=settings.platform, settings=settings)
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{config.name}_{timestamp}_{uuid4().hex[:8]}"
    tracker = BenchmarkTracker(backend=backend, run_id=run_id, settings=settings)
    tracker.log_system_metrics(_collect_system_sample())
    tracker.log_metadata(
        {
            "campaign_name": config.name,
            "protocol": config.protocol,
            "dataset": config.dataset.name,
            "model": config.model.name,
            "model_params": dict(config.model.params),
            "seeds": config.seeds,
            "policies": config.policies,
            "staleness_mode": config.staleness_mode,
            "kernel_gamma": config.kernel_gamma,
        }
    )
    return backend, tracker, run_id


def finalize_campaign_result(
    config: CampaignConfig,
    result: ProtocolResult,
    backend: Any,
    tracker: BenchmarkTracker,
    settings: RuntimeSettings,
    run_id: str,
) -> ProtocolResult:
    """Generate reports and publish campaign and policy aggregates."""
    meta = tracker._metadata
    report_regimes = (
        {str(k): float(v) for k, v in meta["regimes"].items()}
        if isinstance(meta.get("regimes"), dict)
        else (dict(config.regimes) if config.regimes is not None else None)
    )
    grid_bounds: tuple[float, float] | None = None
    if isinstance(meta.get("alpha_grid"), list) and meta["alpha_grid"]:
        vals = [float(v) for v in meta["alpha_grid"]]
        grid_bounds = (min(vals), max(vals))
    elif meta.get("kappa_min") is not None and meta.get("kappa_max") is not None:
        grid_bounds = (float(meta["kappa_min"]), float(meta["kappa_max"]))
    elif config.alpha_grid:
        vals = [float(v) for v in config.alpha_grid]
        grid_bounds = (min(vals), max(vals))
    elif config.kappa_grid:
        vals = [float(v) for v in config.kappa_grid]
        grid_bounds = (min(vals), max(vals))

    report = generate_campaign_report(
        result,
        backend,
        run_id=run_id,
        regimes=report_regimes,
        grid_bounds=grid_bounds,
        generate_plots=settings.generate_plots,
    )
    comparison = compare_with_published_reference(
        result.aggregate_summary,
        paper_id=_PAPER_IDS[config.protocol],
        dataset=result.dataset,
        reference_csv=config.reference_csv,
        backend=backend,
        run_id=run_id,
        generate_plots=settings.generate_plots,
    )
    notebook_bundle = generate_notebook_artifacts(
        result,
        backend,
        run_id=run_id,
        batch_rows=tracker._batch_rows,
        comparison_rows=comparison.get("rows", []),
    )
    tracker.log_system_metrics(_collect_system_sample())
    tracker.log_paper_comparison(comparison.get("rows", []))
    tracker.register_external_artifacts(report.get("artifacts", {}))
    tracker.register_external_artifacts(comparison.get("artifacts", {}))
    tracker.register_external_artifacts(notebook_bundle.get("artifacts", {}))
    tracker.assert_no_checkpoints()
    flushed = tracker.flush()
    result.artifacts.update(flushed)
    result.artifacts.update(report.get("artifacts", {}))
    result.artifacts.update(comparison.get("artifacts", {}))
    result.artifacts.update(notebook_bundle.get("artifacts", {}))
    result.mlflow_receipt = tracker.mlflow_receipt

    return result


def run_campaign(
    config_or_path: CampaignConfig | str | Path,
    *,
    env_name: str | None = None,
    platform: str | None = None,
    output_dir: str | None = None,
    tracking_uri: str | None = None,
    save_model_checkpoints: bool | None = None,
    overrides: dict[str, Any] | None = None,
) -> ProtocolResult:
    """Execute a benchmark campaign end-to-end with backend, tracking, and paper comparison."""
    config = (
        config_or_path
        if isinstance(config_or_path, CampaignConfig)
        else load_campaign_config(config_or_path)
    )
    settings = resolve_campaign_settings(
        config,
        env_name=env_name,
        platform=platform,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        save_model_checkpoints=save_model_checkpoints,
        overrides=overrides,
    )
    backend, tracker, run_id = create_campaign_tracker(config, settings)

    protocol_cls = _PROTOCOLS.get(config.protocol)
    if protocol_cls is None:
        raise ValueError(f"Unsupported protocol {config.protocol!r}")
    protocol = protocol_cls()
    tracker.start_campaign()
    try:
        result = protocol.run(config, backend, tracker)
        return finalize_campaign_result(config, result, backend, tracker, settings, run_id)
    except BaseException:
        tracker.fail_campaign()
        raise
