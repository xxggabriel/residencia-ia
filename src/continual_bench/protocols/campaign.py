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


def run_campaign(
    config_or_path: CampaignConfig | str | Path,
    *,
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

    settings = RuntimeSettings.from_env(overrides=merged_overrides or None)
    backend = resolve_backend(platform=settings.platform, settings=settings)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config.name}_{timestamp}_{uuid4().hex[:8]}"
    tracker = BenchmarkTracker(backend=backend, run_id=run_id, settings=settings)
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
        }
    )

    protocol_cls = _PROTOCOLS.get(config.protocol)
    if protocol_cls is None:
        raise ValueError(f"Unsupported protocol {config.protocol!r}")
    protocol = protocol_cls()
    tracker.start_campaign()
    try:
        result = protocol.run(config, backend, tracker)

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
        )
        comparison = compare_with_published_reference(
            result.aggregate_summary,
            paper_id=_PAPER_IDS[config.protocol],
            dataset=result.dataset,
            reference_csv=config.reference_csv,
            backend=backend,
            run_id=run_id,
        )
        tracker.assert_no_checkpoints()
        tracker.log_paper_comparison(comparison.get("rows", []))
        tracker.register_external_artifacts(report.get("artifacts", {}))
        tracker.register_external_artifacts(comparison.get("artifacts", {}))
        flushed = tracker.flush()
        result.artifacts.update(flushed)
        result.artifacts.update(report.get("artifacts", {}))
        result.artifacts.update(comparison.get("artifacts", {}))

        result.mlflow_receipt = tracker.mlflow_receipt
        return result
    except BaseException:
        tracker.fail_campaign()
        raise
