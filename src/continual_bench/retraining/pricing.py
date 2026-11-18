"""AWS Fargate On-Demand retraining pricing and ``alpha = lambda * custo_retreino_usd``."""

from __future__ import annotations

import math
from typing import Any

DEFAULT_FARGATE_PRICING = {
    "region": "us-east-1",
    "price_date": "2026-11-08",
    "compute_service": "fargate-linux-x86",
    "os_family": "LINUX",
    "cpu_architecture": "X86_64",
    "vcpus": 2.0,
    "memory_gb": 4.0,
    "ephemeral_storage_gb": 50.0,
    "ephemeral_free_gb": 20.0,
    "vcpu_hour_usd": 0.04048,
    "memory_gb_hour_usd": 0.004445,
    "ephemeral_gb_hour_usd": 0.000111,
    "minimum_compute_seconds": 60,
    "pricing_source": "https://aws.amazon.com/fargate/pricing/",
    "compute_source": (
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonECS/current/us-east-1/index.json"
    ),
}

BILLING_MODES = frozenset({"marginal_in_task", "dedicated_task"})


def normalize_fargate_config(aws_cfg: Any = None) -> dict[str, Any]:
    """Return a complete Fargate worker + On-Demand pricing config dictionary."""
    if aws_cfg is None:
        return dict(DEFAULT_FARGATE_PRICING)
    if hasattr(aws_cfg, "model_dump"):
        raw = {
            **aws_cfg.model_dump(),
            "os_family": getattr(aws_cfg, "os_family", DEFAULT_FARGATE_PRICING["os_family"]),
            "cpu_architecture": getattr(
                aws_cfg, "cpu_architecture", DEFAULT_FARGATE_PRICING["cpu_architecture"]
            ),
            "ephemeral_storage_gb": getattr(
                aws_cfg, "ephemeral_storage_gb", DEFAULT_FARGATE_PRICING["ephemeral_storage_gb"]
            ),
            "ephemeral_free_gb": getattr(
                aws_cfg, "ephemeral_free_gb", DEFAULT_FARGATE_PRICING["ephemeral_free_gb"]
            ),
            "ephemeral_gb_hour_usd": getattr(
                aws_cfg, "ephemeral_gb_hour_usd", DEFAULT_FARGATE_PRICING["ephemeral_gb_hour_usd"]
            ),
            "pricing_source": getattr(aws_cfg, "pricing_source", DEFAULT_FARGATE_PRICING["pricing_source"]),
        }
    elif isinstance(aws_cfg, dict):
        raw = dict(aws_cfg)
    else:
        raise TypeError("aws_cfg must be an AwsCostConfig, dict, or None")

    merged = dict(DEFAULT_FARGATE_PRICING)
    for key in DEFAULT_FARGATE_PRICING:
        if key in raw and raw[key] is not None:
            merged[key] = raw[key]
    merged["vcpus"] = float(merged["vcpus"])
    merged["memory_gb"] = float(merged["memory_gb"])
    merged["ephemeral_storage_gb"] = float(merged["ephemeral_storage_gb"])
    merged["ephemeral_free_gb"] = float(merged["ephemeral_free_gb"])
    merged["vcpu_hour_usd"] = float(merged["vcpu_hour_usd"])
    merged["memory_gb_hour_usd"] = float(merged["memory_gb_hour_usd"])
    merged["ephemeral_gb_hour_usd"] = float(merged["ephemeral_gb_hour_usd"])
    merged["minimum_compute_seconds"] = int(merged["minimum_compute_seconds"])
    return merged


def fargate_hourly_rate_usd(aws_cfg: Any = None) -> tuple[float, dict[str, Any]]:
    """Compute the On-Demand USD/hour rate for CPU, RAM, and billable ephemeral storage."""
    cfg = normalize_fargate_config(aws_cfg)
    billable_ephemeral_gb = max(0.0, cfg["ephemeral_storage_gb"] - cfg["ephemeral_free_gb"])
    vcpu_rate_h = cfg["vcpus"] * cfg["vcpu_hour_usd"]
    memory_rate_h = cfg["memory_gb"] * cfg["memory_gb_hour_usd"]
    ephemeral_rate_h = billable_ephemeral_gb * cfg["ephemeral_gb_hour_usd"]
    hourly_rate_usd = vcpu_rate_h + memory_rate_h + ephemeral_rate_h
    return hourly_rate_usd, {
        "config": cfg,
        "billable_ephemeral_gb": billable_ephemeral_gb,
        "unit_rates": {
            "tarifa_usd_vCPU_h": cfg["vcpu_hour_usd"],
            "tarifa_usd_mem_GB_h": cfg["memory_gb_hour_usd"],
            "tarifa_usd_ephemeral_GB_h": cfg["ephemeral_gb_hour_usd"],
        },
        "hourly_breakdown_usd": {
            "vcpu_hour_usd": vcpu_rate_h,
            "memory_hour_usd": memory_rate_h,
            "ephemeral_storage_hour_usd": ephemeral_rate_h,
            "total_hour_usd": hourly_rate_usd,
        },
    }


def fargate_retrain_cost_usd(
    duration_seconds: float,
    *,
    billing_mode: str = "dedicated_task",
    aws_cfg: Any = None,
) -> dict[str, Any]:
    """Compute Fargate On-Demand retraining cost in USD for one retrain action."""
    if billing_mode not in BILLING_MODES:
        raise ValueError(f"billing_mode must be one of {sorted(BILLING_MODES)}")
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("duration_seconds must be finite and nonnegative")

    hourly_rate_usd, rate_info = fargate_hourly_rate_usd(aws_cfg)
    min_seconds = rate_info["config"]["minimum_compute_seconds"]
    if billing_mode == "dedicated_task":
        billed_seconds = float(max(min_seconds, math.ceil(duration_seconds)))
    else:
        billed_seconds = float(duration_seconds)

    cost_usd = hourly_rate_usd * (billed_seconds / 3600.0)
    return {
        "billing_mode": billing_mode,
        "duration_seconds": float(duration_seconds),
        "billed_seconds": billed_seconds,
        "hourly_rate_usd": hourly_rate_usd,
        "custo_retreino_usd": cost_usd,
        **rate_info,
    }


def resolve_retrain_alpha(
    params: dict[str, Any] | None = None,
    *,
    default_alpha: float = 1.0,
    tracking: dict[str, Any] | None = None,
    observed_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Resolve ``alpha`` and On-Demand Fargate pricing for CARA, UPF, and Oracle."""
    params = dict(params or {})
    tracking = dict(tracking or {})
    costs_cfg = tracking.get("costs") or {}
    aws_cfg = costs_cfg.get("aws")

    billing_mode = (
        params.get("fargate_billing_mode") or tracking.get("fargate_billing_mode") or "dedicated_task"
    )
    if billing_mode not in BILLING_MODES:
        raise ValueError(f"fargate_billing_mode must be one of {sorted(BILLING_MODES)}")

    duration_override = params.get("retrain_duration_seconds")
    if duration_override is None:
        duration_override = tracking.get("fargate_retrain_duration_seconds")
    if duration_override is not None:
        duration_seconds = float(duration_override)
    elif observed_duration_seconds is not None and math.isfinite(observed_duration_seconds):
        duration_seconds = float(observed_duration_seconds)
    else:
        duration_seconds = 60.0 if billing_mode == "dedicated_task" else 1.0

    pricing = fargate_retrain_cost_usd(
        duration_seconds,
        billing_mode=billing_mode,
        aws_cfg=aws_cfg,
    )
    if "custo_retreino_usd" in params:
        custo_retreino_usd = float(params["custo_retreino_usd"])
        if not math.isfinite(custo_retreino_usd) or custo_retreino_usd < 0:
            raise ValueError("custo_retreino_usd must be finite and nonnegative")
    else:
        custo_retreino_usd = pricing["custo_retreino_usd"]

    lambda_val = params.get("lambda")
    if lambda_val is None:
        lambda_val = tracking.get("upf_lambda")

    if lambda_val is not None:
        lambda_float = float(lambda_val)
        if not math.isfinite(lambda_float) or lambda_float < 0:
            raise ValueError("lambda must be finite and nonnegative")
        computed_alpha = lambda_float * custo_retreino_usd
        if "alpha" in params and not math.isclose(
            float(params["alpha"]), computed_alpha, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("explicit alpha conflicts with lambda * custo_retreino_usd")
        alpha = computed_alpha
        alpha_source = "lambda_times_custo_retreino_usd"
    else:
        lambda_float = None
        alpha = float(params.get("alpha", default_alpha))
        alpha_source = "explicit_alpha" if "alpha" in params else "default_alpha"

    return {
        "alpha": alpha,
        "alpha_source": alpha_source,
        "lambda": lambda_float,
        "custo_retreino_usd": custo_retreino_usd,
        "reference_cost_per_action_usd": custo_retreino_usd,
        "billing_mode": pricing["billing_mode"],
        "duration_seconds": pricing["duration_seconds"],
        "billed_seconds": pricing["billed_seconds"],
        "hourly_rate_usd": pricing["hourly_rate_usd"],
        "fargate_config": pricing["config"],
        "billable_ephemeral_gb": pricing["billable_ephemeral_gb"],
        "unit_rates": pricing["unit_rates"],
        "hourly_breakdown_usd": pricing["hourly_breakdown_usd"],
        "pricing_source": pricing["config"]["pricing_source"],
        "price_date": pricing["config"]["price_date"],
    }


def summarize_fargate_retraining_cost(
    resolution: dict[str, Any],
    *,
    retrain_count: int,
    retrain_durations: list[float] | None = None,
    operational_execution_mode: str = "local",
    operational_aws_cost_usd: float | None = None,
    operational_aws_cost_reason: str | None = None,
) -> dict[str, Any]:
    """Summarize Fargate On-Demand retraining cost separately from operational execution."""
    if retrain_count < 0:
        raise ValueError("retrain_count must be nonnegative")

    ref_cost = float(resolution["reference_cost_per_action_usd"])
    if retrain_count == 0:
        estimated_cost_usd = 0.0
    elif retrain_durations is not None and len(retrain_durations) == retrain_count:
        estimated_cost_usd = sum(
            fargate_retrain_cost_usd(
                d,
                billing_mode=resolution["billing_mode"],
                aws_cfg=resolution["fargate_config"],
            )["custo_retreino_usd"]
            for d in retrain_durations
        )
    else:
        estimated_cost_usd = float(retrain_count) * ref_cost

    return {
        **resolution,
        "lambda_unavailable_reason": (
            None if resolution.get("lambda") is not None else "lambda_not_configured"
        ),
        "retrain_count": int(retrain_count),
        "estimated_retraining_cost_usd": float(estimated_cost_usd),
        "reference_total_retraining_cost_usd": float(retrain_count) * ref_cost,
        "operational_execution_mode": operational_execution_mode,
        "operational_aws_cost_usd": operational_aws_cost_usd,
        "operational_aws_cost_reason": (
            None if operational_aws_cost_usd is not None else (operational_aws_cost_reason or "not_observed")
        ),
    }
