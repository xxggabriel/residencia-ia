import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from continual_bench.core.policies import BATCH_POLICY_KINDS, POLICIES, PolicyStrategy
from continual_bench.core.tasks import BINARY


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Component(Strict):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    params: dict = Field(default_factory=dict)


class DatasetConfig(Strict):
    name: str = Field(default="sea", pattern=r"^[a-z][a-z0-9_]*$")
    samples: int = Field(default=1200, ge=100, le=100_000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    label_delay: int = Field(default=0, ge=0, le=10_000)
    epoch: int = Field(default=1_735_689_600, ge=0)
    params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def reserved_params(self):
        if set(self.params) & {"name", "samples", "seed", "label_delay", "epoch"}:
            raise ValueError("source params cannot override standard dataset fields")
        return self


class PolicyConfig(Strict):
    name: str = Field(default="periodic", pattern=r"^[a-z][a-z0-9_]*$")
    every: int = Field(default=200, ge=1)
    cooldown: int = Field(default=100, ge=0)
    min_samples: int = Field(default=100, ge=1)
    min_window: int = Field(default=50, ge=1)
    action: Literal["reset_retrain", "learn_one"] = "reset_retrain"
    params: dict = Field(default_factory=dict, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def validate_builtin_policy(self):
        if self.name in BATCH_POLICY_KINDS:
            raise ValueError(f"policy {self.name} decides per batch from a performance matrix; "
                             "run it through the batch-prequential Zia pipeline")
        if self.name in POLICIES and self.params:
            PolicyStrategy(kind=self.name, **self.model_dump(exclude={"name"}))
        return self


class PromotionConfig(Strict):
    """Past-label temporal holdout; candidates never learn its validation suffix."""

    validation_samples: int = Field(default=32, ge=1, le=1000)
    min_improvement: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)


class AwsCostConfig(Strict):
    """Pinned public list prices; resource allocation and retention are scenario inputs."""

    region: str = Field(default="us-east-1", min_length=1)
    price_date: str = Field(default="2026-11-08", min_length=1)
    compute_service: Literal["fargate-linux-x86"] = "fargate-linux-x86"
    storage_class: Literal["s3-standard"] = "s3-standard"
    vcpus: float = Field(default=1, gt=0, allow_inf_nan=False)
    memory_gb: float = Field(default=2, gt=0, allow_inf_nan=False)
    vcpu_hour_usd: float = Field(default=0.04048, ge=0, allow_inf_nan=False)
    memory_gb_hour_usd: float = Field(default=0.004445, ge=0, allow_inf_nan=False)
    storage_gb_month_usd: float = Field(default=0.023, ge=0, allow_inf_nan=False)
    retention_days: float = Field(default=30, ge=0, allow_inf_nan=False)
    billing_month_days: float = Field(default=30, gt=0, allow_inf_nan=False)
    minimum_compute_seconds: int = Field(default=60, ge=60)
    compute_source: str = (
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonECS/current/us-east-1/index.json"
    )
    storage_source: str = (
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-1/index.json"
    )

    @model_validator(mode="before")
    @classmethod
    def explicit_regional_prices(cls, value):
        if isinstance(value, dict) and value.get("region", "us-east-1") != "us-east-1":
            required = {"vcpu_hour_usd", "memory_gb_hour_usd", "storage_gb_month_usd",
                        "compute_source", "storage_source", "price_date"}
            if not required <= value.keys():
                raise ValueError("another AWS region requires explicit prices, sources and price_date")
        return value

    @model_validator(mode="after")
    def supported_allocation(self):
        options = {0.25: {0.5, 1, 2}, 0.5: set(range(1, 5)), 1: set(range(2, 9)),
                   2: set(range(4, 17)), 4: set(range(8, 31)), 8: set(range(16, 61, 4)),
                   16: set(range(32, 121, 8)), 32: {60, 120, 244}}
        if self.memory_gb not in options.get(self.vcpus, set()):
            raise ValueError("unsupported Fargate vCPU/memory allocation")
        return self


class CostConfig(Strict):
    aws: AwsCostConfig | None = Field(default_factory=AwsCostConfig)
    unit: str = Field(default="scenario_unit", min_length=1, max_length=80)
    false_positive: float = Field(default=1, ge=0, allow_inf_nan=False)
    false_negative: float = Field(default=1, ge=0, allow_inf_nan=False)
    classification_error: float = Field(default=1, ge=0, allow_inf_nan=False)
    regression_error: float = Field(default=1, ge=0, allow_inf_nan=False)
    regression_scale: float = Field(default=1, gt=0, allow_inf_nan=False)
    train_sample: float = Field(default=0, ge=0, allow_inf_nan=False)
    update: float = Field(default=0, ge=0, allow_inf_nan=False)
    train_second: float = Field(default=0, ge=0, allow_inf_nan=False)
    predict_second: float = Field(default=0, ge=0, allow_inf_nan=False)


class FutureEvaluationConfig(Strict):
    horizon: int = Field(default=32, ge=1, le=1024)
    gap: int = Field(default=1, ge=0)
    max_windows: int = Field(default=4, ge=1, le=64)


class TrackingConfig(Strict):
    version: Literal["residency-metrics-v1"] = "residency-metrics-v1"
    fading_factor: float = Field(default=0.99, gt=0, lt=1, allow_inf_nan=False)
    detection_window: int = Field(default=500, ge=1)
    recovery_tolerance: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    recovery_windows: int = Field(default=2, ge=1)
    target_accuracy: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)
    probe_samples: int = Field(default=64, ge=1, le=1024)
    continual_references: bool = True
    reference_budget: int | None = Field(default=None, ge=0)
    adaptation_shadows: bool = False
    costs: CostConfig = Field(default_factory=lambda: CostConfig(train_sample=0.001, update=0.05))
    upf_alpha: float = Field(default=1, ge=0, allow_inf_nan=False)
    upf_alphas: tuple[float, ...] = (0, 0.1, 1, 10)
    reference: dict | None = None
    task_evaluation: FutureEvaluationConfig | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def valid_alphas(self):
        import math

        if (len(self.upf_alphas) < 2 or len(self.upf_alphas) > 100
                or any(not math.isfinite(a) or a < 0 for a in self.upf_alphas)
                or tuple(sorted(set(self.upf_alphas))) != self.upf_alphas):
            raise ValueError("upf_alphas must be finite, nonnegative, unique and increasing")
        return self


# Applied when a run omits `tracking`: every run records every metric family by
# default, including adaptation shadows and frozen near-future windows.
DEFAULT_TRACKING = TrackingConfig(adaptation_shadows=True, task_evaluation=FutureEvaluationConfig())


class Config(Strict):
    protocol_version: Literal[1] = 1
    name: str = Field(default="baseline", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    mode: Literal["streaming", "incremental", "offline"] = "streaming"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: Component = Field(default_factory=lambda: Component(name="logistic"))
    detector: Component = Field(default_factory=lambda: Component(name="adwin"))
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    warmup: int = Field(default=100, ge=1)
    replay_window: int = Field(default=256, ge=1, le=10_000)
    metric_window: int = Field(default=100, ge=1, le=10_000)
    batch_size: int = Field(default=32, ge=1, le=1000)
    checkpoint_every: int = Field(default=100, ge=1, le=10_000)
    offline_train_fraction: float = Field(default=0.6, gt=0, lt=1)
    reset_detector_on_retrain: bool = True
    max_seconds: int = Field(default=600, ge=10, le=86_400)
    training_budget: int | None = Field(default=None, ge=0)
    plugins: tuple[str, ...] = ()
    plugin_packages: tuple[str, ...] = ()
    # Keep identities/config dumps of historical configurations byte-compatible.
    tracking: TrackingConfig | None = Field(default=None, exclude_if=lambda value: value is None)
    promotion: PromotionConfig | None = Field(default=None, exclude_if=lambda value: value is None)
    # Groups a campaign in one MLflow experiment; None keeps the per-policy/detector experiments.
    mlflow_experiment: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_./-]{1,200}$",
                                          exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_protocol(self):
        if self.promotion is not None:
            if (self.mode == "offline" or self.policy.action != "reset_retrain"
                    or self.policy.name in {"none", "online"}):
                raise ValueError("promotion gate requires a prequential reset_retrain policy")
            if self.promotion.validation_samples + self.policy.min_window > self.replay_window:
                raise ValueError("promotion must leave min_window training samples in replay")
        if self.warmup >= self.dataset.samples:
            raise ValueError("warmup must leave evaluation samples")
        if self.policy.min_window > self.replay_window:
            raise ValueError("min_window exceeds replay_window")
        if self.mode == "offline" and self.policy.name != "none":
            raise ValueError("offline is a frozen temporal holdout: policy must be none")
        self.validate_components()
        if (
            self.mode == "offline"
            and _dataset_family(self.dataset.name) == "river_abrupt"
            and self.dataset.label_delay >= int(self.dataset.samples * self.offline_train_fraction)
        ):
            raise ValueError("offline label delay leaves no training labels available before test")
        if self.policy.name == "online" and self.policy.action != "learn_one":
            raise ValueError("online policy must explicitly use learn_one")
        if (self.policy.name == "online" and self.training_budget is not None
                and self.training_budget < self.dataset.samples - self.warmup):
            raise ValueError("online budget must cover every post-warmup label")
        if self.mode == "offline" and self.training_budget is not None:
            raise ValueError("training_budget controls post-warmup prequential learning only")
        for plugin in self.plugins:
            if not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", plugin):
                raise ValueError("plugin must be module:function")
        if self.plugin_packages and not self.plugins:
            raise ValueError("plugin_packages requires at least one plugin")
        for package in self.plugin_packages:
            if not re.fullmatch(r"[A-Za-z_][\w.]*", package):
                raise ValueError("plugin_packages must contain importable package names")
        return self

    def tracking_settings(self) -> TrackingConfig:
        """Explicit tracking wins; otherwise the full default profile."""
        return self.tracking if self.tracking is not None else DEFAULT_TRACKING

    def validate_components(self):
        """Built-in catalog compatibility; plugin components are checked when instantiated."""
        from continual_bench.datasets.catalog import CATALOG
        from continual_bench.models import MODELS

        task = CATALOG.get(self.dataset.name).task if self.dataset.name in CATALOG else None
        if task not in (None, BINARY):
            raise ValueError(f"dataset {self.dataset.name} has {task} targets; the drift/retraining "
                             "engine is binary, use the typed or epoch evaluator")
        if self.model.name in MODELS and not MODELS.get(self.model.name).supports(BINARY):
            raise ValueError(f"model {self.model.name} does not support binary targets")

    def identity(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:24]


def _dataset_family(name: str) -> str | None:
    from continual_bench.datasets.catalog import CATALOG

    return CATALOG.get(name).family if name in CATALOG else None


def load_config(path: str | Path) -> Config:
    """Load a run configuration, resolving `extends` bases (see config.loader)."""
    from .loader import load_model

    return load_model(Config, path)
