import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class CostConfig(Strict):
    unit: str = Field(default="scenario_unit", min_length=1, max_length=80)
    false_positive: float = Field(default=1, ge=0, allow_inf_nan=False)
    false_negative: float = Field(default=1, ge=0, allow_inf_nan=False)
    train_sample: float = Field(default=0, ge=0, allow_inf_nan=False)
    update: float = Field(default=0, ge=0, allow_inf_nan=False)
    train_second: float = Field(default=0, ge=0, allow_inf_nan=False)
    predict_second: float = Field(default=0, ge=0, allow_inf_nan=False)


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
    costs: CostConfig | None = None
    upf_alpha: float = Field(default=1, ge=0, allow_inf_nan=False)
    upf_alphas: tuple[float, ...] = (0, 0.1, 1, 10)
    reference: dict | None = None

    @model_validator(mode="after")
    def valid_alphas(self):
        import math

        if (len(self.upf_alphas) < 2 or len(self.upf_alphas) > 100
                or any(not math.isfinite(a) or a < 0 for a in self.upf_alphas)
                or tuple(sorted(set(self.upf_alphas))) != self.upf_alphas):
            raise ValueError("upf_alphas must be finite, nonnegative, unique and increasing")
        return self


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

    @model_validator(mode="after")
    def validate_protocol(self):
        if self.warmup >= self.dataset.samples:
            raise ValueError("warmup must leave evaluation samples")
        if self.policy.min_window > self.replay_window:
            raise ValueError("min_window exceeds replay_window")
        if self.mode == "offline" and self.policy.name != "none":
            raise ValueError("offline is a frozen temporal holdout: policy must be none")
        if (
            self.mode == "offline"
            and self.dataset.name in {"sea", "agrawal", "stagger"}
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

    def identity(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:24]


def load_config(path: str | Path) -> Config:
    with Path(path).open() as f:
        return Config.model_validate(yaml.safe_load(f))
