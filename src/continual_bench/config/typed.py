"""Declarative admission for the bounded regression/multiclass evaluator."""

from typing import Literal

from pydantic import Field, model_validator

from .benchmark import Component, CostConfig, DatasetConfig, Strict


class TypedConfig(Strict):
    costs: CostConfig = Field(default_factory=lambda: CostConfig(train_sample=0.001, update=0.05))
    dataset: DatasetConfig
    policy: Literal["none", "online", "periodic"] = "online"
    warmup: int = Field(default=20, ge=1, le=10_000)
    every: int = Field(default=50, ge=1, le=10_000)
    window: int = Field(default=128, ge=1, le=10_000)
    max_samples: int = Field(default=1000, ge=2, le=10_000)
    max_seconds: int = Field(default=60, ge=1, le=600)
    training_budget: int | None = Field(default=None, ge=0)
    # Model family for the dataset's target type; None selects the task default.
    model: Component | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def useful_prefix(self):
        if self.warmup >= min(self.max_samples, self.dataset.samples):
            raise ValueError("typed warmup must leave evaluation samples")
        return self
