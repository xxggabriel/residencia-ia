"""Admission of bounded repeated-pass evaluation over any catalog dataset."""

from pydantic import Field

from .benchmark import Component, CostConfig, Strict


class EpochConfig(Strict):
    costs: CostConfig = Field(default_factory=lambda: CostConfig(train_sample=0.001, update=0.05))
    dataset: str
    examples: int = Field(default=100, ge=10, le=1000)
    epochs: int = Field(default=2, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    max_seconds: int = Field(default=300, ge=10, le=3600)
    # None keeps the historical per-task default and the historical config dump.
    model: Component | None = Field(default=None, exclude_if=lambda value: value is None)
