"""Base abstractions and configuration models for benchmark protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from continual_bench.backends.base import ExecutionBackend
from continual_bench.datasets.streams_unified import StreamDatasetSpec
from continual_bench.models.adapters import ModelConfig
from continual_bench.tracking.tracker import BenchmarkTracker


class CampaignConfig(BaseModel):
    """Configuration for a reproducible continual-learning benchmark campaign."""

    name: str
    protocol: Literal["cara", "upf"]
    dataset: StreamDatasetSpec
    model: ModelConfig
    seeds: list[int]
    policies: list[str]
    kappa_grid: list[float] | None = None
    alpha_grid: list[float] | None = None
    regimes: dict[str, float] | None = None
    staleness_mode: Literal["relative", "absolute"] = "relative"
    kernel_gamma: float | None = None
    upf_delta: float = 0.95
    upf_mc_samples: int = 100
    platform: str | None = None
    output_dir: str | None = None
    tracking_uri: str | None = None
    save_model_checkpoints: bool | None = None
    max_workers: int = 1
    reference_csv: str = "configs/contracts/published-reference-values.csv"


@dataclass
class ProtocolResult:
    """Structured output returned by a benchmark protocol execution."""

    protocol: str
    dataset: str
    seed_summaries: list[dict[str, Any]] = field(default_factory=list)
    regime_records: list[dict[str, Any]] = field(default_factory=list)
    aggregate_summary: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    mlflow_receipt: dict[str, Any] | None = None


class BenchmarkProtocol(ABC):
    """Abstract base class for continual-learning benchmark protocols."""

    @abstractmethod
    def run(
        self,
        config: CampaignConfig,
        backend: ExecutionBackend,
        tracker: BenchmarkTracker,
    ) -> ProtocolResult:
        """Execute the benchmark protocol and return structured results."""
