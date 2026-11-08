"""Declarative, validated benchmark configuration (Pydantic)."""

from .benchmark import (
    Component,
    Config,
    CostConfig,
    DatasetConfig,
    FutureEvaluationConfig,
    PolicyConfig,
    PromotionConfig,
    Strict,
    TrackingConfig,
    load_config,
)
from .epochs import EpochConfig
from .loader import deep_merge, load_document, load_model
from .typed import TypedConfig

__all__ = [
    "Component", "Config", "CostConfig", "DatasetConfig", "FutureEvaluationConfig", "PolicyConfig",
    "PromotionConfig", "Strict", "TrackingConfig", "load_config", "EpochConfig", "TypedConfig",
    "deep_merge", "load_document", "load_model",
]
