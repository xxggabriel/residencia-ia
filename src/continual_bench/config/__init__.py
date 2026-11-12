"""Declarative, validated benchmark configuration (Pydantic)."""

from .benchmark import (
    DEFAULT_TRACKING,
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
from .iterative import IterativeConfig, IterativeHyperparameters, IterativeTrainingConfig
from .loader import deep_merge, load_document, load_model
from .typed import TypedConfig

__all__ = [
    "Component", "Config", "DEFAULT_TRACKING", "CostConfig", "DatasetConfig", "FutureEvaluationConfig",
    "IterativeConfig", "IterativeHyperparameters", "IterativeTrainingConfig", "PolicyConfig",
    "PromotionConfig", "Strict", "TrackingConfig", "load_config", "EpochConfig", "TypedConfig",
    "deep_merge", "load_document", "load_model",
]
