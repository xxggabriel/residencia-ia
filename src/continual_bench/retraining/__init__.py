"""Cost-aware model-update policies: CARA, UPF and the retrospective Oracle."""

from .forecaster import ElasticNetCV
from .matrix import Learner, MatrixView, PerformanceMatrix, build_matrix, build_paper_matrix
from .pricing import (
    DEFAULT_FARGATE_PRICING,
    fargate_hourly_rate_usd,
    fargate_retrain_cost_usd,
    normalize_fargate_config,
    resolve_retrain_alpha,
    summarize_fargate_retraining_cost,
)
from .rules import RULES, UPF, CaraCT, CaraP, CaraT, Frozen, Markov, Oracle, Rule, StalenessOracle
from .schedule import (
    alpha_for_retrain_bound,
    alpha_max,
    lipschitz_bound,
    min_error_by_retrains,
    optimal_schedule,
    retrain_upper_bound,
    schedule_cost,
)
from .simulate import SimResult, simulate

__all__ = [
    "DEFAULT_FARGATE_PRICING",
    "RULES",
    "UPF",
    "CaraCT",
    "CaraP",
    "CaraT",
    "ElasticNetCV",
    "Frozen",
    "Learner",
    "Markov",
    "MatrixView",
    "Oracle",
    "StalenessOracle",
    "PerformanceMatrix",
    "Rule",
    "SimResult",
    "alpha_for_retrain_bound",
    "alpha_max",
    "build_matrix",
    "build_paper_matrix",
    "fargate_hourly_rate_usd",
    "fargate_retrain_cost_usd",
    "lipschitz_bound",
    "min_error_by_retrains",
    "normalize_fargate_config",
    "optimal_schedule",
    "resolve_retrain_alpha",
    "retrain_upper_bound",
    "schedule_cost",
    "simulate",
    "summarize_fargate_retraining_cost",
]
