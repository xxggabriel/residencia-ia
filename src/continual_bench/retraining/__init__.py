"""Cost-aware model-update policies: CARA, UPF and the retrospective Oracle."""

from .forecaster import ElasticNetCV
from .matrix import Learner, MatrixView, PerformanceMatrix, build_matrix, build_paper_matrix
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
    "RULES", "UPF", "CaraCT", "CaraP", "CaraT", "ElasticNetCV", "Frozen", "Learner", "Markov",
    "MatrixView", "Oracle", "StalenessOracle", "PerformanceMatrix", "Rule", "SimResult",
    "alpha_for_retrain_bound", "alpha_max",
    "build_matrix", "build_paper_matrix", "lipschitz_bound", "min_error_by_retrains", "optimal_schedule",
    "retrain_upper_bound", "schedule_cost", "simulate",
]
