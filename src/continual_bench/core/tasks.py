"""Target-type strategies shared by every evaluator, dataset and model family.

Each task kind owns the rules that used to be scattered as `if task == ...`
branches: target admission, prediction normalization and scenario error units.
"""

import math
from typing import Any

BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
UNTRAINED = "__untrained__"  # Reserved class emitted by a classifier with no learned prior.


class TaskKind:
    name: str
    classification: bool

    def validate_target(self, y: Any) -> None:
        raise NotImplementedError

    def validate_features(self, x: dict) -> None:
        """Feature contract for streams consumed natively by this task's learners."""

    def normalize_prediction(self, prediction: Any) -> Any:
        return prediction

    def error_units(self, target: Any, prediction: Any, costs: dict) -> float:
        raise NotImplementedError


class Binary(TaskKind):
    name, classification = BINARY, True

    def validate_target(self, y):
        if type(y) is not bool:
            raise ValueError("binary targets must be explicit booleans")

    def error_units(self, target, prediction, costs):
        return float(target != prediction) * costs["classification_error"]


class Multiclass(TaskKind):
    name, classification = MULTICLASS, True

    def validate_target(self, y):
        if not isinstance(y, str) or not y or y == UNTRAINED:
            raise ValueError("classification target must be an original class string")

    def normalize_prediction(self, prediction):
        return UNTRAINED if prediction is None else prediction

    def error_units(self, target, prediction, costs):
        return float(target != prediction) * costs["classification_error"]


class Regression(TaskKind):
    name, classification = REGRESSION, False

    def validate_target(self, y):
        if type(y) not in (int, float) or not math.isfinite(y):
            raise ValueError("regression target must be finite numeric")

    def validate_features(self, x):
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in x.values()):
            raise ValueError("regression targets and features must be finite numbers")

    def normalize_prediction(self, prediction):
        if prediction is None or not math.isfinite(prediction):
            raise ValueError("regression model emitted a nonfinite prediction")
        return prediction

    def error_units(self, target, prediction, costs):
        return abs(target - prediction) / costs["regression_scale"] * costs["regression_error"]


TASKS = {kind.name: kind for kind in (Binary(), Multiclass(), Regression())}


def task_kind(name: str) -> TaskKind:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name}; available={sorted(TASKS)}") from None
