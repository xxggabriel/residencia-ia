"""Model adapters and registry for unified batch/stream continual learning."""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field
from sklearn.base import clone as sklearn_clone
from sklearn.ensemble import RandomForestClassifier

from continual_bench.core.registry import resolve


class ModelConfig(BaseModel):
    """Configuration for building a ModelAdapter."""

    name: Literal["random_forest", "xgboost", "sklearn"] | str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42


@runtime_checkable
class ModelAdapter(Protocol):
    """Uniform interface for supervised models used in continual-bench."""

    def fit(self, X: Any, y: np.ndarray) -> "ModelAdapter": ...

    def predict(self, X: Any) -> np.ndarray: ...

    def predict_proba(self, X: Any) -> np.ndarray: ...

    def clone(self, seed: int | None = None) -> "ModelAdapter": ...


def _ensure_binary_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Ensure binary probability output has shape (n_samples, 2) for classes [0, 1]."""
    if proba.ndim == 1:
        return np.column_stack([1.0 - proba, proba])
    if proba.shape[1] == 2 and list(classes) == [0, 1]:
        return proba
    out = np.zeros((proba.shape[0], 2), dtype=np.float64)
    for col_idx, cls in enumerate(classes):
        idx = int(cls)
        if idx in (0, 1):
            out[:, idx] = proba[:, col_idx]
    return out


class RandomForestAdapter:
    """Adapter for scikit-learn's RandomForestClassifier (dense or sparse)."""

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.params = dict(params or {})
        self.seed = int(seed)
        merged = {"n_estimators": 100, "n_jobs": 1, **self.params, "random_state": self.seed}
        self.model = RandomForestClassifier(**merged)

    def fit(self, X: Any, y: np.ndarray) -> "RandomForestAdapter":
        y_arr = np.asarray(y, dtype=np.int64)
        self.model.fit(X, y_arr)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if len(self.model.classes_) <= 2 and set(self.model.classes_) <= {0, 1}:
            return _ensure_binary_proba(proba, self.model.classes_)
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.int64)

    def clone(self, seed: int | None = None) -> "RandomForestAdapter":
        return RandomForestAdapter(params=self.params, seed=self.seed if seed is None else seed)


class XGBoostAdapter:
    """Adapter around ``xgboost.train`` for fast binary/multiclass gradient boosting."""

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.params = dict(params or {})
        self.seed = int(seed)
        self.num_boost_round = int(self.params.get("num_boost_round", 100))
        self._booster: Any = None
        self._single_class: int | None = None

    def _train_params(self) -> dict[str, Any]:
        train_params = {k: v for k, v in self.params.items() if k != "num_boost_round"}
        train_params.setdefault("eta", 0.3)
        train_params.setdefault("objective", "binary:logistic")
        train_params.setdefault("nthread", 2)
        train_params["seed"] = self.seed
        return train_params

    def fit(self, X: Any, y: np.ndarray) -> "XGBoostAdapter":
        import xgboost as xgb

        y_arr = np.asarray(y, dtype=np.int64)
        unique = np.unique(y_arr)
        if len(unique) == 1 and self._train_params().get("objective", "binary:logistic") == "binary:logistic":
            self._single_class = int(unique[0])
            self._booster = None
            return self
        self._single_class = None
        dtrain = xgb.DMatrix(X, label=y_arr)
        self._booster = xgb.train(self._train_params(), dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n_rows = X.shape[0]
        if self._single_class is not None:
            out = np.zeros((n_rows, 2), dtype=np.float64)
            out[:, self._single_class] = 1.0
            return out
        if self._booster is None:
            raise RuntimeError("XGBoostAdapter must be fitted before calling predict_proba")
        import xgboost as xgb

        dmat = xgb.DMatrix(X)
        preds = np.asarray(self._booster.predict(dmat), dtype=np.float64)
        if preds.ndim == 1:
            return np.column_stack([1.0 - preds, preds])
        return preds

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= 0.5).astype(np.int64)
        return np.argmax(proba, axis=1).astype(np.int64)

    def clone(self, seed: int | None = None) -> "XGBoostAdapter":
        return XGBoostAdapter(params=self.params, seed=self.seed if seed is None else seed)


class SklearnAdapter:
    """Generic wrapper for any scikit-learn classifier."""

    def __init__(
        self,
        estimator: Any = None,
        params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        self.params = dict(params or {})
        self.seed = int(seed)

        if estimator is None:
            estimator = self.params.pop("estimator", None)
        if isinstance(estimator, str):
            estimator_cls = resolve(estimator)
            init_params = {k: v for k, v in self.params.items()}
            self._base_estimator = estimator_cls(**init_params)
        elif isinstance(estimator, type):
            self._base_estimator = estimator(**self.params)
        elif estimator is not None:
            self._base_estimator = sklearn_clone(estimator)
            if self.params:
                self._base_estimator.set_params(**self.params)
        else:
            from sklearn.linear_model import LogisticRegression

            self._base_estimator = LogisticRegression(**self.params)

        self.model = sklearn_clone(self._base_estimator)
        if "random_state" in self.model.get_params():
            self.model.set_params(random_state=self.seed)

    def fit(self, X: Any, y: np.ndarray) -> "SklearnAdapter":
        y_arr = np.asarray(y, dtype=np.int64)
        self.model.fit(X, y_arr)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
            classes = getattr(self.model, "classes_", np.arange(proba.shape[1]))
            if len(classes) <= 2 and set(classes) <= {0, 1}:
                return _ensure_binary_proba(proba, classes)
            return proba
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X), dtype=np.float64)
            if scores.ndim == 1:
                pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
                return np.column_stack([1.0 - pos, pos])
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        preds = self.predict(X)
        n_classes = int(max(2, np.max(preds) + 1)) if len(preds) else 2
        out = np.zeros((len(preds), n_classes), dtype=np.float64)
        out[np.arange(len(preds)), preds] = 1.0
        return out

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.int64)

    def clone(self, seed: int | None = None) -> "SklearnAdapter":
        return SklearnAdapter(
            estimator=self._base_estimator,
            params={},
            seed=self.seed if seed is None else seed,
        )


MODEL_ADAPTERS: dict[str, type | Callable[..., ModelAdapter]] = {
    "random_forest": RandomForestAdapter,
    "xgboost": XGBoostAdapter,
    "sklearn": SklearnAdapter,
}


def register_model_adapter(name: str, cls: type | Callable[..., ModelAdapter]) -> None:
    """Register a custom ModelAdapter class or factory under ``name``."""
    MODEL_ADAPTERS[name] = cls


def build_model_adapter(config: ModelConfig) -> ModelAdapter:
    """Instantiate a ModelAdapter from a ``ModelConfig``."""
    if config.name not in MODEL_ADAPTERS:
        raise ValueError(
            f"unknown model adapter {config.name!r}; available={sorted(MODEL_ADAPTERS)}"
        )
    factory = MODEL_ADAPTERS[config.name]
    return factory(params=dict(config.params), seed=config.seed)
