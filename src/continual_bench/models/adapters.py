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


def _is_nonneg_int_classes(classes: Any) -> bool:
    arr = np.asarray(classes)
    return bool(arr.size and np.issubdtype(arr.dtype, np.integer) and int(arr.min()) >= 0)


def _align_class_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Align probability columns to contiguous ``0..max_class`` indices (width >= 2)."""
    if proba.ndim == 1:
        return np.column_stack([1.0 - proba, proba])
    classes_arr = np.asarray(classes)
    if _is_nonneg_int_classes(classes_arr):
        width = max(2, int(classes_arr.max()) + 1)
        if proba.shape[1] == width and np.array_equal(
            classes_arr, np.arange(width, dtype=classes_arr.dtype)
        ):
            return proba
        out = np.zeros((proba.shape[0], width), dtype=np.float64)
        out[:, classes_arr.astype(np.int64)] = proba
        return out
    return proba


def _ensure_binary_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Ensure binary probability output has shape (n_samples, 2) for classes [0, 1]."""
    return _align_class_proba(proba, classes)


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
        proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        return _align_class_proba(proba, self.model.classes_)

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
        self._num_class: int = max(2, int(self.params.get("num_class", 2)))
        self._seen_classes: np.ndarray = np.array([0, 1], dtype=np.int64)
        self._is_multiclass: bool = str(self.params.get("objective", "")).startswith("multi:")

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
        if unique.size == 0:
            raise ValueError("Cannot fit XGBoostAdapter on empty target array")

        train_params = self._train_params()
        objective = str(train_params.get("objective", "binary:logistic"))
        is_multiclass = (
            objective.startswith("multi:")
            or unique.size > 2
            or int(unique.max()) > 1
            or int(unique.min()) < 0
        )
        self._is_multiclass = is_multiclass
        self._seen_classes = unique.astype(np.int64)

        if is_multiclass:
            inferred_num_class = int(unique.max()) + 1 if int(unique.min()) >= 0 else int(unique.size)
            self._num_class = max(2, inferred_num_class, int(self.params.get("num_class", 2)))
            if unique.size == 1:
                self._single_class = int(unique[0])
                self._booster = None
                return self
            self._single_class = None
            if objective == "binary:logistic":
                train_params["objective"] = "multi:softprob"
            contiguous = np.arange(unique.size, dtype=np.int64)
            y_train = (
                y_arr
                if np.array_equal(unique, contiguous)
                else np.searchsorted(unique, y_arr).astype(np.int64)
            )
            train_params["num_class"] = int(unique.size)
            dtrain = xgb.DMatrix(X, label=y_train)
            self._booster = xgb.train(train_params, dtrain, num_boost_round=self.num_boost_round)
            return self

        self._num_class = max(2, int(self.params.get("num_class", 2)))
        if unique.size == 1:
            self._single_class = int(unique[0])
            self._booster = None
            return self
        self._single_class = None
        dtrain = xgb.DMatrix(X, label=y_arr)
        self._booster = xgb.train(train_params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def _raw_predict(self, X: Any) -> np.ndarray:
        import xgboost as xgb

        dmat = xgb.DMatrix(X)
        return np.asarray(self._booster.predict(dmat), dtype=np.float64)

    def predict_proba(self, X: Any) -> np.ndarray:
        n_rows = X.shape[0]
        if self._single_class is not None:
            width = self._num_class if int(self._single_class) >= 0 else 2
            out = np.zeros((n_rows, width), dtype=np.float64)
            col = int(self._single_class) if 0 <= int(self._single_class) < width else 0
            out[:, col] = 1.0
            return out
        if self._booster is None:
            raise RuntimeError("XGBoostAdapter must be fitted before calling predict_proba")

        preds = self._raw_predict(X)
        if not self._is_multiclass:
            if preds.ndim == 1:
                return np.column_stack([1.0 - preds, preds])
            return preds

        if preds.ndim == 1:
            # Handle multi:softmax explicit objective gracefully
            one_hot = np.zeros((n_rows, len(self._seen_classes)), dtype=np.float64)
            if n_rows > 0:
                idx = np.clip(preds.astype(np.int64), 0, len(self._seen_classes) - 1)
                one_hot[np.arange(n_rows), idx] = 1.0
            preds = one_hot

        if int(self._seen_classes.min()) >= 0:
            if preds.shape[1] == self._num_class and np.array_equal(
                self._seen_classes, np.arange(self._num_class, dtype=np.int64)
            ):
                return preds
            out = np.zeros((n_rows, self._num_class), dtype=np.float64)
            out[:, self._seen_classes] = preds
            return out
        return preds

    def predict(self, X: Any) -> np.ndarray:
        if self._single_class is not None:
            return np.full(X.shape[0], self._single_class, dtype=np.int64)
        proba = self.predict_proba(X)
        if self._is_multiclass and int(self._seen_classes.min()) < 0:
            return self._seen_classes[np.argmax(proba, axis=1)].astype(np.int64)
        if proba.shape[1] == 2 and not self._is_multiclass:
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
        self._single_class: int | None = None
        self._num_class: int = 2

    def fit(self, X: Any, y: np.ndarray) -> "SklearnAdapter":
        y_arr = np.asarray(y, dtype=np.int64)
        unique = np.unique(y_arr)
        if unique.size == 0:
            raise ValueError("Cannot fit SklearnAdapter on empty target array")
        if unique.size == 1:
            self._single_class = int(unique[0])
            self._num_class = max(2, int(unique[0]) + 1 if int(unique[0]) >= 0 else 2)
            return self
        self._single_class = None
        self.model.fit(X, y_arr)
        classes = getattr(self.model, "classes_", unique)
        if _is_nonneg_int_classes(classes):
            self._num_class = max(2, int(np.max(classes)) + 1)
        else:
            self._num_class = max(2, len(classes))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n_rows = X.shape[0]
        if self._single_class is not None:
            out = np.zeros((n_rows, self._num_class), dtype=np.float64)
            if 0 <= self._single_class < self._num_class:
                out[:, self._single_class] = 1.0
            return out
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
            classes = getattr(self.model, "classes_", np.arange(proba.shape[1]))
            return _align_class_proba(proba, classes)
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X), dtype=np.float64)
            if scores.ndim == 1:
                pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
                proba = np.column_stack([1.0 - pos, pos])
            else:
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            classes = getattr(self.model, "classes_", np.arange(proba.shape[1]))
            return _align_class_proba(proba, classes)
        preds = self.predict(X)
        classes = getattr(self.model, "classes_", np.unique(preds))
        if _is_nonneg_int_classes(classes):
            n_classes = max(2, int(np.max(classes)) + 1, (int(np.max(preds)) + 1) if len(preds) else 2)
        else:
            n_classes = max(2, int(np.max(preds)) + 1) if len(preds) else 2
        out = np.zeros((len(preds), n_classes), dtype=np.float64)
        if len(preds):
            out[np.arange(len(preds)), preds] = 1.0
        return out

    def predict(self, X: Any) -> np.ndarray:
        if self._single_class is not None:
            return np.full(X.shape[0], self._single_class, dtype=np.int64)
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
