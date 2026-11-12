"""One CPU learner per Zia task, with iterative epoch training, early stopping and loadable model state."""

from __future__ import annotations

import gc
import io
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from continual_bench.config.iterative import IterativeTrainingConfig
from continual_bench.models.training.loop import (
    TrainingResult,
    enforce_single_thread_env,
    train_iterative,
    unpack_split,
)

enforce_single_thread_env()

_XGB_ALLOWED_PARAMS = {
    "objective",
    "tree_method",
    "max_depth",
    "max_bin",
    "eta",
    "reg_lambda",
    "reg_alpha",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "nthread",
    "seed",
    "eval_metric",
    "verbosity",
}


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: int,
) -> dict[str, float]:
    """Compute log_loss, accuracy, and f1_macro without allocating large histograms."""
    total = int(len(labels))
    if total == 0:
        return {"log_loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0}
    labels_i32 = labels.astype(np.int32, copy=False)
    predicted = (
        (probabilities[:, 1] >= 0.5).astype(np.int32)
        if classes == 2
        else probabilities.argmax(axis=1).astype(np.int32)
    )
    correct = int(np.sum(predicted == labels_i32))
    accuracy = float(correct / total)

    observed = probabilities[np.arange(total), labels_i32].astype(np.float64, copy=False)
    log_loss = float(-np.log(np.clip(observed, 1e-15, 1.0)).mean())

    confusion = np.bincount(
        classes * labels_i32 + predicted,
        minlength=classes * classes,
    ).reshape((classes, classes))
    actual = confusion.sum(axis=1)
    pred_counts = confusion.sum(axis=0)
    diag = confusion.diagonal()
    f1_values = [
        float(2 * diag[i] / (pred_counts[i] + actual[i]))
        for i in range(classes)
        if (pred_counts[i] + actual[i]) > 0
    ]
    f1_macro = float(sum(f1_values) / len(f1_values)) if f1_values else 0.0
    return {
        "log_loss": log_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


class FraudOrReturnBooster:
    name = "xgboost"
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": 5,
        "max_bin": 128,
        "eta": 0.01,
        "reg_lambda": 0.001,
        "nthread": 1,
        "seed": 42,
    }
    initial_rounds = 100
    continued_rounds = 100

    def __init__(self, config: IterativeTrainingConfig | dict[str, Any] | None = None):
        enforce_single_thread_env()
        import xgboost

        self.xgb = xgboost
        self.model = None
        self.params = dict(self.params)
        if config is not None:
            cfg = IterativeTrainingConfig.from_params(config)
            self.params["eta"] = cfg.learning_rate
            self.params["reg_lambda"] = cfg.l2_regularization
            self.params["seed"] = cfg.seed
            self.initial_rounds = cfg.max_epochs
            self.continued_rounds = cfg.max_epochs
        self.params["nthread"] = 1
        self._cached_dmatrix = None
        self._cached_dmatrix_key: tuple[int, tuple[int, ...], int, tuple[int, ...]] | None = None
        self.last_training_result: TrainingResult | None = None

    def _xgb_params(self) -> dict[str, Any]:
        clean = {k: v for k, v in self.params.items() if k in _XGB_ALLOWED_PARAMS}
        if "learning_rate" in self.params and "eta" not in clean:
            clean["eta"] = float(self.params["learning_rate"])
        if "l2_regularization" in self.params and "reg_lambda" not in clean:
            clean["reg_lambda"] = float(self.params["l2_regularization"])
        elif "l2" in self.params and "reg_lambda" not in clean:
            clean["reg_lambda"] = float(self.params["l2"])
        clean["nthread"] = 1
        clean.setdefault("verbosity", 0)
        return clean

    def _dmatrix(self, x: np.ndarray, y: np.ndarray):
        key = (id(x), x.shape, id(y), y.shape)
        if self._cached_dmatrix is None or self._cached_dmatrix_key != key:
            self._cached_dmatrix = None
            gc.collect()
            self._cached_dmatrix = self.xgb.QuantileDMatrix(
                x,
                label=y,
                max_bin=int(self.params.get("max_bin", 128)),
                nthread=1,
            )
            self._cached_dmatrix_key = key
        return self._cached_dmatrix

    def release_training_cache(self) -> None:
        self._cached_dmatrix = None
        self._cached_dmatrix_key = None

    def train_epoch(self, train_data: Any) -> dict[str, float]:
        x, y = unpack_split(train_data)
        matrix = self._dmatrix(x, y)
        self.model = self.xgb.train(
            self._xgb_params(),
            matrix,
            num_boost_round=1,
            xgb_model=self.model,
        )
        probabilities = self.predict(x)
        return compute_classification_metrics(y, probabilities, 2)

    def evaluate_split(self, val_data: Any) -> dict[str, float]:
        vx, vy = unpack_split(val_data)
        probabilities = self.predict(vx)
        return compute_classification_metrics(vy, probabilities, 2)

    def get_state(self) -> bytes:
        if self.model is None:
            return b""
        return bytes(self.model.save_raw(raw_format="ubj"))

    def restore_state(self, state: bytes) -> None:
        if not state:
            self.model = None
            return
        self.load(state)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        continuation: bool = False,
        *,
        val_data: Any | None = None,
        config: IterativeTrainingConfig | dict[str, Any] | None = None,
        tracker: Any | None = None,
        emit_stdout: bool = False,
    ) -> int:
        if not continuation:
            self.model = None
        rounds = self.continued_rounds if continuation else self.initial_rounds
        merged_params = {
            "max_epochs": min(100, max(1, int(rounds))),
            "patience": int(self.params.get("patience", 10)),
            "min_delta": float(self.params.get("min_delta", 1e-4)),
            "learning_rate": float(self.params.get("eta", self.params.get("learning_rate", 0.01))),
            "l2_regularization": float(
                self.params.get("reg_lambda", self.params.get("l2_regularization", 0.001))
            ),
            "validation_fraction": float(self.params.get("validation_fraction", 0.1)),
            "seed": int(self.params.get("seed", 42)),
            "restore_best_weights": bool(self.params.get("restore_best_weights", True)),
        }
        if isinstance(config, dict):
            merged_params.update(config)
            iter_cfg = IterativeTrainingConfig.from_params(merged_params)
        elif isinstance(config, IterativeTrainingConfig):
            iter_cfg = config
        else:
            iter_cfg = IterativeTrainingConfig.from_params(merged_params)

        self.params["eta"] = iter_cfg.learning_rate
        self.params["reg_lambda"] = iter_cfg.l2_regularization
        self.params["seed"] = iter_cfg.seed
        self.params["nthread"] = 1

        result = train_iterative(
            self,
            (x, y),
            val_data=val_data,
            config=iter_cfg,
            tracker=tracker,
            emit_stdout=emit_stdout,
        )
        self.last_training_result = result
        return int(self.model.num_boosted_rounds()) if self.model is not None else int(result.epochs_run)

    @property
    def training_summary(self) -> dict[str, Any]:
        return dict(self.last_training_result or {})

    @property
    def rounds(self) -> int:
        return int(self.model.num_boosted_rounds()) if self.model is not None else 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        positive = self.model.inplace_predict(x)
        return np.column_stack((1 - positive, positive))

    predict_proba = predict

    def payload(self) -> tuple[bytes, str]:
        return bytes(self.model.save_raw(raw_format="ubj")), "ubj"

    def load(self, payload: bytes) -> None:
        self.model = self.xgb.Booster(params={"nthread": 1})
        self.model.load_model(bytearray(payload))

    def explanation(self, features: list[str]) -> dict[str, dict[str, float]]:
        kinds = ("gain", "weight", "cover", "total_gain", "total_cover")
        return {
            kind: {
                features[int(key[1:])]: value
                for key, value in self.model.get_score(importance_type=kind).items()
            }
            for kind in kinds
        }


class MultinomialLogistic:
    name = "multinomial_logistic"
    params = {
        "solver": "L-BFGS-B",
        "l2": 0.001,
        "l2_regularization": 0.001,
        "learning_rate": 0.01,
        "seed": 42,
    }
    initial_rounds = 100
    continued_rounds = 100

    def __init__(
        self,
        classes: int,
        config: IterativeTrainingConfig | dict[str, Any] | None = None,
    ):
        enforce_single_thread_env()
        self.classes = int(classes)
        self.weights: np.ndarray | None = None
        self.params = dict(self.params)
        if config is not None:
            cfg = IterativeTrainingConfig.from_params(config)
            self.params["l2"] = cfg.l2_regularization
            self.params["l2_regularization"] = cfg.l2_regularization
            self.params["learning_rate"] = cfg.learning_rate
            self.params["seed"] = cfg.seed
            self.initial_rounds = cfg.max_epochs
            self.continued_rounds = cfg.max_epochs
        self.last_training_result: TrainingResult | None = None

    def _objective_and_grad(
        self,
        x: np.ndarray,
        y: np.ndarray,
        flat_weights: np.ndarray,
        shape: tuple[int, int],
        regularization: float,
    ) -> tuple[float, np.ndarray]:
        weights = flat_weights.reshape(shape)
        logits = (x @ weights[:-1]).astype(np.float64, copy=False) + weights[-1]
        logits = logits - logsumexp(logits, axis=1, keepdims=True)
        index = np.arange(len(y))
        loss = float(
            -logits[index, y].mean() + (regularization / 2.0) * np.square(weights[:-1]).sum()
        )
        residuals = np.exp(logits)
        residuals[index, y] -= 1.0
        gradient = np.empty(shape, dtype=np.float64)
        gradient[:-1] = (x.T @ residuals) / len(y) + regularization * weights[:-1]
        gradient[-1] = residuals.mean(axis=0)
        return loss, gradient.ravel()

    def train_epoch(self, train_data: Any) -> dict[str, float]:
        x, y = unpack_split(train_data)
        shape = (x.shape[1] + 1, self.classes)
        regularization = float(
            self.params.get("l2", self.params.get("l2_regularization", 0.001))
        )
        learning_rate = float(self.params.get("learning_rate", 0.01))
        solver = str(self.params.get("solver", "L-BFGS-B"))

        if self.weights is None or self.weights.shape != shape:
            self.weights = np.zeros(shape, dtype=np.float64)

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            return self._objective_and_grad(x, y, flat, shape, regularization)

        start = self.weights.ravel()
        if solver.upper() in ("SGD", "GD"):
            loss, grad = objective(start)
            if not np.isfinite(loss):
                raise ValueError("logistic optimization produced non-finite loss")
            self.weights = (start - learning_rate * grad).reshape(shape)
        else:
            result = minimize(
                objective,
                start,
                jac=True,
                method=solver,
                options={"maxiter": 2},
            )
            if not np.isfinite(result.fun) or (not result.success and result.status != 1):
                raise ValueError(f"logistic optimization failed: {result.message}")
            candidate = result.x
            if result.nit == 0:
                step_candidate = start - learning_rate * result.jac
                step_loss, _ = objective(step_candidate)
                if np.isfinite(step_loss) and step_loss < result.fun:
                    candidate = step_candidate
            self.weights = np.array(candidate, dtype=np.float64, copy=True).reshape(shape)

        probabilities = self.predict(x)
        return compute_classification_metrics(y, probabilities, self.classes)

    def evaluate_split(self, val_data: Any) -> dict[str, float]:
        vx, vy = unpack_split(val_data)
        probabilities = self.predict(vx)
        return compute_classification_metrics(vy, probabilities, self.classes)

    def get_state(self) -> bytes:
        if self.weights is None:
            return b""
        return self.payload()[0]

    def restore_state(self, state: bytes) -> None:
        if not state:
            self.weights = None
            return
        self.load(state)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        continuation: bool = False,
        *,
        val_data: Any | None = None,
        config: IterativeTrainingConfig | dict[str, Any] | None = None,
        tracker: Any | None = None,
        emit_stdout: bool = False,
    ) -> int:
        if not continuation:
            self.weights = None
        rounds = self.continued_rounds if continuation else self.initial_rounds
        merged_params = {
            "max_epochs": min(100, max(1, int(rounds))),
            "patience": int(self.params.get("patience", 10)),
            "min_delta": float(self.params.get("min_delta", 1e-4)),
            "learning_rate": float(self.params.get("learning_rate", 0.01)),
            "l2_regularization": float(
                self.params.get("l2", self.params.get("l2_regularization", 0.001))
            ),
            "validation_fraction": float(self.params.get("validation_fraction", 0.1)),
            "seed": int(self.params.get("seed", 42)),
            "restore_best_weights": bool(self.params.get("restore_best_weights", True)),
        }
        if isinstance(config, dict):
            merged_params.update(config)
            iter_cfg = IterativeTrainingConfig.from_params(merged_params)
        elif isinstance(config, IterativeTrainingConfig):
            iter_cfg = config
        else:
            iter_cfg = IterativeTrainingConfig.from_params(merged_params)

        self.params["l2"] = iter_cfg.l2_regularization
        self.params["l2_regularization"] = iter_cfg.l2_regularization
        self.params["learning_rate"] = iter_cfg.learning_rate
        self.params["seed"] = iter_cfg.seed

        result = train_iterative(
            self,
            (x, y),
            val_data=val_data,
            config=iter_cfg,
            tracker=tracker,
            emit_stdout=emit_stdout,
        )
        self.last_training_result = result
        return int(result.epochs_run)

    @property
    def training_summary(self) -> dict[str, Any]:
        return dict(self.last_training_result or {})

    def predict(self, x: np.ndarray) -> np.ndarray:
        return softmax((x @ self.weights[:-1]).astype(np.float64, copy=False) + self.weights[-1], axis=1)

    predict_proba = predict

    def payload(self) -> tuple[bytes, str]:
        buffer = io.BytesIO()
        np.save(buffer, self.weights, allow_pickle=False)
        return buffer.getvalue(), "npy"

    def load(self, payload: bytes) -> None:
        self.weights = np.load(io.BytesIO(payload), allow_pickle=False)

    def explanation(self, features: list[str]) -> dict[str, dict[str, list[float]]]:
        return {
            "coefficients": {
                name: row.tolist()
                for name, row in zip([*features, "intercept"], self.weights, strict=True)
            }
        }


class ZiaPredictor:
    """Load a native checkpoint and score prepared numeric features in training order."""

    def __init__(self, directory):
        import hashlib
        import json
        from pathlib import Path

        directory = Path(directory)
        metadata = json.loads((directory / "model-metadata.json").read_text())
        payload = (directory / metadata["payload"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["state_sha256"]:
            raise ValueError("Zia model checkpoint checksum mismatch")
        self.features = metadata["features"]
        self.model = (
            FraudOrReturnBooster()
            if metadata["model"] == "xgboost"
            else MultinomialLogistic(len(metadata["classes"]))
        )
        self.model.load(payload)

    def predict(self, data, params=None):
        if not data.columns.is_unique or set(data.columns) != set(self.features):
            raise ValueError("inference feature schema differs from the Zia model")
        return self.model.predict(data[self.features].to_numpy(dtype=np.float32))


def _load_pyfunc(path):
    return ZiaPredictor(path)
