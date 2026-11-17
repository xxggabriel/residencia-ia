"""One CPU learner per Zia task, with iterative epoch training, early stopping and loadable model state."""

from __future__ import annotations

import gc
import io
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from river import optim
from scipy.optimize import minimize
from scipy.special import expit, softmax

from continual_bench.config.iterative import IterativeTrainingConfig
from continual_bench.core.observability import peak_rss_bytes as process_peak_rss_bytes
from continual_bench.models.classification_metrics import ClassificationCounts, average_precision
from continual_bench.models.observations import quality_scope, training_step
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
    "scale_pos_weight",
}


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: int,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Score a training split with the same decision rule and AP estimator as the Zia evaluation."""
    counts = ClassificationCounts(classes, threshold)
    if len(labels):
        counts.add(labels, probabilities)
    return _training_values(counts)


def _training_values(counts):
    """Preserve the observed-class legacy F1 and the original early-stopping inputs."""
    actual = counts.confusion.sum(axis=1)
    present = counts.confusion.sum(axis=0) + actual
    correct = counts.confusion.diagonal()
    total = int(actual.sum())
    class_ap = [
        average_precision(counts.positive[i], counts.negative[i]) if actual[i] else None
        for i in range(counts.classes)
    ]
    valid_ap = [score for score in class_ap if score is not None]
    f1_values = [float(2 * c / n) for c, n in zip(correct, present, strict=True) if n]
    recalls = [float(c / n) for c, n in zip(correct, actual, strict=True) if n]
    scores = {
        "log_loss": counts.loss / total if total else 0.0,
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "f1_macro": float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "macro_pr_auc": float(sum(valid_ap) / len(valid_ap)) if valid_ap else None,
    }
    if counts.classes == 2:
        scores["pr_auc"] = class_ap[1]
    return scores


def _split_scores(model, scope, features, labels, threshold):
    timer = time.perf_counter()
    probabilities = model.predict(features)
    setattr(model, f"last_{scope}_prediction_seconds", time.perf_counter() - timer)
    counts = ClassificationCounts(len(model.class_labels), threshold)
    counts.add(labels, probabilities)
    metrics = _training_values(counts)
    timer = time.perf_counter()
    setattr(model, f"last_{scope}_quality", counts.quality(model.class_labels))
    if counts.classes == 2:
        setattr(model, f"last_{scope}_scores", counts)
    setattr(model, f"last_{scope}_quality_seconds", time.perf_counter() - timer)
    return metrics


class OnlineHistory:
    """Track the prediction immediately preceding each minibatch's update."""

    def __init__(self, classes, threshold, tracker):
        self.classes = classes
        self.tracker = tracker
        self.counts = ClassificationCounts(len(classes), threshold)
        self.history, self.steps = [], []
        self.examples = 0
        self.started = time.perf_counter()
        self.inference_seconds = self.overhead_seconds = 0.0
        self.task = "binary" if len(classes) == 2 else "multiclass"

    def score(self, labels, probabilities):
        counts = ClassificationCounts(len(self.classes), self.counts.threshold)
        counts.add(labels, probabilities)
        return counts, _training_values(counts)

    def record(self, counts, values, optimizer_steps, seconds, inference_seconds, native_loss):
        step = len(self.history) + 1
        rows = int(counts.confusion.sum())
        point = {
            "minibatch_step": step,
            "train_samples": rows,
            "validation_samples": 0,
            "optimizer_steps": optimizer_steps,
            "training_duration_seconds": seconds,
            **{f"train:{key}": value for key, value in values.items()},
        }
        self.history.append(point)
        timer = time.perf_counter()
        self.examples += rows
        self.inference_seconds += inference_seconds
        self.counts.merge(counts)
        observation = training_step(
            step,
            examples=rows,
            cumulative=self.examples,
            duration_seconds=seconds,
            native_loss=native_loss,
            metrics={
                "batch/log_loss": values["log_loss"],
                "batch/inference_seconds": inference_seconds,
                "training/optimizer_steps": optimizer_steps,
            },
        )
        quality = counts.quality(self.classes)
        # Fit metadata supplies the fixed class order, score method and availability rules.
        observation["scopes"] = {
            "training_pre_update": {
                key: quality[key]
                for key in (
                    "evaluated_examples",
                    "macro_f1",
                    "accuracy",
                    "positive_precision",
                    "positive_recall",
                    "average_precision",
                    "recall_at_fpr_0_01",
                    "confusion_matrix",
                    "class_support",
                    "class_recall",
                )
                if key in quality
            }
        }
        self.steps.append(observation)
        self.overhead_seconds += time.perf_counter() - timer
        if self.tracker is not None and hasattr(self.tracker, "log_minibatch_metrics"):
            try:
                self.tracker.log_minibatch_metrics(point, step=step)
            except Exception:
                # The recorded point remains available for final publication.
                pass

    def finalize(self, *, interrupted, kind, **metadata):
        optimizer_seconds = sum(point["training_duration_seconds"] for point in self.history)
        summary = {
            "epochs_run": 0,
            "epochs_trained": 0,
            "iterations": len(self.history),
            "train_rows": self.examples,
            "validation_rows": 0,
            "training_presentations": self.examples,
            "optimizer_steps": sum(point["optimizer_steps"] for point in self.history),
            "training_duration_seconds": optimizer_seconds,
            "validation_duration_seconds": 0.0,
            "stopped_by_signal": interrupted,
            **metadata,
        }
        if self.tracker is not None and hasattr(self.tracker, "finalize"):
            try:
                self.tracker.finalize(summary)
            except Exception:
                pass
        timer = time.perf_counter()
        quality = quality_scope(self.task, self.counts.quality(self.classes, curves=True))
        self.overhead_seconds += time.perf_counter() - timer
        throughput = self.examples / self.inference_seconds if self.inference_seconds else None
        return TrainingResult(
            **summary,
            history=self.history,
            observation_steps=self.steps,
            observation_epochs=[],
            observed_step_kind=kind,
            observation_scopes={"training_pre_update": quality},
            observation_effort={
                "examples_observed": self.examples,
                "evaluated_examples": self.examples,
                "unique_training_examples": self.examples,
                "training_presentations": self.examples,
                "replay_presentations": 0,
                "steps": len(self.steps),
                "epochs": None,
            },
            observation_resources={
                "duration_seconds": time.perf_counter() - self.started,
                "training_seconds": optimizer_seconds,
                "inference_seconds": self.inference_seconds,
                "predictions_per_second": throughput,
                "peak_rss_bytes": process_peak_rss_bytes(),
                "rss_scope": "process_lifetime_high_water_mark",
                "tracking_overhead_seconds": self.overhead_seconds,
            },
        )


class FraudOrReturnBooster:
    name = "xgboost"
    training_step_kind = "xgboost_boosting_round"
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": 5,
        "max_bin": 128,
        "eta": 0.01,
        "reg_lambda": 0.001,
        "nthread": 2,
        "seed": 42,
    }
    initial_rounds = 100
    continued_rounds = 100
    online_params = {
        "optimizer": "xgboost_refresh",
        "gradient": "river_log_loss",
        "minibatch_rows": 8192,
    }

    def __init__(
        self,
        config: IterativeTrainingConfig | dict[str, Any] | None = None,
        dataset: str | None = None,
        classes: int | list[str] | None = None,
        decision_threshold: float | None = None,
        model_params: dict[str, Any] | None = None,
    ):
        enforce_single_thread_env()
        import xgboost

        self.xgb = xgboost
        self.model = None
        self.params = dict(self.params)
        self.dataset = dataset
        self.model_params = None if model_params is None else dict(model_params)
        self.class_labels = list(classes) if isinstance(classes, list) else [0, 1]
        self.decision_threshold = 0.5 if decision_threshold is None else decision_threshold
        self.params["nthread"] = int(os.environ["OMP_NUM_THREADS"])
        self.iterative_config = IterativeTrainingConfig.from_params(config)
        cfg = self.iterative_config
        if config is not None:
            self.params["eta"] = cfg.learning_rate
            self.params["reg_lambda"] = cfg.l2_regularization
            self.params["seed"] = cfg.seed
            self.initial_rounds = cfg.max_epochs
            self.continued_rounds = cfg.max_epochs
            if isinstance(config, dict):
                if "nthread" in config:
                    self.params["nthread"] = int(config["nthread"])
                if "scale_pos_weight" in config:
                    self.params["scale_pos_weight"] = float(config["scale_pos_weight"])
                if "dataset" in config and not self.dataset:
                    self.dataset = str(config["dataset"])

        for key, value in self._dataset_params().items():
            self.params.setdefault(key, value)

        self.params.setdefault("nthread", 2)
        self._cached_dmatrix = None
        self._cached_dmatrix_key: tuple[int, tuple[int, ...], int, tuple[int, ...]] | None = None
        self.last_training_result: TrainingResult | None = None

    def _dataset_params(self) -> dict[str, Any]:
        """Dataset defaults (e.g. class imbalance) injected by the caller from the stream provider.

        The model layer never resolves datasets itself; see ``datasets.streams``.
        """
        return self.model_params or {}

    def fit_online(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        tracker: Any | None = None,
        decision_threshold: float | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> int:
        """Update the leaves of the existing booster with River's logistic gradient."""
        threshold = self.decision_threshold if decision_threshold is None else decision_threshold
        loss = optim.losses.Log()
        weight = float(self.params.get("scale_pos_weight", 1.0))

        def objective(margins, matrix):
            labels = matrix.get_label()
            gradient = loss.gradient(labels, margins.clip(-80, 80))
            p = expit(margins)
            hessian = np.maximum(p * (1 - p), 1e-16)
            if weight != 1.0:
                pos = labels == 1
                gradient = np.where(pos, gradient * weight, gradient)
                hessian = np.where(pos, hessian * weight, hessian)
            return gradient, hessian

        params = {
            **self.params,
            "process_type": "update",
            "updater": "refresh",
            "refresh_leaf": True,
            "verbosity": 0,
        }
        rounds = self.model.num_boosted_rounds()
        size = self.online_params["minibatch_rows"]
        history = OnlineHistory(self.class_labels, threshold, tracker)
        for start in range(0, len(y), size):
            if stop_requested is not None and stop_requested():
                break
            stop = start + size
            features, labels = x[start:stop], y[start:stop]
            timer = time.perf_counter()
            probabilities = self.predict(features)
            inference_seconds = time.perf_counter() - timer
            counts, metrics = history.score(labels, probabilities)
            timer = time.perf_counter()
            nthread = int(self.params.get("nthread", 2))
            matrix = self.xgb.DMatrix(features, label=labels, nthread=nthread)
            self.model = self.xgb.train(
                params,
                matrix,
                num_boost_round=rounds,
                xgb_model=self.model,
                obj=objective,
            )
            history.record(counts, metrics, rounds, time.perf_counter() - timer, inference_seconds, None)
        self.last_training_result = history.finalize(
            interrupted=stop_requested is not None and stop_requested(),
            kind="xgboost_tree_refresh_minibatch",
            booster_rounds=rounds,
        )
        return int(self.last_training_result.optimizer_steps)

    def _xgb_params(self) -> dict[str, Any]:
        clean = {k: v for k, v in self.params.items() if k in _XGB_ALLOWED_PARAMS}
        if "learning_rate" in self.params and "eta" not in clean:
            clean["eta"] = float(self.params["learning_rate"])
        if "l2_regularization" in self.params and "reg_lambda" not in clean:
            clean["reg_lambda"] = float(self.params["l2_regularization"])
        elif "l2" in self.params and "reg_lambda" not in clean:
            clean["reg_lambda"] = float(self.params["l2"])
        clean["nthread"] = int(self.params.get("nthread", 2))
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
                nthread=int(self.params.get("nthread", 2)),
            )
            self._cached_dmatrix_key = key
        return self._cached_dmatrix

    def release_training_cache(self) -> None:
        self._cached_dmatrix = None
        self._cached_dmatrix_key = None

    def train_epoch(self, train_data: Any) -> dict[str, float | None]:
        x, y = unpack_split(train_data)
        matrix = self._dmatrix(x, y)
        params = self._xgb_params()
        if self.model is None and y.min() == y.max():
            # XGBoost's estimated logistic intercept can hit 0 or 1 on a single-class startup.
            params["base_score"] = 0.5
        self.model = self.xgb.train(
            params,
            matrix,
            num_boost_round=1,
            xgb_model=self.model,
        )
        return _split_scores(self, "train", x, y, self.decision_threshold)

    def evaluate_split(self, val_data: Any) -> dict[str, float | None]:
        vx, vy = unpack_split(val_data)
        return _split_scores(self, "validation", vx, vy, self.decision_threshold)

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
        timestamps: np.ndarray | None = None,
        decision_threshold: float | None = None,
        capture_sigterm: bool = True,
        stop_requested: Callable[[], bool] | None = None,
    ) -> int:
        if decision_threshold is not None:
            self.decision_threshold = decision_threshold
        if not continuation:
            self.model = None
        rounds = self.continued_rounds if continuation else self.initial_rounds
        defaults = self.iterative_config
        merged_params = {
            "max_epochs": min(100, max(1, int(rounds))),
            "patience": int(self.params.get("patience", defaults.patience)),
            "min_delta": float(self.params.get("min_delta", defaults.min_delta)),
            "eval_metric": defaults.eval_metric,
            "learning_rate": float(self.params.get("eta", self.params.get("learning_rate", 0.01))),
            "l2_regularization": float(
                self.params.get("reg_lambda", self.params.get("l2_regularization", 0.001))
            ),
            "validation_fraction": self.params.get("validation_fraction", defaults.validation_fraction),
            "seed": int(self.params.get("seed", 42)),
            "restore_best_weights": self.params.get("restore_best_weights", defaults.restore_best_weights),
        }
        if isinstance(config, dict):
            merged_params.update(config)
            if "scale_pos_weight" in config:
                self.params["scale_pos_weight"] = float(config["scale_pos_weight"])
            if "dataset" in config and not self.dataset:
                self.dataset = str(config["dataset"])
            iter_cfg = IterativeTrainingConfig.from_params(merged_params)
        elif isinstance(config, IterativeTrainingConfig):
            iter_cfg = config
        else:
            iter_cfg = IterativeTrainingConfig.from_params(merged_params)

        self.params["eta"] = iter_cfg.learning_rate
        self.params["reg_lambda"] = iter_cfg.l2_regularization
        self.params["seed"] = iter_cfg.seed
        self.params["nthread"] = int(self.params.get("nthread", 2))
        for key, value in self._dataset_params().items():
            self.params.setdefault(key, value)

        result = train_iterative(
            self,
            (x, y),
            val_data=val_data,
            config=iter_cfg,
            tracker=tracker,
            emit_stdout=emit_stdout,
            timestamps=timestamps,
            capture_sigterm=capture_sigterm,
            stop_requested=stop_requested,
        )
        result["booster_rounds"] = self.rounds
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
        self.model = self.xgb.Booster(params={"nthread": int(self.params.get("nthread", 2))})
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
    training_step_kind = "multinomial_full_batch_solver_call"
    params = {
        "solver": "L-BFGS-B",
        "l2": 0.001,
        "l2_regularization": 0.001,
        "learning_rate": 0.01,
        "seed": 42,
    }
    initial_rounds = 100
    continued_rounds = 100
    online_params = {"optimizer": "river_sgd", "learning_rate": 0.05, "minibatch_rows": 1024}

    def __init__(
        self,
        classes: int | list[str],
        config: IterativeTrainingConfig | dict[str, Any] | None = None,
        decision_threshold: float | None = None,
    ):
        enforce_single_thread_env()
        self.class_labels = list(classes) if isinstance(classes, list) else list(range(classes))
        self.classes = len(self.class_labels)
        self.decision_threshold = 0.5 if decision_threshold is None else decision_threshold
        self.weights: np.ndarray | None = None
        self._online_optimizer = optim.SGD(self.online_params["learning_rate"])
        self.params = dict(self.params)
        self.iterative_config = IterativeTrainingConfig.from_params(config)
        cfg = self.iterative_config
        if config is not None:
            self.params["l2"] = cfg.l2_regularization
            self.params["l2_regularization"] = cfg.l2_regularization
            self.params["learning_rate"] = cfg.learning_rate
            self.params["seed"] = cfg.seed
            self.initial_rounds = cfg.max_epochs
            self.continued_rounds = cfg.max_epochs
        self.last_training_result: TrainingResult | None = None

    @property
    def online_iterations(self) -> int:
        return self._online_optimizer.n_iterations

    def restore_online_iterations(self, iterations: int) -> None:
        self._online_optimizer.n_iterations = iterations

    def fit_online(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        tracker: Any | None = None,
        decision_threshold: float | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> int:
        """Apply River SGD to the same softmax weights and intercept fitted at warm-up."""
        threshold = self.decision_threshold if decision_threshold is None else decision_threshold
        size = self.online_params["minibatch_rows"]
        history = OnlineHistory(self.class_labels, threshold, tracker)
        for start in range(0, len(y), size):
            if stop_requested is not None and stop_requested():
                break
            labels = y[start : start + size]
            features = x[start : start + size]
            timer = time.perf_counter()
            residuals = self.predict(features)
            inference_seconds = time.perf_counter() - timer
            counts, metrics = history.score(labels, residuals)
            timer = time.perf_counter()
            observed = residuals[np.arange(len(labels)), labels]
            native_loss = None
            if np.all(observed > 0):
                regularization = self.params["l2"] * np.square(self.weights[:-1]).sum() / 2
                native_loss = float(-np.log(observed).mean() + regularization)
            history.overhead_seconds += time.perf_counter() - timer
            timer = time.perf_counter()
            residuals[np.arange(len(labels)), labels] -= 1
            gradient = np.empty_like(self.weights)
            gradient[:-1] = features.T @ residuals / len(labels)
            gradient[:-1] += self.params["l2"] * self.weights[:-1]
            gradient[-1] = residuals.mean(axis=0)
            self.weights = self._online_optimizer.step(self.weights, gradient)
            history.record(counts, metrics, 1, time.perf_counter() - timer, inference_seconds, native_loss)
        self.last_training_result = history.finalize(
            interrupted=stop_requested is not None and stop_requested(),
            kind="river_sgd_minibatch",
        )
        return int(self.last_training_result.iterations)

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
        logits -= logits.max(axis=1, keepdims=True)
        index = np.arange(len(y))
        observed_logits = logits[index, y]
        np.exp(logits, out=logits)
        normalizer = logits.sum(axis=1, keepdims=True)
        loss = float(
            (np.log(normalizer[:, 0]) - observed_logits).mean()
            + (regularization / 2.0) * np.square(weights[:-1]).sum()
        )
        logits /= normalizer
        logits[index, y] -= 1.0
        gradient = np.empty(shape, dtype=np.float64)
        gradient[:-1] = (x.T @ logits) / len(y) + regularization * weights[:-1]
        gradient[-1] = logits.mean(axis=0)
        return loss, gradient.ravel()

    def train_epoch(self, train_data: Any) -> dict[str, float | None]:
        x, y = unpack_split(train_data)
        shape = (x.shape[1] + 1, self.classes)
        regularization = float(self.params.get("l2", self.params.get("l2_regularization", 0.001)))
        learning_rate = float(self.params.get("learning_rate", 0.01))
        solver = str(self.params.get("solver", "L-BFGS-B"))

        if self.weights is None or self.weights.shape != shape:
            self.weights = np.zeros(shape, dtype=np.float64)

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            return self._objective_and_grad(x, y, flat, shape, regularization)

        start = self.weights.ravel()
        if solver.upper() in ("SGD", "GD"):
            native_loss, grad = objective(start)
            if not np.isfinite(native_loss):
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
            native_loss = result.fun
            if result.nit == 0:
                step_candidate = start - learning_rate * result.jac
                step_loss, _ = objective(step_candidate)
                if np.isfinite(step_loss) and step_loss < result.fun:
                    candidate, native_loss = step_candidate, step_loss
            self.weights = np.array(candidate, dtype=np.float64, copy=True).reshape(shape)

        metrics = _split_scores(self, "train", x, y, self.decision_threshold)
        return {**metrics, "native_loss": float(native_loss)}

    def evaluate_split(self, val_data: Any) -> dict[str, float | None]:
        vx, vy = unpack_split(val_data)
        return _split_scores(self, "validation", vx, vy, self.decision_threshold)

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
        timestamps: np.ndarray | None = None,
        decision_threshold: float | None = None,
        capture_sigterm: bool = True,
        stop_requested: Callable[[], bool] | None = None,
    ) -> int:
        if decision_threshold is not None:
            self.decision_threshold = decision_threshold
        if not continuation:
            self.weights = None
        rounds = self.continued_rounds if continuation else self.initial_rounds
        defaults = self.iterative_config
        merged_params = {
            "max_epochs": min(100, max(1, int(rounds))),
            "patience": int(self.params.get("patience", defaults.patience)),
            "min_delta": float(self.params.get("min_delta", defaults.min_delta)),
            "eval_metric": defaults.eval_metric,
            "learning_rate": float(self.params.get("learning_rate", 0.01)),
            "l2_regularization": float(self.params.get("l2", self.params.get("l2_regularization", 0.001))),
            "validation_fraction": self.params.get("validation_fraction", defaults.validation_fraction),
            "seed": int(self.params.get("seed", 42)),
            "restore_best_weights": self.params.get("restore_best_weights", defaults.restore_best_weights),
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
            timestamps=timestamps,
            capture_sigterm=capture_sigterm,
            stop_requested=stop_requested,
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
                name: row.tolist() for name, row in zip([*features, "intercept"], self.weights, strict=True)
            }
        }


def make_zia_model(
    name: str,
    classes: int | list[str],
    config: IterativeTrainingConfig | dict[str, Any] | None = None,
    dataset: str | None = None,
    *,
    decision_threshold: float | None = None,
    model_params: dict[str, Any] | None = None,
):
    """Build a stream model; ``model_params`` are the dataset defaults from its stream provider."""
    if name == "xgboost":
        return FraudOrReturnBooster(
            config=config,
            dataset=dataset,
            classes=classes,
            decision_threshold=decision_threshold,
            model_params=model_params,
        )
    if name == "multinomial_logistic":
        return MultinomialLogistic(classes, config=config, decision_threshold=decision_threshold)
    raise ValueError(f"unknown Zia model: {name}")


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
        self.model = make_zia_model(
            metadata["model"], len(metadata["classes"]), dataset=metadata.get("dataset")
        )
        self.model.load(payload)

    def predict(self, data, params=None):
        if not data.columns.is_unique or set(data.columns) != set(self.features):
            raise ValueError("inference feature schema differs from the Zia model")
        return self.model.predict(data[self.features].to_numpy(dtype=np.float32))


def _load_pyfunc(path):
    return ZiaPredictor(path)
