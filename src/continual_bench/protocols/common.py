"""Shared utilities for benchmark protocol implementations (CARA and UPF)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from continual_bench.datasets.streams_unified import BatchSplit
from continual_bench.detectors import make_detector
from continual_bench.models.adapters import ModelAdapter, build_model_adapter
from continual_bench.tracking.tracker import BenchmarkTracker


def canonical_dataset_name(name: str) -> str:
    """Return canonical dataset name for known aliases."""
    lower = name.strip().lower()
    if lower in {"airplanes", "airlines"}:
        return "airlines"
    if lower in {"electricity", "elecnormnew"}:
        return "electricity"
    return lower


def trapezoid_auc(y: list[float] | np.ndarray, x: list[float] | np.ndarray) -> float:
    """Compute trapezoidal AUC using ``np.trapezoid`` with fallback to ``np.trapz``."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


class AdapterLearner:
    """Shared ``Learner`` adapter wrapping ``ModelConfig`` (``fit`` and ``losses``)."""

    def __init__(self, model_cfg: Any) -> None:
        self.model_cfg = model_cfg

    def fit(self, index: int, x: Any, y: np.ndarray, *context: np.ndarray) -> ModelAdapter:
        """Fit a fresh model adapter on batch ``(x, y)``."""
        del index, context
        return build_model_adapter(self.model_cfg).fit(x, y)

    def losses(self, state: ModelAdapter, x: Any, y: np.ndarray) -> np.ndarray:
        """Compute per-sample 0-1 classification losses for ``state`` on ``(x, y)``."""
        preds = state.predict(x)
        return (preds != y).astype(np.float64)


def _build_detector(
    detector_name: str,
    *,
    backend: str = "skmultiflow",
    detector_params: dict[str, Any] | None = None,
) -> Any:
    params = dict(detector_params or {})
    try:
        return make_detector(detector_name, backend=backend, **params)
    except TypeError as exc:
        if "backend" in str(exc):
            return make_detector(detector_name, **params)
        raise


def _batch_errors(
    state: Any,
    batch: BatchSplit,
    split: Literal["train", "test", "both"],
) -> np.ndarray:
    """Compute binary 0-1 prediction errors for ``state`` on the requested batch split."""
    if split == "train":
        preds = state.predict(batch.X_train)
        return (preds != batch.y_train).astype(np.int64)
    if split == "test":
        preds = state.predict(batch.X_test)
        return (preds != batch.y_test).astype(np.int64)
    if split == "both":
        return np.concatenate(
            [
                _batch_errors(state, batch, "train"),
                _batch_errors(state, batch, "test"),
            ]
        )
    raise ValueError(f"Unsupported split: {split!r}")


def run_error_detector_schedule(
    detector_name: str,
    batches: list[BatchSplit],
    states: list[Any],
    offline: int,
    *,
    backend: str = "skmultiflow",
    detector_params: dict[str, Any] | None = None,
    reset_on_drift: bool = False,
    warm_start_on_reset: bool = False,
    initial_model: int | None = None,
    eval_split: Literal["train", "test"] = "test",
    warm_start_split: Literal["train", "test", "both"] = "both",
    include_offline_in_schedule: bool = True,
) -> list[int]:
    """Run sequential binary prediction-error drift detection across online batches.

    Supports both CARA semantics (``reset_on_drift=False``: scans every sample in a batch,
    keeps detector state across triggers, defers model deployment to the next batch) and
    UPF semantics (``reset_on_drift=True``: stops batch scan on first alarm, deploys
    immediately, resets detector, and optionally warm-starts on the new model's batch).
    """
    detector = _build_detector(detector_name, backend=backend, detector_params=detector_params)
    retrains: list[int] = []
    deployed = offline if initial_model is None else initial_model

    if reset_on_drift:
        for t in range(offline, len(batches)):
            errors = _batch_errors(states[deployed], batches[t], eval_split)
            triggered = False
            for err in errors:
                if detector.update(int(err)):
                    triggered = True
                    break
            if triggered:
                retrains.append(t if include_offline_in_schedule else t - offline)
                deployed = t
                if hasattr(detector, "reset"):
                    detector.reset()
                else:
                    detector = _build_detector(
                        detector_name, backend=backend, detector_params=detector_params
                    )
                if warm_start_on_reset:
                    warm_errors = _batch_errors(states[deployed], batches[deployed], warm_start_split)
                    if hasattr(detector, "warm_start"):
                        detector.warm_start(warm_errors)
                    else:
                        for w_err in warm_errors:
                            detector.update(int(w_err))
        return retrains

    detected = False
    for t in range(offline, len(batches)):
        if detected:
            deployed = t - 1
        detected = False
        errors = _batch_errors(states[deployed], batches[t], eval_split)
        for err in errors:
            if detector.update(int(err)):
                detected = True
        if detected:
            retrains.append(t if include_offline_in_schedule else t - offline)
    return retrains


def aggregate_seed_summary(
    records: list[dict[str, Any]],
    *,
    protocol: str,
    dataset: str,
    policies: list[str],
    metrics: tuple[str, ...],
    regime: str = "price_grid",
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate per-seed metrics into mean/std/n_seeds rows per ``(policy, metric)``."""
    dataset_canonical = canonical_dataset_name(dataset)
    aggregate_rows: list[dict[str, Any]] = []
    for pol in policies:
        pol_rows = [
            r
            for r in records
            if r.get("policy") == pol and ("regime" not in r or r.get("regime") == regime)
        ]
        for metric in metrics:
            vals = [float(r[metric]) for r in pol_rows if metric in r and r[metric] is not None]
            row: dict[str, Any] = {
                "protocol": protocol,
                "dataset": dataset_canonical,
                "policy": pol,
                "regime": regime,
                "metric": metric,
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            }
            if extra_fields:
                row.update(extra_fields)
            aggregate_rows.append(row)
    return aggregate_rows


def log_protocol_outputs(
    tracker: BenchmarkTracker,
    seed_summaries: list[dict[str, Any]],
    regime_records: list[dict[str, Any]],
) -> None:
    """Log per-regime records and per-seed summaries to ``tracker``."""
    for r in regime_records:
        payload = {k: v for k, v in r.items() if k not in {"seed", "policy"}}
        tracker.log_regime(r["seed"], r["policy"], payload)
    for s in seed_summaries:
        payload = {k: v for k, v in s.items() if k not in {"seed", "policy"}}
        tracker.log_seed_summary(s["seed"], s["policy"], payload)
