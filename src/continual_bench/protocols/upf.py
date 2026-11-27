"""UPF (2025) benchmark protocol implementation."""

from __future__ import annotations

import gzip
import json
import logging
import math
import time
from typing import Any

import numpy as np
from scipy.sparse import issparse

from continual_bench.backends.base import ExecutionBackend
from continual_bench.datasets.streams_unified import BatchSplit, load_stream_batches
from continual_bench.models.adapters import ModelAdapter
from continual_bench.protocols.base import (
    BenchmarkProtocol,
    CampaignConfig,
    ProtocolResult,
    SpotInterruptedError,
)
from continual_bench.protocols.common import (
    AdapterLearner,
    aggregate_seed_summary,
    canonical_dataset_name,
    log_protocol_outputs,
    run_error_detector_schedule,
    trapezoid_auc,
)
from continual_bench.retraining.matrix import PerformanceMatrix, build_paper_matrix
from continual_bench.retraining.rules import UPF, CaraCT, CaraP, CaraT, Oracle
from continual_bench.retraining.schedule import alpha_max_same_batch
from continual_bench.tracking.tracker import BenchmarkTracker

logger = logging.getLogger(__name__)

_DEFAULT_UPF_POLICIES = (
    "upf",
    "cara",
    "cara_cumulative",
    "cara_periodic",
    "oracle",
    "adwin_5",
    "adwin_50",
    "fhddm_5",
    "fhddm_50",
    "kswin_5",
    "kswin_50",
)

_DETECTOR_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "adwin_5": ("adwin", {"delta": 0.05}),
    "adwin_50": ("adwin", {"delta": 0.50}),
    "fhddm_5": ("fhddm", {"delta": 0.05}),
    "fhddm_50": ("fhddm", {"delta": 0.50}),
    "kswin_5": ("kswin", {"alpha": 0.05}),
    "kswin_50": ("kswin", {"alpha": 0.50}),
}


def _to_dense_float64(x: Any) -> np.ndarray:
    if issparse(x):
        return np.asarray(x.toarray(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _dynamic_alpha_max(
    seed_matrices: list[tuple[int, PerformanceMatrix, dict[str, list[int]]]],
) -> float:
    """Compute smallest ``alpha_max`` where ``Oracle(objective='error')`` yields 0 retrains across seeds."""
    bounds: list[float] = []
    for _, mat, _ in seed_matrices:
        bound = alpha_max_same_batch(
            np.nan_to_num(mat.err.T, nan=0.0),
            mat.initial,
            mat.offline,
            mat.batches - 1,
            retrain_errors=np.nan_to_num(mat.err.diagonal(), nan=0.0),
            allowed_retrains=mat.admissible_retrains,
        )
        if math.isfinite(bound) and bound > 0.0:
            bounds.append(float(bound))
    if not bounds:
        return 1.0
    alpha_max = max(bounds)
    if alpha_max <= 0.0:
        return 1.0

    def _any_seed_retrains(alpha_val: float) -> bool:
        for _, mat, _ in seed_matrices:
            oracle_rule = Oracle(objective="error", same_batch=False)
            oracle_rule.fit_future(mat, alpha_val)
            if oracle_rule.schedule:
                return True
        return False

    for _ in range(10):
        if not _any_seed_retrains(alpha_max):
            break
        alpha_max *= 1.05
    return float(alpha_max)


class _AbsoluteStalenessMatrix(PerformanceMatrix):
    """PerformanceMatrix variant using absolute staleness ``psi(Q_t, D_t, f_m)`` (UPF Appendix A.12)."""

    def staleness(self, t: int, model: int, *, cutoff: int | None = None) -> float:
        if not model < t:
            raise ValueError("staleness is defined for a model trained before batch t")
        cache = self.__dict__.setdefault("_abs_staleness_cache", {})
        key = (t, model, cutoff)
        if key in cache:
            return cache[key]
        queries = self.queries[t]
        gamma = self.kernel_gamma()
        query_norm = (queries**2).sum(axis=1)[:, None]
        data = self.points[t]
        loss = self.sub_loss[model, t]
        if cutoff is not None and self.point_available_at is not None:
            visible = self.point_available_at[t] <= cutoff
            if not visible.any():
                return float("nan")
            data, loss = data[visible], loss[visible]
        distance = query_norm + (data**2).sum(axis=1)[None, :] - 2 * queries @ data.T
        value = float((np.exp(-gamma * np.maximum(distance, 0.0)) * loss[None, :]).mean())
        cache[key] = value
        return value


def _to_absolute_staleness_matrix(
    matrix: PerformanceMatrix,
    gamma: float | None = None,
) -> _AbsoluteStalenessMatrix:
    mat = _AbsoluteStalenessMatrix(
        err=matrix.err,
        offline=matrix.offline,
        points=matrix.points,
        queries=matrix.queries,
        sub_loss=matrix.sub_loss,
        feature_means=matrix.feature_means,
        fit_seconds=matrix.fit_seconds,
        rows=matrix.rows,
        paper_timing=matrix.paper_timing,
        decision_cutoffs=matrix.decision_cutoffs,
        label_available_at=matrix.label_available_at,
        point_available_at=matrix.point_available_at,
        row_loss=matrix.row_loss,
        training_records=matrix.training_records,
    )
    if gamma is not None:
        mat._gamma = float(gamma)
    return mat


def _detector_retrains(
    policy: str,
    batches: list[BatchSplit],
    states: list[ModelAdapter],
    offline: int,
    seed: int = 0,
) -> list[int]:
    det_name, base_params = _DETECTOR_SPECS[policy]
    det_params = dict(base_params)
    eval_window = max(25, len(batches[offline].y_test))
    if det_name == "fhddm":
        det_params["window_size"] = eval_window
    elif det_name == "kswin":
        det_params["window_size"] = eval_window
        det_params["stat_size"] = 30 if eval_window > 30 else max(1, eval_window // 2)
        det_params["seed"] = seed
    return run_error_detector_schedule(
        det_name,
        batches,
        states,
        offline,
        backend="skmultiflow",
        detector_params=det_params,
        reset_on_drift=True,
        warm_start_on_reset=False,
        initial_model=offline - 1,
        eval_split="test",
        warm_start_split="test",
        include_offline_in_schedule=True,
    )


def _evaluate_schedule_with_retrains(
    matrix: PerformanceMatrix, retrains: list[int], alpha: float
) -> tuple[dict[str, float], list[int]]:
    retrain_set = set(retrains)
    deployed = matrix.initial
    total_err = 0.0
    n_retrains = 0
    online_count = matrix.batches - matrix.offline
    executed: list[int] = []
    for t in range(matrix.offline, matrix.batches):
        if t in retrain_set:
            deployed = t
            n_retrains += 1
            executed.append(t)
        total_err += float(matrix.err[deployed, t])
    cost = total_err + float(alpha) * n_retrains
    avg_acc = 1.0 - (total_err / max(1, online_count))
    return (
        {
            "cost": float(cost),
            "retrains": float(n_retrains),
            "average_accuracy": float(avg_acc),
        },
        executed,
    )


def _evaluate_schedule(matrix: PerformanceMatrix, retrains: list[int], alpha: float) -> dict[str, float]:
    metrics, _ = _evaluate_schedule_with_retrains(matrix, retrains, alpha)
    return metrics


def _replay_rule_with_retrains(
    matrix: PerformanceMatrix,
    policy: str,
    alpha: float,
    seed: int,
    upf_delta: float,
    upf_samples: int,
) -> tuple[dict[str, float], list[int]]:
    if policy == "oracle":
        rule = Oracle(objective="error", same_batch=False)
        rule.fit_future(matrix, alpha)
        return _evaluate_schedule_with_retrains(matrix, rule.schedule, alpha)

    if policy == "upf":
        rule = UPF(delta=upf_delta, samples=upf_samples, seed=seed)
        rule.bind(matrix)
        rule.fit_offline(matrix.offline_view(), alpha)
        rule.begin_online()
    elif policy == "cara":
        rule = CaraT(same_batch=True)
        rule.fit_offline(matrix.offline_view(), alpha)
        rule.begin_online()
    elif policy == "cara_cumulative":
        rule = CaraCT(same_batch=True)
        rule.fit_offline(matrix.offline_view(), alpha)
        rule.begin_online()
    elif policy == "cara_periodic":
        rule = CaraP(
            same_batch=True,
            allow_never=False,
            max_period=matrix.offline,
            phase_offset=matrix.offline,
        )
        rule.fit_offline(matrix.offline_view(), alpha)
        rule.begin_online()
    else:
        raise ValueError(f"Unsupported UPF rule policy: {policy!r}")

    retrains: list[int] = []
    deployed = matrix.initial
    for t in range(matrix.offline, matrix.batches):
        if policy == "upf":
            view = matrix.view(t - 1, range(t))
        else:
            view = matrix.view(t, range(t))
        if rule.decide(t, deployed, view):
            retrains.append(t)
            deployed = t
    return _evaluate_schedule_with_retrains(matrix, retrains, alpha)


def _replay_rule(
    matrix: PerformanceMatrix,
    policy: str,
    alpha: float,
    seed: int,
    upf_delta: float,
    upf_samples: int,
) -> dict[str, float]:
    metrics, _ = _replay_rule_with_retrains(matrix, policy, alpha, seed, upf_delta, upf_samples)
    return metrics


class UpfProtocol(BenchmarkProtocol):
    """Strict UPF (Regol et al. 2025) protocol implementation."""

    def build_seed_matrix(
        self,
        config: CampaignConfig,
        seed: int,
        *,
        checkpoint_backend: Any | None = None,
        checkpoint_rel_path: str | None = None,
        stop_requested: Any | None = None,
        checkpoint_interval_seconds: float = 30.0,
    ) -> tuple[int, PerformanceMatrix, dict[str, list[int]]]:
        """Build performance matrix and detector schedules for a single seed with Spot checkpointing."""
        policies = list(config.policies) if config.policies else list(_DEFAULT_UPF_POLICIES)
        staleness_mode = config.staleness_mode
        if "staleness_mode" not in config.model_fields_set:
            staleness_mode = "absolute"

        ds_fields = config.dataset.model_fields_set
        ds_updates: dict[str, Any] = {
            "protocol": "upf",
            "seed": config.dataset.seed + seed,
        }
        if "offline_batches" not in ds_fields:
            ds_updates["offline_batches"] = 7
        if "online_batches" not in ds_fields:
            ds_updates["online_batches"] = 8
        if "num_batches" not in ds_fields:
            off_val = ds_updates.get("offline_batches", config.dataset.offline_batches)
            on_val = ds_updates.get("online_batches", config.dataset.online_batches)
            ds_updates["num_batches"] = (
                off_val + on_val
                if ("offline_batches" in ds_fields or "online_batches" in ds_fields)
                else 15
            )
        if "split_ratio" not in ds_fields:
            ds_updates["split_ratio"] = 0.8
        spec = config.dataset.model_copy(update=ds_updates)
        batches = load_stream_batches(spec)
        offline = sum(1 for b in batches if b.is_offline)
        if offline < 2 or offline > len(batches) - 2:
            offline = min(max(2, spec.offline_batches), max(2, len(batches) - 2))
        raw_batches = [
            (
                _to_dense_float64(b.X_train),
                np.asarray(b.y_train),
                _to_dense_float64(b.X_test),
                np.asarray(b.y_test),
            )
            for b in batches
        ]
        model_cfg = config.model.model_copy(update={"seed": seed})
        learner = AdapterLearner(model_cfg)

        size = len(raw_batches)
        completed_rows: set[int] = set()
        err_init: np.ndarray | None = None
        sub_loss_init: dict[tuple[int, int], np.ndarray] | None = None

        if checkpoint_backend is not None and checkpoint_rel_path:
            raw_ckpt = checkpoint_backend.read_bytes(checkpoint_rel_path)
            if raw_ckpt is not None:
                ckpt = json.loads(gzip.decompress(raw_ckpt).decode("utf-8"))
                if int(ckpt.get("seed", -1)) == seed and len(ckpt.get("err", [])) == size:
                    completed_rows = {int(r) for r in ckpt.get("completed_rows", [])}
                    err_init = np.asarray(ckpt["err"], dtype=np.float64)
                    sub_loss_init = {
                        tuple(int(x) for x in k.split(",")): np.asarray(v, dtype=np.float64)
                        for k, v in ckpt.get("sub_loss", {}).items()
                    }

        last_ckpt = [time.monotonic()]

        def _on_row_complete(
            row_idx: int,
            err_arr: np.ndarray,
            sub_loss_dict: dict[tuple[int, int], np.ndarray],
        ) -> None:
            completed_rows.add(row_idx)
            stopping = bool(stop_requested()) if stop_requested is not None else False
            if checkpoint_backend is not None and checkpoint_rel_path:
                if stopping or (time.monotonic() - last_ckpt[0] >= checkpoint_interval_seconds):
                    payload = {
                        "seed": seed,
                        "size": size,
                        "completed_rows": sorted(completed_rows),
                        "err": err_arr.tolist(),
                        "sub_loss": {f"{i},{j}": v.tolist() for (i, j), v in sub_loss_dict.items()},
                    }
                    checkpoint_backend.write_bytes(
                        checkpoint_rel_path,
                        gzip.compress(json.dumps(payload).encode("utf-8")),
                    )
                    last_ckpt[0] = time.monotonic()
            if stopping:
                raise SpotInterruptedError(
                    f"Spot interruption after row {row_idx} for seed {seed}"
                )

        try:
            use_row_hook = checkpoint_backend is not None or stop_requested is not None
            matrix, states = build_paper_matrix(
                raw_batches,
                learner,
                offline=offline,
                seed=seed,
                completed_rows=completed_rows or None,
                err_init=err_init,
                sub_loss_init=sub_loss_init,
                on_row_complete=_on_row_complete if use_row_hook else None,
                stop_requested=stop_requested,
            )
        except InterruptedError as exc:
            if not isinstance(exc, SpotInterruptedError):
                raise SpotInterruptedError(str(exc)) from exc
            raise
        if staleness_mode == "absolute":
            matrix = _to_absolute_staleness_matrix(matrix, gamma=config.kernel_gamma)
        elif config.kernel_gamma is not None:
            matrix._gamma = float(config.kernel_gamma)

        detector_schedules: dict[str, list[int]] = {}
        for pol in policies:
            if pol in _DETECTOR_SPECS:
                detector_schedules[pol] = _detector_retrains(pol, batches, states, offline, seed=seed)

        return seed, matrix, detector_schedules

    def run_seed_matrix(
        self,
        config: CampaignConfig,
        seed: int,
        tracker: BenchmarkTracker,
    ) -> tuple:
        return tracker.run_matrix(
            seed,
            lambda: self.build_seed_matrix(config, seed),
            lambda result: result[1].fit_seconds,
        )

    def run(
        self,
        config: CampaignConfig,
        backend: ExecutionBackend,
        tracker: BenchmarkTracker,
    ) -> ProtocolResult:
        seeds = list(config.seeds) if config.seeds else list(range(10))
        seed_matrices = backend.run_tasks(
            [lambda s=s: self.run_seed_matrix(config, s, tracker) for s in seeds],
            max_workers=config.max_workers,
        )
        return self.score_and_finalize(config, seed_matrices, backend, tracker)

    def score_and_finalize(
        self,
        config: CampaignConfig,
        seed_matrices: list[tuple[int, PerformanceMatrix, dict[str, list[int]]]],
        backend: ExecutionBackend,
        tracker: BenchmarkTracker,
    ) -> ProtocolResult:
        policies = list(config.policies) if config.policies else list(_DEFAULT_UPF_POLICIES)
        dataset_canonical = canonical_dataset_name(config.dataset.name)

        if config.alpha_grid:
            alpha_grid = [float(v) for v in config.alpha_grid]
        elif dataset_canonical == "electricity":
            alpha_grid = [float(v) for v in np.linspace(0.0, 1.0, 10)]
        elif dataset_canonical == "airlines":
            alpha_grid = [float(v) for v in np.linspace(0.0, 0.7, 10)]
        else:
            alpha_max = _dynamic_alpha_max(seed_matrices)
            alpha_grid = [float(v) for v in np.linspace(0.0, alpha_max, 10)]

        alpha_max_val = float(alpha_grid[-1]) if alpha_grid else 1.0
        if config.regimes is not None:
            regimes = {str(k): float(v) for k, v in config.regimes.items()}
        elif dataset_canonical in {"electricity", "airlines"}:
            regimes = {"low": 0.1, "high": 0.9}
        else:
            regimes = {
                "low": round(0.1 * alpha_max_val, 6),
                "high": round(0.9 * alpha_max_val, 6),
            }

        tracker.log_metadata(
            {
                "alpha_max": alpha_max_val,
                "alpha_grid": list(alpha_grid),
                "regimes": dict(regimes),
            }
        )

        def _score_seed(
            seed_data: tuple[int, PerformanceMatrix, dict[str, list[int]]],
        ) -> tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]:
            seed, matrix, detector_schedules = seed_data
            seed_regime_rows: list[dict[str, Any]] = []
            seed_summary_rows: list[dict[str, Any]] = []
            seed_retrain_events: list[dict[str, Any]] = []
            seed_batch_rows: list[dict[str, Any]] = []

            for pol in policies:
                grid_costs: list[float] = []
                grid_retrains: list[float] = []
                grid_accs: list[float] = []
                for alpha in alpha_grid:
                    if pol in detector_schedules:
                        metrics = _evaluate_schedule(matrix, detector_schedules[pol], alpha)
                    else:
                        metrics = _replay_rule(
                            matrix,
                            pol,
                            alpha,
                            seed,
                            config.upf_delta,
                            config.upf_mc_samples,
                        )
                    grid_costs.append(metrics["cost"])
                    grid_retrains.append(metrics["retrains"])
                    grid_accs.append(metrics["average_accuracy"])
                    seed_regime_rows.append(
                        {
                            "seed": seed,
                            "policy": pol,
                            "regime": "price_grid",
                            "alpha": float(alpha),
                            **metrics,
                        }
                    )

                for reg_idx, (reg_name, alpha_val) in enumerate(regimes.items()):
                    if pol in detector_schedules:
                        metrics, executed = _evaluate_schedule_with_retrains(
                            matrix, detector_schedules[pol], alpha_val
                        )
                    else:
                        metrics, executed = _replay_rule_with_retrains(
                            matrix,
                            pol,
                            alpha_val,
                            seed,
                            config.upf_delta,
                            config.upf_mc_samples,
                        )
                    seed_regime_rows.append(
                        {
                            "seed": seed,
                            "policy": pol,
                            "regime": reg_name,
                            "alpha": float(alpha_val),
                            **metrics,
                        }
                    )
                    if reg_idx == 0:
                        retrain_set = set(executed)
                        deployed = matrix.initial
                        cum_cost = 0.0
                        for t in range(matrix.offline, matrix.batches):
                            is_ret = t in retrain_set
                            if is_ret:
                                deployed = t
                            err_t = float(matrix.err[deployed, t])
                            acc_t = 1.0 - err_t
                            stale_t = float(matrix.staleness(t, deployed)) if deployed < t else 0.0
                            cum_cost += err_t + (float(alpha_val) if is_ret else 0.0)
                            fit_sec = (
                                float(matrix.fit_seconds[t])
                                if is_ret and t < len(matrix.fit_seconds)
                                else 0.0
                            )
                            seed_batch_rows.append(
                                {
                                    "seed": seed,
                                    "policy": pol,
                                    "regime": reg_name,
                                    "batch_index": int(t),
                                    "accuracy": acc_t,
                                    "error": err_t,
                                    "staleness": stale_t if math.isfinite(stale_t) else 0.0,
                                    "model_age": int(t - deployed),
                                    "deployed_model": int(deployed),
                                    "cumulative_cost": cum_cost,
                                    "retrain_decision": is_ret,
                                    "fit_time_s": fit_sec,
                                }
                            )
                    for v_idx, b in enumerate(executed, start=1):
                        fit_sec = float(matrix.fit_seconds[b]) if b < len(matrix.fit_seconds) else 0.0
                        samples_n = int(matrix.rows[b]) if b < len(matrix.rows) else 0
                        post_err = float(matrix.err[b, b]) if b < matrix.err.shape[0] else 0.0
                        v_name = f"{pol}-s{seed}-{reg_name}-v{v_idx}-b{int(b)}"
                        seed_retrain_events.append(
                            {
                                "seed": seed,
                                "policy": pol,
                                "version_index": v_idx,
                                "version_name": v_name,
                                "batch_index": int(b),
                                "step": int(b),
                                "regime": reg_name,
                                "alpha": float(alpha_val),
                                "retrain_cost": float(alpha_val),
                                "fit_seconds": fit_sec,
                                "train_seconds": fit_sec,
                                "train_samples": samples_n,
                                "window_size": samples_n,
                                "post_retrain_error": post_err,
                                "post_retrain_accuracy": 1.0 - post_err,
                            }
                        )

                auc_val = (
                    trapezoid_auc(grid_costs, alpha_grid) if len(alpha_grid) >= 2 else float(grid_costs[0])
                )
                seed_summary_rows.append(
                    {
                        "seed": seed,
                        "policy": pol,
                        "regime": "price_grid",
                        "combined_cost_auc": float(auc_val),
                        "retrains": float(np.mean(grid_retrains)),
                        "average_accuracy": float(np.mean(grid_accs)),
                    }
                )

            return seed_regime_rows, seed_summary_rows, seed_retrain_events, seed_batch_rows

        tasks = [
            lambda sd=sd: tracker.run_seed_scoring(sd[0], policies, lambda: _score_seed(sd))
            for sd in seed_matrices
        ]
        seed_outputs = backend.run_tasks(tasks, max_workers=config.max_workers)

        all_regime_records: list[dict[str, Any]] = []
        all_seed_summaries: list[dict[str, Any]] = []
        for reg_rows, sum_rows, ret_events, batch_rows in seed_outputs:
            all_regime_records.extend(reg_rows)
            all_seed_summaries.extend(sum_rows)
            for ev in ret_events:
                tracker.log_retrain_event(ev["seed"], ev["policy"], ev)
            for br in batch_rows:
                tracker.log_batch(
                    br["seed"],
                    br["policy"],
                    br["regime"],
                    {k: v for k, v in br.items() if k not in {"seed", "policy", "regime"}},
                )

        log_protocol_outputs(tracker, all_seed_summaries, all_regime_records)

        aggregate_rows: list[dict[str, Any]] = aggregate_seed_summary(
            all_seed_summaries,
            protocol="upf",
            dataset=dataset_canonical,
            policies=policies,
            metrics=("combined_cost_auc",),
            regime="price_grid",
        )
        for reg_name in regimes:
            aggregate_rows.extend(
                aggregate_seed_summary(
                    all_regime_records,
                    protocol="upf",
                    dataset=dataset_canonical,
                    policies=policies,
                    metrics=("cost", "retrains", "average_accuracy"),
                    regime=reg_name,
                )
            )

        tracker.log_aggregate_summary(aggregate_rows)
        return ProtocolResult(
            protocol="upf",
            dataset=dataset_canonical,
            seed_summaries=all_seed_summaries,
            regime_records=all_regime_records,
            aggregate_summary=aggregate_rows,
            artifacts={},
        )
