"""CARA (2024) benchmark protocol implementation."""

from __future__ import annotations

import gzip
import json
import logging
import math
import time
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.sparse import issparse

from continual_bench.backends.base import ExecutionBackend
from continual_bench.datasets.streams_unified import load_stream_batches
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
)
from continual_bench.retraining.cara_reference import (
    fit_period,
    fit_threshold,
    oracle_schedule,
    periodic_schedule,
    strategy_cost,
    strategy_scpe,
    threshold_schedule,
)
from continual_bench.retraining.schedule import alpha_max_same_batch
from continual_bench.tracking.tracker import BenchmarkTracker

logger = logging.getLogger(__name__)

_DEFAULT_CARA_POLICIES = (
    "cara",
    "cara_cumulative",
    "cara_periodic",
    "oracle_staleness",
    "frozen",
    "markov",
    "adwin",
    "ddm",
)
_DRIFT_POLICIES = frozenset({"adwin", "ddm"})


def _default_kappa_grid(dataset_name: str) -> list[float]:
    canonical = canonical_dataset_name(dataset_name)
    if canonical == "electricity":
        return [float(v) for v in np.linspace(0.5, 70.0, 99)]
    if canonical == "airlines":
        return [float(v) for v in np.linspace(5.0, 700.0, 99)]
    return [float(v) for v in np.linspace(0.5, 50.0, 99)]


def _row_norms_sq(X: Any) -> np.ndarray:
    if issparse(X):
        return np.asarray(X.power(2).sum(axis=1), dtype=np.float64).ravel()
    arr = np.asarray(X, dtype=np.float64)
    return np.sum(arr * arr, axis=1)


def _rbf_kernel(Q: Any, X: Any, gamma: float) -> np.ndarray:
    q_norm = _row_norms_sq(Q)[:, None]
    x_norm = _row_norms_sq(X)[None, :]
    dot = Q @ X.T
    if issparse(dot):
        dot = dot.toarray()
    dist2 = np.maximum(q_norm + x_norm - 2.0 * np.asarray(dot, dtype=np.float64), 0.0)
    return np.exp(-gamma * dist2)


def _average_accuracy_for_schedule(
    accuracy_online: np.ndarray,
    retrains: list[int],
    *,
    reverse_order: bool = False,
) -> float:
    """Compute CARA prequential test-then-train accuracy with final-batch clipping."""
    online_count = len(accuracy_online)
    if online_count <= 0:
        return 0.0
    accuracy_schedule = retrains[::-1] if reverse_order else retrains
    deployed = np.zeros(online_count, dtype=int)
    for r in accuracy_schedule:
        deployed[min(r + 1, online_count - 1)] = r
    return float(accuracy_online[np.maximum.accumulate(deployed), np.arange(online_count)].mean())


def _resolve_kernel_gamma(config: CampaignConfig, canonical_dataset: str, n_features: int) -> float:
    if config.kernel_gamma is not None:
        return float(config.kernel_gamma)
    if canonical_dataset == "electricity":
        return 20.0
    return 1.0 / max(1, n_features)


class CaraProtocol(BenchmarkProtocol):
    """Strict CARA (Mahadevan & Mathioudakis 2024) protocol implementation."""

    def _prepare_context(
        self, config: CampaignConfig
    ) -> tuple[
        list[Any],
        int,
        float,
        tuple[tuple[int, int], ...],
        dict[int, np.ndarray],
        dict[tuple[int, int], np.ndarray],
    ]:
        dataset_canonical = canonical_dataset_name(config.dataset.name)
        spec = config.dataset.model_copy(update={"protocol": "cara"})
        batches = load_stream_batches(spec)
        n_batches = len(batches)
        offline = (
            min(max(2, spec.offline_batches), n_batches - 2)
            if n_batches >= 4
            else max(1, n_batches // 4)
        )
        n_features = batches[0].X_train.shape[1]
        gamma = _resolve_kernel_gamma(config, dataset_canonical, n_features)

        w_self: dict[int, np.ndarray] = {}
        w_pair: dict[tuple[int, int], np.ndarray] = {}
        phase_ranges: tuple[tuple[int, int], ...] = ((0, offline), (offline, n_batches))
        for phase_start, phase_end in phase_ranges:
            for t in range(phase_start + 1, phase_end):
                b_t = batches[t]
                queries = b_t.X_query if b_t.X_query.shape[0] > 0 else b_t.X_test
                k_s = _rbf_kernel(queries, b_t.X_train, gamma)
                w_self[t] = k_s.sum(axis=0) / float(max(1, b_t.X_train.shape[0]))
                if config.staleness_mode == "relative":
                    for tp in range(phase_start, t):
                        k_p = _rbf_kernel(queries, batches[tp].X_train, gamma)
                        w_pair[(tp, t)] = k_p.sum(axis=0) / float(
                            max(1, batches[tp].X_train.shape[0])
                        )
        return batches, offline, gamma, phase_ranges, w_self, w_pair

    def build_seed_matrix(
        self,
        config: CampaignConfig,
        seed: int,
        *,
        context: tuple[Any, ...] | None = None,
        checkpoint_backend: Any | None = None,
        checkpoint_rel_path: str | None = None,
        stop_requested: Any | None = None,
        checkpoint_interval_seconds: float = 30.0,
    ) -> tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, list[int]],
        list[float],
        list[int],
    ]:
        """Build performance and staleness matrices for a single seed with Spot checkpointing."""
        policies = list(config.policies) if config.policies else list(_DEFAULT_CARA_POLICIES)
        batches, offline, _, phase_ranges, w_self, w_pair = (
            context if context is not None else self._prepare_context(config)
        )
        n_batches = len(batches)
        model_cfg = config.model.model_copy(update={"seed": seed})
        learner = AdapterLearner(model_cfg)
        models: list[ModelAdapter] = []
        train_losses: list[np.ndarray] = []
        fit_times: list[float] = []
        sample_counts: list[int] = []
        for b in batches:
            if stop_requested is not None and stop_requested():
                raise SpotInterruptedError(f"Spot interruption during model fit for seed {seed}")
            t0 = time.perf_counter()
            adapter = learner.fit(b.batch_index, b.X_train, b.y_train)
            fit_times.append(float(time.perf_counter() - t0))
            sample_counts.append(int(len(b.y_train)))
            models.append(adapter)
            train_losses.append(learner.losses(adapter, b.X_train, b.y_train))

        acc_matrix = np.zeros((n_batches, n_batches), dtype=np.float64)
        psi_matrix = np.zeros((n_batches, n_batches), dtype=np.float64)
        completed_batches: set[int] = set()

        if checkpoint_backend is not None and checkpoint_rel_path:
            raw_ckpt = checkpoint_backend.read_bytes(checkpoint_rel_path)
            if raw_ckpt is not None:
                ckpt = json.loads(gzip.decompress(raw_ckpt).decode("utf-8"))
                if int(ckpt.get("seed", -1)) == seed and len(ckpt.get("acc_matrix", [])) == n_batches:
                    acc_matrix = np.asarray(ckpt["acc_matrix"], dtype=np.float64)
                    psi_matrix = np.asarray(ckpt["psi_matrix"], dtype=np.float64)
                    completed_batches = {int(idx) for idx in ckpt.get("completed_batches", [])}

        def _flush_checkpoint() -> None:
            if checkpoint_backend is None or not checkpoint_rel_path:
                return
            payload = {
                "seed": seed,
                "n_batches": n_batches,
                "completed_batches": sorted(completed_batches),
                "acc_matrix": acc_matrix.tolist(),
                "psi_matrix": psi_matrix.tolist(),
            }
            checkpoint_backend.write_bytes(
                checkpoint_rel_path,
                gzip.compress(json.dumps(payload).encode("utf-8")),
            )

        last_ckpt = time.monotonic()
        for phase_start, phase_end in phase_ranges:
            for t in range(phase_start, phase_end):
                if t in completed_batches:
                    continue
                b_t = batches[t]
                eval_X = b_t.X_test if len(b_t.y_test) > 0 else b_t.X_query
                eval_y = b_t.y_test if len(b_t.y_test) > 0 else b_t.y_query
                for tp in range(phase_start, t + 1):
                    if phase_start >= offline:
                        preds_te = models[tp].predict(eval_X)
                        acc_matrix[tp, t] = float(np.mean(preds_te == eval_y))
                    if tp < t:
                        loss_on_t = learner.losses(models[tp], b_t.X_train, b_t.y_train)
                        psi_t = float(w_self[t] @ loss_on_t) if loss_on_t.size else 0.0
                        if config.staleness_mode == "relative":
                            psi_tp = (
                                float(w_pair[(tp, t)] @ train_losses[tp])
                                if train_losses[tp].size
                                else 0.0
                            )
                            psi_matrix[tp, t] = psi_t - psi_tp
                        else:
                            psi_matrix[tp, t] = psi_t
                completed_batches.add(t)
                stopping = bool(stop_requested()) if stop_requested is not None else False
                if checkpoint_backend is not None and checkpoint_rel_path:
                    if stopping or (time.monotonic() - last_ckpt >= checkpoint_interval_seconds):
                        _flush_checkpoint()
                        last_ckpt = time.monotonic()
                if stopping:
                    raise SpotInterruptedError(
                        f"Spot interruption after batch {t} for seed {seed}"
                    )

        offline_cost = np.nan_to_num(psi_matrix[:offline, :offline], nan=0.0)
        online_cost = np.nan_to_num(psi_matrix[offline:, offline:], nan=0.0)
        accuracy_online = acc_matrix[offline:, offline:]

        detector_retrains: dict[str, list[int]] = {}
        for pol in policies:
            if pol in _DRIFT_POLICIES:
                detector_retrains[pol] = run_error_detector_schedule(
                    pol,
                    batches,
                    models,
                    offline,
                    backend="skmultiflow",
                    reset_on_drift=False,
                    warm_start_on_reset=False,
                    initial_model=offline,
                    eval_split="train",
                    include_offline_in_schedule=False,
                )
        return (
            seed,
            offline_cost,
            online_cost,
            accuracy_online,
            detector_retrains,
            fit_times,
            sample_counts,
        )

    def run_seed_matrix(
        self,
        config: CampaignConfig,
        seed: int,
        tracker: BenchmarkTracker,
        *,
        context: tuple[Any, ...] | None = None,
    ) -> tuple:
        return tracker.run_matrix(
            seed,
            lambda: self.build_seed_matrix(config, seed, context=context),
            lambda matrix: matrix[5],
        )

    def run(
        self,
        config: CampaignConfig,
        backend: ExecutionBackend,
        tracker: BenchmarkTracker,
    ) -> ProtocolResult:
        seeds = list(config.seeds) if config.seeds else [0, 1, 2, 3, 4]
        context = self._prepare_context(config)
        seed_matrices = backend.run_tasks(
            [lambda s=s: self.run_seed_matrix(config, s, tracker, context=context) for s in seeds],
            max_workers=config.max_workers,
        )
        return self.score_and_finalize(
            config,
            seed_matrices,
            backend,
            tracker,
            gamma=context[2],
            offline=context[1],
        )

    def score_and_finalize(
        self,
        config: CampaignConfig,
        seed_matrices: list[Any],
        backend: ExecutionBackend,
        tracker: BenchmarkTracker,
        *,
        gamma: float | None = None,
        offline: int | None = None,
    ) -> ProtocolResult:
        policies = list(config.policies) if config.policies else list(_DEFAULT_CARA_POLICIES)
        dataset_canonical = canonical_dataset_name(config.dataset.name)
        if offline is None:
            offline = (
                int(seed_matrices[0][1].shape[0])
                if seed_matrices
                else int(config.dataset.offline_batches)
            )
        if gamma is None:
            gamma = float(config.kernel_gamma or 1.0)

        if config.kappa_grid:
            kappa_grid = [float(v) for v in config.kappa_grid]
        else:
            tops: list[int] = []
            for _, offline_cost, _, _, _, _, _ in seed_matrices:
                if offline_cost.shape[0] >= 2:
                    safe_offline = np.nan_to_num(offline_cost, nan=0.0)
                    bound = alpha_max_same_batch(
                        safe_offline.T, 0, 1, len(safe_offline) - 1
                    )
                    if math.isfinite(bound) and bound > 0:
                        tops.append(math.floor(bound) + 1)
            positive_tops = [t for t in tops if t > 0]
            if positive_tops:
                top = min(positive_tops)
                kappa_grid = [float(v) for v in np.linspace(0.0, float(top), 100)[1:]]
            else:
                kappa_grid = _default_kappa_grid(config.dataset.name)

        tracker.log_metadata(
            {
                "gamma": float(gamma),
                "kappa_min": float(min(kappa_grid)) if kappa_grid else 0.0,
                "kappa_max": float(max(kappa_grid)) if kappa_grid else 0.0,
            }
        )

        def _score_seed(
            seed_data: tuple[
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                dict[str, list[int]],
                list[float],
                list[int],
            ],
        ) -> tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]:
            seed, offline_cost, online_cost, accuracy_online, detector_retrains, fit_times, sample_counts = (
                seed_data
            )
            safe_offline = np.nan_to_num(offline_cost, nan=0.0)
            safe_online = np.nan_to_num(online_cost, nan=0.0)
            online_count = len(safe_online)
            seed_regime_rows: list[dict[str, Any]] = []
            seed_retrain_events: list[dict[str, Any]] = []
            seed_batch_rows: list[dict[str, Any]] = []
            policy_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
            rep_kappa_idx = len(kappa_grid) // 2 if kappa_grid else 0

            for k_idx, kappa in enumerate(kappa_grid):
                oracle_sched = oracle_schedule(safe_online, kappa)
                oracle_cost = strategy_cost(safe_online, oracle_sched, kappa)

                for pol in policies:
                    if pol == "cara":
                        tau = fit_threshold(safe_offline, kappa, cumulative=False, seed=seed)
                        retrains = threshold_schedule(safe_online, tau, cumulative=False)
                    elif pol == "cara_cumulative":
                        tau_cum = fit_threshold(safe_offline, kappa, cumulative=True, seed=seed)
                        retrains = threshold_schedule(safe_online, tau_cum, cumulative=True)
                    elif pol == "cara_periodic":
                        period, offset = fit_period(safe_offline, kappa)
                        retrains = periodic_schedule(online_count, period, offset=offset)
                    elif pol == "oracle_staleness":
                        retrains = oracle_sched
                    elif pol == "frozen":
                        retrains = [0]
                    elif pol == "markov":
                        retrains = threshold_schedule(safe_online, kappa, cumulative=False)
                    elif pol == "always_retrain":
                        retrains = list(range(online_count))
                    elif pol in detector_retrains:
                        retrains = detector_retrains[pol]
                    else:
                        raise ValueError(f"Unsupported CARA policy: {pol!r}")

                    cost_val = strategy_cost(safe_online, retrains, kappa)
                    scpe_val = strategy_scpe(cost_val, oracle_cost)
                    avg_acc = _average_accuracy_for_schedule(
                        accuracy_online,
                        retrains,
                        reverse_order=(pol == "oracle_staleness"),
                    )
                    n_retrains = (
                        float(len(retrains))
                        if pol in _DRIFT_POLICIES
                        else float(max(0, len(retrains) - 1))
                    )

                    rec = {
                        "seed": seed,
                        "policy": pol,
                        "regime": "price_grid",
                        "kappa": float(kappa),
                        "strategy_cost": float(cost_val),
                        "oracle_cost": float(oracle_cost),
                        "strategy_scpe": float(scpe_val) if scpe_val is not None else 0.0,
                        "average_accuracy": float(avg_acc),
                        "retrains": n_retrains,
                    }
                    seed_regime_rows.append(rec)
                    policy_metrics[pol].append(rec)

                    if k_idx == rep_kappa_idx:
                        actual_retrains = [r for r in retrains if (pol in _DRIFT_POLICIES or r > 0)]
                        retrain_set = set(actual_retrains)
                        deployed_rel = 0
                        cum_cost = 0.0
                        for t_rel in range(online_count):
                            if t_rel > 0 and (t_rel - 1) in retrain_set:
                                deployed_rel = t_rel - 1
                            acc_t = (
                                float(accuracy_online[deployed_rel, t_rel])
                                if online_count > 0
                                else 0.0
                            )
                            err_t = 1.0 - acc_t
                            is_ret = t_rel in retrain_set
                            stale_t = (
                                float(safe_online[deployed_rel, t_rel])
                                if deployed_rel < t_rel
                                else 0.0
                            )
                            cum_cost += err_t + (float(kappa) if is_ret else 0.0)
                            b_idx = offline + t_rel
                            seed_batch_rows.append(
                                {
                                    "seed": seed,
                                    "policy": pol,
                                    "regime": "price_grid",
                                    "batch_index": b_idx,
                                    "accuracy": acc_t,
                                    "error": err_t,
                                    "staleness": stale_t,
                                    "model_age": t_rel - deployed_rel,
                                    "deployed_model": offline + deployed_rel,
                                    "cumulative_cost": cum_cost,
                                    "retrain_decision": is_ret,
                                    "fit_time_s": (
                                        fit_times[b_idx] if is_ret and b_idx < len(fit_times) else 0.0
                                    ),
                                }
                            )

                        for v_idx, r in enumerate(actual_retrains, start=1):
                            batch_idx = offline + r
                            post_acc = float(accuracy_online[r, r]) if r < online_count else 0.0
                            fit_sec = (
                                fit_times[batch_idx] if batch_idx < len(fit_times) else 0.0
                            )
                            v_name = f"{pol}-s{seed}-v{v_idx}-b{batch_idx}"
                            seed_retrain_events.append(
                                {
                                    "seed": seed,
                                    "policy": pol,
                                    "version_index": v_idx,
                                    "version_name": v_name,
                                    "batch_index": batch_idx,
                                    "step": batch_idx,
                                    "regime": "price_grid",
                                    "kappa": float(kappa),
                                    "retrain_cost": float(kappa),
                                    "fit_seconds": fit_sec,
                                    "train_seconds": fit_sec,
                                    "train_samples": (
                                        sample_counts[batch_idx] if batch_idx < len(sample_counts) else 0
                                    ),
                                    "window_size": (
                                        sample_counts[batch_idx] if batch_idx < len(sample_counts) else 0
                                    ),
                                    "post_retrain_accuracy": post_acc,
                                    "post_retrain_error": 1.0 - post_acc,
                                    "staleness": float(safe_online[max(0, r - 1), r]) if r > 0 else 0.0,
                                }
                            )

            seed_summary_rows: list[dict[str, Any]] = []
            for pol in policies:
                rows = policy_metrics[pol]
                seed_summary_rows.append(
                    {
                        "seed": seed,
                        "policy": pol,
                        "regime": "price_grid",
                        "strategy_scpe": float(np.mean([r["strategy_scpe"] for r in rows])),
                        "average_accuracy": float(np.mean([r["average_accuracy"] for r in rows])),
                        "retrains": float(np.mean([r["retrains"] for r in rows])),
                        "strategy_cost": float(np.mean([r["strategy_cost"] for r in rows])),
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

        aggregate_rows = aggregate_seed_summary(
            all_seed_summaries,
            protocol="cara",
            dataset=dataset_canonical,
            policies=policies,
            metrics=("strategy_scpe", "average_accuracy", "retrains", "strategy_cost"),
            regime="price_grid",
        )
        tracker.log_aggregate_summary(aggregate_rows)

        return ProtocolResult(
            protocol="cara",
            dataset=dataset_canonical,
            seed_summaries=all_seed_summaries,
            regime_records=all_regime_records,
            aggregate_summary=aggregate_rows,
            artifacts={},
        )
