"""CARA benchmark using the experiment protocol in the authors' source archive."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continual_bench.datasets import retraining_benchmarks as data
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

from .epoch_history import pickle_state
from .retraining_observations import candidate_report, sample_identity
from .retraining_policy_tracking import policy_report, repetition_seeds

BATCHES, OFFLINE, GRID = 100, 25, 100
REFERENCE = "ac2a600a3bbc5729087318087c9def96104c9023"
RULES = ("cara", "cara_cumulative", "cara_periodic", "markov", "frozen", "oracle_staleness", "always_retrain")
DRIFT_RULES = ("adwin", "ddm")
DATASETS = (
    "covcon-d",
    "covcon-s",
    "circle-d",
    "circle-s",
    "gauss-d",
    "gauss-s",
    "electricity",
    "airlines",
    "covertype",
)
# Tables 2-4 of the paper: CARA-T, CARA-CT, CARA-P, Oracle (retrains/accuracy), NR, Markov.
PAPER = {
    "scpe": {
        "covcon-d": (17.72, 19.26, 16.3),
        "covcon-s": (15.58, 22.94, 104.79),
        "circle-d": (45.61, 47.29, 84.96),
        "circle-s": (33.89, 45.15, 79.34),
        "gauss-d": (8.63, 59.93, 135.95),
        "gauss-s": (66.62, 96.38, 202.09),
        "electricity": (8.32, 40.32, 64.07),
        "airlines": (6.11, 5.48, 5.45),
        "covertype": (26.39, 27.92, 38.37),
    },
    "accuracy_oracle": {
        "covcon-d": 0.76,
        "covcon-s": 0.8,
        "circle-d": 0.97,
        "circle-s": 0.96,
        "gauss-d": 0.92,
        "gauss-s": 1.0,
        "electricity": 0.71,
        "airlines": 0.62,
        "covertype": 0.76,
    },
    "accuracy_cara_t": {
        "covcon-d": 0.7,
        "covcon-s": 0.79,
        "circle-d": 0.96,
        "circle-s": 0.93,
        "gauss-d": 0.9,
        "gauss-s": 0.99,
        "electricity": 0.68,
        "airlines": 0.6,
        "covertype": 0.66,
    },
    "retrains_oracle": {
        "covcon-d": 11.2,
        "covcon-s": 8.21,
        "circle-d": 3.89,
        "circle-s": 3.09,
        "gauss-d": 2.23,
        "gauss-s": 1.92,
        "electricity": 2.36,
        "airlines": 25.23,
        "covertype": 19.42,
    },
    "retrains_cara_t": {
        "covcon-d": 13.89,
        "covcon-s": 10.05,
        "circle-d": 5.67,
        "circle-s": 3.56,
        "gauss-d": 2.62,
        "gauss-s": 2.57,
        "electricity": 2.39,
        "airlines": 27.69,
        "covertype": 16.62,
    },
}


class ForestLearner:
    observation_model = "random_forest"
    observation_threshold = None  # Preserve the estimator's argmax and tie rule.
    observation_step_kind = "sklearn_fit_call"

    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, index, x, y):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=self.seed, n_jobs=1).fit(x, y)

    def losses(self, state, x, y):
        return (self.predict(state, x) != y).astype(np.float64)

    @staticmethod
    def predict(forest, x):
        """``forest.predict`` without the per-tree validation overhead (identical output).

        scikit-learn validates the input once per tree; with 100 trees on ~1000 rows that overhead is
        ~10x the prediction, and the matrix needs about 5,000 forest evaluations per seed.
        """
        from scipy.sparse import issparse

        if issparse(x):
            return forest.predict(x)
        scores = ForestLearner._predict_scores(forest, x)
        return forest.classes_[scores.argmax(axis=1)]

    @staticmethod
    def _predict_scores(forest, x):
        rows = np.asarray(x, dtype=np.float32, order="C")
        total = np.zeros((len(rows), len(forest.classes_)))
        for tree in forest.estimators_:
            counts = tree.tree_.predict(rows)
            counts = counts[:, 0, :] if counts.ndim == 3 else counts
            total += counts / counts.sum(axis=1, keepdims=True)
        return total

    @staticmethod
    def observation_predict(forest, x, positive):
        from scipy.sparse import issparse

        if issparse(x):
            totals, denominator = forest.predict_proba(x), 1
        else:
            totals, denominator = ForestLearner._predict_scores(forest, x), len(forest.estimators_)
        prediction = forest.classes_[totals.argmax(axis=1)]
        positions = forest.classes_.tolist()
        if positive is None:
            return prediction, None
        scores = (
            totals[:, positions.index(positive)] / denominator
            if positive in positions
            else np.zeros(x.shape[0])
        )
        return prediction, scores

    @staticmethod
    def observation_durations(_index, elapsed):
        return [elapsed]

    @staticmethod
    def observation_checkpoint(state):
        return pickle_state(state), "pkl"


def batches_for(
    name,
    *,
    directory="~/.cache/continual-bench/datasets",
    electricity_path=None,
    airlines_path=None,
    covertype_path=None,
    batches=BATCHES,
):
    base, _, variant = name.partition("-")
    if base in ("gauss", "circle", "covcon"):
        stream = getattr(data, f"cara_{base}")(batches)
        queries = data.cara_static_queries(base, batches) if variant == "s" else None
        return stream, queries, 20.0
    paths = {"electricity": electricity_path, "airlines": airlines_path, "covertype": covertype_path}
    files = {
        "electricity": "elec.csv",
        "airlines": "airlines.csv",
        "covertype": "covtype.libsvm.binary.scale.bz2",
    }
    path = Path(paths[base] or Path(directory).expanduser() / files[base]).expanduser()
    return data.cara_real_batches(base, path, batches), None, None if base == "airlines" else 20.0


@dataclass
class Phase:
    stale: np.ndarray
    accuracy: np.ndarray
    train_errors: list[list[np.ndarray]]  # indexed by batch, then model
    tracking: dict | None = None
    drift_updates: dict | None = None


def build_phase(
    batches,
    learner,
    gamma,
    queries=None,
    *,
    workers=1,
    drift_baselines=False,
    tracking_dataset=None,
    phase_name=None,
    initial=None,
    drift_backend=None,
    metrics_only=False,
):
    """Train and score each candidate inside a single phase, using the reference's 90/10 split."""
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.model_selection import train_test_split

    started = time.perf_counter()
    splits = [train_test_split(x, y, test_size=0.1, random_state=0) for x, y in batches]
    query_stream = [(q, labels) for _, q, _, labels in splits] if queries is None else queries
    models, fit_seconds = [], []
    for i, (x, _, y, _) in enumerate(splits):
        before = time.perf_counter()
        models.append(learner.fit(i, x, y))
        fit_seconds.append(time.perf_counter() - before)
    own_errors = [
        learner.losses(model, x, y).astype(bool) for model, (x, _, y, _) in zip(models, splits, strict=True)
    ]

    def score(t):
        x, _, y, _ = splits[t]
        query, labels = query_stream[t]
        # The new-data kernel is shared by all models in this column. Average over training
        # points only, then sum query costs in the same units as the reference.
        weights = rbf_kernel(query, x, gamma=gamma).sum(axis=0) / len(y)
        stale, accuracy, errors = [], [], []
        for m, model in enumerate(models[: t + 1]):
            accuracy.append(1.0 - learner.losses(model, query, labels).mean())
            loss = own_errors[m] if m == t else learner.losses(model, x, y).astype(bool)
            if drift_baselines or drift_backend is not None:
                errors.append(loss)
            if m == t:
                stale.append(0.0)
                continue
            old_x, old_loss = splits[m][0], own_errors[m]
            old_cost = (
                rbf_kernel(query, old_x[old_loss], gamma=gamma).sum() / len(old_loss)
                if old_loss.any()
                else 0.0
            )
            stale.append(float(weights[loss].sum() - old_cost))
        return stale, accuracy, errors

    size = len(splits)
    stale, accuracy = (np.full((size, size), np.nan) for _ in range(2))
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, (costs, scores, losses) in enumerate(pool.map(score, range(size))):
            stale[: t + 1, t] = costs
            accuracy[: t + 1, t] = scores
            errors.append(losses)
    phase = Phase(stale, accuracy, errors)
    drift = None
    if drift_backend is not None:
        drift = {"detector_backend": drift_backend, "alarms": [], "coordinate": "batch_number"}
        phase.drift_updates = drift_schedules(phase, drift_backend, alarms=drift["alarms"])
        for detector in phase.drift_updates:
            drift[f"{detector}_alarm_count"] = sum(event["detector"] == detector for event in drift["alarms"])
    if tracking_dataset is not None:
        phase.tracking = candidate_report(
            tracking_dataset,
            learner.seed,
            [(x, y, test, labels) for x, test, y, labels in splits],
            models,
            learner,
            fit_seconds,
            protocol=f"CARA-paper-{phase_name}-candidates",
            initial=initial,
            duration_before_tracking=time.perf_counter() - started,
            drift=drift,
            evaluation={
                "rbf_gamma": gamma,
                "policy_query_scope": "static_queries" if queries is not None else "batch_holdout",
                "static_queries_sha256": sample_identity(queries) if queries is not None else None,
            },
            artifact_mode="metrics_only" if metrics_only else "full",
        )
    return phase


def prequential_accuracy(accuracy, retrains):
    """Match the authors' report, including clipping the last deployment to the final batch."""
    batches = np.arange(len(accuracy))
    deployed = np.zeros(len(accuracy), dtype=int)
    for retrain in retrains:
        deployed[min(retrain + 1, len(accuracy) - 1)] = retrain
    return float(accuracy[np.maximum.accumulate(deployed), batches].mean())


def drift_detectors(backend):
    if backend == "skmultiflow":
        from skmultiflow.drift_detection import ADWIN, DDM
    else:
        from river.drift import ADWIN
        from river.drift.binary import DDM

    return (("adwin", ADWIN()), ("ddm", DDM()))


def drift_schedules(phase, backend, *, alarms=None):
    schedules = {}
    for name, detector in drift_detectors(backend):
        model, retrains = 0, []
        detected = False
        for t, losses in enumerate(phase.train_errors):
            if detected:
                model = t - 1
            detected = False
            for index, loss in enumerate(losses[model], 1):
                if backend == "skmultiflow":
                    detector.add_element(int(loss))
                    alarm = detector.detected_change()
                else:
                    detector.update(int(loss))
                    alarm = detector.drift_detected
                detected |= alarm
                if alarm and alarms is not None:
                    alarms.append({"seq": t + 1, "offset_in_batch": index, "detector": name})
            if detected:
                retrains.append(t)
        schedules[name] = retrains
    return schedules


def evaluate_kappa(offline, online, kappa, seed, baseline_schedules=None):
    tau = fit_threshold(offline.stale, kappa, cumulative=False, seed=seed)
    tau_cum = fit_threshold(offline.stale, kappa, cumulative=True, seed=seed)
    period, offset = fit_period(offline.stale, kappa)
    stale = online.stale
    schedules = {
        "cara": threshold_schedule(stale, tau),
        "cara_cumulative": threshold_schedule(stale, tau_cum, cumulative=True),
        "cara_periodic": periodic_schedule(len(stale), period, offset),
        "markov": threshold_schedule(stale, kappa),
        "frozen": [0],
        "oracle_staleness": oracle_schedule(stale, kappa),
        "always_retrain": list(range(len(stale))),
        **(baseline_schedules or {}),
    }
    optimum = strategy_cost(stale, schedules["oracle_staleness"], kappa)
    report = {}
    parameters = {
        "cara": {"threshold": tau},
        "cara_cumulative": {"threshold": tau_cum},
        "cara_periodic": {"period": period, "offset": offset},
        "markov": {"threshold": kappa},
    }
    for rule, schedule in schedules.items():
        cost = strategy_cost(stale, schedule, kappa)
        # The reference backtracks Oracle in reverse order. If two deployments collide
        # at the final batch after clipping, the assignment order affects accuracy.
        accuracy_schedule = schedule[::-1] if rule == "oracle_staleness" else schedule
        retrain_count = len(schedule) if rule in DRIFT_RULES else len(schedule) - 1
        deployed = np.zeros(len(stale), dtype=int)
        for retrain in accuracy_schedule:
            deployed[min(retrain + 1, len(stale) - 1)] = retrain
        report[rule] = {
            "strategy_cost": cost,
            "strategy_scpe": strategy_scpe(cost, optimum),
            "average_accuracy": prequential_accuracy(online.accuracy, accuracy_schedule),
            "retrains": retrain_count,
            "retrain_points": list(schedule),
            "served_by": np.maximum.accumulate(deployed).tolist(),
            "parameters": parameters.get(rule, {}),
        }
    return report


def _mean(values):
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def _matrix_values(matrix):
    return [[float(value) if math.isfinite(value) else None for value in row] for row in matrix]


def replicate(
    name,
    seeds=5,
    *,
    kappa_points=GRID,
    kappa_max=None,
    progress=None,
    workers=1,
    drift_backend=None,
    metrics_only=False,
    **options,
):
    """Evaluate the CARA variants against a retrospective phase-local Oracle."""
    seed_values = repetition_seeds(seeds)
    if kappa_points < 2 or workers < 1:
        raise ValueError("CARA needs at least one seed and worker, and at least two kappa points")
    if kappa_max is not None and (not math.isfinite(kappa_max) or kappa_max <= 0):
        raise ValueError("kappa_max must be positive and finite")
    if drift_backend not in (None, "skmultiflow", "river"):
        raise ValueError("unknown CARA drift detector backend")
    if drift_backend is not None:
        drift_detectors(drift_backend)
    emit = progress or (lambda event: None)
    batches, queries, gamma = batches_for(name, **options)
    offline_queries = None if queries is None else queries[:OFFLINE]
    online_queries = None if queries is None else queries[OFFLINE:]
    matrices = []
    for seed in seed_values:
        learner = ForestLearner(seed)
        offline = build_phase(
            batches[:OFFLINE],
            learner,
            gamma,
            offline_queries,
            workers=workers,
            tracking_dataset=name,
            phase_name="offline",
            metrics_only=metrics_only,
        )
        online = build_phase(
            batches[OFFLINE:],
            learner,
            gamma,
            online_queries,
            workers=workers,
            drift_backend=drift_backend,
            tracking_dataset=name,
            phase_name="online",
            initial=0,
            metrics_only=metrics_only,
        )
        matrices.append((offline, online))
        emit({"event": "matrix", "dataset": name, "seed": seed})

    # The authors choose the smallest offline boundary over seeds, rounded to the next
    # integer. Solve the same boundary with a bracketed search instead of fsolve on a plateau.
    top = (
        kappa_max
        if kappa_max is not None
        else min(
            math.floor(alpha_max_same_batch(phase.stale.T, 0, 1, len(phase.stale) - 1)) + 1
            for phase, _ in matrices
        )
    )
    kappas = np.linspace(0.0, top, kappa_points).tolist()
    emit({"event": "grid", "dataset": name, "kappa_max": top, "points": kappa_points})
    runs = []
    for seed, (offline, online) in zip(seed_values, matrices, strict=True):
        baselines = online.drift_updates or {}
        runs.append([evaluate_kappa(offline, online, kappa, seed, baselines) for kappa in kappas])
        emit({"event": "analysis", "dataset": name, "seed": seed})

    curves = {
        rule: {
            metric: [_mean(run[i][rule][metric] for run in runs) for i in range(len(kappas))]
            for metric in ("strategy_cost", "strategy_scpe", "average_accuracy", "retrains")
        }
        for rule in (*RULES, *(DRIFT_RULES if drift_backend else ()))
    }
    summary = {
        rule: {
            "strategy_scpe": _mean(metrics["strategy_scpe"][1:]),
            "accuracy": _mean(metrics["average_accuracy"][1:]),
            "retrains": _mean(metrics["retrains"][1:]),
        }
        for rule, metrics in curves.items()
    }
    details = []
    protocol = "CARA-authors-historical-v1"
    for index, (seed, run) in enumerate(zip(seed_values, runs, strict=True)):
        by_policy = {
            rule: [{"kappa": kappa, **row[rule]} for kappa, row in zip(kappas, run, strict=True)]
            for rule in curves
        }
        seed_summary = {
            rule: {
                "strategy_scpe": _mean(row[rule]["strategy_scpe"] for row in run[1:]),
                "accuracy": _mean(row[rule]["average_accuracy"] for row in run[1:]),
                "retrains": _mean(row[rule]["retrains"] for row in run[1:]),
            }
            for rule in curves
        }
        detail = {
            "seed": seed,
            "kappa": kappas,
            "curves_by_policy": by_policy,
            "summary": seed_summary,
            "offline_batches": OFFLINE,
            "online_batches": len(matrices[index][1].stale),
            "matrices": {
                phase_name: {
                    "staleness": _matrix_values(phase.stale),
                    "accuracy": _matrix_values(phase.accuracy),
                }
                for phase_name, phase in zip(("offline", "online"), matrices[index], strict=True)
            },
        }
        detail["policy_tracking"] = policy_report(
            name,
            seed,
            protocol=protocol,
            analysis=detail,
            candidate_reports=[matrices[index][0].tracking, matrices[index][1].tracking],
            settings={
                "kappa_grid": kappas,
                "drift_backend": drift_backend,
                "cost_timing": "same_batch_with_initial_fit_charged",
                "accuracy_timing": "reference_prequential_shift_with_final_batch_clipping",
                "preprocessing": "historical_full_stream_attributes_for_real_datasets",
            },
            artifact_mode="metrics_only" if metrics_only else "full",
        )
        details.append(detail)
    return {
        "dataset": name,
        "seeds": len(seed_values),
        "seed_values": seed_values,
        "reference": REFERENCE,
        "kappa_max": top,
        "summary": summary,
        "kappa": kappas,
        "curves": curves,
        "drift_backend": drift_backend,
        "kappa_range": "offline-oracle" if kappa_max is None else "fixed",
        "kernel_gamma": gamma if gamma is not None else "1/n_features",
        "oracle_retrains": summary["oracle_staleness"]["retrains"],
        "candidate_tracking": [{"offline": a.tracking, "online": b.tracking} for a, b in matrices],
        "trials_detail": details,
        "protocol": protocol,
        "dispersion": {
            "n": len(seed_values),
            "ddof": 0,
            "divisor": len(seed_values),
            "kind": "population_std",
        },
        "evaluation": {
            "cost_timing": "same_batch_with_initial_fit_charged",
            "accuracy_timing": "reference_prequential_shift_with_final_batch_clipping",
            "preprocessing": "historical_full_stream_attributes_for_real_datasets",
            "causal_preprocessing_variant": False,
        },
    }


def schedule_showcase(
    name="covcon-d", seeds=5, *, batches=11, offline=4, fraction=0.25, metrics_only=False, **options
):
    """Illustrate a CARA-T schedule on a short prefix of the reference data stream."""
    seed_values = repetition_seeds(seeds)
    if not 2 <= offline <= batches - 2 or fraction < 0:
        raise ValueError("invalid CARA showcase horizon, seeds or cost fraction")
    stream, queries, gamma = batches_for(name, **options)
    if batches > len(stream):
        raise ValueError("showcase horizon exceeds the available batches")
    accuracy, deployed, changes, kappas, tracking = [], [], [], [], []
    for seed in seed_values:
        phase = build_phase(
            stream[:batches],
            ForestLearner(seed),
            gamma,
            None if queries is None else queries[:batches],
            tracking_dataset=name,
            phase_name="showcase",
            initial=offline,
            metrics_only=metrics_only,
        )
        tracking.append(phase.tracking)
        history, online = phase.stale[:offline, :offline], phase.stale[offline:, offline:]
        kappa = fraction * alpha_max_same_batch(history.T, 0, 1, offline - 1)
        tau = fit_threshold(history, kappa, cumulative=False, seed=seed)
        retrains = threshold_schedule(online, tau)
        models = np.zeros(len(online), dtype=int)
        models[retrains] = retrains
        models = np.maximum.accumulate(models)
        accuracy.append(np.where(np.isnan(phase.accuracy), 0.0, phase.accuracy))
        deployed.append(phase.accuracy[offline + models, np.arange(offline, batches)])
        changes.append({offline + t for t in retrains[1:]})
        kappas.append(kappa)
    stack, scores = np.asarray(accuracy), np.asarray(deployed)
    marked = [t for t in range(offline, batches) if sum(t in run for run in changes) * 2 >= len(seed_values)]
    return {
        "dataset": name,
        "rule": "cara",
        "seeds": len(seed_values),
        "seed_values": seed_values,
        "batches": batches,
        "offline": offline,
        "alpha": float(np.mean(kappas)),
        "accuracy_mean": stack.mean(axis=0).tolist(),
        "accuracy_std": stack.std(axis=0).tolist(),
        "schedule_mean": scores.mean(axis=0).tolist(),
        "schedule_std": scores.std(axis=0).tolist(),
        "retrain_batches": marked,
        "candidate_tracking": tracking,
    }
