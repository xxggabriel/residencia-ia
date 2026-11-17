"""Replication harness for the UPF paper protocol (Regol et al. 2025, sections 5-6, Tables 1 and 4).

Per trial: ``w + T`` datasets (w = 7 offline, T = 8 online); a fresh XGBoost model per dataset with the
library defaults; ``C_hat`` AUC over 10 values of alpha in ``[0, alpha_max]`` with ``alpha_max`` from
UPF Table 4; UPF with delta = 0.95. ADWIN/KSWIN/FHDDM baselines are not reproduced here.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.retraining import build_paper_matrix
from continual_bench.retraining.analysis import DEFAULT_RULES, analyze, trapezoid

from .retraining_observations import candidate_report

OFFLINE, ONLINE, ALPHAS = 7, 8, 10
COMPARISON_DATASETS = ("electricity", "airlines")
COMPARISON_BATCH_SIZE = 2000
# Extra resolution at low costs, where the per-query staleness objective changes schedules.
COMPARISON_GRID = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
                   0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
COMPARISON_METRICS = ("strategy_cost", "error_oracle_gap", "strategy_oracle_gap")
ALPHA_MAX = {"gauss": 0.5, "circles": 0.25, "electricity": 1.0, "airlines": 0.7}  # UPF Table 4
# UPF Table 1 (AUC of C_hat, lower is better): published values, for comparison only.
PAPER_AUC = {
    "gauss": {"cara": 0.3841, "cara_cumulative": 0.3862, "cara_periodic": 0.4678, "upf": 0.3829,
              "oracle": 0.3724},
    "circles": {"cara": 0.0726, "cara_cumulative": 0.0731, "cara_periodic": 0.0800, "upf": 0.0668,
                "oracle": 0.0627},
    "electricity": {"cara": 2.7198, "cara_cumulative": 2.7147, "cara_periodic": 2.8986, "upf": 2.5782,
                    "oracle": 2.4217},
    "airlines": {"cara": 2.2753, "cara_cumulative": 2.2900, "cara_periodic": 2.4061, "upf": 2.2865,
                 "oracle": 2.2298},
}
DEFAULT_CACHE = "~/.cache/continual-bench/datasets"
_ARRAYS: dict = {}


class XGBoostLearner:
    """XGBoost with the library defaults (eta 0.3, depth 6, 100 rounds, threshold 0.5); 0-1 loss.

    Use the native API to preserve the learner settings of the original benchmark.
    """

    observation_model = "xgboost"
    observation_threshold = 0.5
    observation_step_kind = "xgboost_boosting_round"

    def __init__(self, seed: int):
        self.params = {"objective": "binary:logistic", "seed": seed, "nthread": 2, "verbosity": 0}
        self._round_durations = {}

    def fit(self, index, x, y):
        import xgboost

        durations = self._round_durations[index] = []

        class ObserveRounds(xgboost.callback.TrainingCallback):
            def before_iteration(self, model, epoch, evals_log):
                self.started = time.perf_counter()
                return False

            def after_iteration(self, model, epoch, evals_log):
                durations.append(time.perf_counter() - self.started)
                return False

        return xgboost.train(
            self.params, xgboost.DMatrix(x, label=y), num_boost_round=100, callbacks=[ObserveRounds()],
        )

    def losses(self, state, x, y):
        prediction, _ = self.observation_predict(state, x, 1)
        return (prediction != y).astype(np.float64)

    def observation_predict(self, state, x, positive):
        import xgboost

        scores = state.predict(xgboost.DMatrix(x))
        return (scores >= self.observation_threshold).astype(np.int64), scores if positive == 1 else None

    def observation_durations(self, index, _elapsed):
        return self._round_durations[index]

    @staticmethod
    def observation_checkpoint(state):
        return bytes(state.save_raw(raw_format="ubj")), "ubj"


def trial_batches(name: str, trial: int, electricity_path=None, size=None, directory=DEFAULT_CACHE, *,
                  airlines_path=None, airlines_disk_cache=True):
    length = OFFLINE + ONLINE
    rng = np.random.default_rng(10_000 + trial)
    if name == "gauss":
        pool = data.upf_gauss(length, size or 5000, seed=trial)
    elif name == "circles":
        pool = data.upf_circles(length, size or 5000, seed=trial)
    elif name == "electricity":
        every = data.electricity_batches(electricity_path, size or 2000)
        start = int(rng.integers(0, len(every) - length + 1))
        pool = every[start:start + length]
    elif name == "airlines":
        path = Path(airlines_path or Path(directory).expanduser() / "airlines.arff").expanduser().resolve()
        cache_key = (path, airlines_disk_cache)
        if cache_key not in _ARRAYS:
            _ARRAYS[cache_key] = data.airlines_arrays(path, use_disk_cache=airlines_disk_cache)
        x, y = _ARRAYS[cache_key]
        size = size or 3000
        start = int(rng.integers(0, len(y) // size - length + 1))
        pool = data.windowed_batches(x, y, size, length, start)
    else:
        raise ValueError(f"unknown dataset {name}")
    return [data.split_batch(x, y, rng) for x, y in pool]


def run_trial(name, trial, *, electricity_path=None, airlines_path=None, size=None, samples=100, rules=None,
              compare_oracles=False, decision_timing="same-batch"):
    started = time.perf_counter()
    if compare_oracles:
        if name not in COMPARISON_DATASETS:
            raise ValueError("the shared CARA/UPF benchmark supports Electricity and Airlines")
        size = size or COMPARISON_BATCH_SIZE
    if decision_timing not in ("same-batch", "next-batch"):
        raise ValueError("unknown decision timing")
    if decision_timing == "next-batch" and not compare_oracles:
        raise ValueError("next-batch timing requires the shared benchmark")
    # In paired runs, the reported source hash must identify the parsed rows.
    batches = trial_batches(name, trial, electricity_path, size, airlines_path=airlines_path,
                            airlines_disk_cache=not compare_oracles)
    learner = XGBoostLearner(trial)
    matrix, states = build_paper_matrix(batches, learner, OFFLINE, seed=trial,
                                        separate_query_split=compare_oracles)
    tracking = candidate_report(
        name, trial, batches, states, learner, matrix.fit_seconds.tolist(), protocol="UPF-paper-candidates",
        initial=matrix.initial, duration_before_tracking=time.perf_counter() - started,
        evaluation={"separate_query_split": compare_oracles, "decision_timing": decision_timing},
    )
    selected_rules = tuple(rules) if rules else DEFAULT_RULES
    if compare_oracles:
        selected_rules = (*selected_rules, "oracle_staleness")
        if decision_timing == "next-batch":
            errors = matrix.err.copy()
            np.fill_diagonal(errors, np.nan)
            matrix = replace(matrix, err=errors, paper_timing=False)
    result = analyze(matrix, n_alphas=ALPHAS, alpha_top=ALPHA_MAX[name], samples=samples, seed=trial,
                     rules=selected_rules, alpha_grid=COMPARISON_GRID if compare_oracles else None)
    result["candidate_tracking"] = tracking
    return result


def replicate(name, trials=10, *, compare_oracles=False, **options):
    runs = [run_trial(name, trial, compare_oracles=compare_oracles, **options) for trial in range(trials)]
    metrics = ("cost", "retrains", "average_accuracy")
    if compare_oracles:
        metrics = (*metrics, *COMPARISON_METRICS)
    summary = {}
    for rule in runs[0]["rules"]:
        auc = [run["rules"][rule]["auc"] for run in runs]
        summary[rule] = {"auc_mean": float(np.mean(auc)), "auc_std": float(np.std(auc)),
                         "paper_auc": None if compare_oracles else PAPER_AUC[name].get(rule)}
        if compare_oracles:
            strategy_auc = [
                trapezoid(run["alphas"], [r["strategy_cost"] for r in run["rules"][rule]["curve"]])
                for run in runs
            ]
            summary[rule]["strategy_auc_mean"] = float(np.mean(strategy_auc))
            summary[rule]["strategy_auc_std"] = float(np.std(strategy_auc))
    curves = {}
    for rule in runs[0]["rules"]:
        rows = [run["rules"][rule]["curve"] for run in runs]
        curves[rule] = {key: [float(np.mean([r[i][key] for r in rows])) for i in range(len(rows[0]))]
                        for key in metrics}
    table = {}
    for level, by_rule in runs[0]["table"].items():
        table[level] = {}
        for rule, row in by_rule.items():
            stats = {}
            for key in metrics:
                values = [run["table"][level][rule][key] for run in runs]
                stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
            table[level][rule] = {"alpha": row["alpha"], **stats}
    result = {"dataset": name, "trials": trials, "alpha_max": runs[0]["setup"]["alpha_max"],
            "summary": summary, "alphas": runs[0]["alphas"], "curves": curves, "table": table,
            "oracle_alpha_max_per_trial": [run["setup"]["oracle_alpha_max"] for run in runs],
            "candidate_tracking": [run["candidate_tracking"] for run in runs]}
    if compare_oracles:
        result["comparison"] = {
            "same_batch": options.get("decision_timing", "same-batch") == "same-batch",
            "batch_size": options.get("size") or COMPARISON_BATCH_SIZE, "train_fraction": 0.8,
            "offline_batches": OFFLINE, "online_batches": ONLINE, "seeds": list(range(trials)),
            "decision_points": runs[0]["setup"]["decision_points"],
            "upf_delta": 0.95, "upf_samples": options.get("samples", 100),
            "staleness": "RBF-weighted losses from labeled train rows; queries from unlabeled test features",
            "cost_input": "Common numeric grid, in the units of each objective",
        }
        result["trials_detail"] = [{"seed": seed, "table": run["table"],
                                    "curves": {rule: info["curve"] for rule, info in run["rules"].items()}}
                                   for seed, run in enumerate(runs)]
    return result
