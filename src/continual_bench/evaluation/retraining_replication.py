"""Replication harness for the UPF paper protocol (Regol et al. 2025, sections 5-6, Tables 1 and 4).

Per trial: ``w + T`` datasets (w = 7 offline, T = 8 online); a fresh XGBoost model per dataset with the
library defaults; ``C_hat`` AUC over 10 values of alpha in ``[0, alpha_max]`` with ``alpha_max`` from
UPF Table 4; UPF with delta = 0.95. ADWIN/KSWIN/FHDDM baselines are not reproduced here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.retraining import build_paper_matrix
from continual_bench.retraining.analysis import analyze

OFFLINE, ONLINE, ALPHAS = 7, 8, 10
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

    def __init__(self, seed: int):
        self.params = {"objective": "binary:logistic", "seed": seed, "nthread": 2, "verbosity": 0}

    def fit(self, index, x, y):
        import xgboost

        return xgboost.train(self.params, xgboost.DMatrix(x, label=y), num_boost_round=100)

    def losses(self, state, x, y):
        import xgboost

        return ((state.predict(xgboost.DMatrix(x)) >= 0.5).astype(np.int64) != y).astype(np.float64)


def trial_batches(name: str, trial: int, electricity_path=None, size=None, directory=DEFAULT_CACHE, *,
                  airlines_path=None):
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
        if path not in _ARRAYS:
            _ARRAYS[path] = data.airlines_arrays(path)
        x, y = _ARRAYS[path]
        size = size or 3000
        start = int(rng.integers(0, len(y) // size - length + 1))
        pool = data.windowed_batches(x, y, size, length, start)
    else:
        raise ValueError(f"unknown dataset {name}")
    return [data.split_batch(x, y, rng) for x, y in pool]


def run_trial(name, trial, *, electricity_path=None, airlines_path=None, size=None, samples=100, rules=None):
    batches = trial_batches(name, trial, electricity_path, size, airlines_path=airlines_path)
    matrix, _ = build_paper_matrix(batches, XGBoostLearner(trial), OFFLINE, seed=trial)
    return analyze(matrix, n_alphas=ALPHAS, alpha_top=ALPHA_MAX[name], samples=samples, seed=trial,
                   **({"rules": tuple(rules)} if rules else {}))


def replicate(name, trials=10, **options):
    runs = [run_trial(name, trial, **options) for trial in range(trials)]
    metrics = ("cost", "retrains", "average_accuracy")
    summary = {}
    for rule in runs[0]["rules"]:
        auc = [run["rules"][rule]["auc"] for run in runs]
        summary[rule] = {"auc_mean": float(np.mean(auc)), "auc_std": float(np.std(auc)),
                         "paper_auc": PAPER_AUC[name].get(rule)}
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
    return {"dataset": name, "trials": trials, "alpha_max": ALPHA_MAX[name], "summary": summary,
            "alphas": runs[0]["alphas"], "curves": curves, "table": table,
            "oracle_alpha_max_per_trial": [run["setup"]["oracle_alpha_max"] for run in runs]}
