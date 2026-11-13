"""Replication harness for the CARA paper (Mahadevan & Mathioudakis 2024, sections 7-9, Tables 1-4).

100 batches (1000 points for the synthetic sets), 25 offline, 75 online; Random Forest classifier from
scikit-learn with library defaults; queries = 10% of every batch ("-D") or a static Gaussian ("-S");
5 random seeds; kappa swept over a grid and averaged. Baselines NR, Markov and CARA-T/-CT/-P plus the
Oracle are reproduced; ADWIN and DDM (River detectors on the error stream) are not.

``alpha = kappa / n_queries``: the paper sums the staleness over queries, this code averages.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.retraining import alpha_max, build_matrix, simulate
from continual_bench.retraining.analysis import analyze, make_rule

BATCHES, OFFLINE = 100, 25
RULES = ("cara", "cara_cumulative", "cara_periodic", "markov", "frozen", "oracle_staleness")
# kappa upper limits read from Figs. 4-7 (CovCon-D, Airlines, Electricity, Covertype). The paper does not
# state the other ranges, so they use the smallest kappa at which this code's Oracle stops retraining.
KAPPA_MAX = {"covcon-d": 150, "airlines": 620, "electricity": 68, "covertype": 14}
DATASETS = ("covcon-d", "covcon-s", "circle-d", "circle-s", "gauss-d", "gauss-s", "electricity", "airlines",
            "covertype")
# Tables 2-4 of the paper: CARA-T, CARA-CT, CARA-P, Oracle (retrains/accuracy), NR, Markov.
PAPER = {
    "scpe": {"covcon-d": (17.72, 19.26, 16.3), "covcon-s": (15.58, 22.94, 104.79),
             "circle-d": (45.61, 47.29, 84.96), "circle-s": (33.89, 45.15, 79.34),
             "gauss-d": (8.63, 59.93, 135.95), "gauss-s": (66.62, 96.38, 202.09),
             "electricity": (8.32, 40.32, 64.07), "airlines": (6.11, 5.48, 5.45),
             "covertype": (26.39, 27.92, 38.37)},
    "accuracy_oracle": {"covcon-d": 0.76, "covcon-s": 0.8, "circle-d": 0.97, "circle-s": 0.96,
                        "gauss-d": 0.92, "gauss-s": 1.0, "electricity": 0.71, "airlines": 0.62,
                        "covertype": 0.76},
    "accuracy_cara_t": {"covcon-d": 0.7, "covcon-s": 0.79, "circle-d": 0.96, "circle-s": 0.93,
                        "gauss-d": 0.9, "gauss-s": 0.99, "electricity": 0.68, "airlines": 0.6,
                        "covertype": 0.66},
    "retrains_oracle": {"covcon-d": 11.2, "covcon-s": 8.21, "circle-d": 3.89, "circle-s": 3.09,
                        "gauss-d": 2.23},
    "retrains_cara_t": {"covcon-d": 13.89, "covcon-s": 10.05, "circle-d": 5.67, "circle-s": 3.56,
                        "gauss-d": 2.62},
}


class ForestLearner:
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
        rows = np.asarray(x, dtype=np.float32)
        total = np.zeros((len(rows), len(forest.classes_)))
        for tree in forest.estimators_:
            counts = tree.tree_.predict(rows)
            counts = counts[:, 0, :] if counts.ndim == 3 else counts
            total += counts / counts.sum(axis=1, keepdims=True)
        return forest.classes_[total.argmax(axis=1)]


def batches_for(name: str, seed: int, directory="~/.cache/continual-bench/datasets", electricity_path=None):
    """``(data batches, query stream or None, n_points, n_queries)`` of one dataset and seed."""
    base, _, variant = name.partition("-")
    cache = Path(directory).expanduser()
    if base in ("gauss", "circle", "covcon"):
        batches = getattr(data, f"cara_{base}")(BATCHES, 1000, seed)
        queries = data.cara_static_queries(base, BATCHES, 100, seed) if variant == "s" else None
        return batches, queries, 1000, 100
    if base == "electricity":
        every = data.electricity_batches(electricity_path, 45312 // BATCHES)
        batches = every[:BATCHES]
    elif base == "airlines":
        batches = data.airlines_batches(cache / "airlines.arff", BATCHES)
    elif base == "covertype":
        batches = data.covertype_batches(cache / "covtype.libsvm.binary.bz2", BATCHES)
    else:
        raise ValueError(f"unknown dataset {name}")
    rows = len(batches[0][1])
    return batches, None, min(rows, 512), min(max(rows // 10, 1), 600)


def run_seed(name, seed, *, kappa_points=16, samples=1, **options):
    batches, queries, n_points, n_queries = batches_for(name, seed, **options)
    matrix, _ = build_matrix(iter(batches), ForestLearner(seed), offline=OFFLINE, seed=seed,
                             n_points=n_points, n_queries=n_queries, query_stream=queries,
                             workers=os.cpu_count() or 1)
    n_q = len(queries[0][1]) if queries else n_queries
    probe = analyze(matrix, rules=("oracle_staleness",), n_alphas=2, samples=samples, seed=seed)
    top = KAPPA_MAX[name] / n_q if name in KAPPA_MAX else probe["setup"]["alpha_max"] * 1.0
    # for datasets without a published range use the staleness Oracle's own alpha_max (stops retraining)
    return analyze(matrix, rules=RULES, n_alphas=kappa_points, alpha_top=top, samples=samples, seed=seed), n_q


def mean_over_kappa(analysis, rule, key):
    values = [c[key] for c in analysis["rules"][rule]["curve"] if c[key] is not None]
    return float(np.mean(values)) if values else None


def replicate(name, seeds=5, **options):
    runs = [run_seed(name, seed, **options) for seed in range(seeds)]
    summary = {}
    for rule in RULES:
        summary[rule] = {
            "strategy_scpe": float(np.mean([mean_over_kappa(a, rule, "strategy_scpe") or np.nan
                                            for a, _ in runs])),
            "accuracy": float(np.mean([mean_over_kappa(a, rule, "average_accuracy") for a, _ in runs])),
            "retrains": float(np.mean([mean_over_kappa(a, rule, "retrains") for a, _ in runs])),
        }
    curves = {}
    for rule in RULES:
        curves[rule] = {
            key: [float(np.nanmean([np.nan if a["rules"][rule]["curve"][i][key] is None
                                    else a["rules"][rule]["curve"][i][key] for a, _ in runs]))
                  for i in range(len(runs[0][0]["alphas"]))]
            for key in ("strategy_cost", "retrains", "average_accuracy", "strategy_scpe")}
    kappa = [alpha * runs[0][1] for alpha in runs[0][0]["alphas"]]
    return {"dataset": name, "seeds": seeds, "kappa_max": KAPPA_MAX.get(name), "summary": summary,
            "kappa": kappa, "curves": curves}


def schedule_showcase(name="covcon-d", seeds=5, *, batches=11, offline=4, fraction=0.25, rule="cara",
                      **options):
    """Data of the CARA Fig. 1 plot: accuracy of every model on every later batch, and the schedule.

    ``batches`` consecutive batches (the paper draws 11, ``D0..D10``); the retraining cost is
    ``fraction`` of the largest alpha at which the error Oracle still retrains. Accuracy is ``1 - err``
    averaged over seeds, with the spread across seeds; the schedule line is the accuracy of the model
    deployed at each online batch, and a batch is marked when at least half the seeds retrain for it.
    """
    base, _, _ = name.partition("-")
    accuracy, deployed, changes, alphas = [], [], [], []
    for seed in range(seeds):
        if base in ("gauss", "circle", "covcon"):
            stream = getattr(data, f"cara_{base}")(batches, 1000, seed)
            queries, n_points, n_queries = None, 1000, 100
        else:
            stream, queries, n_points, n_queries = batches_for(name, seed, **options)
            stream = stream[:batches]
        matrix, _ = build_matrix(iter(stream), ForestLearner(seed), offline=offline, seed=seed,
                                 n_points=n_points, n_queries=n_queries, query_stream=queries,
                                 workers=os.cpu_count() or 1)
        first, last = matrix.offline, matrix.batches - 1
        alpha = fraction * alpha_max(matrix.err, matrix.initial, first, last)
        result = simulate(make_rule(rule, seed=seed), matrix, alpha)
        alphas.append(alpha)
        accuracy.append(1.0 - matrix.err)
        models = list(result.deployed)
        deployed.append([1.0 - matrix.err[m, j] for j, m in zip(range(first, matrix.batches), models,
                                                                strict=True)])
        changes.append({j for (j, m), previous in zip(zip(range(first, matrix.batches), models),
                                                      [matrix.initial, *models]) if m != previous})
    stack = np.asarray(accuracy)
    online = np.asarray(deployed)
    marked = sorted(j for j in range(batches) if sum(j in c for c in changes) * 2 >= seeds and j > 0)
    return {"dataset": name, "rule": rule, "seeds": seeds, "batches": batches, "offline": offline,
            "alpha": float(np.mean(alphas)),
            "accuracy_mean": np.nan_to_num(np.mean(stack, axis=0), nan=0.0).tolist(),
            "accuracy_std": np.nan_to_num(np.std(stack, axis=0), nan=0.0).tolist(),
            "schedule_mean": online.mean(axis=0).tolist(), "schedule_std": online.std(axis=0).tolist(),
            "retrain_batches": marked}
