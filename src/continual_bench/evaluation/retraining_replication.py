"""Replication harness for the UPF paper protocol (Regol et al. 2025, sections 5-6, Tables 1 and 4).

Per trial: ``w + T`` datasets (w = 7 offline, T = 8 online); a fresh XGBoost model per dataset with the
library defaults; ``C_hat`` AUC over 10 values of alpha in ``[0, alpha_max]`` with ``alpha_max`` from
UPF Table 4; UPF with delta = 0.95. ADWIN/KSWIN/FHDDM baselines are not reproduced here.
"""

from __future__ import annotations

import time
from dataclasses import replace
from hashlib import file_digest
from pathlib import Path

import numpy as np

from continual_bench.core.metrics import Metrics
from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.retraining import build_paper_matrix
from continual_bench.retraining.analysis import DEFAULT_RULES, analyze, trapezoid

from .retraining_observations import candidate_report
from .retraining_policy_tracking import policy_report, repetition_seeds

OFFLINE, ONLINE, ALPHAS = 7, 8, 10
COMPARISON_DATASETS = ("electricity", "airlines")
COMPARISON_BATCH_SIZE = 2000
# Extra resolution at low costs, where the per-query staleness objective changes schedules.
COMPARISON_GRID = (
    0.0,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)
COMPARISON_METRICS = ("strategy_cost", "error_oracle_gap", "strategy_oracle_gap")
ALPHA_MAX = {"gauss": 0.5, "circles": 0.25, "electricity": 1.0, "airlines": 0.7}  # UPF Table 4
# UPF Table 1 (AUC of C_hat, lower is better): published values, for comparison only.
PAPER_AUC = {
    "gauss": {
        "cara": 0.3841,
        "cara_cumulative": 0.3862,
        "cara_periodic": 0.4678,
        "upf": 0.3829,
        "oracle": 0.3724,
    },
    "circles": {
        "cara": 0.0726,
        "cara_cumulative": 0.0731,
        "cara_periodic": 0.0800,
        "upf": 0.0668,
        "oracle": 0.0627,
    },
    "electricity": {
        "cara": 2.7198,
        "cara_cumulative": 2.7147,
        "cara_periodic": 2.8986,
        "upf": 2.5782,
        "oracle": 2.4217,
    },
    "airlines": {
        "cara": 2.2753,
        "cara_cumulative": 2.2900,
        "cara_periodic": 2.4061,
        "upf": 2.2865,
        "oracle": 2.2298,
    },
}
DEFAULT_CACHE = "~/.cache/continual-bench/datasets"
_ARRAYS: dict = {}


class TrialBatches(list):
    """Split arrays with the source selection and row coverage that produced them."""

    def __init__(self, batches, *, source, coverage):
        super().__init__(batches)
        self.source = source
        self.coverage = coverage


def source_identity(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return {"kind": "file", "file": path.name, "sha256": digest, "bytes": path.stat().st_size}


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
            self.params,
            xgboost.DMatrix(x, label=y),
            num_boost_round=100,
            callbacks=[ObserveRounds()],
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


def trial_batches(
    name: str,
    trial: int,
    electricity_path=None,
    size=None,
    directory=DEFAULT_CACHE,
    *,
    airlines_path=None,
    airlines_disk_cache=True,
    full_source=False,
):
    length = OFFLINE + ONLINE
    rng = np.random.default_rng(10_000 + trial)
    if full_source and size is not None and name in COMPARISON_DATASETS:
        raise ValueError("full-source partitioning cannot use a fixed batch size")
    offset = 0
    if name == "gauss":
        pool = data.upf_gauss(length, size or 5000, seed=trial)
        source_rows = length * (size or 5000)
        source = {"kind": "generator", "generator": "upf_gauss", "seed": trial}
    elif name == "circles":
        pool = data.upf_circles(length, size or 5000, seed=trial)
        source_rows = length * (size or 5000)
        source = {"kind": "generator", "generator": "upf_circles", "seed": trial}
    elif name == "electricity":
        source = source_identity(electricity_path)
        x, y = data.electricity_arrays(electricity_path)
        source_rows = len(y)
        if full_source:
            pool = data.full_source_batches(x, y, length)
        else:
            size = size or 2000
            start = int(rng.integers(0, len(y) // size - length + 1))
            pool = data.windowed_batches(x, y, size, length, start)
            offset = start * size
    elif name == "airlines":
        path = Path(airlines_path or Path(directory).expanduser() / "airlines.arff").expanduser().resolve()
        source = source_identity(path)
        cache_key = (path, airlines_disk_cache, source["sha256"])
        if cache_key not in _ARRAYS:
            _ARRAYS[cache_key] = data.airlines_arrays(path, use_disk_cache=airlines_disk_cache)
        x, y = _ARRAYS[cache_key]
        source_rows = len(y)
        if full_source:
            pool = data.full_source_batches(x, y, length)
        else:
            size = size or 3000
            start = int(rng.integers(0, len(y) // size - length + 1))
            pool = data.windowed_batches(x, y, size, length, start)
            offset = start * size
    else:
        raise ValueError(f"unknown dataset {name}")
    splits = [data.split_batch(x, y, rng) for x, y in pool]
    ranges = []
    for index, (_, y, _, yt) in enumerate(splits):
        rows = len(y) + len(yt)
        ranges.append(
            {
                "batch": index,
                "start": offset,
                "stop": offset + rows,
                "train_rows": len(y),
                "holdout_rows": len(yt),
            }
        )
        offset += rows
    selected_rows = sum(len(y) for _, y in pool)
    coverage = {
        "full_source": full_source,
        "selection": "all_rows_contiguous_15_batches"
        if full_source
        else "all_generated_batches"
        if source["kind"] == "generator"
        else "random_contiguous_window",
        "source_rows": source_rows,
        "raw_rows": source_rows,
        "total_rows": source_rows,
        "selected_rows": selected_rows,
        "included_rows": selected_rows,
        "excluded_rows": source_rows - selected_rows,
        "batch_rows": [len(y) for _, y in pool],
        "source_row_ranges": ranges,
        "included_row_ranges": [[row["start"], row["stop"]] for row in ranges],
        "source_range_coordinate": "zero_based_half_open_before_within_batch_split",
        "train_fraction": 0.8,
        "split_rng_seed": 10_000 + trial,
        "offline_rows": sum(len(y) for _, y in pool[:OFFLINE]),
        "online_rows": sum(len(y) for _, y in pool[OFFLINE:]),
        "tail_handling": "preserved" if selected_rows == source_rows else "outside_selected_window",
    }
    if full_source and (
        selected_rows != source_rows
        or ranges[0]["start"] != 0
        or ranges[-1]["stop"] != source_rows
        or any(left["stop"] != right["start"] for left, right in zip(ranges, ranges[1:]))
    ):
        raise ValueError("full-source partition failed contiguous complete row coverage")
    coverage["all_source_rows_included"] = selected_rows == source_rows
    return TrialBatches(splits, source=source, coverage=coverage)


def add_policy_quality(analysis, batches, states, learner, matrix):
    """Score completed schedules on their pooled held-outs, without feeding decisions or storing scores."""
    began = time.perf_counter()
    probabilities = {}
    quality_by_schedule = {}

    def quality(row):
        deployed = matrix.initial
        changes = set(row["retrain_points"])
        served = []
        for batch in range(matrix.offline, matrix.batches):
            if matrix.paper_timing and batch in changes:
                deployed = batch
            served.append(deployed)
            if not matrix.paper_timing and batch in changes:
                deployed = batch
        row["served_by"] = served
        schedule = tuple(served)
        if schedule not in quality_by_schedule:
            accumulator = Metrics()
            for batch, model in zip(range(matrix.offline, matrix.batches), served, strict=True):
                x, y = batches[batch][2:]
                key = model, batch
                if key not in probabilities:
                    _, scores = learner.observation_predict(states[model], x, 1)
                    if scores is None:
                        raise ValueError("policy quality requires positive-class probabilities")
                    scores = np.asarray(scores, dtype=np.float64)
                    if scores.shape != (len(y),):
                        raise ValueError("policy probabilities must match the held-out rows")
                    probabilities[key] = scores
                for target, probability in zip(y, probabilities[key], strict=True):
                    accumulator.add(bool(target), float(probability))
            quality_by_schedule[schedule] = accumulator.values()
        row["quality"] = dict(quality_by_schedule[schedule])
        row["quality_scope"] = "pooled_online_heldout_retrospective_only"

    for rule in analysis["rules"].values():
        for row in rule["curve"]:
            quality(row)
    for table in analysis["table"].values():
        for row in table.values():
            quality(row)
    analysis["evaluation_quality"] = {
        "scope": "pooled_online_heldout",
        "retrospective_only": True,
        "used_for_decisions": False,
        "decision_threshold": 0.5,
        "positive_class": 1,
        "quality_accuracy": "weighted_by_heldout_row_count",
        "average_accuracy": "unweighted_mean_of_online_batch_accuracy",
        "probability_cache_pairs": len(probabilities),
        "unique_schedules": len(quality_by_schedule),
        "elapsed_seconds": time.perf_counter() - began,
    }


def run_trial(
    name,
    trial,
    *,
    electricity_path=None,
    airlines_path=None,
    size=None,
    samples=100,
    rules=None,
    compare_oracles=False,
    decision_timing="same-batch",
    metrics_only=False,
    full_source=False,
):
    started = time.perf_counter()
    if compare_oracles:
        if name not in COMPARISON_DATASETS:
            raise ValueError("the shared CARA/UPF benchmark supports Electricity and Airlines")
        if not full_source:
            size = size or COMPARISON_BATCH_SIZE
    if decision_timing not in ("same-batch", "next-batch"):
        raise ValueError("unknown decision timing")
    if decision_timing == "next-batch" and not compare_oracles:
        raise ValueError("next-batch timing requires the shared benchmark")
    # A stale NPZ sidecar must not replace the explicitly supplied source in a corrective run.
    batches = trial_batches(
        name,
        trial,
        electricity_path,
        size,
        airlines_path=airlines_path,
        airlines_disk_cache=False,
        full_source=full_source,
    )
    learner = XGBoostLearner(trial)
    # Held-out labels score candidates but never define the CARA decision signal.
    # This also applies to the individual UPF harness, not only the paired benchmark.
    matrix, states = build_paper_matrix(batches, learner, OFFLINE, seed=trial, separate_query_split=True)
    version = "v3-full-source" if full_source else "v2-window"
    protocol = (
        f"UPF-train-reference-{version}-{'paired' if compare_oracles else 'individual'}-{decision_timing}"
    )
    selected_rows = sum(len(y) + len(yt) for _, y, _, yt in batches)
    coverage = getattr(batches, "coverage", {"full_source": full_source, "selected_rows": selected_rows})
    source = getattr(batches, "source", {"kind": "provided_batches"})
    artifact_mode = "metrics_only" if metrics_only else "full"
    tracking = candidate_report(
        name,
        trial,
        batches,
        states,
        learner,
        matrix.fit_seconds.tolist(),
        protocol=f"{protocol}-candidates",
        initial=matrix.initial,
        duration_before_tracking=time.perf_counter() - started,
        evaluation={
            "separate_query_split": True,
            "decision_timing": decision_timing,
            "source": source,
            "coverage": coverage,
        },
        artifact_mode=artifact_mode,
    )
    selected_rules = tuple(rules) if rules else DEFAULT_RULES
    if compare_oracles:
        selected_rules = (*selected_rules, "oracle_staleness")
        if decision_timing == "next-batch":
            errors = matrix.err.copy()
            np.fill_diagonal(errors, np.nan)
            matrix = replace(matrix, err=errors, paper_timing=False)
    result = analyze(
        matrix,
        n_alphas=ALPHAS,
        alpha_top=ALPHA_MAX[name],
        samples=samples,
        seed=trial,
        rules=selected_rules,
        alpha_grid=COMPARISON_GRID if compare_oracles else None,
    )
    if full_source:
        add_policy_quality(result, batches, states, learner, matrix)
    result["candidate_tracking"] = tracking
    result["protocol"] = protocol
    result["source"] = {**source, "selected_data_sha256": tracking["contract"].get("data_sha256")}
    result["coverage"] = coverage
    result["policy_tracking"] = policy_report(
        name,
        trial,
        protocol=protocol,
        analysis=result,
        candidate_reports=[tracking],
        settings={
            "separate_query_split": True,
            "decision_timing": decision_timing,
            "paired": compare_oracles,
            "alpha_grid": result["alphas"],
            "samples": samples,
            "rules": list(result["rules"]),
            "full_source": full_source,
            "source": source,
            "coverage": coverage,
        },
        artifact_mode=artifact_mode,
    )
    return result


def replicate(name, trials=None, *, seeds=None, compare_oracles=False, **options):
    if seeds is None and trials is None and options.get("full_source"):
        seeds = [42, 43, 44]
    seed_values = repetition_seeds((10 if trials is None else trials) if seeds is None else seeds)
    if seeds is not None and trials is not None and trials != len(seed_values):
        raise ValueError("trial count differs from the explicit seeds")
    trials = len(seed_values)
    runs = [run_trial(name, seed, compare_oracles=compare_oracles, **options) for seed in seed_values]
    metrics = ("cost", "retrains", "average_accuracy")
    if compare_oracles:
        metrics = (*metrics, *COMPARISON_METRICS)
    summary = {}
    for rule in runs[0]["rules"]:
        auc = [run["rules"][rule]["auc"] for run in runs]
        summary[rule] = {
            "auc_mean": float(np.mean(auc)),
            "auc_std": float(np.std(auc)),
            "paper_auc": None if compare_oracles or options.get("full_source") else PAPER_AUC[name].get(rule),
        }
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
        curves[rule] = {
            key: [float(np.mean([r[i][key] for r in rows])) for i in range(len(rows[0]))] for key in metrics
        }
    table = {}
    for level, by_rule in runs[0]["table"].items():
        table[level] = {}
        for rule, row in by_rule.items():
            stats = {}
            for key in metrics:
                values = [run["table"][level][rule][key] for run in runs]
                stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
            table[level][rule] = {"alpha": row["alpha"], **stats}
    result = {
        "dataset": name,
        "trials": trials,
        "seeds": seed_values,
        "alpha_max": runs[0]["setup"]["alpha_max"],
        "summary": summary,
        "alphas": runs[0]["alphas"],
        "curves": curves,
        "table": table,
        "oracle_alpha_max_per_trial": [run["setup"]["oracle_alpha_max"] for run in runs],
        "candidate_tracking": [run["candidate_tracking"] for run in runs],
        "protocol": runs[0]["protocol"],
        "source": {
            key: value
            for key, value in runs[0]["source"].items()
            if key not in ("seed", "selected_data_sha256")
        },
        "coverage_per_seed": [
            {"seed": seed, **run["coverage"]} for seed, run in zip(seed_values, runs, strict=True)
        ],
        "sources_per_seed": [
            {"seed": seed, **run["source"]} for seed, run in zip(seed_values, runs, strict=True)
        ],
        "published_reference_auc": PAPER_AUC[name],
        "paper_reference": {
            "auc": PAPER_AUC[name],
            "comparable_to_paper": False,
            "reason": "full_source_partition_changes_the_evaluation_protocol"
            if options.get("full_source")
            else "local_train_reference_variant_and_declared_data_assumptions",
        },
        "dispersion": {"n": trials, "ddof": 0, "divisor": trials, "kind": "population_std"},
        "evaluation": {
            "separate_query_split": True,
            "full_source": options.get("full_source", False),
            "policy_reference": "labeled_training_partition",
            "policy_queries": "unlabeled_heldout_attributes",
            "heldout_labels": "retrospective_scoring_only",
            "paper_values": "published_reference_values_not_a_claim_of_identical_reproduction",
            "comparable_to_paper": False,
        },
        "trials_detail": [
            {
                "seed": seed,
                "setup": run["setup"],
                "source": run["source"],
                "coverage": run["coverage"],
                "table": run["table"],
                "curves": {rule: info["curve"] for rule, info in run["rules"].items()},
                "summary": {
                    rule: {key: value for key, value in info.items() if key != "curve"}
                    for rule, info in run["rules"].items()
                },
                "performance_matrix": run["performance_matrix"],
                "staleness_matrix": run["staleness_matrix"],
                "forecast": run["forecast"],
                "wrong_alpha": run["wrong_alpha"],
                "cumulative_strategy_cost": run["cumulative_strategy_cost"],
                "oracle_schedule_at_focus": run["oracle_schedule_at_focus"],
                "evaluation_quality": run.get("evaluation_quality"),
                "candidate_run_key": run["candidate_tracking"]["run_key"],
                "policy_tracking": run["policy_tracking"],
            }
            for seed, run in zip(seed_values, runs, strict=True)
        ],
    }
    if compare_oracles:
        result["comparison"] = {
            "same_batch": options.get("decision_timing", "same-batch") == "same-batch",
            "batch_size": None
            if options.get("full_source")
            else options.get("size") or COMPARISON_BATCH_SIZE,
            "batch_rows": runs[0]["coverage"].get("batch_rows"),
            "full_source": options.get("full_source", False),
            "train_fraction": 0.8,
            "offline_batches": OFFLINE,
            "online_batches": ONLINE,
            "seeds": seed_values,
            "source": runs[0]["source"],
            "decision_points": runs[0]["setup"]["decision_points"],
            "upf_delta": 0.95,
            "upf_samples": options.get("samples", 100),
            "staleness": "RBF-weighted losses from labeled train rows; queries from unlabeled test features",
            "cost_input": "Common numeric grid, in the units of each objective",
        }
    return result
