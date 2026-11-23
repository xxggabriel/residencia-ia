"""Benchmark UPF and export the statistics for Tables 1 and 2 of Regol et al. (2025)."""

import argparse
import json
from pathlib import Path

from continual_bench.evaluation.retraining_replication import (
    ALPHA_MAX,
    COMPARISON_DATASETS,
    DEFAULT_CACHE,
    repetition_seeds,
    replicate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=sorted(ALPHA_MAX))
    repetitions = parser.add_mutually_exclusive_group()
    repetitions.add_argument("--trials", type=int, help="legacy repetition count; default 10 seeds from zero")
    repetitions.add_argument("--seeds", type=int, nargs="+", help="explicit distinct repetition seeds")
    parser.add_argument(
        "--full-source",
        action="store_true",
        help="use every real source row in 15 contiguous batches, preserving the remainder",
    )
    parser.add_argument("--electricity")
    parser.add_argument(
        "--airlines",
        type=Path,
        default=Path(DEFAULT_CACHE).expanduser() / "airlines.arff",
        help="numeric Scikit-Multiflow CSV or OpenML ARFF",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--metrics-only", action="store_true", help="retain metrics and provenance, no models"
    )
    parser.add_argument(
        "--compare-oracles",
        action="store_true",
        help="score CARA/UPF and both Oracles on shared Electricity or Airlines windows",
    )
    parser.add_argument(
        "--decision-timing", choices=("same-batch", "next-batch"), help="shared benchmark deployment timing"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.compare_oracles and any(name not in COMPARISON_DATASETS for name in args.datasets):
        parser.error("--compare-oracles supports electricity and airlines")
    if args.decision_timing and not args.compare_oracles:
        parser.error("--decision-timing requires --compare-oracles")
    if args.full_source and args.trials is not None:
        parser.error("--trials belongs to legacy windows; use --seeds with --full-source (default 42 43 44)")
    if "electricity" in args.datasets and not args.electricity:
        parser.error("electricity requires --electricity PATH")
    if (args.trials is not None and args.trials < 1) or args.samples < 2:
        parser.error("use at least one trial and two Monte Carlo samples")
    try:
        repetition_seeds(args.seeds if args.seeds is not None else args.trials or 10)
    except ValueError as error:
        parser.error(str(error))
    if args.electricity:
        args.electricity = Path(args.electricity).expanduser()
    args.airlines = args.airlines.expanduser()
    paths = {"electricity": args.electricity, "airlines": args.airlines}
    for name in args.datasets:
        if name in paths and (paths[name] is None or not paths[name].is_file()):
            parser.error(f"{name} input file does not exist: {paths[name]}")
    results = []
    for name in args.datasets:
        result = replicate(
            name,
            args.trials,
            seeds=args.seeds,
            electricity_path=args.electricity,
            airlines_path=args.airlines,
            samples=args.samples,
            compare_oracles=args.compare_oracles,
            decision_timing=args.decision_timing or "same-batch",
            metrics_only=args.metrics_only,
            full_source=args.full_source,
        )
        import xgboost

        result["learner"] = {
            "name": "XGBoost",
            "version": xgboost.__version__,
            "rounds": 100,
            "eta": 0.3,
            "max_depth": 6,
            "threshold": 0.5,
            "nthread": 2,
        }
        if args.compare_oracles:
            result["comparison"]["learner"] = result["learner"]
        results.append(result)
        print(f"\n{result['dataset']} ({result['trials']} trials, alpha_max={result['alpha_max']})")
        if args.full_source:
            print("Full-source evaluation; published paper values belong to a different protocol.")
        for rule, row in result["summary"].items():
            paper = "n/a" if row["paper_auc"] is None else f"{row['paper_auc']:.4f}"
            if args.compare_oracles:
                print(
                    f"  {rule:16s} AUC error {row['auc_mean']:.4f}   "
                    f"AUC staleness {row['strategy_auc_mean']:.4f}"
                )
                continue
            print(f"  {rule:16s} ours {row['auc_mean']:.4f} +- {row['auc_std']:.4f}   paper {paper}")
        if args.compare_oracles or name == "electricity":
            for by_rule in result["table"].values():
                print(f"  α = {by_rule['upf']['alpha']:g}")
                selected = ("cara", "cara_cumulative", "cara_periodic", "upf", "oracle", "oracle_staleness")
                for rule in selected if args.compare_oracles else ("upf", "oracle"):
                    row = by_rule[rule]
                    metrics = "   ".join(
                        f"{key} {row[key]['mean']:.4f} ± {row[key]['std']:.4f}"
                        for key in ("retrains", "average_accuracy", "cost")
                    )
                    if args.compare_oracles:
                        metrics += f"   staleness cost {row['strategy_cost']['mean']:.4f}"
                    print(f"    {rule:6s} {metrics}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
