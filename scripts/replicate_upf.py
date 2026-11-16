"""Benchmark UPF and export the statistics for Tables 1 and 2 of Regol et al. (2025)."""

import argparse
import json
from hashlib import file_digest
from pathlib import Path

from continual_bench.evaluation.retraining_replication import ALPHA_MAX, replicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=sorted(ALPHA_MAX))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--electricity")
    parser.add_argument("--airlines", help="numeric Scikit-Multiflow CSV or OpenML ARFF")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--compare-oracles", action="store_true",
                        help="score CARA/UPF and both Oracles on the same Electricity windows")
    parser.add_argument("--decision-timing", choices=("same-batch", "next-batch"),
                        help="shared benchmark deployment timing")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.compare_oracles and args.datasets != ["electricity"]:
        parser.error("--compare-oracles requires electricity")
    if args.decision_timing and not args.compare_oracles:
        parser.error("--decision-timing requires --compare-oracles")
    if "electricity" in args.datasets and not args.electricity:
        parser.error("electricity requires --electricity PATH")
    if args.trials < 1 or args.samples < 2:
        parser.error("use at least one trial and two Monte Carlo samples")
    if args.electricity:
        args.electricity = Path(args.electricity).expanduser()
    results = []
    for name in args.datasets:
        result = replicate(name, args.trials, electricity_path=args.electricity,
                           airlines_path=args.airlines, samples=args.samples,
                           compare_oracles=args.compare_oracles,
                           decision_timing=args.decision_timing or "same-batch")
        if args.compare_oracles:
            import xgboost

            path = Path(args.electricity).expanduser()
            with path.open("rb") as source:
                digest = file_digest(source, "sha256").hexdigest()
            result["comparison"]["source"] = {"file": path.name, "sha256": digest}
            result["comparison"]["learner"] = {
                "name": "XGBoost", "version": xgboost.__version__, "rounds": 100,
                "eta": 0.3, "max_depth": 6, "threshold": 0.5,
            }
        results.append(result)
        print(f"\n{result['dataset']} ({result['trials']} trials, alpha_max={result['alpha_max']})")
        for rule, row in result["summary"].items():
            paper = "n/a" if row["paper_auc"] is None else f"{row['paper_auc']:.4f}"
            if args.compare_oracles:
                print(f"  {rule:16s} AUC error {row['auc_mean']:.4f}   "
                      f"AUC staleness {row['strategy_auc_mean']:.4f}")
                continue
            print(f"  {rule:16s} ours {row['auc_mean']:.4f} +- {row['auc_std']:.4f}   paper {paper}")
        if name == "electricity":
            for by_rule in result["table"].values():
                print(f"  α = {by_rule['upf']['alpha']:g}")
                selected = ("cara", "cara_cumulative", "cara_periodic", "upf", "oracle", "oracle_staleness")
                for rule in (selected if args.compare_oracles else ("upf", "oracle")):
                    row = by_rule[rule]
                    metrics = "   ".join(f"{key} {row[key]['mean']:.4f} ± {row[key]['std']:.4f}"
                                         for key in ("retrains", "average_accuracy", "cost"))
                    if args.compare_oracles:
                        metrics += f"   staleness cost {row['strategy_cost']['mean']:.4f}"
                    print(f"    {rule:6s} {metrics}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
