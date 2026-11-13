"""Replicate UPF Table 1 (AUC of C_hat) on datasets that need no download. Usage:

    python scripts/replicate_upf.py gauss circles [electricity --electricity PATH.arff] [--trials 10]
"""

import argparse
import json

from continual_bench.evaluation.retraining_replication import ALPHA_MAX, replicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=sorted(ALPHA_MAX))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--electricity")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    results = [replicate(name, args.trials, electricity_path=args.electricity, samples=args.samples)
               for name in args.datasets]
    for result in results:
        print(f"\n{result['dataset']} ({result['trials']} trials, alpha_max={result['alpha_max']})")
        for rule, row in result["summary"].items():
            paper = "n/a" if row["paper_auc"] is None else f"{row['paper_auc']:.4f}"
            print(f"  {rule:16s} ours {row['auc_mean']:.4f} +- {row['auc_std']:.4f}   paper {paper}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
