"""Replicate CARA Tables 2-4 (SCPE, query accuracy, #retrains). Usage:

    python scripts/replicate_cara.py covcon-d circle-d electricity covertype [--seeds 5]
"""

import argparse
import json

from continual_bench.evaluation.cara_replication import DATASETS, PAPER, replicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--kappa-points", type=int, default=16)
    parser.add_argument("--electricity")
    parser.add_argument("--output")
    args = parser.parse_args()
    results = [replicate(name, args.seeds, kappa_points=args.kappa_points, electricity_path=args.electricity)
               for name in args.datasets]
    for result in results:
        name = result["dataset"]
        print(f"\n{name} ({result['seeds']} seeds, kappa_max={result['kappa_max']})")
        for index, rule in enumerate(("cara", "cara_cumulative", "cara_periodic")):
            row = result["summary"][rule]
            paper = PAPER["scpe"][name][index]
            print(f"  {rule:16s} SCPE {row['strategy_scpe']:7.2f} (paper {paper})  "
                  f"acc {row['accuracy']:.3f}  "
                  f"retrains {row['retrains']:.1f}")
        for rule in ("markov", "frozen", "oracle_staleness"):
            row = result["summary"][rule]
            print(f"  {rule:16s} SCPE {row['strategy_scpe']:7.2f}  acc {row['accuracy']:.3f}  "
                  f"retrains {row['retrains']:.1f}")
        print(f"  paper accuracy: oracle {PAPER['accuracy_oracle'][name]}, "
              f"CARA-T {PAPER['accuracy_cara_t'][name]}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
