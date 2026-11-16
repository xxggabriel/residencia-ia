"""Replicate CARA Tables 2-4 (SCPE, query accuracy, #retrains). Usage:

python scripts/replicate_cara.py covcon-d circle-d electricity covertype [--seeds 5] \\
    --output docs/results/cara.json --log docs/results/cara-progress.jsonl

Progress lines go to stdout (flushed) and, with --log, as one JSON object per event. The result JSON is
rewritten after every finished dataset, so partial results can be plotted while the run continues.
"""

import argparse
import json
import time

from continual_bench.evaluation.cara_replication import DATASETS, GRID, PAPER, replicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--kappa-points", type=int, default=GRID)
    parser.add_argument("--kappa-max", type=float, help="override the offline Oracle's grid endpoint")
    parser.add_argument("--electricity", help="original elec.csv")
    parser.add_argument("--airlines", help="original airlines.csv")
    parser.add_argument("--covertype", help="original scaled LibSVM Covertype file")
    parser.add_argument("--workers", type=int, default=1, help="parallel matrix columns")
    parser.add_argument("--drift-backend", choices=("skmultiflow", "river"),
                        help="optionally include ADWIN and DDM; River is an alternative backend")
    parser.add_argument("--output")
    parser.add_argument("--log", help="JSON-lines file with every progress event")
    args = parser.parse_args()
    started = time.time()
    total_steps = len(args.datasets) * (2 * args.seeds + 1)
    done = [0]
    log = open(args.log, "a") if args.log else None

    def progress(event):
        done[0] += 1
        elapsed = time.time() - started
        eta = elapsed / done[0] * (total_steps - done[0])
        event = {
            **event,
            "step": f"{done[0]}/{total_steps}",
            "elapsed_min": round(elapsed / 60, 1),
            "eta_min": round(eta / 60, 1),
        }
        print(
            f"[{event['step']} {event['elapsed_min']:5.1f}min eta {event['eta_min']:5.1f}min] "
            + json.dumps(event),
            flush=True,
        )
        if log:
            log.write(json.dumps(event) + "\n")
            log.flush()

    results = []
    for name in args.datasets:
        result = replicate(
            name,
            args.seeds,
            kappa_points=args.kappa_points,
            electricity_path=args.electricity,
            airlines_path=args.airlines,
            covertype_path=args.covertype,
            kappa_max=args.kappa_max,
            workers=args.workers,
            drift_backend=args.drift_backend,
            progress=progress,
        )
        results.append(result)
        print(f"\n{name} ({result['seeds']} seeds, kappa_max={result['kappa_max']:.2f})")
        for rule, row in result["summary"].items():
            scpe = "n/a" if row["strategy_scpe"] is None else f"{row['strategy_scpe']:.2f}%"
            print(f"  {rule:16s} SCPE {scpe:>9}  acc {row['accuracy']:.3f}  retrains {row['retrains']:.1f}")
        print(
            f"  oracle retrains {result['oracle_retrains']:.2f} (paper {PAPER['retrains_oracle'][name]}); "
            f"paper accuracy: oracle {PAPER['accuracy_oracle'][name]}, "
            f"CARA-T {PAPER['accuracy_cara_t'][name]}\n",
            flush=True,
        )
        if args.output:
            with open(args.output, "w") as handle:
                json.dump(results, handle, indent=2, allow_nan=False)

    if log:
        log.close()


if __name__ == "__main__":
    main()
