"""Compute and draw the CARA Fig. 1 plot (accuracy of each model per batch + retraining schedule).

python scripts/plot_schedule.py covcon-d electricity --seeds 5 --output docs/results
"""

import argparse
import json
from pathlib import Path

from continual_bench.evaluation.cara_replication import DATASETS, schedule_showcase
from continual_bench.evaluation.replication_plots import schedule_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--batches", type=int, default=11)
    parser.add_argument("--offline", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--electricity")
    parser.add_argument("--output", type=Path, default=Path("docs/results"))
    args = parser.parse_args()
    options = {"electricity_path": args.electricity} if args.electricity else {}
    for name in args.datasets:
        showcase = schedule_showcase(
            name, args.seeds, batches=args.batches, offline=args.offline, fraction=args.fraction, **options
        )
        (args.output / f"cara-schedule-{name}.json").write_text(json.dumps([showcase], indent=2))
        plots = args.output / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        file, figure = schedule_figure(showcase)
        figure.savefig(plots / file, dpi=130, bbox_inches="tight")
        print(plots / file, "retrains at", showcase["retrain_batches"])


if __name__ == "__main__":
    main()
