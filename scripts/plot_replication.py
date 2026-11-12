"""Draw the replication figures from the result JSON files. Usage:

    python scripts/plot_replication.py docs/results/upf-*.json docs/results/cara-*.json \
        --output docs/results/plots
"""

import argparse
import json
from pathlib import Path

from continual_bench.evaluation.replication_plots import cara_figures, upf_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/results/plots"))
    args = parser.parse_args()
    upf, cara = {}, {}
    for path in args.results:
        for entry in json.loads(path.read_text()):
            (upf if "trials" in entry else cara)[entry["dataset"]] = entry
    args.output.mkdir(parents=True, exist_ok=True)
    figures = []
    if upf:
        figures += upf_figures(list(upf.values()))
    if cara:
        figures += cara_figures(list(cara.values()))
    for name, figure in figures:
        figure.savefig(args.output / name, dpi=130, bbox_inches="tight")
        print(args.output / name)


if __name__ == "__main__":
    main()
