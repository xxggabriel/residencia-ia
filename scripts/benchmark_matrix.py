"""Explicit larger experiments; separate processes keep RSS measurements isolated."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from continual_bench.application.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="state/full-benchmark")
    parser.add_argument("--samples", type=int, default=30_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--services", action="store_true")
    parser.add_argument("--zenml", action="store_true")
    args = parser.parse_args()
    root = Path(args.state).resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in args.seeds:
        for dataset in ("sea", "agrawal", "stagger"):
            for mode in ("streaming", "incremental", "offline"):
                for policy in ["none"] if mode == "offline" else ["none", "periodic", "drift"]:
                    config = Config.model_validate(
                        {
                            "name": "matrix-v1",
                            "mode": mode,
                            "dataset": {
                                "name": dataset,
                                "samples": args.samples,
                                "seed": seed,
                                "label_delay": 3,
                            },
                            "policy": {"name": policy},
                        }
                    )
                    path = root / f"{config.identity()}.json"
                    path.write_text(config.model_dump_json(indent=2))
                    command = [
                        sys.executable,
                        "-m",
                        "continual_bench.cli",
                        "run",
                        str(path),
                        "--state",
                        str(root),
                    ]
                    command += [
                        flag
                        for flag, enabled in (("--services", args.services), ("--zenml", args.zenml))
                        if enabled
                    ]
                    subprocess.run(command, check=True)
                    summary.append(json.loads((root / config.identity() / "report.json").read_text()))
                    (root / "matrix-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
