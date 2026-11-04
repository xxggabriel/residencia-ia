"""Homelab admission: exactly 100 unique records and two train-only epochs per source."""

import argparse
import json
from pathlib import Path

from continual_bench.adapters.dataset_loading import DATASETS
from continual_bench.adapters.local import atomic_write
from continual_bench.application.epoch_benchmark import EpochConfig, dataset_fingerprint, run_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--real-directory", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = Path(args.state)
    results = []
    receipt = {"version": "semana09-all-datasets-smoke-v1", "complete": False, "datasets": results}
    atomic_write(root / "dataset-smoke-complete.json", json.dumps(receipt).encode())
    for dataset in DATASETS:
        try:
            report = run_epochs(EpochConfig(dataset=dataset), root, args.cache,
                                download=args.download, real_directory=args.real_directory)
            row = {"dataset": dataset, "complete": True, "status": "passed",
                   "run_key": report["run_key"], "unique_examples": report["unique_examples"],
                   "epochs": len(report["epochs"]), "metrics": [e["metrics"] for e in report["epochs"]],
                   "code_sha256": report["contract"]["environment"]["code_sha256"],
                   "runtime_fingerprint": report["contract"]["runtime_fingerprint"]}
        except Exception as error:
            row = {"dataset": dataset, "complete": False, "status": "blocked_or_failed",
                   "error_type": type(error).__name__, "error": str(error)}
        results.append(row)
        atomic_write(root / "dataset-smoke-progress.json", json.dumps(receipt, indent=2).encode())
        print(json.dumps(row), flush=True)
    receipt["complete"] = len(results) == len(DATASETS) and all(row["complete"] for row in results)
    receipt["runtime_fingerprint"] = dataset_fingerprint()
    atomic_write(root / "dataset-smoke-complete.json", json.dumps(receipt, indent=2).encode())
    if not receipt["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
