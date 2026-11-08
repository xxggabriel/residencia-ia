"""Real-data admission check: every catalog dataset, 100 unique examples, 1 epoch."""

import json
import sys
import time
import traceback
from pathlib import Path

from continual_bench.config import EpochConfig
from continual_bench.datasets.catalog import DATASETS
from continual_bench.evaluation.environment import environment
from continual_bench.evaluation.epochs import run_epochs

state, cache, real, download = sys.argv[1], sys.argv[2], sys.argv[3], "--download" in sys.argv
only = [a for a in sys.argv[4:] if not a.startswith("--")] or DATASETS
rows = []
for dataset in only:
    started = time.monotonic()
    try:
        report = run_epochs(EpochConfig(dataset=dataset, examples=100, epochs=1), state, cache,
                            download=download, real_directory=real)
        epoch = report["epochs"][0]
        row = {"dataset": dataset, "status": "PASS", "task": report["contract"]["source"]["task"],
               "train": epoch["train_presentations"], "test": epoch["test_examples"],
               "split": report["contract"]["split"]["kind"], "metrics": epoch["metrics"],
               "run_key": report["run_key"]}
    except Exception as error:
        row = {"dataset": dataset, "status": "FAIL", "error_type": type(error).__name__,
               "error": str(error)[:300], "where": traceback.extract_tb(error.__traceback__)[-1].name}
    row["seconds"] = round(time.monotonic() - started, 2)
    rows.append(row)
    print(json.dumps(row), flush=True)
summary = {"code_sha256": environment()["code_sha256"], "examples": 100, "epochs": 1,
           "passed": sum(r["status"] == "PASS" for r in rows), "total": len(rows), "datasets": rows}
Path(state, "epoch-check-summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "datasets"}))
