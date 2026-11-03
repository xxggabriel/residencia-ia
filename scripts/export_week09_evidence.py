"""Export benchmark-owned evidence from the same runtime that wrote the files."""

import argparse
import hashlib
import json
from pathlib import Path


def collect(root):
    result = {
        "scope": "functional smoke only; no ranking or comparative campaign",
        "binary": [], "typed": [], "services": [],
    }
    for path in sorted((root / "smoke").rglob("report.json")):
        report = json.loads(path.read_text())
        result["binary"].append({
            "run_key": report["run_key"], "dataset": report["config"]["dataset"]["name"],
            "samples": report["observed"], "policy": report["config"]["policy"]["name"],
            "detector": report["config"]["detector"]["name"], "metrics": report["metrics"],
            "code_sha256": report["environment"]["code_sha256"],
            "dataset_sha256": report["dataset"]["sha256"], "environment": report["environment"],
            "near_future": {k: v for k, v in report["week09"]["near_future"].items() if k != "windows"},
            "promotion": report["promotion"], "training": report["training"],
            "compute_seconds": report["compute_seconds"],
            "peak_rss_bytes_process": report["peak_rss_bytes_process"],
        })
    for path in sorted((root / "typed").rglob("report.json")):
        result["typed"].append(json.loads(path.read_text()))
    for path in sorted((root / "service-tests").rglob("service-evidence.json")):
        evidence = json.loads(path.read_text())
        report = json.loads((path.parent / "report.json").read_text())
        evidence["code_sha256"] = report["environment"]["code_sha256"]
        evidence["dataset_sha256"] = report["dataset"]["sha256"]
        evidence["supplement_sha256"] = hashlib.sha256(
            (path.parent / "week09-metrics.json").read_bytes()
        ).hexdigest()
        result["services"].append(evidence)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="/state")
    print(json.dumps(collect(Path(parser.parse_args().state)), indent=2))


if __name__ == "__main__":
    main()
