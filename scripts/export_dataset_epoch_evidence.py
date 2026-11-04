"""Export small, inspectable epoch evidence using the runtime that owns the state."""

import argparse
import json
from pathlib import Path


def collect(root):
    root = Path(root)
    receipt = json.loads((root / "dataset-smokes/dataset-smoke-complete.json").read_text())
    reports = []
    for row in receipt["datasets"]:
        if not row.get("complete"):
            continue
        path = root / "dataset-smokes" / row["run_key"] / "epoch-report.json"
        report = json.loads(path.read_text())
        source = report["contract"]["source"]
        reports.append({
            "dataset": row["dataset"], "run_key": report["run_key"],
            "unique_examples": report["unique_examples"], "epochs": report["epochs"],
            "training_examples": len(report["train_ids"]), "test_examples": len(report["test_ids"]),
            "split": report["contract"]["split"], "seconds": report["seconds"],
            "peak_rss_bytes_process": report["peak_rss_bytes_process"],
            "environment": report["contract"]["environment"],
            "runtime_fingerprint": report["contract"]["runtime_fingerprint"],
            "samples_sha256": report["contract"]["samples_sha256"],
            "source": {key: source[key] for key in
                       ("name", "version", "source", "source_version", "license", "schema",
                        "adapter", "source_variant", "timestamp_kind", "natural_chronology",
                        "total_range_bytes", "whole_parquet_lfs_checksums_verified") if key in source},
        })
    services = []
    code = reports[0]["environment"]["code_sha256"] if reports else None
    for directory in ("binary-service-final", "epoch-service-final", "reviewed-services"):
        for path in sorted((root / directory).rglob("*service-evidence.json")):
            evidence = json.loads(path.read_text())
            if "code_sha256" not in evidence:
                report = json.loads((path.parent / "report.json").read_text())
                evidence["code_sha256"] = report["environment"]["code_sha256"]
            if evidence["code_sha256"] == code:
                services.append(evidence)
    return {"scope": "initial functional validation, not comparative campaign results",
            "admission": receipt, "reports": reports, "services": services}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    print(json.dumps(collect(parser.parse_args().state), indent=2, allow_nan=False))
