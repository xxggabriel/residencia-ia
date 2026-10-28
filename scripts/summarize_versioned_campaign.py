"""Summarize the verified v4 evidence and check the unchanged v3 synthetic arms."""

import argparse
import json
from pathlib import Path


def config_key(config):
    return json.dumps({k: v for k, v in config.items() if k != "name"}, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="reports/generated/versioned-v4-inputs.json")
    parser.add_argument("--previous", default="reports/generated/controlled-v3-inputs.json")
    parser.add_argument("--output", default="reports/versioned-v4-validation.json")
    args = parser.parse_args()
    current = json.loads(Path(args.current).read_text())
    assert current["complete_campaign"] and current["runs_verified"] == 289
    previous = {
        config_key(r["report"]["config"]): r["report"]
        for r in json.loads(Path(args.previous).read_text())
        if r["report"]["config"]["name"] == "controlled-v3"
    }
    same = []
    prediction_matches = 0
    offline_matches = 0
    real = []
    for record in current["records"]:
        report = record["report"]
        cfg = report["config"]
        if report["dataset"]["name"] in {"electricity", "airlines"}:
            real.append(
                {
                    "run_key": report["run_key"],
                    "dataset": report["dataset"]["name"],
                    "mode": cfg["mode"],
                    "policy": cfg["policy"]["name"],
                    "detector": cfg["detector"]["name"],
                    "budget": cfg["training_budget"],
                    "training": report.get("training"),
                    "metrics": report["metrics"],
                    "alarms": len(report["alarms"]),
                    "updates": len(report["updates"]),
                    "model_versions": report["model_versions"]["count"],
                    "model_uri": record["receipt"]["model_uri"],
                }
            )
        else:
            old = previous[config_key(cfg)]
            assert old["metrics"] == report["metrics"]
            if cfg["mode"] == "offline":
                assert old["split"] == report["split"]
                assert old["windows"] == report["windows"]
                offline_matches += 1
            else:
                assert old["prediction_digest"] == report["prediction_digest"]
                prediction_matches += 1
            assert old["dataset"]["sha256"] == report["dataset"]["sha256"]
            same.append(report["run_key"])
    assert len(same) == 255 and len(real) == 34
    summary = {k: v for k, v in current.items() if k != "records"}
    summary.update(
        {
            "synthetic_v3_exact_matches": len(same),
            "prequential_prediction_digest_matches": prediction_matches,
            "offline_split_and_window_matches": offline_matches,
            "comparison_fields": [
                "metrics",
                "dataset.sha256",
                "prequential prediction_digest",
                "offline split/windows",
            ],
            "real_results": real,
            "limitations": [
                "3000-row fixed prefix per real dataset; no independent seed repetitions",
                "Source order preserved; original chronology and label delay not established",
                "Electricity upstream normalization/imputation fit scope undocumented",
                "No real ground-truth change points; alarms do not measure detection recall/delay",
                "Training budget counts sample presentations, not equal CPU time or detector consumption",
                "Model Registry contains final versions; every update is versioned in immutable journals",
            ],
        }
    )
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"real_results", "limitations"}}, indent=2)
    )


if __name__ == "__main__":
    main()
