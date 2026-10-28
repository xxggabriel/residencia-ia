"""Read frozen events in the worker; emit post-hoc label-only references, without training runs."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/app/configs/campaign-versioned-v4.json")
    parser.add_argument("--state", default="/state/runs")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    seen = set()
    rows = []
    for config in manifest["configurations"]:
        cfg = config["config"]
        key = (cfg["dataset"]["name"], cfg["dataset"]["seed"])
        if key in seen:
            continue
        seen.add(key)
        directory = Path(args.state) / config["run_key"]
        raw = (directory / "events.jsonl").read_bytes()
        report = json.loads((directory / "report.json").read_text())
        assert hashlib.sha256(raw).hexdigest() == report["dataset"]["sha256"]
        events = [json.loads(line) for line in raw.splitlines()]
        labels = [event["y"] for event in events]
        split = int(len(labels) * cfg["offline_train_fraction"])
        trained = sum(e["seq"] < split and e["available_at"] < events[split]["timestamp"] for e in events)
        for mode, train_end, eval_start in (
            ("prequential", cfg["warmup"], cfg["warmup"]),
            ("offline", trained, split),
        ):
            train, test = labels[:train_end], labels[eval_start:]
            q, prior = sum(test) / len(test), sum(train) / len(train)
            rows.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "mode": mode,
                    "dataset_sha256": report["dataset"]["sha256"],
                    "train_positive_rate": prior,
                    "evaluation_positive": sum(test),
                    "evaluation_n": len(test),
                    "evaluation_positive_rate": q,
                    "always_positive": {"accuracy": q, "f1": 2 * q / (1 + q)},
                    "always_negative": {"accuracy": 1 - q, "f1": 0},
                }
            )
    assert len(seen) == 17
    print(
        json.dumps(
            {
                "scope": "Post-hoc constant-class references, not registered campaign arms; "
                "class rates are descriptive, not held-out model selection evidence",
                "references": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
