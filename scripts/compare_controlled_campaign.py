"""Validate and summarize controlled-v3 reports against the immutable v2 campaign."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def arm(report):
    config = report["config"]
    if config["name"].endswith("bridge"):
        return "sparse_bridge"
    policy = config["policy"]["name"]
    if policy == "drift":
        policy = config["detector"]["name"]
    if policy in ("periodic", "adwin", "ddm"):
        return f"{policy}@{config['training_budget']}"
    return policy


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def analyze(records, historical, manifest):
    expected = {row["run_key"] for row in manifest["configurations"]}
    expected_config = {row["run_key"]: row["config"] for row in manifest["configurations"]}
    assert len(records) == len(expected) == 261
    reports = [row["report"] for row in records]
    assert {r["run_key"] for r in reports} == expected
    assert all(r["complete"] for r in reports)
    assert len({r["environment"]["code_sha256"] for r in reports}) == 1
    assert reports[0]["environment"]["code_sha256"] == manifest["source_sha256"]
    for record in records:
        assert record["publication_verified"] and record["report_sha256_verified"]
    index = {}
    groups = defaultdict(list)
    for r in reports:
        c = r["config"]
        assert c == expected_config[r["run_key"]]
        key = (r["protocol"], c["dataset"]["name"], c["dataset"]["seed"], arm(r))
        assert key not in index
        index[key] = r
        used = sum(update["train_samples"] for update in r["updates"])
        if r["protocol"] != "temporal_holdout":
            assert r["training"]["used"] == used
        if c["training_budget"] is not None:
            assert used <= c["training_budget"]
        if c["policy"]["name"] == "online":
            assert len(r["updates"]) == used == c["dataset"]["samples"] - c["warmup"] == 2900
            assert [u["seq"] for u in r["updates"]] == list(range(100, 3000))
        groups[(r["protocol"], c["dataset"]["name"], arm(r))].append(r)
    summary = []
    for (protocol, dataset, strategy), group in sorted(groups.items()):
        summary.append({
            "protocol": protocol, "dataset": dataset, "strategy": strategy,
            "seeds": sorted(r["config"]["dataset"]["seed"] for r in group),
            "metrics": {m: describe([r["metrics"][m] for r in group])
                        for m in ("accuracy", "f1", "brier", "log_loss")},
            "updates": describe([len(r["updates"]) for r in group]),
            "alarms": describe([len(r["alarms"]) for r in group]),
            "training_used": None if protocol == "temporal_holdout" else describe(
                [sum(u["train_samples"] for u in r["updates"]) for r in group]
            ),
        })
    bridge = []
    for row in historical:
        old = row["report"]
        c = old["config"]
        policy = c["policy"]["name"]
        strategy = ("sparse_bridge" if policy == "online" else
                    "periodic@5800" if policy == "periodic" else
                    c["detector"]["name"] + "@5800" if policy == "drift" else "none")
        key = (old["protocol"], c["dataset"]["name"], 42, strategy)
        new = index[key]
        assert new["dataset"]["sha256"] == old["dataset"]["sha256"]
        assert new["metrics"] == old["metrics"]
        if "prediction_digest" in old:
            assert new["prediction_digest"] == old["prediction_digest"]
        bridge.append({"old_run_key": old["run_key"], "new_run_key": new["run_key"],
                       "metrics_identical": True, "dataset_identical": True,
                       "prediction_digest_identical": True if "prediction_digest" in old else None})
    assert len(bridge) == 33
    paired = []
    for protocol in ("event_prequential", "batch_prequential"):
        for dataset in ("sea", "agrawal", "stagger"):
            online = [index[(protocol, dataset, seed, "online")] for seed in range(42, 47)]
            for comparator in ("periodic@2900", "periodic@5800", "adwin@2900", "ddm@2900"):
                controls = [index[(protocol, dataset, seed, comparator)] for seed in range(42, 47)]
                differences = [a["metrics"]["accuracy"] - b["metrics"]["accuracy"]
                               for a, b in zip(online, controls)]
                paired.append({"protocol": protocol, "dataset": dataset,
                               "comparison": "online - " + comparator,
                               "accuracy_difference": describe(differences),
                               "online_wins": sum(d > 0 for d in differences)})
    detectors = []
    equal_budget_pairs = 0
    for protocol in ("event_prequential", "batch_prequential"):
        for detector in ("adwin", "ddm"):
            selected = []
            for dataset in ("sea", "agrawal", "stagger"):
                for seed in range(42, 47):
                    low = index[(protocol, dataset, seed, detector + "@2900")]
                    high = index[(protocol, dataset, seed, detector + "@5800")]
                    assert low["prediction_digest"] == high["prediction_digest"]
                    equal_budget_pairs += 1
                    selected.append(low)
            detectors.append({
                "protocol": protocol, "detector": detector, "runs": len(selected),
                "alarms": sum(len(r["alarms"]) for r in selected),
                "updates": sum(len(r["updates"]) for r in selected),
                "alarms_before_first_change": sum(a["seq"] < 1000 for r in selected for a in r["alarms"]),
                "cooldown_blocked_alarms": sum(a["decision"]["reason"] == "cooldown"
                                               for r in selected for a in r["alarms"]),
                "training_used": describe([r["training"]["used"] for r in selected]),
            })
    return {"complete": True, "runs": len(reports), "seeds": list(range(42, 47)),
            "historical_reproductions": bridge, "summary": summary, "paired_differences": paired,
            "detectors": detectors, "identical_drift_budget_pairs": equal_budget_pairs,
            "training_cost_scope": "Post-warmup learn calls, including replays; offline cost not compared",
            "uncertainty": "Sample SD across five seeds; no significance claim or pooled-protocol ranking"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports/generated/controlled-v3-inputs.json")
    parser.add_argument("--historical", default="reports/benchmark-analysis-inputs-20261027.json")
    parser.add_argument("--manifest", default="configs/campaign-controlled-v3.json")
    parser.add_argument("--label-balance", default="reports/controlled-v3-label-balance.json")
    parser.add_argument("--output", default="reports/controlled-v3-comparison.json")
    args = parser.parse_args()
    result = analyze(json.loads(Path(args.reports).read_text()),
                     json.loads(Path(args.historical).read_text()),
                     json.loads(Path(args.manifest).read_text()))
    balances = json.loads(Path(args.label_balance).read_text())
    assert len(balances) == 15
    constant = []
    for dataset in ("sea", "agrawal", "stagger"):
        selected = [b for b in balances if b["dataset"] == dataset]
        assert len(selected) == 5
        counts = [(sum(s["positive"] for s in b["segments"]), sum(s["n"] for s in b["segments"]))
                  for b in selected]
        assert all(n == 2900 for _, n in counts)
        constant.append({"dataset": dataset,
                         "accuracy": describe([positive / n for positive, n in counts]),
                         "f1": describe([2 * positive / (n + positive) for positive, n in counts])})
    result["retrospective_always_positive_reference"] = {
        "note": "Post-hoc label-based diagnostic, not a preregistered campaign arm", "datasets": constant}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"runs": result["runs"],
                      "historical_reproductions": len(result["historical_reproductions"])}))


if __name__ == "__main__":
    main()
