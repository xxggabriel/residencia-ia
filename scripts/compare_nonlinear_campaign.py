"""Compare verified nonlinear-v5 runs with paired logistic controls and frozen v4 evidence."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

MODELS = ("logistic", "hoeffding_tree", "knn")
METRICS = ("accuracy", "f1", "brier", "log_loss")
REAL_DATASETS = {"electricity", "airlines"}


def case_key(config):
    """Keep data, seed, protocol, policy and budget; only learner and campaign may differ."""
    return json.dumps(
        {key: value for key, value in config.items() if key not in {"name", "model"}},
        sort_keys=True,
        separators=(",", ":"),
    )


def arm(report):
    config = report["config"]
    name = config["policy"]["name"]
    if name == "drift":
        name = config["detector"]["name"]
    if name in {"periodic", "adwin", "ddm"}:
        return f"{name}@{config['training_budget']}"
    return name


def describe(values):
    assert values and all(math.isfinite(value) for value in values), "nonfinite or empty observations"
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def _reports(payload):
    assert payload["complete_campaign"], "input campaign is incomplete"
    records = payload["records"]
    assert payload["runs_verified"] == len(records), "verified run count differs from records"
    reports = []
    for record in records:
        report = record["report"]
        assert record["publication_verified"] and report["complete"], "unverified/incomplete run"
        for metric in METRICS:
            value = report["metrics"][metric]
            assert math.isfinite(value) and value >= 0, f"invalid {metric}"
            assert metric == "log_loss" or value <= 1, f"invalid {metric}"
        reports.append(report)
    assert len({r["run_key"] for r in reports}) == len(reports), "duplicate run_key"
    return reports


def _observations(report):
    used = sum(update["train_samples"] for update in report["updates"])
    if report.get("training") is not None:
        assert used == report["training"]["used"], "training consumption differs from update ledger"
    budget = report["config"].get("training_budget")
    assert budget is None or used <= budget, "training budget exceeded"
    return {
        "training_used": used,
        "alarms": len(report["alarms"]),
        "updates": len(report["updates"]),
        "compute_seconds": report["compute_seconds"],
        "wall_seconds": report["wall_seconds_this_attempt"],
        "peak_rss_bytes": report["peak_rss_bytes_process"],
    }


def _new_bridge_counter():
    return {
        "cases": 0,
        "metrics_exact_matches": 0,
        "prequential_cases": 0,
        "prediction_digest_exact_matches": 0,
        "missing_prediction_digests": 0,
        "offline_cases": 0,
        "offline_split_exact_matches": 0,
        "offline_window_exact_matches": 0,
        "offline_split_and_window_exact_matches": 0,
    }


def _bridge(current, previous):
    counters = {key: _new_bridge_counter() for key in ("all", "synthetic", "real")}
    differences = []
    for key, old in previous.items():
        new = current[key]
        config = old["config"]
        selected = (
            counters["all"],
            counters["real" if config["dataset"]["name"] in REAL_DATASETS else "synthetic"],
        )
        metrics_equal = old["metrics"] == new["metrics"]
        detail = {
            "dataset": config["dataset"]["name"],
            "mode": config["mode"],
            "seed": config["dataset"]["seed"],
            "strategy": arm(old),
            "previous_run_key": old["run_key"],
            "current_run_key": new["run_key"],
            "metrics_equal": metrics_equal,
        }
        for counter in selected:
            counter["cases"] += 1
            counter["metrics_exact_matches"] += metrics_equal
        changed = not metrics_equal
        if not metrics_equal:
            detail.update(
                {
                    "previous_metrics": old["metrics"],
                    "current_metrics": new["metrics"],
                    "metric_deltas": {m: new["metrics"][m] - old["metrics"][m] for m in METRICS},
                }
            )
        if config["mode"] == "offline":
            split_equal = old.get("split") is not None and old["split"] == new.get("split")
            windows_equal = old.get("windows") is not None and old["windows"] == new.get("windows")
            for counter in selected:
                counter["offline_cases"] += 1
                counter["offline_split_exact_matches"] += split_equal
                counter["offline_window_exact_matches"] += windows_equal
                counter["offline_split_and_window_exact_matches"] += split_equal and windows_equal
            detail.update({"split_equal": split_equal, "windows_equal": windows_equal})
            changed |= not (split_equal and windows_equal)
            if not split_equal:
                detail.update({"previous_split": old.get("split"), "current_split": new.get("split")})
            if not windows_equal:
                detail.update({"previous_windows": old.get("windows"), "current_windows": new.get("windows")})
        else:
            missing = not old.get("prediction_digest") or not new.get("prediction_digest")
            digest_equal = not missing and old["prediction_digest"] == new["prediction_digest"]
            for counter in selected:
                counter["prequential_cases"] += 1
                counter["prediction_digest_exact_matches"] += digest_equal
                counter["missing_prediction_digests"] += bool(missing)
            detail["prediction_digest_equal"] = digest_equal
            changed |= not digest_equal
            if not digest_equal:
                detail.update(
                    {
                        "previous_prediction_digest": old.get("prediction_digest"),
                        "current_prediction_digest": new.get("prediction_digest"),
                    }
                )
        if changed:
            differences.append(detail)
    return {
        **counters,
        "differences": differences,
        "interpretation": "Exact equality is measured, never assumed from matching configurations.",
    }


def _authorized_sources(reports, manifest):
    counts = defaultdict(int)
    for report in reports:
        counts[report["environment"]["code_sha256"]] += 1
    if manifest is None:
        assert len(counts) == 1, "v5 code changed during campaign"
        return next(iter(counts)), dict(sorted(counts.items()))
    rows = manifest["configurations"]
    expected = {row["run_key"]: row for row in rows}
    assert len(expected) == len(rows) == len(reports), "manifest run count differs"
    assert manifest["expected_runs"] == len(reports), "manifest expected run count differs"
    assert set(expected) == {r["run_key"] for r in reports}, "manifest run coverage differs"
    primary = manifest["source_sha256"]
    assert isinstance(primary, str) and primary, "invalid primary source"
    for report in reports:
        row = expected[report["run_key"]]
        assert report["config"] == row["config"], "run configuration differs from manifest"
        source = row.get("source_sha256", primary)
        assert isinstance(source, str) and source, "invalid per-run source override"
        assert report["environment"]["code_sha256"] == source, "unauthorized per-run code source"
    return primary, dict(sorted(counts.items()))


def analyze(current, previous, *, expected_cases=289, synthetic_replicates=5, manifest=None):
    """Testing may reduce expected_cases/replicates; the CLI audits the full 867-vs-289 campaign."""
    old_reports, reports = _reports(previous), _reports(current)
    assert len(old_reports) == expected_cases, "unexpected v4 case count"
    assert len(reports) == len(MODELS) * expected_cases, "unexpected v5 run count"
    previous_index = {}
    for report in old_reports:
        assert report["config"]["model"]["name"] == "logistic", "v4 baseline must be logistic"
        key = case_key(report["config"])
        assert key not in previous_index, "duplicate v4 case"
        previous_index[key] = report
    indexes = {model: {} for model in MODELS}
    groups = defaultdict(list)
    for report in reports:
        config = report["config"]
        model = config["model"]["name"]
        assert model in indexes, "unexpected model family"
        key = case_key(config)
        assert key not in indexes[model], "duplicate v5 model/case"
        assert key in previous_index, "v5 case is not present in v4"
        old = previous_index[key]
        assert report["dataset"]["sha256"] == old["dataset"]["sha256"], "paired dataset SHA mismatch"
        assert report["metrics"]["n"] == old["metrics"]["n"], "paired evaluation count mismatch"
        indexes[model][key] = report
        groups[(model, config["dataset"]["name"], config["mode"], arm(report))].append(report)
    assert all(set(index) == set(previous_index) for index in indexes.values()), "model case coverage differs"
    primary_source, source_counts = _authorized_sources(reports, manifest)
    aggregates = []
    paired = []
    for (model, dataset, mode, strategy), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["config"]["dataset"]["seed"])
        expected_replicates = 1 if dataset in REAL_DATASETS else synthetic_replicates
        assert len(rows) == expected_replicates, "unexpected group replicate count"
        assert len({r["config"]["dataset"]["seed"] for r in rows}) == len(rows), "duplicate group seed"
        costs = [_observations(r) for r in rows]
        aggregates.append(
            {
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "strategy": strategy,
                "seeds": [r["config"]["dataset"]["seed"] for r in rows],
                "metrics": {m: describe([r["metrics"][m] for r in rows]) for m in METRICS},
                **{name: describe([cost[name] for cost in costs]) for name in costs[0]},
                "run_keys": [r["run_key"] for r in rows],
            }
        )
        if model == "logistic":
            continue
        controls = [indexes["logistic"][case_key(r["config"])] for r in rows]
        comparisons = {}
        for metric in METRICS:
            deltas = [r["metrics"][metric] - control["metrics"][metric] for r, control in zip(rows, controls)]
            sign = 1 if metric in {"accuracy", "f1"} else -1
            comparisons[metric] = {
                **describe(deltas),
                "wins": sum(sign * d > 0 for d in deltas),
                "ties": sum(d == 0 for d in deltas),
                "losses": sum(sign * d < 0 for d in deltas),
                "higher_is_better": sign == 1,
            }
        paired.append(
            {
                "model": model,
                "baseline": "logistic",
                "dataset": dataset,
                "mode": mode,
                "strategy": strategy,
                "seeds": [r["config"]["dataset"]["seed"] for r in rows],
                "delta_definition": "nonlinear minus paired logistic v5",
                "metrics": comparisons,
                "run_keys": [r["run_key"] for r in rows],
                "control_run_keys": [r["run_key"] for r in controls],
            }
        )
    return {
        "campaign": "nonlinear-v5",
        "complete": True,
        "runs": len(reports),
        "previous_runs": len(old_reports),
        "models": list(MODELS),
        "cases_per_model": expected_cases,
        "source_sha256": primary_source,
        "source_revisions": sorted(source_counts),
        "source_counts": source_counts,
        "previous_source_sha256": sorted({r["environment"]["code_sha256"] for r in old_reports}),
        "dataset_sha_pairs_verified": len(reports),
        "groups": aggregates,
        "paired_model_comparisons": paired,
        "logistic_v4_comparison": _bridge(indexes["logistic"], previous_index),
        "cost_scope": {
            "training_used": "Post-warmup presentations, including replays; offline initial fit excluded",
            "compute_seconds": "Reported evaluation timer; not controlled hardware CPU cost",
            "wall_seconds": "Evaluation-attempt wall timer, not full orchestration/publication or CPU cost",
            "peak_rss_bytes": "Process peak RSS, including interpreter/dependencies; not model-only memory",
        },
        "limitations": [
            "Five synthetic seeds; real datasets have one fixed prefix, without independent replicates",
            "The 867 configurations share data and are not 867 independent experimental replicates",
            "Sample SD is descriptive; no significance claim or pooling of incompatible protocols",
            "Same presentation ceiling does not equal same consumption, retained history or compute cost",
            "Online continues learning; periodic/drift reset and retrain on at most 256 released examples",
            "KNN stores at most 256 neighbors, including offline; the full training prefix is not retained",
            "KNN preserves admission-time scaling; its evolving scaler can change coordinate systems",
            "Hoeffding-tree growth/memory/nominal settings are fixed; no independent hyperparameter tuning",
            "Tree native inputs differ from scaled/encoded logistic and KNN; capacity alone is not isolated",
            "Real timestamps and label delay are simulated; real drift points are not annotated",
            "Electricity upstream normalization/imputation fit scope remains undocumented",
            "Wall/compute timers include different versioning overhead; neither is a controlled CPU assay",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="reports/generated/nonlinear-v5-inputs.json")
    parser.add_argument("--previous", default="reports/generated/versioned-v4-inputs.json")
    parser.add_argument("--output", default="reports/nonlinear-v5-comparison.json")
    parser.add_argument("--manifest", default="configs/campaign-nonlinear-v5.json")
    args = parser.parse_args()
    raw_current, raw_previous = Path(args.current).read_bytes(), Path(args.previous).read_bytes()
    raw_manifest = Path(args.manifest).read_bytes()
    current, previous, manifest = json.loads(raw_current), json.loads(raw_previous), json.loads(raw_manifest)
    # The CLI checks row-level source/config authorization before calculating comparisons.
    _authorized_sources(_reports(current), manifest)
    result = analyze(current, previous, manifest=manifest)
    result["input_sha256"] = {
        "current": hashlib.sha256(raw_current).hexdigest(),
        "previous": hashlib.sha256(raw_previous).hexdigest(),
        "manifest": hashlib.sha256(raw_manifest).hexdigest(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": result["runs"],
                "groups": len(result["groups"]),
                "logistic_v4_comparison": result["logistic_v4_comparison"]["all"],
            }
        )
    )


if __name__ == "__main__":
    main()
