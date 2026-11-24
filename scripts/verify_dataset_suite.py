"""Reconstruct ordered-stream scores and paired comparison cohorts from predictions.

No model state is loaded. Ranking scores use the evaluator's declared 4,096-bin
estimator; this verifier does not describe those scores as exact ranking areas.
The optional MLflow readback checks every finite point of the parent tracking_v1
catalog, without inspecting the service database or assuming latest-value views
contain the full history.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from continual_bench.evaluation.zia_artifacts import tracking_report
from continual_bench.evaluation.zia_metrics import SCORE_BINS, ClassificationMetrics
from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files
from continual_bench.infrastructure.metric_tracking import metric_points
from continual_bench.infrastructure.storage import atomic_write


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    atomic_write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_metrics(observed, expected, scope):
    """Null availability is part of equality; it is never silently replaced by zero."""
    checked = 0
    for key, value in observed.items():
        require(key in expected, f"{scope}: missing {key}")
        recorded = expected[key]
        if value is None or recorded is None:
            require(value is None and recorded is None, f"{scope}: availability differs for {key}")
        else:
            require(math.isfinite(recorded), f"{scope}: nonfinite {key}")
            require(math.isclose(value, recorded, rel_tol=1e-10, abs_tol=1e-11), f"{scope}: differs {key}")
        checked += 1
    return checked


def verify_retention(directory, report):
    verify_metrics_only_files(directory)
    for path in Path(directory).rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        require(not set(relative.parts) & {"models", "candidates", ".zia-resume"}, "model-state directory")
        if path.suffix == ".json":
            value = read(path)
            require(not isinstance(value, dict) or "learner" not in value, "serialized booster payload")
    require(
        report.get("artifact_mode", report["contract"].get("artifact_mode")) == "metrics_only",
        "report does not declare metrics_only",
    )
    require(all(v.get("file") is None for v in report["model_versions"]), "persisted deployed model")
    for fit in report["training_history"]:
        checkpoint = fit.get("checkpoint") or {}
        require(checkpoint.get("file") is None, "persisted candidate checkpoint")


def calibration_end(report):
    retraining = report.get("retraining")
    if not retraining:
        return int(report["metrics"]["initial_train_rows"])
    offline = int(retraining["setup"]["offline_batches"])
    windows = [w for w in report["windows"] if w["scope"] == "prequential"]
    require(offline >= 2 and len(windows) >= offline - 1, "invalid calibration boundary")
    return int(windows[offline - 2]["end_seq"])


def verify_case(directory, case, *, common_range=None):
    directory = Path(directory)
    report, config = read(directory / "report.json"), case["config"]
    require(report.get("complete") is True, "evaluation is incomplete")
    require(
        (report["dataset"], report["scope"], report["policy"])
        == (config["dataset"], config["scope"], config["policy"]),
        "case coordinates differ",
    )
    require(report["contract"]["input"]["sha256"] == config["input_sha256"], "input identity differs")
    require(
        report["contract"]["iterative_training"] == config["iterative_training"],
        "iterative training or seed contract differs",
    )
    require(report["model_params"].get("seed") == config["seed"], "effective model seed differs")
    require(report["contract"]["score_bins"] == SCORE_BINS, "ranking estimator differs")
    fits = report["training_history"]
    require(bool(fits), "missing per-fit training history")
    require(all(fit.get("seed") == config["seed"] for fit in fits), "per-fit seed is absent or differs")
    verify_retention(directory, report)
    index = read(directory / "predictions-manifest.json")
    classes = index["classes"]
    require(index["dataset_sha256"] == config["input_sha256"], "prediction input identity differs")
    threshold = report["contract"].get("decision_threshold")
    threshold = 0.5 if threshold is None else threshold
    metrics = {
        "prequential": ClassificationMetrics(len(classes), threshold),
        "holdout": ClassificationMetrics(len(classes), threshold),
    }
    windows = report["windows"]
    window_ends = np.array([int(w["end_seq"]) for w in windows], dtype=np.int64)
    require(len(windows) > 0 and np.all(np.diff(window_ends) > 0), "invalid prediction windows")
    window_counts = [ClassificationMetrics(len(classes), threshold) for _ in windows]
    common = ClassificationMetrics(len(classes), threshold)
    warmup = int(report["metrics"]["initial_train_rows"])
    total = int(report["contract"]["input"]["rows"])
    holdout_start = int(report["contract"]["holdout_start"])
    require(
        warmup + report["metrics"]["n"] + report["holdout"]["n"] == total,
        "warmup plus evaluation plus holdout does not cover the source",
    )
    require(report["metrics"]["dataset_rows"] == total, "dataset rows differ")
    next_seq, parts = warmup, []
    independent_confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for part in index["parts"]:
        name = part["file"]
        require(Path(name).name == name and name.endswith(".parquet"), "invalid prediction part path")
        path = directory / "predictions" / name
        require(
            path.stat().st_size == part["bytes"] and sha256(path) == part["sha256"],
            "prediction part checksum differs",
        )
        part_rows = 0
        with pq.ParquetFile(path) as reader:
            for batch in reader.iter_batches(batch_size=65_536, use_threads=False):
                data = batch.to_pydict()
                seq = np.asarray(data["seq"], dtype=np.int64)
                labels = np.asarray(data["label"], dtype=np.uint8)
                probabilities = np.column_stack([data[f"p_{i}"] for i in range(len(classes))])
                held = np.asarray(data["holdout"], dtype=bool)
                predicted = np.asarray(data["prediction"], dtype=np.int32)
                serving = np.asarray(data["model_end_seq"], dtype=np.int64)
                require(
                    np.array_equal(seq, np.arange(next_seq, next_seq + len(seq))),
                    "prediction sequence has gaps, duplicates or reordering",
                )
                require(np.all((labels >= 0) & (labels < len(classes))), "labels outside class order")
                require(
                    np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)),
                    "invalid probabilities",
                )
                require(
                    np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-7),
                    "probabilities do not sum to one",
                )
                decisions = (
                    (probabilities[:, 1] >= threshold).astype(np.int32)
                    if len(classes) == 2
                    else probabilities.argmax(axis=1)
                )
                require(np.array_equal(decisions, predicted), "prediction decision rule differs")
                require(np.all(serving <= seq), "future model used for an earlier prediction")
                require(np.array_equal(held, seq >= holdout_start), "holdout membership differs")
                for scope, mask in (("prequential", ~held), ("holdout", held)):
                    if mask.any():
                        metrics[scope].add(labels[mask], probabilities[mask])
                independent_confusion += np.bincount(
                    len(classes) * labels.astype(np.int64) + predicted, minlength=len(classes) ** 2
                ).reshape(len(classes), -1)
                membership = np.searchsorted(window_ends, seq, side="right")
                require(np.all(membership < len(windows)), "prediction outside recorded windows")
                for i in np.unique(membership):
                    mask = membership == i
                    window_counts[i].add(labels[mask], probabilities[mask])
                if common_range:
                    mask = (~held) & (seq >= common_range[0]) & (seq < common_range[1])
                    if mask.any():
                        common.add(labels[mask], probabilities[mask])
                next_seq += len(seq)
                part_rows += len(seq)
        require(part_rows == part["rows"], "prediction part row count differs")
        parts.append({"file": name, "rows": part_rows, "sha256": part["sha256"]})
    require(next_seq == total and index["rows"] == total - warmup, "prediction coverage incomplete")
    require(sum(p["rows"] for p in parts) == index["rows"], "prediction index row count differs")
    checks = compare_metrics(metrics["prequential"].metrics()[0], report["metrics"], "prequential")
    checks += compare_metrics(metrics["holdout"].metrics()[0], report["holdout"], "holdout")
    for i, counts in enumerate(window_counts):
        checks += compare_metrics(counts.metrics()[0], windows[i], f"window {i}")
    require(
        np.array_equal(
            independent_confusion, metrics["prequential"].confusion + metrics["holdout"].confusion
        ),
        "independent confusion reconstruction differs",
    )
    adapted = tracking_report(report, index)
    points = metric_points(adapted["tracking"])
    for key, value in adapted["tracking"]["metrics"].items():
        availability = adapted["tracking"]["availability"][key]
        require(
            availability["status"] == ("available" if value is not None else "unavailable"),
            "tracking availability differs from recorded value",
        )
    result = {
        "complete": True,
        "case_id": case["case_id"],
        "dataset": report["dataset"],
        "policy": report["policy"],
        "seed": config["seed"],
        "source_rows": total,
        "input_sha256": config["input_sha256"],
        "prediction_rows": index["rows"],
        "prediction_parts": parts,
        "metric_checks": checks,
        "fit_seed_checks": len(fits),
        "tracking_catalog_points": len(points),
        "tracking_null_points": sum(v is None for _, _, v in points),
        "calibration_end_seq": calibration_end(report),
        "holdout_start": holdout_start,
        "classes": classes,
        "ranking_estimator": "uniform_histogram_4096_bins",
        "report_sha256": sha256(directory / "report.json"),
    }
    if common_range:
        values, by_class = common.metrics()
        require(values["n"] == common_range[1] - common_range[0], "common cohort is incomplete")
        values["brier_half_sum"] = values["brier"] / (1 if len(classes) == 2 else 2)
        result["common_comparison"] = {
            "seq_range": list(common_range),
            "metrics": values,
            "class_metrics": dict(zip(classes, by_class, strict=True)),
            "brier_half_sum_definition": "sum_k((p_k-1[y=k])^2)/(2*n), range 0..1",
            "positive_ranking": "pr_auc/roc_auc describe class index 1 only for binary targets",
            "macro_ranking": "macro_pr_auc/macro_roc_auc average defined one-vs-rest class scores",
        }
    return result


def verify_native(directory, report, receipt, client):
    require(receipt and receipt.get("complete") is True, "tracking receipt is incomplete")
    run_id = receipt["mlflow_run_id"]
    run = client.get_run(run_id)
    require(run.info.status == "FINISHED", "native run is not FINISHED")
    index = read(Path(directory) / "predictions-manifest.json")
    expected = defaultdict(dict)
    for key, step, value in metric_points(tracking_report(report, index)["tracking"]):
        if value is not None:
            expected[key][step] = value
    checked = 0
    for key, points in expected.items():
        history = client.get_metric_history(run_id, key)
        observed = defaultdict(list)
        for item in history:
            observed[item.step].append(item.value)
        require(set(observed) == set(points), f"native metric axis differs for {key}")
        for step, value in points.items():
            require(len(observed[step]) == 1, f"native metric coordinate duplicated for {key}/{step}")
            require(
                math.isclose(observed[step][0], value, rel_tol=1e-12, abs_tol=1e-12),
                f"native metric differs for {key}/{step}",
            )
            checked += 1
    return {
        "complete": True,
        "run_id": run_id,
        "series": len(expected),
        "points": checked,
        "scope": "all finite parent tracking_v1 catalog coordinates",
    }


def comparison_tables(results, output):
    rows, groups = [], defaultdict(list)
    for result in results:
        values = result["common_comparison"]["metrics"]
        for metric, value in values.items():
            if not (value is None or isinstance(value, (int, float))):
                continue
            row = {
                "dataset": result["dataset"],
                "policy": result["policy"],
                "seed": result["seed"],
                "metric": metric,
                "value": value,
                "cohort_n": values["n"],
            }
            rows.append(row)
            groups[(row["dataset"], row["policy"], metric)].append(row)
    aggregate = []
    for (dataset, policy, metric), group in sorted(groups.items()):
        finite = [r["value"] for r in group if r["value"] is not None]
        aggregate.append(
            {
                "dataset": dataset,
                "policy": policy,
                "metric": metric,
                "n_seeds_present": len(group),
                "n_defined": len(finite),
                "n_undefined": len(group) - len(finite),
                "mean": statistics.mean(finite) if finite else None,
                "sample_sd": statistics.stdev(finite) if len(finite) >= 2 else None,
                "sd_ddof": 1,
            }
        )
    for name, data in (("comparison-by-seed.csv", rows), ("comparison-mean-sd.csv", aggregate)):
        path = Path(output) / name
        if data:
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    return {"seed_rows": len(rows), "aggregate_rows": len(aggregate), "sample_sd_ddof": 1}


def verify_suite(output, *, native_tracking=False):
    output = Path(output)
    plan = read(output / "suite-plan.json")
    require(plan["schema"] == "ordered-dataset-suite-v1", "unsupported suite plan")
    cases, expected = plan["cases"], set()
    groups, receipts = defaultdict(list), {}
    for case in cases:
        key = case["case_id"]
        require(key not in expected, "duplicate case in suite")
        expected.add(key)
        receipt = read(output / "results" / (key + ".json"))
        require(
            receipt["status"] == "passed" and receipt["case"] == case and receipt["code"] == plan["code"],
            "case receipt differs from suite plan",
        )
        directory = output / "cases" / key
        for path, digest in receipt["files"].items():
            require(sha256(directory / path) == digest, "case evidence changed after execution")
        report = read(directory / "report.json")
        group_key = case["config"]["dataset"], case["config"]["input_sha256"]
        groups[group_key].append((case, report))
        receipts[key] = receipt
    ranges = {}
    for key, group in groups.items():
        starts = [calibration_end(report) for _, report in group]
        ends = [report["contract"]["holdout_start"] for _, report in group]
        start, end = max(starts), min(ends)
        require(start < end, "no common post-calibration prequential cohort")
        ranges[key] = (start, end)
    client = None
    if native_tracking:
        from mlflow.tracking import MlflowClient

        require(plan["tracking_required"], "suite was not configured to publish tracking")
        client = MlflowClient()
    results = []
    for key, group in groups.items():
        for case, report in group:
            directory = output / "cases" / case["case_id"]
            result = verify_case(directory, case, common_range=ranges[key])
            if plan["tracking_required"]:
                require(
                    receipts[case["case_id"]].get("tracking", {}).get("complete") is True,
                    "tracking receipt is incomplete",
                )
            if client is not None:
                result["native_tracking"] = verify_native(
                    directory, report, receipts[case["case_id"]]["tracking"], client
                )
            results.append(result)
    require(len(results) == len(cases), "suite verification is incomplete")
    report = {
        "schema": "ordered-dataset-suite-qa-v1",
        "complete": True,
        "scope": plan["scope"],
        "code": plan["code"],
        "suite_plan_sha256": sha256(output / "suite-plan.json"),
        "coverage": {
            "expected_cases": len(cases),
            "passed_cases": len(results),
            "datasets": sorted({c["config"]["dataset"] for c in cases}),
            "policies": sorted({c["config"]["policy"] for c in cases}),
            "seeds": sorted({c["config"]["seed"] for c in cases}),
        },
        "checks": {
            "prediction_integrity": True,
            "metrics_reconstructed": True,
            "no_model_files": True,
            "fit_seeds_verified": True,
            "tracking_receipts_complete": plan["tracking_required"],
            "native_tracking_catalog_verified": native_tracking,
        },
        "comparison_tables": comparison_tables(results, output),
        "cases": results,
        "interpretation": {
            "paired_comparisons": "same dataset, seed and common source sequence range",
            "cross_dataset": "descriptive normalized metrics, not identical tasks or independent populations",
            "oracle": "retrospective reference for its objective, not a universal quality ceiling",
            "undefined": "nulls are retained; aggregate sample sizes report only defined values",
        },
    }
    save(output / "suite-qa.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-tracking", action="store_true")
    args = parser.parse_args()
    report = verify_suite(args.output, native_tracking=args.native_tracking)
    print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "interpretation"}}, indent=2))


if __name__ == "__main__":
    main()
