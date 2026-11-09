"""Batch-prequential benchmarks on the full, ordered Zia streams."""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from river.drift import ADWIN

from continual_bench.config import TrackingConfig
from continual_bench.datasets.zia import DATASETS, SCOPES, load_manifest, sha256_file
from continual_bench.evaluation.costs import account, aws_account, profile
from continual_bench.evaluation.environment import encoded, environment, peak_rss_bytes
from continual_bench.evaluation.system_metrics import SystemMetricsSampler
from continual_bench.evaluation.tracking import metric_catalog
from continual_bench.evaluation.zia_metrics import SCORE_BINS, ClassificationMetrics, ratio
from continual_bench.infrastructure.storage import atomic_write
from continual_bench.models.zia import FraudOrReturnBooster, MultinomialLogistic

POLICIES = ("frozen", "periodic", "drift", "incremental")
BLOCK_ROWS = {"smoke": 1000, "full": 250_000}
REPLAY_BLOCKS = 4
HOLDOUT_FRACTION = 0.1
ADWIN_DELTA = 0.002


def chronological_blocks(path, size):
    """Carry timestamp ties forward to keep a whole timestamp in one block."""
    tail = []
    with pq.ParquetFile(path) as source:
        for batch in source.iter_batches(batch_size=size, use_threads=False):
            if any(batch[i].null_count for i in (0, 1)):
                raise ValueError("null timestamp or label")
            timestamps, y = (batch[i].to_numpy() for i in (0, 1))
            x = np.column_stack([col.cast(pa.float32()).to_numpy(zero_copy_only=False)
                                 for col in batch.columns[2:]])
            end = int(np.searchsorted(timestamps, timestamps[-1]))
            if end:
                tail.append((timestamps[:end], x[:end], y[:end]))
            if tail and (end or tail[-1][0][-1] < timestamps[0]):
                yield tuple(np.concatenate(parts) for parts in zip(*tail, strict=True))
                tail.clear()
            tail.append((timestamps[end:], x[end:], y[end:]))
    if tail:
        yield tuple(np.concatenate(parts) for parts in zip(*tail, strict=True))


def batches(path, size, split):
    offset = 0
    holdout = False
    for timestamps, x, y in chronological_blocks(path, size):
        cut = split - offset
        if not holdout and 0 <= cut < len(y):
            cut = int(np.searchsorted(timestamps, timestamps[cut]))
            if cut:
                yield timestamps[:cut], x[:cut], y[:cut], False
            yield timestamps[cut:], x[cut:], y[cut:], True
            holdout = True
        else:
            yield timestamps, x, y, holdout
        offset += len(y)


def run(root, dataset, scope, policy, output, tracking_settings):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path, manifest = load_manifest(root, dataset, scope)
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError("prepared dataset checksum differs from its manifest")
    spec = DATASETS[dataset]
    classes = len(spec["classes"])
    model = MultinomialLogistic(classes) if dataset == "social" else FraudOrReturnBooster()
    monitoring = SystemMetricsSampler(output, interval=0.5 if scope == "smoke" else 5)
    replay = deque(maxlen=REPLAY_BLOCKS)
    detector = ADWIN(delta=ADWIN_DELTA, clock=1) if policy == "drift" else None
    online, holdout = ClassificationMetrics(classes), ClassificationMetrics(classes)
    windows, versions = [], []
    class_counts = np.zeros(classes, dtype=np.int64)
    train_seconds = predict_seconds = 0.0
    presentations = unique = resets = alarms = rows = stream_blocks = 0
    last_training_end = last_timestamp = None
    block_rows = BLOCK_ROWS[scope]
    warmup = 0
    split = manifest["rows"] - int(manifest["rows"] * HOLDOUT_FRACTION)
    with monitoring:
        for step, (timestamps, x, y, is_holdout) in enumerate(batches(path, block_rows, split)):
            if ((last_timestamp is not None and timestamps[0] < last_timestamp)
                    or np.any(timestamps[1:] < timestamps[:-1]) or np.any(y >= classes)):
                raise ValueError("unordered timestamps or invalid labels")
            last_timestamp = int(timestamps[-1])
            start, rows = rows, rows + len(y)
            class_counts += np.bincount(y, minlength=classes)
            action = "initial" if step == 0 else None
            if not step:
                warmup = len(y)
            if step:
                timer = time.monotonic()
                probabilities = model.predict(x)
                elapsed = time.monotonic() - timer
                predict_seconds += elapsed
                window = ClassificationMetrics(classes)
                window.add(y, probabilities)
                (holdout if is_holdout else online).merge(window)
                metrics, _ = window.metrics()
                windows.append({
                    "end_seq": rows, "scope": "holdout" if is_holdout else "prequential",
                    "start_time_ms": int(timestamps[0]), "end_time_ms": last_timestamp,
                    "prediction_seconds": elapsed, "prediction_throughput_per_second": len(y) / elapsed,
                    **metrics,
                })
                if not is_holdout:
                    stream_blocks += 1
                    if policy in ("periodic", "drift"):
                        replay.append((start, x, y))
                    if detector:
                        detector.update(1 - metrics["accuracy"])
                        alarms += int(detector.drift_detected)
                    if policy == "incremental":
                        action = "continue"
                    elif ((policy == "periodic" and stream_blocks % REPLAY_BLOCKS == 0)
                          or (detector and detector.drift_detected)):
                        action = "reset_retrain"
            if action:
                timer = time.monotonic()
                if action == "reset_retrain":
                    train_start = replay[0][0]
                    fit_x = np.concatenate([item[1] for item in replay])
                    fit_y = np.concatenate([item[2] for item in replay])
                    resets += 1
                else:
                    train_start, fit_x, fit_y = start, x, y
                iterations = model.fit(fit_x, fit_y, action == "continue")
                elapsed = time.monotonic() - timer
                train_seconds += elapsed
                presentations += len(fit_y)
                unique += rows - max(train_start, last_training_end or 0)
                last_training_end = rows
                payload, extension = model.payload()
                filename = f"models/{len(versions):04d}.{extension}"
                atomic_write(output / filename, payload)
                versions.append({
                    "end_seq": rows, "file": filename, "action": action, "start_seq": train_start,
                    "sha256": sha256_file(output / filename), "bytes": len(payload),
                    "train_samples": len(fit_y), "train_seconds": elapsed, "iterations": iterations,
                })
            if step % 10 == 0:
                atomic_write(output / "progress.json", encoded({
                    "rows": rows, "total_rows": manifest["rows"], "model_versions_count": len(versions),
                    "elapsed_seconds": time.monotonic() - started,
                }))
    if rows != manifest["rows"] or not online.confusion.sum() or not holdout.confusion.sum():
        raise ValueError("dataset coverage or temporal split is incomplete")
    summary, by_class = online.metrics()
    final, final_by_class = holdout.metrics()
    errors = int(online.confusion.sum() - online.confusion.trace())
    updates = len(versions) - 1
    model_bytes = sum(v["bytes"] for v in versions)
    costs = profile(tracking_settings["costs"])
    error_units = (summary["fp"] * costs["false_positive"] + summary["fn"] * costs["false_negative"]
                   if classes == 2 else errors * costs["classification_error"])
    cost_report = account(costs, errors=error_units, presentations=presentations, updates=updates,
                          train_seconds=train_seconds, predict_seconds=predict_seconds,
                          model_storage_bytes=model_bytes)
    compute_seconds = time.monotonic() - started
    system = monitoring.report()
    rss_samples = [(sample["elapsed_seconds"], sample["values"]["process_memory_rss_megabytes"] * 1024**2)
                   for sample in system["samples"] if "process_memory_rss_megabytes" in sample["values"]]
    rss_integral = sum((end - start) * (before + after) / 2
                       for (start, before), (end, after) in zip(rss_samples, rss_samples[1:]))
    wall_cost = aws_account(costs["aws"], train_seconds=compute_seconds, predict_seconds=0,
                            model_storage_bytes=model_bytes)
    summary.update(
        dataset_rows=rows, initial_train_rows=warmup, holdout_rows=final["n"],
        training_presentations=presentations, training_used_post_warmup=presentations - warmup,
        training_unique_examples=unique, training_repeated_presentations=presentations - unique,
        update_count=updates, retrain_count=resets, alarm_count=alarms,
        updates_per_1000_evaluated=1000 * updates / summary["n"],
        train_seconds=train_seconds, initial_train_seconds=versions[0]["train_seconds"],
        update_train_seconds=train_seconds - versions[0]["train_seconds"],
        postwarmup_train_samples_per_second=ratio(presentations - warmup,
                                                  train_seconds - versions[0]["train_seconds"]),
        prediction_seconds=predict_seconds,
        prediction_latency_mean_seconds=ratio(predict_seconds, rows - warmup),
        prediction_throughput_per_second=ratio(rows - warmup, predict_seconds),
        evaluation_compute_seconds=compute_seconds, evaluation_attempt_wall_seconds=compute_seconds,
        process_peak_rss_bytes=peak_rss_bytes(), model_versions_count=len(versions),
        model_storage_bytes=model_bytes, mean_window_accuracy=sum(
            w["accuracy"] for w in windows if w["scope"] == "prequential"
        ) / stream_blocks,
        drift_alarm_count=alarms, peak_rss_bytes=peak_rss_bytes(),
        sampled_rss_byte_seconds=rss_integral if len(rss_samples) > 1 else None,
        samples_to_target_accuracy=next(
            (w["end_seq"] for w in windows if w["scope"] == "prequential"
             and w["accuracy"] >= tracking_settings["target_accuracy"]), None
        ),
        scenario_total_cost=cost_report["metrics"]["cost_total"],
        **cost_report["metrics"], **{"aws_" + k: v for k, v in cost_report["aws"]["metrics"].items()},
        **{"aws_evaluation_" + k: v for k, v in wall_cost["metrics"].items()},
    )
    error_sum = sum(w["error_rate"] for w in windows if w["scope"] == "prequential")
    alpha = tracking_settings["upf_alpha"]
    summary["upf_total_cost"] = alpha * resets + error_sum if policy != "incremental" else None
    summary["posthoc_update_event_cost"] = alpha * updates + error_sum
    sensitivity = [[a, a * updates + error_sum] for a in tracking_settings["upf_alphas"]]
    summary["cost_sensitivity_auc"] = sum((b - a) * (v + w) / 2
                                          for (a, v), (b, w) in zip(sensitivity, sensitivity[1:]))
    availability = {
        name: {"status": "unavailable", "reason": "undefined_or_requires_other_evaluation_evidence"}
        for name in metric_catalog()[0]["definitions"]
        if summary.get(name) is None
    }
    env = environment()
    if model.name == "xgboost":
        env["packages"]["xgboost"] = model.xgb.__version__
    report = {
        "complete": True, "protocol": "batch_prequential_with_final_temporal_holdout",
        "dataset": dataset, "scope": scope, "policy": policy, "model": model.name,
        "model_params": {**model.params, "initial_rounds": model.initial_rounds,
                         "continued_rounds": model.continued_rounds},
        "contract": {"input": manifest, "environment": env, "block_rows": block_rows,
                     "holdout_start": rows - final["n"], "replay_blocks": REPLAY_BLOCKS,
                     "adwin_delta": ADWIN_DELTA, "score_bins": SCORE_BINS,
                     "decision_rule": "binary: p>=0.5; multiclass: argmax, ties favor lower class ID"},
        "class_counts": class_counts.tolist(),
        "metrics": summary, "holdout": final, "windows": windows,
        "prequential_confusion": online.confusion.tolist(), "holdout_confusion": holdout.confusion.tolist(),
        "class_metrics": dict(zip(spec["classes"], by_class, strict=True)),
        "holdout_class_metrics": dict(zip(spec["classes"], final_by_class, strict=True)),
        "model_versions": versions, "model_explanation": model.explanation(manifest["features"]),
        "costs": cost_report, "evaluation_aws_cost": wall_cost, "cost_sensitivity": sensitivity,
        "availability": availability, "system_metrics": system,
    }
    atomic_write(output / "report.json", encoded(report))
    atomic_write(output / "progress.json", encoded({"rows": rows, "total_rows": rows, "complete": True}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("policy", choices=POLICIES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tracking", type=Path, default=Path("configs/base/tracking.yaml"))
    parser.add_argument("--campaign", help="Publish the run to the configured MLflow server")
    args = parser.parse_args()
    settings = TrackingConfig.model_validate(yaml.safe_load(args.tracking.read_text())).model_dump()
    if args.campaign:
        from mlflow.tracking import MlflowClient

        from continual_bench.infrastructure.zia_tracking import begin, publish

        client = MlflowClient()
        run_id, experiment_id = begin(
            client, args.output, args.dataset, args.scope, args.policy, args.campaign,
        )
        try:
            report_path = args.output / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
            else:
                report = run(
                    args.root, args.dataset, args.scope, args.policy, args.output, settings,
                )
            publish(client, args.output, run_id, experiment_id, report)
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise
    else:
        report = run(args.root, args.dataset, args.scope, args.policy, args.output, settings)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
