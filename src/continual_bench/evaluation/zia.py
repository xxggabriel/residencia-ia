"""Batch-prequential benchmarks on the full, ordered Zia streams."""

import argparse
import gc
import json
import os
import time
from collections import deque
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from river.drift import ADWIN  # noqa: E402
from river.drift.binary import DDM  # noqa: E402

from continual_bench.config import IterativeTrainingConfig, TrackingConfig  # noqa: E402
from continual_bench.core.contracts import DecisionContext  # noqa: E402
from continual_bench.core.policies import BATCH_POLICY_KINDS, PolicyStrategy, ZiaPolicy  # noqa: E402
from continual_bench.datasets.zia import DATASETS, SCOPES, load_manifest, sha256_file  # noqa: E402
from continual_bench.evaluation import zia_retraining  # noqa: E402
from continual_bench.evaluation.costs import account, aws_account, profile  # noqa: E402
from continual_bench.evaluation.environment import encoded, environment, peak_rss_bytes  # noqa: E402
from continual_bench.evaluation.system_metrics import SystemMetricsSampler  # noqa: E402
from continual_bench.evaluation.tracking import metric_catalog  # noqa: E402
from continual_bench.evaluation.zia_metrics import (  # noqa: E402
    SCORE_BINS,
    ClassificationMetrics,
    decide,
    decision_threshold,
    ratio,
)
from continual_bench.evaluation.zia_observations import TemporalHoldoutProbes, run_observations  # noqa: E402
from continual_bench.evaluation.zia_predictions import (  # noqa: E402
    PredictionArchive,
    prediction_manifest,
)
from continual_bench.evaluation.zia_training_history import TrainingHistory, unique_train_rows  # noqa: E402
from continual_bench.infrastructure.storage import atomic_write, run_lock  # noqa: E402
from continual_bench.models.zia import make_zia_model  # noqa: E402

POLICIES = get_args(ZiaPolicy)
BLOCK_ROWS = {"smoke": 1000, "full": 250_000}
REPLAY_BLOCKS = 4
HOLDOUT_FRACTION = 0.1
ADWIN_DELTA = 0.002
ADWIN_CLOCK = 32
DDM_PARAMS = {"warm_start": 30, "warning_threshold": 2.0, "drift_threshold": 3.0}


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


def drift_alarms(detector, policy, probabilities, labels, metrics, threshold=0.5, *, on_alarm=None):
    if policy == "drift":
        detector.update(1 - metrics["accuracy"])
        if on_alarm is not None and detector.drift_detected:
            on_alarm(len(labels) - 1)
        return int(detector.drift_detected)
    decisions = decide(probabilities, threshold)
    alarms = 0
    for index, error in enumerate((decisions != labels).tolist()):
        detector.update(int(error))
        if detector.drift_detected:
            alarms += 1
            if on_alarm is not None:
                on_alarm(index)
    return alarms


def run(
    root,
    dataset,
    scope,
    policy,
    output,
    tracking_settings,
    *,
    tracker=None,
    iterative_config=None,
    stop_requested=None,
    progress=None,
):
    """Evaluate the stream and optionally report live metrics after completed blocks.

    The callback step counts rows visited across matrix construction (if used)
    and evaluation. The evaluation's ``rows_processed`` excludes the matrix pass.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown Zia policy: {policy}")
    started = time.monotonic()
    tracking_seconds = 0.0

    def publish_progress(step, metrics):
        nonlocal tracking_seconds
        mark = time.monotonic()
        progress(step, metrics)
        tracking_seconds += time.monotonic() - mark

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path, manifest = load_manifest(root, dataset, scope)
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError("prepared dataset checksum differs from its manifest")
    if progress is not None:
        publish_progress(0, {"rows_processed": 0, "rows_total": manifest["rows"], "evaluated": 0})
    spec = DATASETS[dataset]
    classes = len(spec["classes"])
    iter_config = IterativeTrainingConfig.from_params(iterative_config)
    costs = profile(tracking_settings["costs"])
    threshold = decision_threshold(costs) if classes == 2 else None
    model = make_zia_model(spec["model"], spec["classes"], config=iter_config, dataset=dataset,
                           decision_threshold=threshold)
    history = TrainingHistory(output, stop_requested)
    monitoring = SystemMetricsSampler(output, interval=0.5 if scope == "smoke" else 5)
    replay = deque(maxlen=REPLAY_BLOCKS)
    detector = None
    policy_settings = {}
    block_rows = BLOCK_ROWS[scope]
    batch_policy = None
    retraining_options = zia_retraining.settings(tracking_settings) if policy in BATCH_POLICY_KINDS else None
    if policy == "online":
        policy_settings = model.online_params
    elif policy in ("drift", "drift_adwin"):
        params = {"delta": ADWIN_DELTA, "clock": 1 if policy == "drift" else ADWIN_CLOCK}
        detector = ADWIN(**params)
        policy_settings = {"detector": "adwin", "params": params,
                           "signal": "block_error_rate" if policy == "drift" else "binary_error_per_example"}
    elif policy == "drift_ddm":
        detector = DDM(**DDM_PARAMS)
        policy_settings = {"detector": "ddm", "params": DDM_PARAMS, "signal": "binary_error_per_example"}
    elif retraining_options is not None:
        alpha = tracking_settings["upf_alpha"]
        params = {"alpha": alpha}
        if policy == "upf":
            params.update(delta=retraining_options["delta"], samples=retraining_options["mc_samples"],
                          seed=retraining_options["seed"])
        batch_policy = PolicyStrategy(policy, params=params)
        policy_settings = {**params, "replay_blocks": REPLAY_BLOCKS, "objective": "error_rate_0_1",
                           "offline_blocks": retraining_options["offline_blocks"],
                           "timing": "retrain after block t serves t+1 onward"}
    matrix = states = None

    def scores():
        return ClassificationMetrics(classes, threshold)

    online, holdout = scores(), scores()
    windows, versions, alarm_events = [], [], []
    observed_windows = []
    probes = TemporalHoldoutProbes()
    class_counts = np.zeros(classes, dtype=np.int64)
    train_seconds = predict_seconds = 0.0
    resets = rows = stream_blocks = 0
    last_timestamp = None
    warmup = 0
    split = manifest["rows"] - int(manifest["rows"] * HOLDOUT_FRACTION)
    candidate_dir = TemporaryDirectory(prefix="zia-candidates-") if batch_policy else nullcontext()
    progress_offset = 0
    with monitoring, candidate_dir as scratch, PredictionArchive(
        output, spec["classes"], manifest["sha256"], "native", tracking_settings["fading_factor"],
        threshold=threshold,
    ) as predictions:
        if batch_policy is not None:
            matrix, states = zia_retraining.stream_matrix(
                ((x, y, ts) for ts, x, y, held_out in batches(path, block_rows, split) if not held_out),
                make_zia_model(spec["model"], spec["classes"], config=iter_config, dataset=dataset,
                               decision_threshold=threshold),
                threshold, iter_config, REPLAY_BLOCKS, retraining_options,
                history=history, scratch=Path(scratch), tracker=tracker,
                progress=publish_progress if progress is not None else None,
            )
            # Matrix construction reads the training prefix before the evaluation pass.
            progress_offset = int(matrix.rows.sum())
            batch_policy.prepare(matrix)
            policy_settings["offline_blocks"] = matrix.offline
        for step, (timestamps, x, y, is_holdout) in enumerate(batches(path, block_rows, split)):
            history.check_interrupted()
            if ((last_timestamp is not None and timestamps[0] < last_timestamp)
                    or np.any(timestamps[1:] < timestamps[:-1]) or np.any((y < 0) | (y >= classes))):
                raise ValueError("unordered timestamps or invalid labels")
            last_timestamp = int(timestamps[-1])
            start, rows = rows, rows + len(y)
            class_counts += np.bincount(y, minlength=classes)
            action = "initial" if step == 0 else None
            chosen, phase = (states[0] if step == 0 and states is not None else None), None
            if not step:
                if is_holdout:
                    raise ValueError("no warm-up rows strictly before the temporal holdout")
                warmup = len(y)
            if step:
                timer = time.monotonic()
                probabilities = model.predict(x)
                elapsed = time.monotonic() - timer
                predict_seconds += elapsed
                window = scores()
                window.add(y, probabilities)
                predictions.add(start, timestamps, y, probabilities, is_holdout, versions[-1]["end_seq"])
                (holdout if is_holdout else online).merge(window)
                metrics, _ = window.metrics()
                windows.append({
                    "end_seq": rows, "scope": "holdout" if is_holdout else "prequential",
                    "start_time_ms": int(timestamps[0]), "end_time_ms": last_timestamp,
                    "prediction_seconds": elapsed, "prediction_throughput_per_second": len(y) / elapsed,
                    **metrics,
                })
                mark = time.monotonic()
                serving = versions[-1]
                observed_windows.append({
                    "step": rows, "start_seq": start, "scope": windows[-1]["scope"],
                    "quality": window.quality(spec["classes"]), "model_version": len(versions) - 1,
                    "fit_id": serving["fit_id"], "checkpoint": serving["sha256"],
                    "inference_seconds": elapsed, "start_time_ms": int(timestamps[0]),
                    "end_time_ms": last_timestamp,
                })
                if is_holdout:
                    probes.add(start, timestamps, x, y)
                tracking_seconds += time.monotonic() - mark
                if not is_holdout:
                    stream_blocks += 1
                    if policy == "periodic" or detector is not None:
                        replay.append((start, x, y, timestamps))
                    block_alarms = 0
                    if detector is not None:
                        alarm_offsets = []
                        block_alarms = drift_alarms(
                            detector, policy, probabilities, y, metrics, threshold,
                            on_alarm=alarm_offsets.append,
                        )
                        alarm_events.extend({
                            "seq": start + offset + 1, "block_index": step, "block_end_seq": rows,
                            "timestamp_ms": int(timestamps[offset]), "detector": policy_settings["detector"],
                            "signal": policy_settings["signal"],
                        } for offset in alarm_offsets)
                    if policy == "online":
                        action = "online"
                    elif policy == "incremental":
                        action = "continue"
                    elif (policy == "periodic" and stream_blocks % REPLAY_BLOCKS == 0) or block_alarms:
                        action = "reset_retrain"
                    elif batch_policy is not None:
                        if step == matrix.offline - 1:
                            action, chosen, phase = "reset_retrain", states[step], "offline_final"
                        elif matrix.offline <= step <= matrix.batches - 2:
                            decision = batch_policy.decide(DecisionContext(
                                labeled=step, window_size=len(y), drift=False, batch=step))
                            if decision.action == "reset_retrain":
                                action, chosen, phase = decision.action, states[step], "online"
            if action:
                timer = time.monotonic()
                if chosen is not None:
                    # Pre-trained for the matrix with this exact window; no second fit.
                    fit = chosen["fit"]
                    payload = chosen["path"].read_bytes()
                    extension = chosen["path"].suffix[1:]
                    model.load(payload)
                else:
                    if action == "reset_retrain":
                        train_start = replay[0][0]
                        fit_x = np.concatenate([item[1] for item in replay])
                        fit_y = np.concatenate([item[2] for item in replay])
                        fit_timestamps = np.concatenate([item[3] for item in replay])
                    else:
                        train_start, fit_x, fit_y, fit_timestamps = start, x, y, timestamps
                    if action == "online":
                        model.fit_online(fit_x, fit_y, tracker=tracker, stop_requested=stop_requested)
                    else:
                        model.fit(
                            fit_x, fit_y, action == "continue", config=iter_config, tracker=tracker,
                            timestamps=fit_timestamps, capture_sigterm=stop_requested is None,
                            stop_requested=stop_requested,
                        )
                    fit = history.capture(
                        model.training_summary, action=action, block_index=step, start=train_start,
                        timestamps=fit_timestamps, seconds=time.monotonic() - timer,
                        current_block_start=start, source_model_version=(
                            len(versions) - 1 if action in ("continue", "online") else None
                        ),
                    )
                    payload, extension = model.payload()
                    del fit_x, fit_y, fit_timestamps
                if fit["end_seq"] != rows:
                    raise ValueError("checkpoint fit window differs from its deployment block")
                resets += action == "reset_retrain"
                train_seconds += fit["train_seconds"]
                filename = f"models/{len(versions):04d}.{extension}"
                atomic_write(output / filename, payload)
                fit["deployed_version"] = len(versions)
                fit["deployment_action"] = action
                fit["promotion_status"] = "promoted"
                fit["replaces_model_version"] = len(versions) - 1 if versions else None
                fit["checkpoint"] = {
                    "file": filename, "sha256": sha256(payload).hexdigest(), "bytes": len(payload),
                }
                history.save_partial(fit)
                versions.append({
                    "end_seq": rows, "action": action, "start_seq": fit["start_seq"], "fit_id": fit["fit_id"],
                    **fit["checkpoint"],
                    "train_samples": fit["train_samples"], "validation_samples": fit["validation_samples"],
                    "training_presentations": fit["training_presentations"],
                    "train_seconds": fit["train_seconds"],
                    "iterations": fit["summary"]["iterations"],
                    "iteration_unit": fit["steps"][0]["kind"],
                    **({"phase": phase} if phase else {}),
                })
                if action == "reset_retrain" and chosen is None:
                    gc.collect()
            if step % 10 == 0:
                atomic_write(output / "progress.json", encoded({
                    "rows": rows, "total_rows": manifest["rows"], "model_versions_count": len(versions),
                    "elapsed_seconds": time.monotonic() - started,
                }))
            if progress is not None:
                mark = time.monotonic()
                live = {
                    "rows_processed": rows, "rows_total": manifest["rows"], "evaluated": rows - warmup,
                    "updates": len(versions) - 1, "retrain_count": resets,
                    "train_seconds": train_seconds, "prediction_seconds": predict_seconds,
                }
                if detector is not None:
                    live["drift_alarms"] = len(alarm_events)
                if step:
                    current = holdout if is_holdout else online
                    cumulative, _ = current.metrics()
                    scope_name = windows[-1]["scope"]
                    for key in ("n", "accuracy", "macro_f1", "log_loss", "average_precision"):
                        if cumulative.get(key) is not None:
                            live[f"{scope_name}/{key}"] = cumulative[key]
                tracking_seconds += time.monotonic() - mark
                publish_progress(progress_offset + rows, live)
    history.check_interrupted()
    if rows != manifest["rows"] or not online.confusion.sum() or not holdout.confusion.sum():
        raise ValueError("dataset coverage or temporal split is incomplete")
    retraining, retraining_metrics = None, {}
    if batch_policy is not None:
        zia_retraining.verify_realised(matrix, windows, batch_policy)
        retraining, retraining_metrics = zia_retraining.finalize(
            batch_policy, matrix, states, retraining_options)
        for fit in history.fits:
            if fit["deployed_version"] is None:
                fit["promotion_status"] = "rejected"
                fit["checkpoint_unavailable_reason"] = "matrix_candidate_checkpoint_is_transient"
    summary, by_class = online.metrics()
    final, final_by_class = holdout.metrics()
    errors = int(online.confusion.sum() - online.confusion.trace())
    updates = len(versions) - 1
    alarms = len(alarm_events)
    deployed_fits = [fit for fit in history.fits if fit["deployed_version"] is not None]
    presentations = sum(fit["training_presentations"] for fit in deployed_fits)
    after_warmup = sum(fit["training_presentations"] for fit in deployed_fits[1:])
    unique = unique_train_rows(deployed_fits)
    if batch_policy is not None:
        retraining_metrics["retraining_matrix_training_presentations"] = sum(
            fit["training_presentations"] for fit in history.fits
        )
    model_bytes = sum(v["bytes"] for v in versions)
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
    wall_cost = (
        aws_account(
            costs["aws"], train_seconds=compute_seconds, predict_seconds=0, model_storage_bytes=model_bytes,
        )
        if costs["aws"] is not None else None
    )
    summary.update(
        dataset_rows=rows, initial_train_rows=warmup, holdout_rows=final["n"],
        training_presentations=presentations, training_used_post_warmup=after_warmup,
        training_unique_examples=unique, training_repeated_presentations=presentations - unique,
        update_count=updates, retrain_count=resets, alarm_count=alarms,
        updates_per_1000_evaluated=1000 * updates / summary["n"],
        train_seconds=train_seconds, initial_train_seconds=versions[0]["train_seconds"],
        update_train_seconds=train_seconds - versions[0]["train_seconds"],
        postwarmup_train_samples_per_second=ratio(after_warmup,
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
        **retraining_metrics,
        **cost_report["metrics"],
        **{"aws_" + k: v for k, v in cost_report.get("aws", {}).get("metrics", {}).items()},
        **{"aws_evaluation_" + k: v for k, v in (wall_cost or {}).get("metrics", {}).items()},
    )
    error_sum = sum(w["error_rate"] for w in windows if w["scope"] == "prequential")
    alpha = tracking_settings["upf_alpha"]
    summary["upf_total_cost"] = (
        alpha * resets + error_sum if policy not in ("incremental", "online") else None
    )
    summary["posthoc_update_event_cost"] = alpha * updates + error_sum
    sensitivity = [[a, a * updates + error_sum] for a in tracking_settings["upf_alphas"]]
    summary["cost_sensitivity_auc"] = sum((b - a) * (v + w) / 2
                                          for (a, v), (b, w) in zip(sensitivity, sensitivity[1:]))
    availability = {
        name: {"status": "unavailable", "reason": "undefined_or_requires_other_evaluation_evidence"}
        for name in metric_catalog()[0]["definitions"]
        if summary.get(name) is None
    }
    decision_rule = (f"binary: p>={threshold:.6g} (c_FP/(c_FP+c_FN))" if classes == 2
                     else "multiclass: argmax, ties favor lower class ID")
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
                     "adwin_delta": ADWIN_DELTA, "score_bins": SCORE_BINS, "policy_settings": policy_settings,
                     "iterative_training": iter_config.model_dump(),
                     "decision_threshold": threshold, "decision_rule": decision_rule},
        "class_counts": class_counts.tolist(),
        "metrics": summary, "holdout": final, "windows": windows,
        "prequential_confusion": online.confusion.tolist(), "holdout_confusion": holdout.confusion.tolist(),
        "class_metrics": dict(zip(spec["classes"], by_class, strict=True)),
        "holdout_class_metrics": dict(zip(spec["classes"], final_by_class, strict=True)),
        "model_versions": versions, "model_explanation": model.explanation(manifest["features"]),
        "training_history": history.fits, "alarms": alarm_events,
        "costs": cost_report, "evaluation_aws_cost": wall_cost, "cost_sensitivity": sensitivity,
        "availability": availability, "system_metrics": system,
        **({"retraining": retraining} if retraining is not None else {}),
    }
    mark = time.monotonic()
    prequential_observation = online.quality(spec["classes"], curves=True)
    holdout_observation = holdout.quality(spec["classes"], curves=True)
    retention = probes.evaluate(model, output, versions, threshold)
    tracking_seconds += time.monotonic() - mark
    report["tracking_observations"] = run_observations(
        report=report, history=history, windows=observed_windows,
        prequential=prequential_observation, holdout=holdout_observation, retention=retention,
        has_detector=detector is not None, overhead=tracking_seconds, started=started,
    )
    history.check_interrupted()
    history.save()
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
    with run_lock(args.output.parent, args.output.name):
        report = _run_cli(args, settings)
    print(json.dumps(report["metrics"], indent=2))


def _run_cli(args, settings):
    if args.campaign:
        from mlflow.tracking import MlflowClient

        from continual_bench.evaluation.zia_publication import publish
        from continual_bench.infrastructure.zia_tracking import begin

        client = MlflowClient()
        run_id, experiment_id = begin(
            client, args.output, args.dataset, args.scope, args.policy, args.campaign,
        )
        try:
            report_path = args.output / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                _, manifest = load_manifest(args.root, args.dataset, args.scope)
                if report["contract"]["input"] != manifest or report["policy"] != args.policy:
                    raise ValueError("existing report does not match the requested Zia run")
            else:
                report = run(
                    args.root, args.dataset, args.scope, args.policy, args.output, settings,
                )
            publish(client, args.output, run_id, experiment_id, report, args.root)
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise
    else:
        from continual_bench.evaluation.zia_artifacts import tracking_report, write_zia_artifacts

        report = run(args.root, args.dataset, args.scope, args.policy, args.output, settings)
        evidence = prediction_manifest(args.output, report)
        write_zia_artifacts(args.output, report, tracking_report(report, evidence))
    return report


if __name__ == "__main__":
    main()
