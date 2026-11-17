"""Diagnostic evidence for shared candidates in the paper replication matrices."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from continual_bench.core.evaluation_metrics import task_matrix_metrics
from continual_bench.infrastructure.storage import atomic_write, run_lock

from .environment import encoded, environment, peak_rss_bytes
from .epoch_history import training_step, write_state
from .tracking_observations import batch_quality, observations


def sample_identity(splits):
    """Fingerprint each batch's ordered arrays without densifying sparse inputs."""
    from scipy.sparse import issparse

    digest = hashlib.sha256()
    for groups in splits:
        for values in groups:
            arrays = (values.data, values.indices, values.indptr) if issparse(values) else (values,)
            digest.update(encoded(list(values.shape)))
            for array in arrays:
                array = np.asarray(array)
                digest.update(encoded([array.dtype.str, list(array.shape)]))
                if array.dtype.hasobject:
                    digest.update(encoded(array.tolist()))
                else:
                    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def reserved_retention(accuracy, checkpoint_ids):
    """Summarize repeated scores on the same per-period held-out examples."""
    series = []
    for index in range(len(accuracy)):
        result = task_matrix_metrics(accuracy[:index + 1])["metrics"]
        series.append({
            "step": index + 1, "absolute_retention": result["past_task_retention_accuracy"],
            "gem_bwt": result["gem_backward_transfer"], "post_learning_fm": result["postlearning_forgetting"],
        })
    return {
        "matrix": accuracy, "checkpoints": checkpoint_ids,
        "periods": [f"lote {index + 1}" for index in range(len(accuracy))],
        "series": series, "metric": "accuracy", "scope": "matched_reserved_partition_by_batch",
        "checkpoint_lineage": "independent_refits",
        **{key: value for key, value in series[-1].items() if key != "step"},
    }


def candidate_report(
    name, seed, splits, states, learner, fit_seconds, *, protocol, initial=None,
    duration_before_tracking=0.0, drift=None, evaluation=None,
):
    """Reuse the protocol's train/holdout partitions. The decision matrix never consumes these scores."""
    if initial is not None and (type(initial) is not int or not 0 <= initial < len(splits)):
        raise ValueError("initial candidate index must point to a fitted batch")
    started = time.perf_counter()
    source_digest = sample_identity(splits)
    env = environment()
    detector = drift or {}
    contract = {
        "dataset": name, "seed": seed, "data_sha256": source_digest, "protocol": protocol,
        "model": learner.observation_model, "environment": env, "initial_candidate_index": initial,
        "decision_threshold": learner.observation_threshold,
        "evaluation": evaluation or {}, "detector_backend": detector.get("detector_backend"),
        "detector_evidence_sha256": hashlib.sha256(encoded({
            key: detector[key] for key in ("alarms", "coordinate") if key in detector
        })).hexdigest() if drift is not None else None,
    }
    run_key = hashlib.sha256(encoded(contract)).hexdigest()[:24]
    if "MLFLOW_TRACKING_URI" in os.environ:
        root = Path(os.environ.get("BENCH_STATE", "state")) / "retraining-replication"
        with run_lock(root, run_key):
            directory = root / run_key
            path = directory / "report.json"
            if path.exists():
                report = json.loads(path.read_bytes())
                if report.get("run_key") != run_key or report.get("contract") != contract:
                    raise ValueError("retained candidate report does not match its execution identity")
            else:
                report = _record_candidates(contract, run_key, splits, states, learner, fit_seconds,
                                            started, duration_before_tracking, initial, directory, drift)
                from .training_evidence import write_training_artifacts

                report["artifacts"] = write_training_artifacts(directory, report)
                atomic_write(path, encoded(report))
            from continual_bench.infrastructure.benchmark_tracking import publish_local_benchmark

            try:
                receipt = publish_local_benchmark(
                    directory, report, experiment_name="continual-bench/retraining-replication",
                )
            except Exception as error:  # The retained report supports retry after a publication failure.
                receipt = {"complete": False, "error": f"{type(error).__name__}: {str(error)[:500]}"}
            return {**report, "mlflow": receipt}
    return _record_candidates(contract, run_key, splits, states, learner, fit_seconds,
                              started, duration_before_tracking, initial, None, drift)


def _record_candidates(contract, run_key, splits, states, learner, fit_seconds, started, previous, initial,
                       directory, drift):
    classes = sorted({label for _, train, _, holdout in splits
                      for labels in (train, holdout) for label in np.unique(labels).tolist()})
    positive = next((value for value in (1, "1", "UP") if value in classes), None)
    if classes and set(classes).issubset({0, 1}):
        classes = [0, 1]
        positive = 1
    task = "binary" if len(classes) == 2 else "multiclass"
    accuracy = [[None for _ in splits] for _ in states]
    fits, windows = [], []
    offset = evaluated = global_step = 0
    predict_seconds = 0.0
    candidates = zip(splits, states, fit_seconds, strict=True)
    for index, ((_, y_train, _, _), state, elapsed) in enumerate(candidates):
        number = index + 1
        fit_id = f"fit-{number:06d}"
        fit_predict_seconds = 0.0
        fit_evaluated = 0
        for period, (_, _, x, y) in enumerate(splits[:number]):
            before = time.perf_counter()
            prediction, scores = learner.observation_predict(state, x, positive)
            fit_predict_seconds += time.perf_counter() - before
            fit_evaluated += len(y)
            accuracy[index][period] = float(np.mean(prediction == y))
        predict_seconds += fit_predict_seconds
        evaluated += fit_evaluated
        quality = batch_quality(
            y, prediction, classes=classes, positive_class=positive if task == "binary" else None,
            scores=scores, decision_threshold=learner.observation_threshold,
        )
        raw, extension = learner.observation_checkpoint(state)
        checkpoint = (write_state(directory, f"models/{fit_id}.{extension}", raw, format=extension)
                      if directory is not None else
                      {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "format": extension})
        durations = learner.observation_durations(index, elapsed)
        train_rows, step_count = len(y_train), len(durations)
        steps = [
            training_step(step, examples=train_rows, cumulative=step * train_rows, duration_seconds=seconds)
            for step, seconds in enumerate(durations, 1)
        ]
        effort = {
            "evaluated_examples": fit_evaluated, "unique_training_examples": train_rows,
            "training_presentations": step_count * train_rows, "steps": step_count,
            "epochs": None, "replay_presentations": 0,
        }
        resources = {
            "training_seconds": elapsed, "inference_seconds": fit_predict_seconds,
            "predictions_per_second": fit_evaluated / fit_predict_seconds if fit_predict_seconds else None,
        }
        promoted = index == initial
        fits.append({
            "fit_id": fit_id, "action": "initial" if promoted else "candidate", "status": "complete",
            "block_index": index, "start_seq": offset, "end_seq": offset + train_rows + len(y),
            "promotion_step": number,
            "data_scope": {"source_batch": index, "train_partition_rows": train_rows,
                           "holdout_partition_rows": len(y)},
            "train_samples": train_rows, "validation_samples": 0, "train_seconds": elapsed,
            "training_presentations": effort["training_presentations"], "checkpoint": checkpoint,
            "deployed_version": index if promoted else None,
            "promotion_status": "promoted" if promoted else "candidate",
            "steps": [
                {"step": step, "global_step": global_step + step, "kind": "update", "metrics": row["metrics"]}
                for step, row in enumerate(steps, 1)
            ],
            "summary": {
                "steps": step_count, "epochs_unavailable_reason": "native_batch_fit_has_no_epoch_axis",
            },
            "observation_steps": steps, "observation_epochs": None,
            "observed_step_kind": learner.observation_step_kind, "observation_effort": effort,
            "observation_resources": resources, "observation_scopes": {"batch_holdout": quality},
        })
        windows.append({"step": number, "scope": "batch_holdout", "coordinate": "candidate",
                        "quality": quality, "model_version": index, "fit_id": fit_id})
        offset += train_rows + len(y)
        global_step += step_count
    retention = reserved_retention(accuracy, [fit["fit_id"] for fit in fits])
    overhead = time.perf_counter() - started
    return {
        "run_key": run_key, "complete": True, "dataset": contract["dataset"], "contract": contract,
        "training_history": fits,
        "tracking_observations": observations(
            task, {}, windows=windows, retention=retention, drift=drift,
            effort={
                "examples_observed": offset, "evaluated_examples": evaluated,
                "unique_training_examples": sum(fit["train_samples"] for fit in fits),
                "training_presentations": sum(fit["training_presentations"] for fit in fits),
                "steps": global_step, "replay_presentations": 0,
                "promoted_versions": int(initial is not None),
            },
            resources={
                "duration_seconds": previous + overhead, "training_seconds": sum(fit_seconds),
                "inference_seconds": predict_seconds, "peak_rss_bytes": peak_rss_bytes(),
                "rss_scope": "process_lifetime_high_water_mark", "tracking_overhead_seconds": overhead,
                "predictions_per_second": evaluated / predict_seconds if predict_seconds else None,
            },
            lineage={
                **contract, "dataset_version": contract["data_sha256"], "policy": "shared_candidate_matrix",
                "code_revision": contract["environment"]["code_sha256"],
            },
        ),
    }
