"""Translate streaming training actions into a history based on observed updates."""

import hashlib
import json
from bisect import bisect_right
from itertools import groupby, islice
from pathlib import Path


def _retained_versions(report, directory):
    if directory is None:
        return ()
    path = directory / "model-versions.json"
    if not path.is_file():
        return ()
    data = path.read_bytes()
    metadata = report.get("model_versions") or {}
    index_hash = metadata.get("index_sha256") if isinstance(metadata, dict) else None
    if index_hash and hashlib.sha256(data).hexdigest() != index_hash:
        raise ValueError("retained model index differs from the report")
    return json.loads(data)["versions"]


def _bind_checkpoints(versions, retained_fits, directory):
    for entry in versions:
        if "training" in entry:
            training = entry["training"]
            action, bounds = training["action"], training["seq_range"]
            file = "models.bin"
            # Use the version's own coordinate. A promotion gate supplies its range before validation.
            coordinate = (action, tuple(bounds))
        else:
            file = entry["file"]
            coordinate = (entry["action"], entry["seq"])
        fit = retained_fits.get(coordinate)
        if fit is None:
            continue
        if file == "models.bin":
            checkpoint = {"sha256": entry["state_sha256"], "bytes": entry["size"], "offset": entry["offset"]}
        else:
            checkpoint = {key: entry[key] for key in ("sha256", "bytes")}
        checkpoint["file"] = file if directory is None or (directory / file).is_file() else None
        fit.update(checkpoint=checkpoint, deployed_version=entry["ordinal"])


def _typed_bounds(report, directory):
    if directory is None or not (path := directory / "samples.jsonl").is_file():
        return ()
    hashed = hashlib.sha256()
    train = []
    with path.open("rb") as source:
        for line in source:
            hashed.update(line)
            row = json.loads(line)
            if row["split"] == "train":
                train.append(row["seq"])
    digest = report.get("config", {}).get("samples_sha256")
    if digest and hashed.hexdigest() != digest:
        raise ValueError("retained typed samples differ from the report")
    return train


def _typed_training(report, directory):
    config = report.get("config", {})
    training = report.get("training", {})
    warmup = training.get("warmup_samples")
    initial = training.get("initial_fit")
    updates = report.get("updates", ())
    need_bounds = (not initial and warmup) or any("train_seq_range" not in row for row in updates)
    train_seqs = _typed_bounds(report, directory) if need_bounds else ()
    if not initial and warmup:
        bounds = [train_seqs[0], train_seqs[warmup - 1] + 1] if len(train_seqs) >= warmup else [None, None]
        initial = {
            "train_seq_range": bounds,
            "seq": bounds[1] - 1 if bounds[1] is not None else None,
            "train_samples": warmup,
        }

    def rows():
        for row in updates:
            bounds = row.get("train_seq_range")
            if bounds is None:
                end = bisect_right(train_seqs, row["seq"])
                count = row["train_samples"]
                if row["action"] == "learn_one":
                    bounds = [row["seq"], row["seq"] + 1]
                else:
                    start = train_seqs[end - count] if end >= count else None
                    bounds = [start, row["seq"] + 1]
            yield {**row, "train_seq_range": bounds}

    return initial, rows(), config.get("window", 128)


def _binary_training(report, versions):
    config = report.get("config", {})
    bounds = report.get("split", {}).get("train")
    warmup = config.get("warmup")
    if bounds is None and warmup and report.get("labeled", 0) >= warmup:
        bounds = [0, warmup]
    if bounds is None:
        bounds = next(
            (
                version["training"]["seq_range"]
                for version in versions
                if version.get("training", {}).get("action") == "initial_fit"
            ),
            None,
        )
    initial = None
    if bounds and bounds[1] > bounds[0]:
        initial = {
            "train_seq_range": bounds,
            "seq": bounds[1] - 1,
            "train_samples": bounds[1] - bounds[0],
            "seconds": (report.get("tracking") or {}).get("metrics", {}).get("initial_train_seconds"),
        }

    def rows():
        for row in report.get("candidates", report.get("updates", ())):
            validation_samples = row.get("validation_samples", 0)
            end = row["seq"] + 1 - validation_samples
            bounds = row.get("train_seq_range", (end - row["train_samples"], end))
            yield {**row, "train_seq_range": bounds}

    return initial, rows(), config.get("metric_window", 128)


def _sum_known(values):
    return sum(values) if all(value is not None for value in values) else None


def stream_training_history(report, directory=None, *, versions=None, captured=None, quality=None):
    """Build fit groups from recorded learning. Online points summarize bounded chunks of actual updates."""
    typed = report.get("schema_version") == "task_evaluation-typed-v1"
    if not typed and not ("updates" in report and ("config" in report or "split" in report)):
        return []
    directory = Path(directory) if directory is not None else None
    if versions is None:
        versions = _retained_versions(report, directory)
    if typed:
        initial, updates, width = _typed_training(report, directory)
    else:
        initial, updates, width = _binary_training(report, versions)
    fits = []
    retained_fits = {}
    global_step = 0
    seen_training_seqs = set()
    retained = {fit["fit_id"]: fit for fit in report.get("training_history", ())}

    def record(action, rows):
        nonlocal global_step
        grouped = action == "learn_one"
        fit = {
            "fit_id": f"fit-{len(fits) + 1:06d}",
            "action": action,
            "status": "complete",
            "block_index": None,
            "validation_samples": 0,
            "train_samples": 0,
            "steps": [],
        }
        last = None
        duration = 0.0
        actions = 0
        actual = []
        while chunk := list(islice(rows, max(1, width))):
            global_step += 1
            first, last = chunk[0], chunk[-1]
            count = sum(row["train_samples"] for row in chunk)
            seconds = _sum_known([row.get("seconds") for row in chunk])
            duration = _sum_known([duration, seconds])
            actions += len(chunk)
            metrics = {
                "training_presentations": count,
                "learning_calls": count,
                "training_seconds": seconds,
                "completed_training_actions": len(chunk),
            }
            if first["train_seq_range"][0] is not None:
                metrics["source_start_seq"] = first["train_seq_range"][0]
            if last["train_seq_range"][1] is not None:
                metrics["source_end_seq_exclusive"] = last["train_seq_range"][1]
            if not grouped and "candidate_accuracy" in last:
                metrics["promotion/candidate/accuracy"] = last["candidate_accuracy"]
                for key, name in (
                    ("incumbent_accuracy", "incumbent/accuracy"),
                    ("accuracy_gain", "accuracy_gain"),
                    ("validation_seconds", "validation_seconds"),
                ):
                    if key in last:
                        metrics[f"promotion/{name}"] = last[key]
                metrics["promotion/accepted"] = int(last.get("promoted", True))
            if not fit["steps"]:
                fit["start_seq"] = first["train_seq_range"][0]
            fit["end_seq"] = last["train_seq_range"][1]
            fit["train_samples"] += count
            fit["validation_samples"] += sum(row.get("validation_samples", 0) for row in chunk)
            fit["steps"].append(
                {
                    "step": len(fit["steps"]) + 1,
                    "global_step": global_step,
                    "kind": "update",
                    "metrics": metrics,
                }
            )
            if captured is not None:
                if action == "initial_fit":
                    actual.extend(captured["initial"])
                else:
                    actual.extend(captured["updates"][str(row["seq"])] for row in chunk)
        if last is None:
            return
        coordinate = (action, last["seq"] if typed else tuple(last["train_seq_range"]))
        promoted = last.get("promoted", True)
        if promoted:
            retained_fits[coordinate] = fit
        if "validation_seq_range" in last:
            fit["validation_seq_range"] = last["validation_seq_range"]
        fit.update(
            training_presentations=fit["train_samples"],
            train_seconds=duration,
            checkpoint=None,
            deployed_version=None,
            summary={
                "promoted": promoted,
                "training_seconds": duration,
                "learning_calls": fit["train_samples"],
                "completed_training_actions": actions,
                "source_coordinates": "half_open_training_span",
                "aggregation": "online_update_chunks" if grouped else "recorded_fit",
            },
        )
        if actual:
            _add_fit_observations(fit, action, actual, last, seen_training_seqs, quality)
        elif saved := retained.get(fit["fit_id"]):
            fit.update(
                {
                    key: value
                    for key, value in saved.items()
                    if key.startswith("observation_")
                    or key
                    in (
                        "observed_step_kind",
                        "promotion_status",
                        "source_model_version",
                        "prediction_model_versions",
                        "source_model_reason",
                        "fit_sequence",
                        "epoch_availability",
                        "native_loss_availability",
                    )
                }
            )
            if not promoted:
                fit["checkpoint"] = saved.get("checkpoint")
        fits.append(fit)

    if initial:
        record("initial_fit", iter((initial,)))
    for key, group in groupby(updates, key=lambda row: None if row["action"] == "learn_one" else row["seq"]):
        if key is None:
            record("learn_one", group)
        else:
            for row in group:
                record(row["action"], iter((row,)))
    _bind_checkpoints(versions, retained_fits, directory)
    if captured is not None:
        deployed = None
        version_ids = {entry["ordinal"]: entry.get("version_id", entry["ordinal"]) for entry in versions}
        for index, fit in enumerate(fits, 1):
            fit["fit_sequence"] = index
            fit["source_model_version"] = deployed if fit["action"] == "learn_one" else None
            fit["source_model_reason"] = None if fit["action"] == "learn_one" else "fresh_initialization"
            if (ordinal := fit["deployed_version"]) is not None:
                deployed = version_ids[ordinal]
    return fits


def _add_fit_observations(fit, action, actual, last, trained, quality):
    from .stream_observations import binary_summary_quality, single_prediction_metrics
    from .tracking_observations import quality_scope

    steps, predictions, prediction_versions = [], [], set()
    presentations = repeated = 0
    for index, row in enumerate(actual, 1):
        size = row["size"]
        if "repeated_presentations" in row:
            repeated += row["repeated_presentations"]
        else:
            if action == "reset_retrain" and "train_seqs" in last:
                source = last["train_seqs"]
            else:
                # Binary replay uses the contiguous suffix recorded by the engine and gate.
                end = last["train_seq_range"][1] if action == "reset_retrain" else row["seq"] + 1
                source = range(end - size, end)
            repeated += sum(seq in trained for seq in source)
            trained.update(source)
        presentations += size
        metrics = {
            "training/batch_size": size,
            "training/examples_processed": size,
            "training/presentations_cumulative": presentations,
            "training/replay_presentations_cumulative": repeated,
            "training/duration_seconds": row["seconds"],
            "training/native_loss": None,
        }
        prediction_version = None
        if prediction := row.get("prediction"):
            predictions.append(prediction)
            prediction_version = prediction.get("model_version")
            prediction_versions.add(prediction_version)
            metrics.update(single_prediction_metrics(prediction, quality))
        steps.append(
            {
                "step": index,
                "source_seq": row["seq"],
                "metrics": metrics,
                "prediction_model_version": prediction_version,
            }
        )
    scopes = {}
    if predictions:
        scopes["prequential_before_learning"] = quality_scope(
            quality.task,
            quality.collect(predictions),
        )
    if validation := actual[-1].get("validation"):
        scopes.update(
            {
                scope: quality_scope("binary", binary_summary_quality(values))
                for scope, values in validation.items()
            }
        )
    step_kind = "one_reset_and_replay_pass" if action == "reset_retrain" else "one_learner_call"
    fit.update(
        observation_steps=steps,
        observation_epochs=None,
        observed_step_kind=step_kind,
        epoch_availability={"status": "unavailable", "reason": "single_pass_learning_has_no_epochs"},
        native_loss_availability={"status": "unavailable", "reason": "learner_does_not_expose_loss"},
        promotion_status="promoted" if last.get("promoted", True) else "rejected",
        observation_effort={
            "steps": len(steps),
            "epochs": None,
            "unique_training_examples": presentations,
            "training_presentations": presentations,
            "replay_presentations": repeated,
        },
        observation_resources={"training_seconds": fit["train_seconds"]},
        observation_scopes=scopes,
        prediction_model_versions=sorted(prediction_versions - {None}, key=str),
    )
    if last.get("validation_seconds") is not None:
        fit["observation_resources"]["validation_seconds"] = last["validation_seconds"]
    if checkpoint := actual[-1].get("checkpoint"):
        fit["checkpoint"] = checkpoint
