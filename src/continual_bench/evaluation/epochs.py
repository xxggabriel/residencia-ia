"""Explicit repeated training passes over a bounded dataset, with held-out scoring.

Every catalog dataset can run with every model family supporting its target type.
"""

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

from continual_bench.config import EpochConfig
from continual_bench.core.tasks import BINARY, MULTICLASS, REGRESSION, task_kind
from continual_bench.datasets.catalog import DATASETS, load_dataset
from continual_bench.infrastructure.storage import atomic_write, run_lock
from continual_bench.infrastructure.training_tracking import HISTORY_FILE, training_checkpoint_paths
from continual_bench.models import MODELS

from .artifacts import epoch_figures, system_figures, write_artifacts
from .costs import account
from .environment import dataset_fingerprint, encoded, environment, peak_rss_bytes
from .epoch_history import epoch_fit, holdout_epoch_rows, pickle_state, training_step, write_state
from .scoring import metric_set
from .system_metrics import SystemMetricsSampler
from .tracking_observations import batch_quality, observations, quality_scope
from .training_evidence import write_training_artifacts
from .typed import resolve_model

__all__ = ["EpochConfig", "dataset_fingerprint", "run_epochs", "smoke_all", "split_records",
           "validate_records"]


def validate_records(records, task):
    kind = task_kind(task)
    for seq, row in enumerate(records):
        if (type(row.seq) is not int or row.seq != seq or not isinstance(row.sample_id, str)
                or not row.sample_id or type(row.timestamp) is not int or row.timestamp < 0
                or type(row.available_at) is not int or row.available_at < row.timestamp):
            raise ValueError("invalid sample identity, sequence, timestamp or label availability")
        if not row.x or any(not isinstance(k, str) for k in row.x):
            raise ValueError("sample features must be nonempty and string-keyed")
        for value in row.x.values():
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise ValueError("nonfinite feature")
            elif not isinstance(value, str) or not value:
                raise ValueError("unsupported feature value")
        kind.validate_target(row.y)


def split_records(records, manifest):
    """Official membership wins; unpartitioned streams use a purged temporal suffix."""
    if not records or len({row.sample_id for row in records}) != len(records):
        raise ValueError("epoch input must be nonempty and contain unique original sample IDs")
    if any(row.split not in {"train", "test", "validation"} for row in records):
        raise ValueError("unknown input split")
    official = any(row.split != "train" for row in records)
    if official:
        train = [row for row in records if row.split == "train"]
        test = [row for row in records if row.split == "test"]
        if not test:
            raise ValueError("official holdout requires a test partition")
        contract = {"kind": "official_holdout", "calendar_future_claim": False, "purged": 0}
    else:
        if any(a.seq >= b.seq or a.timestamp > b.timestamp for a, b in zip(records, records[1:])):
            raise ValueError("unpartitioned stream must preserve source order")
        boundary = int(len(records) * 0.8)
        test = records[boundary:]
        train = [row for row in records[:boundary] if row.available_at < test[0].timestamp]
        contract = {"kind": "purged_temporal_holdout", "cutoff": test[0].timestamp,
                    "purged": boundary - len(train),
                    "calendar_future_claim": manifest.get("timestamp_kind", "").startswith("natural")}
    if not train or not test or {r.sample_id for r in train} & {r.sample_id for r in test}:
        raise ValueError("training and test must be nonempty and disjoint")
    return train, test, contract


def run_epochs(config, root, cache, *, download=False, real_directory=None, source=None, model_factory=None,
               models=MODELS, progress=None, checkpoint=None):
    started = time.monotonic()
    source = source or load_dataset(config.dataset, cache, examples=config.examples, seed=config.seed,
                                    download=download, real_directory=real_directory)
    manifest = source.manifest()
    records = list(islice(source.samples_iter(), config.examples + 1))
    if len(records) != config.examples:
        raise ValueError(f"expected exactly {config.examples} unique examples, got {len(records)}")
    task = manifest["task"]
    if task not in {MULTICLASS, REGRESSION}:
        raise ValueError("unsupported epoch target type")
    custom_factory = model_factory is not None
    if custom_factory:
        model_identity = {"family": getattr(model_factory, "__qualname__", "custom"), "task": task}
    else:
        model_factory, model_identity = resolve_model(task, config.model, models=models)
    validate_records(records, task)
    train, test, split = split_records(records, manifest)
    binary_adapter = task == MULTICLASS and manifest.get("target_conversion") == (
        "explicit binary boolean to class strings, no information loss"
    )
    classes = None
    if task == MULTICLASS:
        # Keep the manifest order even when the selected prefix or the holdout omits a class.
        labels = manifest.get("class_order", manifest.get("labels", []))
        classes = list(dict.fromkeys([*(str(label) for label in labels), *sorted({r.y for r in records})]))
    payload = b"".join(encoded(asdict(row)) + b"\n" for row in records)
    if len(payload) > 128 * 1024**2:
        raise ValueError("selected epoch dataset exceeds 128 MiB")
    env = environment()
    contract = json.loads(encoded({"config": config.model_dump(), "source": manifest,
                                   "samples_sha256": hashlib.sha256(payload).hexdigest(),
                                   "runtime_fingerprint": dataset_fingerprint(env),
                                   "environment": env, "split": split}))
    key = hashlib.sha256(encoded(contract)).hexdigest()[:24]
    directory = Path(root) / key
    snapshot = checkpoint.load() if checkpoint is not None else None
    if snapshot is not None and snapshot["contract"] != contract:
        raise ValueError("epoch checkpoint contract changed")
    with run_lock(root, key):
        report_path = directory / "epoch-report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["contract"] != contract:
                raise ValueError("completed epoch contract changed")
            if (directory / "samples.jsonl").read_bytes() != payload:
                raise ValueError("completed epoch sample artifact checksum changed")
            if report.get("training_history"):
                expected = {"version": 1, "fits": report["training_history"]}
                if json.loads((directory / HISTORY_FILE).read_bytes()) != expected:
                    raise ValueError("completed epoch training history differs from the report")
                training_checkpoint_paths(directory, report)
            return report
        costs = config.costs.model_dump()
        state = snapshot["state"] if snapshot is not None else EpochState(model_factory())
        previous_duration = state.elapsed_seconds

        def save(*, force=False):
            if checkpoint is None or not (force or checkpoint.due()):
                return
            requested = checkpoint.requested()
            state.elapsed_seconds = previous_duration + time.monotonic() - started
            checkpoint.save({"contract": contract, "state": state}, force=force or requested)
            if requested:
                raise InterruptedError("benchmark interrupted at an epoch checkpoint")

        with SystemMetricsSampler(root) as sampler:
            epochs, totals = _train_epochs(
                config, state, task, train, test, costs, started, progress=progress, save=save,
                classes=classes, positive_class="1" if binary_adapter else None,
                target_unit=manifest.get("target_unit", "target_unit") if task == REGRESSION else None,
            )
        blob = pickle_state(state.model, optional=custom_factory)
        model_checkpoint = write_state(directory, "model.pkl", blob, format="pickle") if blob else None
        duration = previous_duration + time.monotonic() - started
        effort = {
            "examples_observed": len(records), "evaluated_examples": len(test) * len(epochs),
            "unique_evaluated_examples": len(test),
            "unique_training_examples": len(train), "training_presentations": totals["presentations"],
            "replay_presentations": 0, "steps": len(totals["steps"]), "epochs": len(epochs),
            "updates": 0, "initial_training_presentations": totals["presentations"],
            "initial_unique_training_examples": len(train), "initial_steps": len(totals["steps"]),
            "initial_epochs": len(epochs), "update_training_presentations": 0, "update_steps": 0,
            "update_epochs": 0, "promoted_versions": int(model_checkpoint is not None),
        }
        resources = {
            "duration_seconds": duration,
            "training_seconds": totals["learner_seconds"],
            "inference_seconds": totals["predict_seconds"],
            "predictions_per_second": effort["evaluated_examples"] / totals["predict_seconds"]
            if totals["predict_seconds"] else None,
            "peak_rss_bytes": peak_rss_bytes(),
            "rss_scope": "process_lifetime_high_water_mark",
            "tracking_overhead_seconds": totals["overhead_seconds"],
        }
        final_holdout = totals["observation_epochs"][-1]["scopes"]["holdout"]
        observation_task = BINARY if binary_adapter else task
        scopes = {
            "holdout": final_holdout,
            "train": quality_scope(observation_task, None),
            "validation": quality_scope(observation_task, None),
        }
        history = [epoch_fit(
            holdout_epoch_rows(epochs), action="initial", train_samples=len(train), validation_samples=0,
            train_seconds=totals["train_seconds"], checkpoint=model_checkpoint,
            deployed_version=0 if model_checkpoint else None,
            summary={"holdout_examples": len(test), "last_epoch": len(epochs),
                     "checkpoint_epoch": len(epochs) if model_checkpoint else None},
            observation_steps=totals["steps"], observation_epochs=totals["observation_epochs"],
            observed_step_kind="learn_one",
            promotion_status="promoted" if model_checkpoint else "unavailable",
            observation_effort=effort, observation_resources=resources,
            observation_scopes=scopes,
            data_scope={
                "train_ids": "epoch-report.json#train_ids", "holdout_ids": "epoch-report.json#test_ids",
            },
        )]
        report = {"version": "semana09-epochs-v1", "run_key": key, "complete": True,
                  "dataset": config.dataset, "unique_examples": len(records), "epochs": epochs,
                  "train_ids": [r.sample_id for r in train], "test_ids": [r.sample_id for r in test],
                  "training_history": history,
                  "contract": contract, "seconds": duration,
                  "peak_rss_bytes_process": peak_rss_bytes(),
                  "scope": "train-only passes; frozen holdout after each epoch, no prequential claim"}
        report["costs"] = account(costs, errors=totals["errors"],
                                  presentations=len(train) * config.epochs, updates=config.epochs,
                                  train_seconds=totals["train_seconds"],
                                  predict_seconds=totals["predict_seconds"],
                                  model_storage_bytes=model_checkpoint["bytes"] if model_checkpoint else 0)
        cost_total = report["costs"]["metrics"]["cost_total"]
        resources.update({
            "cost_total": cost_total,
            "cost_per_1000_evaluated": 1000 * cost_total / effort["evaluated_examples"],
            "cost_currency": "scenario_cost_unit",
            "cost_kind": report["costs"]["kind"],
        })
        report["system_metrics"] = sampler.report()
        report["tracking_observations"] = observations(
            observation_task, scopes, effort=effort, resources=resources,
            windows=[
                {"step": row["epoch"], "scope": "holdout", "coordinate": "epoch",
                 "quality": row["scopes"]["holdout"]}
                for row in totals["observation_epochs"]
            ],
            lineage={
                "dataset": config.dataset,
                "dataset_version": manifest.get("version") or contract["samples_sha256"],
                "model": model_identity["family"], "model_identity": model_identity,
                "policy": "initial_only", "seed": config.seed, "environment": env,
                "checkpoint": model_checkpoint, "code_revision": env["code_sha256"],
                "protocol": split["kind"],
            },
        )
        report["artifacts"] = write_artifacts(
            directory, epoch_figures(report) + system_figures(report["system_metrics"]),
            {"summary/epochs": epochs, "costs/scenario": report["costs"],
             "system/samples": report["system_metrics"]})
        report["artifacts"] = write_training_artifacts(directory, report)
        atomic_write(directory / "samples.jsonl", payload)
        atomic_write(report_path, encoded(report))
        return report


@dataclass
class EpochPass:
    first_step: int
    scores: dict
    training_offset: int = 0
    frozen: object | None = None
    learner_seconds: float = 0.0
    training_seconds: float = 0.0
    prediction_seconds: float = 0.0
    errors: float = 0.0
    predictions: list = field(default_factory=list)


@dataclass
class EpochState:
    model: object
    epochs: list = field(default_factory=list)
    totals: dict = field(default_factory=lambda: {
        "errors": 0.0, "train_seconds": 0.0, "predict_seconds": 0.0, "learner_seconds": 0.0,
        "overhead_seconds": 0.0, "presentations": 0, "steps": [], "observation_epochs": [],
    })
    current: EpochPass | None = None
    elapsed_seconds: float = 0.0


def _train_epochs(config, state, task, train, test, costs, started, *, save, progress=None,
                  classes=None, positive_class=None, target_unit=None):
    """Train-only passes, each followed by scoring a frozen copy on the holdout."""
    kind = task_kind(task)
    model, epochs, totals = state.model, state.epochs, state.totals
    for epoch in range(len(epochs), config.epochs):
        if state.current is None:
            state.current = EpochPass(len(totals["steps"]) + 1, metric_set(task))
        current = state.current
        for row in islice(train, current.training_offset, None):
            if time.monotonic() - started > config.max_seconds:
                raise TimeoutError("epoch benchmark deadline exceeded")
            step_started = time.perf_counter()
            model.learn_one(dict(row.x), row.y)
            step_finished = time.perf_counter()
            duration = step_finished - step_started
            current.learner_seconds += duration
            totals["presentations"] += 1
            totals["steps"].append(training_step(
                len(totals["steps"]) + 1, examples=1, cumulative=totals["presentations"],
                duration_seconds=duration,
            ))
            totals["overhead_seconds"] += time.perf_counter() - step_finished
            current.training_seconds += time.perf_counter() - step_started
            current.training_offset += 1
            save()
        if current.frozen is None:
            current.frozen = copy.deepcopy(model)
        for row in islice(test, len(current.predictions), None):
            if time.monotonic() - started > config.max_seconds:
                raise TimeoutError("epoch evaluation deadline exceeded")
            before_prediction = time.perf_counter()
            prediction = current.frozen.predict_one(dict(row.x))
            current.prediction_seconds += time.perf_counter() - before_prediction
            prediction = kind.normalize_prediction(prediction)
            current.errors += kind.error_units(row.y, prediction, costs)
            for metric in current.scores.values():
                metric.update(row.y, prediction)
            current.predictions.append({"sample_id": row.sample_id, "prediction": prediction})
            save()
        tracking_started = time.perf_counter()
        ranking_scores = _ranking_scores(model, test, positive_class)
        quality = batch_quality(
            [row.y for row in test], [item["prediction"] for item in current.predictions],
            classes=classes, positive_class=positive_class, scores=ranking_scores, target_unit=target_unit,
        )
        quality = quality_scope(BINARY if positive_class is not None else task, quality)
        totals["observation_epochs"].append({
            "epoch": epoch + 1,
            "first_step": current.first_step,
            "last_step": len(totals["steps"]),
            "metrics": {
                "training/examples_processed": len(train),
                "training/duration_seconds": current.learner_seconds,
                "holdout/duration_seconds": current.prediction_seconds,
            },
            "scopes": {"holdout": quality},
        })
        totals["learner_seconds"] += current.learner_seconds
        totals["overhead_seconds"] += time.perf_counter() - tracking_started
        totals["errors"] += current.errors
        totals["train_seconds"] += current.training_seconds
        totals["predict_seconds"] += current.prediction_seconds
        epochs.append({"epoch": epoch + 1, "train_presentations": len(train), "test_examples": len(test),
                       "training_seconds": current.training_seconds,
                       "prediction_seconds": current.prediction_seconds,
                       "costs": account(costs, errors=current.errors, presentations=len(train), updates=1,
                                        train_seconds=current.training_seconds,
                                        predict_seconds=current.prediction_seconds, include_aws=False),
                       "metrics": {name: metric.get() for name, metric in current.scores.items()},
                       "prediction_digest": hashlib.sha256(encoded(current.predictions)).hexdigest()})
        state.current = None
        if progress is not None:
            progress(epoch + 1, epochs[-1]["metrics"])
        save(force=len(epochs) == config.epochs)
    return epochs, totals


def _ranking_scores(model, test, positive_class):
    """Read probabilities from a separate frozen copy without changing legacy predictions or timers."""
    if positive_class is None or not callable(getattr(model, "predict_proba_one", None)):
        return None
    predict = copy.deepcopy(model).predict_proba_one
    scores = []
    for row in test:
        try:
            distribution = predict(dict(row.x))
            if not distribution:
                return None
            score = float(distribution.get(positive_class, 0.0))
        except (ValueError, TypeError, AttributeError, NotImplementedError):
            return None
        if not math.isfinite(score) or not 0 <= score <= 1:
            return None
        scores.append(score)
    return scores


def smoke_all(root, cache, *, download=False, real_directory=None):
    """Fail closed: a missing source never becomes a synthetic replacement or a skip."""
    results = []
    receipt_path = Path(root) / "dataset-smoke-complete.json"
    receipt = {"version": "semana09-all-datasets-smoke-v1", "complete": False, "datasets": results}
    # A failed repeat must revoke an older admission before attempting any source.
    atomic_write(receipt_path, encoded(receipt))
    for dataset in DATASETS:
        report = run_epochs(EpochConfig(dataset=dataset), root, cache, download=download,
                            real_directory=real_directory)
        results.append({"dataset": dataset, "run_key": report["run_key"], "complete": report["complete"],
                        "unique_examples": report["unique_examples"], "epochs": len(report["epochs"]),
                        "runtime_fingerprint": report["contract"]["runtime_fingerprint"],
                        "code_sha256": report["contract"]["environment"]["code_sha256"]})
        atomic_write(Path(root) / "dataset-smoke-progress.json", encoded(results))
    receipt["complete"] = True
    atomic_write(receipt_path, encoded(receipt))
    return receipt
