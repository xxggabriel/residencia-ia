"""Explicit repeated training passes over a bounded dataset, with held-out scoring.

Every catalog dataset can run with every model family supporting its target type.
"""

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from continual_bench.config import EpochConfig
from continual_bench.core.tasks import MULTICLASS, REGRESSION, task_kind
from continual_bench.datasets.catalog import DATASETS, load_dataset
from continual_bench.infrastructure.storage import atomic_write, run_lock
from continual_bench.models import MODELS

from .artifacts import epoch_figures, system_figures, write_artifacts
from .costs import account
from .environment import dataset_fingerprint, encoded, environment, peak_rss_bytes
from .scoring import metric_set
from .system_metrics import SystemMetricsSampler
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
               models=MODELS):
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
    if model_factory is None:
        model_factory, _ = resolve_model(task, config.model, models=models)
    validate_records(records, task)
    train, test, split = split_records(records, manifest)
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
    with run_lock(root, key):
        report_path = directory / "epoch-report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["contract"] != contract:
                raise ValueError("completed epoch contract changed")
            if (directory / "samples.jsonl").read_bytes() != payload:
                raise ValueError("completed epoch sample artifact checksum changed")
            return report
        costs = config.costs.model_dump()
        with SystemMetricsSampler(root) as sampler:
            epochs, totals = _train_epochs(config, model_factory(), task, train, test, costs, started)
        report = {"version": "semana09-epochs-v1", "run_key": key, "complete": True,
                  "dataset": config.dataset, "unique_examples": len(records), "epochs": epochs,
                  "train_ids": [r.sample_id for r in train], "test_ids": [r.sample_id for r in test],
                  "contract": contract, "seconds": time.monotonic() - started,
                  "peak_rss_bytes_process": peak_rss_bytes(),
                  "scope": "train-only passes; frozen holdout after each epoch, no prequential claim"}
        report["costs"] = account(costs, errors=totals["errors"],
                                  presentations=len(train) * config.epochs, updates=config.epochs,
                                  train_seconds=totals["train_seconds"],
                                  predict_seconds=totals["predict_seconds"])
        report["system_metrics"] = sampler.report()
        report["artifacts"] = write_artifacts(
            directory, epoch_figures(report) + system_figures(report["system_metrics"]),
            {"summary/epochs": epochs, "costs/scenario": report["costs"],
             "system/samples": report["system_metrics"]})
        atomic_write(directory / "samples.jsonl", payload)
        atomic_write(report_path, encoded(report))
        return report


def _train_epochs(config, model, task, train, test, costs, started):
    """Train-only passes, each followed by scoring a frozen copy on the holdout."""
    kind = task_kind(task)
    epochs = []
    totals = {"errors": 0.0, "train_seconds": 0.0, "predict_seconds": 0.0}
    for epoch in range(config.epochs):
        before = time.perf_counter()
        for row in train:
            if time.monotonic() - started > config.max_seconds:
                raise TimeoutError("epoch benchmark deadline exceeded")
            model.learn_one(copy.deepcopy(row.x), row.y)
        training_seconds = time.perf_counter() - before
        frozen = copy.deepcopy(model)
        scores = metric_set(task)
        predictions = []
        prediction_seconds = errors = 0.0
        for row in test:
            if time.monotonic() - started > config.max_seconds:
                raise TimeoutError("epoch evaluation deadline exceeded")
            before_prediction = time.perf_counter()
            prediction = frozen.predict_one(copy.deepcopy(row.x))
            prediction_seconds += time.perf_counter() - before_prediction
            prediction = kind.normalize_prediction(prediction)
            errors += kind.error_units(row.y, prediction, costs)
            for metric in scores.values():
                metric.update(row.y, prediction)
            predictions.append({"sample_id": row.sample_id, "prediction": prediction})
        totals["errors"] += errors
        totals["train_seconds"] += training_seconds
        totals["predict_seconds"] += prediction_seconds
        epochs.append({"epoch": epoch + 1, "train_presentations": len(train), "test_examples": len(test),
                       "training_seconds": training_seconds,
                       "prediction_seconds": prediction_seconds,
                       "costs": account(costs, errors=errors, presentations=len(train), updates=1,
                                        train_seconds=training_seconds,
                                        predict_seconds=prediction_seconds),
                       "metrics": {name: metric.get() for name, metric in scores.items()},
                       "prediction_digest": hashlib.sha256(encoded(predictions)).hexdigest()})
    return epochs, totals


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
