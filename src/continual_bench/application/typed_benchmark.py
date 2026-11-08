"""Bounded native River evaluation for regression and prepared multiclass data."""

import hashlib
import json
import math
import resource
import sys
import time
from collections import deque
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from river import linear_model, metrics, optim, preprocessing, tree

from continual_bench.adapters.local import atomic_write, run_lock

from .costs import account, error_units, profile
from .runner import environment


class _Regression:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.model = linear_model.LinearRegression(optimizer=optim.SGD(0.01))

    def predict_one(self, x):
        return self.model.predict_one(self.scaler.transform_one(x))

    def learn_one(self, x, y):
        self.scaler.learn_one(x)
        self.model.learn_one(self.scaler.transform_one(x), y)


def _model(task):
    return (_Regression() if task == "regression" else
            tree.HoeffdingTreeClassifier(grace_period=25, max_depth=12, max_size=64))


def _metric_set(task):
    return ({"mae": metrics.MAE(), "rmse": metrics.RMSE()} if task == "regression" else
            {"accuracy": metrics.Accuracy(), "macro_f1": metrics.MacroF1()})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def run_typed(source, root, *, task=None, policy="online", warmup=20, every=50, window=128,
              max_samples=1000, max_seconds=60, training_budget=None, costs=None):
    """Evaluate a deterministic prefix; never learn validation/test labels.

    `source` exposes samples_iter() of TaskSample plus manifest(). Labels are
    released after prediction at the current timestamp, including at delay zero.
    Complete reruns are idempotent by config/data/code identity; no partial resume.
    """
    costs = profile(costs)
    manifest = source.manifest()
    task = task or manifest.get("task")
    if task not in {"regression", "multiclass"} or manifest.get("task") != task:
        raise ValueError("typed task must match the source's regression/multiclass declaration")
    if policy not in {"none", "online", "periodic"}:
        raise ValueError("typed evaluator supports none/online/periodic; cost/drift policies are unsupported")
    for name, value, minimum, maximum in (("warmup", warmup, 1, 10_000),
                                          ("every", every, 1, 10_000),
                                          ("window", window, 1, 10_000),
                                          ("max_samples", max_samples, 2, 10_000),
                                          ("max_seconds", max_seconds, 1, 600)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid {name} budget")
    if training_budget is not None and (type(training_budget) is not int or training_budget < 0):
        raise ValueError("training_budget must be a nonnegative integer")
    started = time.monotonic()
    records, seen, last_stamp, last_available = [], set(), -1, -1
    payload_bytes = 0
    for sample in islice(source.samples_iter(), max_samples):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("typed source deadline exceeded")
        if (sample.seq != len(records) or sample.timestamp < last_stamp
                or sample.available_at < sample.timestamp or sample.available_at < last_available):
            raise ValueError("typed samples require contiguous sequence and ordered availability")
        if sample.sample_id in seen:
            raise ValueError("duplicate sample identity across typed splits")
        if sample.split not in {"train", "validation", "test"}:
            raise ValueError("unknown typed split")
        if not sample.x or any(not isinstance(k, str) for k in sample.x):
            raise ValueError("typed features require nonempty string-key dictionaries")
        if task == "regression":
            if (type(sample.y) not in (int, float) or not math.isfinite(sample.y)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in sample.x.values())):
                raise ValueError("regression targets and features must be finite numbers")
        elif not isinstance(sample.y, str) or sample.y == "__untrained__":
            raise ValueError("multiclass targets must be strings other than reserved __untrained__")
        raw = _json(asdict(sample))
        payload_bytes += len(raw) + 1
        if payload_bytes > 64 * 1024**2:
            raise ValueError("typed materialization exceeds 64 MiB")
        records.append(sample)
        seen.add(sample.sample_id)
        last_stamp, last_available = sample.timestamp, sample.available_at
    if sum(s.split == "train" for s in records) <= warmup:
        raise ValueError("typed warmup must leave at least one training-stream evaluation sample")
    payload = b"".join(_json(asdict(sample)) + b"\n" for sample in records)
    env = environment()
    config = {"schema_version": "task_evaluation-typed-v1", "task": task, "policy": policy, "warmup": warmup,
              "every": every, "window": window, "max_samples": max_samples, "max_seconds": max_seconds,
              "training_budget": training_budget, "costs": costs, "source": manifest,
              "samples_sha256": hashlib.sha256(payload).hexdigest(), "environment": env,
              "model": ({"name": "river.LinearRegression", "scaler": "StandardScaler", "sgd": 0.01}
                        if task == "regression" else {"name": "river.HoeffdingTreeClassifier",
                            "grace_period": 25, "max_depth": 12, "max_size_mib": 64})}
    # Native tuple-valued descriptors must match the persisted JSON shape on replay.
    config = json.loads(_json(config))
    identity = hashlib.sha256(_json(config)).hexdigest()[:24]
    directory = Path(root) / identity
    with run_lock(root, identity):
        report_path = directory / "report.json"
        if report_path.exists():
            if (directory / "samples.jsonl").read_bytes() != payload:
                raise ValueError("typed materialized samples changed for a completed run")
            return json.loads(report_path.read_text())
        model = _model(task)
        metric_sets = {split: _metric_set(task) for split in ("train", "validation", "test")}
        counts = {split: 0 for split in metric_sets}
        per_task, predictions, updates = {}, [], []
        replay, pending = deque(maxlen=window), deque()
        train_arrived = released_train = training_used = budget_blocked = 0
        prediction_seconds = training_seconds = 0.0
        prediction_errors = 0.0

        def release(now):
            nonlocal model, released_train, training_used, training_seconds, budget_blocked, prediction_errors
            while pending and pending[0][0].available_at <= now:
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("typed evaluation deadline exceeded")
                sample, prediction, train_index = pending.popleft()
                scored = sample.split != "train" or train_index >= warmup
                if scored:
                    prediction_errors += error_units(task, sample.y, prediction, costs)
                    counts[sample.split] += 1
                    for metric in metric_sets[sample.split].values():
                        metric.update(sample.y, prediction)
                    if sample.task_id is not None:
                        key = f"{sample.split}:{sample.task_id}"
                        entry = per_task.setdefault(key, {"n": 0, "metrics": _metric_set(task)})
                        entry["n"] += 1
                        for metric in entry["metrics"].values():
                            metric.update(sample.y, prediction)
                if sample.split != "train":
                    continue
                released_train += 1
                replay.append(sample)
                initial = train_index < warmup
                reset = not initial and policy == "periodic" and (released_train - warmup) % every == 0
                learn = initial or policy == "online" or reset
                if not learn:
                    continue
                selected = list(replay) if reset else [sample]
                if (not initial and training_budget is not None
                        and training_used + len(selected) > training_budget):
                    budget_blocked += 1
                    continue
                before = time.perf_counter()
                if reset:
                    model = _model(task)
                for row in selected:
                    model.learn_one(dict(row.x), row.y)
                seconds = time.perf_counter() - before
                training_seconds += seconds
                if not initial:
                    training_used += len(selected)
                    updates.append({"seq": sample.seq, "released_at": now, "train_samples": len(selected),
                                    "action": "reset_retrain" if reset else "learn_one", "seconds": seconds})

        for sample in records:
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("typed evaluation deadline exceeded")
            before = time.perf_counter()
            prediction = model.predict_one(dict(sample.x))
            prediction_seconds += time.perf_counter() - before
            if task == "regression" and (prediction is None or not math.isfinite(prediction)):
                raise ValueError("regression model emitted a nonfinite prediction")
            if task == "multiclass" and prediction is None:
                prediction = "__untrained__"
            index = train_arrived if sample.split == "train" else None
            if sample.split == "train":
                train_arrived += 1
            predictions.append({"sample_id": sample.sample_id, "seq": sample.seq,
                                "prediction": prediction, "split": sample.split,
                                "training_labels_available": released_train})
            pending.append((sample, prediction, index))
            release(sample.timestamp)
        if pending:
            release(pending[-1][0].available_at)
        report = {"schema_version": "task_evaluation-typed-v1", "identity": identity,
                  "run_key": identity, "complete": True, "config": config,
                  "observed": len(records), "metrics": {
                      split: {"n": counts[split], **{k: m.get() if counts[split] else None
                                                     for k, m in values.items()}}
                      for split, values in metric_sets.items()},
                  "per_task": {k: {"n": e["n"], **{n: m.get() for n, m in e["metrics"].items()}}
                               for k, e in per_task.items()},
                  "prediction_digest": hashlib.sha256(_json(predictions)).hexdigest(),
                  "updates": updates, "training": {"warmup_samples": warmup,
                      "postwarmup_samples": training_used, "budget": training_budget,
                      "blocked_decisions": budget_blocked},
                  "compute": {"elapsed_seconds": time.monotonic() - started,
                              "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                              * (1 if sys.platform == "darwin" else 1024),
                              "memory_scope": ("process lifetime high-water mark; "
                                               "includes harness and dependencies")},
                  "protocol": {"evaluation": "predict_before_label_release_and_train",
                               "selection": "bounded published prefix", "heldout_training": False,
                               "promotion": "automatic baseline; all trained candidates promoted",
                               "task_matrix": "not computed; per-task prequential metrics are not forgetting",
                               "resume": "complete-run idempotency; interrupted runs restart from scratch"}}
        report["costs"] = account(costs, errors=prediction_errors,
                                  presentations=warmup + training_used, updates=len(updates),
                                  train_seconds=training_seconds, predict_seconds=prediction_seconds)
        atomic_write(directory / "samples.jsonl", payload)
        atomic_write(directory / "predictions.json", _json(predictions))
        atomic_write(report_path, _json(report))
        return report
