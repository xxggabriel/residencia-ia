"""Bounded prequential evaluation for regression and multiclass targets.

Any dataset (binary streams are adapted to two explicit classes) can be paired
with any model family that supports the dataset's target type.
"""

import hashlib
import json
import time
from collections import deque
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from continual_bench.core.tasks import MULTICLASS, REGRESSION, task_kind
from continual_bench.datasets.base import as_sample_source
from continual_bench.infrastructure.storage import atomic_write, run_lock
from continual_bench.models import MODELS, default_model

from .artifacts import system_figures, typed_figures, write_artifacts
from .costs import account, profile
from .environment import environment, peak_rss_bytes
from .scoring import metric_set
from .system_metrics import SystemMetricsSampler

TYPED_TASKS = (REGRESSION, MULTICLASS)
TYPED_POLICIES = ("none", "online", "periodic")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def resolve_model(task, model=None, *, models=MODELS):
    """Return (factory, descriptor) for a {"name", "params"} request or the task default."""
    if model is not None and not isinstance(model, dict):
        model = model.model_dump()
    name = model["name"] if model else default_model(task)
    params = dict(model.get("params", {})) if model else {}
    return models.get(name).factory(task, **params), {"family": name, "task": task, "params": params}


class PeriodicSchedule:
    """Typed update schedule: warmup fit, then none/online/periodic full-window resets."""

    def __init__(self, policy, warmup, every):
        self.policy, self.warmup, self.every = policy, warmup, every

    def plan(self, train_index, released_train):
        """Return (initial, reset, learn) for the released training label."""
        initial = train_index < self.warmup
        reset = (not initial and self.policy == "periodic"
                 and (released_train - self.warmup) % self.every == 0)
        return initial, reset, initial or self.policy == "online" or reset


def _admit(source, task, max_samples, max_seconds, started):
    kind = task_kind(task)
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
        kind.validate_features(sample.x)
        kind.validate_target(sample.y)
        raw = _json(asdict(sample))
        payload_bytes += len(raw) + 1
        if payload_bytes > 64 * 1024**2:
            raise ValueError("typed materialization exceeds 64 MiB")
        records.append(sample)
        seen.add(sample.sample_id)
        last_stamp, last_available = sample.timestamp, sample.available_at
    return records


def run_typed(source, root, *, task=None, policy="online", warmup=20, every=50, window=128,
              max_samples=1000, max_seconds=60, training_budget=None, costs=None, model=None,
              model_factory=None, models=MODELS):
    """Evaluate a deterministic prefix; never learn validation/test labels.

    `source` exposes samples_iter() of TaskSample plus manifest(); binary Event
    sources are adapted to classes "0"/"1". Labels are released after prediction
    at the current timestamp, including at delay zero. `model` selects a family
    ({"name", "params"}); `model_factory` injects a learner directly. Complete
    reruns are idempotent by config/data/code identity; no partial resume.
    """
    costs = profile(costs)
    source = as_sample_source(source)
    manifest = source.manifest()
    task = task or manifest.get("task")
    if task not in TYPED_TASKS or manifest.get("task") != task:
        raise ValueError("typed task must match the source's regression/multiclass declaration")
    if policy not in TYPED_POLICIES:
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
    if model_factory is None:
        model_factory, descriptor = resolve_model(task, model, models=models)
    else:
        descriptor = {"family": "injected", "task": task}
    kind = task_kind(task)
    started = time.monotonic()
    records = _admit(source, task, max_samples, max_seconds, started)
    if sum(s.split == "train" for s in records) <= warmup:
        raise ValueError("typed warmup must leave at least one training-stream evaluation sample")
    payload = b"".join(_json(asdict(sample)) + b"\n" for sample in records)
    config = {"schema_version": "task_evaluation-typed-v1", "task": task, "policy": policy, "warmup": warmup,
              "every": every, "window": window, "max_samples": max_samples, "max_seconds": max_seconds,
              "training_budget": training_budget, "costs": costs, "source": manifest,
              "samples_sha256": hashlib.sha256(payload).hexdigest(), "environment": environment(),
              "model": descriptor}
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
        with SystemMetricsSampler(root) as sampler:
            model = model_factory()
            schedule = PeriodicSchedule(policy, warmup, every)
            metric_sets = {split: metric_set(task) for split in ("train", "validation", "test")}
            counts = {split: 0 for split in metric_sets}
            per_task, predictions, updates = {}, [], []
            replay, pending = deque(maxlen=window), deque()
            train_arrived = released_train = training_used = budget_blocked = 0
            prediction_seconds = training_seconds = 0.0
            prediction_errors = 0.0

            def score(sample, prediction):
                nonlocal prediction_errors
                prediction_errors += kind.error_units(sample.y, prediction, costs)
                counts[sample.split] += 1
                for metric in metric_sets[sample.split].values():
                    metric.update(sample.y, prediction)
                if sample.task_id is not None:
                    key = f"{sample.split}:{sample.task_id}"
                    entry = per_task.setdefault(key, {"n": 0, "metrics": metric_set(task)})
                    entry["n"] += 1
                    for metric in entry["metrics"].values():
                        metric.update(sample.y, prediction)

            def release(now):
                nonlocal model, released_train, training_used, training_seconds, budget_blocked
                while pending and pending[0][0].available_at <= now:
                    if time.monotonic() - started > max_seconds:
                        raise TimeoutError("typed evaluation deadline exceeded")
                    sample, prediction, train_index = pending.popleft()
                    if sample.split != "train" or train_index >= warmup:
                        score(sample, prediction)
                    if sample.split != "train":
                        continue
                    released_train += 1
                    replay.append(sample)
                    initial, reset, learn = schedule.plan(train_index, released_train)
                    if not learn:
                        continue
                    selected = list(replay) if reset else [sample]
                    if (not initial and training_budget is not None
                            and training_used + len(selected) > training_budget):
                        budget_blocked += 1
                        continue
                    before = time.perf_counter()
                    if reset:
                        model = model_factory()
                    for row in selected:
                        model.learn_one(dict(row.x), row.y)
                    seconds = time.perf_counter() - before
                    training_seconds += seconds
                    if not initial:
                        training_used += len(selected)
                        updates.append({"seq": sample.seq, "released_at": now,
                                        "train_samples": len(selected),
                                        "action": "reset_retrain" if reset else "learn_one",
                                        "seconds": seconds})

            for sample in records:
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("typed evaluation deadline exceeded")
                before = time.perf_counter()
                prediction = model.predict_one(dict(sample.x))
                prediction_seconds += time.perf_counter() - before
                prediction = kind.normalize_prediction(prediction)
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
                                  "training_seconds": training_seconds,
                                  "prediction_seconds": prediction_seconds,
                                  "process_peak_rss_bytes": peak_rss_bytes(),
                                  "memory_scope": ("process lifetime high-water mark; "
                                                   "includes harness and dependencies")},
                      "protocol": {"evaluation": "predict_before_label_release_and_train",
                                   "selection": "bounded published prefix", "heldout_training": False,
                                   "promotion": "automatic baseline; all trained candidates promoted",
                                   "task_matrix": ("not computed; per-task prequential metrics are not "
                                                   "forgetting"),
                                   "resume": ("complete-run idempotency; interrupted runs restart "
                                              "from scratch")}}
            report["costs"] = account(costs, errors=prediction_errors,
                                      presentations=warmup + training_used, updates=len(updates),
                                      train_seconds=training_seconds, predict_seconds=prediction_seconds,
                                      model_storage_bytes=0)
        report["system_metrics"] = sampler.report()
        timeline = [(r.seq, r.split, r.y, p["prediction"]) for r, p in zip(records, predictions)]
        report["artifacts"] = write_artifacts(
            directory, typed_figures(report, timeline) + system_figures(report["system_metrics"]),
            {"summary/metrics": {"metrics": report["metrics"], "per_task": report["per_task"],
                                 "training": report["training"], "compute": report["compute"]},
             "costs/scenario": report["costs"], "events/updates": report["updates"],
             "series/predictions": [{"seq": seq, "split": split, "target": y, "prediction": p}
                                    for seq, split, y, p in timeline],
             "system/samples": report["system_metrics"]})
        atomic_write(directory / "samples.jsonl", payload)
        atomic_write(directory / "predictions.json", _json(predictions))
        atomic_write(report_path, _json(report))
        return report
