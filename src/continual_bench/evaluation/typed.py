"""Bounded prequential evaluation for regression and multiclass targets.

Any dataset (binary streams are adapted to two explicit classes) can be paired
with any model family that supports the dataset's target type.
"""

import hashlib
import json
import math
import pickle
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

from river.base import Classifier

from continual_bench.core.tasks import BINARY, MULTICLASS, REGRESSION, task_kind
from continual_bench.datasets.base import as_sample_source
from continual_bench.infrastructure.storage import atomic_write, run_lock
from continual_bench.models import MODELS, default_model
from continual_bench.models.encoding import EncodedEstimator, NativeEstimator

from .artifacts import system_figures, typed_figures, write_artifacts
from .costs import account, profile
from .environment import environment, peak_rss_bytes
from .scoring import metric_set
from .stream_observations import (
    StreamQuality,
    adapted_binary_positive_class,
    declared_class_labels,
    typed_observations,
)
from .stream_training_history import stream_training_history
from .system_metrics import SystemMetricsSampler
from .training_evidence import write_training_artifacts

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
        reset = not initial and self.policy == "periodic" and (released_train - self.warmup) % self.every == 0
        return initial, reset, initial or self.policy == "online" or reset


def _admit(source, task, max_samples, max_seconds, started):
    kind = task_kind(task)
    records, seen, last_stamp, last_available = [], set(), -1, -1
    payload = bytearray()
    for sample in islice(source.samples_iter(), max_samples):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("typed source deadline exceeded")
        if (
            sample.seq != len(records)
            or sample.timestamp < last_stamp
            or sample.available_at < sample.timestamp
            or sample.available_at < last_available
        ):
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
        if len(payload) + len(raw) + 1 > 64 * 1024**2:
            raise ValueError("typed materialization exceeds 64 MiB")
        payload.extend(raw)
        payload.extend(b"\n")
        records.append(sample)
        seen.add(sample.sample_id)
        last_stamp, last_available = sample.timestamp, sample.available_at
    return records, payload


def _record_model(directory, versions, model, action, seq):
    try:
        payload = pickle.dumps(model, protocol=5)
    except (pickle.PicklingError, AttributeError, TypeError):
        # A runtime-provided estimator may have no serializable state.
        return
    ordinal = len(versions)
    file = f"models/{ordinal:04d}.pkl"
    atomic_write(directory / file, payload)
    versions.append(
        {
            "ordinal": ordinal,
            "action": action,
            "seq": seq,
            "file": file,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    )


def _predict(model, x, positive_class):
    """Observe scores using the same one inference call as River's default argmax."""
    if (
        positive_class is not None
        and isinstance(model, (EncodedEstimator, NativeEstimator))
        and type(model.estimator).predict_one is Classifier.predict_one
    ):
        probabilities = model.predict_proba_one(x)
        if probabilities:
            prediction = max(probabilities, key=probabilities.get)
            probability = float(probabilities.get(positive_class, 0.0))
            return prediction, probability if math.isfinite(probability) and 0 <= probability <= 1 else None
        return None, None
    return model.predict_one(x), None


@dataclass
class TypedState:
    model: object
    metric_sets: dict
    counts: dict
    replay: deque
    pending: deque = field(default_factory=deque)
    per_task: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)
    updates: list = field(default_factory=list)
    versions: list = field(default_factory=list)
    trained_seqs: set = field(default_factory=set)
    captured: dict = field(default_factory=lambda: {"initial": [], "updates": {}})
    labeled: dict = field(
        default_factory=lambda: {split: [] for split in ("prequential", "validation", "test")}
    )
    next_sample: int = 0
    release_at: int | None = None
    pending_progress: bool = False
    train_arrived: int = 0
    released_train: int = 0
    training_used: int = 0
    budget_blocked: int = 0
    prediction_seconds: float = 0.0
    prediction_errors: float = 0.0
    training_seconds: float = 0.0
    tracking_seconds: float = 0.0
    initial_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    initial_first: int | None = None
    initial_fit: dict | None = None
    model_version: str | None = None
    finished: bool = False


def run_typed(
    source,
    root,
    *,
    task=None,
    policy="online",
    warmup=20,
    every=50,
    window=128,
    max_samples=1000,
    max_seconds=60,
    training_budget=None,
    costs=None,
    model=None,
    model_factory=None,
    models=MODELS,
    progress=None,
    checkpoint=None,
):
    """Evaluate a deterministic prefix; never learn validation/test labels.

    `source` exposes samples_iter() of TaskSample plus manifest(); binary Event
    sources are adapted to classes "0"/"1". Labels are released after prediction
    at the current timestamp, including at delay zero. `model` selects a family
    ({"name", "params"}); `model_factory` injects a learner directly. Complete
    reruns are idempotent by config/data/code identity. A durable checkpoint
    can resume at a sample or released-label boundary.
    """
    costs = profile(costs)
    source = as_sample_source(source)
    manifest = source.manifest()
    task = task or manifest.get("task")
    if task not in TYPED_TASKS or manifest.get("task") != task:
        raise ValueError("typed task must match the source's regression/multiclass declaration")
    if policy not in TYPED_POLICIES:
        raise ValueError("typed evaluator supports none/online/periodic; cost/drift policies are unsupported")
    for name, value, minimum, maximum in (
        ("warmup", warmup, 1, 10_000),
        ("every", every, 1, 10_000),
        ("window", window, 1, 10_000),
        ("max_samples", max_samples, 2, 10_000),
        ("max_seconds", max_seconds, 1, 600),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid {name} budget")
    if training_budget is not None and (type(training_budget) is not int or training_budget < 0):
        raise ValueError("training_budget must be a nonnegative integer")
    if model_factory is None:
        model_factory, descriptor = resolve_model(task, model, models=models)
    else:
        descriptor = {"family": "injected", "task": task}
    kind = task_kind(task)
    positive_class = adapted_binary_positive_class(manifest) if task == MULTICLASS else None
    started = time.monotonic()
    records, payload = _admit(source, task, max_samples, max_seconds, started)
    train_count = sum(s.split == "train" for s in records)
    if train_count <= warmup:
        raise ValueError("typed warmup must leave at least one training-stream evaluation sample")
    online_last = train_count if training_budget is None else min(train_count, warmup + training_budget)
    config = {
        "schema_version": "task_evaluation-typed-v1",
        "task": task,
        "policy": policy,
        "warmup": warmup,
        "every": every,
        "window": window,
        "max_samples": max_samples,
        "max_seconds": max_seconds,
        "training_budget": training_budget,
        "costs": costs,
        "source": manifest,
        "samples_sha256": hashlib.sha256(payload).hexdigest(),
        "environment": environment(),
        "model": descriptor,
    }
    # Native tuple-valued descriptors must match the persisted JSON shape on replay.
    config = json.loads(_json(config))
    identity = hashlib.sha256(_json(config)).hexdigest()[:24]
    directory = Path(root) / identity
    snapshot = checkpoint.load() if checkpoint is not None else None
    if snapshot is not None and snapshot["contract"] != config:
        raise ValueError("typed checkpoint contract changed")
    with run_lock(root, identity):
        report_path = directory / "report.json"
        if report_path.exists():
            if (directory / "samples.jsonl").read_bytes() != payload:
                raise ValueError("typed materialized samples changed for a completed run")
            return json.loads(report_path.read_text())
        with SystemMetricsSampler(root) as sampler:
            if snapshot is None:
                metric_sets = {split: metric_set(task) for split in ("train", "validation", "test")}
                state = TypedState(
                    model_factory(), metric_sets, {split: 0 for split in metric_sets}, deque(maxlen=window)
                )
            else:
                state = snapshot["state"]
            previous_duration = state.elapsed_seconds
            schedule = PeriodicSchedule(policy, warmup, every)
            metric_sets, counts, per_task = state.metric_sets, state.counts, state.per_task
            predictions, updates, versions = state.predictions, state.updates, state.versions
            replay, pending = state.replay, state.pending
            captured, labeled, trained_seqs = state.captured, state.labeled, state.trained_seqs

            def save(*, force=False):
                if checkpoint is None or not (force or checkpoint.due()):
                    return
                requested = checkpoint.requested()
                state.elapsed_seconds = previous_duration + time.monotonic() - started
                checkpoint.save({"contract": config, "state": state}, force=force or requested)
                if requested:
                    raise InterruptedError("benchmark interrupted at a typed checkpoint")

            def score(sample, prediction):
                state.prediction_errors += kind.error_units(sample.y, prediction, costs)
                counts[sample.split] += 1
                for metric in metric_sets[sample.split].values():
                    metric.update(sample.y, prediction)
                if sample.task_id is not None:
                    key = f"{sample.split}:{sample.task_id}"
                    if key not in per_task:
                        per_task[key] = {"n": 0, "metrics": metric_set(task)}
                    entry = per_task[key]
                    entry["n"] += 1
                    for metric in entry["metrics"].values():
                        metric.update(sample.y, prediction)

            def consume_label(sample, prediction, train_index, observed):
                if sample.split != "train" or train_index >= warmup:
                    score(sample, prediction)
                    tick = time.perf_counter()
                    split = "prequential" if sample.split == "train" else sample.split
                    labeled[split].append(observed)
                    state.tracking_seconds += time.perf_counter() - tick
                if sample.split != "train":
                    return
                state.released_train += 1
                replay.append(sample)
                initial, reset, learn = schedule.plan(train_index, state.released_train)
                if not learn:
                    return
                selected = list(replay) if reset else [sample]
                if (
                    not initial
                    and training_budget is not None
                    and state.training_used + len(selected) > training_budget
                ):
                    state.budget_blocked += 1
                    return
                before = time.perf_counter()
                if reset:
                    state.model = model_factory()
                for row in selected:
                    state.model.learn_one(dict(row.x), row.y)
                seconds = time.perf_counter() - before
                state.training_seconds += seconds
                tick = time.perf_counter()
                repeated = sum(row.seq in trained_seqs for row in selected)
                trained_seqs.update(row.seq for row in selected)
                state.model_version = f"trained-through-{sample.seq}"
                step = {
                    "seq": sample.seq,
                    "seconds": seconds,
                    "size": len(selected),
                    "repeated_presentations": repeated,
                }
                if not reset:
                    step["prediction"] = observed
                if initial:
                    captured["initial"].append(step)
                else:
                    captured["updates"][str(sample.seq)] = step
                state.tracking_seconds += time.perf_counter() - tick
                if initial:
                    if state.initial_first is None:
                        state.initial_first = sample.seq
                    state.initial_seconds += seconds
                    if train_index == warmup - 1:
                        state.initial_fit = {
                            "seq": sample.seq,
                            "train_seq_range": [state.initial_first, sample.seq + 1],
                            "train_samples": warmup,
                            "seconds": state.initial_seconds,
                        }
                        _record_model(directory, versions, state.model, "initial_fit", sample.seq)
                else:
                    state.training_used += len(selected)
                    action = "reset_retrain" if reset else "learn_one"
                    updates.append(
                        {
                            "seq": sample.seq,
                            "released_at": state.release_at,
                            "train_samples": len(selected),
                            "action": action,
                            "train_seq_range": [selected[0].seq, selected[-1].seq + 1],
                            "seconds": seconds,
                        }
                    )
                    if reset or (policy == "online" and state.released_train == online_last):
                        _record_model(directory, versions, state.model, action, sample.seq)

            def release(now):
                state.release_at = now
                while pending and pending[0][0].available_at <= now:
                    if time.monotonic() - started > max_seconds:
                        raise TimeoutError("typed evaluation deadline exceeded")
                    consume_label(*pending.popleft())
                    save()
                state.release_at = None

            def publish_progress(step):
                if progress is not None and (step == 1 or step % window == 0):
                    observed = {
                        f"{split}/{name}": metric.get()
                        for split, metrics in metric_sets.items()
                        if counts[split]
                        for name, metric in metrics.items()
                    }
                    progress(step, {"observed": step, **observed})

            if state.release_at is not None:
                release(state.release_at)
            if state.pending_progress:
                publish_progress(state.next_sample)
                state.pending_progress = False
            for step, sample in enumerate(islice(records, state.next_sample, None), state.next_sample + 1):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("typed evaluation deadline exceeded")
                before = time.perf_counter()
                prediction, probability = _predict(state.model, dict(sample.x), positive_class)
                state.prediction_seconds += time.perf_counter() - before
                prediction = kind.normalize_prediction(prediction)
                index = state.train_arrived if sample.split == "train" else None
                if sample.split == "train":
                    state.train_arrived += 1
                predictions.append(
                    {
                        "sample_id": sample.sample_id,
                        "seq": sample.seq,
                        "prediction": prediction,
                        "split": sample.split,
                        "training_labels_available": state.released_train,
                    }
                )
                tick = time.perf_counter()
                observation = {
                    "seq": sample.seq,
                    "prediction": prediction,
                    "label": sample.y,
                    "probability": probability,
                    "model_version": state.model_version,
                }
                pending.append((sample, prediction, index, observation))
                state.tracking_seconds += time.perf_counter() - tick
                state.next_sample = step
                state.pending_progress = True
                release(sample.timestamp)
                publish_progress(step)
                state.pending_progress = False
                save()
            if pending:
                release(pending[-1][0].available_at)
            if not state.finished:
                state.finished = True
                save(force=True)
            report = {
                "schema_version": "task_evaluation-typed-v1",
                "identity": identity,
                "run_key": identity,
                "complete": True,
                "config": config,
                "observed": len(records),
                "metrics": {
                    split: {
                        "n": counts[split],
                        **{k: m.get() if counts[split] else None for k, m in values.items()},
                    }
                    for split, values in metric_sets.items()
                },
                "per_task": {
                    k: {"n": e["n"], **{n: m.get() for n, m in e["metrics"].items()}}
                    for k, e in per_task.items()
                },
                "prediction_digest": hashlib.sha256(_json(predictions)).hexdigest(),
                "updates": updates,
                "training": {
                    "warmup_samples": warmup,
                    "postwarmup_samples": state.training_used,
                    "budget": training_budget,
                    "blocked_decisions": state.budget_blocked,
                    "initial_fit": state.initial_fit,
                },
                "compute": {
                    "elapsed_seconds": previous_duration + time.monotonic() - started,
                    "training_seconds": state.training_seconds,
                    "prediction_seconds": state.prediction_seconds,
                    "process_peak_rss_bytes": peak_rss_bytes(),
                    "memory_scope": ("process lifetime high-water mark; includes harness and dependencies"),
                },
                "protocol": {
                    "evaluation": "predict_before_label_release_and_train",
                    "selection": "bounded published prefix",
                    "heldout_training": False,
                    "promotion": "automatic baseline; all trained candidates promoted",
                    "task_matrix": ("not computed; per-task prequential metrics are not forgetting"),
                    "resume": (
                        "checkpointed sample and label boundaries"
                        if checkpoint is not None
                        else "complete-run idempotency; interrupted runs restart from scratch"
                    ),
                },
            }
            model_index = _json({"format_version": 1, "versions": versions})
            if versions:
                atomic_write(directory / "model-versions.json", model_index)
            model_storage = sum(entry["bytes"] for entry in versions) + (len(model_index) if versions else 0)
            report["model_versions"] = {
                "count": len(versions),
                "storage_bytes": model_storage,
                "index_sha256": hashlib.sha256(model_index).hexdigest() if versions else None,
            }
            report["costs"] = account(
                costs,
                errors=state.prediction_errors,
                presentations=warmup + state.training_used,
                updates=len(updates),
                train_seconds=state.training_seconds,
                predict_seconds=state.prediction_seconds,
                model_storage_bytes=model_storage,
            )
        report["system_metrics"] = sampler.report()
        tick = time.perf_counter()
        if task == REGRESSION:
            quality = StreamQuality(task, target_unit=manifest.get("target_unit") or "target_unit")
        else:
            declared = declared_class_labels(manifest)
            admitted = sorted({sample.y for sample in records} - set(declared))
            classes = (*declared, *admitted)
            prediction_classes = {row["prediction"] for row in predictions} - set(classes)
            quality = StreamQuality(
                BINARY if positive_class else MULTICLASS,
                classes,
                positive_class,
                confusion_classes=tuple(sorted(prediction_classes, key=str)),
            )
        report["training_history"] = stream_training_history(
            report,
            directory,
            versions=versions,
            captured=captured,
            quality=quality,
        )
        report["tracking_observations"] = typed_observations(
            report,
            quality,
            labeled,
            unique_training=len(trained_seqs),
            tracking_seconds=state.tracking_seconds + time.perf_counter() - tick,
        )
        timeline = [(r.seq, r.split, r.y, p["prediction"]) for r, p in zip(records, predictions)]
        report["artifacts"] = write_artifacts(
            directory,
            typed_figures(report, timeline) + system_figures(report["system_metrics"]),
            {
                "summary/metrics": {
                    "metrics": report["metrics"],
                    "per_task": report["per_task"],
                    "training": report["training"],
                    "compute": report["compute"],
                },
                "costs/scenario": report["costs"],
                "events/updates": report["updates"],
                "series/predictions": [
                    {"seq": seq, "split": split, "target": y, "prediction": p}
                    for seq, split, y, p in timeline
                ],
                "series/tracking-observations": labeled,
                "system/samples": report["system_metrics"],
            },
        )
        report["artifacts"] = write_training_artifacts(directory, report)
        atomic_write(directory / "samples.jsonl", payload)
        atomic_write(directory / "predictions.json", _json(predictions))
        atomic_write(report_path, _json(report))
        return report
