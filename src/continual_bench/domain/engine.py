"""Only this engine controls the per-sample/batch evaluation and update cycle."""

import copy
import hashlib
import json
import time
from collections import deque
from dataclasses import asdict
from typing import Callable

from .contracts import DecisionContext, Event
from .metrics import Metrics


class Engine:
    schema_version = 1

    def __init__(
        self,
        model_factory: Callable,
        detector_factory: Callable,
        policy,
        *,
        warmup: int,
        window: int,
        metric_window: int,
        reset_detector: bool = True,
        training_budget: int | None = None,
        min_retrain_window: int = 1,
        versions=None,
        observer=None,
    ):
        self.model_factory = model_factory
        self.detector_factory = detector_factory
        self.model = model_factory()
        self.detector = detector_factory()
        self.policy = policy
        self.warmup = warmup
        self.replay = deque(maxlen=window)
        self.metric_window = metric_window
        self.reset_detector = reset_detector
        self.training_budget = training_budget
        self.min_retrain_window = min_retrain_window
        self.versions = versions
        self.observer = observer
        self.training_used = 0
        self.budget_blocked = 0
        self.next_seq = 0
        self.last_timestamp = -1
        self.last_available_at = -1
        self.pending = deque()
        self.labeled = 0
        self.total = Metrics()
        self.current = Metrics()
        self.windows = []
        self.alarms = []
        self.updates = []
        self.seen_concepts = set()
        self.retention = []
        self.best_probe = {}
        self.audit = hashlib.sha256(b"continual-bench-v1").hexdigest()
        self.compute_seconds = 0.0

    def observe_batch(self, batch: list[Event]) -> None:
        start = time.perf_counter()
        accepted = []
        # All batch predictions precede all learning from this batch.
        for event in batch:
            if event.seq < self.next_seq:
                continue  # Verified source manifest makes seq a bounded replay watermark.
            if event.seq != self.next_seq:
                raise ValueError(f"out-of-order event: expected {self.next_seq}, got {event.seq}")
            if event.timestamp <= self.last_timestamp or event.available_at < event.timestamp:
                raise ValueError("invalid event-time/label-availability order")
            if event.available_at < self.last_available_at:
                raise ValueError("label availability must preserve source order")
            prediction_started = time.perf_counter()
            p = self.model.predict(dict(event.x))
            prediction_seconds = time.perf_counter() - prediction_started
            if not 0 <= p <= 1:
                raise ValueError("invalid prediction")
            trace = (self.observer.before_prediction(event, self.model, self.labeled, prediction_seconds)
                     if self.observer is not None else {})
            self.pending.append((event, p, trace))
            self.next_seq += 1
            self.last_timestamp = event.timestamp
            self.last_available_at = event.available_at
            accepted.append(event)
            self.audit = hashlib.sha256(
                (self.audit + json.dumps([event.event_id, p], separators=(",", ":"))).encode()
            ).hexdigest()
        if accepted:
            self.release(accepted[-1].timestamp)
        self.compute_seconds += time.perf_counter() - start
        if self.observer is not None:
            self.observer.sample_memory()

    def release(self, now: int) -> None:
        # Dataset adapters require monotonically increasing available_at.
        while self.pending and self.pending[0][0].available_at <= now:
            event, p, trace = self.pending.popleft()
            self.labeled += 1
            self.replay.append((dict(event.x), event.y))
            if event.seq < self.warmup:
                train_started = time.perf_counter()
                self.model.learn(dict(event.x), event.y)
                if self.observer is not None:
                    self.observer.warmup_train_seconds += time.perf_counter() - train_started
                    self.observer.warmup_train_samples += 1
                self.policy.updated(self.labeled)
                if self.versions is not None and event.seq == self.warmup - 1:
                    self.versions.record(self.model, {"action": "initial_fit", "seq_range": [0, self.warmup],
                                                      "label_available_at": event.available_at,
                                                      "released_at": now})
                if self.observer is not None:
                    self.observer.after_label(event, p, trace, now, self.model, False)
                continue
            self.total.add(event.y, p)
            self.current.add(event.y, p)
            if event.concept is not None:
                self.seen_concepts.add(event.concept)
            monitor_started = time.perf_counter()
            alarm = self.detector.update(int((p >= 0.5) != event.y))
            decision = self.policy.decide(DecisionContext(self.labeled, len(self.replay), alarm))
            if self.observer is not None:
                self.observer.monitor_seconds += time.perf_counter() - monitor_started
            if alarm:
                self.alarms.append(
                    {
                        "event_id": event.event_id,
                        "seq": event.seq,
                        "available_at": event.available_at,
                        "released_at": now,
                        "decision": asdict(decision),
                    }
                )
            if decision.action != "none":
                train_samples = len(self.replay) if decision.action == "reset_retrain" else 1
                if self.training_budget is not None:
                    train_samples = min(train_samples, self.training_budget - self.training_used)
                    minimum = self.min_retrain_window if decision.action == "reset_retrain" else 1
                    if train_samples < minimum:
                        self.budget_blocked += 1
                        if self.current.n == self.metric_window:
                            self.close_window(event.seq)
                        if self.observer is not None:
                            self.observer.after_label(event, p, trace, now, self.model, True)
                        continue
                previous = self.observer.before_update(self.model) if self.observer is not None else None
                started = time.perf_counter()
                if decision.action == "reset_retrain":
                    self.model = self.model_factory()
                    for x, y in list(self.replay)[-train_samples:]:
                        self.model.learn(dict(x), y)
                    if self.reset_detector:
                        self.detector = self.detector_factory()
                elif decision.action == "learn_one":
                    self.model.learn(dict(event.x), event.y)
                else:
                    raise ValueError(f"unknown update action: {decision.action}")
                self.policy.updated(self.labeled)
                self.training_used += train_samples
                self.updates.append(
                    {
                        "seq": event.seq,
                        "labeled": self.labeled,
                        "action": decision.action,
                        "reason": decision.reason,
                        "train_samples": train_samples,
                        "seconds": time.perf_counter() - started,
                    }
                )
                if self.observer is not None:
                    self.observer.after_update(previous, self.updates[-1])
                if self.versions is not None:
                    start = event.seq - train_samples + 1 if decision.action == "reset_retrain" else event.seq
                    self.versions.record(self.model, {"action": decision.action,
                                                      "seq_range": [start, event.seq + 1],
                                                      "label_available_at": event.available_at,
                                                      "released_at": now})
            if self.current.n == self.metric_window:
                self.close_window(event.seq)
            if self.observer is not None:
                self.observer.after_label(event, p, trace, now, self.model, True)

    def close_window(self, end_seq: int) -> None:
        if self.current.n:
            self.windows.append({"end_seq": end_seq, **self.current.values()})
            self.current = Metrics()

    def probe(self, probes: dict[int, list[tuple[dict, bool]]]) -> None:
        # Separate, never-trained synthetic reference probes, evaluated only for seen concepts.
        frozen = copy.deepcopy(self.model)
        for concept in sorted(self.seen_concepts):
            samples = probes.get(concept, [])
            if not samples:
                continue
            accuracy = sum((frozen.predict(dict(x)) >= 0.5) == y for x, y in samples) / len(samples)
            self.best_probe[concept] = max(self.best_probe.get(concept, 0), accuracy)
            self.retention.append(
                {
                    "observed_seq": self.next_seq - 1,
                    "concept": concept,
                    "accuracy": accuracy,
                    "drop_from_best_probe": self.best_probe[concept] - accuracy,
                }
            )

    def finish(self) -> None:
        # Explicit label-drain horizon after the last feature, never used to predict earlier data.
        if self.pending:
            self.release(self.pending[-1][0].available_at)
        self.close_window(self.next_seq - 1)

    def report(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "observed": self.next_seq,
            "labeled": self.labeled,
            "metrics": self.total.values(),
            "windows": self.windows,
            "alarms": self.alarms,
            "updates": self.updates,
            "training": {"budget": self.training_budget, "used": self.training_used,
                         "blocked_decisions": self.budget_blocked, "excludes_warmup": True},
            "retention_probes": self.retention,
            "prediction_digest": self.audit,
            "compute_seconds": self.compute_seconds,
        }
