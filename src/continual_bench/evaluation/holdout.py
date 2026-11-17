"""Offline protocol: temporal training prefix, purged label delay, frozen test suffix.

There is no online detector/policy cycle; only the initial fit is versioned.
"""

import time
from dataclasses import dataclass, field

from continual_bench.core.metrics import Metrics
from continual_bench.core.observability import OnlineObservability

from .binary import BinaryProtocol, BinaryRun, Outcome
from .environment import current_rss_bytes


@dataclass
class HoldoutState:
    model: object
    observer: OnlineObservability
    split: int
    test_start: int
    metrics: Metrics = field(default_factory=Metrics)
    current: Metrics = field(default_factory=Metrics)
    windows: list = field(default_factory=list)
    trained: int = 0
    next_seq: int = 0
    evaluation_duration_seconds: float = 0.0


class TemporalHoldout(BinaryProtocol):
    name = "temporal_holdout"

    def execute(self, run: BinaryRun) -> Outcome:
        config = run.config
        split = int(config.dataset.samples * config.offline_train_fraction)
        state = run.snapshot["state"] if run.snapshot else None
        if state is None:
            test_start = next(run.events(split)).timestamp
            observer = OnlineObservability(run.settings, memory_sampler=current_rss_bytes, warmup=split)
            observer.sample_memory()
            state = HoldoutState(run.model_factory(), observer, split, test_start)
        previous_duration = state.evaluation_duration_seconds
        previously_evaluated = state.next_seq == config.dataset.samples
        model, observer = state.model, state.observer
        for event in run.events(state.next_seq):
            if time.monotonic() - run.started > config.max_seconds:
                raise TimeoutError("offline deadline exceeded")
            if event.seq < split:
                if event.available_at < state.test_start:
                    train_started = time.perf_counter()
                    model.learn(dict(event.x), event.y)
                    elapsed = time.perf_counter() - train_started
                    observer.warmup_train_seconds += elapsed
                    observer.warmup_train_samples += 1
                    observer.record_learning(event.seq, elapsed, 1, initial=True)
                    state.trained += 1
            else:
                prediction_started = time.perf_counter()
                p = model.predict(dict(event.x))
                elapsed = time.perf_counter() - prediction_started
                trace = observer.before_prediction(event, model, state.trained, elapsed)
                observer.after_label(event, p, trace, event.available_at, model, True)
                state.metrics.add(event.y, p)
                state.current.add(event.y, p)
                if state.current.n == config.metric_window:
                    state.windows.append({"end_seq": event.seq, **state.current.values()})
                    if run.progress is not None:
                        run.progress(event.seq, state.windows[-1])
                    state.current = Metrics()
            if event.seq % config.metric_window == 0:
                observer.sample_memory()
            state.next_seq = event.seq + 1
            if run.durable_checkpoint_due():
                state.evaluation_duration_seconds = previous_duration + time.monotonic() - run.started
                run.checkpoint(state)
        state.evaluation_duration_seconds = previous_duration + time.monotonic() - run.started
        if run.checkpoint_due is not None and not previously_evaluated:
            run.checkpoint(state, force=True)
        windows = list(state.windows)
        if state.current.n:
            windows.append({"end_seq": config.dataset.samples - 1, **state.current.values()})
        if state.trained == 0:
            raise ValueError("offline prefix has no labels available before test start")
        report = {
            "observed": config.dataset.samples,
            "labeled": config.dataset.samples,
            "metrics": state.metrics.values(),
            "windows": windows,
            "updates": [],
            "alarms": [],
            "retention_probes": [],
            "split": {
                "train": [0, state.trained],
                "purged_label_delay": [state.trained, split],
                "test": [split, config.dataset.samples],
                "intervals": "half-open",
            },
            "compute_seconds": state.evaluation_duration_seconds,
        }
        run.versions.record(
            model,
            {"action": "initial_fit", "seq_range": [0, state.trained], "label_cutoff": state.test_start},
        )
        observer.sample_memory()
        return Outcome(report, state, model, run.versions, observer)
