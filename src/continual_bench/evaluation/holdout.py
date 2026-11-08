"""Offline protocol: temporal training prefix, purged label delay, frozen test suffix.

There is no online detector/policy cycle; only the initial fit is versioned.
"""

import itertools
import time

from continual_bench.core.metrics import Metrics
from continual_bench.core.observability import OnlineObservability

from .binary import BinaryProtocol, BinaryRun, Outcome
from .environment import current_rss_bytes


class TemporalHoldout(BinaryProtocol):
    name = "temporal_holdout"

    def execute(self, run: BinaryRun) -> Outcome:
        config = run.config
        model = run.model_factory()
        split = int(config.dataset.samples * config.offline_train_fraction)
        test_start = next(itertools.islice(run.events(), split, split + 1)).timestamp
        metrics, current, windows = Metrics(), Metrics(), []
        observer = OnlineObservability(run.settings, memory_sampler=current_rss_bytes, warmup=split)
        observer.sample_memory()
        trained = 0
        for event in run.events():
            if time.monotonic() - run.started > config.max_seconds:
                raise TimeoutError("offline deadline exceeded")
            if event.seq < split:
                if event.available_at < test_start:
                    train_started = time.perf_counter()
                    model.learn(dict(event.x), event.y)
                    observer.warmup_train_seconds += time.perf_counter() - train_started
                    observer.warmup_train_samples += 1
                    trained += 1
            else:
                prediction_started = time.perf_counter()
                p = model.predict(dict(event.x))
                elapsed = time.perf_counter() - prediction_started
                trace = observer.before_prediction(event, model, trained, elapsed)
                observer.after_label(event, p, trace, event.available_at, model, True)
                metrics.add(event.y, p)
                current.add(event.y, p)
                if current.n == config.metric_window:
                    windows.append({"end_seq": event.seq, **current.values()})
                    current = Metrics()
            if event.seq % config.metric_window == 0:
                observer.sample_memory()
        if current.n:
            windows.append({"end_seq": config.dataset.samples - 1, **current.values()})
        if trained == 0:
            raise ValueError("offline prefix has no labels available before test start")
        report = {
            "observed": config.dataset.samples,
            "labeled": config.dataset.samples,
            "metrics": metrics.values(),
            "windows": windows,
            "updates": [],
            "alarms": [],
            "retention_probes": [],
            "split": {
                "train": [0, trained],
                "purged_label_delay": [trained, split],
                "test": [split, config.dataset.samples],
                "intervals": "half-open",
            },
            "compute_seconds": time.monotonic() - run.started,
        }
        run.versions.record(model, {"action": "initial_fit", "seq_range": [0, trained],
                                    "label_cutoff": test_start})
        observer.sample_memory()
        return Outcome(report, {"model": model, "split": split}, model, run.versions, observer)
