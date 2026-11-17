"""Prequential protocols: predict, release delayed labels, monitor, then maybe update.

Streaming evaluates one event at a time (optionally over a real transport);
incremental predicts a whole batch before learning from any of its labels.
"""

import itertools
import time

from continual_bench.core.engine import Engine
from continual_bench.core.observability import OnlineObservability
from continual_bench.core.policies import BATCH_POLICY_KINDS

from .binary import BinaryProtocol, BinaryRun, Outcome, Progress, continual_evaluation
from .environment import current_rss_bytes


class Prequential(BinaryProtocol):
    def __init__(self, name: str, *, batch_size: int, use_transport: bool):
        self.name, self.batch_size, self.use_transport = name, batch_size, use_transport

    def build_engine(self, run: BinaryRun) -> Engine:
        config, settings = run.config, run.settings
        return Engine(
            run.model_factory,
            run.detector_factory,
            run.policy_factory(),
            warmup=config.warmup,
            window=config.replay_window,
            metric_window=config.metric_window,
            reset_detector=config.reset_detector_on_retrain,
            training_budget=config.training_budget,
            min_retrain_window=config.policy.min_window,
            versions=run.versions,
            observer=OnlineObservability(
                settings,
                continual_evaluation(run.source, run.model_factory, settings),
                current_rss_bytes,
                warmup=config.warmup,
            ),
            promotion=config.promotion.model_dump() if config.promotion else None,
        )

    def stream(self, run: BinaryRun, start: int):
        if self.use_transport and run.transport:
            return run.transport.events(run.manifest, run.events(), run.config.max_seconds)
        return run.events(start)

    def execute(self, run: BinaryRun) -> Outcome | Progress:
        config = run.config
        if config.policy.name in BATCH_POLICY_KINDS:
            from .binary_retraining import run_batch_retraining

            return run_batch_retraining(run)
        engine = run.snapshot["state"] if run.snapshot else self.build_engine(run)
        # EC2 retries recreate the output tree at a new absolute path.
        engine.versions.directory = run.versions.directory
        if getattr(engine, "evaluation_finished", False):
            return Outcome(engine.report(), engine, engine.model, engine.versions, engine.observer)
        previous_duration = getattr(engine, "evaluation_duration_seconds", 0.0)
        events = iter(self.stream(run, engine.next_seq))
        next_checkpoint = engine.next_seq + config.checkpoint_every
        probes = run.source.probes() if hasattr(run.source, "probes") else {}
        for batch in iter(lambda: list(itertools.islice(events, self.batch_size)), []):
            if time.monotonic() - run.started > config.max_seconds:
                raise TimeoutError("benchmark deadline exceeded; resume from last checkpoint")
            before = len(engine.windows)
            engine.observe_batch(batch)
            if len(engine.windows) > before:
                engine.probe(probes)
                if run.progress is not None:
                    for window in engine.windows[before:]:
                        run.progress(window["end_seq"], window)
            bounded_stop = run.stop_after is not None and engine.next_seq >= run.stop_after
            if engine.next_seq >= next_checkpoint or run.durable_checkpoint_due() or bounded_stop:
                engine.versions.flush()
                engine.evaluation_duration_seconds = previous_duration + time.monotonic() - run.started
                run.checkpoint(engine)
                next_checkpoint = engine.next_seq + config.checkpoint_every
            if bounded_stop:
                return Progress(engine.next_seq)
        if engine.next_seq != config.dataset.samples:
            raise ValueError(f"source ended early at {engine.next_seq}")
        engine.finish()
        engine.observer.sample_memory()
        engine.probe(probes)
        engine.evaluation_duration_seconds = previous_duration + time.monotonic() - run.started
        if run.checkpoint_due is not None:
            engine.evaluation_finished = True
            engine.versions.flush()
            run.checkpoint(engine, force=True)
        return Outcome(engine.report(), engine, engine.model, engine.versions, engine.observer)
