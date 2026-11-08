"""Binary evaluation protocols over a materialized, checksummed Event stream.

Strategy pattern: `protocol_for(mode)` selects the temporal holdout (offline),
event prequential (streaming) or batch prequential (incremental) protocol. The
application runner owns locks, checkpoints, contracts and report persistence;
a protocol only turns events into a report, final model and observer.
"""

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from continual_bench.core.continual import ContinualEvaluation
from continual_bench.core.contracts import Event


def read_events(path):
    with Path(path).open() as f:
        for line in f:
            yield Event(**json.loads(line))


def continual_evaluation(source, factory, settings):
    """Episode probes exist only for sources declaring synthetic generators and boundaries."""
    if not hasattr(source, "generator") or not hasattr(source, "boundaries"):
        return None
    probes = {}
    for task, function in enumerate(source.functions):
        generator = source.generator(function, source.seed + 2_000_000 + task)
        probes[task] = [({str(k): v for k, v in x.items()}, bool(y))
                        for x, y in itertools.islice(generator, settings["probe_samples"])]
    return ContinualEvaluation(factory, probes, source.boundaries + [source.samples],
                               reference_budget=settings["reference_budget"],
                               references_enabled=settings["continual_references"])


@dataclass
class BinaryRun:
    """Everything a protocol needs; built once per attempt by the application runner."""

    config: Any
    source: Any
    events_path: Path
    model_factory: Callable
    detector_factory: Callable
    policy_factory: Callable
    versions: Any
    settings: dict
    started: float
    checkpoint: Callable[[Any], None]
    snapshot: dict | None = None
    transport: Any = None
    manifest: dict = field(default_factory=dict)
    stop_after: int | None = None

    def events(self):
        return read_events(self.events_path)


@dataclass
class Outcome:
    """A finished protocol: report body plus objects needed for tracking and versioning."""

    report: dict
    state: Any
    model: Any
    versions: Any
    observer: Any


@dataclass
class Progress:
    """A bounded, checkpointed stop before the stream ended."""

    observed: int


class BinaryProtocol:
    name: str  # Reported protocol label.

    def execute(self, run: BinaryRun) -> Outcome | Progress:
        raise NotImplementedError


def protocol_for(mode: str, batch_size: int) -> BinaryProtocol:
    from .holdout import TemporalHoldout
    from .prequential import Prequential

    if mode == "offline":
        return TemporalHoldout()
    if mode == "streaming":
        return Prequential("event_prequential", batch_size=1, use_transport=True)
    if mode == "incremental":
        return Prequential("batch_prequential", batch_size=batch_size, use_transport=False)
    raise ValueError(f"unknown evaluation mode {mode}")
