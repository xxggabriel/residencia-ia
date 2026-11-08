"""Shared dataset helpers and adapters between binary Events and typed TaskSamples."""

import hashlib
import json

from continual_bench.core.contracts import TaskSample


def stable_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_bounds(samples, seed, label_delay, epoch):
    if not 1 <= samples <= 100_000 or not 0 <= seed < 2**31 or label_delay < 0 or epoch < 0:
        raise ValueError("invalid bounded source parameters")


class EventSamples:
    """Adapter: present a binary Event stream as an explicit two-class TaskSample stream."""

    def __init__(self, source):
        self.source = source

    def samples_iter(self):
        for event in self.source.events():
            yield TaskSample(event.event_id, event.seq, event.timestamp, event.available_at,
                             event.x, "1" if event.y else "0", task_id=event.concept)

    def manifest(self):
        return {**self.source.manifest(), "task": "multiclass", "labels": ["0", "1"],
                "target_conversion": "explicit binary boolean to class strings, no information loss"}


def as_sample_source(source):
    """Typed evaluators consume TaskSamples; binary-only sources are adapted losslessly."""
    return source if hasattr(source, "samples_iter") else EventSamples(source)
