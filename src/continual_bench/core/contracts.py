"""Service-free domain contracts. Adapters own third-party implementation details."""

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class Event:
    """Binary prequential record consumed by the drift/retraining engine."""

    event_id: str
    seq: int
    timestamp: int
    available_at: int
    x: dict[str, Any]
    y: bool
    concept: int | None = None  # Synthetic ground truth; never passed to detector/policy.


@dataclass(frozen=True)
class TaskSample:
    """Task-agnostic record: class string for classification, float for regression.

    Binary Event is deliberately not reused for multiclass or regression targets.
    """

    sample_id: str
    seq: int
    timestamp: int
    available_at: int
    x: dict[str, Any]
    y: str | float
    task_id: int | None = None
    split: str = "train"
    group: str | None = None


def event_json(event: Event) -> str:
    """Canonical JSONL encoding; its SHA-256 is the materialized dataset version."""
    return json.dumps(asdict(event), sort_keys=True, separators=(",", ":"))


class Model(Protocol):
    """Binary probability learner used by the prequential engine."""

    def predict(self, x: dict[str, Any]) -> float: ...
    def learn(self, x: dict[str, Any], y: bool) -> None: ...


class Estimator(Protocol):
    """Task-generic learner: label/None for classification, finite float for regression."""

    def predict_one(self, x: dict[str, Any]) -> Any: ...
    def learn_one(self, x: dict[str, Any], y: Any) -> None: ...


class Detector(Protocol):
    def update(self, error: int) -> bool: ...


@dataclass(frozen=True)
class DecisionContext:
    labeled: int
    window_size: int
    drift: bool
    recent_error: float | None = None
    monitored: int = 0
    batch: int | None = None  # Index of the batch just observed (batch policies only).


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


class Policy(Protocol):
    def decide(self, context: DecisionContext) -> Decision: ...
    def updated(self, labeled: int) -> None: ...


class Source(Protocol):
    """Binary stream: ordered Events plus a provenance manifest."""

    def events(self) -> Iterable[Event]: ...
    def manifest(self) -> dict: ...


class SampleSource(Protocol):
    """Typed stream: ordered TaskSamples; manifest()["task"] declares the target type."""

    def samples_iter(self) -> Iterable[TaskSample]: ...
    def manifest(self) -> dict: ...


class Checkpoints(Protocol):
    def load(self, identity: str) -> Any | None: ...
    def save(self, identity: str, state: Any) -> None: ...


class Tracker(Protocol):
    def publish(self, identity: str, config: dict, report: dict, files: list[str]) -> str: ...


class ModelVersions(Protocol):
    def record(self, model: Model, training: dict) -> str: ...


class Objects(Protocol):
    def upload(self, key: str, path: str) -> str: ...
