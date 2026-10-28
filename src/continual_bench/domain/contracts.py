"""Service-free domain contracts. Adapters own third-party implementation details."""

from dataclasses import dataclass
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class Event:
    event_id: str
    seq: int
    timestamp: int
    available_at: int
    x: dict[str, Any]
    y: bool
    concept: int | None = None  # Synthetic ground truth; never passed to detector/policy.


class Model(Protocol):
    def predict(self, x: dict[str, Any]) -> float: ...
    def learn(self, x: dict[str, Any], y: bool) -> None: ...


class Detector(Protocol):
    def update(self, error: int) -> bool: ...


@dataclass(frozen=True)
class DecisionContext:
    labeled: int
    window_size: int
    drift: bool


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


class Policy(Protocol):
    def decide(self, context: DecisionContext) -> Decision: ...
    def updated(self, labeled: int) -> None: ...


class Source(Protocol):
    def events(self) -> Iterable[Event]: ...
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
