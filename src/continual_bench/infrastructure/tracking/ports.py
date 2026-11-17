"""The tracking port: the only MLflow operations the publication algorithms need."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MetricPoint:
    key: str
    value: float
    step: int = 0
    timestamp: int = 0


@dataclass
class RunRecord:
    run_id: str
    status: str = "RUNNING"
    artifact_uri: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    metric_keys: set[str] = field(default_factory=set)


@runtime_checkable
class TrackingBackend(Protocol):
    """Transport-neutral MLflow operations (Strategy implemented per environment)."""

    #: Upper bound of concurrent requests this transport tolerates.
    concurrency: int

    def search_runs(self, experiment_id: str, filter_string: str) -> list[RunRecord]: ...

    def create_run(
        self, experiment_id: str, tags: dict[str, str], run_name: str | None = None,
    ) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord: ...

    def get_metric_history(self, run_id: str, key: str) -> list[tuple[int, float]]: ...

    def log_batch(
        self,
        run_id: str,
        metrics: list[MetricPoint] = (),
        params: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None: ...

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None: ...
