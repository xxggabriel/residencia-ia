"""Platform-agnostic storage and execution backends for continual_bench."""

from continual_bench.backends.aws import AwsBackend
from continual_bench.backends.base import (
    CHECKPOINT_EXTENSIONS,
    VALID_PLATFORMS,
    ExecutionBackend,
    RuntimeSettings,
    StorageBackend,
)
from continual_bench.backends.factory import resolve_backend
from continual_bench.backends.homelab import HomelabBackend
from continual_bench.backends.local import LocalBackend

__all__ = [
    "AwsBackend",
    "CHECKPOINT_EXTENSIONS",
    "ExecutionBackend",
    "HomelabBackend",
    "LocalBackend",
    "RuntimeSettings",
    "StorageBackend",
    "VALID_PLATFORMS",
    "resolve_backend",
]
