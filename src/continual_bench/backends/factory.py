"""Backend factory for resolving execution and storage backends by platform."""

from __future__ import annotations

from continual_bench.backends.aws import AwsBackend
from continual_bench.backends.base import ExecutionBackend, RuntimeSettings
from continual_bench.backends.homelab import HomelabBackend
from continual_bench.backends.local import LocalBackend


def resolve_backend(
    platform: str | None = None,
    settings: RuntimeSettings | None = None,
) -> ExecutionBackend:
    """Resolve an ExecutionBackend instance for the requested or configured platform."""
    resolved_settings = settings if settings is not None else RuntimeSettings.from_env()
    target = (platform or resolved_settings.platform).strip().lower()
    if target == "local":
        return LocalBackend(resolved_settings)
    if target == "homelab":
        return HomelabBackend(resolved_settings)
    if target == "aws":
        return AwsBackend(resolved_settings)
    raise ValueError(f"Unsupported backend platform: {target!r}")
