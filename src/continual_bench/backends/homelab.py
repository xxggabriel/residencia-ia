"""Homelab-optimized local/shared filesystem execution backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from continual_bench.backends.base import RuntimeSettings
from continual_bench.backends.local import LocalBackend

T = TypeVar("T")


class HomelabBackend(LocalBackend):
    """Execution backend tuned for homelab multi-core hosts with shared filesystem output."""

    platform_name: str = "homelab"

    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__(settings)
        self.cpu_count = os.cpu_count() or 1

    def run_tasks(self, tasks: list[Callable[[], T]], max_workers: int = 1) -> list[T]:
        if not tasks:
            return []
        effective_workers = min(max(1, max_workers), self.cpu_count, len(tasks))
        if effective_workers <= 1:
            return [task() for task in tasks]
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            return [future.result() for future in futures]
