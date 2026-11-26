"""Local filesystem storage and thread-pool execution backend."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from continual_bench.backends.base import (
    CHECKPOINT_EXTENSIONS,
    ExecutionBackend,
    RuntimeSettings,
)

T = TypeVar("T")


def _safe_relative_path(root: Path, rel_path: str) -> Path:
    candidate = (root / rel_path).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Path {rel_path!r} escapes output directory {resolved_root}")
    return candidate


def _serialize_checkpoint(model_obj: Any) -> bytes:
    if isinstance(model_obj, (bytes, bytearray)):
        return bytes(model_obj)
    return pickle.dumps(model_obj)


class LocalBackend(ExecutionBackend):
    """Filesystem-backed execution backend for local runs and unit tests."""

    platform_name: str = "local"

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.output_dir = Path(settings.output_dir)

    def write_bytes(self, rel_path: str, data: bytes) -> str:
        target = _safe_relative_path(self.output_dir, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    def write_text(self, rel_path: str, text: str) -> str:
        return self.write_bytes(rel_path, text.encode("utf-8"))

    def write_json(self, rel_path: str, payload: Any) -> str:
        serialized = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        return self.write_text(rel_path, serialized)

    def save_model_checkpoint(self, rel_path: str, model_obj: Any) -> str | None:
        if not self.settings.save_model_checkpoints:
            return None
        return self.write_bytes(rel_path, _serialize_checkpoint(model_obj))

    def verify_no_checkpoints(self) -> list[str]:
        if not self.output_dir.exists():
            return []
        violations: list[str] = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS:
                violations.append(path.relative_to(self.output_dir).as_posix())
        return violations

    def run_tasks(self, tasks: list[Callable[[], T]], max_workers: int = 1) -> list[T]:
        if not tasks:
            return []
        if max_workers <= 1 or len(tasks) == 1:
            return [task() for task in tasks]
        workers = min(max_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            return [future.result() for future in futures]
