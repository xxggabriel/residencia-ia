"""Homelab-optimized local/MinIO/S3 execution and storage backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from continual_bench.backends.aws import _parse_s3_uri
from continual_bench.backends.base import CHECKPOINT_EXTENSIONS, RuntimeSettings
from continual_bench.backends.local import LocalBackend, _safe_relative_path

T = TypeVar("T")


class HomelabBackend(LocalBackend):
    """Execution backend tuned for homelab/Docker hosts with MinIO/S3 or local filesystem output."""

    platform_name: str = "homelab"

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        s3_client: Any | None = None,
        staging_dir: Path | None = None,
    ) -> None:
        self.cpu_count = os.cpu_count() or 1
        self._s3_client = s3_client
        raw_output = str(settings.output_dir)
        if raw_output.startswith("s3://"):
            parsed_bucket, parsed_prefix = _parse_s3_uri(raw_output)
            self.s3_bucket: str | None = parsed_bucket
            self.s3_prefix: str = parsed_prefix
            self.output_dir = staging_dir or Path("state/benchmarks/homelab_staging")
            self.settings = settings
        else:
            self.s3_bucket = None
            self.s3_prefix = ""
            super().__init__(settings)
        self.endpoint_url = settings.s3_endpoint_url or os.environ.get("S3_ENDPOINT_URL", "")

    def _get_s3_client(self) -> Any | None:
        if self._s3_client is not None:
            return self._s3_client
        if not self.s3_bucket:
            return None
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            return None
        kwargs: dict[str, Any] = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        self._s3_client = boto3.client("s3", **kwargs)
        return self._s3_client

    def _s3_key(self, rel_path: str) -> str:
        clean_rel = rel_path.lstrip("/")
        if self.s3_prefix:
            return f"{self.s3_prefix}/{clean_rel}"
        return clean_rel

    def write_bytes(self, rel_path: str, data: bytes) -> str:
        target = _safe_relative_path(self.output_dir, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        client = self._get_s3_client()
        if client is not None and self.s3_bucket:
            key = self._s3_key(rel_path)
            client.put_object(Bucket=self.s3_bucket, Key=key, Body=data)
            return f"s3://{self.s3_bucket}/{key}"
        return str(target)

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
        effective_workers = min(max(1, max_workers), self.cpu_count, len(tasks))
        if effective_workers <= 1:
            return [task() for task in tasks]
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            return [future.result() for future in futures]
