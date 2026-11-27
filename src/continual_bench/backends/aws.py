"""AWS adapter backend supporting S3 persistence with a local staging mirror."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from continual_bench.backends.base import (
    CHECKPOINT_EXTENSIONS,
    ExecutionBackend,
    RuntimeSettings,
)
from continual_bench.backends.local import _safe_relative_path, _serialize_checkpoint

T = TypeVar("T")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri[len("s3://") :].strip("/")
    if not without_scheme:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix.strip("/")


class AwsBackend(ExecutionBackend):
    """Platform-agnostic AWS backend using S3 (when configured) and a local staging mirror."""

    platform_name: str = "aws"

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        s3_client: Any | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str = "",
        staging_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self._s3_client = s3_client

        raw_output = str(settings.output_dir)
        if raw_output.startswith("s3://"):
            parsed_bucket, parsed_prefix = _parse_s3_uri(raw_output)
            self.s3_bucket = s3_bucket or parsed_bucket
            self.s3_prefix = s3_prefix.strip("/") or parsed_prefix
            self.staging_dir = staging_dir or Path("state/benchmarks/aws_staging")
        else:
            self.s3_bucket = s3_bucket or os.environ.get("BENCH_S3_BUCKET") or None
            self.s3_prefix = s3_prefix.strip("/")
            self.staging_dir = staging_dir or Path(settings.output_dir)

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
        endpoint = self.settings.s3_endpoint_url or os.environ.get("S3_ENDPOINT_URL", "")
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        self._s3_client = boto3.client("s3", **kwargs)
        return self._s3_client

    def _s3_key(self, rel_path: str) -> str:
        clean_rel = rel_path.lstrip("/")
        if self.s3_prefix:
            return f"{self.s3_prefix}/{clean_rel}"
        return clean_rel

    def write_bytes(self, rel_path: str, data: bytes) -> str:
        target = _safe_relative_path(self.staging_dir, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        client = self._get_s3_client()
        if client is not None and self.s3_bucket:
            key = self._s3_key(rel_path)
            client.put_object(Bucket=self.s3_bucket, Key=key, Body=data)
            return f"s3://{self.s3_bucket}/{key}"
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
        if not self.staging_dir.exists():
            return []
        violations: list[str] = []
        for path in sorted(self.staging_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS:
                violations.append(path.relative_to(self.staging_dir).as_posix())
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
