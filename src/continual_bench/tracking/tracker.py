"""Structured benchmark metric tracking with zero-checkpoint enforcement."""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

from continual_bench.backends.base import ExecutionBackend, RuntimeSettings

logger = logging.getLogger(__name__)


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _rows_to_parquet_bytes(rows: list[dict[str, Any]]) -> bytes | None:
    if not rows:
        return None
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    except Exception as exc:  # pragma: no cover - optional parquet engine fallback
        logger.warning("Skipping Parquet serialization: %s", exc)
        return None


class BenchmarkTracker:
    """Records structured benchmark metrics and flushes them through an ExecutionBackend."""

    def __init__(
        self,
        backend: ExecutionBackend,
        run_id: str,
        settings: RuntimeSettings,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.settings = settings
        self._metadata: dict[str, Any] = {
            "run_id": run_id,
            "platform": settings.platform,
            "save_model_checkpoints": settings.save_model_checkpoints,
            "tracking_enabled": settings.tracking_enabled,
        }
        self._batch_rows: list[dict[str, Any]] = []
        self._regime_rows: list[dict[str, Any]] = []
        self._seed_summaries: list[dict[str, Any]] = []
        self._aggregate_rows: list[dict[str, Any]] = []

    def _rel(self, filename: str) -> str:
        return f"{self.run_id}/{filename}"

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        if not self.settings.tracking_enabled:
            return
        self._metadata.update(metadata)

    def log_batch(
        self,
        seed: int,
        policy: str,
        regime: str | float,
        batch_record: dict[str, Any],
    ) -> None:
        if not self.settings.tracking_enabled:
            return
        row = {
            "run_id": self.run_id,
            "seed": seed,
            "policy": policy,
            "regime": regime,
            **batch_record,
        }
        self._batch_rows.append(row)

    def log_regime(
        self,
        seed: int,
        policy: str,
        regime_record: dict[str, Any],
    ) -> None:
        if not self.settings.tracking_enabled:
            return
        row = {
            "run_id": self.run_id,
            "seed": seed,
            "policy": policy,
            **regime_record,
        }
        self._regime_rows.append(row)

    def log_seed_summary(
        self,
        seed: int,
        policy: str,
        summary: dict[str, Any],
    ) -> None:
        if not self.settings.tracking_enabled:
            return
        row = {
            "run_id": self.run_id,
            "seed": seed,
            "policy": policy,
            **summary,
        }
        self._seed_summaries.append(row)

    def log_aggregate_summary(self, rows: list[dict[str, Any]]) -> None:
        if not self.settings.tracking_enabled:
            return
        for row in rows:
            self._aggregate_rows.append({"run_id": self.run_id, **row})

    def flush(self) -> dict[str, str]:
        """Write buffered metrics and metadata via backend in JSON, CSV, and Parquet."""
        if not self.settings.tracking_enabled:
            return {}

        written: dict[str, str] = {}
        written["metadata.json"] = self.backend.write_json(self._rel("metadata.json"), self._metadata)

        if self._batch_rows:
            written["batch_metrics.csv"] = self.backend.write_text(
                self._rel("batch_metrics.csv"),
                _rows_to_csv(self._batch_rows),
            )
            parquet_bytes = _rows_to_parquet_bytes(self._batch_rows)
            if parquet_bytes is not None:
                written["batch_metrics.parquet"] = self.backend.write_bytes(
                    self._rel("batch_metrics.parquet"),
                    parquet_bytes,
                )

        if self._regime_rows:
            written["regime_metrics.csv"] = self.backend.write_text(
                self._rel("regime_metrics.csv"),
                _rows_to_csv(self._regime_rows),
            )
            parquet_bytes = _rows_to_parquet_bytes(self._regime_rows)
            if parquet_bytes is not None:
                written["regime_metrics.parquet"] = self.backend.write_bytes(
                    self._rel("regime_metrics.parquet"),
                    parquet_bytes,
                )

        if self._seed_summaries:
            written["seed_summaries.csv"] = self.backend.write_text(
                self._rel("seed_summaries.csv"),
                _rows_to_csv(self._seed_summaries),
            )
            written["seed_summaries.json"] = self.backend.write_json(
                self._rel("seed_summaries.json"),
                self._seed_summaries,
            )

        if self._aggregate_rows:
            written["aggregate_summary.csv"] = self.backend.write_text(
                self._rel("aggregate_summary.csv"),
                _rows_to_csv(self._aggregate_rows),
            )
            written["aggregate_summary.json"] = self.backend.write_json(
                self._rel("aggregate_summary.json"),
                self._aggregate_rows,
            )

        self._maybe_log_mlflow()
        return written

    def _maybe_log_mlflow(self) -> None:
        uri = self.settings.tracking_uri.strip()
        if not uri or not uri.startswith(("http://", "https://", "mlflow")):
            return
        try:
            import mlflow  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("MLflow tracking_uri configured (%s) but mlflow is not installed.", uri)
            return

        try:
            mlflow.set_tracking_uri(uri)
            with mlflow.start_run(run_name=self.run_id):
                mlflow.log_params(
                    {k: str(v)[:250] for k, v in self._metadata.items() if v is not None}
                )
                for row in self._aggregate_rows:
                    policy = str(row.get("policy", "aggregate"))
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            mlflow.log_metric(f"{policy}.{key}", float(value))
        except Exception as exc:
            logger.warning("MLflow logging failed gracefully for %s: %s", uri, exc)

    def assert_no_checkpoints(self) -> None:
        """Raise RuntimeError if any checkpoint files exist when save_model_checkpoints=False."""
        if self.settings.save_model_checkpoints:
            return
        violations = self.backend.verify_no_checkpoints()
        if violations:
            raise RuntimeError(
                f"Model checkpoints are forbidden when save_model_checkpoints=False: {violations}"
            )
