"""Structured benchmark metric tracking with zero-checkpoint enforcement."""

from __future__ import annotations

import csv
import io
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from continual_bench.backends.base import ExecutionBackend, RuntimeSettings
from continual_bench.tracking.mlflow_publisher import (
    TRACKING_URI_PREFIXES,
    CampaignRuns,
    publish_campaign_to_mlflow,
)

T = TypeVar("T")
SeedRows = tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]


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
        self._retrain_events: list[dict[str, Any]] = []
        self._paper_comparison_rows: list[dict[str, Any]] = []
        self._system_metrics: list[dict[str, Any]] = []
        self._artifact_paths: dict[str, str] = {}
        self.mlflow_receipt: dict[str, Any] | None = None
        self._active_runs: CampaignRuns | None = None
        self._seed_artifacts: dict[tuple[str, int], dict[str, str]] = {}

    def start_campaign(self, *, resume: bool = False, recover_evaluations: bool = False) -> None:
        if self.settings.tracking_enabled and self.settings.tracking_uri.startswith(TRACKING_URI_PREFIXES):
            self._active_runs = self._try_mlflow(
                lambda: CampaignRuns(
                    self.settings, self.run_id, self._metadata,
                    resume=resume, recover_evaluations=recover_evaluations,
                )
            )

    def fail_campaign(self) -> None:
        if self._active_runs is not None:
            self._active_runs.fail_campaign()

    def _try_mlflow(self, operation: Callable[[], T]) -> T | None:
        try:
            return operation()
        except Exception:
            logger.warning("MLflow tracking failed for campaign %s", self.run_id, exc_info=True)
            return None

    def run_matrix(
        self,
        seed: int,
        build: Callable[[], T],
        fit_times: Callable[[T], Iterable[float]],
    ) -> T:
        tracking = self._active_runs
        if tracking is None:
            return build()
        run_id = self._try_mlflow(lambda: tracking.start_matrix(seed))
        started = time.perf_counter()
        try:
            result = build()
        except BaseException:
            if run_id is not None:
                tracking.fail(run_id)
            raise
        if run_id is not None:
            elapsed = time.perf_counter() - started

            def publish() -> bool:
                with tracking.publication_slots:
                    tracking.finish_matrix(run_id, seed, elapsed, fit_times(result))
                return True

            if not self._try_mlflow(publish):
                tracking.fail(run_id)
        return result

    def run_seed_scoring(
        self, seed: int, policies: list[str], score: Callable[[], SeedRows]
    ) -> SeedRows:
        tracking = self._active_runs
        if tracking is None:
            return score()
        started = {
            pol: run_id
            for pol in dict.fromkeys(policies)
            if (run_id := self._try_mlflow(lambda: tracking.start_seed(pol, seed))) is not None
        }
        try:
            rows = score()
            for pol, run_id in started.items():
                if (pol, seed) in tracking.completed_seeds:
                    continue
                policy_rows = self._seed_rows(pol, seed, rows)
                paths = self._write_seed_artifacts(pol, seed, policy_rows)
                self._seed_artifacts[pol, seed] = paths
                self._try_mlflow(
                    lambda: tracking.finish_seed(
                        pol, seed, policy_rows, self._local_artifact_paths(paths)
                    )
                )
                if (pol, seed) not in tracking.completed_seeds:
                    tracking.fail(run_id)
        except BaseException:
            for pol, run_id in started.items():
                if (pol, seed) not in tracking.completed_seeds:
                    tracking.fail(run_id)
            raise
        return rows

    def _seed_rows(self, policy: str, seed: int, rows: SeedRows) -> SeedRows:
        def select(group):
            return [
                {"run_id": self.run_id, **row}
                for row in group
                if row.get("policy") == policy and int(row.get("seed", -1)) == seed
            ]

        regimes, summaries, retrains, batches = rows
        return select(regimes), select(summaries), select(retrains), select(batches)

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

    def log_retrain_event(self, seed: int, policy: str, event: dict[str, Any]) -> None:
        if not self.settings.tracking_enabled:
            return
        self._retrain_events.append(
            {
                "run_id": self.run_id,
                "seed": seed,
                "policy": policy,
                **event,
            }
        )

    def log_paper_comparison(self, rows: list[dict[str, Any]]) -> None:
        if not self.settings.tracking_enabled:
            return
        for row in rows:
            self._paper_comparison_rows.append({"run_id": self.run_id, **row})

    def log_system_metrics(self, sample: dict[str, Any]) -> None:
        if not self.settings.tracking_enabled:
            return
        self._system_metrics.append(sample)

    def register_external_artifacts(self, artifacts: dict[str, str]) -> None:
        self._artifact_paths.update(artifacts)

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

        if self._retrain_events:
            written["retrain_events.csv"] = self.backend.write_text(
                self._rel("retrain_events.csv"),
                _rows_to_csv(self._retrain_events),
            )
            written["retrain_events.json"] = self.backend.write_json(
                self._rel("retrain_events.json"),
                self._retrain_events,
            )

        written.update(self._write_per_level_artifacts())
        self._artifact_paths.update(written)
        self._maybe_log_mlflow()
        return written

    def _write_per_level_artifacts(self) -> dict[str, str]:
        """Persist per-policy, per-seed, and per-retrain-version artifacts at end of run."""
        written: dict[str, str] = {}
        policies: list[str] = list(self._metadata.get("policies") or [])
        for row in self._seed_summaries:
            pol = str(row.get("policy", ""))
            if pol and pol not in policies:
                policies.append(pol)

        for pol in policies:
            pol_aggs = [r for r in self._aggregate_rows if r.get("policy") == pol]
            pol_seeds = [r for r in self._seed_summaries if r.get("policy") == pol]
            pol_paper = [r for r in self._paper_comparison_rows if r.get("policy") == pol]
            if pol_aggs:
                key = f"policies/{pol}/policy_summary.json"
                written[key] = self.backend.write_json(self._rel(key), pol_aggs)
            if pol_seeds:
                key = f"policies/{pol}/policy_seed_summaries.csv"
                written[key] = self.backend.write_text(self._rel(key), _rows_to_csv(pol_seeds))
            if pol_paper:
                key = f"policies/{pol}/policy_paper_comparison.json"
                written[key] = self.backend.write_json(self._rel(key), pol_paper)

            seed_ids = sorted({int(r["seed"]) for r in pol_seeds if "seed" in r})
            for seed_val in seed_ids:
                seed_artifacts = self._seed_artifacts.get((pol, seed_val))
                if seed_artifacts is None:
                    seed_artifacts = self._write_seed_artifacts(
                        pol,
                        seed_val,
                        self._seed_rows(
                            pol, seed_val,
                            (self._regime_rows, pol_seeds, self._retrain_events, self._batch_rows),
                        ),
                    )
                written.update(seed_artifacts)
        return written

    def _write_seed_artifacts(self, policy: str, seed: int, rows: SeedRows) -> dict[str, str]:
        regimes, summaries, retrains, batches = rows
        prefix = f"policies/{policy}/seeds/seed-{seed}"
        written: dict[str, str] = {}
        for filename, records in (
            ("seed_summary.json", summaries),
            ("seed_retrain_events.json", retrains),
        ):
            if records:
                key = f"{prefix}/{filename}"
                written[key] = self.backend.write_json(self._rel(key), records)
        for filename, records in (
            ("seed_timeline.csv", batches),
            ("seed_regimes.csv", regimes),
        ):
            if records:
                key = f"{prefix}/{filename}"
                written[key] = self.backend.write_text(self._rel(key), _rows_to_csv(records))
        for event in retrains:
            version_name = str(
                event.get("version_name") or f"{policy}-s{seed}-b{event.get('batch_index', 0)}"
            )
            key = f"{prefix}/retrains/{version_name}.json"
            written[key] = self.backend.write_json(self._rel(key), event)
        return written

    def _local_artifact_paths(self, artifacts: dict[str, str]) -> dict[str, str]:
        # Remote execution backends return S3 URIs and keep a local staging copy.
        staging_dir = getattr(self.backend, "staging_dir", getattr(self.backend, "output_dir", None))
        return {
            name: str(staging_dir / self._rel(name))
            if path.startswith("s3://") and staging_dir is not None else path
            for name, path in artifacts.items()
        }

    def _maybe_log_mlflow(self) -> None:
        if not self.settings.tracking_uri.startswith(TRACKING_URI_PREFIXES):
            return
        self.mlflow_receipt = self._try_mlflow(
            lambda: publish_campaign_to_mlflow(
                settings=self.settings,
                run_id=self.run_id,
                metadata=self._metadata,
                aggregate_rows=self._aggregate_rows,
                seed_summaries=self._seed_summaries,
                regime_rows=self._regime_rows,
                batch_rows=self._batch_rows,
                retrain_events=self._retrain_events,
                paper_comparison_rows=self._paper_comparison_rows,
                system_metrics=self._system_metrics,
                artifact_paths=self._local_artifact_paths(self._artifact_paths),
                active_runs=self._active_runs,
            )
        )
        if self.mlflow_receipt is None:
            self.fail_campaign()

    def assert_no_checkpoints(self) -> None:
        """Raise RuntimeError if any checkpoint files exist when save_model_checkpoints=False."""
        if self.settings.save_model_checkpoints:
            return
        violations = self.backend.verify_no_checkpoints()
        if violations:
            raise RuntimeError(
                f"Model checkpoints are forbidden when save_model_checkpoints=False: {violations}"
            )
