"""Live MLflow metrics for a cloud matrix case."""

import math
import time

from continual_bench.infrastructure.tracking import AsyncMetricWriter

from .benchmark_tracking import BenchmarkMlflowRun
from .mlflow_tracking import sanitize_mlflow_name

FLUSH_SECONDS = 2
FLUSH_POINTS = 500


class MatrixMlflowRun(BenchmarkMlflowRun):
    def __init__(self, unit, attempt_key, job_id, tracking_uri, log):
        labels = unit.labels
        super().__init__(
            tracking_uri=tracking_uri,
            dataset=labels["dataset"],
            policy=labels.get("policy") or "none",
            model_name=labels.get("model") or "default",
            scope=unit.params["evaluator"],
            campaign_id=unit.campaign_id,
            run_id=f"{unit.run_id}-{attempt_key}",
            job_name=unit.job_name,
            git_sha=unit.contract["release"]["git_sha"],
            logger=log,
            timeout_seconds=15,
        )
        self.unit = unit
        self.job_id = job_id
        self.attempt_key = attempt_key
        self.pending = []
        self.last_flush = None
        self.step = 0

    @property
    def experiment_name(self):
        short = self.short_sha
        prefix = f"{short}-" if short else ""
        policy = self.policy or "none"
        raw = f"{prefix}{self.project}-matrix-{policy}" if self.project else f"{prefix}matrix-{policy}"
        return sanitize_mlflow_name(raw)

    @property
    def logical_run_id(self):
        return self.unit.run_id

    @property
    def run_name(self):
        detector = self.unit.labels.get("detector")
        strategy = f"{self.policy}-{detector}" if detector and detector != "none" else self.policy
        mode = self.unit.labels.get("mode")
        parts = [
            self.campaign_id,
            self.unit.run_id[:8],
            self.dataset,
            self.model_name,
            *([mode] if mode else []),
            strategy,
        ]
        base = "-".join(str(part) for part in parts if part)
        return sanitize_mlflow_name(base if self.spot else f"{base}-{self.attempt_key}")

    @property
    def mlflow_tags(self):
        tags = {tag["key"]: tag["value"] for tag in super().mlflow_tags}
        tags.update(
            {
                "cb-unit-id": self.unit.run_id,
                "cb-evaluator": self.scope,
                "cb-contract-sha256": self.unit.contract_sha256,
            }
        )
        if not self.spot:
            tags.update({"cb-batch-job-id": self.job_id, "cb-attempt": self.attempt_key})
        if self.unit.params.get("source_sha256"):
            tags["cb-source-sha256"] = self.unit.params["source_sha256"]
        return [{"key": key, "value": str(value)} for key, value in tags.items()]

    def metrics(self, step, values, prefix="progress"):
        if not self.mlflow_run_id and not self.spot:
            return
        self.step = max(self.step, step)
        if (self.unit.params.get("config") or {}).get("artifact_mode") == "metrics_only":
            # Restarted evaluation has new clocks and measurements. Keep each attempt's
            # live series independent while the final report retains the logical run.
            prefix = f"attempts/{self.attempt_key}/{prefix}"
        stamp = int(time.time() * 1000)
        points = [
            {"key": f"{prefix}/{key}", "value": float(value), "step": step, "timestamp": stamp}
            for key, value in values.items()
            if type(value) in (float, int) and math.isfinite(value)
        ]
        if self.native_logger is not None:
            # MLflow's async queue keeps every step and sends it from its own thread pool.
            self.log_progress_metrics(points)
            return
        if not self.spot:
            # Every step is kept (no coalescing); a daemon thread sends them off the training thread.
            if self._live_writer is None:
                self._live_writer = AsyncMetricWriter(
                    self.log_progress_metrics,
                    FLUSH_SECONDS,
                    coalesce=False,
                    max_pending=FLUSH_POINTS,
                )
            self._live_writer.submit(points)
            return
        # Spot reconciliation reads remote histories in order, so it stays on the caller thread.
        self.pending.extend(points)
        now = time.monotonic()
        due = self.last_flush is None or now - self.last_flush >= FLUSH_SECONDS
        if due or len(self.pending) >= FLUSH_POINTS:
            self.flush()

    def flush(self):
        if self.pending:
            self.log_progress_metrics(self.pending)
            self.pending.clear()
            self.last_flush = time.monotonic()
        return super().flush()

    def finish(self, status="FINISHED"):
        self.flush()
        return super().finish(status)
