"""Live MLflow metrics for an individual AWS Batch matrix attempt."""

import math
import time

from .mlflow_tracking import SageMakerMlflowTracker, sanitize_mlflow_name


class MatrixMlflowRun(SageMakerMlflowTracker):
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
            timeout_seconds=3,
        )
        self.unit = unit
        self.job_id = job_id
        self.attempt_key = attempt_key
        self.pending = []
        self.last_flush = None
        self.step = 0

    @property
    def experiment_name(self):
        return sanitize_mlflow_name(f"{self.project}-matrix")

    @property
    def run_name(self):
        return f"{self.campaign_id}-{self.unit.run_id[:12]}-{self.attempt_key}"

    @property
    def mlflow_tags(self):
        tags = {tag["key"]: tag["value"] for tag in super().mlflow_tags if tag["key"] != "zia_run_key"}
        tags.update({
            "cb-unit-id": self.unit.run_id,
            "cb-batch-job-id": self.job_id,
            "cb-attempt": self.attempt_key,
            "cb-evaluator": self.scope,
            "cb-contract-sha256": self.unit.contract_sha256,
        })
        if self.unit.params.get("source_sha256"):
            tags["cb-source-sha256"] = self.unit.params["source_sha256"]
        return [{"key": key, "value": str(value)} for key, value in tags.items()]

    def metrics(self, step, values, prefix="progress"):
        if not self.mlflow_run_id:
            return
        self.step = max(self.step, step)
        stamp = int(time.time() * 1000)
        self.pending.extend(
            {"key": f"{prefix}/{key}", "value": float(value), "step": step, "timestamp": stamp}
            for key, value in values.items()
            if type(value) in (float, int) and math.isfinite(value)
        )
        now = time.monotonic()
        if self.last_flush is None or now - self.last_flush >= 2 or len(self.pending) >= 500:
            self.flush()

    def flush(self):
        if self.pending:
            self.log_batch(metrics=self.pending)
            self.pending.clear()
            self.last_flush = time.monotonic()

    def finish(self, status="FINISHED"):
        self.flush()
        return super().finish(status)
