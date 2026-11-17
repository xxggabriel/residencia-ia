"""Publish reports and grouped training evidence to the managed MLflow App."""

import json
import math
import os
from pathlib import Path

from continual_bench.infrastructure.benchmark_tracking import benchmark_metric_points
from continual_bench.infrastructure.metric_tracking import metric_points
from continual_bench.infrastructure.object_store import split_uri
from continual_bench.infrastructure.observation_tracking import observation_tags, parent_observation_points
from continual_bench.infrastructure.training_tracking import HISTORY_FILE, training_checkpoint_paths

from .mlflow_tracking import (
    SageMakerMlflowTracker,
    sanitize_mlflow_name,
    upload_directory_to_mlflow_artifacts,
)


def aws_metric_points(report, headline=None):
    """Extend the common report coordinates with the managed run's live and supplemental evidence."""
    points = benchmark_metric_points(report)
    points.extend(parent_observation_points(report))
    if (tracking := report.get("tracking")) and tracking.get("version"):
        points.extend((key, step, value) for key, step, value in metric_points(tracking) if value is not None)
    points.extend(
        (f"final/{key}", 0, value)
        for key, value in (headline or {}).items()
        if type(value) in (int, float) and math.isfinite(value)
    )
    items = [
        {"key": sanitize_mlflow_name(key), "value": float(value), "step": step, "timestamp": 0}
        for key, step, value in points
    ]
    for seq, sample in enumerate((report.get("system_metrics") or {}).get("samples", [])):
        items.extend(
            {
                "key": sanitize_mlflow_name(f"system/{key}"),
                "value": float(value),
                "step": seq,
                "timestamp": sample["timestamp_ms"],
            }
            for key, value in sample["values"].items()
            if type(value) in (int, float) and math.isfinite(value)
        )
    return items


class BenchmarkMlflowRun(SageMakerMlflowTracker):
    run_key_tag = "cb_training_run_key"

    def publish_training_evidence(self, report, directory, *, store=None, headline=None, outputs_uri=None):
        """Return a receipt only after all required metrics, children, and artifacts are stored."""
        directory = Path(directory)
        if "training_history" in report:
            if json.loads((directory / HISTORY_FILE).read_bytes()) != {
                "version": 1,
                "fits": report["training_history"],
            }:
                raise ValueError("retained training history differs from the benchmark report")
            if not (directory / "artifacts" / "data" / "index.json").is_file():
                raise ValueError("benchmark artifact index is missing")
            training_checkpoint_paths(directory, report)
        run_id = self.ensure_run()
        if run_id is None:
            raise RuntimeError("MLflow did not create the benchmark run")
        tags = {"benchmark_complete": "false", **observation_tags(report)}
        if not self.log_batch(tags=[{"key": key, "value": val} for key, val in tags.items()]):
            raise RuntimeError("MLflow did not mark benchmark publication started")
        if not self.log_completed_metrics(aws_metric_points(report, headline)):
            raise RuntimeError("MLflow did not publish benchmark metrics")
        child_runs = self.publish_training_rounds(report)
        if store is None and self.artifact_uri and self.artifact_uri.startswith("s3://"):
            from .s3 import S3ObjectStore

            bucket, _ = split_uri(self.artifact_uri)
            store = S3ObjectStore(bucket, kms_key_id=os.environ.get("BENCH_KMS_KEY_ID"))
        self.stage_observation_artifacts(directory, report, child_runs, store)
        artifacts = upload_directory_to_mlflow_artifacts(
            store,
            directory,
            self.artifact_uri,
            immutable=self.spot,
        )
        if not artifacts["files"]:
            raise RuntimeError("MLflow did not publish benchmark artifacts")
        tags = [
            {"key": "cb-training-rounds", "value": str(len(child_runs))},
            {"key": "cb-evidence-status", "value": "complete"},
        ]
        if outputs_uri:
            tags.append({"key": "cb-outputs-uri", "value": outputs_uri[:500]})
        if not self.log_batch(tags=tags):
            raise RuntimeError("MLflow did not mark benchmark evidence complete")
        return {
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": self.experiment_id,
            "artifact_uri": self.artifact_uri,
            "artifacts": artifacts,
            "child_runs": child_runs,
        }

    def finish_publication(self):
        if not self.finish() or not self.log_batch(tags=[{"key": "benchmark_complete", "value": "true"}]):
            raise RuntimeError("MLflow did not finalize the benchmark")
