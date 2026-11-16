"""Track a SageMaker Processing job, Training job, or individual AMT trial."""

import logging
import os
from contextlib import contextmanager

from .mlflow_tracking import SageMakerMlflowTracker, sanitize_mlflow_name


class NativeJobRun(SageMakerMlflowTracker):
    def __init__(self, uri, execution, stage, job_name):
        super().__init__(
            tracking_uri=uri, dataset="social", policy=stage, model_name="multinomial-logistic",
            scope=os.environ.get("BENCH_NATIVE_SCOPE", "smoke"),
            campaign_id=execution, run_id=job_name, job_name=job_name, timeout_seconds=3,
        )
        self.execution = execution
        self.stage = stage

    @property
    def experiment_name(self):
        return sanitize_mlflow_name(f"{self.project}-native")

    @property
    def run_name(self):
        return self.job_name

    @property
    def mlflow_tags(self):
        tags = [tag for tag in super().mlflow_tags if tag["key"] != "zia_run_key"]
        tags.extend([
            {"key": "cb-sagemaker-job", "value": self.job_name},
            {"key": "cb-pipeline-execution-id", "value": self.execution},
            {"key": "cb-native-stage", "value": self.stage},
        ])
        if arn := os.environ.get("TRAINING_JOB_ARN"):
            tags.append({"key": "cb-sagemaker-job-arn", "value": arn})
        return tags


@contextmanager
def track_native_job(tracking_uri, execution, stage, job_name):
    tracker = NativeJobRun(tracking_uri, execution, stage, job_name)
    if not tracker.start()["run_id"]:
        raise RuntimeError("SageMaker MLflow did not create a run for this job")
    try:
        yield tracker
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            tracker.finish("KILLED")
        else:
            tracker.mark_failed(str(error))
        raise
    else:
        if not tracker.finish():
            logging.getLogger(__name__).warning("Could not finalize MLflow run %s", tracker.mlflow_run_id)
