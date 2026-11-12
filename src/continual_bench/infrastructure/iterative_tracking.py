"""MLflow publication of iterative runs with the SageMaker tracker's metric contract.

Mirrors `aws.mlflow_tracking.SageMakerMlflowTracker`: per-epoch metrics on one global step axis
(`train:*`, `validation:*`, `best_epoch`, `patience_consumed`, `early_stopped`, ...), per-round
`training_round/*`, and at the end `summary/*`, `<scope>/windows/*`, `class_metrics/*`, `training/*`,
costs, `system/*`, plots/data artifacts, parameters, tags and dataset lineage. Transient per-epoch
failures never abort training; they are counted and reported.
"""

import math
import re
import time

from .metric_tracking import log_metric_points
from .run_artifacts import publish_run_artifacts

EXPERIMENT = "continual-bench/semana09/iterative-catalog-v1"


def sanitize(name):
    return re.sub(r"[^0-9a-zA-Z_\-. /:]", "_", str(name))[:250]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class IterativeMlflowTracker:
    def __init__(self, uri, *, dataset, task, model, campaign, git_sha="uncommitted", client=None):
        from mlflow.tracking import MlflowClient

        self.client = client or MlflowClient(tracking_uri=uri)
        self.dataset, self.task, self.model = dataset, task, model
        self.campaign, self.git_sha = campaign, git_sha
        self.experiment_id = self.run_id = None
        self.global_step = self.round = 0
        self.errors = []

    def _guard(self, action, call):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - tracker failures must not abort training
            self.errors.append(f"{action}: {type(error).__name__}: {str(error)[:300]}")
            return None

    def start(self, params, samples):
        from mlflow.entities import Dataset, DatasetInput, InputTag, Param, RunTag

        client = self.client
        experiment = client.get_experiment_by_name(EXPERIMENT)
        self.experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
        short = self.git_sha[:7]
        key = f"{self.campaign}-{self.dataset}"
        tags = {"cb-dataset": self.dataset, "cb-policy": "iterative", "cb-scope": "full",
                "cb-model": self.model, "cb-run-id": key, "cb-campaign-id": self.campaign,
                "cb-git-sha": self.git_sha, "zia_run_key": key, "mlflow.source.type": "JOB",
                "mlflow.source.name": "continual-bench iterative-worker",
                "mlflow.runName": f"{self.dataset}-iterative-full-{short}-{self.campaign}"}
        run = client.create_run(self.experiment_id, tags=tags)
        self.run_id = run.info.run_id
        raw = {"dataset": self.dataset, "task": self.task, "model": self.model, "samples": samples, **params}
        client.log_batch(self.run_id, params=[Param(sanitize(k), str(v)[:500]) for k, v in raw.items()
                                              if v is not None], synchronous=True)
        dataset = Dataset(name=self.dataset, digest=f"{self.dataset}-{samples}"[:32], source_type="local",
                          source="{}", profile=f'{{"num_rows": {samples}}}')
        client.log_inputs(self.run_id, datasets=[DatasetInput(
            dataset, tags=[InputTag("mlflow.data.context", "iterative-continual")])])
        client.log_batch(self.run_id, tags=[RunTag("benchmark_complete", "false")], synchronous=True)
        return self.run_id

    def _batch(self, points, step=None):
        from mlflow.entities import Metric

        now = int(time.time() * 1000)
        metrics = [Metric(sanitize(k), float(v), now, int(step if step is not None else s))
                   for k, v, s in points if _finite(v) or isinstance(v, bool)]
        for offset in range(0, len(metrics), 500):
            self.client.log_batch(self.run_id, metrics=metrics[offset:offset + 500], synchronous=True)

    def log_epoch(self, metrics):
        self.global_step += 1
        step = self.global_step
        self._guard("log_epoch", lambda: self._batch(
            [(k, float(v), step) for k, v in metrics.items() if _finite(v) or isinstance(v, bool)]))

    def finalize_round(self, summary):
        self.round += 1
        step = self.round
        self._guard("finalize_round", lambda: self._batch(
            [(f"training_round/{k}", float(v), step) for k, v in summary.items()
             if _finite(v) or isinstance(v, bool)]))

    def publish(self, report, directory):
        """Final points, artifacts and run termination; returns a receipt dict."""
        from mlflow.entities import RunTag

        points = []
        for key, value in (report.get("summary") or {}).items():
            points.append((f"summary/{key}", 0, value))
        for window in report.get("windows") or []:
            scope = window.get("scope", "prequential")
            for key, value in window.items():
                if key not in {"scope", "end_seq"}:
                    points.append((f"{scope}/windows/{key}", int(window["end_seq"]), value))
        for label, row in (report.get("class_metrics") or {}).items():
            points.extend((f"class_metrics/{label}/{key}", 0, value) for key, value in row.items())
        for version in report.get("model_versions") or []:
            points.extend((f"training/{key}", int(version["end_seq"]), version[key])
                          for key in ("train_samples", "train_seconds", "iterations", "bytes")
                          if key in version)
        costs = report.get("costs") or {}
        points.extend(("costs/total/" + k, 0, v) for k, v in (costs.get("metrics") or {}).items())
        aws = (costs.get("aws") or {}).get("metrics") or {}
        points.extend(("costs/aws/" + k, 0, v) for k, v in aws.items())
        points = [(k, s, v) for k, s, v in points if _finite(v)]
        log_metric_points(self.client, self.run_id, points)
        artifacts = publish_run_artifacts(self.client, self.run_id, directory, report)
        self.client.log_artifact(self.run_id, str(directory / "result.json"))
        model = directory / "model.pkl"
        if model.exists():
            self.client.log_artifact(self.run_id, str(model), artifact_path="models")
        note = (f"### Continual-Bench iterative run: `{self.dataset}`\n- Campaign: `{self.campaign}`\n"
                f"- Model: `{self.model}`\n- Prequential accuracy: "
                f"`{(report.get('summary') or {}).get('accuracy', 'n/a')}`\n"
                f"- Updates: `{len(report.get('windows') or [])}`")
        self.client.log_batch(self.run_id, tags=[RunTag("mlflow.note.content", note[:4500]),
                                                 RunTag("benchmark_complete", "true"),
                                                 RunTag("zia_artifacts_status", "complete")],
                                synchronous=True)
        self.client.set_terminated(self.run_id, "FINISHED")
        return {"mlflow_run_id": self.run_id, "mlflow_experiment_id": self.experiment_id,
                "metric_points": len(points), **artifacts, "tracker_errors": list(self.errors)}

    def fail(self, message):
        from mlflow.entities import RunTag

        if self.run_id:
            self._guard("fail", lambda: (
                self.client.log_batch(self.run_id, tags=[RunTag("cb-error", str(message)[:1000])],
                                      synchronous=True),
                self.client.set_terminated(self.run_id, "FAILED")))
