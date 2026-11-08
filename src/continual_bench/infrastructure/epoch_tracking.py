"""Publish epoch evidence under a separate immutable namespace and MLflow schema."""

import hashlib
import json
import os
from pathlib import Path

from .catalog import put_immutable
from .run_artifacts import publish_run_artifacts
from .services import S3Objects
from .storage import atomic_write, publication_lock


def publish_epochs(report, root):
    from mlflow.entities import Param, RunTag
    from mlflow.tracking import MlflowClient

    key = report["run_key"]
    directory = Path(root) / key
    code = report["contract"]["environment"]["code_sha256"]
    files = {}
    if hashlib.sha256((directory / "samples.jsonl").read_bytes()).hexdigest() != \
            report["contract"]["samples_sha256"]:
        raise ValueError("epoch sample artifact differs from admitted report checksum")
    if json.loads((directory / "epoch-report.json").read_text()) != report:
        raise ValueError("epoch report artifact differs from evaluated result")
    objects = S3Objects()
    with publication_lock(root, "epoch-" + key):
        for filename in ("samples.jsonl", "epoch-report.json"):
            path = directory / filename
            uri = put_immutable(objects, f"epoch-benchmarks/{code}/{key}/{filename}", path)
            files[filename] = {"uri": uri, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        name = "continual-bench/semana09/dataset-epochs-v1"
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        matches = client.search_runs([experiment_id], f"tags.epoch_run_key = '{key}'")
        if len(matches) > 1:
            raise ValueError("duplicate epoch tracking run; explicit reconciliation required")
        run = matches[0] if matches else client.create_run(experiment_id, tags={"epoch_run_key": key})
        run_id = run.info.run_id
        config = report["contract"]["config"]
        params = {**config, "code_sha256": code, "sample_sha256": report["contract"]["samples_sha256"],
                  "evaluation": report["scope"]}
        client.log_batch(run_id, params=[Param(k, str(v)) for k, v in params.items()],
                         tags=[RunTag("dataset", report["dataset"])], synchronous=True)
        from .metric_tracking import log_metric_points

        points = [(name, epoch["epoch"], value)
                  for epoch in report["epochs"] for name, value in epoch["metrics"].items()]
        points.extend(("costs/" + name, epoch["epoch"], value)
                      for epoch in report["epochs"] for name, value in epoch["costs"]["metrics"].items())
        points.extend(("costs/total/" + name, 0, value)
                      for name, value in report["costs"]["metrics"].items())
        points.extend(("compute/" + name, epoch["epoch"], epoch[name])
                      for epoch in report["epochs"] for name in ("training_seconds", "prediction_seconds"))
        log_metric_points(client, run_id, points)
        for filename in files:
            client.log_artifact(run_id, str(directory / filename))
        artifacts = publish_run_artifacts(client, run_id, directory, report)
        client.set_terminated(run_id, "FINISHED")
        receipt = {"run_key": key, "mlflow_run_id": run_id, "mlflow_experiment_id": experiment_id,
                   "files": files, "run_artifacts": artifacts, "complete": True}
        atomic_write(directory / "epoch-receipt.json", json.dumps(receipt, indent=2).encode())
    return receipt
