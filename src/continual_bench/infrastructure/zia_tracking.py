"""Idempotent MLflow publication for Zia runs and model versions."""

import json
from pathlib import Path

from .metric_tracking import log_metric_points
from .storage import publication_lock


def begin(client, directory, dataset, scope, policy, campaign):
    directory = Path(directory)
    key = directory.name
    name = f"continual-bench/zia-{dataset}-{scope}-{campaign}"
    with publication_lock(directory.parent, name):
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        matches = client.search_runs([experiment_id], f"tags.zia_run_key = '{key}'")
        if len(matches) > 1:
            raise ValueError("multiple MLflow runs for the same Zia job")
        run = matches[0] if matches else client.create_run(experiment_id, tags={
            "zia_run_key": key, "mlflow.runName": f"{dataset}-{policy}",
            "dataset": dataset, "scope": scope, "policy": policy, "campaign": campaign,
        })
    return run.info.run_id, experiment_id


def publish_run_metrics(client, directory, run_id, report):
    from mlflow.entities import Dataset, DatasetInput, InputTag, Param

    contract = report["contract"]
    source = contract["input"]["source"]
    env = contract["environment"]
    params = {
        "model": report["model"], "policy": report["policy"], "protocol": report["protocol"],
        "dataset_source": f"ziadatalabs/{source['repo']}", "dataset_revision": source["revision"],
        "dataset_sha256": contract["input"]["sha256"], "dataset_license": contract["input"]["license"],
        "samples": contract["input"]["rows"], "holdout_start": contract["holdout_start"],
        "block_rows": contract["block_rows"], "model_params": json.dumps(report["model_params"]),
        "code_sha256": env["code_sha256"], "git_revision": env["git_revision"],
        "aws_region": report["costs"]["aws"]["settings"]["region"],
    }
    client.log_batch(
        run_id, params=[Param(name, str(value)) for name, value in params.items()], synchronous=True,
    )
    inputs = client.get_run(run_id).inputs
    if not inputs or not inputs.dataset_inputs:
        dataset = Dataset(
            name=params["dataset_source"], digest=params["dataset_sha256"][:32], source_type="http",
            source=json.dumps({"url": f"https://huggingface.co/datasets/{params['dataset_source']}"
                                      f"/resolve/{source['revision']}/{source['file']}"}),
            profile=json.dumps({
                "num_rows": params["samples"], "num_features": len(contract["input"]["features"]),
            }),
        )
        client.log_inputs(run_id, datasets=[DatasetInput(
            dataset, tags=[InputTag("mlflow.data.context", report["protocol"])],
        )])
    points = [(f"summary_{k}", 0, v) for k, v in report["metrics"].items()]
    points.extend((f"tracking_v1/summary/{k}", 0, v) for k, v in report["metrics"].items())
    points.extend((f"holdout/{k}", 0, v) for k, v in report["holdout"].items())
    for window in report["windows"]:
        points.extend((f"{window['scope']}/windows/{k}", window["end_seq"], v) for k, v in window.items()
                      if k not in {"scope", "end_seq", "start_time_ms", "end_time_ms"})
    for group in ("class_metrics", "holdout_class_metrics"):
        points.extend((f"{group}/{label}/{key}", 0, value) for label, row in report[group].items()
                      for key, value in row.items())
    for version in report["model_versions"]:
        points.extend((f"training/{k}", version["end_seq"], version[k])
                      for k in ("train_samples", "train_seconds", "iterations", "bytes"))
    points.extend(("tracking_v1/posthoc_cost_sensitivity", index, value)
                  for index, (_, value) in enumerate(report["cost_sensitivity"]))
    log_metric_points(client, run_id, points)
    client.log_artifact(run_id, str(directory / "report.json"))
    # Only recorded model versions count toward AWS storage and are published.
    for model in report["model_versions"]:
        client.log_artifact(run_id, str(directory / model["file"]), artifact_path="models")
