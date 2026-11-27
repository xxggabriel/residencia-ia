"""Idempotent MLflow publication for Zia runs and model versions."""

import json
from pathlib import Path

from .artifact_policy import metrics_only
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
        run = (
            matches[0]
            if matches
            else client.create_run(
                experiment_id,
                tags={
                    "zia_run_key": key,
                    "mlflow.runName": f"{dataset}-{policy}",
                    "dataset": dataset,
                    "scope": scope,
                    "policy": policy,
                    "campaign": campaign,
                },
            )
        )
    return run.info.run_id, experiment_id


def publish_run_metrics(client, directory, run_id, report):
    from mlflow.entities import Dataset, DatasetInput, InputTag, Param

    contract = report["contract"]
    source = contract["input"]["source"]
    env = contract["environment"]
    # Older immutable real-stream manifests keep their canonical license in source.
    license_name = contract["input"].get("license")
    if license_name is None:
        license_name = source.get("license")
    if not isinstance(license_name, str) or not license_name.strip():
        raise ValueError("dataset license is absent from the prepared input and source")
    if source.get("license") is not None and source["license"] != license_name:
        raise ValueError("dataset license differs between the prepared input and source")
    # Preserve raw-source identity separately from the prepared Parquet digest.
    dataset_source = (
        f"ziadatalabs/{source['repo']}"
        if "repo" in source
        else (
            f"pinned-real/{report['dataset']}"
            if source.get("url")
            else f"catalog/{source.get('catalog', report['dataset'])}"
        )
    )
    source_uri = source.get("url") or (
        f"https://huggingface.co/datasets/{dataset_source}/resolve/{source['revision']}/{source['file']}"
        if "repo" in source
        else f"catalog://{source.get('catalog', report['dataset'])}"
    )
    params = {
        "model": report["model"],
        "policy": report["policy"],
        "protocol": report["protocol"],
        "dataset_source": dataset_source,
        "dataset_source_uri": source_uri,
        "dataset_revision": (
            source.get("revision") or source.get("raw_sha256") or source.get("source_version") or "catalog"
        ),
        "dataset_source_version": source.get("source_version"),
        "dataset_raw_sha256": source.get("raw_sha256"),
        "dataset_sha256": contract["input"]["sha256"],
        "dataset_license": license_name,
        "samples": contract["input"]["rows"],
        "holdout_start": contract["holdout_start"],
        "block_rows": contract["block_rows"],
        "model_params": json.dumps(report["model_params"]),
        "code_sha256": env["code_sha256"],
        "git_revision": env["git_revision"],
        "seed": (contract.get("iterative_training") or {}).get("seed"),
        "artifact_mode": report.get("artifact_mode", "full"),
    }
    if aws := report["costs"].get("aws"):
        params["aws_region"] = aws["settings"]["region"]
    client.log_batch(
        run_id,
        params=[Param(name, str(value)) for name, value in params.items() if value is not None],
        synchronous=True,
    )
    inputs = client.get_run(run_id).inputs
    if not inputs or not inputs.dataset_inputs:
        dataset = Dataset(
            name=params["dataset_source"],
            digest=params["dataset_sha256"][:32],
            source_type="http",
            source=json.dumps({"url": source_uri}),
            profile=json.dumps(
                {
                    "num_rows": params["samples"],
                    "num_features": len(contract["input"]["features"]),
                }
            ),
        )
        client.log_inputs(
            run_id,
            datasets=[
                DatasetInput(
                    dataset,
                    tags=[InputTag("mlflow.data.context", report["protocol"])],
                )
            ],
        )
    points = [(f"summary_{k}", 0, v) for k, v in report["metrics"].items()]
    points.extend((f"tracking_v1/summary/{k}", 0, v) for k, v in report["metrics"].items())
    points.extend((f"holdout/{k}", 0, v) for k, v in report["holdout"].items())
    for window in report["windows"]:
        points.extend(
            (f"{window['scope']}/windows/{k}", window["end_seq"], v)
            for k, v in window.items()
            if k not in {"scope", "end_seq", "start_time_ms", "end_time_ms"}
        )
    for group in ("class_metrics", "holdout_class_metrics"):
        points.extend(
            (f"{group}/{label}/{key}", 0, value)
            for label, row in report[group].items()
            for key, value in row.items()
        )
    for version in report["model_versions"]:
        fields = (
            "train_samples",
            "validation_samples",
            "training_presentations",
            "train_seconds",
            "iterations",
            "bytes",
        )
        points.extend(
            (f"training/{name}", version["end_seq"], version[name]) for name in fields if name in version
        )
        unit = {"epoch": "epochs", "minibatch": "minibatches"}.get(version.get("iteration_unit"))
        if unit:
            points.append((f"training/{unit}", version["end_seq"], version["iterations"]))
    points.extend(
        ("tracking_v1/posthoc_cost_sensitivity", index, value)
        for index, (_, value) in enumerate(report["cost_sensitivity"])
    )
    if metrics_only(report):
        log_metric_points(client, run_id, points, max_points_per_key=None)
    else:
        log_metric_points(client, run_id, points)
    client.log_artifact(run_id, str(directory / "report.json"))
    # Only recorded model versions count toward AWS storage and are published.
    for model in report["model_versions"]:
        if model.get("file"):
            client.log_artifact(run_id, str(directory / model["file"]), artifact_path="models")
