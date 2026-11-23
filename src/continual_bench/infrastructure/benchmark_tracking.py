"""Parent run metrics and publication for completed benchmark reports."""

import hashlib
import json
import math
from pathlib import Path

from continual_bench.infrastructure.artifact_policy import metrics_only, verify_metrics_only_files

from .metric_tracking import log_metric_points, metric_points
from .observation_tracking import observation_tags, parent_observation_points
from .run_artifacts import publish_run_artifacts
from .storage import atomic_write, publication_lock
from .training_tracking import sanitize_mlflow_name


def benchmark_metric_points(report):
    """Flatten recorded metrics while retaining their partition and coordinate."""
    points = []

    def add(prefix, data, step):
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            name = f"{prefix}/{key}"
            if isinstance(value, dict):
                add(name, value, step)
            elif type(value) in (int, float) and math.isfinite(value):
                points.append((sanitize_mlflow_name(name), step, value))

    add("summary", report.get("metrics") or report.get("summary") or {}, 0)
    for key in (
        "training",
        "validation",
        "holdout",
        "per_task",
        "class_metrics",
        "holdout_class_metrics",
        "compute",
    ):
        add(key, report.get(key) or {}, 0)
    for item in report.get("windows", []):
        step = item.get("end_seq")
        if type(step) is int and step >= 0:
            add(
                f"{item.get('scope', 'prequential')}/windows",
                {
                    key: value
                    for key, value in item.items()
                    if key not in ("start_seq", "end_seq", "start_time_ms", "end_time_ms", "scope")
                },
                step,
            )
    for epoch in report.get("epochs", []):
        step = epoch["epoch"]
        add("holdout/epochs", epoch["metrics"], step)
        add(
            "compute/epochs",
            {
                key: epoch[key]
                for key in (
                    "training_seconds",
                    "prediction_seconds",
                    "train_presentations",
                    "test_examples",
                )
                if key in epoch
            },
            step,
        )
        add("costs/epochs", (epoch.get("costs") or {}).get("metrics", {}), step)
    costs = report.get("costs") or {}
    add("costs/total", costs.get("metrics", {}), 0)
    add("costs/aws", (costs.get("aws") or {}).get("metrics", {}), 0)
    model_metadata = report.get("model_versions")
    if isinstance(model_metadata, dict) and model_metadata.get("artifact_mode") == "metrics_only":
        add(
            "model_state/summary",
            {key: model_metadata.get(key) for key in ("serialized_state_bytes", "storage_bytes", "count")},
            0,
        )
        for point in model_metadata.get("state_measurements", []):
            add("model_state/sampled", {"serialized_bytes": point["serialized_bytes"]}, point["ordinal"])
    return points


def publish_local_benchmark(directory, report, *, experiment_name, client=None):
    """Publish the completed local report without changing its evaluation protocol."""
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    directory = Path(directory)
    retain_models = not metrics_only(report)
    if not retain_models:
        verify_metrics_only_files(directory)
    client = client or MlflowClient()
    path = next(
        path
        for name in ("report.json", "epoch-report.json", "result.json")
        if (path := directory / name).is_file()
    )
    data = path.read_bytes()
    if json.loads(data) != report or report.get("complete") is not True:
        raise ValueError("completed benchmark report differs from the retained artifact")
    digest = hashlib.sha256(data).hexdigest()
    key = report["run_key"]
    contract = report.get("contract") or {}
    config = report.get("config") or {}
    dataset = report.get("dataset") or contract.get("dataset") or config.get("dataset") or {}
    dataset = dataset.get("name") if isinstance(dataset, dict) else dataset
    seed = report.get("seed", contract.get("seed", (config.get("dataset") or {}).get("seed")))
    protocol = report.get("protocol") or contract.get("protocol") or config.get("mode")
    coordinates = {
        "cb-dataset": dataset,
        "cb-seed": seed,
        "cb-protocol": protocol,
        "cb-artifact-mode": "full" if retain_models else "metrics_only",
    }
    tags = {name: str(value) for name, value in coordinates.items() if value is not None}
    run_name = key
    if not retain_models:
        run_name = sanitize_mlflow_name(
            "-".join(str(v) for v in (dataset, protocol, f"seed-{seed}", key[:8]) if v is not None)
        )
    with publication_lock(directory.parent, f"benchmark-{key}"):
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            try:
                experiment_id = client.create_experiment(experiment_name)
            except MlflowException as error:
                if error.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
                experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        else:
            experiment_id = experiment.experiment_id
        runs = client.search_runs([experiment_id], f"tags.`cb-local-run-key` = '{key}'", max_results=2)
        if len(runs) > 1:
            raise ValueError("duplicate local benchmark runs in MLflow")
        run = (
            runs[0]
            if runs
            else client.create_run(
                experiment_id,
                tags={
                    "cb-local-run-key": key,
                    "mlflow.runName": run_name,
                    "cb-report-sha256": digest,
                    **tags,
                },
            )
        )
        if run.data.tags.get("cb-report-sha256") != digest:
            raise ValueError("local benchmark report changed for the existing MLflow run")
        run_id = run.info.run_id
        for tag_key, tag_value in tags.items():
            client.set_tag(run_id, tag_key, tag_value)
        client.set_tag(run_id, "benchmark_complete", "false")
        try:
            points = benchmark_metric_points(report)
            points.extend(parent_observation_points(report))
            if (tracking := report.get("tracking")) and tracking.get("version"):
                points.extend(
                    (key, step, value) for key, step, value in metric_points(tracking) if value is not None
                )
            log_metric_points(
                client, run_id, points, **({"max_points_per_key": None} if not retain_models else {})
            )
            for tag_key, tag_value in observation_tags(report).items():
                client.set_tag(run_id, tag_key, tag_value)
            client.log_artifact(run_id, str(path))
            artifacts = publish_run_artifacts(
                client, run_id, directory, report, include_checkpoints=retain_models
            )
            client.set_tag(run_id, "benchmark_complete", "true")
            client.set_terminated(run_id, "FINISHED")
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise
        receipt = {
            "complete": True,
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": experiment_id,
            "report_sha256": digest,
            **artifacts,
        }
        atomic_write(directory / "benchmark-receipt.json", json.dumps(receipt, indent=2).encode())
        return receipt
