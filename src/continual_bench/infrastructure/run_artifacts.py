"""Publish a run's system metrics, plots and plot data to its MLflow run.

System samples become `system/<name>` metrics (MLflow's "System metrics" tab)
with their original wall-clock timestamps; `artifacts/plots` and `artifacts/data`
are uploaded to the run artifact root as `plots/<group>/` and `data/<group>/`.
Republishing is idempotent: existing metric points are checked, not duplicated.
"""

from pathlib import Path

from .metric_tracking import missing_metric_points

SYSTEM_PREFIX = "system/"


def system_metric_points(system) -> tuple[list[tuple[str, int, float]], dict]:
    points, stamps = [], {}
    for step, sample in enumerate((system or {}).get("samples", [])):
        for name, value in sample["values"].items():
            key = SYSTEM_PREFIX + name
            points.append((key, step, value))
            stamps[key, step] = sample["timestamp_ms"]
    return points, stamps


def publish_run_artifacts(client, run_id, directory, report) -> dict:
    from mlflow.entities import Metric

    points, stamps = system_metric_points(report.get("system_metrics"))
    missing = missing_metric_points(client, run_id, points)
    for offset in range(0, len(missing), 500):
        client.log_batch(
            run_id,
            metrics=[
                Metric(key, value, stamps[key, step], step)
                for key, step, value in missing[offset : offset + 500]
            ],
            synchronous=True,
        )
    root = Path(directory) / "artifacts"
    uploaded = 0
    for folder in ("plots", "data"):
        if (root / folder).is_dir():
            client.log_artifacts(run_id, str(root / folder), artifact_path=folder)
            uploaded += sum(1 for path in (root / folder).rglob("*") if path.is_file())
    system = report.get("system_metrics") or {}
    client.set_tag(run_id, "system_metrics_collector", system.get("collector", "absent"))
    client.set_tag(run_id, "run_artifacts", f"{uploaded} files; plots/ and data/")
    return {"system_metric_points": len(points), "artifact_files": uploaded}
