"""Publish a run's system metrics, plots and plot data to its MLflow run.

System samples become `system/<name>` metrics (MLflow's "System metrics" tab)
with their original wall-clock timestamps; `artifacts/plots` and `artifacts/data`
are uploaded to the run artifact root as `plots/<group>/` and `data/<group>/`.
Republishing is idempotent: existing metric points are checked, not duplicated.
"""

from pathlib import Path

from continual_bench.infrastructure.artifact_policy import metrics_only, verify_metrics_only_files

from .metric_tracking import downsample_points, missing_metric_points
from .observation_tracking import publish_sdk_observations
from .training_tracking import publish_training_history

SYSTEM_PREFIX = "system/"
DEFAULT_MAX_SYSTEM_POINTS = 200


def system_metric_points(system) -> tuple[list[tuple[str, int, float]], dict]:
    points, stamps = [], {}
    for step, sample in enumerate((system or {}).get("samples", [])):
        for name, value in sample["values"].items():
            key = SYSTEM_PREFIX + name
            points.append((key, step, value))
            stamps[key, step] = sample["timestamp_ms"]
    return points, stamps


def publish_run_artifacts(client, run_id, directory, report, *, include_checkpoints=True) -> dict:
    from .tracking import MetricPoint, as_backend

    backend = as_backend(client)
    complete_series = metrics_only(report)
    if complete_series:
        verify_metrics_only_files(directory)
        include_checkpoints = False
    points, stamps = system_metric_points(report.get("system_metrics"))
    sampled = downsample_points(
        points, max_points_per_key=None if complete_series else DEFAULT_MAX_SYSTEM_POINTS
    )
    missing = missing_metric_points(client, run_id, sampled)
    backend.log_batch(
        run_id, metrics=[MetricPoint(key, value, step, stamps[key, step]) for key, step, value in missing]
    )
    root = Path(directory) / "artifacts"
    uploaded = 0
    for folder in ("plots", "data"):
        if (root / folder).is_dir():
            client.log_artifacts(run_id, str(root / folder), artifact_path=folder)
            uploaded += sum(1 for path in (root / folder).rglob("*") if path.is_file())
    figures = log_detailed_figures(report, run_id)
    system = report.get("system_metrics") or {}
    backend.log_batch(
        run_id,
        tags={
            "detailed_figures": str(len(figures)),
            "system_metrics_collector": system.get("collector", "absent"),
            "run_artifacts": f"{uploaded} files; plots/ and data/",
        },
    )
    result = {
        "system_metric_points": len(points),
        "artifact_files": uploaded,
        "detailed_figures": len(figures),
    }
    if "training_history" in report:
        result["training_rounds"] = publish_training_history(
            client,
            run_id,
            directory,
            report,
            include_checkpoints=include_checkpoints,
        )
    manifest = publish_sdk_observations(
        client,
        run_id,
        directory,
        report,
        result.get("training_rounds", []),
        **({"max_points_per_key": None} if complete_series else {}),
    )
    if manifest:
        result["observation_plots"] = manifest
    return result


def log_detailed_figures(report, run_id) -> list[str]:
    """Matplotlib dashboards via `mlflow.log_figure`; a plotting failure never fails publication."""
    try:
        from continual_bench.infrastructure.mlflow_figures import log_report_figures

        return log_report_figures(report, run_id=run_id)
    except Exception:  # matplotlib missing, or MLflow rejected a figure: SVG artifacts still ship
        return []
