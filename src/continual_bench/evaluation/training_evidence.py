"""Persist the recorded training coordinates and render one set of plots per fit."""

import math
from pathlib import Path

from continual_bench.evaluation.artifacts import Figure, write_artifacts
from continual_bench.evaluation.environment import encoded
from continual_bench.infrastructure.storage import atomic_write
from continual_bench.infrastructure.training_tracking import HISTORY_FILE, training_round_mlflow

AXES = {"epoch": "época", "minibatch": "minibatch", "update": "atualização", "example": "exemplo"}
QUALITY = {
    "accuracy", "balanced_accuracy", "f1", "f1_macro", "macro_f1", "precision", "recall",
    "macro_precision", "macro_recall", "pr_auc", "macro_pr_auc", "roc_auc", "macro_roc_auc", "r2",
}
LOSSES = {"loss", "log_loss", "mae", "mse", "rmse", "brier", "error_rate"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _plot_group(metric):
    name = metric.replace(":", "/").rsplit("/", 1)[-1]
    if name in LOSSES:
        return "loss"
    if name in QUALITY:
        return "quality"
    if name.endswith("_seconds"):
        return "timing"
    if name.endswith(("_presentations", "_samples", "_examples")) or name == "presentations":
        return "volume"
    return None


def training_figures(fits):
    figures = []
    descriptions = {
        "loss": ("Perda e erro", "valor"),
        "quality": ("Métricas de qualidade", "valor"),
        "timing": ("Tempo por step", "s"),
        "volume": ("Exemplos e apresentações", "contagem"),
    }
    for fit in fits:
        steps = fit["steps"]
        if not steps:
            continue
        kind = steps[0]["kind"]
        context = {key: value for key, value in fit.items() if key not in ("summary", "steps")}
        keys = sorted({key for row in steps for key, value in row["metrics"].items() if _finite(value)})
        for group, (label, unit) in descriptions.items():
            series = []
            for key in keys:
                if _plot_group(key) != group:
                    continue
                points = []
                for row in steps:
                    value = row["metrics"].get(key)
                    points.append([row["step"], value if _finite(value) else None])
                series.append({"name": key, "points": points})
            if series:
                title = f"{fit['fit_id']} · {fit['action']} · {label}"
                if kind == "minibatch" and group in {"loss", "quality"}:
                    title += " antes da atualização"
                figures.append(Figure(f"training/{fit['fit_id']}", group, title, "line", {
                    "series": series, "x_label": AXES[kind], "y_label": unit, "step_kind": kind,
                    "fit": context, "global_steps": [[row["step"], row["global_step"]] for row in steps],
                }))
    return figures


def write_training_artifacts(directory, report):
    """Append grouped plots to the existing protocol's artifact index."""
    fits = report["training_history"]
    # The same validation is used by the tracking adapters; fail before writing partial evidence.
    for _ in training_round_mlflow("", report):
        pass
    document = {"version": 1, "fits": fits}
    atomic_write(Path(directory) / HISTORY_FILE, encoded(document))
    return write_artifacts(
        directory, training_figures(fits), {"training/history": document}, preserve_existing=True,
    )
