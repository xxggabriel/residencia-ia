"""Adapt retained Zia evidence to the shared tracking and artifact contracts."""

import hashlib
import math
from itertools import accumulate

from continual_bench.evaluation.artifacts import (
    Figure,
    binary_figures,
    retraining_figures,
    system_figures,
    write_artifacts,
)
from continual_bench.evaluation.tracking import VERSION, metric_catalog
from continual_bench.evaluation.training_evidence import training_figures
from continual_bench.infrastructure.artifact_policy import metrics_only
from continual_bench.infrastructure.observation_tracking import write_observation_evidence
from continual_bench.infrastructure.storage import atomic_write


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def tracking_report(report, predictions=None):
    manifest = report["contract"]["input"]
    summary = dict(report["metrics"])
    for name, source in {
        "adaptation_count": "update_count",
        "prediction_throughput": "prediction_throughput_per_second",
        "ram_time_bytes_seconds": "sampled_rss_byte_seconds",
    }.items():
        summary[name] = summary.get(source)
    # The legacy catalog includes metrics that require evidence outside this protocol.
    definitions, raw = metric_catalog()
    fading = predictions["fading_accuracy"] if predictions else []
    if fading:
        summary["fading_accuracy"] = fading[-1]["value"]
    if predictions:
        summary["prediction_archive_bytes"] = sum(part["bytes"] for part in predictions["parts"])
    for name in definitions["definitions"]:
        summary.setdefault(name, None)
    availability = {
        name: {
            "status": "available" if value is not None else "unavailable",
            "reason": None if value is not None else "undefined_or_requires_other_evaluation_evidence",
            "scope": report["protocol"],
        }
        for name, value in summary.items()
    }
    windows = [w for w in report["windows"] if w["scope"] == "prequential"]
    series = {}
    if fading:
        series["fading_accuracy"] = fading
    for window in windows:
        for name, value in window.items():
            if name not in {"scope", "end_seq", "start_time_ms", "end_time_ms"}:
                series.setdefault(name, []).append({"step": window["end_seq"], "value": value})
    sensitivity = [{"alpha": a, "value": v} for a, v in report["cost_sensitivity"]]
    series["posthoc_cost_sensitivity"] = [
        {"step": index, "value": row["value"]} for index, row in enumerate(sensitivity)
    ]
    tracking = {
        "version": VERSION,
        "definitions_sha256": hashlib.sha256(raw).hexdigest(),
        "metrics": summary,
        "series": series,
        "availability": availability,
        "costs": report["costs"],
        "cost_sensitivity": sensitivity,
        "settings": {
            "protocol": report["protocol"],
            "decision_rule": report["contract"]["decision_rule"],
            "block_rows": report["contract"]["block_rows"],
            "score_bins": report["contract"]["score_bins"],
            "costs": report["costs"]["settings"],
            "upf_alphas": [row["alpha"] for row in sensitivity],
        },
    }
    retraining = report.get("retraining")
    if retraining:
        for name, rule in retraining["rules"].items():
            for key in ("cost", "retrains", "average_accuracy", "scpe", "strategy_cost"):
                series[f"retraining/{key}_vs_alpha/{name}"] = [
                    {"step": index, "value": row[key]}
                    for index, row in enumerate(rule["curve"])
                    if row[key] is not None
                ]
        first = retraining["setup"]["offline_batches"]
        for name, values in retraining["cumulative_strategy_cost"].items():
            series[f"retraining/cumulative_strategy_cost/{name}"] = [
                {"step": first + index, "value": value}
                for index, value in enumerate(values)
                if value is not None
            ]
        tracking["settings"]["retraining"] = {
            "alphas": retraining["alphas"],
            "alpha_max": retraining["setup"]["alpha_max"],
            "focus_alpha": retraining["setup"]["focus_alpha"],
            "timing": retraining["setup"]["timing"],
            "offline_batches": first,
            "online_batches": retraining["setup"]["online_batches"],
            "series_step": "alpha grid index (cost/retrains/accuracy/scpe) or online batch index",
        }
    if predictions:
        tracking["settings"]["fading_factor"] = predictions["fading_factor"]
        tracking["prediction_archive"] = {
            "path": "predictions",
            "origin": predictions["origin"],
            "rows": predictions["rows"],
        }
    updates = [{"seq": v["end_seq"], **v} for v in report["model_versions"][1:]]
    alarms = report.get("alarms")
    if alarms is None and report["policy"] == "drift":
        alarms = [
            {"seq": u["seq"], "inferred_from": "drift_policy_reset"}
            for u in updates
            if u["action"] == "reset_retrain"
        ]
    return {
        "artifact_mode": report.get("artifact_mode", "full"),
        "dataset": manifest,
        "environment": report["contract"]["environment"],
        "tracking": tracking,
        "windows": windows,
        "updates": updates,
        "alarms": alarms or [],
        "training": {"used": summary["training_used_post_warmup"]},
    }


def write_zia_artifacts(directory, report, adapted):
    tracking = adapted["tracking"]
    atomic_write(directory / "metric-definitions.json", metric_catalog()[1])
    plot_tracking = {**tracking, "metrics": {k: v for k, v in tracking["metrics"].items() if v is not None}}
    classes = report["contract"]["input"]["source"]["classes"]
    figures = binary_figures({**adapted, "tracking": plot_tracking, "classes": classes})
    figures.extend(system_figures(report["system_metrics"]))
    history = report.get("training_history")
    if history:
        figures.extend(training_figures(history))
    if report.get("retraining"):
        figures.extend(retraining_figures(report["retraining"]))
    for scope, field in (("prequential", "prequential_confusion"), ("holdout", "holdout_confusion")):
        matrix = report[field]
        title = f"Matriz de confusão ({scope})"
        figures.append(
            Figure(
                "performance",
                f"{scope}_confusion",
                title,
                "heatmap",
                {
                    "rows": classes,
                    "columns": classes,
                    "matrix": matrix,
                    "x_label": "Predição",
                    "y_label": "Classe real",
                    "value_range": (0, max(map(max, matrix))),
                },
            )
        )
    windows = [w for w in report["windows"] if w["scope"] == "holdout"]
    ranking = ("roc_auc", "average_precision") if len(classes) == 2 else ("macro_roc_auc", "macro_pr_auc")
    for group, name, title, keys in (
        (
            "performance",
            "holdout_accuracy",
            "Acurácia no holdout",
            ("accuracy", "balanced_accuracy", "f1_macro"),
        ),
        ("probabilistic", "holdout_ranking", "Ranqueamento no holdout", ranking),
        ("probabilistic", "holdout_losses", "Brier e log loss no holdout", ("brier", "log_loss")),
    ):
        series = [
            {"name": key, "points": [[window["end_seq"], window.get(key)] for window in windows]}
            for key in keys
            if any(_finite(window.get(key)) for window in windows)
        ]
        if series:
            figures.append(
                Figure(
                    group,
                    name,
                    title,
                    "line",
                    {
                        "series": series,
                        "x_label": "seq (fim da janela)",
                        "y_label": "valor",
                    },
                )
            )
    versions = report["model_versions"]
    figures.append(
        Figure(
            "adaptation",
            "model_storage",
            "Armazenamento acumulado dos modelos",
            "line",
            {
                "series": [
                    {
                        "name": "bytes",
                        "points": list(
                            zip(
                                (v["end_seq"] for v in versions),
                                accumulate(v["bytes"] for v in versions),
                                strict=True,
                            )
                        ),
                    }
                ],
                "x_label": "seq",
                "y_label": "bytes",
            },
        )
    )
    write_observation_evidence(directory, report)
    return write_artifacts(
        directory,
        [] if metrics_only(report) else figures,
        {
            "summary/metrics": {
                "metrics": tracking["metrics"],
                "availability": tracking["availability"],
                "report_metrics": report["metrics"],
                "holdout": report["holdout"],
            },
            "summary/classes": {
                "counts": report["class_counts"],
                "prequential": report["class_metrics"],
                "holdout": report["holdout_class_metrics"],
            },
            "series/tracking": tracking["series"],
            "series/windows": report["windows"],
            "events/alarms": adapted["alarms"],
            "events/adaptations": {"updates": adapted["updates"], "training": adapted["training"]},
            "system/samples": report["system_metrics"],
            "costs/accounting": {"learner": report["costs"], "evaluation": report["evaluation_aws_cost"]},
            "model/explanation": report["model_explanation"],
            "retraining/analysis": report.get("retraining"),
            "training/history": None if history is None else {"version": 1, "fits": history},
            **(
                {
                    "rendering/status": {
                        "status": "deferred",
                        "reason": "metrics_only_preserves_all_plot_data_without_rendering",
                    },
                    **{
                        f"figures/{figure.group}/{figure.name}": {
                            "title": figure.title,
                            "group": figure.group,
                            "name": figure.name,
                            "kind": figure.kind,
                            **figure.data,
                        }
                        for figure in figures
                    },
                }
                if metrics_only(report)
                else {}
            ),
        },
    )


def write_zia_campaign_artifacts(directory, units):
    """Compare policy outcomes on matching partitions of one campaign snapshot."""
    succeeded = [unit for unit in units if unit["status"] == "SUCCEEDED"]
    metrics = {
        "prequential": (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "macro_roc_auc",
            "macro_pr_auc",
            "log_loss",
            "train_seconds",
            "retrain_count",
            "scenario_total_cost",
            "aws_cost_total_usd",
        ),
        "holdout": ("accuracy", "balanced_accuracy", "f1_macro", "macro_roc_auc", "macro_pr_auc", "log_loss"),
    }
    figures = []
    for scope, names in metrics.items():
        field = "metrics" if scope == "prequential" else "holdout"
        for name in names:
            values = [(unit.get(field) or {}).get(name) for unit in succeeded]
            if any(_finite(value) for value in values):
                title = f"{scope} · {name} por política"
                figures.append(
                    Figure(
                        f"comparison/{scope}",
                        name,
                        title,
                        "bar",
                        {
                            "labels": [unit["labels"]["policy"] for unit in succeeded],
                            "values": [value if _finite(value) else None for value in values],
                            "x_label": name,
                        },
                    )
                )
    return write_artifacts(
        directory,
        figures,
        {
            "comparison/policies": [
                {
                    key: unit[key]
                    for key in ("run_id", "labels", "status", "final_uri", "metrics", "holdout")
                    if key in unit
                }
                for unit in units
            ],
        },
    )
