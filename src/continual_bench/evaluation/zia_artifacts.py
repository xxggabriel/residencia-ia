"""Adapt retained Zia evidence to the shared tracking and artifact contracts."""

import hashlib
from itertools import accumulate

from continual_bench.evaluation.artifacts import Figure, binary_figures, system_figures, write_artifacts
from continual_bench.evaluation.tracking import VERSION, metric_catalog
from continual_bench.infrastructure.storage import atomic_write


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
        name: {"status": "available" if value is not None else "unavailable",
               "reason": None if value is not None else "undefined_or_requires_other_evaluation_evidence",
               "scope": report["protocol"]}
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
        "version": VERSION, "definitions_sha256": hashlib.sha256(raw).hexdigest(),
        "metrics": summary, "series": series, "availability": availability, "costs": report["costs"],
        "cost_sensitivity": sensitivity,
        "settings": {"protocol": report["protocol"], "decision_rule": report["contract"]["decision_rule"],
                     "block_rows": report["contract"]["block_rows"],
                     "score_bins": report["contract"]["score_bins"], "costs": report["costs"]["settings"],
                     "upf_alphas": [row["alpha"] for row in sensitivity]},
    }
    if predictions:
        tracking["settings"]["fading_factor"] = predictions["fading_factor"]
        tracking["prediction_archive"] = {"path": "predictions", "origin": predictions["origin"],
                                          "rows": predictions["rows"]}
    updates = [{"seq": v["end_seq"], **v} for v in report["model_versions"][1:]]
    alarms = report.get("alarms")
    if alarms is None and report["policy"] == "drift":
        alarms = [{"seq": u["seq"], "inferred_from": "drift_policy_reset"} for u in updates
                  if u["action"] == "reset_retrain"]
    return {"dataset": manifest, "environment": report["contract"]["environment"],
            "tracking": tracking, "windows": windows, "updates": updates, "alarms": alarms or [],
            "training": {"used": summary["training_used_post_warmup"]}}


def write_zia_artifacts(directory, report, adapted):
    tracking = adapted["tracking"]
    atomic_write(directory / "metric-definitions.json", metric_catalog()[1])
    plot_tracking = {**tracking, "metrics": {k: v for k, v in tracking["metrics"].items() if v is not None}}
    figures = binary_figures({**adapted, "tracking": plot_tracking})
    figures.extend(system_figures(report["system_metrics"]))
    classes = report["contract"]["input"]["source"]["classes"]
    for scope, field in (("prequential", "prequential_confusion"), ("holdout", "holdout_confusion")):
        matrix = report[field]
        title = f"Matriz de confusão ({scope})"
        figures.append(Figure("performance", f"{scope}_confusion", title, "heatmap", {
            "rows": classes, "columns": classes, "matrix": matrix,
            "x_label": "Predição", "y_label": "Classe real", "value_range": (0, max(map(max, matrix))),
        }))
    windows = [w for w in report["windows"] if w["scope"] == "holdout"]
    figures.append(Figure("performance", "holdout_accuracy", "Acurácia no holdout", "line", {
        "series": [{"name": name, "points": [[w["end_seq"], w[name]] for w in windows]}
                   for name in ("accuracy", "balanced_accuracy")],
        "x_label": "seq (fim da janela)", "y_label": "valor",
    }))
    versions = report["model_versions"]
    figures.append(Figure("adaptation", "model_storage", "Armazenamento acumulado dos modelos", "line", {
        "series": [{"name": "bytes", "points": list(zip(
            (v["end_seq"] for v in versions), accumulate(v["bytes"] for v in versions), strict=True,
        ))}],
        "x_label": "seq", "y_label": "bytes",
    }))
    return write_artifacts(directory, figures, {
        "summary/metrics": {"metrics": tracking["metrics"], "availability": tracking["availability"],
                            "report_metrics": report["metrics"], "holdout": report["holdout"]},
        "summary/classes": {"counts": report["class_counts"], "prequential": report["class_metrics"],
                            "holdout": report["holdout_class_metrics"]},
        "series/tracking": tracking["series"],
        "series/windows": report["windows"],
        "events/alarms": adapted["alarms"],
        "events/adaptations": {"updates": adapted["updates"], "training": adapted["training"]},
        "system/samples": report["system_metrics"],
        "costs/accounting": {"learner": report["costs"], "evaluation": report["evaluation_aws_cost"]},
        "model/explanation": report["model_explanation"],
    })
