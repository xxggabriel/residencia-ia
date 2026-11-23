"""Plots from the split-aware tracking observations, independent of the MLflow adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from functools import partial
from html import escape
from itertools import chain, islice, pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile

from .mlflow_figures import FIGURE_DPI, MARKER_STYLE, PALETTE, _pyplot, matplotlib_available
from .retraining_pages import PAGE_SCRIPT, PAGE_STYLE, page_data
from .storage import atomic_write

VERSION = "benchmark-plots-v1"
MAX_POINTS = 800
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
STYLES = ("-", "--", "-.", ":")
FIT_STYLES = {
    "initial": ("o", "#0072B2", "Treino inicial"),
    "promoted": ("^", "#007f63", "Checkpoint promovido"),
    "rejected": ("x", "#D55E00", "Candidato rejeitado"),
    "candidate": ("D", "#882255", "Candidato"),
    "update": ("s", "#444444", "Atualização"),
    "interrupted": ("X", "#6b3332", "Ajuste interrompido"),
}
EVENT_STYLES = {
    **MARKER_STYLE,
    "bound": ("#7354a0", ":", "Última época"),
    "censored": ("#7354a0", "-.", "Mudança com acompanhamento incompleto"),
}
QUALITY_LABELS = {
    "macro_f1": "Macro-F1",
    "accuracy": "Acurácia",
    "positive_precision": "Precisão positiva",
    "positive_recall": "Recall positivo",
    "average_precision": "AP",
    "recall_at_fpr_0_01": "Recall a 1% de FPR",
    "mae": "MAE",
    "rmse": "RMSE",
}
QUALITY_KEYS = {
    "binary": tuple(QUALITY_LABELS)[:6],
    "multiclass": ("macro_f1", "accuracy"),
    "regression": ("mae", "rmse"),
}
WINDOW_LABELS = {
    "epoch": ("Qualidade por época", "Época"),
    "candidate": ("Qualidade por candidato", "Candidato (ordem do ajuste)"),
}
EFFORT = {
    "unique_training_examples": "Exemplos únicos",
    "training_presentations": "Apresentações",
    "replay_presentations": "Apresentações de replay",
    "steps": "Steps",
    "epochs": "Épocas",
    "updates": "Atualizações",
}
RESOURCE_PANELS = (
    (
        "Tempo",
        "s",
        {
            "training_seconds": "Treino",
            "inference_seconds": "Inferência",
            "served_inference_seconds": "Inferência servida",
            "duration_seconds": "Duração total",
            "tracking_overhead_seconds": "Overhead do tracking",
        },
    ),
    (
        "Vazão de inferência",
        "predições/s",
        {
            "predictions_per_second": "Inferência",
            "served_predictions_per_second": "Inferência servida",
        },
    ),
    ("Pico de RSS", "bytes", {"peak_rss_bytes": "Pico de RSS"}),
    ("Custo total", None, {"cost_total": "Custo total"}),
    ("Custo por 1.000 exemplos avaliados", None, {"cost_per_1000_evaluated": "Custo normalizado"}),
)


@dataclass
class Series:
    name: str
    points: list[dict]
    input_count: int
    style: str = "line"

    @property
    def reduced(self):
        return len(self.points) < self.input_count


@dataclass
class Panel:
    title: str
    x_label: str
    y_label: str
    series: list[Series] = field(default_factory=list)
    y_lim: tuple[float, float] | None = None
    markers: list[dict] = field(default_factory=list)
    zero: bool = False
    annotate: bool = False
    fit_axis: bool = False
    integer_x: bool = False
    primary_annotation: bool = False


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _point(x, y, **detail):
    return {"x": x, "y": _number(y), **{key: value for key, value in detail.items() if value is not None}}


def _sample(points, count, *, limit=MAX_POINTS, keep_x_extrema=False):
    """Keep bucket extrema, gap edges, and explicit event/version boundaries."""
    if count <= limit:
        return list(points)
    retained = {}
    previous = None
    x_low = x_high = None
    iterator = enumerate(points)
    width = math.ceil(count / max(1, limit // 4))
    while first := next(iterator, None):
        retained[first[0]] = first[1]
        low = high = None
        for index, entry in chain((first,), islice(iterator, width - 1)):
            value = entry["y"]
            if value is not None:
                if low is None or value < low[1]["y"]:
                    low = (index, entry)
                if high is None or value > high[1]["y"]:
                    high = (index, entry)
                if keep_x_extrema:
                    if x_low is None or entry["x"] < x_low[1]["x"]:
                        x_low = (index, entry)
                    if x_high is None or entry["x"] > x_high[1]["x"]:
                        x_high = (index, entry)
            if index and (
                (value is None) != (previous["y"] is None) or entry.get("version") != previous.get("version")
            ):
                retained[index - 1] = previous
                retained[index] = entry
            if entry.get("event"):
                retained[index] = entry
            previous = entry
        retained[index] = entry
        if low is not None:
            retained[low[0]] = low[1]
            retained[high[0]] = high[1]
    if x_low is not None:
        retained[x_low[0]] = x_low[1]
        retained[x_high[0]] = x_high[1]
    return [retained[index] for index in sorted(retained)]


def _series(name, points, count, style="line"):
    retained = _sample(points, count, keep_x_extrema=style == "scatter")
    if any(point["y"] is not None for point in retained):
        return Series(name, retained, count, style)
    return None


def _add(panel, name, points, count, style="line"):
    if series := _series(name, points, count, style):
        panel.series.append(series)


def _fit_detail(fit):
    checkpoint = fit.get("checkpoint") or {}
    summary = fit.get("summary") or {}
    promotion = fit.get("promotion_status")
    if promotion is None or promotion not in FIT_STYLES:
        if summary.get("promoted") is False or fit.get("status") == "rejected":
            promotion = "rejected"
        elif fit.get("action") in ("initial", "initial_fit"):
            promotion = "initial"
        elif fit.get("deployed_version") is not None or summary.get("promoted"):
            promotion = "promoted"
        elif "candidate" in fit.get("action", ""):
            promotion = "candidate"
        else:
            promotion = "update"
    if fit.get("action") in ("initial", "initial_fit") and promotion == "promoted":
        promotion = "initial"
    return {
        "fit": fit["fit_id"],
        "action": fit.get("action"),
        "status": promotion,
        "version": fit.get("deployed_version"),
        "source_version": fit.get("source_model_version"),
        "checkpoint": checkpoint.get("sha256") or checkpoint.get("file"),
    }


def _quality_entry(item):
    return item.get("quality", item)


def _quality_keys(task):
    return QUALITY_KEYS.get(task, ())


def _target_unit(value):
    return "unidade do alvo" if value in (None, "target_unit") else value


def _window_panels(observations, fits, *, primary_only=False, markers=None):
    task = observations.get("task")
    keys = _quality_keys(task)
    if not keys:
        return []
    details = {fit["fit_id"]: _fit_detail(fit) for fit in fits}
    groups = {}
    for row in observations.get("windows") or []:
        if _number(row.get("step")) is not None:
            groups.setdefault((row["scope"], row.get("coordinate")), []).append(row)
    panels = []
    for (scope, coordinate), rows in groups.items():
        # The report's target unit belongs to this split. A window may omit redundant metadata.
        summary = _quality_entry((observations.get("scopes") or {}).get(scope) or {})
        unit = summary.get("target_unit") or (rows[0].get("quality") or {}).get("target_unit")
        label = f"Erro ({_target_unit(unit)})" if task == "regression" else "Proporção"
        if markers is None:
            transitions = [
                {"x": row["step"], "kind": "update", "label": "Primeira janela da nova versão"}
                for previous, row in pairwise(rows)
                if previous.get("model_version") is not None
                and row.get("model_version") is not None
                and previous["model_version"] != row["model_version"]
            ]
        else:
            transitions = markers
        title, x_label = WINDOW_LABELS.get(coordinate, ("Qualidade por janela", "Evento (fim da janela)"))
        panel = Panel(
            f"{title} · {scope}",
            x_label,
            label,
            y_lim=None if task == "regression" else (0, 1.045),
            markers=transitions if coordinate not in WINDOW_LABELS else [],
            integer_x=True,
        )
        for key in keys[:1] if primary_only else keys:
            _add(
                panel,
                QUALITY_LABELS[key],
                (
                    _window_point(row, scope, quality, key, details.get(row.get("fit_id"), {}))
                    for row in rows
                    for quality in [row.get("quality") or {}]
                ),
                len(rows),
            )
        if panel.series:
            panels.append(panel)
    return panels


def _window_point(row, scope, quality, key, detail):
    context = {
        **detail,
        "fit": row.get("fit_id"),
        "version": row.get("model_version"),
        "checkpoint": row.get("checkpoint_sha256") or row.get("checkpoint") or detail.get("checkpoint"),
    }
    return _point(
        row["step"], quality.get(key), scope=scope, support=quality.get("evaluated_examples"), **context
    )


def _fit_quality_panels(observations, fits):
    task = observations.get("task")
    keys = _quality_keys(task)
    if not keys:
        return []
    scopes = dict.fromkeys(scope for fit in fits for scope in fit.get("observation_scopes") or {})
    panels = []
    for scope in scopes:
        rows = [_quality_entry((fit.get("observation_scopes") or {}).get(scope) or {}) for fit in fits]
        unit = next((row["target_unit"] for row in rows if row.get("target_unit")), None)
        panel = Panel(
            f"Qualidade por ajuste · {scope}",
            "Ajuste na sequência",
            f"Erro ({_target_unit(unit)})" if task == "regression" else "Proporção",
            y_lim=None if task == "regression" else (0, 1.18),
            fit_axis=True,
            primary_annotation=True,
        )
        for key in keys[:2]:
            _add(
                panel,
                QUALITY_LABELS[key],
                (
                    _point(
                        index,
                        row.get(key),
                        scope=scope,
                        support=row.get("evaluated_examples"),
                        **_fit_detail(fit),
                    )
                    for index, (row, fit) in enumerate(zip(rows, fits, strict=True), 1)
                ),
                len(rows),
                "comparison",
            )
        if panel.series:
            panels.append(panel)
    return panels


def _official_overview(observations):
    task = observations.get("task")
    keys = _quality_keys(task)
    panels = []
    for scope, scoped in (observations.get("scopes") or {}).items():
        quality = _quality_entry(scoped)
        unit = _target_unit(quality.get("target_unit"))
        panel = Panel(
            f"Resumo oficial · {scope}",
            "Resumo do split",
            f"Erro ({unit})" if task == "regression" else "Proporção",
            y_lim=None if task == "regression" else (0, 1.18),
            annotate=True,
        )
        for index, key in enumerate(keys, 1):
            _add(
                panel,
                QUALITY_LABELS[key],
                (_point(index, quality.get(key), scope=scope, support=quality.get("evaluated_examples")),),
                1,
                "scatter",
            )
        if panel.series:
            panels.append(panel)
    return panels


def _key_label(key):
    leaf = key.rsplit("/", 1)[-1]
    return f"{QUALITY_LABELS[leaf]} · {key.rsplit('/', 1)[0]}" if leaf in QUALITY_LABELS else key


def _fit_series(fit, dimension):
    kind = fit.get("observed_step_kind")
    rows = fit.get(f"observation_{dimension}s") or []
    if not rows:
        return []
    keys = sorted({key for row in rows for key, value in _row_metrics(row) if _number(value) is not None})
    dimension_name = "Época local" if dimension == "epoch" else "Step local"
    title = f"{dimension_name} · {fit['fit_id']}"
    if dimension == "step" and isinstance(kind, str):
        title += f" ({kind})"
    markers = []
    if dimension == "epoch":
        summary = fit.get("summary") or {}
        best = fit.get("best_epoch", summary.get("best_epoch"))
        last = fit.get("last_epoch", summary.get("last_epoch", rows[-1][dimension]))
        promoted = fit.get("promoted_epoch") or summary.get("promoted_epoch")
        if promoted is None and _fit_detail(fit)["status"] in ("initial", "promoted"):
            promoted = fit.get("checkpoint_epoch", summary.get("checkpoint_epoch"))
            if promoted is None:
                promoted = (fit.get("checkpoint") or {}).get("epoch")
        if _number(best) and best > 0:
            markers.append({"x": best, "kind": "update", "label": "Melhor época"})
        if _number(last) and last > 0:
            stopped = summary.get("early_stopped") or fit.get("early_stopped")
            markers.append(
                {
                    "x": last,
                    "kind": "alarm" if stopped else "bound",
                    "label": "Última época · early stopping" if stopped else "Última época",
                }
            )
        if _number(promoted) is not None:
            markers.append({"x": promoted, "kind": "changepoint", "label": "Checkpoint promovido"})
    unit = next(
        (
            quality.get("target_unit")
            for quality in (fit.get("observation_scopes") or {}).values()
            if quality.get("target_unit")
        ),
        None,
    )
    panels = (
        Panel(f"Perda · {title}", dimension_name, "Perda", markers=markers, integer_x=True),
        Panel(
            f"Qualidade · {title}",
            dimension_name,
            "Proporção",
            y_lim=(0, 1.045),
            markers=markers,
            integer_x=True,
        ),
        Panel(
            f"Erro de regressão · {title}",
            dimension_name,
            _target_unit(unit),
            markers=markers,
            integer_x=True,
        ),
        Panel(f"Tempo · {title}", dimension_name, "s", markers=markers, integer_x=True),
        Panel(f"Volume por chamada · {title}", dimension_name, "Contagem", markers=markers, integer_x=True),
        Panel(f"Volume acumulado · {title}", dimension_name, "Contagem", markers=markers, integer_x=True),
        Panel(f"Pico de RSS do processo · {title}", dimension_name, "bytes", markers=markers, integer_x=True),
    )
    for key in keys:
        leaf = key.rsplit("/", 1)[-1]
        if leaf in ("loss", "log_loss", "objective", "native_loss") or leaf.endswith("_loss"):
            panel = panels[0]
        elif leaf in ("mae", "rmse"):
            panel = panels[2]
        elif leaf in QUALITY_LABELS:
            panel = panels[1]
        elif leaf.endswith("_seconds"):
            panel = panels[3]
        elif leaf.endswith(("_size", "_presentations", "_examples", "_samples")) or leaf in (
            "presentations",
            "examples_processed",
        ):
            panel = panels[4]
        elif leaf.endswith("_cumulative"):
            panel = panels[5]
        elif leaf == "peak_rss_bytes":
            panel = panels[6]
        else:
            continue
        scope = key.rsplit("/", 1)[0] if "/" in key else None
        detail = _fit_detail(fit)
        _add(
            panel,
            _key_label(key),
            (_fit_row_point(row, dimension, scope, key, detail) for row in rows),
            len(rows),
        )
    return panels


def _row_metrics(row):
    yield from (row.get("metrics") or {}).items()
    for scope, quality in (row.get("scopes") or {}).items():
        yield from ((f"{scope}/{key}", value) for key, value in quality.items() if key in QUALITY_LABELS)


def _fit_row_point(row, dimension, scope, key, detail):
    values = row.get("metrics") or {}
    quality = (row.get("scopes") or {}).get(scope) or {}
    leaf = key.rsplit("/", 1)[-1]
    if scope and "prequential" in scope and "prediction_model_version" in row:
        version = row["prediction_model_version"]
        if version != detail.get("version"):
            detail = {**detail, "version": version, "checkpoint": row.get("prediction_checkpoint")}
    return _point(
        row[dimension],
        quality.get(leaf, values.get(key)),
        scope=scope,
        support=quality.get("evaluated_examples", values.get(f"{scope}/evaluated_examples")),
        **detail,
    )


def _comparison_panel(title, unit, fields, fits, summary, container, *, caption=None):
    panel = Panel(
        title, "Ajuste na sequência" if fits else "Run principal", unit, annotate=True, fit_axis=bool(fits)
    )
    panel.title = f"{title} · {caption}" if caption else title
    for index, (key, label) in enumerate(fields.items(), 1):
        points = (
            (
                _point(position, (fit.get(container) or {}).get(key), **_fit_detail(fit))
                for position, fit in enumerate(fits, 1)
            )
            if fits
            else (_point(index, summary.get(key)),)
        )
        _add(panel, label, points, len(fits) if fits else 1, "comparison" if fits else "scatter")
    return panel


def _effort_panels(fits, summary):
    return [
        panel
        for title, fields in (
            (
                "Exemplos",
                {
                    **dict(list(EFFORT.items())[:3]),
                    "examples_observed": "Observados",
                    "evaluated_examples": "Predições avaliadas",
                },
            ),
            ("Steps, épocas e atualizações", dict(list(EFFORT.items())[3:])),
        )
        if (panel := _comparison_panel(title, "Contagem", fields, fits, summary, "observation_effort")).series
    ]


def _resource_panels(fits, summary):
    panels = []
    for title, unit, fields in RESOURCE_PANELS:
        if unit is None:
            if fits:
                groups = {}
                for fit in fits:
                    data = fit.get("observation_resources") or {}
                    groups.setdefault((data.get("cost_currency"), data.get("cost_kind")), True)
                for currency, kind in groups:
                    if currency is None or kind is None:
                        continue
                    selected = [
                        fit
                        if (
                            (fit.get("observation_resources") or {}).get("cost_currency"),
                            (fit.get("observation_resources") or {}).get("cost_kind"),
                        )
                        == (currency, kind)
                        else {**fit, "observation_resources": {}}
                        for fit in fits
                    ]
                    panel = _comparison_panel(
                        title, currency, fields, selected, summary, "observation_resources", caption=kind
                    )
                    if panel.series:
                        panels.append(panel)
                continue
            unit, caption = summary.get("cost_currency"), summary.get("cost_kind")
            if unit is None or caption is None:
                continue
        elif unit == "bytes":
            scopes = (
                {(fit.get("observation_resources") or {}).get("rss_scope") for fit in fits}
                if fits
                else {summary.get("rss_scope")}
            )
            # Values from different RSS scopes cannot share a panel.
            for scope in sorted(scopes - {None}):
                selected = [
                    fit
                    if (fit.get("observation_resources") or {}).get("rss_scope") == scope
                    else {**fit, "observation_resources": {}}
                    for fit in fits
                ]
                panel = _comparison_panel(
                    title, unit, fields, selected, summary, "observation_resources", caption=scope
                )
                if panel.series:
                    panels.append(panel)
            continue
        else:
            caption = None
        panel = _comparison_panel(
            title, unit, fields, fits, summary, "observation_resources", caption=caption
        )
        if panel.series:
            panels.append(panel)
    return panels


def _ranking_panel(quality, task, kind, scope):
    axes = (
        ("recall", "precision", "Recall", "Precisão")
        if kind == "pr"
        else ("fpr", "tpr", "Taxa de falsos positivos", "Recall")
    )
    curve = quality.get(f"{kind}_curve") or {}
    ranking = quality.get("ranking_method")
    if isinstance(ranking, dict):
        ranking = ", ".join(f"{key}={value}" for key, value in ranking.items() if value is not None)
    elif ranking == "exact_ties_grouped":
        ranking = "Exata (empates agrupados)"
    elif isinstance(ranking, str) and (match := re.fullmatch(r"uniform_histogram_(\d+)_bins", ranking)):
        ranking = f"Histograma aproximado ({match[1]} bins uniformes)"
    ranking = ranking or "método não identificado"
    positive = quality.get("positive_class")
    decision_support = quality.get("evaluated_examples")
    support = quality.get("ranking_evaluated_examples", decision_support)
    positive_support = quality.get("ranking_positive_examples")
    if positive_support is None and support == decision_support:
        classes, supports = quality.get("classes") or [], quality.get("class_support") or []
        if positive in classes and len(classes) == len(supports):
            positive_support = supports[classes.index(positive)]
    method = f"{ranking} · Classe positiva={positive} · n={support}"
    if support != decision_support:
        method += f" de {decision_support} avaliados"
    if positive_support is not None:
        method += f" · n positivo={positive_support}"
    if kind == "pr" and _number(quality.get("average_precision")) is not None:
        method = f"AP={quality['average_precision']:.5g} · {method}"
    panel = Panel(
        f"{'Precisão–recall' if kind == 'pr' else 'ROC'} · {scope}\n{method}",
        axes[2],
        axes[3],
        y_lim=(0, 1.045),
    )
    x, y = curve.get(axes[0]) or [], curve.get(axes[1]) or []
    if task == "binary" and x and len(x) == len(y):
        style = "line"
        if kind == "pr":
            # Histogram curves advance from recall 0; sklearn supplies the reverse order.
            style = "steps-pre" if x[0] < x[-1] else "steps-post"
        _add(
            panel,
            "Escores da classe positiva",
            (
                _point(a, b, scope=scope, support=support)
                for a, b in zip(x, y, strict=True)
                if _number(a) is not None
            ),
            len(x),
            style,
        )
        if kind == "roc" and _number(quality.get("recall_at_fpr_0_01")) is not None:
            target = quality["recall_at_fpr_0_01"]
            operating = [
                a
                for a, b in zip(x, y, strict=True)
                if _number(a) is not None
                and a <= 0.01
                and _number(b) is not None
                and math.isclose(b, target, abs_tol=1e-12)
            ]
            panel.markers.append({"x": 0.01, "kind": "alarm", "label": "Limite de FPR (1%)"})
            if operating:
                _add(
                    panel,
                    f"Ponto operacional (recall={target:.4g})",
                    (_point(max(operating), target, scope=scope, support=support),),
                    1,
                    "scatter",
                )
    return panel


def _residual_panel(quality, scope):
    unit = _target_unit(quality.get("target_unit"))
    panel = Panel(f"Resíduos · {scope}", f"Predição ({unit})", f"Resíduo ({unit})", zero=True)
    residuals = quality.get("residuals") or {}
    x, y = residuals.get("prediction") or [], residuals.get("error") or []
    if x and len(x) == len(y):
        _add(
            panel,
            "Resíduos",
            (_point(a, b, scope=scope) for a, b in zip(x, y, strict=True) if _number(a) is not None),
            len(x),
            "scatter",
        )
    return panel


def _retention_panel(retention):
    rows = retention.get("series") or []
    panel = Panel(
        "Retenção e esquecimento · conjuntos históricos reservados", "Checkpoint", "Escore", zero=True
    )
    for key, name in (
        ("absolute_retention", "Retenção absoluta"),
        ("gem_bwt", "BWT de GEM"),
        ("post_learning_fm", "FM pós-aprendizado"),
    ):
        _add(
            panel,
            name,
            (_point(row["step"], row.get(key), version=row.get("model_version")) for row in rows),
            len(rows),
        )
    return panel


def _drift_panels(observations, fits):
    drift = observations.get("drift") or {}
    markers = []
    matching = drift.get("matching") or []
    censored = {match.get("change_seq") for match in matching if match.get("censored")}
    for key, kind in (("alarms", "alarm"), ("changes", "changepoint")):
        for event in drift.get(key) or []:
            x = event.get("seq") if isinstance(event, dict) else event
            if _number(x) is not None:
                markers.append({"x": x, "kind": "censored" if key == "changes" and x in censored else kind})
    for fit in fits:
        detail = _fit_detail(fit)
        if detail["status"] not in ("promoted", "rejected", "update"):
            continue
        x = fit.get("promotion_step")
        if x is None:
            x = fit.get("end_seq")
        if _number(x) is not None:
            markers.append({"x": x, "kind": "rejected" if detail["status"] == "rejected" else "update"})
    timeline = []
    if markers or drift.get("alarms") is not None:
        timeline = _window_panels(observations, fits, primary_only=True, markers=markers)
        if not timeline and markers:
            lanes = tuple(dict.fromkeys(marker["kind"] for marker in markers))
            panel = Panel("Eventos de drift e atualização", "Evento", "Tipos de evento")
            for level, lane in enumerate(lanes, 1):
                points = [_point(marker["x"], level) for marker in markers if marker["kind"] == lane]
                _add(panel, EVENT_STYLES[lane][2], points, len(points), "scatter")
            timeline = [panel]
    panel = Panel("Qualidade da detecção", "Métrica", "Proporção", y_lim=(0, 1.18), annotate=True)
    for i, (key, name) in enumerate((("precision", "Precisão"), ("recall", "Recall"), ("f1", "F1")), 1):
        _add(panel, name, (_point(i, drift.get(key)),), 1, "scatter")
    delay_unit = drift.get("delay_unit") or "unidade do horizonte do protocolo"
    incomplete = drift.get("censored_count", drift.get("changes_censored"))
    if incomplete is None and drift.get("matching") is not None:
        incomplete = len(censored)
    suffix = f" · censuradas={incomplete}" if incomplete is not None else ""
    if panel.series and suffix:
        panel.title += suffix
    return timeline, panel, drift.get("delays") or [], delay_unit, suffix


def _style(index):
    return PALETTE[index % len(PALETTE)], STYLES[index % len(STYLES)]


def _draw_panel(ax, panel):
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    for index, series in enumerate(panel.series):
        color, style = _style(index)
        x = [point["x"] for point in series.points]
        y = [point["y"] if point["y"] is not None else np.nan for point in series.points]
        if series.style == "comparison":
            # Join equal-scope points, excluding rejected/unselected checkpoints from the trajectory.
            linked = [
                p["y"] if p.get("status") not in ("rejected", "candidate") else None for p in series.points
            ]
            ax.plot(x, linked, color=color, linestyle=style, linewidth=1.25, label=series.name)
            for status, (marker, _, _) in FIT_STYLES.items():
                selected = [p for p in series.points if p.get("status") == status and p["y"] is not None]
                if selected:
                    ax.scatter(
                        [p["x"] for p in selected],
                        [p["y"] for p in selected],
                        marker=marker,
                        color=color,
                        zorder=3,
                    )
        elif series.style == "scatter":
            ax.scatter(x, y, color=color, marker="o", label=series.name, alpha=0.75, zorder=3)
        else:
            ax.plot(
                x,
                y,
                color=color,
                linestyle=style,
                linewidth=1.55,
                label=series.name,
                drawstyle=series.style if series.style.startswith("steps-") else "default",
                marker="o" if len(x) <= 35 else None,
                markersize=3,
            )
        if panel.annotate:
            for point in _label_points(series.points, 24 // len(panel.series)):
                support = f" · n={point['support']:,}" if point.get("support") is not None else ""
                below = series.style == "comparison" and index % 2 == 1 and point["y"] > 0.05
                rightmost = point["x"] == x[-1] and len(x) > 1
                ax.annotate(
                    f"{point['y']:.4g}{support}",
                    (point["x"], point["y"]),
                    xytext=(-4 if rightmost else 4, -6 if below else 6 + 9 * (index % 2)),
                    textcoords="offset points",
                    color="#212936",
                    fontsize=8,
                    va="top" if below else "bottom",
                    ha="right" if rightmost else "left",
                )
    if panel.primary_annotation:
        _annotate_primary(ax, panel.series)
    seen = set()
    for event in panel.markers:
        color, style, default = EVENT_STYLES[event["kind"]]
        label = event.get("label", default)
        ax.axvline(
            event["x"],
            color=color,
            linestyle=style,
            linewidth=1,
            alpha=0.7,
            label=None if label in seen else label,
        )
        seen.add(label)
    if panel.zero:
        ax.axhline(0, color="#556274", linewidth=0.9, linestyle=":")
    if panel.y_lim:
        ax.set_ylim(*panel.y_lim)
    elif panel.primary_annotation:
        ax.margins(y=0.15)
    if panel.fit_axis:
        fits = {point["x"]: point["fit"] for series in panel.series for point in series.points}
        if len(fits) <= 12:
            positions = sorted(fits)
            ax.set_xticks(
                positions,
                [fits[position] for position in positions],
                fontsize=8,
                rotation=25 if len(fits) > 7 else 0,
            )
    if panel.integer_x or (panel.fit_axis and len(fits) > 12):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_title(panel.title, fontsize=11, loc="left", fontweight="bold")
    ax.set_xlabel(panel.x_label)
    ax.set_ylabel(panel.y_label)
    ax.grid(axis="y", color="#CDD7E5", alpha=0.65, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=min(3, max(1, (len(panel.series) + len(seen) + 2) // 3)))
    if any(series.reduced for series in panel.series):
        ax.text(
            1,
            -0.27,
            "Série reduzida no desenho · dados completos no artefato",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color="#475569",
        )


def _label_points(points, limit):
    present = [point for point in points if point["y"] is not None]
    if len(present) <= limit:
        return present
    selected = [
        present[0],
        present[-1],
        min(present, key=lambda row: row["y"]),
        max(present, key=lambda row: row["y"]),
    ]
    return {point["x"]: point for point in selected}.values()


def _annotate_primary(ax, series):
    primary = series[0]
    top = {}
    for line in series:
        for point in line.points:
            if point["y"] is not None:
                top[point["x"]] = max(top.get(point["x"], -math.inf), point["y"])
    for point in _label_points(primary.points, 12):
        x = point["x"]
        support = f"\nn={point['support']:,}" if point.get("support") is not None else ""
        ax.annotate(
            f"{point['y']:.4g}{support}",
            (x, top[x]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_style(0)[0],
        )


def _panel_figure(plt, panels, title, subtitle=None):
    from matplotlib.lines import Line2D

    columns = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / columns)
    fig = plt.figure(figsize=(14 if columns == 2 else 10, 3.75 * rows + 1), layout="constrained")
    grid = fig.add_gridspec(rows, columns)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        slot = grid[row, :] if column == 0 and index == len(panels) - 1 else grid[row, column]
        _draw_panel(fig.add_subplot(slot), panel)
    if any(panel.fit_axis for panel in panels):
        statuses = {
            point.get("status") for panel in panels for series in panel.series for point in series.points
        }
        handles = [
            Line2D([], [], marker=marker, color=color, linestyle="None", label=label, markersize=6)
            for status, (marker, color, label) in FIT_STYLES.items()
            if status in statuses
        ]
        if handles:
            fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 5), fontsize=9)
    fig.suptitle(title + (f"\n{subtitle}" if subtitle else ""), fontsize=13, fontweight="bold")
    return fig


def _heatmap(ax, matrix, x_labels, y_labels, *, x_label, y_label, title, fraction=False):
    import numpy as np

    data = np.asarray(matrix, dtype=float)
    from matplotlib import colormaps

    color_map = colormaps["Blues"].with_extremes(bad="#c1c9d5")
    view = ax.imshow(
        np.ma.masked_invalid(data),
        cmap=color_map,
        aspect="auto",
        vmin=0 if fraction else None,
        vmax=1 if fraction else None,
    )
    ax.figure.colorbar(view, ax=ax, fraction=0.045)
    for setter, labels, rotate in ((ax.set_xticks, x_labels, 50), (ax.set_yticks, y_labels, 0)):
        positions = range(0, len(labels), max(1, math.ceil(len(labels) / 30)))
        setter(list(positions), [labels[i] for i in positions], rotation=rotate, fontsize=8)
    if max(data.shape) <= 13:
        finite = data[np.isfinite(data)]
        shade = 0.5 if fraction else (finite.max() / 2 if finite.size else 0)
        for (i, j), value in np.ndenumerate(data):
            label = (
                ("–" if not math.isfinite(value) else f"{value:.1%}")
                if fraction
                else (f"{value:,.0f}" if math.isfinite(value) else "–")
            )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > shade else "#182238",
            )
    ax.set(xlabel=x_label, ylabel=y_label, title=title)


def _matrix_available(classes, matrix):
    return (
        bool(classes)
        and isinstance(matrix, (list, tuple))
        and len(matrix) == len(classes)
        and all(isinstance(row, (list, tuple)) and len(row) == len(classes) for row in matrix)
        and any(_number(value) and value > 0 for row in matrix for value in row)
    )


def _confusion(plt, quality, scope):
    classes = [str(label) for label in quality["classes"]]
    matrix = quality["confusion_matrix"]
    normalized = [
        [value / total for value in row] if (total := sum(row)) else [None] * len(row) for row in matrix
    ]
    height = max(5, min(12, len(classes) * 0.28))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, height), layout="constrained")
    _heatmap(
        left, matrix, classes, classes, x_label="Classe prevista", y_label="Classe real", title="Contagens"
    )
    _heatmap(
        right,
        normalized,
        classes,
        classes,
        x_label="Classe prevista",
        y_label="Classe real",
        title="Normalizada pela classe real · cinza sem suporte",
        fraction=True,
    )
    threshold = quality.get("decision_threshold")
    context = ""
    if quality.get("positive_class") is not None:
        context = f" · Classe positiva={quality['positive_class']}"
        if threshold is not None:
            context += f" · Limiar={threshold:.4g}"
    fig.suptitle(f"Matriz de confusão · {scope}{context}")
    return fig


def _class_recall(plt, quality, scope):
    classes, recalls, supports = quality["classes"], quality["class_recall"], quality["class_support"]
    rows = list(zip(classes, recalls, supports, strict=True))
    columns = math.ceil(len(rows) / 40)
    height = max(4.8, 2.3 + min(40, len(rows)) * 0.31)
    fig, axes = plt.subplots(
        1, columns, figsize=(max(8, 5.2 * columns), height), squeeze=False, layout="constrained"
    )
    for index, ax in enumerate(axes.flat):
        subset = rows[40 * index : 40 * (index + 1)]
        positions = list(range(len(subset)))
        ax.barh(positions, [recall or 0 for _, recall, _ in subset], color="#0072B2")
        ax.set_yticks(positions, [str(label) for label, _, _ in subset], fontsize=8)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1.30), xlabel="Recall por classe")
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
        ax.grid(axis="x", linewidth=0.5, alpha=0.45)
        for pos, (_, value, support) in enumerate(subset):
            text = f"n={support:,}" if support is not None else "n indisponível"
            if value is None or not support:
                text = "Sem suporte" if support == 0 else f"Indisponível · {text}"
            ax.text((value or 0) + 0.02, pos, text, va="center", fontsize=8)
    fig.suptitle(f"Recall e suporte por classe · {scope}")
    return fig


def _retention_matrix(plt, retention):
    rows, columns = retention["checkpoints"], retention["periods"]
    width = max(8, min(17, 5 + len(columns) * 0.5))
    height = max(4.8, min(15, 3.5 + len(rows) * 0.36))
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    _heatmap(
        ax,
        retention["matrix"],
        columns,
        rows,
        x_label="Tarefa ou período histórico",
        y_label="Checkpoint",
        title="Cinza indica avaliação indisponível",
        fraction=retention.get("metric") in ("accuracy", "macro_f1", "recall"),
    )
    fig.suptitle(f"Retenção histórica · {retention.get('metric') or 'escore'}")
    return fig


def _detection_delays(plt, delays, unit, note):
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.hist(delays, bins=min(20, 1 + math.isqrt(len(delays))), color="#0072B2", edgecolor="white")
    ax.set(
        title=f"Atraso das detecções completas{note}",
        xlabel=f"Atraso ({unit})",
        ylabel="Detecções com correspondência",
    )
    return fig


_HTML_SCRIPT = """
const NS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs, parent, content) {
  const el = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (content !== undefined) el.textContent = content;
  parent.append(el);
  return el;
}
function node(tag, parent, content) {
  const el = document.createElement(tag);
  if (content !== undefined) el.textContent = content;
  parent.append(el);
  return el;
}
const format = new Intl.NumberFormat('pt-BR', {maximumSignificantDigits: 5});
const integerFormat = new Intl.NumberFormat('pt-BR', {maximumFractionDigits: 0});
function extent(values) {
  let low = Infinity, high = -Infinity;
  values.forEach(v => { low = Math.min(low, v); high = Math.max(high, v); });
  return low === high ? [low - 1, high + 1] : [low, high];
}
function renderPanels(panels, target) {
panels.forEach(panel => {
  const section = node('section', target);
  node('h2', section, panel.title);
  const legend = node('fieldset', section);
  node('legend', legend, 'Séries visíveis');
  const plot = svg('svg', {viewBox: '0 0 720 310', role: 'group', 'aria-label': panel.title}, section);
  const readout = node('output', section, 'Aponte ou selecione um ponto para ver os detalhes.');
  const visible = panel.series.flatMap(series => series.points.filter(p => p.y !== null));
  const xrange = panel.xRange || extent(panel.series.flatMap(series => series.points.map(p => p.x)).concat(
    panel.markers.map(m => m.x)
  ));
  const xr = xrange.every(Number.isFinite) ? xrange : [0, 1];
  const yr = panel.yLim || (visible.length ? extent(visible.map(p => p.y)
    .concat(panel.zero ? [0] : [])) : [0, 1]);
  const integerX = panel.integerX || panel.fitAxis;
  const formatX = integerX ? integerFormat : format;
  const px = x => 76 + 604 * (x - xr[0]) / (xr[1] - xr[0]);
  const py = y => 252 - 228 * (y - yr[0]) / (yr[1] - yr[0]);
  for (let tick = 0; tick < 6; ++tick) {
    const y = yr[0] + (yr[1] - yr[0]) * tick / 5;
    svg('line', {x1: 76, y1: py(y), x2: 680, y2: py(y), stroke: '#dce4ef'}, plot);
    svg('text', {x: 69, y: py(y) + 3, 'text-anchor': 'end'}, plot, format.format(y));
  }
  const step = integerX ? Math.max(1, Math.ceil((xr[1] - xr[0]) / 5)) : (xr[1] - xr[0]) / 5;
  const start = integerX ? Math.ceil(xr[0] / step) * step : xr[0];
  for (let index = 0; index < 6; ++index) {
    const x = start + index * step;
    if (x <= xr[1]) {
      svg('text', {x: px(x), y: 272, 'text-anchor': 'middle'}, plot, formatX.format(x));
    }
  }
  svg('line', {x1: 76, y1: 252, x2: 680, y2: 252, stroke: '#52627a'}, plot);
  svg('line', {x1: 76, y1: 24, x2: 76, y2: 252, stroke: '#52627a'}, plot);
  svg('text', {x: 378, y: 299, 'text-anchor': 'middle'}, plot, panel.xLabel);
  svg('text', {x: 15, y: 138, transform: 'rotate(-90 15 138)', 'text-anchor': 'middle'}, plot,
      panel.yLabel);
  if (panel.zero) svg('line', {x1: 76, y1: py(0), x2: 680, y2: py(0), stroke: '#52627a',
                              'stroke-dasharray': '3 3'}, plot);
  const duplicateLanes = new Map();
  panel.markers.forEach(marker => {
    if (marker.fitIndex !== undefined) {
      const lane = duplicateLanes.get(marker.x) || 0;
      duplicateLanes.set(marker.x, lane + 1);
      const point = svg('circle', {cx:px(marker.x), cy:18 + 11 * (lane % 20), r:5,
        fill:marker.color, stroke:'#fff', tabindex:0, role:'button', 'aria-label':marker.label,
        'data-fit-marker':marker.fitIndex}, plot);
      svg('title', {}, point, marker.label);
      const activate = () => panel.selectFit(marker.fitIndex);
      point.addEventListener('click', activate);
      point.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(); }
      });
      return;
    }
    const ruler = svg('line', {x1: px(marker.x), y1: 24, x2: px(marker.x), y2: 252,
      stroke: marker.color, 'stroke-dasharray': '5 4'}, plot);
    svg('title', {}, ruler, marker.label);
  });
  panel.series.forEach(series => {
    const label = node('label', legend);
    const checkbox = node('input', label);
    checkbox.type = 'checkbox';
    checkbox.checked = true;
    const swatch = node('span', label, series.name);
    swatch.style.borderColor = series.color;
    const group = svg('g', {}, plot);
    checkbox.addEventListener('change', () => { group.style.display = checkbox.checked ? '' : 'none'; });
    let path = '', connected = false;
    series.points.forEach(point => {
      if (point.y === null || (series.style === 'comparison' &&
          ['rejected','candidate'].includes(point.status))) { connected = false; return; }
      path += (connected ? 'L' : 'M') + px(point.x) + ' ' + py(point.y) + ' ';
      connected = true;
    });
    if (series.style !== 'scatter') {
      svg('path', {d: path, fill: 'none', stroke: series.color, 'stroke-width': 1.7,
                   'stroke-dasharray': series.dash}, group);
    }
    series.points.filter(p => p.y !== null).forEach(p => {
      const x = px(p.x), y = py(p.y);
      const attrs = {fill: series.color, tabindex: 0};
      const marker = p.status === 'promoted'
        ? svg('path', {...attrs, d: 'M' + x + ' ' + (y - 4) + 'L' + (x + 4) + ' ' + (y + 4) +
                   'L' + (x - 4) + ' ' + (y + 4) + 'Z'}, group)
        : svg('circle', {...attrs, cx: x, cy: y, r: series.points.length <= 50 ? 3 : 1.8,
            fill: p.status === 'rejected' ? '#fff' : series.color, stroke: series.color}, group);
      const meta = [['Ajuste',p.fit], ['Ação',p.action], ['Status',p.status], ['Versão',p.version],
        ['Checkpoint',p.checkpoint], ['Split',p.scope], ['Suporte',p.support]];
      const detail = series.name + ' = ' + format.format(p.y) + ' · ' + panel.xLabel +
        ' = ' + formatX.format(p.x) + meta.filter(entry => entry[1] !== undefined)
        .map(entry => ' · ' + entry[0] + ' = ' + entry[1]).join('');
      svg('title', {}, marker, detail);
      marker.setAttribute('aria-label', detail);
      marker.addEventListener('pointerenter', () => { readout.textContent = detail; });
      marker.addEventListener('focus', () => { readout.textContent = detail; });
    });
  });
  if (panel.series.some(series => series.retained < series.count)) {
    node('p', section, 'Séries reduzidas no desenho. Valores completos nos artefatos de dados.');
  }
});
}
"""


def _chart_data(panels):
    dash = ("none", "7 4", "7 3 2 3", "2 3")
    return [
        {
            "title": panel.title,
            "xLabel": panel.x_label,
            "yLabel": panel.y_label,
            "yLim": panel.y_lim,
            "zero": panel.zero,
            "fitAxis": panel.fit_axis,
            "integerX": panel.integer_x,
            "markers": [
                {
                    "x": m["x"],
                    "color": EVENT_STYLES[m["kind"]][0],
                    "label": m.get("label") or EVENT_STYLES[m["kind"]][2],
                }
                for m in panel.markers
            ],
            "series": [
                {
                    "name": series.name,
                    "points": series.points,
                    "style": series.style,
                    "color": _style(index)[0],
                    "dash": dash[index % len(dash)],
                    "count": series.input_count,
                    "retained": len(series.points),
                }
                for index, series in enumerate(panel.series)
            ],
        }
        for panel in panels
    ]


def _write_html(path, panels, title, *, page=None):
    data = json.dumps(_chart_data(panels), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    page_json = json.dumps(page or {}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    document = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<script>try{{var s=localStorage.getItem("cb-color-scheme");
if(s==="light"||s==="dark")document.querySelector('meta[name="color-scheme"]').content=s;
}}catch(e){{}}</script>
<title>{escape(title)}</title>
<style>
{PAGE_STYLE}
</style></head><body>
<header class="app-header">
  <div class="header-top"><h1>{escape(title)}</h1><div class="header-actions"></div></div>
</header>
<div id="controls"></div><main></main>
<noscript>Ative JavaScript para interagir ou abra a versão PNG da visão geral.</noscript>
<script type="application/json" id="chart-data">{data}</script>
<script type="application/json" id="page-data">{page_json}</script>
<script type="application/json" id="run-lineage">{{}}</script>
<script>{_HTML_SCRIPT}
{PAGE_SCRIPT}</script></body></html>
"""
    atomic_write(path, document.encode())


def _write_json(path, value):
    # The full fit can contain a long series. Stream JSON without another in-memory copy.
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".pending-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _selector(value):
    return str(value).replace("~", "~0").replace("/", "~1")


class _Publication:
    """Accumulate artifact paths and connect each figure to its complete data."""

    def __init__(self, root, observations, parent_run_id, child_run_ids, allow_html, render_static=True):
        self.root = root
        self.render_static = render_static
        self.plt = _pyplot() if render_static and matplotlib_available() else None
        self.allow_html = allow_html
        self.version = observations.get("version") or observations.get("schema_version")
        self.parent_run_id = parent_run_id
        self.child_run_ids = child_run_ids
        self.parent = []
        self.children = {}
        self.figures = []

    def artifact(self, path, fit=None):
        target = self.children.setdefault(fit["fit_id"], []) if fit else self.parent
        target.append(path)

    def chart(
        self,
        path,
        builder,
        *,
        source,
        source_pointer,
        scope=None,
        fit=None,
        additional_sources=(),
        panels=None,
        absent_panels=(),
        reason="dados necessários indisponíveis",
        html=False,
    ):
        path = path + (".html" if html else ".png")
        checkpoint = (fit.get("checkpoint") or {}) if fit else {}
        detail = _fit_detail(fit) if fit else {}
        availability = None
        if builder is None:
            availability = reason
        elif html and not self.allow_html:
            availability = "HTML indisponível neste publicador"
        elif not html and not self.render_static:
            availability = "rendering_deferred"
        elif not html and self.plt is None:
            availability = "Matplotlib indisponível"
        fit_id = detail.get("fit")
        entry = {
            "filename": path,
            "path": path if availability is None else None,
            "artifact_path": (
                f"tracking-observations/{Path(path).name if fit else path}" if availability is None else None
            ),
            "run_id": self.child_run_ids.get(fit_id) if fit_id else self.parent_run_id,
            "parent_run_id": self.parent_run_id,
            "fit_id": fit_id,
            "model_version": detail.get("version"),
            "promotion_status": detail.get("status"),
            "checkpoint": checkpoint or None,
            "scope": scope,
            "metric_version": self.version,
            "availability": {
                "status": "available" if availability is None else "unavailable",
                "reason": availability,
            },
            "source": {"artifact": source, "pointers": source_pointer},
        }
        if additional_sources:
            entry["additional_sources"] = [
                {"artifact": artifact, "pointers": pointers} for artifact, pointers in additional_sources
            ]
        if panels:
            entry["series"] = [
                {
                    "panel": panel.title,
                    "name": series.name,
                    "data_points": series.input_count,
                    "rendered_points": len(series.points),
                    "decimated": series.reduced,
                }
                for panel in panels
                for series in panel.series
            ]
        if absent_panels:
            entry["unavailable_panels"] = [
                {"title": panel.title, "reason": "sem observações finitas para as métricas do painel"}
                for panel in absent_panels
            ]
        if availability is None:
            destination = self.root / path
            if html:
                builder(destination)
            else:
                fig = builder(self.plt)
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with NamedTemporaryFile(dir=destination.parent, prefix=".pending-", delete=False) as f:
                        temporary = Path(f.name)
                    try:
                        fig.savefig(temporary, format="png", dpi=FIGURE_DPI, facecolor="white")
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                finally:
                    self.plt.close(fig)
            self.artifact(path, fit)
        elif not html and not self.render_static:
            # A changed render-mode signature must not leave a stale cached PNG that
            # adapters would mistake for an artifact of the current metric evidence.
            (self.root / path).unlink(missing_ok=True)
        self.figures.append(entry)

    def line(self, stem, panels, title, *, subtitle=None, **metadata):
        panels_with_data = [panel for panel in panels if panel.series]
        source = (
            partial(_panel_figure, panels=panels_with_data, title=title, subtitle=subtitle)
            if (panels_with_data)
            else None
        )
        self.chart(
            stem,
            source,
            panels=panels_with_data,
            absent_panels=[panel for panel in panels if not panel.series],
            **metadata,
        )

    def scoped(self, location, scopes, task, *, source, pointer, fit=None):
        for scope, scoped in scopes.items():
            # Internal scopes are known names, but reports can come from user-owned plugins.
            name = re.sub(r"[^A-Za-z0-9_-]", "_", scope)
            if name != scope:
                name += "-" + hashlib.sha256(scope.encode()).hexdigest()[:8]
            quality = _quality_entry(scoped)
            metadata = {
                "scope": scope,
                "fit": fit,
                "source": source,
                "source_pointer": [f"{pointer}/{_selector(scope)}"],
            }
            prefix = f"{location}/{name}"
            if task in ("binary", "multiclass"):
                classes, matrix = quality.get("classes"), quality.get("confusion_matrix")
                renderer = (
                    partial(_confusion, quality=quality, scope=scope)
                    if (_matrix_available(classes, matrix))
                    else None
                )
                self.chart(
                    f"{prefix}-confusion",
                    renderer,
                    **metadata,
                    reason="matriz de confusão com suporte indisponível",
                )
                recalls, supports = quality.get("class_recall"), quality.get("class_support")
                renderer = (
                    partial(_class_recall, quality=quality, scope=scope)
                    if (classes and recalls and supports and len(classes) == len(recalls) == len(supports))
                    else None
                )
                self.chart(
                    f"{prefix}-class-recall",
                    renderer,
                    **metadata,
                    reason="recall ou suporte ordenado por classe indisponível",
                )
            if task == "binary":
                for kind in ("pr", "roc"):
                    panel = _ranking_panel(quality, task, kind, scope)
                    self.line(
                        f"{prefix}-{kind}",
                        [panel] if panel.series else [],
                        "Ranqueamento binário",
                        **metadata,
                        reason="escores da classe positiva ou classes exigidas indisponíveis",
                    )
            if task == "regression":
                panel = _residual_panel(quality, scope)
                self.line(
                    f"{prefix}-residuals",
                    [panel] if panel.series else [],
                    "Erros de predição",
                    **metadata,
                    reason="resíduos ou predições preservadas indisponíveis",
                )

    def manifest(self, elapsed_seconds):
        value = {
            "version": VERSION,
            "parent_run_id": self.parent_run_id,
            "metric_version": self.version,
            "rendering_seconds": elapsed_seconds,
            "parent": self.parent,
            "children": self.children,
            "figures": self.figures,
            "static_rendering": {
                "enabled": self.render_static,
                "reason": None if self.render_static else "rendering_deferred",
            },
        }
        for fit_id, paths in self.children.items():
            prefix = f"fits/{fit_id}/"
            child_path = prefix + "plots-manifest.json"
            paths.append(child_path)
            _write_json(
                self.root / child_path,
                {
                    "version": VERSION,
                    "parent_run_id": self.parent_run_id,
                    "run_id": self.child_run_ids.get(fit_id),
                    "fit_id": fit_id,
                    "metric_version": self.version,
                    "artifacts": paths,
                    "figures": [entry for entry in self.figures if entry["fit_id"] == fit_id],
                },
            )
        self.parent.append("plots-manifest.json")
        _write_json(self.root / "plots-manifest.json", value)
        return value


def _parent_charts(publication, observations, fits, report):
    task = observations.get("task")
    windows = _window_panels(observations, fits)
    fit_quality = _fit_quality_panels(observations, fits)
    panels = windows + fit_quality + _official_overview(observations)
    title = "Visão geral do benchmark"
    metadata = {
        "source": "observations.json",
        "source_pointer": ["/scopes", "/windows"],
        "scope": "por painel",
    }
    # The adjustment comparison uses the canonical fit history retained by the parent run.
    if fit_quality:
        metadata["additional_sources"] = [
            ("../training-history.json", ["/fits/*/observation_scopes"]),
        ]
    publication.line("overview/overview", panels, title, **metadata)
    pages = page_data(report, fits, _chart_data, _window_panels, _fit_series)
    for stem, page_title, page_panels in (
        ("overview/overview", title, panels),
        ("retraining-timeline", "Timeline de retreinamentos", []),
        ("retraining-explorer", "Explorador de retreinamentos", []),
    ):
        kind = stem.rsplit("/", 1)[-1]
        publication.chart(
            stem,
            partial(_write_html, panels=page_panels, title=page_title, page=pages[kind]),
            html=True,
            panels=page_panels,
            source="observations.json",
            source_pointer=["/lineage", "/scopes", "/windows", "/drift", "/effort", "/resources"],
            additional_sources=[
                ("../training-history.json", ["/fits"]),
                ("mlflow-lineage.json", ["/children"]),
            ],
            scope="run",
        )
    publication.scoped(
        "overview", observations.get("scopes") or {}, task, source="observations.json", pointer="/scopes"
    )

    for family, title, collector in (
        ("effort", "Volume e esforço", _effort_panels),
        ("resources", "Tempo, memória, vazão e custo", _resource_panels),
    ):
        total = observations.get(family) or {}
        publication.line(
            f"overview/{family}",
            collector([], total),
            f"{title} da run",
            source="observations.json",
            source_pointer=[f"/{family}"],
            scope="run",
        )
        publication.line(
            f"overview/{family}-by-fit",
            collector(fits, {}) if fits else [],
            f"{title} por ajuste",
            source="../training-history.json",
            source_pointer=[f"/fits/*/observation_{family}"],
            scope="ajustes",
        )
    retention = observations.get("retention") or {}
    matrix, rows, columns = (retention.get(key) for key in ("matrix", "checkpoints", "periods"))
    renderer = None
    if (
        matrix
        and rows
        and columns
        and len(matrix) == len(rows)
        and all(len(row) == len(columns) for row in matrix)
    ):
        renderer = partial(_retention_matrix, retention=retention)
    publication.chart(
        "overview/retention-matrix",
        renderer,
        source="observations.json",
        source_pointer=["/retention/matrix", "/retention/checkpoints", "/retention/periods"],
        scope="holdout histórico",
        reason="avaliações históricas comparáveis indisponíveis",
    )
    panel = _retention_panel(retention)
    publication.line(
        "overview/retention",
        [panel] if panel.series else [],
        "Retenção histórica",
        source="observations.json",
        source_pointer=["/retention/series"],
        scope="holdout histórico",
        reason="séries de retenção ou esquecimento indisponíveis",
    )
    timeline, panel, delays, unit, note = _drift_panels(observations, fits)
    publication.line(
        "overview/drift",
        timeline,
        "Qualidade e eventos de drift no fluxo",
        source="observations.json",
        source_pointer=["/windows", "/drift"],
        scope="fluxo",
        additional_sources=[("../training-history.json", ["/fits"])] if fits else (),
        reason="trajetória com eventos de drift indisponível",
    )
    publication.line(
        "overview/detector-quality",
        [panel] if panel.series else [],
        "Qualidade do detector",
        source="observations.json",
        source_pointer=["/drift"],
        scope="detecção",
        reason="mudanças anotadas ou acompanhamento suficiente indisponíveis",
    )
    publication.chart(
        "overview/detection-delays",
        partial(_detection_delays, delays=delays, unit=unit, note=note) if delays else None,
        source="observations.json",
        source_pointer=["/drift/delays", "/drift/delay_unit"],
        scope="detecção",
        reason="atrasos de detecção com acompanhamento completo indisponíveis",
    )


def _child_charts(publication, observations, fit):
    fit_id = fit["fit_id"]
    if not SAFE_COMPONENT.fullmatch(fit_id):
        raise ValueError(f"invalid benchmark fit ID {fit_id!r}")
    location = f"fits/{fit_id}"
    source = "data.json"
    _write_json(publication.root / location / source, fit)
    publication.artifact(f"{location}/{source}", fit)
    status = FIT_STYLES[_fit_detail(fit)["status"]][2]
    caption = f"{fit_id} · {fit.get('action')} · {status}"
    for dimension, heading in (("step", "Curvas de step"), ("epoch", "Curvas de época")):
        reason = "treinador sem épocas observáveis" if dimension == "epoch" else "steps indisponíveis"
        publication.line(
            f"{location}/{dimension}s",
            _fit_series(fit, dimension),
            heading,
            subtitle=caption,
            source=source,
            source_pointer=[f"/observation_{dimension}s"],
            fit=fit,
            scope="por série",
            reason=reason,
        )
    for family, title, collector in (
        ("effort", "Volume e esforço", _effort_panels),
        ("resources", "Tempo, memória, vazão e custo", _resource_panels),
    ):
        publication.line(
            f"{location}/{family}",
            collector([], fit.get(f"observation_{family}") or {}),
            title,
            subtitle=caption,
            source=source,
            source_pointer=[f"/observation_{family}"],
            fit=fit,
            scope="ajuste",
        )
    publication.scoped(
        location,
        fit.get("observation_scopes") or {},
        observations.get("task"),
        source=source,
        pointer="/observation_scopes",
        fit=fit,
    )


def write_observation_figures(
    directory, report, *, parent_run_id=None, child_run_ids=None, allow_html=True, render_static=True
):
    """Stage static and interactive plots alongside explicit availability and lineage.

    Returned paths are relative to the tracking directory; the adapter logs the
    parent paths on the main run and each fit's paths on its linked child.
    """
    observations = report.get("tracking_observations")
    if observations is None:
        return {"parent": [], "children": {}, "figures": []}
    started = time.perf_counter()
    publication = _Publication(
        Path(directory), observations, parent_run_id, child_run_ids or {}, allow_html, render_static
    )
    fits = report.get("training_history") or []
    _parent_charts(publication, observations, fits, report)
    for fit in fits:
        _child_charts(publication, observations, fit)
    return publication.manifest(time.perf_counter() - started)
