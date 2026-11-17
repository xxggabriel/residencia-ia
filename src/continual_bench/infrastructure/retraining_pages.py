"""Prepare self-contained HTML views from the retained observation and fit contracts."""

import math

HTML_REVISION = 2
PAGE_PATHS = ("overview/overview.html", "retraining-timeline.html", "retraining-explorer.html")
AXES = {
    "sequence": "Evento (sequência do fluxo)",
    "epoch": "Época",
    "candidate": "Candidato (ordem do ajuste)",
    "time": "Timestamp (ms desde Unix epoch)",
}
FRACTIONS = {
    "accuracy",
    "macro_f1",
    "positive_precision",
    "positive_recall",
    "average_precision",
    "recall_at_fpr_0_01",
    "precision",
    "recall",
    "f1",
    "absolute_retention",
    "gem_bwt",
    "post_learning_fm",
    "postlearning_forgetting",
    "cl_forgetting",
    "cl_backward_transfer",
    "cl_forward_transfer",
}
INITIAL = {"initial", "initial_fit", "warmup"}
_MAX_SYSTEM_SAMPLES = 240
_SKIP_SCALAR_KEYS = {
    "confusion_matrix",
    "classes",
    "class_support",
    "class_recall",
    "pr_curve",
    "roc_curve",
    "residuals",
    "ranking_method",
    "target_unit",
    "positive_class",
}


def _coordinate(value):
    return type(value) in (int, float) and math.isfinite(value)


def _unit(metric, scope):
    if metric in FRACTIONS:
        return "proporção"
    if metric in {"mae", "rmse"}:
        return scope.get("target_unit") or "unidade do alvo"
    if metric.endswith("_seconds"):
        return "s"
    if metric.endswith("_bytes"):
        return "bytes"
    if metric.endswith("per_second"):
        return "predições/s"
    if metric == "release_delay_mean":
        return scope.get("release_delay_unit") or "unidade do relógio da fonte"
    if metric == "cost_per_1000_evaluated":
        return f"{scope.get('cost_currency') or 'unidade de custo'} / 1.000 exemplos"
    if metric.startswith("cost_"):
        return scope.get("cost_currency") or "unidade de custo do protocolo"
    if metric.endswith("_examples") or metric.endswith("_samples"):
        return "exemplos"
    if metric.endswith("_presentations"):
        return "apresentações"
    if metric in {"updates", "steps", "epochs", "alarm_count", "promoted_versions", "fit_count"}:
        return "contagem"
    return "unidade do protocolo"


def _metrics(scopes, position=None):
    for split, values in scopes.items():
        for key, value in (values or {}).items():
            reason = (values.get("availability") or {}).get(key) or {}
            if _coordinate(value) or value is None:
                # Scalar availability belongs in the table. Array/metadata keys belong in the sources.
                if key in _SKIP_SCALAR_KEYS:
                    continue
                yield [split, key, value, _unit(key, values), position, reason.get("reason")]


def _extract_confusion_matrices(scopes):
    matrices = []
    for split, values in (scopes or {}).items():
        if not isinstance(values, dict):
            continue
        matrix = values.get("confusion_matrix")
        classes = values.get("classes")
        if matrix is None or classes is None:
            continue
        matrices.append(
            {
                "split": split,
                "classes": list(classes),
                "matrix": matrix,
                "support": values.get("class_support"),
                "recall": values.get("class_recall"),
            }
        )
    return matrices


def _downsample_samples(samples, max_samples=_MAX_SYSTEM_SAMPLES):
    if not isinstance(samples, list) or len(samples) <= max_samples:
        return list(samples) if isinstance(samples, list) else []
    step = (len(samples) - 1) / (max_samples - 1)
    indices = sorted({0, len(samples) - 1, *(round(i * step) for i in range(max_samples))})
    return [samples[i] for i in indices]


def _extract_drift(observations):
    drift = observations.get("drift") or {}
    alarms = drift.get("alarms") or []
    changes = drift.get("changes") or []
    return {
        "alarms": alarms,
        "changes": changes,
        "delays": drift.get("delays") or [],
        "delay_unit": drift.get("delay_unit"),
        "alarm_count": len(alarms),
        "change_count": len(changes),
        "censored_count": drift.get("censored_count"),
        "precision": drift.get("precision"),
        "recall": drift.get("recall"),
        "f1": drift.get("f1"),
    }


def _extract_policy(base, observations, records):
    resources = observations.get("resources") or {}
    effort = observations.get("effort") or {}
    retention = observations.get("retention") or {}
    promoted = sum(1 for row in records if row.get("promotion") == "promoted")
    rejected = sum(1 for row in records if row.get("promotion") == "rejected")
    costs = {key: value for key, value in resources.items() if key.startswith("cost_")}
    retention_summary = (
        {
            "metric": retention.get("metric"),
            "checkpoints": retention.get("checkpoints") or [],
            "periods": retention.get("periods") or [],
            "series": retention.get("series") or [],
        }
        if retention
        else {}
    )
    return {
        **base,
        "effort": effort,
        "resources": resources,
        "costs": costs,
        "retention": retention_summary,
        "fit_counts": {
            "total": len(records),
            "promoted": promoted,
            "rejected": rejected,
            "other": len(records) - promoted - rejected,
        },
    }


def _extract_forgetting(observations, report):
    retention = observations.get("retention") or {}
    report_metrics = report.get("metrics") or {}
    tracking_metrics = (report.get("tracking") or {}).get("metrics") or {}

    def pick(*keys):
        for source in (retention, report_metrics, tracking_metrics):
            for key in keys:
                val = source.get(key)
                if _coordinate(val):
                    return val
        return None

    return {
        "metric": retention.get("metric"),
        "absolute_retention": pick("absolute_retention", "past_task_retention_accuracy"),
        "gem_bwt": pick("gem_bwt", "gem_backward_transfer"),
        "post_learning_fm": pick("post_learning_fm", "postlearning_forgetting"),
        "cl_forgetting": pick("cl_forgetting"),
        "cl_backward_transfer": pick("cl_backward_transfer"),
        "cl_forward_transfer": pick("cl_forward_transfer"),
        "checkpoints": retention.get("checkpoints") or [],
        "periods": retention.get("periods") or [],
        "matrix": retention.get("matrix") or [],
        "series": retention.get("series") or [],
    }


def _extract_system_metrics(report):
    sys_raw = report.get("system_metrics")
    if not isinstance(sys_raw, dict):
        return None
    raw_samples = sys_raw.get("samples") or []
    return {
        "collector": sys_raw.get("collector"),
        "summary": sys_raw.get("summary") or {},
        "total_samples": len(raw_samples) if isinstance(raw_samples, list) else 0,
        "samples": _downsample_samples(raw_samples),
    }


def _fit_record(index, fit, candidates, total_fits):
    fit_id = fit["fit_id"]
    coordinate = fit.get("promotion_step")
    if not _coordinate(coordinate):
        coordinate = fit.get("end_seq")
    axis = "sequence"
    if fit_id in candidates:
        axis, coordinate = "candidate", candidates[fit_id]
    elif not _coordinate(coordinate) and _coordinate(fit.get("end_time_ms")):
        axis, coordinate = "time", fit["end_time_ms"]
    elif not _coordinate(coordinate) and fit.get("observation_epochs") and total_fits == 1:
        axis, coordinate = "epoch", fit["observation_epochs"][-1]["epoch"]
    if not _coordinate(coordinate):
        axis, coordinate = None, None
    effort = fit.get("observation_effort") or {}
    resources = fit.get("observation_resources") or {}
    scopes = {**(fit.get("observation_scopes") or {}), "esforço": effort, "recursos": resources}
    return {
        "id": fit_id,
        "index": index,
        "initial": fit.get("action") in INITIAL or fit.get("deployment_action") in INITIAL,
        "action": fit.get("action"),
        "status": fit.get("status"),
        "promotion": fit.get("promotion_status"),
        "axis": axis,
        "x": coordinate,
        "start": fit.get("start_seq"),
        "end": fit.get("end_seq"),
        "time": fit.get("end_time_ms"),
        "checkpoint": fit.get("checkpoint"),
        "version": fit.get("deployed_version"),
        "duration": resources.get("duration_seconds", fit.get("train_seconds")),
        "volume": effort.get("training_presentations", fit.get("training_presentations")),
        "metrics": list(_metrics(scopes)),
        "confusion_matrices": _extract_confusion_matrices(fit.get("observation_scopes")),
    }


def page_data(report, fits, chart_data, window_panels, fit_series):
    observations = report["tracking_observations"]
    windows = observations.get("windows") or []
    lineage = observations.get("lineage") or {}
    parent_scopes = {
        **(observations.get("scopes") or {}),
        **{key: observations.get(key) for key in ("effort", "resources", "drift", "retention")},
    }
    base = {key: lineage.get(key) or report.get(key) for key in ("dataset", "policy", "protocol")}
    overview = list(_metrics(parent_scopes))
    for window in windows:
        coordinate = AXES.get(window.get("coordinate") or "sequence", "Coordenada do protocolo")
        overview.extend(_metrics({window["scope"]: window["quality"]}, f"{coordinate} · {window['step']}"))

    candidates = {row.get("fit_id"): row["step"] for row in windows if row.get("coordinate") == "candidate"}
    records = [_fit_record(index, fit, candidates, len(fits)) for index, fit in enumerate(fits, 1)]

    confusion_matrices = {
        "global": _extract_confusion_matrices(observations.get("scopes")),
        "by_fit": {row["id"]: row["confusion_matrices"] for row in records if row["confusion_matrices"]},
    }
    drift_data = _extract_drift(observations)
    policy_data = _extract_policy(base, observations, records)
    system_metrics = _extract_system_metrics(report)

    timeline = {}
    for axis in AXES:
        compatible = [row for row in windows if (row.get("coordinate") or "sequence") == axis]
        timeline[axis] = chart_data(window_panels({**observations, "windows": compatible}, fits))
    drift = observations.get("drift") or {}
    events = []
    for key, label, color in (("alarms", "Alarme", "#aa2335"), ("changes", "Mudança de conceito", "#5f40a1")):
        for event in drift.get(key) or []:
            x = event.get("seq") if isinstance(event, dict) else event
            if _coordinate(x):
                events.append({"x": x, "label": label, "color": color})

    fit_summaries = [{key: value for key, value in row.items() if key != "metrics"} for row in records]
    forgetting_data = _extract_forgetting(observations, report)
    return {
        "overview": {
            "kind": "overview",
            "meta": base,
            "metrics": overview,
            "fits": fit_summaries,
            "confusion_matrices": confusion_matrices,
            "drift": drift_data,
            "policy": policy_data,
            "forgetting": forgetting_data,
            "system_metrics": system_metrics,
        },
        "retraining-timeline": {
            "kind": "timeline",
            "meta": base,
            "axes": AXES,
            "events": events,
            "timeline": timeline,
            "fits": fit_summaries,
            "drift": drift_data,
            "policy": policy_data,
            "forgetting": forgetting_data,
            "system_metrics": system_metrics,
        },
        "retraining-explorer": {
            "kind": "explorer",
            "meta": base,
            "axes": AXES,
            "confusion_matrices": confusion_matrices,
            "drift": drift_data,
            "policy": policy_data,
            "forgetting": forgetting_data,
            "system_metrics": system_metrics,
            "fits": [
                {
                    **record,
                    "series": {
                        axis: chart_data([panel for panel in fit_series(fit, axis) if panel.series])
                        for axis in ("step", "epoch")
                    },
                }
                for record, fit in zip(records, fits, strict=True)
            ],
        },
    }


PAGE_STYLE = """
:root {
  color-scheme: light dark;
  --bg-light: #f4f7fb; --bg-dark: #0d131d;
  --surface-light: #ffffff; --surface-dark: #151e2c;
  --surface-alt-light: #eef4fb; --surface-alt-dark: #1c2839;
  --border-light: #cdd9e8; --border-dark: #2c3c54;
  --text-light: #10233c; --text-dark: #e6edf7;
  --muted-light: #475569; --muted-dark: #9fb0c7;
  --accent-light: #00568e; --accent-dark: #58b4ff;
  --accent-bg-light: #e5f1fc; --accent-bg-dark: #16324c;
  --ok-light: #0a6e4d; --ok-dark: #4ad295;
  --ok-bg-light: #e6f7ef; --ok-bg-dark: #113426;
  --warn-light: #9a5200; --warn-dark: #f7b84b;
  --warn-bg-light: #fff4e2; --warn-bg-dark: #38270d;
  --bad-light: #a3261b; --bad-dark: #ff7b72;
  --bad-bg-light: #fdeceb; --bad-bg-dark: #3e1917;
  --grid-light: #dce4ef; --grid-dark: #253449;

  --bg: var(--bg-light); --surface: var(--surface-light); --surface-alt: var(--surface-alt-light);
  --border: var(--border-light); --text: var(--text-light); --muted: var(--muted-light);
  --accent: var(--accent-light); --accent-bg: var(--accent-bg-light);
  --ok: var(--ok-light); --ok-bg: var(--ok-bg-light);
  --warn: var(--warn-light); --warn-bg: var(--warn-bg-light);
  --bad: var(--bad-light); --bad-bg: var(--bad-bg-light);
  --grid: var(--grid-light);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: var(--bg-dark); --surface: var(--surface-dark); --surface-alt: var(--surface-alt-dark);
    --border: var(--border-dark); --text: var(--text-dark); --muted: var(--muted-dark);
    --accent: var(--accent-dark); --accent-bg: var(--accent-bg-dark);
    --ok: var(--ok-dark); --ok-bg: var(--ok-bg-dark);
    --warn: var(--warn-dark); --warn-bg: var(--warn-bg-dark);
    --bad: var(--bad-dark); --bad-bg: var(--bad-bg-dark);
    --grid: var(--grid-dark);
  }
}
@supports (color: light-dark(white, black)) {
  :root {
    --bg: light-dark(var(--bg-light), var(--bg-dark));
    --surface: light-dark(var(--surface-light), var(--surface-dark));
    --surface-alt: light-dark(var(--surface-alt-light), var(--surface-alt-dark));
    --border: light-dark(var(--border-light), var(--border-dark));
    --text: light-dark(var(--text-light), var(--text-dark));
    --muted: light-dark(var(--muted-light), var(--muted-dark));
    --accent: light-dark(var(--accent-light), var(--accent-dark));
    --accent-bg: light-dark(var(--accent-bg-light), var(--accent-bg-dark));
    --ok: light-dark(var(--ok-light), var(--ok-dark));
    --ok-bg: light-dark(var(--ok-bg-light), var(--ok-bg-dark));
    --warn: light-dark(var(--warn-light), var(--warn-dark));
    --warn-bg: light-dark(var(--warn-bg-light), var(--warn-bg-dark));
    --bad: light-dark(var(--bad-light), var(--bad-dark));
    --bad-bg: light-dark(var(--bad-bg-light), var(--bad-bg-dark));
    --grid: light-dark(var(--grid-light), var(--grid-dark));
  }
}
:root:has(> head > meta[name="color-scheme"][content="dark"]) { color-scheme: dark; }
:root:has(> head > meta[name="color-scheme"][content="light"]) { color-scheme: light; }
html {
  accent-color: var(--accent);
  scrollbar-color: var(--border) var(--bg);
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 1520px; padding: 1.25rem 1.5rem 2.5rem;
  color: var(--text); background: var(--bg);
  font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.app-header { margin-bottom: 1rem; display: grid; gap: .75rem; }
.header-top {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .75rem;
}
.header-actions { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }
h1 { font-size: clamp(1.35rem, 2.2vw, 1.8rem); margin: 0; letter-spacing: -.01em; }
h2 { font-size: 1.05rem; margin: 0 0 .7rem; }
h3 { font-size: .95rem; margin: 0 0 .5rem; color: var(--muted); }
nav.page-nav { display: flex; flex-wrap: wrap; gap: .45rem; }
nav.page-nav a {
  text-decoration: none; color: var(--text); background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px; padding: .35rem .75rem;
  font-weight: 500; font-size: .9rem;
}
nav.page-nav a[aria-current="page"] {
  background: var(--accent-bg); color: var(--accent); border-color: var(--accent); font-weight: 600;
}
main { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 500px), 1fr)); gap: 1.15rem; }
section, .panel-card {
  background: var(--surface); border: 1px solid var(--border); padding: 1rem 1.1rem; border-radius: 10px;
}
.wide { grid-column: 1 / -1; min-width: 0; }
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 11.5rem), 1fr));
  gap: .75rem; margin-bottom: .35rem;
}
.kpi-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: .75rem .9rem; display: grid; gap: .2rem;
}
.kpi-label {
  font-size: .78rem; color: var(--muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: .03em;
}
.kpi-value { font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: .8rem; color: var(--muted); }
.badge {
  display: inline-flex; align-items: center; gap: .3rem; padding: .15rem .5rem;
  border-radius: 999px; font-size: .78rem; font-weight: 600; border: 1px solid transparent;
  white-space: nowrap;
}
.badge-ok { background: var(--ok-bg); color: var(--ok); border-color: var(--ok); }
.badge-warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn); }
.badge-bad { background: var(--bad-bg); color: var(--bad); border-color: var(--bad); }
.badge-info { background: var(--accent-bg); color: var(--accent); border-color: var(--accent); }
.badge-neutral { background: var(--surface-alt); color: var(--muted); border-color: var(--border); }

#controls, .filter-bar {
  display: flex; flex-wrap: wrap; align-items: end; gap: .65rem; margin-bottom: .85rem;
}
#controls label, .filter-bar label {
  display: grid; gap: .22rem; margin: 0; font-size: .84rem; color: var(--muted);
}
input, select, button {
  font: inherit; padding: .38rem .58rem; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface); color: var(--text); min-height: 2rem;
}
button { background: var(--surface-alt); color: var(--text); cursor: pointer; font-weight: 500; }
button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
button:disabled { opacity: .48; cursor: not-allowed; }
button:focus-visible, input:focus-visible, select:focus-visible, a:focus-visible, [tabindex]:focus-visible {
  outline: 3px solid var(--accent); outline-offset: 2px;
}
fieldset { border: 0; margin: 0 0 .4rem; padding: 0; }
legend { font-size: .82rem; color: var(--muted); }
fieldset label {
  display: inline-flex; align-items: center; gap: .3em; margin-right: .75em; font-size: .88rem;
}
fieldset label span { border-bottom: 2px solid; padding-bottom: .08em; }
svg { width: 100%; height: auto; }
svg text { font: 11px system-ui, sans-serif; fill: var(--muted); }
svg line[stroke="#dce4ef"] { stroke: var(--grid); }
svg line[stroke="#52627a"] { stroke: var(--muted); }
output {
  display: block; background: var(--surface-alt); border-radius: 6px;
  padding: .45rem .7rem; min-height: 2.2rem; overflow-wrap: anywhere; font-size: .88rem;
}
dl {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 12.5rem), 1fr));
  gap: .55rem; margin: .4rem 0;
}
dl div { background: var(--surface-alt); padding: .5rem .65rem; border-radius: 6px; overflow-wrap: anywhere; }
dt { font-weight: 600; font-size: .8rem; color: var(--muted); }
dd { margin: .15rem 0 0; font-variant-numeric: tabular-nums; }

/* Accessible Tabs */
.tabs { grid-column: 1 / -1; min-width: 0; }
[role="tablist"] {
  display: flex; overflow-x: auto; gap: .25rem; border-bottom: 1px solid var(--border);
  margin-bottom: 1rem; scroll-snap-type: inline proximity;
}
[role="tab"] {
  flex: none; scroll-snap-align: start; position: relative; border: 0; border-radius: 6px 6px 0 0;
  background: transparent; color: var(--muted); padding: .55rem .9rem; font-weight: 600; min-height: 2.5rem;
}
[role="tab"][aria-selected="true"] { color: var(--accent); }
[role="tab"][aria-selected="true"]::after {
  content: ""; position: absolute; inset: auto 0 0; border-block-end: 3px solid var(--accent);
}
[role="tabpanel"] { min-width: 0; }
.tabpanel-inner {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 500px), 1fr)); gap: 1.15rem;
}

/* Responsive & Sortable Data Table */
.table-wrapper {
  overflow: auto; max-inline-size: 100%; max-block-size: min(560px, 75vh);
  container-type: inline-size; border: 1px solid var(--border); border-radius: 8px;
}
table.data-table {
  width: 100%; border-collapse: separate; border-spacing: 0; text-align: left;
  font-variant-numeric: tabular-nums; font-size: .88rem;
}
table.data-table caption {
  position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%);
}
table.data-table th, table.data-table td {
  border-bottom: 1px solid var(--border); padding: .45rem .6rem; vertical-align: top; overflow-wrap: anywhere;
}
table.data-table thead th {
  background: var(--surface-alt); position: sticky; inset-block-start: 0; z-index: 2; font-weight: 600;
}
table.data-table th:has(> button) { padding: 0; }
table.data-table th > button {
  width: 100%; min-height: 2.25rem; border: 0; border-radius: 0; background: transparent;
  text-align: left; padding: .45rem .6rem; font-weight: 600;
  display: flex; align-items: center; justify-content: space-between; gap: .35rem;
}
table.data-table th[aria-sort="ascending"] > button::after {
  content: "▲"; font-size: .7rem; color: var(--accent);
}
table.data-table th[aria-sort="descending"] > button::after {
  content: "▼"; font-size: .7rem; color: var(--accent);
}
.table-footer {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: .6rem; margin-top: .65rem;
}
.table-footer-group { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; }
@container (width < 600px) {
  table.data-table, table.data-table thead, table.data-table tbody,
  table.data-table tr, table.data-table th, table.data-table td { display: block; }
  table.data-table thead { display: none; }
  table.data-table tr {
    margin-block-end: .85rem; border: 1px solid var(--border); border-radius: 8px; padding: .35rem;
  }
  table.data-table td::before { font-weight: 600; color: var(--muted); margin-right: .4rem; }
  table.data-table td:nth-child(1)::before { content: var(--label-1, "") ": "; }
  table.data-table td:nth-child(2)::before { content: var(--label-2, "") ": "; }
  table.data-table td:nth-child(3)::before { content: var(--label-3, "") ": "; }
  table.data-table td:nth-child(4)::before { content: var(--label-4, "") ": "; }
  table.data-table td:nth-child(5)::before { content: var(--label-5, "") ": "; }
  table.data-table td:nth-child(6)::before { content: var(--label-6, "") ": "; }
  table.data-table td:nth-child(7)::before { content: var(--label-7, "") ": "; }
  table.data-table td:nth-child(8)::before { content: var(--label-8, "") ": "; }
  table.data-table td:nth-child(9)::before { content: var(--label-9, "") ": "; }
}

/* Confusion Matrix Heatmap */
.cm-controls { display: flex; flex-wrap: wrap; align-items: center; gap: .65rem; margin-bottom: .75rem; }
.cm-grid { overflow: auto; }
table.cm-table { border-collapse: collapse; font-variant-numeric: tabular-nums; margin-bottom: .5rem; }
table.cm-table th, table.cm-table td {
  border: 1px solid var(--border); padding: .5rem .7rem; text-align: center; min-width: 4.5rem;
}
table.cm-table th { background: var(--surface-alt); font-size: .82rem; }
table.cm-table td.cm-cell { cursor: pointer; transition: outline-color .1s; font-weight: 600; }
table.cm-table td.cm-cell:hover, table.cm-table td.cm-cell:focus-visible {
  outline: 2px solid var(--accent); outline-offset: -2px;
}

.split-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 1rem;
}
.fit-list { display: grid; gap: .4rem; padding-left: 1.2rem; margin: .5rem 0; }
[data-fit-marker] { cursor: pointer; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""

PAGE_SCRIPT = """
const page = JSON.parse(document.querySelector('#page-data').textContent);
const overviewPanels = JSON.parse(document.querySelector('#chart-data').textContent);
const lineage = JSON.parse(document.querySelector('#run-lineage').textContent);
const main = document.querySelector('main');
const controls = document.querySelector('#controls');
const header = document.querySelector('header');
const headerActions = document.querySelector('.header-actions');

const names = {
  dataset: 'Dataset', policy: 'Política', protocol: 'Protocolo',
  complete: 'Concluído', interrupted: 'Interrompido', partial: 'Parcial',
  promoted: 'Promovido', rejected: 'Rejeitado', candidate: 'Candidato',
  pending: 'Pendente', initial: 'Treino inicial', initial_fit: 'Treino inicial',
  reset_retrain: 'Retreinamento', continue: 'Continuação do treino',
  learn_one: 'Aprendizado online', online: 'Atualização online', failed: 'Falhou'
};

const display = v => v === null || v === undefined || v === '' ? 'Indisponível' :
  typeof v === 'number' ? v.toLocaleString('pt-BR', {maximumSignificantDigits: 12}) :
  typeof v === 'object' ? (v.name || v.family || JSON.stringify(v)) : (names[v] || String(v));

function badge(parent, value) {
  const span = node('span', parent, display(value));
  const v = String(value || '').toLowerCase();
  const cls = ['promoted', 'complete'].includes(v) ? 'badge-ok' :
    ['rejected', 'failed'].includes(v) ? 'badge-bad' :
    ['partial', 'interrupted', 'candidate', 'pending'].includes(v) ? 'badge-warn' :
    ['initial', 'initial_fit'].includes(v) ? 'badge-info' : 'badge-neutral';
  span.className = 'badge ' + cls;
  return span;
}

// Theme toggle (2-state override following dark-mode-toggle guidance)
(function setupThemeToggle() {
  if (!headerActions) return;
  const meta = document.querySelector('meta[name="color-scheme"]');
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const systemDark = () => mq.matches;
  const renderedDark = () => meta.content === 'dark' || (meta.content === 'light dark' && systemDark());
  const btn = node('button', headerActions);
  btn.type = 'button';
  function syncLabel() {
    const dark = renderedDark();
    btn.textContent = dark ? 'Tema claro' : 'Tema escuro';
    btn.setAttribute('aria-pressed', String(meta.content !== 'light dark'));
  }
  btn.addEventListener('click', () => {
    const target = renderedDark() ? 'light' : 'dark';
    const sys = systemDark() ? 'dark' : 'light';
    if (target === sys) {
      meta.content = 'light dark';
      try { localStorage.removeItem('cb-color-scheme'); } catch (e) {}
    } else {
      meta.content = target;
      try { localStorage.setItem('cb-color-scheme', target); } catch (e) {}
    }
    syncLabel();
  });
  mq.addEventListener('change', syncLabel);
  window.addEventListener('storage', e => {
    if (e.key === 'cb-color-scheme') {
      meta.content = (e.newValue === 'light' || e.newValue === 'dark') ? e.newValue : 'light dark';
      syncLabel();
    }
  });
  syncLabel();
})();

// Cross-page navigation (semantic <nav> with <a> links, never role=tablist)
(function setupNav() {
  const nav = node('nav', header);
  nav.className = 'page-nav';
  nav.setAttribute('aria-label', 'Páginas do benchmark');
  const isOverview = page.kind === 'overview';
  const links = [
    ['overview', 'Visão Geral', isOverview ? 'overview.html' : 'overview/overview.html'],
    ['timeline', 'Timeline', isOverview ? '../retraining-timeline.html' : 'retraining-timeline.html'],
    ['explorer', 'Explorador de Treinos',
      isOverview ? '../retraining-explorer.html' : 'retraining-explorer.html']
  ];
  links.forEach(([kind, label, href]) => {
    const a = node('a', nav, label);
    a.href = href;
    if (page.kind === kind) a.setAttribute('aria-current', 'page');
  });
})();

function facts(parent, values) {
  const list = node('dl', parent);
  values.forEach(([title, value]) => {
    const item = node('div', list);
    node('dt', item, title);
    node('dd', item, display(value));
  });
}

if (page.meta && Object.keys(page.meta).length) {
  facts(header, Object.entries(page.meta).map(([k, v]) => [names[k] || k, v]));
}

function section(title, target = main) {
  const box = node('section', target);
  box.className = 'wide';
  node('h2', box, title);
  return box;
}

function field(parent, label, tag, attrs = {}) {
  const wrap = node('label', parent, label);
  const input = node(tag, wrap);
  Object.entries(attrs).forEach(([k, v]) => input.setAttribute(k, v));
  return input;
}

function action(parent, label, tap) {
  const button = node('button', parent, label);
  button.type = 'button';
  button.addEventListener('click', tap);
  return button;
}

// Top KPI summary cards
function renderKPIs(target) {
  const policy = page.policy || {};
  const drift = page.drift || {};
  const counts = policy.fit_counts || {};
  const fitsList = page.fits || [];
  const totalFits = counts.total ?? fitsList.length;
  const promoted = counts.promoted ?? fitsList.filter(f => f.promotion === 'promoted').length;
  const rejected = counts.rejected ?? fitsList.filter(f => f.promotion === 'rejected').length;
  const resources = policy.resources || {};
  const costTotal = resources.cost_total;
  const costCur = resources.cost_currency || '';
  const sys = page.system_metrics;

  const fg = page.forgetting || {};
  const retVal = fg.absolute_retention != null ? display(fg.absolute_retention) : 'Indisponível';
  const bwtVal = fg.gem_bwt != null ? `BWT ${display(fg.gem_bwt)}` : '';
  const fmRaw = fg.post_learning_fm ?? fg.cl_forgetting;
  const fmVal = fmRaw != null ? `FM ${display(fmRaw)}` : '';
  const fgSub = [bwtVal, fmVal].filter(Boolean).join(' · ') || 'Sem série de esquecimento';

  const grid = node('div', target);
  grid.className = 'kpi-grid wide';
  const cards = [
    ['Ajustes (Fits)', String(totalFits), `${promoted} promovidos · ${rejected} rejeitados`],
    ['Drift & Alarmes', String(drift.alarm_count ?? 0) + ' alarmes',
      `${drift.change_count ?? 0} mudanças` + (drift.f1 != null ? ` · F1 ${display(drift.f1)}` : '')],
    ['Retenção & Esquecimento', retVal, fgSub],
    ['Duração Total',
      resources.duration_seconds != null ? display(resources.duration_seconds) + ' s' : 'Indisponível',
      costTotal != null ? `Custo: ${display(costTotal)} ${costCur}`.trim() : 'Sem custo estimado'],
    ['Sistema', sys ? (sys.collector || 'Coletor ativo') : 'Sem coleta',
      sys ? `${sys.total_samples} amostras` : 'Métricas de sistema ausentes']
  ];
  cards.forEach(([label, value, sub]) => {
    const card = node('div', grid);
    card.className = 'kpi-card';
    node('span', card, label).className = 'kpi-label';
    node('span', card, value).className = 'kpi-value';
    node('span', card, sub).className = 'kpi-sub';
  });
}

// Accessible Tabbed Interface
function createTabs(target, idPrefix, items) {
  const wrap = node('div', target);
  wrap.className = 'tabs wide';
  const tablist = node('div', wrap);
  tablist.setAttribute('role', 'tablist');
  tablist.setAttribute('focusgroup', 'tablist nomemory');
  tablist.setAttribute('aria-label', 'Seções da página');

  const tabs = [];
  const panels = [];

  function activate(idx, moveFocus = false) {
    tabs.forEach((t, i) => {
      const sel = i === idx;
      t.setAttribute('aria-selected', String(sel));
      t.tabIndex = sel ? 0 : -1;
      t.toggleAttribute('focusgroupstart', sel);
      if (sel) {
        panels[i].removeAttribute('hidden');
        panels[i].tabIndex = 0;
        t.scrollIntoView({block: 'nearest', inline: 'nearest'});
      } else {
        panels[i].hidden = 'until-found';
        panels[i].removeAttribute('tabindex');
      }
    });
    if (moveFocus) tabs[idx].focus();
  }

  items.forEach((item, idx) => {
    const tab = node('button', tablist, item.label);
    tab.type = 'button';
    tab.id = `${idPrefix}-tab-${idx}`;
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-selected', idx === 0 ? 'true' : 'false');
    tab.tabIndex = idx === 0 ? 0 : -1;
    if (idx === 0) tab.setAttribute('focusgroupstart', '');

    const panel = node('div', wrap);
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', tab.id);
    if (idx !== 0) panel.hidden = 'until-found';
    else panel.tabIndex = 0;

    const inner = node('div', panel);
    inner.className = 'tabpanel-inner';

    tab.addEventListener('click', () => activate(idx, false));
    tab.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight') { e.preventDefault(); activate((idx + 1) % tabs.length, true); }
      else if (e.key === 'ArrowLeft') {
        e.preventDefault(); activate((idx - 1 + tabs.length) % tabs.length, true);
      }
      else if (e.key === 'Home') { e.preventDefault(); activate(0, true); }
      else if (e.key === 'End') { e.preventDefault(); activate(tabs.length - 1, true); }
    });
    panel.addEventListener('beforematch', () => activate(idx, false));

    tabs.push(tab);
    panels.push(panel);
    item.render(inner);
  });
  return wrap;
}

// Export helper
function downloadFile(filename, mime, content) {
  const blob = new Blob([content], {type: mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Advanced responsive + sortable + filterable + exportable table
function table(parent, headers, rows, options = {}) {
  const searchable = typeof options === 'boolean' ? options : Boolean(options.searchable);
  const filterCols = (typeof options === 'object' && options.filterCols) ? options.filterCols : [];
  const captionText = (typeof options === 'object' && options.caption) || 'Tabela de dados';
  const exportName = (typeof options === 'object' && options.exportName) || 'dados';
  const badgeCols = new Set((typeof options === 'object' && options.badgeCols) || []);

  const bar = node('div', parent);
  bar.className = 'filter-bar';

  const search = searchable
    ? field(bar, 'Buscar', 'input', {type: 'search', placeholder: 'Filtrar texto...'})
    : null;
  const selectFilters = filterCols.map(({col, label}) => {
    const sel = field(bar, label, 'select');
    const opts = [...new Set(
      rows.map(r => r[col]).filter(v => v !== null && v !== undefined && v !== '')
    )];
    [['', 'Todos'], ...opts.map(v => [String(v), display(v)])].forEach(([val, txt]) => {
      const o = node('option', sel, txt);
      o.value = val;
    });
    return {col, sel};
  });

  const availSel = (typeof options === 'object' && options.availabilityCol !== undefined)
    ? (() => {
        const sel = field(bar, 'Disponibilidade', 'select');
        [['', 'Todos'], ['available', 'Disponíveis'], ['unavailable', 'Indisponíveis']]
          .forEach(([v, t]) => {
            const o = node('option', sel, t); o.value = v;
          });
        return {col: options.availabilityCol, sel};
      })()
    : null;

  let pageSize = searchable ? 25 : rows.length || 100;
  let sizeSel = null;
  if (searchable) {
    sizeSel = field(bar, 'Linhas/página', 'select');
    [['25', '25'], ['50', '50'], ['100', '100'], ['all', 'Todos']].forEach(([v, t]) => {
      const o = node('option', sizeSel, t); o.value = v;
    });
  }

  const wrapId = 'tbl-' + Math.random().toString(36).slice(2, 8);
  const scroll = node('div', parent);
  scroll.className = 'table-wrapper';
  scroll.tabIndex = 0;
  scroll.setAttribute('role', 'region');
  scroll.setAttribute('aria-labelledby', wrapId);

  const grid = node('table', scroll);
  grid.className = 'data-table';
  headers.forEach((h, idx) => {
    const lbl = `"${String(h).replace(/"/g, '\\"')}"`;
    grid.style.setProperty(`--label-${idx + 1}`, lbl);
  });
  const cap = node('caption', grid, captionText);
  cap.id = wrapId;
  const head = node('tr', node('thead', grid));
  const body = node('tbody', grid);
  const collator = new Intl.Collator('pt-BR', {numeric: true, sensitivity: 'base'});
  const thCells = [];
  let sort = -1, direction = 1, offset = 0;
  let previous, next, pagination;
  let lastFiltered = rows;

  function filterAndSort() {
    const q = search ? search.value.trim().toLocaleLowerCase('pt-BR') : '';
    let items = rows.filter(row => {
      if (q && !row.some(v => display(v).toLocaleLowerCase('pt-BR').includes(q))) return false;
      for (const {col, sel} of selectFilters) {
        if (sel.value && String(row[col] ?? '') !== sel.value) return false;
      }
      if (availSel && availSel.sel.value) {
        const val = row[availSel.col];
        const isAvail = val !== null && val !== undefined && val !== '';
        if (availSel.sel.value === 'available' && !isAvail) return false;
        if (availSel.sel.value === 'unavailable' && isAvail) return false;
      }
      return true;
    });
    if (sort >= 0) {
      items = [...items].sort((a, b) => {
        const x = a[sort], y = b[sort];
        if (x == null || y == null) return x == null && y == null ? 0 : x == null ? 1 : -1;
        return direction * (typeof x === 'number' && typeof y === 'number'
          ? x - y
          : collator.compare(display(x), display(y)));
      });
    }
    return items;
  }

  function draw() {
    lastFiltered = filterAndSort();
    const total = lastFiltered.length;
    const limit = sizeSel && sizeSel.value === 'all' ? Math.max(1, total) : pageSize;
    if (offset >= total) offset = 0;

    body.replaceChildren();
    if (!total) {
      const msg = rows.length ? 'Nenhum resultado para os filtros.' : 'Nenhum registro.';
      const cell = node('td', node('tr', body), msg);
      cell.colSpan = headers.length;
    } else {
      const slice = searchable ? lastFiltered.slice(offset, offset + limit) : lastFiltered;
      slice.forEach(row => {
        const tr = node('tr', body);
        row.forEach((value, colIdx) => {
          const td = node('td', tr);
          if (badgeCols.has(colIdx) && value != null && value !== '') badge(td, value);
          else td.textContent = display(value);
        });
      });
    }

    if (searchable) {
      previous.disabled = offset === 0;
      next.disabled = offset + limit >= total;
      pagination.textContent = total
        ? `${offset + 1}–${Math.min(total, offset + limit)} de ${total}`
        : '0 resultados';
    }
  }

  headers.forEach((name, index) => {
    const th = node('th', head);
    th.scope = 'col';
    thCells.push(th);
    if (!searchable) { th.textContent = name; return; }
    action(th, name, () => {
      direction = sort === index ? -direction : 1;
      sort = index;
      offset = 0;
      thCells.forEach(cell => cell.removeAttribute('aria-sort'));
      th.setAttribute('aria-sort', direction === 1 ? 'ascending' : 'descending');
      draw();
    });
  });

  if (searchable) {
    const footer = node('div', parent);
    footer.className = 'table-footer';
    const navGrp = node('div', footer);
    navGrp.className = 'table-footer-group';
    previous = action(navGrp, 'Página anterior', () => {
      const limit = sizeSel.value === 'all' ? lastFiltered.length : pageSize;
      offset = Math.max(0, offset - limit);
      draw();
    });
    next = action(navGrp, 'Próxima página', () => {
      const limit = sizeSel.value === 'all' ? lastFiltered.length : pageSize;
      offset += limit;
      draw();
    });
    pagination = node('output', navGrp);
    pagination.setAttribute('role', 'status');

    const expGrp = node('div', footer);
    expGrp.className = 'table-footer-group';
    action(expGrp, 'Exportar CSV', () => {
      const esc = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
      const csv = [headers.map(esc).join(',')]
        .concat(lastFiltered.map(r => r.map(v => esc(display(v))).join(',')))
        .join('\\n');
      downloadFile(exportName + '.csv', 'text/csv;charset=utf-8', csv);
    });
    action(expGrp, 'Exportar JSON', () => {
      const objs = lastFiltered.map(r => Object.fromEntries(headers.map((h, i) => [h, r[i]])));
      downloadFile(
        exportName + '.json', 'application/json;charset=utf-8', JSON.stringify(objs, null, 2)
      );
    });

    search.addEventListener('input', () => { offset = 0; draw(); });
    selectFilters.forEach(({sel}) => sel.addEventListener('change', () => { offset = 0; draw(); }));
    if (availSel) availSel.sel.addEventListener('change', () => { offset = 0; draw(); });
    sizeSel.addEventListener('change', () => {
      if (sizeSel.value !== 'all') pageSize = Number(sizeSel.value);
      offset = 0;
      draw();
    });
  }
  draw();
}

const metricHeaders = [
  'Split ou grupo', 'Métrica', 'Valor', 'Unidade', 'Posição', 'Motivo de indisponibilidade'
];

// Interactive Confusion Matrix Viewer
function renderConfusionMatrices(target, cmData, fixedFitId = null) {
  const box = section('Matrizes de Confusão', target);
  const globalList = (cmData && cmData.global) || [];
  const byFit = (cmData && cmData.by_fit) || {};
  const entries = [];
  if (!fixedFitId) {
    globalList.forEach(item => {
      entries.push({key: `global:${item.split}`, label: `Global · ${item.split}`, ...item});
    });
    Object.entries(byFit).forEach(([fitId, list]) => {
      list.forEach(item => {
        entries.push({key: `${fitId}:${item.split}`, label: `${fitId} · ${item.split}`, ...item});
      });
    });
  } else {
    (byFit[fixedFitId] || []).forEach(item => {
      entries.push({key: `${fixedFitId}:${item.split}`, label: item.split, ...item});
    });
  }

  if (!entries.length) {
    node('p', box, 'Nenhuma matriz de confusão disponível.');
    return;
  }

  const ctrl = node('div', box);
  ctrl.className = 'cm-controls';
  const sel = field(ctrl, 'Escopo / Split', 'select');
  entries.forEach((e, i) => {
    const o = node('option', sel, e.label);
    o.value = String(i);
  });
  const modeSel = field(ctrl, 'Exibição', 'select');
  [['count', 'Contagens absolutas'], ['recall', 'Normalizado por linha (% Recall)']].forEach(([v, t]) => {
    const o = node('option', modeSel, t);
    o.value = v;
  });

  const gridWrap = node('div', box);
  gridWrap.className = 'cm-grid';
  const readout = node('output', box, 'Passe o cursor ou foque uma célula para detalhes.');

  function draw() {
    gridWrap.replaceChildren();
    const item = entries[Number(sel.value) || 0];
    const classes = item.classes || [];
    const matrix = item.matrix || [];
    const rowSums = matrix.map(row => row.reduce((a, b) => a + (Number(b) || 0), 0));
    const maxCount = Math.max(1, ...matrix.flat().map(v => Number(v) || 0));
    const norm = modeSel.value === 'recall';

    const tbl = node('table', gridWrap);
    tbl.className = 'cm-table';
    const thead = node('thead', tbl);
    const hr = node('tr', thead);
    node('th', hr, 'Real \\\\ Predito').scope = 'col';
    classes.forEach(c => { const th = node('th', hr, String(c)); th.scope = 'col'; });
    node('th', hr, 'Suporte').scope = 'col';

    const tbody = node('tbody', tbl);
    matrix.forEach((row, rIdx) => {
      const tr = node('tr', tbody);
      const rh = node('th', tr, String(classes[rIdx] ?? rIdx));
      rh.scope = 'row';
      const sum = rowSums[rIdx] || 0;
      row.forEach((val, cIdx) => {
        const count = Number(val) || 0;
        const pct = sum > 0 ? (count / sum) * 100 : 0;
        const intensity = norm ? (pct / 100) : (count / maxCount);
        const alpha = (0.08 + 0.55 * intensity).toFixed(2);
        const text = norm ? `${pct.toFixed(1)}%` : display(count);
        const td = node('td', tr, text);
        td.className = 'cm-cell';
        td.tabIndex = 0;
        td.style.backgroundColor = `rgba(0, 118, 192, ${alpha})`;
        const detail = `Real: ${classes[rIdx]} · Predito: ${classes[cIdx]} · ` +
          `Contagem: ${count} · Recall da linha: ${pct.toFixed(2)}%`;
        td.setAttribute('aria-label', detail);
        td.addEventListener('pointerenter', () => { readout.textContent = detail; });
        td.addEventListener('focus', () => { readout.textContent = detail; });
      });
      node('td', tr, display(sum));
    });
  }

  sel.addEventListener('change', draw);
  modeSel.addEventListener('change', draw);
  draw();
}

// Drift & Policy Panel
function renderDriftAndPolicy(target) {
  const drift = page.drift || {};
  const policy = page.policy || {};

  const dBox = section('Monitoramento de Drift e Alarmes', target);
  dBox.className = '';
  facts(dBox, [
    ['Alarmes', drift.alarm_count ?? 0],
    ['Mudanças reais', drift.change_count ?? 0],
    ['Precisão do detector', drift.precision],
    ['Recall do detector', drift.recall],
    ['F1 do detector', drift.f1],
    ['Censurados', drift.censored_count],
    ['Unidade de atraso', drift.delay_unit],
    ['Atrasos registrados', (drift.delays || []).length ? drift.delays.map(display).join(', ') : null]
  ]);
  if ((drift.alarms || []).length || (drift.changes || []).length) {
    const rows = [
      ...(drift.alarms || []).map(a => ['Alarme', typeof a === 'object' ? (a.seq ?? JSON.stringify(a)) : a]),
      ...(drift.changes || []).map(c => [
        'Mudança de conceito', typeof c === 'object' ? (c.seq ?? JSON.stringify(c)) : c
      ])
    ];
    table(dBox, ['Evento', 'Posição / Detalhe'], rows, {caption: 'Eventos de drift', badgeCols: [0]});
  }

  const pBox = section('Política, Custos e Retenção', target);
  pBox.className = '';
  const counts = policy.fit_counts || {};
  const res = policy.resources || {};
  const eff = policy.effort || {};
  const ret = policy.retention || {};
  facts(pBox, [
    ['Dataset', policy.dataset],
    ['Política', policy.policy],
    ['Protocolo', policy.protocol],
    ['Ajustes promovidos', counts.promoted],
    ['Ajustes rejeitados', counts.rejected],
    ['Exemplos únicos de treino', eff.unique_training_examples],
    ['Apresentações totais', eff.training_presentations],
    ['Duração total (s)', res.duration_seconds],
    ['Tipo de custo', res.cost_kind],
    ['Moeda', res.cost_currency],
    ['Custo total', res.cost_total],
    ['Custo / 1.000 avaliados', res.cost_per_1000_evaluated],
    ['Métrica de retenção', ret.metric],
    ['Checkpoints na retenção', (ret.checkpoints || []).length]
  ]);
}

// Forgetting & Retention Panels
function renderForgetting(target) {
  const fg = page.forgetting || {};
  const box = section('Métricas de Esquecimento e Retenção', target);
  const hasAny = fg.absolute_retention != null || fg.gem_bwt != null || fg.post_learning_fm != null ||
    fg.cl_forgetting != null || (fg.series || []).length || (fg.matrix || []).length;
  if (!hasAny) {
    node('p', box, 'Métricas de esquecimento e retenção histórica indisponíveis nesta execução.');
    return;
  }
  facts(box, [
    ['Métrica base', fg.metric],
    ['Retenção absoluta (absolute_retention)', fg.absolute_retention],
    ['GEM Backward Transfer (gem_bwt)', fg.gem_bwt],
    ['Forgetting Measure pós-aprendizado (post_learning_fm)', fg.post_learning_fm],
    ['CL Forgetting (cl_forgetting)', fg.cl_forgetting],
    ['CL Backward Transfer (cl_backward_transfer)', fg.cl_backward_transfer],
    ['CL Forward Transfer (cl_forward_transfer)', fg.cl_forward_transfer],
    ['Checkpoints avaliados', (fg.checkpoints || []).length],
    ['Períodos avaliados', (fg.periods || []).length]
  ]);

  const series = fg.series || [];
  if (series.length) {
    const keys = [
      ['absolute_retention', 'Retenção absoluta', '#007a62'],
      ['gem_bwt', 'GEM BWT', '#00568e'],
      ['post_learning_fm', 'FM pós-aprendizado', '#aa2335']
    ].filter(([k]) => series.some(row => typeof row[k] === 'number'));
    if (keys.length) {
      const panel = {
        title: 'Evolução de Retenção e Esquecimento',
        xLabel: 'Checkpoint (step)',
        yLabel: 'Valor',
        integerX: true,
        markers: [],
        series: keys.map(([k, label, color], idx) => ({
          name: label,
          color,
          dash: idx === 0 ? '' : '4 3',
          style: 'line',
          points: series
            .filter(r => typeof r[k] === 'number' && Number.isFinite(r[k]))
            .map(r => ({x: r.step, y: r[k]}))
        }))
      };
      renderPanels([panel], target);
    }
    const sHeaders = ['Checkpoint (step)', 'Retenção absoluta', 'GEM BWT', 'FM pós-aprendizado'];
    const sRows = series.map(r => [r.step, r.absolute_retention, r.gem_bwt, r.post_learning_fm]);
    table(section('Série de Esquecimento por Checkpoint', target), sHeaders, sRows, {
      searchable: true,
      caption: 'Série de esquecimento',
      exportName: 'esquecimento-serie'
    });
  }

  const matrix = fg.matrix || [];
  const rowsLabels = fg.checkpoints || [];
  const colLabels = fg.periods || [];
  if (matrix.length && rowsLabels.length && colLabels.length) {
    const mBox = section('Matriz de Retenção Histórica (Checkpoints × Períodos)', target);
    const info = node('output', mBox, 'Passe o mouse ou foque uma célula para ver o valor exato.');
    const wrap = node('div', mBox);
    wrap.className = 'cm-grid';
    const tbl = node('table', wrap);
    tbl.className = 'cm-table';
    const thead = node('thead', tbl);
    const hr = node('tr', thead);
    node('th', hr, 'Checkpoint \\ Período').scope = 'col';
    colLabels.forEach(col => { node('th', hr, String(col)).scope = 'col'; });
    const tbody = node('tbody', tbl);
    matrix.forEach((row, rIdx) => {
      const tr = node('tr', tbody);
      node('th', tr, String(rowsLabels[rIdx] ?? rIdx)).scope = 'row';
      row.forEach((val, cIdx) => {
        const td = node('td', tr);
        td.tabIndex = 0;
        if (typeof val === 'number' && Number.isFinite(val)) {
          const clamped = Math.max(0, Math.min(1, val));
          td.style.background = `rgba(0, 122, 98, ${(0.08 + 0.72 * clamped).toFixed(3)})`;
          td.style.color = clamped > 0.55 ? '#ffffff' : 'inherit';
          td.textContent = (val * 100).toFixed(1) + '%';
        } else {
          td.textContent = '—';
        }
        const announce = () => {
          const cp = rowsLabels[rIdx] ?? rIdx;
          const per = colLabels[cIdx] ?? cIdx;
          info.textContent = `Checkpoint ${cp} · Período ${per} · Valor: ${display(val)}`;
        };
        td.addEventListener('mouseenter', announce);
        td.addEventListener('focus', announce);
      });
    });
  }
}

// System Metrics Charts & Summary
function renderSystemMetrics(target) {
  const sys = page.system_metrics;
  const box = section('Métricas de Sistema', target);
  if (!sys) {
    node('p', box, 'Nenhuma métrica de sistema registrada nesta execução.');
    return;
  }
  facts(box, [
    ['Coletor', sys.collector],
    ['Total de amostras', sys.total_samples],
    ...Object.entries(sys.summary || {}).map(([k, v]) => [k, v])
  ]);
  const samples = sys.samples || [];
  if (!samples.length) return;
  const numericKeys = Object.keys(samples[0]).filter(k =>
    k !== 'timestamp' && k !== 'time' && k !== 'step' && samples.some(s => typeof s[k] === 'number')
  );
  const colors = ['#00568e', '#007a62', '#aa2335', '#5f40a1', '#9a5200'];
  const panels = numericKeys.slice(0, 6).map((key, idx) => ({
    title: `Sistema · ${key}`,
    xLabel: samples[0].step !== undefined ? 'Amostra / Step' : 'Índice da amostra',
    yLabel: key,
    yLim: null,
    zero: true,
    fitAxis: false,
    integerX: true,
    markers: [],
    series: [{
      name: key,
      style: 'line',
      color: colors[idx % colors.length],
      dash: 'none',
      count: sys.total_samples,
      retained: samples.length,
      points: samples.map((s, i) => ({
        x: typeof s.step === 'number' ? s.step : i + 1,
        y: typeof s[key] === 'number' ? s[key] : null
      }))
    }]
  }));
  if (panels.length) renderPanels(panels, target);
}

// Multi-Training Comparison Table
function renderFitsComparison(target, fitsList, onSelectFit = null) {
  const box = section('Comparação de Treinos (Fits)', target);
  if (!fitsList.length) {
    node('p', box, 'Nenhum ajuste registrado.');
    return;
  }
  const headers = [
    'Ordem', 'Ajuste (fit_id)', 'Tipo', 'Status', 'Promoção',
    'Versão', 'Duração (s)', 'Volume', 'Run filha (ID)'
  ];
  const rows = fitsList.map(f => [
    f.index,
    f.id,
    f.initial ? 'Treino inicial' : f.action,
    f.status,
    f.promotion,
    f.version,
    f.duration,
    f.volume,
    lineage.children?.[f.id]
  ]);
  table(box, headers, rows, {
    searchable: true,
    caption: 'Comparação de treinos',
    exportName: 'comparacao-fits',
    filterCols: [{col: 3, label: 'Status'}, {col: 4, label: 'Promoção'}],
    badgeCols: [3, 4]
  });
  if (onSelectFit) {
    const list = node('div', box);
    list.className = 'filter-bar';
    fitsList.forEach(f => action(list, `Inspecionar ${f.id}`, () => onSelectFit(f.index)));
  }
}

function describe(box, fit) {
  const type = fit.initial ? 'Treino inicial' : fit.action;
  const badges = node('div', box);
  badges.className = 'filter-bar';
  badge(badges, fit.status);
  badge(badges, fit.promotion);
  badge(badges, type);
  facts(box, [
    ['Ajuste', fit.id],
    ['Posição no histórico', fit.index],
    ['Tipo', type],
    ['Status', fit.status],
    ['Promoção', fit.promotion],
    ['Posição no fluxo', fit.axis ? (page.axes[fit.axis] + ' · ' + display(fit.x)) : null],
    ['Intervalo de treino',
      fit.start == null && fit.end == null ? null : display(fit.start) + ' → ' + display(fit.end)],
    ['Fim registrado (UTC)', fit.time == null ? null : new Date(fit.time).toISOString()],
    ['Versão promovida', fit.version],
    ['Run filha (ID)', lineage.children?.[fit.id]],
    ['Checkpoint', fit.checkpoint?.file],
    ['SHA-256 do checkpoint', fit.checkpoint?.sha256],
    ['Duração (s)', fit.duration],
    ['Apresentações ao modelo', fit.volume]
  ]);
}

function explore() {
  renderKPIs(main);
  const fits = page.fits;
  const details = section('Ajuste selecionado');
  const cmTarget = node('div', main);
  cmTarget.className = 'wide';
  const plots = node('div', main);
  plots.className = 'wide';

  const bounds = {min: 1, max: Math.max(1, fits.length), step: 1};
  const fitPicker = field(controls, 'Selecionar ajuste (fit_id)', 'select');
  fits.forEach(f => {
    const o = node('option', fitPicker, `${f.index}. ${f.id} (${display(f.promotion)})`);
    o.value = String(f.index);
  });
  const slider = field(controls, 'Posição no histórico', 'input', {type: 'range', ...bounds});
  const direct = field(controls, 'Ir para a posição', 'input', {type: 'number', ...bounds});
  const state = node('output', controls);
  state.setAttribute('aria-live', 'polite');
  fitPicker.disabled = slider.disabled = direct.disabled = fits.length < 2;

  function show(index) {
    details.replaceChildren();
    cmTarget.replaceChildren();
    plots.replaceChildren();
    if (!fits.length) {
      node('h2', details, 'Ajuste selecionado');
      state.textContent = 'Nenhum ajuste registrado.';
      return;
    }
    const fit = fits[index - 1];
    fitPicker.value = slider.value = direct.value = String(index);
    state.textContent = `${index} de ${fits.length} · ${fit.id}`;
    node('h2', details, `Ajuste selecionado · ${fit.id}`);
    describe(details, fit);
    table(details, metricHeaders, fit.metrics, {
      searchable: true,
      caption: `Métricas de ${fit.id}`,
      exportName: `${fit.id}-metricas`,
      filterCols: [{col: 0, label: 'Split/Grupo'}, {col: 3, label: 'Unidade'}],
      availabilityCol: 2
    });
    if ((fit.confusion_matrices || []).length) {
      renderConfusionMatrices(cmTarget, page.confusion_matrices, fit.id);
    }
    Object.entries(fit.series).forEach(([axis, panels]) => {
      if (panels.length) renderPanels(panels, plots);
      else {
        const msg = 'Série por ' + (axis === 'step' ? 'step' : 'época') + ' indisponível neste ajuste.';
        node('p', plots, msg);
      }
    });
  }

  fitPicker.addEventListener('change', () => show(Number(fitPicker.value)));
  slider.addEventListener('input', () => show(Number(slider.value)));
  const select = () => { if (direct.value && direct.validity.valid) show(Number(direct.value)); };
  direct.addEventListener('input', select);
  direct.addEventListener('change', () => {
    show(Math.max(1, Math.min(fits.length, Math.round(Number(direct.value) || 1))));
  });
  show(1);
}

function timeline() {
  renderKPIs(main);
  const fits = page.fits;
  const status = field(controls, 'Status ou promoção', 'select');
  const statuses = [...new Set(fits.flatMap(f => [f.status, f.promotion]).filter(Boolean))];
  [['', 'Todos'], ...statuses.map(v => [v, display(v)])]
    .forEach(([value, label]) => { const option = node('option', status, label); option.value = value; });
  const axis = field(controls, 'Eixo do intervalo', 'select');
  [['', 'Todos os eixos'], ...Object.entries(page.axes)].forEach(([value, label]) => {
    const option = node('option', axis, label); option.value = value;
  });
  const lo = field(controls, 'De', 'input', {type: 'number', step: 'any'});
  const hi = field(controls, 'Até', 'input', {type: 'number', step: 'any'});

  const selection = section('Detalhes do ajuste');
  node('p', selection, 'Selecione um marcador ou um ajuste na lista.');
  const charts = node('div', main);
  charts.className = 'wide';
  const records = section('Ajustes no recorte');

  const select = index => {
    selection.replaceChildren();
    node('h2', selection, 'Detalhes do ajuste');
    describe(selection, fits[index - 1]);
  };
  function limits(values) {
    return values.map(input => input.value === '' ? null : Number(input.value));
  }
  function draw() {
    charts.replaceChildren();
    records.replaceChildren();
    selection.replaceChildren();
    node('h2', selection, 'Detalhes do ajuste');
    node('p', selection, 'Selecione um marcador ou um ajuste na lista.');
    const [a, b] = limits([lo, hi]);
    const between = x => Number.isFinite(x) && (a == null || x >= a) && (b == null || x <= b);
    const rows = fits.filter(f => (!status.value || [f.status, f.promotion].includes(status.value)) &&
      (!axis.value || f.axis === axis.value) && (a == null && b == null || between(f.x)));
    node('h2', records, 'Ajustes no recorte · ' + rows.length + ' de ' + fits.length);
    if (!rows.length) {
      node('p', records, fits.length ? 'Nenhum ajuste neste recorte.' : 'Nenhum ajuste registrado.');
    }
    const list = node('ol', records);
    list.className = 'fit-list';
    rows.forEach(f => {
      const item = node('li', list);
      action(item, (f.initial ? 'Treino inicial · ' : '') + f.id, () => select(f.index));
      node('span', item, ' · ');
      badge(item, f.status);
      node('span', item, ' ');
      badge(item, f.promotion);
      const pos = f.axis ? page.axes[f.axis] + ' · ' + display(f.x) : 'Posição indisponível';
      node('span', item, ' · ' + pos);
    });
    table(records, ['Ordem', 'Ajuste', 'Tipo', 'Status', 'Promoção', 'Eixo', 'Posição', 'Run filha'],
      rows.map(f => [f.index, f.id, f.initial ? 'Treino inicial' : f.action, f.status, f.promotion,
        page.axes[f.axis], f.x, lineage.children?.[f.id]]),
      {searchable: true, caption: 'Ajustes no recorte', exportName: 'timeline-fits', badgeCols: [3, 4]});

    Object.entries(page.axes).forEach(([kind, label]) => {
      if (axis.value && kind !== axis.value) return;
      const markers = rows.filter(f => f.axis === kind).map(f => ({
        x: f.x, fitIndex: f.index,
        label: (f.initial ? 'Treino inicial · ' : '') + f.id + ' · ' + display(f.promotion),
        color: f.promotion === 'rejected' ? '#a33224' : f.initial ? '#00568e' : '#007a62'
      })).concat(kind === 'sequence' ? page.events.filter(e => between(e.x)) : []);
      const panels = page.timeline[kind];
      if (!panels.length && !markers.length) return;
      const metricPositions = panels.flatMap(p => p.series.flatMap(s => s.points.map(pt => pt.x)));
      const xValues = markers.map(m => m.x).concat(metricPositions);
      const [left, right] = extent(xValues);
      const min = a ?? left, max = b ?? right;
      if (!(min <= max) || ![min, max].every(Number.isFinite)) return;
      const xRange = min === max ? [min - 1, max + 1] : [min, max];
      const basePanels = panels.length
        ? panels
        : [{title: 'Ajustes e eventos', series: [], yLim: [0, 1], yLabel: 'Eventos'}];
      renderPanels(basePanels.map(p => ({
        ...p, xLabel: label, integerX: true, xRange, markers, selectFit: select,
        series: p.series.map(s => ({...s, points: s.points.filter(pt => between(pt.x))}))
      })), charts);
    });
  }
  function axisChange() {
    lo.disabled = hi.disabled = !axis.value;
    lo.value = hi.value = '';
    draw();
  }
  axis.addEventListener('change', axisChange);
  status.addEventListener('change', draw);
  [lo, hi].forEach(input => input.addEventListener('change', draw));
  action(controls, 'Ampliar ×2', () => {
    if (!axis.value) {
      const preferred = fits.find(f => f.axis)?.axis;
      if (!preferred) return;
      axis.value = preferred;
      axisChange();
    }
    const values = fits.filter(f => f.axis === axis.value).map(f => f.x);
    if (!values.length) return;
    const [a, b] = extent(values), [low, high] = limits([lo, hi]);
    const start = low ?? a, end = high ?? b, quarter = (end - start) / 4;
    lo.value = start + quarter;
    hi.value = end - quarter;
    draw();
  });
  action(controls, 'Restaurar intervalo', () => { lo.value = hi.value = ''; draw(); });
  axisChange();
}

function overview() {
  renderKPIs(main);
  createTabs(main, 'ov', [
    {
      label: 'Métricas & Gráficos',
      render: target => {
        if (page.metrics) {
          table(section('Resultados da execução', target), metricHeaders, page.metrics, {
            searchable: true,
            caption: 'Resultados da execução',
            exportName: 'metricas-execucao',
            filterCols: [{col: 0, label: 'Split ou grupo'}, {col: 3, label: 'Unidade'}],
            availabilityCol: 2
          });
        }
        renderPanels(overviewPanels, target);
      }
    },
    {
      label: 'Treinos (Fits)',
      render: target => renderFitsComparison(target, page.fits || [])
    },
    {
      label: 'Matrizes de Confusão',
      render: target => renderConfusionMatrices(target, page.confusion_matrices)
    },
    {
      label: 'Drift & Política',
      render: target => renderDriftAndPolicy(target)
    },
    {
      label: 'Esquecimento & Retenção',
      render: target => renderForgetting(target)
    },
    {
      label: 'Métricas de Sistema',
      render: target => renderSystemMetrics(target)
    }
  ]);
}

if (page.kind === 'explorer') explore();
else if (page.kind === 'timeline') timeline();
else overview();
"""
