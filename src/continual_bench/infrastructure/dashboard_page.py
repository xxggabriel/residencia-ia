"""Self-contained interactive HTML dashboard written at the root of each benchmark artifact."""
# ruff: noqa: E501

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .storage import atomic_write


def _finite(value: Any) -> float | int | None:
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


def _extract_windows(report: dict[str, Any]) -> list[dict[str, Any]]:
    obs_windows = (report.get("tracking_observations") or {}).get("windows") or []
    if obs_windows:
        rows = []
        for item in obs_windows:
            metrics = {
                k: v
                for k, v in (item.get("quality") or {}).items()
                if type(v) in (int, float) and math.isfinite(v)
            }
            rows.append(
                {
                    "step": item.get("step"),
                    "start_seq": item.get("start_seq"),
                    "scope": item.get("scope", "prequential"),
                    "model_version": item.get("model_version"),
                    "metrics": metrics,
                }
            )
        return rows
    rows = []
    for item in report.get("windows") or []:
        step = item.get("end_seq")
        if step is None:
            continue
        metrics = {
            k: v
            for k, v in item.items()
            if k not in ("end_seq", "start_seq", "scope") and type(v) in (int, float) and math.isfinite(v)
        }
        rows.append(
            {
                "step": step,
                "start_seq": item.get("start_seq"),
                "scope": item.get("scope", "prequential"),
                "model_version": None,
                "metrics": metrics,
            }
        )
    return rows


def _extract_all_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(group: str, name: str, value: Any, status: str = "available", reason: str | None = None) -> None:
        key = (group, name)
        if key in seen:
            return
        seen.add(key)
        clean = _finite(value)
        if clean is None and value is not None and isinstance(value, (str, bool)):
            clean = value
        rows.append(
            {
                "group": group,
                "name": name,
                "value": clean,
                "status": status if clean is not None else "unavailable",
                "reason": None if clean is not None else (reason or "unavailable"),
            }
        )

    tracking = report.get("tracking") or {}
    t_avail = tracking.get("availability") or {}
    for k, v in (tracking.get("metrics") or {}).items():
        info = t_avail.get(k) or {}
        add("tracking", k, v, info.get("status", "available"), info.get("reason"))

    for k, v in (report.get("metrics") or {}).items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                add(f"summary/{k}", sub_k, sub_v)
        else:
            add("summary", k, v)

    obs = report.get("tracking_observations") or {}
    for scope_name, scope_vals in (obs.get("scopes") or {}).items():
        avail = (scope_vals or {}).get("availability") or {}
        for k, v in (scope_vals or {}).items():
            if k in (
                "availability",
                "confusion_matrix",
                "classes",
                "class_support",
                "class_recall",
                "pr_curve",
                "roc_curve",
                "residuals",
            ):
                continue
            info = avail.get(k) or {}
            add(f"scope/{scope_name}", k, v, info.get("status", "available"), info.get("reason"))

    for section in ("effort", "resources", "retention", "drift"):
        sec = obs.get(section) or {}
        avail = sec.get("availability") or {}
        for k, v in sec.items():
            if k in (
                "availability",
                "matrix",
                "checkpoints",
                "periods",
                "series",
                "alarms",
                "changes",
                "delays",
                "release_delays",
                "matching",
            ):
                continue
            info = avail.get(k) or {}
            add(section, k, v, info.get("status", "available"), info.get("reason"))
    return rows


def build_dashboard_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, JSON-safe payload for the interactive HTML dashboard."""
    obs = report.get("tracking_observations") or {}
    lineage = obs.get("lineage") or {}
    config = report.get("config") or {}
    tracking = report.get("tracking") or {}

    def _name(obj: Any) -> Any:
        return obj.get("name") if isinstance(obj, dict) else obj

    meta = {
        "run_key": report.get("run_key") or report.get("identity") or "",
        "dataset": _name(lineage.get("dataset") or report.get("dataset") or config.get("dataset")),
        "model": _name(lineage.get("model") or config.get("model")),
        "policy": _name(lineage.get("policy") or config.get("policy")),
        "detector": _name(config.get("detector")),
        "mode": config.get("mode") or report.get("protocol"),
        "protocol": report.get("protocol") or lineage.get("protocol"),
        "seed": lineage.get("seed") or (config.get("dataset") or {}).get("seed"),
        "complete": bool(report.get("complete")),
    }

    windows = _extract_windows(report)
    fits = [
        {
            "fit_id": f.get("fit_id"),
            "action": f.get("action"),
            "promotion_status": f.get("promotion_status"),
            "start_seq": f.get("start_seq"),
            "end_seq": f.get("end_seq"),
            "train_samples": f.get("train_samples"),
            "train_seconds": _finite(f.get("train_seconds")),
            "deployed_version": f.get("deployed_version"),
            "training_presentations": f.get("training_presentations"),
        }
        for f in (report.get("training_history") or [])
    ]
    updates = [
        {
            "seq": u.get("seq"),
            "action": u.get("action"),
            "reason": u.get("reason"),
            "train_samples": u.get("train_samples"),
            "seconds": _finite(u.get("seconds")),
            "promoted": u.get("promoted", True),
        }
        for u in (report.get("updates") or [])
    ]
    alarms = [{"seq": a.get("seq") if isinstance(a, dict) else a} for a in (report.get("alarms") or [])]

    retention = obs.get("retention") or {}
    t_series = {
        k: v for k, v in (tracking.get("series") or {}).items() if k.startswith(("cl_", "retention_"))
    }

    confusion = {}
    for scope_name, scope_vals in (obs.get("scopes") or {}).items():
        if isinstance(scope_vals, dict) and scope_vals.get("confusion_matrix") is not None:
            confusion[scope_name] = {
                "classes": scope_vals.get("classes") or [],
                "matrix": scope_vals.get("confusion_matrix") or [],
            }

    retraining_raw = report.get("retraining") or {}
    retraining = None
    if retraining_raw and retraining_raw.get("rules"):
        retraining = {
            "rules": {
                name: {
                    "auc": _finite(info.get("auc")),
                    "c_hat": _finite(info.get("c_hat")),
                    "retrains": info.get("retrains"),
                    "alphas": info.get("alphas") or [],
                    "costs": info.get("costs") or [],
                }
                for name, info in (retraining_raw.get("rules") or {}).items()
                if isinstance(info, dict)
            }
        }

    return {
        "meta": meta,
        "all_metrics": _extract_all_metrics(report),
        "windows": windows,
        "fits": fits,
        "updates": updates,
        "alarms": alarms,
        "retention": {
            "matrix": retention.get("matrix") or [],
            "checkpoints": retention.get("checkpoints") or [],
            "periods": retention.get("periods") or [],
            "metric": retention.get("metric") or "accuracy",
            "absolute_retention": _finite(retention.get("absolute_retention")),
            "gem_bwt": _finite(retention.get("gem_bwt")),
            "post_learning_fm": _finite(retention.get("post_learning_fm")),
            "series": retention.get("series") or [],
            "tracking_series": t_series,
        },
        "confusion": confusion,
        "retraining": retraining,
    }


def render_dashboard_html(report: dict[str, Any]) -> str:
    """Render a self-contained, interactive HTML5 benchmark dashboard."""
    payload = build_dashboard_payload(report)
    raw_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", r"\u003c")
    title = (
        f"Benchmark Dashboard · {payload['meta'].get('dataset') or payload['meta'].get('run_key') or 'run'}"
    )
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA_JSON__", raw_json)


def write_dashboard_html(directory: str | Path, report: dict[str, Any]) -> Path:
    """Write ``dashboard.html`` at the root of the benchmark artifact directory."""
    target = Path(directory) / "dashboard.html"
    atomic_write(target, render_dashboard_html(report).encode("utf-8"))
    return target


_HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f6f8fb;
    --surface: #ffffff;
    --surface-2: #f0f4f9;
    --ink: #132238;
    --muted: #52637a;
    --border: #d7e0ec;
    --accent: #0f5cc0;
    --accent-soft: #e6effc;
    --good: #0a7a5c;
    --warn: #b45309;
    --bad: #b42318;
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--ink);
    line-height: 1.45;
  }
  header {
    background: var(--surface);
    border-block-end: 1px solid var(--border);
    padding: 1rem 1.5rem;
  }
  .header-row {
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1.5rem;
    align-items: center;
    justify-content: space-between;
  }
  h1 { font-size: 1.25rem; margin: 0; }
  .meta-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .pill {
    font-size: 0.78rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .pill strong { color: var(--ink); }
  main { max-width: 1400px; margin: 0 auto; padding: 1.25rem 1.5rem 3rem; }

  /* Tabs (WAI-ARIA + modern-web-guidance) */
  .tablist {
    display: flex;
    overflow-x: auto;
    gap: 0.25rem;
    border-block-end: 1px solid var(--border);
    margin-block-end: 1.25rem;
    scroll-snap-type: inline proximity;
  }
  .tablist [role="tab"] {
    position: relative;
    flex: none;
    background: transparent;
    border: 0;
    padding: 0.65rem 1rem;
    font: inherit;
    font-size: 0.92rem;
    color: var(--muted);
    cursor: pointer;
    scroll-snap-align: start;
  }
  .tablist [role="tab"][aria-selected="true"] {
    color: var(--accent);
  }
  .tablist [role="tab"][aria-selected="true"]::after {
    content: "";
    position: absolute;
    inset: auto 0 0;
    border-block-end: 3px solid var(--accent);
  }
  .tablist [role="tab"]:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-block-end: 1rem;
  }
  .card h2 {
    margin: 0 0 0.75rem;
    font-size: 1.02rem;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
    gap: 0.75rem;
    margin-block-end: 1rem;
  }
  .kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 0.9rem;
  }
  .kpi .label { font-size: 0.76rem; color: var(--muted); }
  .kpi .val { font-size: 1.22rem; font-weight: 700; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }
  .kpi .sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.1rem; }

  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 0.9rem 1.4rem;
    align-items: center;
    padding: 0.75rem 0.9rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin-block-end: 0.9rem;
  }
  .control-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
  }
  .control-group select, .control-group input[type="search"], .control-group button {
    font: inherit;
    font-size: 0.84rem;
    padding: 0.3rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--ink);
  }
  .control-group button { cursor: pointer; }
  .control-group button.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }

  /* Native range slider (modern-web-guidance) */
  input[type="range"] {
    --slider-track-size: 0.45rem;
    --slider-thumb-size: 1.1rem;
    --slider-track-radius: 999px;
    --slider-thumb-radius: 50%;
    --slider-track-color: #cbd5e1;
    --slider-fill-color: var(--accent);
    --slider-thumb-color: var(--accent);
    --attr-min: attr(min type(<number>), 0);
    --attr-max: attr(max type(<number>), 100);
    --control-value: attr(value type(<number>), 50);
    --slider-progress: calc(progress(var(--control-value), var(--attr-min), var(--attr-max)) * 100%);
    appearance: none;
    inline-size: 100%;
    min-inline-size: 160px;
    block-size: 2rem;
    margin: 0;
    padding: 0;
    border: 0;
    background: transparent;
    cursor: pointer;
  }
  @supports not (opacity: progress(1, 0, 100)) {
    input[type="range"] {
      --slider-progress: calc((var(--control-value) - var(--attr-min)) / max(1, (var(--attr-max) - var(--attr-min))) * 100%);
    }
  }
  input[type="range"]::-webkit-slider-runnable-track {
    block-size: var(--slider-track-size);
    border-radius: var(--slider-track-radius);
    background: linear-gradient(to right, var(--slider-fill-color) var(--slider-progress), var(--slider-track-color) var(--slider-progress));
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    inline-size: var(--slider-thumb-size);
    block-size: var(--slider-thumb-size);
    margin-block-start: calc((var(--slider-track-size) - var(--slider-thumb-size)) / 2);
    border: 0;
    border-radius: var(--slider-thumb-radius);
    background: var(--slider-thumb-color);
  }
  input[type="range"]::-moz-range-track {
    block-size: var(--slider-track-size);
    border-radius: var(--slider-track-radius);
    background: var(--slider-track-color);
  }
  input[type="range"]::-moz-range-progress {
    block-size: var(--slider-track-size);
    border-radius: var(--slider-track-radius);
    background: var(--slider-fill-color);
  }
  input[type="range"]::-moz-range-thumb {
    inline-size: var(--slider-thumb-size);
    block-size: var(--slider-thumb-size);
    border: 0;
    border-radius: var(--slider-thumb-radius);
    background: var(--slider-thumb-color);
  }
  input[type="range"]:focus-visible {
    outline: 2px solid var(--slider-fill-color);
    outline-offset: 4px;
  }

  .series-toggles { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-block-end: 0.75rem; }
  .series-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface);
    cursor: pointer;
  }
  .swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

  .chart-wrap { position: relative; width: 100%; overflow: hidden; }
  .chart-svg { width: 100%; height: 320px; display: block; }
  .tooltip {
    position: absolute;
    pointer-events: none;
    background: rgba(19, 34, 56, 0.94);
    color: #fff;
    padding: 0.45rem 0.65rem;
    border-radius: 6px;
    font-size: 0.78rem;
    white-space: nowrap;
    z-index: 10;
    display: none;
  }

  /* Responsive table (modern-web-guidance) */
  .table-wrapper {
    overflow: auto;
    max-inline-size: 100%;
    max-block-size: min(480px, 75vh);
    container-type: inline-size;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.84rem;
    font-variant-numeric: tabular-nums;
  }
  caption {
    text-align: left;
    font-weight: 600;
    padding: 0.55rem 0.75rem;
    background: var(--surface-2);
    border-block-end: 1px solid var(--border);
  }
  thead {
    position: sticky;
    inset-block-start: 0;
    z-index: 1;
    background: var(--surface-2);
  }
  th, td {
    padding: 0.48rem 0.7rem;
    border-block-end: 1px solid var(--border);
    text-align: left;
  }
  tbody th {
    position: sticky;
    inset-inline-start: 0;
    background: var(--surface);
    font-weight: 600;
  }
  tr.active-row td, tr.active-row th { background: var(--accent-soft); }
  .badge-ok { color: var(--good); font-weight: 600; }
  .badge-na { color: var(--warn); font-weight: 600; }

  @container (width < 560px) {
    table.responsive-stack, .responsive-stack thead, .responsive-stack tbody,
    .responsive-stack tr, .responsive-stack th, .responsive-stack td { display: block; }
    .responsive-stack thead { display: none; }
    .responsive-stack tr { margin-block-end: 0.75rem; border: 1px solid var(--border); border-radius: 6px; }
    .responsive-stack tbody th { position: sticky; inset-block-start: 0; z-index: 2; background: var(--surface-2); }
    .responsive-stack td::before { content: attr(data-label) ": "; font-weight: 600; color: var(--muted); }
  }

  .matrix-cell { text-align: center; font-weight: 600; border-radius: 4px; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 1rem; }
  .muted { color: var(--muted); font-size: 0.84rem; }
</style>
</head>
<body>
<header>
  <div class="header-row">
    <div>
      <h1>Painel Interativo de Benchmark</h1>
      <div id="meta-pills" class="meta-pills" style="margin-top:0.35rem;"></div>
    </div>
    <div id="run-status" class="pill"></div>
  </div>
</header>

<main>
  <div class="tablist" role="tablist" aria-label="Seções do painel">
    <button type="button" role="tab" id="tab-overview" aria-selected="true">Visão Geral &amp; Timeline</button>
    <button type="button" role="tab" id="tab-retention" aria-selected="false">Retenção &amp; Esquecimento</button>
    <button type="button" role="tab" id="tab-retraining" aria-selected="false">Retreinamento &amp; Políticas</button>
    <button type="button" role="tab" id="tab-metrics" aria-selected="false">Todas as Métricas</button>
  </div>

  <!-- TAB 1: OVERVIEW & LIVE TIMELINE -->
  <section role="tabpanel" id="panel-overview" aria-labelledby="tab-overview" tabindex="0">
    <div id="kpi-grid" class="kpi-grid"></div>

    <div class="card">
      <h2>Timeline Interativa &amp; Retreinamento ao Vivo</h2>
      <div class="controls">
        <div class="control-group">
          <span>Modo:</span>
          <button type="button" id="mode-global" class="active">Resultado Global</button>
          <button type="button" id="mode-live">Ao Vivo por Retreinamento</button>
        </div>
        <div class="control-group" style="flex:1; min-width:240px;">
          <label for="timeline-slider">Passo / Retreinamento:</label>
          <input type="range" id="timeline-slider" min="0" max="1" step="1" value="1">
          <span id="slider-readout" class="pill"></span>
          <button type="button" id="play-btn" title="Reproduzir evolução">▶ Play</button>
        </div>
      </div>

      <div class="controls">
        <div class="control-group">
          <label for="smooth-strategy">Suavização:</label>
          <select id="smooth-strategy">
            <option value="raw">Bruto (sem suavização)</option>
            <option value="sma" selected>Média Móvel Simples (SMA)</option>
            <option value="ema">Média Móvel Exponencial (EMA)</option>
            <option value="median">Mediana Móvel</option>
          </select>
        </div>
        <div class="control-group">
          <label for="smooth-window">Janela / Fator:</label>
          <input type="range" id="smooth-window" min="1" max="25" step="1" value="3" style="max-width:160px;">
          <span id="smooth-readout" class="pill">W = 3</span>
        </div>
      </div>

      <div id="timeline-toggles" class="series-toggles"></div>
      <div class="chart-wrap">
        <svg id="timeline-chart" class="chart-svg" role="img" aria-label="Gráfico de timeline das métricas"></svg>
        <div id="timeline-tooltip" class="tooltip"></div>
      </div>
      <p class="muted" id="timeline-legend-note"></p>
    </div>

    <div class="grid-2">
      <div class="card" id="confusion-card">
        <h2>Matriz de Confusão</h2>
        <div id="confusion-container"></div>
      </div>
      <div class="card">
        <h2>Resumo do Passo Selecionado</h2>
        <div id="live-summary" class="muted"></div>
      </div>
    </div>
  </section>

  <!-- TAB 2: RETENTION & FORGETTING -->
  <section role="tabpanel" id="panel-retention" aria-labelledby="tab-retention" hidden>
    <div class="card">
      <h2>Evolução de Retenção Histórica &amp; Esquecimento</h2>
      <div id="retention-toggles" class="series-toggles"></div>
      <div class="chart-wrap">
        <svg id="retention-chart" class="chart-svg" role="img" aria-label="Gráfico de retenção e esquecimento"></svg>
        <div id="retention-tooltip" class="tooltip"></div>
      </div>
    </div>
    <div class="card">
      <h2>Matriz de Retenção Histórica (Checkpoints × Períodos)</h2>
      <div id="retention-matrix-container"></div>
    </div>
  </section>

  <!-- TAB 3: RETRAINING & POLICIES -->
  <section role="tabpanel" id="panel-retraining" aria-labelledby="tab-retraining" hidden>
    <div class="card" id="policy-compare-card">
      <h2>Comparação de Políticas de Retreinamento (CARA / UPF / Oracle)</h2>
      <div class="chart-wrap">
        <svg id="policy-chart" class="chart-svg" role="img" aria-label="Comparação de custo por alfa entre políticas"></svg>
        <div id="policy-tooltip" class="tooltip"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h2>Histórico de Ajustes / Fits</h2>
        <div id="fits-table-container"></div>
      </div>
      <div class="card">
        <h2>Eventos de Retreinamento &amp; Alarmes</h2>
        <div id="updates-table-container"></div>
      </div>
    </div>
  </section>

  <!-- TAB 4: ALL METRICS -->
  <section role="tabpanel" id="panel-metrics" aria-labelledby="tab-metrics" hidden>
    <div class="card">
      <h2>Todas as Métricas Registradas</h2>
      <div class="controls">
        <div class="control-group">
          <label for="metric-search">Filtrar:</label>
          <input type="search" id="metric-search" placeholder="Nome, grupo ou status...">
        </div>
        <div class="control-group">
          <label for="metric-status">Status:</label>
          <select id="metric-status">
            <option value="all">Todos</option>
            <option value="available">Disponíveis</option>
            <option value="unavailable">Indisponíveis</option>
          </select>
        </div>
      </div>
      <div id="all-metrics-container"></div>
    </div>
  </section>
</main>

<script type="application/json" id="benchmark-data">__DATA_JSON__</script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById('benchmark-data').textContent);
  const PALETTE = ['#0f5cc0', '#0a7a5c', '#b42318', '#b45309', '#6d28d9', '#0e7490', '#be185d'];

  const fmt = (v, digits = 4) => {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'boolean') return v ? 'Sim' : 'Não';
    if (typeof v === 'string') return v;
    if (Math.abs(v) >= 1000) return v.toLocaleString('pt-BR', { maximumFractionDigits: 2 });
    return Number(v).toFixed(digits);
  };

  function syncSliderCSS(el) {
    el.style.setProperty('--attr-min', el.min || 0);
    el.style.setProperty('--attr-max', el.max || 100);
    el.style.setProperty('--control-value', el.value);
  }

  // Meta pills
  const metaPills = document.getElementById('meta-pills');
  [['Dataset', DATA.meta.dataset], ['Modelo', DATA.meta.model], ['Política', DATA.meta.policy],
   ['Detector', DATA.meta.detector], ['Modo', DATA.meta.mode], ['Seed', DATA.meta.seed]].forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== '') {
      const span = document.createElement('span');
      span.className = 'pill';
      span.innerHTML = `<strong>${k}:</strong> ${String(v)}`;
      metaPills.appendChild(span);
    }
  });
  document.getElementById('run-status').textContent = DATA.meta.complete ? 'Execução Completa' : 'Em Progresso';

  // Tabs (accessible keyboard + panel switching)
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const panels = [...document.querySelectorAll('[role="tabpanel"]')];
  function selectTab(target) {
    tabs.forEach((t, i) => {
      const active = t === target;
      t.setAttribute('aria-selected', String(active));
      panels[i].hidden = !active;
      if (active) panels[i].setAttribute('tabindex', '0');
      else panels[i].removeAttribute('tabindex');
    });
    renderAll();
  }
  tabs.forEach((tab, idx) => {
    tab.addEventListener('click', () => selectTab(tab));
    tab.addEventListener('keydown', (e) => {
      let next = null;
      if (e.key === 'ArrowRight') next = tabs[(idx + 1) % tabs.length];
      if (e.key === 'ArrowLeft') next = tabs[(idx - 1 + tabs.length) % tabs.length];
      if (e.key === 'Home') next = tabs[0];
      if (e.key === 'End') next = tabs[tabs.length - 1];
      if (next) { e.preventDefault(); next.focus(); selectTab(next); }
    });
  });

  // Smoothing algorithms
  function smoothValues(points, strategy, windowSize) {
    if (strategy === 'raw' || windowSize <= 1 || points.length <= 1) return points.map(p => ({ ...p }));
    if (strategy === 'sma') {
      return points.map((p, idx) => {
        const start = Math.max(0, idx - windowSize + 1);
        const slice = points.slice(start, idx + 1).filter(x => x.y !== null);
        const avg = slice.length ? slice.reduce((a, b) => a + b.y, 0) / slice.length : null;
        return { ...p, y: avg };
      });
    }
    if (strategy === 'ema') {
      const alpha = 2 / (windowSize + 1);
      let prev = null;
      return points.map(p => {
        if (p.y === null) return { ...p };
        prev = prev === null ? p.y : alpha * p.y + (1 - alpha) * prev;
        return { ...p, y: prev };
      });
    }
    if (strategy === 'median') {
      return points.map((p, idx) => {
        const start = Math.max(0, idx - windowSize + 1);
        const vals = points.slice(start, idx + 1).map(x => x.y).filter(v => v !== null).sort((a, b) => a - b);
        if (!vals.length) return { ...p, y: null };
        const mid = Math.floor(vals.length / 2);
        return { ...p, y: vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2 };
      });
    }
    return points;
  }

  // Generic SVG Line Chart with crosshair tooltip & vertical event markers
  function drawLineChart(svgEl, tooltipEl, seriesList, markers = [], cutoffX = null) {
    const W = 960, H = 310, PAD = { l: 56, r: 20, t: 18, b: 36 };
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svgEl.innerHTML = '';

    const active = seriesList.filter(s => s.visible && s.points.some(p => p.y !== null));
    if (!active.length) {
      svgEl.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="#52637a" font-size="14">Sem pontos disponíveis para os filtros selecionados</text>`;
      return;
    }
    const allX = active.flatMap(s => s.points.map(p => p.x));
    const allY = active.flatMap(s => s.points.map(p => p.y).filter(y => y !== null));
    let minX = Math.min(...allX), maxX = Math.max(...allX);
    if (minX === maxX) { minX -= 1; maxX += 1; }
    let minY = Math.min(...allY), maxY = Math.max(...allY);
    if (minY === maxY) { minY -= 0.05; maxY += 0.05; }
    const padY = (maxY - minY) * 0.08;
    minY -= padY; maxY += padY;

    const sx = x => PAD.l + ((x - minX) / (maxX - minX)) * (W - PAD.l - PAD.r);
    const sy = y => H - PAD.b - ((y - minY) / (maxY - minY)) * (H - PAD.t - PAD.b);

    const ns = 'http://www.w3.org/2000/svg';
    const el = (tag, attrs) => {
      const node = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      return node;
    };

    // Gridlines
    for (let i = 0; i <= 4; i++) {
      const gy = PAD.t + (i / 4) * (H - PAD.t - PAD.b);
      const val = maxY - (i / 4) * (maxY - minY);
      svgEl.appendChild(el('line', { x1: PAD.l, y1: gy, x2: W - PAD.r, y2: gy, stroke: '#e6edf5' }));
      const lbl = el('text', { x: PAD.l - 8, y: gy + 4, 'text-anchor': 'end', fill: '#52637a', 'font-size': '11' });
      lbl.textContent = fmt(val, 3);
      svgEl.appendChild(lbl);
    }
    for (let i = 0; i <= 5; i++) {
      const gx = PAD.l + (i / 5) * (W - PAD.l - PAD.r);
      const val = Math.round(minX + (i / 5) * (maxX - minX));
      const lbl = el('text', { x: gx, y: H - 10, 'text-anchor': 'middle', fill: '#52637a', 'font-size': '11' });
      lbl.textContent = val;
      svgEl.appendChild(lbl);
    }

    // Event markers (updates / alarms)
    markers.forEach(m => {
      if (m.x < minX || m.x > maxX) return;
      const mx = sx(m.x);
      svgEl.appendChild(el('line', {
        x1: mx, y1: PAD.t, x2: mx, y2: H - PAD.b,
        stroke: m.color, 'stroke-dasharray': m.dash || '4,3', 'stroke-width': 1.3, opacity: 0.75
      }));
    });

    if (cutoffX !== null && cutoffX >= minX && cutoffX <= maxX) {
      const cx = sx(cutoffX);
      svgEl.appendChild(el('line', {
        x1: cx, y1: PAD.t, x2: cx, y2: H - PAD.b,
        stroke: '#132238', 'stroke-width': 2
      }));
    }

    // Series paths
    active.forEach(s => {
      let d = '';
      s.points.forEach(p => {
        if (p.y === null) return;
        d += (d ? ' L ' : 'M ') + sx(p.x).toFixed(2) + ' ' + sy(p.y).toFixed(2);
      });
      if (d) {
        svgEl.appendChild(el('path', {
          d, fill: 'none', stroke: s.color, 'stroke-width': 2.2, 'stroke-linejoin': 'round'
        }));
      }
    });

    // Crosshair hover overlay
    const cross = el('line', { x1: 0, y1: PAD.t, x2: 0, y2: H - PAD.b, stroke: '#52637a', 'stroke-dasharray': '2,2', visibility: 'hidden' });
    svgEl.appendChild(cross);
    const overlay = el('rect', { x: PAD.l, y: PAD.t, width: W - PAD.l - PAD.r, height: H - PAD.t - PAD.b, fill: 'transparent' });
    svgEl.appendChild(overlay);

    overlay.addEventListener('mousemove', (ev) => {
      const rect = svgEl.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * W;
      const targetX = minX + ((relX - PAD.l) / (W - PAD.l - PAD.r)) * (maxX - minX);
      let nearestX = null, bestDist = Infinity;
      active[0].points.forEach(p => {
        const dist = Math.abs(p.x - targetX);
        if (dist < bestDist) { bestDist = dist; nearestX = p.x; }
      });
      if (nearestX === null) return;
      const cx = sx(nearestX);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
      const lines = [`<strong>Passo ${nearestX}</strong>`];
      active.forEach(s => {
        const pt = s.points.find(p => p.x === nearestX);
        if (pt && pt.y !== null) {
          lines.push(`<span style="color:${s.color}">●</span> ${s.label}: ${fmt(pt.y)}`);
        }
      });
      tooltipEl.innerHTML = lines.join('<br>');
      tooltipEl.style.display = 'block';
      const tipX = Math.min(rect.width - 170, Math.max(8, ev.clientX - rect.left + 12));
      tooltipEl.style.left = `${tipX}px`;
      tooltipEl.style.top = '12px';
    });
    overlay.addEventListener('mouseleave', () => {
      cross.setAttribute('visibility', 'hidden');
      tooltipEl.style.display = 'none';
    });
  }

  // Timeline state & controls
  const windows = DATA.windows || [];
  const windowMetricKeys = [...new Set(windows.flatMap(w => Object.keys(w.metrics || {})))].filter(k =>
    ['accuracy', 'macro_f1', 'positive_precision', 'positive_recall', 'average_precision', 'brier', 'mae', 'rmse'].includes(k)
  );
  const timelineState = {
    mode: 'global',
    sliderIdx: Math.max(0, windows.length - 1),
    strategy: 'sma',
    smoothWindow: 3,
    visibleMetrics: new Set(windowMetricKeys.slice(0, 3)),
    playing: null,
  };

  const slider = document.getElementById('timeline-slider');
  const sliderReadout = document.getElementById('slider-readout');
  const smoothStrategy = document.getElementById('smooth-strategy');
  const smoothWindow = document.getElementById('smooth-window');
  const smoothReadout = document.getElementById('smooth-readout');
  const modeGlobal = document.getElementById('mode-global');
  const modeLive = document.getElementById('mode-live');
  const playBtn = document.getElementById('play-btn');

  if (windows.length > 0) {
    slider.min = 0;
    slider.max = windows.length - 1;
    slider.value = windows.length - 1;
  } else {
    slider.disabled = true;
  }
  syncSliderCSS(slider);
  syncSliderCSS(smoothWindow);

  modeGlobal.addEventListener('click', () => {
    timelineState.mode = 'global';
    modeGlobal.classList.add('active'); modeLive.classList.remove('active');
    renderOverview();
  });
  modeLive.addEventListener('click', () => {
    timelineState.mode = 'live';
    modeLive.classList.add('active'); modeGlobal.classList.remove('active');
    renderOverview();
  });
  slider.addEventListener('input', () => {
    timelineState.sliderIdx = Number(slider.value);
    syncSliderCSS(slider);
    if (timelineState.mode === 'global') {
      timelineState.mode = 'live';
      modeLive.classList.add('active'); modeGlobal.classList.remove('active');
    }
    renderOverview();
  });
  smoothStrategy.addEventListener('change', () => {
    timelineState.strategy = smoothStrategy.value;
    renderAll();
  });
  smoothWindow.addEventListener('input', () => {
    timelineState.smoothWindow = Number(smoothWindow.value);
    syncSliderCSS(smoothWindow);
    smoothReadout.textContent = timelineState.strategy === 'ema'
      ? `α = ${(2 / (timelineState.smoothWindow + 1)).toFixed(2)}`
      : `W = ${timelineState.smoothWindow}`;
    renderAll();
  });
  playBtn.addEventListener('click', () => {
    if (timelineState.playing) {
      clearInterval(timelineState.playing);
      timelineState.playing = null;
      playBtn.textContent = '▶ Play';
      return;
    }
    if (!windows.length) return;
    timelineState.mode = 'live';
    modeLive.classList.add('active'); modeGlobal.classList.remove('active');
    if (timelineState.sliderIdx >= windows.length - 1) timelineState.sliderIdx = 0;
    playBtn.textContent = '⏸ Pausar';
    timelineState.playing = setInterval(() => {
      if (timelineState.sliderIdx >= windows.length - 1) {
        clearInterval(timelineState.playing);
        timelineState.playing = null;
        playBtn.textContent = '▶ Play';
        return;
      }
      timelineState.sliderIdx += 1;
      slider.value = timelineState.sliderIdx;
      syncSliderCSS(slider);
      renderOverview();
    }, 350);
  });

  // Toggles for timeline
  const togglesEl = document.getElementById('timeline-toggles');
  windowMetricKeys.forEach((key, idx) => {
    const label = document.createElement('label');
    label.className = 'series-pill';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = timelineState.visibleMetrics.has(key);
    cb.addEventListener('change', () => {
      if (cb.checked) timelineState.visibleMetrics.add(key);
      else timelineState.visibleMetrics.delete(key);
      renderOverview();
    });
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = PALETTE[idx % PALETTE.length];
    label.append(cb, sw, document.createTextNode(key));
    togglesEl.appendChild(label);
  });

  function renderKPIs(cutoffStep) {
    const findMetric = (name) => DATA.all_metrics.find(m => m.name === name)?.value;
    const slicedWins = timelineState.mode === 'live' && windows.length
      ? windows.slice(0, timelineState.sliderIdx + 1)
      : windows;
    const meanWin = (k) => {
      const vals = slicedWins.map(w => w.metrics[k]).filter(v => v !== undefined && v !== null);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    const updatesCount = timelineState.mode === 'live'
      ? (DATA.updates || []).filter(u => u.seq <= cutoffStep).length
      : (DATA.updates || []).length;

    const kpis = [
      ['Acurácia', timelineState.mode === 'live' ? meanWin('accuracy') : (findMetric('accuracy') ?? meanWin('accuracy')), timelineState.mode === 'live' ? 'média até o passo' : 'global'],
      ['Macro F1', timelineState.mode === 'live' ? meanWin('macro_f1') : (findMetric('macro_f1') ?? meanWin('macro_f1')), 'classificação'],
      ['Retenção Absoluta', findMetric('absolute_retention'), 'períodos históricos'],
      ['GEM BWT', findMetric('gem_bwt'), 'backward transfer'],
      ['Post-Learning FM', findMetric('post_learning_fm') ?? findMetric('cl_forgetting'), 'esquecimento'],
      ['CL Forgetting', findMetric('cl_forgetting'), 'matriz CL'],
      ['Retreinamentos', updatesCount, timelineState.mode === 'live' ? `até seq ${cutoffStep}` : 'total'],
      ['Duração (s)', findMetric('duration_seconds'), 'execução'],
    ];
    const grid = document.getElementById('kpi-grid');
    grid.innerHTML = '';
    kpis.forEach(([label, val, sub]) => {
      const card = document.createElement('div');
      card.className = 'kpi';
      card.innerHTML = `<div class="label">${label}</div><div class="val">${fmt(val)}</div><div class="sub">${sub}</div>`;
      grid.appendChild(card);
    });
  }

  function renderOverview() {
    const activeWin = windows[timelineState.sliderIdx];
    const cutoffStep = activeWin ? activeWin.step : Infinity;
    sliderReadout.textContent = activeWin
      ? `seq ${activeWin.step} (${timelineState.sliderIdx + 1}/${windows.length})`
      : 'sem janelas';
    slider.setAttribute('aria-valuetext', sliderReadout.textContent);

    renderKPIs(cutoffStep);

    const shown = timelineState.mode === 'live' && windows.length
      ? windows.slice(0, timelineState.sliderIdx + 1)
      : windows;

    const seriesList = windowMetricKeys.map((key, idx) => {
      const rawPts = shown.map(w => ({ x: w.step, y: w.metrics[key] ?? null }));
      return {
        label: key,
        color: PALETTE[idx % PALETTE.length],
        visible: timelineState.visibleMetrics.has(key),
        points: smoothValues(rawPts, timelineState.strategy, timelineState.smoothWindow),
      };
    });

    const markers = [
      ...(DATA.updates || []).map(u => ({ x: u.seq, color: '#b45309', dash: '4,3' })),
      ...(DATA.alarms || []).map(a => ({ x: a.seq, color: '#b42318', dash: '2,2' })),
    ];

    drawLineChart(
      document.getElementById('timeline-chart'),
      document.getElementById('timeline-tooltip'),
      seriesList,
      markers,
      timelineState.mode === 'live' && activeWin ? activeWin.step : null
    );
    document.getElementById('timeline-legend-note').textContent =
      `Linhas tracejadas: laranja = retreinamento (${(DATA.updates || []).length}), vermelho = alarme de drift (${(DATA.alarms || []).length}).`;

    // Live summary
    const liveBox = document.getElementById('live-summary');
    if (activeWin) {
      const activeUpdates = (DATA.updates || []).filter(u => u.seq <= activeWin.step);
      const lastUpdate = activeUpdates[activeUpdates.length - 1];
      liveBox.innerHTML = `
        <p><strong>Passo atual:</strong> seq ${activeWin.step} (escopo: ${activeWin.scope})</p>
        <p><strong>Versão do modelo:</strong> ${activeWin.model_version || 'inicial'}</p>
        <p><strong>Retreinamentos acumulados:</strong> ${activeUpdates.length}${lastUpdate ? ` (último em seq ${lastUpdate.seq}, ${lastUpdate.action})` : ''}</p>
        <p><strong>Métricas da janela:</strong> ${Object.entries(activeWin.metrics).slice(0, 6).map(([k, v]) => `${k}=${fmt(v, 3)}`).join(' · ')}</p>
      `;
    } else {
      liveBox.textContent = 'Sem janelas registradas para este benchmark.';
    }

    // Confusion matrix
    const confEl = document.getElementById('confusion-container');
    const confEntries = Object.entries(DATA.confusion || {});
    if (!confEntries.length) {
      confEl.innerHTML = '<p class="muted">Matriz de confusão não aplicável ou indisponível.</p>';
    } else {
      const [scopeName, conf] = confEntries[0];
      const classes = conf.classes || [];
      const mat = conf.matrix || [];
      let html = `<div class="table-wrapper" tabindex="0" role="region" aria-label="Matriz de confusão ${scopeName}"><table><caption>Escopo: ${scopeName}</caption><thead><tr><th>Real \\ Predito</th>`;
      classes.forEach(c => { html += `<th>${c}</th>`; });
      html += '</tr></thead><tbody>';
      mat.forEach((row, rIdx) => {
        html += `<tr><th>${classes[rIdx] ?? rIdx}</th>`;
        row.forEach(val => { html += `<td>${val}</td>`; });
        html += '</tr>';
      });
      html += '</tbody></table></div>';
      confEl.innerHTML = html;
    }
  }

  // Retention tab
  const retSeriesVisible = new Set(['absolute_retention', 'gem_bwt', 'post_learning_fm', 'cl_forgetting']);
  function renderRetention() {
    const ret = DATA.retention || {};
    const checkpoints = ret.checkpoints || [];
    const series = ret.series || [];
    const tSeries = ret.tracking_series || {};

    const combined = [];
    ['absolute_retention', 'gem_bwt', 'post_learning_fm'].forEach((k, idx) => {
      const pts = series.map((r, i) => ({ x: r.step ?? i, y: r[k] ?? null }));
      if (pts.some(p => p.y !== null)) {
        combined.push({ key: k, label: k, color: PALETTE[idx], points: pts });
      }
    });
    Object.entries(tSeries).forEach(([k, pts], idx) => {
      const mapped = pts.map(p => ({ x: p.step, y: p.value ?? null }));
      if (mapped.some(p => p.y !== null)) {
        combined.push({ key: k, label: k, color: PALETTE[(idx + 3) % PALETTE.length], points: mapped });
      }
    });

    const retToggles = document.getElementById('retention-toggles');
    if (!retToggles.children.length) {
      combined.forEach(s => {
        const label = document.createElement('label');
        label.className = 'series-pill';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = retSeriesVisible.has(s.key) || combined.length <= 4;
        if (cb.checked) retSeriesVisible.add(s.key);
        cb.addEventListener('change', () => {
          if (cb.checked) retSeriesVisible.add(s.key);
          else retSeriesVisible.delete(s.key);
          renderRetention();
        });
        const sw = document.createElement('span');
        sw.className = 'swatch';
        sw.style.background = s.color;
        label.append(cb, sw, document.createTextNode(s.label));
        retToggles.appendChild(label);
      });
    }

    drawLineChart(
      document.getElementById('retention-chart'),
      document.getElementById('retention-tooltip'),
      combined.map(s => ({
        ...s,
        visible: retSeriesVisible.has(s.key),
        points: smoothValues(s.points, timelineState.strategy, timelineState.smoothWindow),
      }))
    );

    const matContainer = document.getElementById('retention-matrix-container');
    const matrix = ret.matrix || [];
    const periods = ret.periods || [];
    if (!matrix.length) {
      matContainer.innerHTML = '<p class="muted">Matriz de retenção histórica indisponível para este protocolo.</p>';
      return;
    }
    let html = `<div class="table-wrapper" tabindex="0" role="region" aria-label="Matriz de retenção histórica"><table><caption>Métrica: ${ret.metric} · Retenção absoluta: ${fmt(ret.absolute_retention)} · GEM BWT: ${fmt(ret.gem_bwt)} · FM: ${fmt(ret.post_learning_fm)}</caption><thead><tr><th>Checkpoint</th>`;
    periods.forEach(p => { html += `<th>${p}</th>`; });
    html += '</tr></thead><tbody>';
    matrix.forEach((row, rIdx) => {
      html += `<tr><th>${checkpoints[rIdx] || `step-${rIdx}`}</th>`;
      row.forEach(val => {
        if (val === null || val === undefined) {
          html += '<td class="matrix-cell">—</td>';
        } else {
          const alpha = Math.max(0.08, Math.min(0.65, val * 0.65));
          html += `<td class="matrix-cell" style="background:rgba(10,122,92,${alpha.toFixed(2)})">${fmt(val, 3)}</td>`;
        }
      });
      html += '</tr>';
    });
    html += '</tbody></table></div>';
    matContainer.innerHTML = html;
  }

  // Retraining & Policies tab
  function renderRetraining() {
    const policyCard = document.getElementById('policy-compare-card');
    if (DATA.retraining && DATA.retraining.rules) {
      policyCard.hidden = false;
      const rules = Object.entries(DATA.retraining.rules);
      const series = rules.map(([name, info], idx) => ({
        label: `${name} (AUC=${fmt(info.auc, 3)})`,
        color: PALETTE[idx % PALETTE.length],
        visible: true,
        points: (info.alphas || []).map((a, i) => ({ x: a, y: info.costs[i] ?? null })),
      }));
      drawLineChart(document.getElementById('policy-chart'), document.getElementById('policy-tooltip'), series);
    } else {
      policyCard.hidden = true;
    }

    const fitsEl = document.getElementById('fits-table-container');
    if (!(DATA.fits || []).length) {
      fitsEl.innerHTML = '<p class="muted">Nenhum ajuste registrado.</p>';
    } else {
      let html = '<div class="table-wrapper" tabindex="0" role="region" aria-label="Histórico de ajustes"><table class="responsive-stack"><thead><tr><th>Fit</th><th>Ação</th><th>Status</th><th>Seq</th><th>Amostras</th><th>Tempo (s)</th></tr></thead><tbody>';
      DATA.fits.forEach(f => {
        html += `<tr><th>${f.fit_id}</th><td data-label="Ação">${f.action}</td><td data-label="Status">${f.promotion_status}</td><td data-label="Seq">${f.start_seq ?? '—'}..${f.end_seq ?? '—'}</td><td data-label="Amostras">${f.train_samples ?? '—'}</td><td data-label="Tempo (s)">${fmt(f.train_seconds, 4)}</td></tr>`;
      });
      html += '</tbody></table></div>';
      fitsEl.innerHTML = html;
    }

    const updEl = document.getElementById('updates-table-container');
    if (!(DATA.updates || []).length) {
      updEl.innerHTML = '<p class="muted">Nenhum retreinamento pós-warmup registrado.</p>';
    } else {
      let html = '<div class="table-wrapper" tabindex="0" role="region" aria-label="Eventos de retreinamento"><table class="responsive-stack"><thead><tr><th>Seq</th><th>Ação</th><th>Motivo</th><th>Amostras</th><th>Tempo (s)</th></tr></thead><tbody>';
      DATA.updates.forEach(u => {
        html += `<tr><th>${u.seq}</th><td data-label="Ação">${u.action}</td><td data-label="Motivo">${u.reason || '—'}</td><td data-label="Amostras">${u.train_samples ?? '—'}</td><td data-label="Tempo (s)">${fmt(u.seconds, 4)}</td></tr>`;
      });
      html += '</tbody></table></div>';
      updEl.innerHTML = html;
    }
  }

  // All Metrics tab
  const metricSearch = document.getElementById('metric-search');
  const metricStatus = document.getElementById('metric-status');
  function renderMetricsTable() {
    const q = (metricSearch.value || '').toLowerCase();
    const st = metricStatus.value;
    const rows = (DATA.all_metrics || []).filter(m => {
      if (st !== 'all' && m.status !== st) return false;
      if (!q) return true;
      return `${m.group} ${m.name} ${m.reason || ''}`.toLowerCase().includes(q);
    });
    const el = document.getElementById('all-metrics-container');
    let html = `<div class="table-wrapper" tabindex="0" role="region" aria-label="Tabela completa de métricas"><table class="responsive-stack"><caption>${rows.length} métrica(s)</caption><thead><tr><th>Métrica</th><th>Grupo</th><th>Valor</th><th>Status</th><th>Motivo (se indisponível)</th></tr></thead><tbody>`;
    rows.forEach(m => {
      const cls = m.status === 'available' ? 'badge-ok' : 'badge-na';
      html += `<tr><th>${m.name}</th><td data-label="Grupo">${m.group}</td><td data-label="Valor">${fmt(m.value)}</td><td data-label="Status" class="${cls}">${m.status}</td><td data-label="Motivo">${m.reason || '—'}</td></tr>`;
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
  }
  metricSearch.addEventListener('input', renderMetricsTable);
  metricStatus.addEventListener('change', renderMetricsTable);

  function renderAll() {
    renderOverview();
    renderRetention();
    renderRetraining();
    renderMetricsTable();
  }
  renderAll();
})();
</script>
</body>
</html>
"""
