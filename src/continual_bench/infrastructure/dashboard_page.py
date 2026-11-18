"""Self-contained interactive HTML dashboard written at the root of each benchmark artifact.

Implements interactive benchmark visualizations and full replicas of the figures and tables from:
1. Mahadevan & Mathioudakis (2024), "Cost-aware retraining for machine learning" (CARA):
   - Upper-triangular Staleness / Cost Matrix (Psi_bar) & Retraining Strategy Trajectory (Fig. 3a, A.8, A.9)
   - Optimal offline CARA parameters (tau*, tau_cum*, phi*) vs retraining cost kappa=alpha (Fig. 3b)
   - Synchronized multi-panel evaluation across kappa=alpha: Strategy Cost, # Retrain Decisions, Query Accuracy & SCPE (Figs. 4-7)
   - Cumulative Strategy Cost vs Online Batches t (Fig. B.10)
   - Accuracy vs Retraining Effort / # Retrains Pareto Trade-off Plot with distinct point shapes (Fig. E.14)
   - Summary comparison tables (Tables 2, 3, 4: Mean SCPE, Mean Query Accuracy, Mean # Retrains, AUC)
2. Regol et al. (2025), "When to retrain a machine learning model" (UPF):
   - Model Performance Decay (D_i evaluated on D_j) & Active Retraining Schedule Trajectory (Fig. 1)
   - Cost C_hat_alpha(theta) and # Retrains vs alpha with distinct marker shapes & line styles (Fig. 2)
   - Proposition 3.1 Theoretical Upper Bounds on r* & Lipschitz constant L (Fig. 4)
   - Robustness / Sensitivity to Wrong alpha Heatmap (provided alpha vs real alpha, % cost increase, Fig. 3)
   - Forecaster Quality Diagnostics: Predicted vs Observed Error scatter & Residual Histogram (Figs. 5-8, App. A.7)
   - AUC Bar Chart & Low (0.1 alpha_max) vs Focus vs High (0.9 alpha_max) Retrain Cost Regime Table (Tables 1 & 2)
Plus full stream/continual-learning/drift/probabilistic/ROC-PR/system-resources diagnostics and interpretability guides.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dashboard_payload import build_dashboard_payload
from .storage import atomic_write

__all__ = [
    "build_dashboard_payload",
    "render_dashboard_html",
    "write_dashboard_html",
]


def render_dashboard_html(report: dict[str, Any]) -> str:
    """Render a self-contained, interactive HTML5 benchmark dashboard."""
    payload = build_dashboard_payload(report)
    raw_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", r"\u003c")
    meta = payload.get("meta") or {}
    title = f"Benchmark Dashboard · {meta.get('dataset') or meta.get('run_key') or 'run'}"
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA_JSON__", raw_json)


def write_dashboard_html(directory: str | Path, report: dict[str, Any]) -> Path:
    """Write ``dashboard.html`` at the root of the benchmark artifact directory."""
    target = Path(directory) / "dashboard.html"
    atomic_write(target, render_dashboard_html(report).encode("utf-8"))
    return target


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f6f8fb;
  --surface: #ffffff;
  --surface-alt: #f0f4f9;
  --surface-elevated: #ffffff;
  --border: #d8e1ec;
  --border-strong: #b6c5d8;
  --text: #0f172a;
  --text-muted: #475569;
  --text-soft: #64748b;
  --primary: #0f62fe;
  --primary-soft: #e0ecff;
  --accent: #0072b2;
  --success: #0a7a5c;
  --success-soft: #dcfce7;
  --warning: #b45309;
  --warning-soft: #fef3c7;
  --danger: #be123c;
  --danger-soft: #ffe4e6;
  --purple: #6d28d9;
  --purple-soft: #ede9fe;
  --radius: 10px;
  --radius-sm: 6px;
  --shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
}
[data-theme="dark"] {
  --bg: #0b1120;
  --surface: #111827;
  --surface-alt: #1e293b;
  --surface-elevated: #1f2937;
  --border: #334155;
  --border-strong: #475569;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --text-soft: #64748b;
  --primary: #3b82f6;
  --primary-soft: rgba(59, 130, 246, 0.18);
  --accent: #38bdf8;
  --success: #10b981;
  --success-soft: rgba(16, 185, 129, 0.18);
  --warning: #f59e0b;
  --warning-soft: rgba(245, 158, 11, 0.18);
  --danger: #f43f5e;
  --danger-soft: rgba(244, 63, 94, 0.18);
  --purple: #a78bfa;
  --purple-soft: rgba(167, 139, 250, 0.18);
  --shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
}
header.app-header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  position: sticky;
  top: 0;
  z-index: 40;
  box-shadow: var(--shadow);
}
.header-top {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.brand-title {
  display: flex;
  align-items: center;
  gap: 10px;
}
.brand-title h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.meta-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  color: var(--text);
}
.pill strong { color: var(--text-muted); font-weight: 500; }
.pill.badge-ok { background: var(--success-soft); color: var(--success); border-color: transparent; }
.pill.badge-warn { background: var(--warning-soft); color: var(--warning); border-color: transparent; }

.global-toolbar {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 14px;
}
.toolbar-group {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--surface-alt);
  padding: 5px 10px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  font-size: 12px;
}
.toolbar-group label {
  font-weight: 600;
  color: var(--text-muted);
  white-space: nowrap;
}
select, input[type="text"], input[type="search"], input[type="number"] {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border-strong);
  border-radius: 5px;
  padding: 4px 8px;
  font-size: 12px;
  font-family: inherit;
}
input[type="range"] {
  accent-color: var(--primary);
  cursor: pointer;
}
button.btn {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border-strong);
  border-radius: 5px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}
button.btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}
button.btn.active, button.btn-primary {
  background: var(--primary);
  color: #ffffff;
  border-color: var(--primary);
}
.mode-switch {
  display: inline-flex;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--border-strong);
}
.mode-switch button {
  border: none;
  background: var(--surface);
  color: var(--text-muted);
  padding: 5px 11px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.mode-switch button.active {
  background: var(--primary);
  color: #fff;
}

.live-banner {
  margin-top: 8px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  background: var(--primary-soft);
  border: 1px solid var(--primary);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  font-size: 12.5px;
}

nav.tabs-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 12px 24px 0;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
.tab-btn {
  background: transparent;
  color: var(--text-muted);
  border: none;
  border-bottom: 3px solid transparent;
  padding: 9px 14px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active {
  color: var(--primary);
  border-bottom-color: var(--primary);
}

main.container {
  max-width: 1640px;
  margin: 0 auto;
  padding: 20px 24px 60px;
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }

.insights-box {
  background: linear-gradient(135deg, var(--surface) 0%, var(--surface-alt) 100%);
  border: 1px solid var(--border-strong);
  border-left: 4px solid var(--primary);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-bottom: 18px;
  box-shadow: var(--shadow);
}
.insights-box h2 {
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 8px;
}
.insights-list {
  margin: 0;
  padding-left: 18px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 6px 20px;
  font-size: 13px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  box-shadow: var(--shadow);
}
.kpi-card .kpi-label {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.kpi-card .kpi-val {
  font-size: 21px;
  font-weight: 700;
  font-family: var(--font-mono);
  margin: 4px 0 2px;
  color: var(--text);
}
.kpi-card .kpi-sub {
  font-size: 11.5px;
  color: var(--text-soft);
}

.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}
.grid-4 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 14px;
  margin-bottom: 16px;
}
@media (max-width: 768px) {
  .grid-2, .grid-4 { grid-template-columns: 1fr; }
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}
.card-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 10px;
}
.card-title-group h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 8px;
}
.paper-tag {
  display: inline-block;
  padding: 2px 7px;
  font-size: 10.5px;
  font-weight: 700;
  border-radius: 4px;
  background: var(--purple-soft);
  color: var(--purple);
  letter-spacing: 0.02em;
}
.card-subtitle {
  margin: 3px 0 0;
  font-size: 12.5px;
  color: var(--text-muted);
}

details.guide-box {
  margin: 8px 0 12px;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 8px 12px;
  font-size: 12.5px;
}
details.guide-box summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--primary);
  user-select: none;
}
.formula-block {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 10px;
  margin: 6px 0;
  overflow-x: auto;
  color: var(--text);
}

.chart-wrap {
  position: relative;
  width: 100%;
  user-select: none;
}
.chart-svg {
  width: 100%;
  height: 310px;
  display: block;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}
.chart-svg.short { height: 245px; }
.chart-svg.tall { height: 370px; }

.legend-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 8px 0 6px;
}
.legend-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 600;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  cursor: pointer;
  transition: all 0.15s ease;
  color: var(--text);
}
.legend-pill:hover {
  border-color: var(--primary);
  transform: translateY(-1px);
}
.legend-pill.muted-series {
  opacity: 0.38;
  text-decoration: line-through;
}
.legend-marker-svg {
  width: 24px;
  height: 14px;
  flex-shrink: 0;
}

.chart-tooltip {
  position: absolute;
  pointer-events: none;
  background: rgba(15, 23, 42, 0.94);
  color: #f8fafc;
  padding: 8px 11px;
  border-radius: 6px;
  font-size: 11.5px;
  line-height: 1.45;
  z-index: 30;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28);
  max-width: 340px;
  border: 1px solid rgba(255,255,255,0.15);
}
.chart-tooltip .tt-title {
  font-weight: 700;
  border-bottom: 1px solid rgba(255,255,255,0.18);
  padding-bottom: 3px;
  margin-bottom: 4px;
}
.chart-tooltip .tt-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-family: var(--font-mono);
}

.table-wrapper {
  width: 100%;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}
th, td {
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
thead th {
  background: var(--surface-alt);
  color: var(--text-muted);
  font-weight: 700;
  position: sticky;
  top: 0;
}
td.num, th.num {
  text-align: right;
  font-family: var(--font-mono);
}
tr:hover td {
  background: rgba(15, 98, 254, 0.04);
}
.best-val {
  font-weight: 800;
  color: var(--success);
}
.second-val {
  text-decoration: underline;
  font-weight: 600;
}
.matrix-cell {
  text-align: center;
  font-family: var(--font-mono);
  font-size: 11.5px;
  padding: 5px 6px;
}
.muted { color: var(--text-muted); font-style: italic; }
.badge-ok { color: var(--success); font-weight: 700; }
.badge-na { color: var(--text-soft); }
</style>
</head>
<body>
<script id="benchmark-data" type="application/json">__DATA_JSON__</script>

<header class="app-header">
  <div class="header-top">
    <div class="brand-title">
      <h1>📊 Benchmark &amp; Cost-Aware Retraining Dashboard</h1>
      <span class="paper-tag">CARA (Mahadevan &amp; Mathioudakis 2024) + UPF (Regol et al. 2025)</span>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
      <button type="button" class="btn" id="expand-all-guides" title="Expandir/Recolher fórmulas e guias de interpretação">📖 Fórmulas &amp; Guias</button>
      <button type="button" class="btn" id="theme-toggle" title="Alternar tema claro/escuro">🌗 Tema</button>
    </div>
  </div>
  <div class="meta-pills" id="meta-pills"></div>

  <div class="global-toolbar" role="toolbar" aria-label="Controles globais de simulação, suavização e ponto de operação">
    <div class="toolbar-group">
      <label>Modo de Visão:</label>
      <div class="mode-switch">
        <button type="button" id="mode-global" class="active">Resultado Global</button>
        <button type="button" id="mode-live">Simulação ao Vivo</button>
      </div>
    </div>

    <div class="toolbar-group">
      <label for="timeline-slider">Passo / Lote (<span id="timeline-step-label">100%</span>):</label>
      <button type="button" class="btn" id="timeline-prev" title="Passo anterior">⏮</button>
      <button type="button" class="btn" id="timeline-play" title="Reproduzir simulação de retreinamento passo a passo">▶ Play</button>
      <button type="button" class="btn" id="timeline-next" title="Próximo passo">⏭</button>
      <input type="range" id="timeline-slider" min="1" max="100" value="100" style="width:130px;">
    </div>

    <div class="toolbar-group">
      <label for="smooth-strategy">Suavização:</label>
      <select id="smooth-strategy">
        <option value="raw">Bruto (Sem Suavização)</option>
        <option value="sma">Média Móvel Simples (SMA)</option>
        <option value="ema">Média Móvel Exponencial (EMA)</option>
        <option value="median">Mediana Móvel (Robusta a Outliers)</option>
        <option value="savgol">Savitzky-Golay / Ponderada (WMA)</option>
      </select>
      <label for="smooth-window">Janela (<span id="smooth-window-val">5</span>):</label>
      <input type="range" id="smooth-window" min="1" max="25" value="5" style="width:85px;">
    </div>

    <div class="toolbar-group" id="alpha-toolbar-group">
      <label for="policy-selector">Política Foco:</label>
      <select id="policy-selector"></select>
      <label for="alpha-slider">Custo α=κ (<span id="alpha-value-label">—</span>):</label>
      <input type="range" id="alpha-slider" min="0" max="0" step="1" value="0" style="width:120px;">
    </div>
  </div>

  <div class="live-banner" id="live-status-banner" hidden></div>
</header>

<nav class="tabs-nav" role="tablist" aria-label="Seções do Dashboard">
  <button type="button" class="tab-btn active" data-tab="tab-overview" role="tab" aria-selected="true">1. Visão Geral &amp; Fluxo Prequencial</button>
  <button type="button" class="tab-btn" data-tab="tab-paper-retraining" role="tab" aria-selected="false">2. Artigos Base: CARA &amp; UPF (Retreinamento)</button>
  <button type="button" class="tab-btn" data-tab="tab-retention" role="tab" aria-selected="false">3. Continual Learning, Esquecimento &amp; Retenção</button>
  <button type="button" class="tab-btn" data-tab="tab-classification" role="tab" aria-selected="false">4. Diagnósticos ROC, PR &amp; Confusão</button>
  <button type="button" class="tab-btn" data-tab="tab-drift-system" role="tab" aria-selected="false">5. Drift, Esforço, Custos &amp; Sistema</button>
  <button type="button" class="tab-btn" data-tab="tab-catalog" role="tab" aria-selected="false">6. Catálogo Completo de Métricas &amp; Fórmulas</button>
</nav>

<main class="container">
  <!-- Diagnostic Insights Banner -->
  <section class="insights-box" id="insights-banner" aria-label="Interpretação Automática do Experimento">
    <h2>💡 Síntese Diagnóstica &amp; Interpretação Automática da Execução</h2>
    <ul class="insights-list" id="insights-list"></ul>
  </section>

  <!-- KPI Summary Strip -->
  <section class="kpi-grid" id="kpi-strip" aria-label="Indicadores Chave de Desempenho"></section>

  <!-- TAB 1: OVERVIEW & PREQUENTIAL STREAM -->
  <div class="tab-panel active" id="tab-overview" role="tabpanel">
    <div class="card" id="prequential-card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>Evolução Prequencial (Test-Then-Train) &amp; Eventos de Adaptação <span class="paper-tag">Stream Evaluation</span></h3>
          <p class="card-subtitle">Séries temporais avaliadas por janela com marcadores geométricos distintos por métrica, alarmes de drift (linhas tracejadas vermelhas) e promoções de modelos (linhas verdes).</p>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <label for="scope-selector" style="font-size:12px; font-weight:600;">Escopo:</label>
          <select id="scope-selector"></select>
        </div>
      </div>
      <details class="guide-box">
        <summary>Como ler este gráfico e o efeito da suavização</summary>
        <p>Cada ponto $t$ representa o desempenho prequencial na janela encerrada no passo $t$ antes do modelo ser atualizado com aqueles exemplos. Use a barra superior para alternar entre <strong>Bruto</strong>, <strong>SMA</strong>, <strong>EMA</strong>, <strong>Mediana Móvel</strong> ou <strong>Savitzky-Golay/WMA</strong>, ou ative a <strong>Simulação ao Vivo</strong> para percorrer passo a passo o histórico de retreinamentos.</p>
      </details>
      <div class="legend-bar" id="prequential-legend"></div>
      <div class="chart-wrap">
        <svg class="chart-svg tall" id="prequential-chart" role="img" aria-label="Gráfico de evolução prequencial"></svg>
        <div class="chart-tooltip" id="prequential-tooltip" hidden></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card" id="overview-upf-decay-preview-card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Degradação por Modelo $f_i$ &amp; Trajetória Ativa <span class="paper-tag">UPF Fig. 1</span></h3>
            <p class="card-subtitle">Leque de acurácia de cada modelo $f_i$ (treinado no lote $D_i$) nos lotes futuros $D_t$ ($t \ge i$) + trajetória real da política selecionada.</p>
          </div>
        </div>
        <div class="legend-bar" id="upf-fig1-overview-legend"></div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="upf-fig1-overview-chart" role="img" aria-label="UPF Figura 1 preview"></svg>
          <div class="chart-tooltip" id="upf-fig1-overview-tooltip" hidden></div>
        </div>
      </div>

      <div class="card" id="overview-fading-card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Série Temporal Fading Accuracy &amp; Brier / Perda Calibrada <span class="paper-tag">Prequential Tracking</span></h3>
            <p class="card-subtitle">Acurácia com fator de esquecimento exponencial e métricas probabilísticas ao longo do fluxo.</p>
          </div>
        </div>
        <div class="legend-bar" id="fading-legend"></div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="fading-chart" role="img" aria-label="Série temporal de acurácia fading"></svg>
          <div class="chart-tooltip" id="fading-tooltip" hidden></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: CARA & UPF PAPER PLOTS -->
  <div class="tab-panel" id="tab-paper-retraining" role="tabpanel">
    <!-- UPF Fig 1 Full Card -->
    <div class="card" id="upf-fig1-card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>UPF Fig. 1 — Degradação de Desempenho por Modelo ($f_i$) &amp; Trajetória de Retreinamento Ativa <span class="paper-tag">Regol et al. 2025 · Fig. 1</span></h3>
          <p class="card-subtitle">Cada curva fina mostra a acurácia $1 - pe_{i,t}$ do modelo $f_i$ (ajustado em $D_i$) avaliado nos lotes subsequentes $D_t$. A linha espessa destacada segue o modelo em produção e salta para uma nova curva $f_t$ em cada decisão de retreinamento.</p>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <label for="upf-fig1-metric" style="font-size:12px; font-weight:600;">Eixo Y:</label>
          <select id="upf-fig1-metric">
            <option value="accuracy">Acurácia (1 - pe_{i,t})</option>
            <option value="error">Taxa de Erro (pe_{i,t})</option>
          </select>
        </div>
      </div>
      <details class="guide-box">
        <summary>Fórmula e Interpretação (UPF Seção 2 &amp; Fig. 1)</summary>
        <div class="formula-block">pe_{i,t} = Erro do modelo f_i (treinado até o lote i) avaliado no lote de consulta D_t (t ≥ i) · Acurácia_{i,t} = 1 - pe_{i,t}</div>
        <p>Quando não há retreinamento (<em>Frozen / NR</em>), o sistema permanece na curva inicial $f_{\text{init}}$, sofrendo degradação por concept drift. Ao retreinar no lote $r_k$, a trajetória salta para a curva $f_{r_k}$ (marcada com estrela/diamante), restaurando a acurácia ao custo de $\alpha$ unidades.</p>
      </details>
      <div class="legend-bar" id="upf-fig1-legend"></div>
      <div class="chart-wrap">
        <svg class="chart-svg tall" id="upf-fig1-chart" role="img" aria-label="UPF Fig. 1 Degradação por modelo e trajetória"></svg>
        <div class="chart-tooltip" id="upf-fig1-tooltip" hidden></div>
      </div>
    </div>

    <!-- CARA Figs 4-7 & UPF Fig 2 / Fig 4: 4-Panel Synchronized Curves vs alpha -->
    <div class="card" id="alpha-curves-card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>CARA Figs. 4–7 &amp; UPF Figs. 2, 4 — Avaliação em Função do Custo de Retreinamento $\alpha$ ($\kappa$) <span class="paper-tag">CARA Figs. 4–7 · UPF Figs. 2 &amp; 4</span></h3>
          <p class="card-subtitle">Painel quádruplo sincronizado ao longo de $\alpha \in [0, \alpha_{\max}]$ com formas de marcadores distintas por algoritmo e limites teóricos da Proposição 3.1 ($r^* \le 0, 1, 2$).</p>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <label for="cost-type-selector" style="font-size:12px; font-weight:600;">Tipo de Custo no Painel (a):</label>
          <select id="cost-type-selector">
            <option value="cost">Custo de Erro Normalizado C_hat_α(θ) (UPF Eq. 2)</option>
            <option value="strategy_cost">Custo de Staleness Estratégico (CARA Eq. 3)</option>
          </select>
        </div>
      </div>
      <details class="guide-box">
        <summary>Fórmulas Matemáticas: Custo $\hat{C}_\alpha(\theta)$, SCPE e Limite Teórico de Lipschitz (Proposição 3.1)</summary>
        <div class="formula-block">UPF Custo Normalizado: C_hat_α(θ) = (1 / C_0) · [ Σ_{t=m}^{T} pe_{m(t), t} + α · |R(θ)| ]   onde C_0 = Σ_{t=m}^{T} pe_{init, t}</div>
        <div class="formula-block">CARA Strategy Cost: S_κ(π) = Σ_{t=m}^{T} [ Ψ_bar_{t, m(t)} + κ · I(retrain em t) ]   ·   SCPE(%) = 100 · (C(θ) - C(Oracle)) / C(Oracle)</div>
        <div class="formula-block">UPF Proposição 3.1 (Limite Superior de Retreinamentos): Se |pe_{i,t} - pe_{i+1,t}| ≤ L, então r*(α) ≤ max(0, T - √(α / L))  ⇔  r* ≤ r quando α ≥ L · (T - r)²</div>
      </details>
      <div class="legend-bar" id="alpha-panels-legend"></div>
      <div class="grid-2">
        <div class="chart-wrap">
          <div style="font-size:12px; font-weight:700; margin-bottom:4px; color:var(--text-muted);">(a) Custo Estratégico / Normalizado $\hat{C}_\alpha(\theta)$ vs $\alpha$ + Limites Prop. 3.1</div>
          <svg class="chart-svg short" id="alpha-cost-chart" role="img" aria-label="Custo vs alpha"></svg>
          <div class="chart-tooltip" id="alpha-cost-tooltip" hidden></div>
        </div>
        <div class="chart-wrap">
          <div style="font-size:12px; font-weight:700; margin-bottom:4px; color:var(--text-muted);">(b) # Decisões de Retreinamento $|R(\theta)|$ vs $\alpha$ + Curva Limite $T - \sqrt{\alpha/L}$</div>
          <svg class="chart-svg short" id="alpha-retrains-chart" role="img" aria-label="Número de retreinamentos vs alpha"></svg>
          <div class="chart-tooltip" id="alpha-retrains-tooltip" hidden></div>
        </div>
        <div class="chart-wrap">
          <div style="font-size:12px; font-weight:700; margin-bottom:4px; color:var(--text-muted);">(c) Acurácia Prequencial Média (Query Accuracy) vs $\alpha$</div>
          <svg class="chart-svg short" id="alpha-accuracy-chart" role="img" aria-label="Acurácia vs alpha"></svg>
          <div class="chart-tooltip" id="alpha-accuracy-tooltip" hidden></div>
        </div>
        <div class="chart-wrap">
          <div style="font-size:12px; font-weight:700; margin-bottom:4px; color:var(--text-muted);">(d) SCPE (%) — Erro Percentual Relativo ao Oráculo vs $\alpha$</div>
          <svg class="chart-svg short" id="alpha-scpe-chart" role="img" aria-label="SCPE vs alpha"></svg>
          <div class="chart-tooltip" id="alpha-scpe-tooltip" hidden></div>
        </div>
      </div>
    </div>

    <!-- CARA Fig 3(a) & Fig A.8(c,e): Upper-Triangular Staleness Matrix & Performance Matrix with Overlaid Strategy Path -->
    <div class="card" id="matrices-trajectory-card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>CARA Fig. 3(a) &amp; Fig. A.8(c,e) — Matriz Triangular de Staleness $\bar{\Psi}_{t,t'}$ &amp; Matriz de Erro $pe_{i,j}$ com Trajetória da Política <span class="paper-tag">CARA Fig. 3a / A.8</span></h3>
          <p class="card-subtitle">Mapas de calor triangulares superiores ($t' \le t$) sobrepostos com a trajetória em degraus da política ativa no custo $\alpha$ selecionado: segmentos horizontais indicam <strong>Manter Modelo (Keep)</strong> e saltos diagonais indicam <strong>Retreinar (Retrain)</strong>.</p>
        </div>
      </div>
      <details class="guide-box">
        <summary>Como ler a Matriz de Staleness $\bar{\Psi}_{t,t'}$ e a Trajetória de Decisão (CARA Fig. 3a)</summary>
        <div class="formula-block">Ψ_bar_{t, t'} = Perda excedente (staleness) ao servir o lote de consulta D_t usando o modelo antigo treinado no lote t' (t' ≤ t), relativo ao modelo atualizado em t.</div>
        <p>No eixo horizontal está o lote avaliado $t$ e no eixo vertical o lote em que o modelo em uso foi treinado $t'$. Uma política que nunca retreina permanece na linha horizontal $t' = t_{\text{init}}$. Cada decisão de retreinamento no lote $t$ move a trajetória para a diagonal principal ($t' = t$, onde $\bar{\Psi}_{t,t} = 0$).</p>
      </details>
      <div class="grid-2">
        <div>
          <div style="font-size:12.5px; font-weight:700; margin-bottom:6px;">Matriz de Staleness $\bar{\Psi}_{t,t'}$ + Trajetória (<span id="staleness-traj-label">Política</span>)</div>
          <div class="chart-wrap">
            <svg class="chart-svg tall" id="staleness-matrix-chart" role="img" aria-label="Mapa de calor da matriz de staleness com trajetória"></svg>
            <div class="chart-tooltip" id="staleness-matrix-tooltip" hidden></div>
          </div>
        </div>
        <div>
          <div style="font-size:12.5px; font-weight:700; margin-bottom:6px;">Matriz de Erro / Desempenho $pe_{i,j}$ + Trajetória &amp; Oráculo</div>
          <div class="chart-wrap">
            <svg class="chart-svg tall" id="performance-matrix-chart" role="img" aria-label="Mapa de calor da matriz de desempenho com trajetória"></svg>
            <div class="chart-tooltip" id="performance-matrix-tooltip" hidden></div>
          </div>
        </div>
      </div>
    </div>

    <!-- CARA Fig 3(b) Optimal Parameters & CARA Fig B.10 Cumulative Strategy Cost -->
    <div class="grid-2">
      <div class="card" id="cara-params-card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>CARA Fig. 3(b) — Parâmetros Ótimos ($\tau^*, \tau^*_{\text{cum}}, \phi^*$) vs $\alpha$ ($\kappa$) <span class="paper-tag">CARA Fig. 3b</span></h3>
            <p class="card-subtitle">Limiares ótimos aprendidos nos lotes offline por CARA ($\tau^*$), CARA-CT ($\tau^*_{\text{cum}}$) e período ótimo CARA-P ($\phi^*$) em função de $\alpha$.</p>
          </div>
        </div>
        <div class="legend-bar" id="cara-params-legend"></div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="cara-params-chart" role="img" aria-label="Parâmetros ótimos CARA vs alpha"></svg>
          <div class="chart-tooltip" id="cara-params-tooltip" hidden></div>
        </div>
      </div>

      <div class="card" id="cumulative-cost-card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>CARA Fig. B.10 — Custo Estratégico Acumulado vs Lotes Online ($t$) <span class="paper-tag">CARA Fig. B.10</span></h3>
            <p class="card-subtitle">Evolução lote a lote do custo estratégico acumulado $\sum_{s=m}^{t} (\bar{\Psi}_{s, m(s)} + \alpha \cdot \mathbb{I}_{\text{retrain}})$ no $\alpha_{\text{focus}}$ para todas as estratégias.</p>
          </div>
        </div>
        <div class="legend-bar" id="cumulative-cost-legend"></div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="cumulative-cost-chart" role="img" aria-label="Custo estratégico acumulado por lote online"></svg>
          <div class="chart-tooltip" id="cumulative-cost-tooltip" hidden></div>
        </div>
      </div>
    </div>

    <!-- CARA Fig E.14 Pareto Trade-off & UPF Fig 3 Wrong-Alpha Sensitivity Heatmap -->
    <div class="grid-2">
      <div class="card" id="tradeoff-pareto-card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>CARA Fig. E.14 — Trade-off Acurácia Prequencial vs Esforço de Retreinamento <span class="paper-tag">CARA Fig. E.14</span></h3>
            <p class="card-subtitle">Fronteira de Pareto entre número de retreinamentos (ou tempo de decisão) e acurácia média de consulta ao variar $\alpha$, com símbolos geométricos distintos por política.</p>
          </div>
        </div>
        <div class="legend-bar" id="tradeoff-pareto-legend"></div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="tradeoff-pareto-chart" role="img" aria-label="Trade-off acurácia vs retreinamentos"></svg>
          <div class="chart-tooltip" id="tradeoff-pareto-tooltip" hidden></div>
        </div>
      </div>

      <div class="card" id="wrong-alpha-card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>UPF Fig. 3 (Apêndice A.2) — Sensibilidade a $\alpha$ Incorreto (`wrong_alpha`) <span class="paper-tag">UPF Fig. 3</span></h3>
            <p class="card-subtitle">Aumento percentual no custo $\hat{C}_{\alpha_{\text{real}}}$ quando a política é configurada com $\alpha_{\text{fornecido}} \ne \alpha_{\text{real}}$.</p>
          </div>
          <div style="display:flex; gap:6px; align-items:center;">
            <label for="wrong-alpha-rule" style="font-size:12px; font-weight:600;">Política:</label>
            <select id="wrong-alpha-rule">
              <option value="upf">UPF</option>
              <option value="cara">CARA</option>
            </select>
          </div>
        </div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="wrong-alpha-heatmap" role="img" aria-label="Mapa de calor de sensibilidade a alpha incorreto"></svg>
          <div class="chart-tooltip" id="wrong-alpha-tooltip" hidden></div>
        </div>
      </div>
    </div>

    <!-- UPF Figs 5-8 Forecaster Diagnostics -->
    <div class="card" id="forecast-diagnostics-card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>UPF Figs. 5–8 (Apêndice A.7) — Diagnóstico de Qualidade do Previsor de Desempenho (`forecast`) <span class="paper-tag">UPF Figs. 5–8</span></h3>
          <p class="card-subtitle">Calibração das previsões de erro futuro $\hat{pe}_{i,j}$ geradas pelo filtro probabilístico UPF vs erro observado $pe_{i,j}$, MAE e viés (Bias).</p>
        </div>
        <div id="forecast-kpi-badges" style="display:flex; gap:8px;"></div>
      </div>
      <div class="grid-2">
        <div class="chart-wrap">
          <div style="font-size:12px; font-weight:700; margin-bottom:4px; color:var(--text-muted);">Erro Previsto $\hat{pe}_{i,j}$ vs Erro Observado $pe_{i,j}$ (Linha Ideal $y=x$)</div>
          <svg class="chart-svg short" id="forecast-scatter-chart" role="img" aria-label="Dispersão erro previsto vs observado"></svg>
          <div class="chart-tooltip" id="forecast-scatter-tooltip" hidden></div>
        </div>
        <div class="chart-wrap">
          <div style="font-size:12px; font-weight:700; margin-bottom:4px; color:var(--text-muted);">Distribuição dos Resíduos de Previsão ($\hat{pe}_{i,j} - pe_{i,j}$)</div>
          <svg class="chart-svg short" id="forecast-hist-chart" role="img" aria-label="Histograma de resíduos do previsor UPF"></svg>
          <div class="chart-tooltip" id="forecast-hist-tooltip" hidden></div>
        </div>
      </div>
    </div>

    <!-- Paper Summary Tables (CARA Tables 2-4 & UPF Tables 1-2) -->
    <div class="card" id="paper-summary-tables-card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>Tabelas Comparativas dos Artigos Base (CARA Tables 2–4 &amp; UPF Tables 1–2) <span class="paper-tag">Paper Benchmarks</span></h3>
          <p class="card-subtitle">Comparação consolidada de AUC de $\hat{C}_\alpha(\theta)$, Mean SCPE (%), Query Accuracy, # Retrains e Regimes de Custo Baixo ($\alpha_{\text{low}} = 0.1\alpha_{\max}$), Foco e Alto ($\alpha_{\text{high}} = 0.9\alpha_{\max}$). <strong>Negrito</strong> indica o melhor método (excluindo Oráculo) e <u>sublinhado</u> o segundo melhor.</p>
        </div>
      </div>
      <div class="grid-2" style="margin-bottom:14px;">
        <div>
          <h4 style="margin:0 0 8px; font-size:13.5px;">Tabela 1 — Resumo Global ao Longo da Grade de $\alpha$ (CARA Tables 2–4 / UPF Table 1)</h4>
          <div id="cara-summary-table"></div>
        </div>
        <div>
          <h4 style="margin:0 0 8px; font-size:13.5px;">Gráfico Comparativo de AUC $\hat{C}_\alpha(\theta)$ &amp; Mean SCPE (%)</h4>
          <div class="chart-wrap">
            <svg class="chart-svg short" id="auc-bar-chart" role="img" aria-label="Comparação de AUC e SCPE por política"></svg>
            <div class="chart-tooltip" id="auc-bar-tooltip" hidden></div>
          </div>
        </div>
      </div>
      <h4 style="margin:10px 0 8px; font-size:13.5px;">Tabela 2 — Comparação por Regime de Custo: Baixo ($\alpha_{\text{low}}$), Foco ($\alpha_{\text{focus}}$) e Alto ($\alpha_{\text{high}}$) (UPF Table 2)</h4>
      <div id="upf-regime-table"></div>
    </div>
  </div>

  <!-- TAB 3: CONTINUAL LEARNING & RETENTION -->
  <div class="tab-panel" id="tab-retention" role="tabpanel">
    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>Curvas de Continual Learning, Esquecimento Catastrófico &amp; Transferência Temporal <span class="paper-tag">CL &amp; Retention</span></h3>
          <p class="card-subtitle">Evolução de `cl_average_accuracy`, `cl_average_incremental_accuracy`, `cl_forgetting`, `cl_backward_transfer`, `cl_forward_transfer`, `cl_intransigence`, `absolute_retention`, `gem_bwt` e `post_learning_fm`.</p>
        </div>
      </div>
      <details class="guide-box">
        <summary>Fórmulas de Esquecimento (Forgetting), GEM BWT e Retenção Histórica</summary>
        <div class="formula-block">Forgetting (FM): F_k = mean_{j &lt; k} max_{l ∈ {j..k-1}} (R_{l,j} - R_{k,j})   ·   GEM BWT = (1 / (k-1)) Σ_{j=1}^{k-1} (R_{k,j} - R_{j,j})</div>
        <div class="formula-block">Absolute Retention = Média da acurácia nos probes históricos de episódios anteriores   ·   Post-Learning FM = Queda máxima pós-aprendizado por período</div>
      </details>
      <div class="legend-bar" id="retention-legend"></div>
      <div class="chart-wrap">
        <svg class="chart-svg tall" id="retention-chart" role="img" aria-label="Curvas de retenção e esquecimento"></svg>
        <div class="chart-tooltip" id="retention-tooltip" hidden></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Matriz de Retenção Histórica $R_{i,j}$ (Checkpoints × Períodos) <span class="paper-tag">Historical Probes</span></h3>
            <p class="card-subtitle">Cada célula $(i, j)$ exibe o desempenho do checkpoint $i$ avaliado nas amostras independentes de retenção do período $j$.</p>
          </div>
        </div>
        <div id="retention-matrix-container"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Matriz de Tarefas Continual Learning (Task × Episode + Baselines Single/Joint) <span class="paper-tag">Continual Matrix</span></h3>
            <p class="card-subtitle">Desempenho por tarefa e comparação com os baselines de referência `single_accuracy` e `joint_accuracy`.</p>
          </div>
        </div>
        <div id="continual-matrix-container"></div>
      </div>
    </div>
  </div>

  <!-- TAB 4: CLASSIFICATION, ROC, PR & CONFUSION -->
  <div class="tab-panel" id="tab-classification" role="tabpanel">
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Curva ROC (FPR vs TPR) &amp; Recall@FPR=1% <span class="paper-tag">ROC Diagnostics</span></h3>
            <p class="card-subtitle">Taxa de Verdadeiros Positivos vs Taxa de Falsos Positivos comparada à diagonal aleatória ($y=x$).</p>
          </div>
        </div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="roc-chart" role="img" aria-label="Curva ROC"></svg>
          <div class="chart-tooltip" id="roc-tooltip" hidden></div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Curva Precision-Recall (Recall vs Precision) &amp; Average Precision <span class="paper-tag">PR Diagnostics</span></h3>
            <p class="card-subtitle">Equilíbrio entre Precisão e Revocação na classe positiva ao variar o limiar de decisão.</p>
          </div>
        </div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="pr-chart" role="img" aria-label="Curva Precision-Recall"></svg>
          <div class="chart-tooltip" id="pr-tooltip" hidden></div>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Matriz de Confusão por Escopo (Contagens Absolutas &amp; % por Classe Real) <span class="paper-tag">Confusion Matrix</span></h3>
            <p class="card-subtitle">Distribuição de acertos e erros entre classes verdadeiras (linhas) e classes preditas (colunas).</p>
          </div>
          <div>
            <select id="confusion-scope-selector" aria-label="Escopo da matriz de confusão"></select>
          </div>
        </div>
        <div id="confusion-matrix-container"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Suporte &amp; Recall por Classe (`class_support` / `class_recall`) <span class="paper-tag">Per-Class Balance</span></h3>
            <p class="card-subtitle">Comparação de sensibilidade (recall) e volume de exemplos avaliados para cada classe.</p>
          </div>
        </div>
        <div class="chart-wrap">
          <svg class="chart-svg" id="class-recall-chart" role="img" aria-label="Suporte e recall por classe"></svg>
          <div class="chart-tooltip" id="class-recall-tooltip" hidden></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 5: DRIFT, EFFORT, COSTS & SYSTEM TELEMETRY -->
  <div class="tab-panel" id="tab-drift-system" role="tabpanel">
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Detecção de Concept Drift: Alarmes, Changepoints &amp; Atrasos (`delays`) <span class="paper-tag">Drift Recovery</span></h3>
            <p class="card-subtitle">Correspondência entre mudanças reais de conceito, alarmes disparados pelo detector e atraso até a detecção.</p>
          </div>
        </div>
        <div class="chart-wrap">
          <svg class="chart-svg short" id="drift-timeline-chart" role="img" aria-label="Linha do tempo de drift e atrasos"></svg>
          <div class="chart-tooltip" id="drift-timeline-tooltip" hidden></div>
        </div>
        <div id="drift-summary-table" style="margin-top:10px;"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Esforço de Treinamento por Ajuste (`fits`) &amp; Curva Interna (`observation_steps`) <span class="paper-tag">Training Effort</span></h3>
            <p class="card-subtitle">Amostras/apresentações consumidas e tempo (s) em cada evento de ajuste, além da evolução interna por passo.</p>
          </div>
          <div>
            <select id="fit-selector" aria-label="Selecionar ajuste para curva interna"></select>
          </div>
        </div>
        <div class="chart-wrap">
          <svg class="chart-svg short" id="effort-fits-chart" role="img" aria-label="Esforço por ajuste"></svg>
          <div class="chart-tooltip" id="effort-fits-tooltip" hidden></div>
        </div>
        <div class="chart-wrap" style="margin-top:10px;">
          <svg class="chart-svg short" id="fit-internal-steps-chart" role="img" aria-label="Passos internos do ajuste"></svg>
          <div class="chart-tooltip" id="fit-internal-steps-tooltip" hidden></div>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Telemetria de Recursos do Sistema (Memória RSS MB &amp; CPU %) <span class="paper-tag">System Telemetry</span></h3>
            <p class="card-subtitle">Monitoramento de consumo de memória e processador ao longo do tempo de execução.</p>
          </div>
        </div>
        <div class="chart-wrap">
          <svg class="chart-svg short" id="system-telemetry-chart" role="img" aria-label="Telemetria de memória RSS e CPU"></svg>
          <div class="chart-tooltip" id="system-telemetry-tooltip" hidden></div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title-group">
            <h3>Decomposição de Custos Computacionais, Orçamento &amp; Histórico de Retreinamentos <span class="paper-tag">Cost Accounting</span></h3>
            <p class="card-subtitle">Resumo financeiro/operacional, orçamento de treinamento e tabela detalhada de fits e atualizações.</p>
          </div>
        </div>
        <div id="costs-breakdown-container"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3 style="margin-top:0; font-size:14px;">Histórico Completo de Ajustes (`fits`)</h3>
        <div id="fits-table-container"></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0; font-size:14px;">Eventos de Decisão / Atualização (`updates` &amp; `candidates`)</h3>
        <div id="updates-table-container"></div>
      </div>
    </div>
  </div>

  <!-- TAB 6: COMPLETE METRICS CATALOG & MATHEMATICAL FORMULAS GUIDE -->
  <div class="tab-panel" id="tab-catalog" role="tabpanel">
    <div class="card" id="formulas-guide-card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>📘 Guia de Interpretabilidade &amp; Glossário Matemático dos Artigos Base <span class="paper-tag">Reference Guide</span></h3>
          <p class="card-subtitle">Definições formais das métricas e estruturas visuais de Mahadevan &amp; Mathioudakis (2024, CARA) e Regol et al. (2025, UPF).</p>
        </div>
      </div>
      <div class="grid-2">
        <div class="guide-box" style="margin:0;">
          <strong>1. Custo de Retreinamento $\alpha$ ($\kappa$) e Matriz de Desempenho $pe_{i,j}$ (UPF / CARA)</strong>
          <div class="formula-block">pe_{i,j} = Taxa de erro do modelo f_i (treinado até lote i) avaliado no lote futuro D_j (j ≥ i)</div>
          <p>O hiperparâmetro $\alpha$ (denotado $\kappa$ em CARA) expressa o custo relativo de executar um retreinamento completo em unidades equivalentes de erro de classificação por lote. Quando $\alpha = 0$, retreinar em todos os lotes é ótimo; quando $\alpha \ge \alpha_{\max}$, nunca retreinar (<em>Frozen</em>) torna-se ótimo.</p>
          <strong>2. Matriz de Staleness $\bar{\Psi}_{t,t'}$ e Parâmetros $(\tau^*, \tau^*_{\text{cum}}, \phi^*)$ (CARA)</strong>
          <div class="formula-block">Ψ_bar_{t, t'} = Perda de obsolescência (staleness) no lote t ao usar modelo do lote t' &lt; t</div>
          <p><strong>CARA ($\tau^*$):</strong> dispara retreinamento quando o staleness estimado excede $\tau^*$. <strong>CARA-CT ($\tau^*_{\text{cum}}$):</strong> acumula o staleness desde o último retreinamento e dispara ao ultrapassar $\tau^*_{\text{cum}}$. <strong>CARA-P ($\phi^*$):</strong> retreina periodicamente a cada $\phi^*$ lotes.</p>
        </div>
        <div class="guide-box" style="margin:0;">
          <strong>3. Custo Normalizado $\hat{C}_\alpha(\theta)$, SCPE e Proposição 3.1 (UPF)</strong>
          <div class="formula-block">C_hat_α(θ) = (Erro Acumulado(θ) + α · |R(θ)|) / Erro Acumulado(Frozen)</div>
          <div class="formula-block">SCPE(%) = 100 · [ C_α(θ) - C_α(Oracle) ] / C_α(Oracle)   ·   AUC = ∫_0^{α_max} C_hat_α(θ) dα</div>
          <p><strong>Proposição 3.1 (UPF Fig. 4):</strong> Sob suavidade de Lipschitz $L = \max_{i,t} |pe_{i,t} - pe_{i+1,t}|$, o número ótimo de retreinamentos $r^*$ em um horizonte de $T$ lotes online satisfaz $r^* \le \max(0, T - \sqrt{\alpha / L})$. Logo, nenhuma política ótima retreina mais que $r$ vezes quando $\alpha \ge L(T - r)^2$.</p>
          <strong>4. Retenção Histórica, GEM BWT &amp; Post-Learning Forgetting</strong>
          <div class="formula-block">cl_forgetting · cl_backward_transfer · absolute_retention · gem_bwt · post_learning_fm</div>
          <p>Avaliam o impacto da plasticidade na estabilidade: se o aprendizado de novos conceitos degrada o desempenho em conceitos antigos medidos em conjuntos de prova independentes (<em>probes</em>).</p>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <h3>Catálogo Completo de Métricas do Experimento (`all_metrics`)</h3>
          <p class="card-subtitle">Tabela pesquisável com todas as métricas escalares extraídas do relatório, grupo semântico e status de disponibilidade.</p>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center;">
          <input type="search" id="metric-search" placeholder="Filtrar por nome ou grupo..." style="min-width:220px;">
          <select id="metric-group">
            <option value="all">Todos os grupos</option>
          </select>
          <select id="metric-status">
            <option value="all">Todos os status</option>
            <option value="available">Disponíveis (available)</option>
            <option value="unavailable">Indisponíveis (unavailable)</option>
          </select>
        </div>
      </div>
      <div id="all-metrics-container"></div>
    </div>
  </div>
</main>

<script>
(() => {
  'use strict';

  // Colorblind-safe high-contrast publication palette (Okabe-Ito + extended distinct hues)
  const PALETTE = [
    '#0072B2', // Blue
    '#D55E00', // Vermilion / Orange-Red
    '#009E73', // Bluish Green
    '#CC79A7', // Reddish Purple
    '#E69F00', // Amber / Orange
    '#56B4E9', // Sky Blue
    '#332288', // Indigo
    '#117733', // Forest Green
    '#882255', // Wine
    '#44AA99'  // Teal
  ];

  // Distinct geometric point marker shapes and line dash styles per series
  const MARKER_SHAPES = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'star', 'triangle-down'];
  const DASH_STYLES = ['', '6,3', '2,2', '8,3,2,3', '4,4', '10,3,2,3,2,3', '5,2'];

  const RULE_STYLE = {
    oracle:           { color: '#111827', shape: 'star',          dash: '3,3',       order: 0 },
    oracle_staleness: { color: '#475569', shape: 'cross',         dash: '2,2',       order: 1 },
    upf:              { color: '#0072B2', shape: 'circle',        dash: '',          order: 2 },
    cara:             { color: '#D55E00', shape: 'square',        dash: '',          order: 3 },
    cara_cumulative:  { color: '#009E73', shape: 'triangle-up',   dash: '6,3',       order: 4 },
    cara_periodic:    { color: '#CC79A7', shape: 'diamond',       dash: '8,3,2,3',   order: 5 },
    markov:           { color: '#E69F00', shape: 'triangle-down', dash: '4,4',       order: 6 },
    frozen:           { color: '#64748b', shape: 'cross',         dash: '5,4',       order: 7 }
  };

  function getRuleAppearance(ruleName, idx) {
    if (RULE_STYLE[ruleName]) return RULE_STYLE[ruleName];
    return {
      color: PALETTE[idx % PALETTE.length],
      shape: MARKER_SHAPES[idx % MARKER_SHAPES.length],
      dash: DASH_STYLES[idx % DASH_STYLES.length],
      order: 10 + idx
    };
  }

  function isFiniteNum(v) {
    return typeof v === 'number' && Number.isFinite(v);
  }

  function fmt(v, digits = 4) {
    if (v === null || v === undefined || v === '') return '—';
    if (typeof v === 'boolean') return v ? 'Sim' : 'Não';
    if (typeof v === 'number') {
      if (!Number.isFinite(v)) return '—';
      if (Number.isInteger(v) && Math.abs(v) < 1e9) return String(v);
      if (Math.abs(v) > 0 && Math.abs(v) < 0.0001) return v.toExponential(2);
      return Number(v.toFixed(digits)).toString();
    }
    return String(v);
  }

  function deriveServedBy(retrainPoints, initialModel, offlineBatches, batches, sameBatch) {
    const rSet = new Set((retrainPoints || []).map(Number));
    let deployed = Number.isFinite(initialModel) ? initialModel : Math.max(0, offlineBatches - 1);
    const served = [];
    for (let j = offlineBatches; j < batches; j++) {
      if (sameBatch) {
        if (rSet.has(j)) deployed = j;
        served.push(deployed);
      } else {
        served.push(deployed);
        if (rSet.has(j)) deployed = j;
      }
    }
    return served;
  }

  // Normalize payload across all schema variations from dashboard_payload.py
  function normalizePayload(raw) {
    const d = raw || {};
    d.meta = d.meta || {};
    d.all_metrics = Array.isArray(d.all_metrics) ? d.all_metrics : [];
    d.windows = Array.isArray(d.windows) ? d.windows : [];
    d.fits = Array.isArray(d.fits) ? d.fits : [];
    d.updates = Array.isArray(d.updates) ? d.updates : (Array.isArray(d.candidates) ? d.candidates : []);
    d.alarms = Array.isArray(d.alarms) ? d.alarms : [];

    // Effort / resources / system / costs unification
    const ers = d.effort_resources_system || {};
    d.effort = d.effort || ers.effort || {};
    d.resources = d.resources || ers.resources || {};
    d.costs = d.costs || ers.costs || {};
    d.system_metrics = d.system_metrics || ers.system_metrics || {};
    d.drift = d.drift || ers.drift || { alarms: d.alarms, changepoints: [], recovery: [], delays: [] };

    // Retention unification
    d.retention = d.retention || {};
    d.retention.matrix = Array.isArray(d.retention.matrix) ? d.retention.matrix : [];
    d.retention.checkpoints = Array.isArray(d.retention.checkpoints) ? d.retention.checkpoints : [];
    d.retention.periods = Array.isArray(d.retention.periods) ? d.retention.periods : [];
    d.retention.series = Array.isArray(d.retention.series) ? d.retention.series : [];
    d.retention.tracking_series = d.retention.tracking_series || {};
    d.retention.continual_matrix = d.retention.continual_matrix || [];

    // Curves & confusion unification
    d.confusion = d.confusion || {};
    d.ranking_curves = d.ranking_curves || {};
    if (d.curves && typeof d.curves === 'object') {
      const c = d.curves;
      const roc = c.roc_curve || c.roc;
      const pr = c.pr_curve || c.pr;
      if ((roc || pr) && !d.ranking_curves.prequential) {
        d.ranking_curves.prequential = {
          roc: roc && Array.isArray(roc.fpr) && roc.fpr.length ? roc : null,
          pr: pr && Array.isArray(pr.recall) && pr.recall.length ? pr : null,
          residuals: c.residuals || null
        };
      }
      if (c.by_scope && typeof c.by_scope === 'object') {
        Object.entries(c.by_scope).forEach(([sc, scVal]) => {
          if (!scVal || typeof scVal !== 'object') return;
          const sRoc = scVal.roc_curve || scVal.roc;
          const sPr = scVal.pr_curve || scVal.pr;
          d.ranking_curves[sc] = {
            roc: sRoc && Array.isArray(sRoc.fpr) && sRoc.fpr.length ? sRoc : null,
            pr: sPr && Array.isArray(sPr.recall) && sPr.recall.length ? sPr : null,
            residuals: scVal.residuals || null
          };
        });
      }
      const cm = c.class_metrics || {};
      if (Array.isArray(cm.classes) && cm.classes.length) {
        if (!d.confusion.prequential) {
          d.confusion.prequential = { classes: cm.classes.map(String), matrix: [], support: cm.class_support || [], recall: cm.class_recall || [] };
        } else {
          d.confusion.prequential.support = d.confusion.prequential.support || cm.class_support || [];
          d.confusion.prequential.recall = d.confusion.prequential.recall || cm.class_recall || [];
        }
      }
      if (Array.isArray(c.fading_accuracy) && c.fading_accuracy.length) {
        d.fading_series = c.fading_accuracy.map((pt, idx) =>
          typeof pt === 'object' && pt !== null
            ? { step: pt.step ?? pt.x ?? idx + 1, value: pt.value ?? pt.y ?? null }
            : { step: idx + 1, value: isFiniteNum(pt) ? pt : null }
        );
      }
    }
    if (!d.fading_series) {
      const ts = (d.tracking_series && d.tracking_series.fading_accuracy) || (d.retention.tracking_series && d.retention.tracking_series.fading_accuracy);
      if (Array.isArray(ts)) d.fading_series = ts;
    }

    // Retraining normalization
    if (d.retraining && typeof d.retraining === 'object' && d.retraining.rules && Object.keys(d.retraining.rules).length) {
      const ret = d.retraining;
      const setup = ret.setup || {};
      let perfMat = ret.performance_matrix || {};
      if (Array.isArray(perfMat)) {
        perfMat = { err: perfMat };
      }
      perfMat.err = Array.isArray(perfMat.err) ? perfMat.err : (Array.isArray(perfMat.matrix) ? perfMat.matrix : []);
      const batches = Number(setup.batches || perfMat.err.length || 0);
      const offline = Number(setup.offline_batches ?? setup.offline ?? perfMat.offline ?? 0);
      const online = Number(setup.online_batches ?? setup.online ?? Math.max(0, batches - offline));
      const initial = Number(setup.initial_model ?? setup.initial ?? perfMat.initial ?? Math.max(0, offline - 1));
      const timingStr = String(setup.timing || '');
      const sameBatch = Boolean(perfMat.paper_timing || timingStr.includes('same_batch') || timingStr.includes('CARA accounting'));

      setup.batches = batches;
      setup.offline_batches = offline;
      setup.online_batches = online;
      setup.initial_model = initial;
      ret.setup = setup;
      ret.performance_matrix = perfMat;

      if (ret.staleness_matrix && !Array.isArray(ret.staleness_matrix) && Array.isArray(ret.staleness_matrix.matrix)) {
        ret.staleness_matrix = ret.staleness_matrix.matrix;
      }
      ret.staleness_matrix = Array.isArray(ret.staleness_matrix) ? ret.staleness_matrix : [];

      // Proposition 3.1 bounds normalization
      const pb = ret.proposition_bounds || {};
      const L = setup.lipschitz_L ?? pb.L ?? pb.lipschitz_L ?? null;
      setup.lipschitz_L = L;
      const boundAlphas = Object.assign({}, setup.bound_alphas || {}, pb.bound_alphas || pb.thresholds || pb.bounds || {});
      if (isFiniteNum(L) && online > 0 && Object.keys(boundAlphas).length === 0) {
        for (let r = 0; r < 3; r++) {
          boundAlphas[`r<=${r}`] = L * Math.pow(Math.max(0, online - r), 2);
        }
      }
      setup.bound_alphas = boundAlphas;

      // Normalize each rule's curve & parallel arrays
      const topAlphas = Array.isArray(ret.alphas) ? ret.alphas : [];
      Object.entries(ret.rules).forEach(([rName, rInfo]) => {
        if (!rInfo || typeof rInfo !== 'object') return;
        const rAlphas = Array.isArray(rInfo.alphas) && rInfo.alphas.length ? rInfo.alphas : topAlphas;
        if (!Array.isArray(rInfo.curve) || !rInfo.curve.length) {
          const costs = rInfo.costs || [];
          const retrainsC = rInfo.retrains_curve || [];
          const accC = rInfo.accuracies_curve || [];
          const scpeC = rInfo.scpe_curve || [];
          const stratC = rInfo.strategy_costs_curve || [];
          const errGapC = rInfo.error_oracle_gap_curve || [];
          const stratGapC = rInfo.strategy_oracle_gap_curve || [];
          const rPtsByAlpha = rInfo.retrain_points_by_alpha || [];
          const paramsByAlpha = rInfo.parameters_by_alpha || [];
          rInfo.curve = rAlphas.map((a, idx) => {
            const rPts = Array.isArray(rPtsByAlpha)
              ? (rPtsByAlpha[idx] || [])
              : (rPtsByAlpha && rPtsByAlpha[String(a)]) || [];
            const prms = Array.isArray(paramsByAlpha)
              ? (paramsByAlpha[idx] || {})
              : (paramsByAlpha && paramsByAlpha[String(a)]) || {};
            return {
              alpha: a,
              cost: costs[idx] ?? null,
              retrains: retrainsC[idx] ?? (Array.isArray(rPts) ? rPts.length : 0),
              retrain_points: Array.isArray(rPts) ? rPts : [],
              average_accuracy: accC[idx] ?? null,
              scpe: scpeC[idx] ?? null,
              strategy_cost: stratC[idx] ?? null,
              error_oracle_gap: errGapC[idx] ?? null,
              strategy_oracle_gap: stratGapC[idx] ?? null,
              parameters: prms || {}
            };
          });
        }
        rInfo.curve.forEach(pt => {
          pt.retrain_points = Array.isArray(pt.retrain_points) ? pt.retrain_points : [];
          if (!Array.isArray(pt.served_by) || !pt.served_by.length) {
            if (batches > offline) {
              pt.served_by = deriveServedBy(pt.retrain_points, initial, offline, batches, sameBatch && rName !== 'upf' && rName !== 'oracle');
            } else {
              pt.served_by = [];
            }
          }
        });
        if (!Array.isArray(rInfo.alphas) || !rInfo.alphas.length) {
          rInfo.alphas = rInfo.curve.map(c => c.alpha);
        }
        if (!Array.isArray(rInfo.costs) || !rInfo.costs.length) {
          rInfo.costs = rInfo.curve.map(c => c.cost);
        }
      });

      if (!Array.isArray(ret.alphas) || !ret.alphas.length) {
        const firstRule = Object.values(ret.rules)[0];
        ret.alphas = firstRule ? (firstRule.alphas || []) : [];
      }

      // Normalize wrong_alpha
      const waRaw = ret.wrong_alpha || {};
      const waNorm = {};
      if (waRaw.percent_increase && typeof waRaw.percent_increase === 'object') {
        const prov = waRaw.provided_alphas || waRaw.alphas || ret.alphas || [];
        const real = waRaw.real_alphas || waRaw.alphas || ret.alphas || [];
        Object.entries(waRaw.percent_increase).forEach(([rk, mat]) => {
          if (Array.isArray(mat)) {
            waNorm[rk] = { provided: prov, real: real, pct_increase: mat };
          }
        });
      }
      Object.entries(waRaw).forEach(([rk, val]) => {
        if (rk === 'percent_increase' || rk === 'alphas' || rk === 'real_alphas' || rk === 'provided_alphas') return;
        if (val && typeof val === 'object') {
          const mat = val.pct_increase || val.percent_increase || val.matrix || [];
          if (Array.isArray(mat) && mat.length) {
            waNorm[rk] = {
              provided: val.provided || val.provided_alphas || val.alphas || ret.alphas || [],
              real: val.real || val.real_alphas || val.alphas || ret.alphas || [],
              pct_increase: mat
            };
          }
        }
      });
      ret.wrong_alpha = waNorm;
    } else {
      d.retraining = null;
    }
    return d;
  }

  const RAW_DATA = JSON.parse(document.getElementById('benchmark-data').textContent || '{}');
  const DATA = normalizePayload(RAW_DATA);

  // Global Interactive State
  const state = {
    mode: 'global',           // 'global' | 'live'
    sliderPct: 100,           // 1..100
    playing: false,
    playTimer: null,
    smoothStrategy: 'raw',    // 'raw' | 'sma' | 'ema' | 'median' | 'savgol'
    smoothWindow: 5,
    selectedPolicy: 'cara',
    alphaIndex: 0,
    costMetricType: 'cost',   // 'cost' | 'strategy_cost'
    upfFig1Metric: 'accuracy',
    wrongAlphaRule: 'upf',
    selectedScope: 'prequential',
    confusionScope: 'prequential',
    selectedFitIdx: 0,
    hiddenSeries: new Set(),
    hoveredSeries: null
  };

  // Initialize default policy & alpha index
  if (DATA.retraining && DATA.retraining.rules) {
    const ruleKeys = Object.keys(DATA.retraining.rules);
    if (DATA.meta.policy && DATA.retraining.rules[DATA.meta.policy]) {
      state.selectedPolicy = DATA.meta.policy;
    } else if (DATA.retraining.rules.cara) {
      state.selectedPolicy = 'cara';
    } else if (DATA.retraining.rules.upf) {
      state.selectedPolicy = 'upf';
    } else if (ruleKeys.length) {
      state.selectedPolicy = ruleKeys[0];
    }
    const alphas = DATA.retraining.alphas || [];
    const focusA = DATA.retraining.setup?.focus_alpha;
    if (alphas.length && isFiniteNum(focusA)) {
      let bestIdx = 0;
      let bestDist = Infinity;
      alphas.forEach((a, i) => {
        const dist = Math.abs(a - focusA);
        if (dist < bestDist) { bestDist = dist; bestIdx = i; }
      });
      const selCurve = DATA.retraining.rules[state.selectedPolicy]?.curve || [];
      const hasActiveLower = selCurve.some(c => (c.retrains || 0) > 0);
      const matchedRetrains = selCurve[bestIdx]?.retrains || 0;
      if (focusA >= alphas[alphas.length - 1] || (matchedRetrains === 0 && hasActiveLower)) {
        state.alphaIndex = Math.max(0, Math.min(alphas.length - 2, Math.floor(alphas.length / 3)));
      } else {
        state.alphaIndex = bestIdx;
      }
    } else if (alphas.length) {
      state.alphaIndex = Math.max(0, Math.min(alphas.length - 2, Math.floor(alphas.length / 3)));
    }
    if (DATA.retraining.wrong_alpha) {
      if (!DATA.retraining.wrong_alpha[state.wrongAlphaRule]) {
        const waKeys = Object.keys(DATA.retraining.wrong_alpha);
        if (waKeys.length) state.wrongAlphaRule = waKeys[0];
      }
    }
  }

  // Smoothing implementations
  function applySmoothing(points, strategy, windowSize) {
    if (!Array.isArray(points) || !points.length || strategy === 'raw' || windowSize <= 1) {
      return points;
    }
    const w = Math.max(2, Math.round(windowSize));
    const ys = points.map(p => p.y);
    const out = [];
    const emaAlpha = 2 / (w + 1);
    let emaState = 0;
    let emaStarted = false;
    for (let i = 0; i < points.length; i++) {
      if (!isFiniteNum(ys[i])) {
        out.push({ ...points[i] });
        continue;
      }
      if (strategy === 'sma') {
        let sum = 0, cnt = 0;
        for (let j = Math.max(0, i - w + 1); j <= i; j++) {
          if (isFiniteNum(ys[j])) { sum += ys[j]; cnt++; }
        }
        out.push({ ...points[i], y: cnt ? sum / cnt : ys[i] });
      } else if (strategy === 'ema') {
        emaState = emaStarted ? emaAlpha * ys[i] + (1 - emaAlpha) * emaState : ys[i];
        emaStarted = true;
        out.push({ ...points[i], y: emaState });
      } else if (strategy === 'median') {
        const slice = [];
        for (let j = Math.max(0, i - w + 1); j <= i; j++) {
          if (isFiniteNum(ys[j])) slice.push(ys[j]);
        }
        slice.sort((a, b) => a - b);
        const mid = Math.floor(slice.length / 2);
        const med = slice.length % 2 !== 0 ? slice[mid] : (slice[mid - 1] + slice[mid]) / 2;
        out.push({ ...points[i], y: slice.length ? med : ys[i] });
      } else if (strategy === 'savgol') {
        // Weighted triangular / Savitzky-Golay local polynomial approximation
        let wSum = 0, weightTotal = 0;
        const half = Math.max(1, Math.floor(w / 2));
        for (let k = -half; k <= half; k++) {
          const idx = i + k;
          if (idx >= 0 && idx < ys.length && isFiniteNum(ys[idx])) {
            const weight = (half + 1 - Math.abs(k));
            wSum += ys[idx] * weight;
            weightTotal += weight;
          }
        }
        out.push({ ...points[i], y: weightTotal > 0 ? wSum / weightTotal : ys[i] });
      } else {
        out.push({ ...points[i] });
      }
    }
    return out;
  }

  // SVG Point Marker path generator (circle, square, triangle-up, diamond, cross, star, triangle-down)
  function markerSvgElement(shape, cx, cy, r, color, fill = color) {
    const ns = 'http://www.w3.org/2000/svg';
    const s = shape || 'circle';
    if (s === 'circle') {
      const el = document.createElementNS(ns, 'circle');
      el.setAttribute('cx', cx);
      el.setAttribute('cy', cy);
      el.setAttribute('r', r);
      el.setAttribute('fill', fill);
      el.setAttribute('stroke', color);
      el.setAttribute('stroke-width', '1.5');
      return el;
    }
    const path = document.createElementNS(ns, 'path');
    let d = '';
    if (s === 'square') {
      const h = r * 0.9;
      d = `M ${cx - h} ${cy - h} L ${cx + h} ${cy - h} L ${cx + h} ${cy + h} L ${cx - h} ${cy + h} Z`;
    } else if (s === 'triangle-up') {
      const h = r * 1.15;
      d = `M ${cx} ${cy - h} L ${cx + h} ${cy + h * 0.85} L ${cx - h} ${cy + h * 0.85} Z`;
    } else if (s === 'triangle-down') {
      const h = r * 1.15;
      d = `M ${cx - h} ${cy - h * 0.85} L ${cx + h} ${cy - h * 0.85} L ${cx} ${cy + h} Z`;
    } else if (s === 'diamond') {
      const h = r * 1.25;
      d = `M ${cx} ${cy - h} L ${cx + h} ${cy} L ${cx} ${cy + h} L ${cx - h} ${cy} Z`;
    } else if (s === 'cross') {
      const h = r * 1.05;
      d = `M ${cx - h} ${cy - h} L ${cx + h} ${cy + h} M ${cx - h} ${cy + h} L ${cx + h} ${cy - h}`;
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '2.2');
      path.setAttribute('d', d);
      return path;
    } else if (s === 'star') {
      const outer = r * 1.35;
      const inner = r * 0.58;
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const rad = (Math.PI / 5) * i - Math.PI / 2;
        const radius = i % 2 === 0 ? outer : inner;
        pts.push(`${(cx + Math.cos(rad) * radius).toFixed(2)} ${(cy + Math.sin(rad) * radius).toFixed(2)}`);
      }
      d = `M ${pts.join(' L ')} Z`;
    } else {
      const h = r;
      d = `M ${cx - h} ${cy - h} L ${cx + h} ${cy - h} L ${cx + h} ${cy + h} L ${cx - h} ${cy + h} Z`;
    }
    path.setAttribute('d', d);
    path.setAttribute('fill', fill);
    path.setAttribute('stroke', color);
    path.setAttribute('stroke-width', '1.5');
    return path;
  }

  // Interactive Legend Pill Builder with embedded SVG line + point marker shape
  function renderLegendBar(container, seriesList, onToggle) {
    if (!container) return;
    container.innerHTML = '';
    seriesList.forEach((s, idx) => {
      const key = s.key || s.label || `s-${idx}`;
      const isHidden = state.hiddenSeries.has(key);
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'legend-pill' + (isHidden ? ' muted-series' : '');
      pill.title = 'Clique para exibir/ocultar série; passe o mouse para destacar';

      const ns = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(ns, 'svg');
      svg.setAttribute('class', 'legend-marker-svg');
      svg.setAttribute('viewBox', '0 0 24 14');
      const line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', '1');
      line.setAttribute('y1', '7');
      line.setAttribute('x2', '23');
      line.setAttribute('y2', '7');
      line.setAttribute('stroke', s.color || PALETTE[idx % PALETTE.length]);
      line.setAttribute('stroke-width', '2.2');
      if (s.dash) line.setAttribute('stroke-dasharray', s.dash);
      svg.appendChild(line);
      svg.appendChild(markerSvgElement(s.shape || MARKER_SHAPES[idx % MARKER_SHAPES.length], 12, 7, 3.8, s.color || PALETTE[idx % PALETTE.length]));

      const lbl = document.createElement('span');
      lbl.textContent = s.label;
      pill.append(svg, lbl);

      pill.addEventListener('click', () => {
        if (state.hiddenSeries.has(key)) state.hiddenSeries.delete(key);
        else state.hiddenSeries.add(key);
        if (onToggle) onToggle();
        else renderAll();
      });
      pill.addEventListener('mouseenter', () => {
        state.hoveredSeries = key;
        if (onToggle) onToggle();
      });
      pill.addEventListener('mouseleave', () => {
        state.hoveredSeries = null;
        if (onToggle) onToggle();
      });
      container.appendChild(pill);
    });
  }

  // Core Rich Interactive Multi-Series SVG Line/Scatter Chart Engine
  function drawInteractiveChart(svgEl, tooltipEl, seriesList, options = {}) {
    if (!svgEl) return;
    svgEl.innerHTML = '';
    const ns = 'http://www.w3.org/2000/svg';
    const W = svgEl.clientWidth || 680;
    const H = svgEl.clientHeight || 300;
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);

    const pad = {
      top: options.padTop ?? 20,
      right: options.padRight ?? 24,
      bottom: options.padBottom ?? 38,
      left: options.padLeft ?? 56
    };
    const plotW = Math.max(80, W - pad.left - pad.right);
    const plotH = Math.max(60, H - pad.top - pad.bottom);

    const visibleSeries = (seriesList || []).filter((s, idx) => {
      const k = s.key || s.label || `s-${idx}`;
      return s.visible !== false && !state.hiddenSeries.has(k);
    });

    const allPts = [];
    visibleSeries.forEach(s => {
      (s.points || []).forEach(p => {
        if (isFiniteNum(p.x) && isFiniteNum(p.y)) allPts.push(p);
      });
    });

    if (!allPts.length) {
      const txt = document.createElementNS(ns, 'text');
      txt.setAttribute('x', W / 2);
      txt.setAttribute('y', H / 2);
      txt.setAttribute('text-anchor', 'middle');
      txt.setAttribute('fill', 'var(--text-muted)');
      txt.setAttribute('font-size', '13');
      txt.textContent = options.emptyMessage || 'Sem pontos numéricos disponíveis para esta visualização.';
      svgEl.appendChild(txt);
      return;
    }

    let xMin = options.xMin !== undefined ? options.xMin : Math.min(...allPts.map(p => p.x));
    let xMax = options.xMax !== undefined ? options.xMax : Math.max(...allPts.map(p => p.x));
    let yMin = options.yMin !== undefined ? options.yMin : Math.min(...allPts.map(p => p.y));
    let yMax = options.yMax !== undefined ? options.yMax : Math.max(...allPts.map(p => p.y));

    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    if (yMin === yMax) { yMin -= 0.05; yMax += 0.05; }
    else if (options.yMin === undefined && options.yMax === undefined) {
      const span = yMax - yMin;
      yMin -= span * 0.06;
      yMax += span * 0.08;
    }

    const sx = x => pad.left + ((x - xMin) / (xMax - xMin)) * plotW;
    const sy = y => pad.top + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

    // Grid & Y ticks
    const gridG = document.createElementNS(ns, 'g');
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + (i / yTicks) * (yMax - yMin);
      const py = sy(v);
      const line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', pad.left);
      line.setAttribute('y1', py);
      line.setAttribute('x2', pad.left + plotW);
      line.setAttribute('y2', py);
      line.setAttribute('stroke', 'var(--border)');
      line.setAttribute('stroke-dasharray', i === 0 ? '' : '3,3');
      gridG.appendChild(line);

      const lbl = document.createElementNS(ns, 'text');
      lbl.setAttribute('x', pad.left - 7);
      lbl.setAttribute('y', py + 4);
      lbl.setAttribute('text-anchor', 'end');
      lbl.setAttribute('font-size', '10.5');
      lbl.setAttribute('fill', 'var(--text-muted)');
      lbl.textContent = fmt(v, 3);
      gridG.appendChild(lbl);
    }

    // X ticks
    const xTicks = Math.min(7, Math.max(2, allPts.length));
    for (let i = 0; i <= xTicks; i++) {
      const v = xMin + (i / xTicks) * (xMax - xMin);
      const px = sx(v);
      const lbl = document.createElementNS(ns, 'text');
      lbl.setAttribute('x', px);
      lbl.setAttribute('y', pad.top + plotH + 16);
      lbl.setAttribute('text-anchor', 'middle');
      lbl.setAttribute('font-size', '10.5');
      lbl.setAttribute('fill', 'var(--text-muted)');
      lbl.textContent = fmt(v, 3);
      gridG.appendChild(lbl);
    }

    // Axis titles
    if (options.xLabel) {
      const xl = document.createElementNS(ns, 'text');
      xl.setAttribute('x', pad.left + plotW / 2);
      xl.setAttribute('y', H - 5);
      xl.setAttribute('text-anchor', 'middle');
      xl.setAttribute('font-size', '11');
      xl.setAttribute('font-weight', '600');
      xl.setAttribute('fill', 'var(--text-muted)');
      xl.textContent = options.xLabel;
      gridG.appendChild(xl);
    }
    if (options.yLabel) {
      const yl = document.createElementNS(ns, 'text');
      yl.setAttribute('transform', `translate(13, ${pad.top + plotH / 2}) rotate(-90)`);
      yl.setAttribute('text-anchor', 'middle');
      yl.setAttribute('font-size', '11');
      yl.setAttribute('font-weight', '600');
      yl.setAttribute('fill', 'var(--text-muted)');
      yl.textContent = options.yLabel;
      gridG.appendChild(yl);
    }
    svgEl.appendChild(gridG);

    // Optional diagonal reference line (e.g. y = x for ROC or Forecast Calibration)
    if (options.diagonalLine) {
      const diag = document.createElementNS(ns, 'line');
      diag.setAttribute('x1', sx(xMin));
      diag.setAttribute('y1', sy(xMin));
      diag.setAttribute('x2', sx(xMax));
      diag.setAttribute('y2', sy(xMax));
      diag.setAttribute('stroke', 'var(--text-soft)');
      diag.setAttribute('stroke-width', '1.4');
      diag.setAttribute('stroke-dasharray', '5,4');
      svgEl.appendChild(diag);
    }

    // Vertical annotation lines (e.g. Proposition 3.1 bounds, active alpha, drift alarms, retrain points)
    (options.vLines || []).forEach(vl => {
      if (!isFiniteNum(vl.x) || vl.x < xMin || vl.x > xMax) return;
      const px = sx(vl.x);
      const l = document.createElementNS(ns, 'line');
      l.setAttribute('x1', px);
      l.setAttribute('y1', pad.top);
      l.setAttribute('x2', px);
      l.setAttribute('y2', pad.top + plotH);
      l.setAttribute('stroke', vl.color || 'var(--danger)');
      l.setAttribute('stroke-width', vl.width || '1.4');
      if (vl.dash) l.setAttribute('stroke-dasharray', vl.dash);
      svgEl.appendChild(l);
      if (vl.label) {
        const t = document.createElementNS(ns, 'text');
        t.setAttribute('x', px + 3);
        t.setAttribute('y', pad.top + (vl.labelOffsetY || 12));
        t.setAttribute('font-size', '9.5');
        t.setAttribute('font-weight', '700');
        t.setAttribute('fill', vl.color || 'var(--danger)');
        t.textContent = vl.label;
        svgEl.appendChild(t);
      }
    });

    // Plot series paths + distinct geometric markers
    const seriesG = document.createElementNS(ns, 'g');
    seriesG.setAttribute('class', 'series-markers');
    visibleSeries.forEach((s, sIdx) => {
      const key = s.key || s.label || `s-${sIdx}`;
      const color = s.color || PALETTE[sIdx % PALETTE.length];
      const shape = s.shape || MARKER_SHAPES[sIdx % MARKER_SHAPES.length];
      const dash = s.dash !== undefined ? s.dash : DASH_STYLES[sIdx % DASH_STYLES.length];
      const isHovered = state.hoveredSeries === key;
      const isDimmed = state.hoveredSeries && state.hoveredSeries !== key;
      const opacity = isDimmed ? '0.22' : (s.opacity || '1');
      const strokeW = isHovered ? (s.width || 2.3) + 1.2 : (s.width || 2.2);

      const pts = (s.points || []).filter(p => isFiniteNum(p.x) && isFiniteNum(p.y));
      if (!pts.length) return;

      if (!s.scatterOnly && pts.length > 1) {
        const path = document.createElementNS(ns, 'path');
        const dCmd = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`).join(' ');
        path.setAttribute('d', dCmd);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', String(strokeW));
        path.setAttribute('opacity', opacity);
        if (dash) path.setAttribute('stroke-dasharray', dash);
        seriesG.appendChild(path);
      }

      // Draw point markers (subsampled if dense so markers remain crisp and legible)
      const showMarkers = s.showMarkers !== false;
      if (showMarkers) {
        const maxMarkers = s.maxMarkers || 32;
        const stride = Math.max(1, Math.ceil(pts.length / maxMarkers));
        pts.forEach((p, idx) => {
          const isSpecial = Boolean(p.highlight || idx === 0 || idx === pts.length - 1 || idx % stride === 0);
          if (!isSpecial) return;
          const mRadius = p.highlight ? 5.2 : (s.markerRadius || 3.6);
          const mFill = p.highlight ? '#ffffff' : color;
          const mEl = markerSvgElement(p.shape || shape, sx(p.x), sy(p.y), mRadius, color, mFill);
          mEl.setAttribute('opacity', opacity);
          seriesG.appendChild(mEl);
        });
      }
    });
    svgEl.appendChild(seriesG);

    // Interactive Crosshair & Multi-Series Tooltip
    const crossLine = document.createElementNS(ns, 'line');
    crossLine.setAttribute('y1', pad.top);
    crossLine.setAttribute('y2', pad.top + plotH);
    crossLine.setAttribute('stroke', 'var(--primary)');
    crossLine.setAttribute('stroke-width', '1.2');
    crossLine.setAttribute('stroke-dasharray', '3,3');
    crossLine.setAttribute('visibility', 'hidden');
    svgEl.appendChild(crossLine);

    const overlay = document.createElementNS(ns, 'rect');
    overlay.setAttribute('x', pad.left);
    overlay.setAttribute('y', pad.top);
    overlay.setAttribute('width', plotW);
    overlay.setAttribute('height', plotH);
    overlay.setAttribute('fill', 'transparent');
    overlay.style.cursor = 'crosshair';

    overlay.addEventListener('mousemove', ev => {
      const rect = svgEl.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * W;
      const targetX = xMin + ((relX - pad.left) / plotW) * (xMax - xMin);
      crossLine.setAttribute('x1', Math.max(pad.left, Math.min(pad.left + plotW, relX)));
      crossLine.setAttribute('x2', Math.max(pad.left, Math.min(pad.left + plotW, relX)));
      crossLine.setAttribute('visibility', 'visible');

      if (!tooltipEl) return;
      const rows = [];
      let snapX = targetX;
      visibleSeries.forEach((s, idx) => {
        if (s.hideInTooltip) return;
        const pts = (s.points || []).filter(p => isFiniteNum(p.x) && isFiniteNum(p.y));
        if (!pts.length) return;
        let nearest = pts[0];
        let bestD = Math.abs(pts[0].x - targetX);
        for (let i = 1; i < pts.length; i++) {
          const d = Math.abs(pts[i].x - targetX);
          if (d < bestD) { bestD = d; nearest = pts[i]; }
        }
        snapX = nearest.x;
        rows.push({
          label: s.label,
          color: s.color || PALETTE[idx % PALETTE.length],
          val: nearest.y,
          extra: nearest.extra || ''
        });
      });

      if (!rows.length) {
        tooltipEl.hidden = true;
        return;
      }
      let html = `<div class="tt-title">${options.xLabel || 'x'} = ${fmt(snapX, 4)}</div>`;
      rows.slice(0, 10).forEach(r => {
        html += `<div class="tt-row"><span style="color:${r.color}">● ${r.label}</span><span>${fmt(r.val, 4)}${r.extra ? ' ' + r.extra : ''}</span></div>`;
      });
      tooltipEl.innerHTML = html;
      tooltipEl.hidden = false;
      const leftPx = Math.min(rect.width - 240, Math.max(8, ev.clientX - rect.left + 14));
      const topPx = Math.max(8, ev.clientY - rect.top - 24);
      tooltipEl.style.left = `${leftPx}px`;
      tooltipEl.style.top = `${topPx}px`;
    });

    overlay.addEventListener('mouseleave', () => {
      crossLine.setAttribute('visibility', 'hidden');
      if (tooltipEl) tooltipEl.hidden = true;
    });
    svgEl.appendChild(overlay);
  }

  // Render Header Metadata Pills & KPI Strip & Diagnostic Insights
  function renderHeaderAndInsights() {
    const m = DATA.meta || {};
    const pillsEl = document.getElementById('meta-pills');
    const items = [
      ['Dataset', m.dataset || '—'],
      ['Modelo', m.model || '—'],
      ['Política', m.policy || '—'],
      ['Detector', m.detector || '—'],
      ['Modo', m.mode || m.protocol || '—'],
      ['Seed', m.seed ?? '—'],
      ['Exemplos', m.observed ?? m.labeled ?? '—'],
      ['Status', m.complete ? 'Completo ✓' : 'Parcial']
    ];
    pillsEl.innerHTML = items.map(([k, v]) =>
      `<span class="pill ${k === 'Status' ? (m.complete ? 'badge-ok' : 'badge-warn') : ''}"><strong>${k}:</strong> ${v}</span>`
    ).join('');

    // Build Automatic Diagnostic Insights
    const insights = [];
    const metricMap = {};
    (DATA.all_metrics || []).forEach(row => {
      if (isFiniteNum(row.value)) {
        metricMap[row.name] = row.value;
        metricMap[`${row.group}/${row.name}`] = row.value;
      }
    });

    if (DATA.retraining && DATA.retraining.rules) {
      const rules = DATA.retraining.rules;
      const nonOracle = Object.entries(rules).filter(([k]) => !k.startsWith('oracle') && k !== 'frozen');
      if (nonOracle.length) {
        const byAuc = [...nonOracle].filter(([, r]) => isFiniteNum(r.auc)).sort((a, b) => a[1].auc - b[1].auc);
        if (byAuc.length) {
          const [bestKey, bestR] = byAuc[0];
          insights.push(`<strong>Melhor Política Cost-Aware (AUC de Ĉ_α):</strong> <code>${bestR.label || bestKey}</code> atingiu a menor área sob a curva de custo (AUC = <strong>${fmt(bestR.auc, 4)}</strong>, Mean SCPE = <strong>${fmt(bestR.mean_scpe, 2)}%</strong>).`);
        }
      }
      const pol = rules[state.selectedPolicy];
      const curPt = pol && pol.curve && pol.curve[state.alphaIndex];
      if (curPt) {
        insights.push(`<strong>Ponto de Operação Ativo (α = ${fmt(curPt.alpha, 4)}):</strong> A política <code>${pol.label || state.selectedPolicy}</code> realiza <strong>${curPt.retrains}</strong> retreinamento(s) nos lotes <code>[${(curPt.retrain_points || []).join(', ')}]</code> com acurácia média de <strong>${fmt(curPt.average_accuracy, 4)}</strong> e SCPE de <strong>${fmt(curPt.scpe, 2)}%</strong> em relação ao Oráculo.`);
      }
      const L = DATA.retraining.setup?.lipschitz_L;
      const b0 = DATA.retraining.setup?.bound_alphas?.['r<=0'];
      if (isFiniteNum(L)) {
        insights.push(`<strong>Suavidade de Lipschitz &amp; Proposição 3.1 (UPF Fig. 4):</strong> Constante empírica $L = ${fmt(L, 4)}$; para $\\alpha \\ge ${fmt(b0, 4)}$, o limite teórico garante $r^* = 0$ (nunca retreinar é ótimo).`);
      }
    }

    const ret = DATA.retention || {};
    const clForg = metricMap.cl_forgetting ?? metricMap['tracking/cl_forgetting'];
    if (isFiniteNum(ret.absolute_retention) || isFiniteNum(clForg)) {
      insights.push(`<strong>Estabilidade vs Plasticidade (Continual Learning):</strong> Retenção histórica absoluta = <strong>${fmt(ret.absolute_retention, 4)}</strong>, GEM BWT = <strong>${fmt(ret.gem_bwt, 4)}</strong>, Esquecimento (<code>cl_forgetting</code>) = <strong>${fmt(clForg, 4)}</strong>, Post-Learning FM = <strong>${fmt(ret.post_learning_fm, 4)}</strong>.`);
    }

    const drift = DATA.drift || {};
    if ((drift.alarms || []).length || (drift.changepoints || []).length) {
      insights.push(`<strong>Detecção de Concept Drift:</strong> <strong>${(drift.alarms || []).length}</strong> alarme(s) disparado(s) para <strong>${(drift.changepoints || []).length}</strong> changepoint(s) reais (F1 = <strong>${fmt(drift.f1, 3)}</strong>, atraso médio = <strong>${fmt((drift.delays || [])[0], 1)}</strong> passos).`);
    }

    if (!insights.length) {
      insights.push(`<strong>Execução concluída:</strong> ${DATA.windows.length} janelas prequenciais avaliadas e ${(DATA.all_metrics || []).length} métricas catalogadas.`);
    }
    document.getElementById('insights-list').innerHTML = insights.map(i => `<li>${i}</li>`).join('');

    // KPI Cards
    const kpiEl = document.getElementById('kpi-strip');
    const preqAcc = metricMap['scope/prequential/accuracy'] ?? metricMap.accuracy ?? metricMap.cl_average_accuracy;
    const rocAuc = metricMap.roc_auc ?? metricMap['scope/prequential/roc_auc'];
    const polInfo = DATA.retraining?.rules?.[state.selectedPolicy];
    const activeCurvePt = polInfo?.curve?.[state.alphaIndex];

    const kpis = [
      { label: 'Acurácia Prequencial', val: fmt(preqAcc, 4), sub: `ROC-AUC: ${fmt(rocAuc, 4)}` },
      { label: 'Retenção Absoluta', val: fmt(ret.absolute_retention, 4), sub: `GEM BWT: ${fmt(ret.gem_bwt, 4)}` },
      { label: 'Esquecimento (cl_forgetting)', val: fmt(clForg, 4), sub: `Post-Learning FM: ${fmt(ret.post_learning_fm, 4)}` },
      { label: `Custo Ĉ_α (${state.selectedPolicy.toUpperCase()})`, val: fmt(activeCurvePt?.cost ?? polInfo?.c_hat, 4), sub: `α = ${fmt(activeCurvePt?.alpha ?? DATA.retraining?.setup?.focus_alpha, 4)}` },
      { label: 'AUC Ĉ_α / Mean SCPE', val: fmt(polInfo?.auc, 4), sub: `Mean SCPE: ${fmt(polInfo?.mean_scpe, 2)}%` },
      { label: 'Retreinamentos / Fits', val: fmt(activeCurvePt?.retrains ?? (DATA.updates?.length || DATA.fits?.length || 0)), sub: `${DATA.fits?.length || 0} fit(s) registrados` }
    ];
    kpiEl.innerHTML = kpis.map(k => `
      <div class="kpi-card">
        <div class="kpi-label">${k.label}</div>
        <div class="kpi-val">${k.val}</div>
        <div class="kpi-sub">${k.sub}</div>
      </div>
    `).join('');

    // Live simulation status banner
    const liveBanner = document.getElementById('live-status-banner');
    if (state.mode === 'live') {
      liveBanner.hidden = false;
      const winCount = DATA.windows.length;
      const cutoffIdx = Math.max(0, Math.min(winCount - 1, Math.round((state.sliderPct / 100) * (winCount - 1))));
      const curWin = DATA.windows[cutoffIdx] || {};
      let batchMsg = '';
      if (DATA.retraining && DATA.retraining.setup) {
        const s = DATA.retraining.setup;
        const maxBatch = s.offline_batches + Math.max(1, Math.round((state.sliderPct / 100) * s.online_batches)) - 1;
        const rPts = (activeCurvePt?.retrain_points || []).filter(b => b <= maxBatch);
        const servedModel = activeCurvePt?.served_by?.[maxBatch - s.offline_batches] ?? s.initial_model;
        batchMsg = ` · Lote Online Atual: <strong>D_${maxBatch}</strong> · Modelo em Produção: <strong>f_${servedModel}</strong> · Retreinamentos até aqui: <strong>${rPts.length}</strong> (<code>[${rPts.join(', ')}]</code>)`;
      }
      liveBanner.innerHTML = `
        <div>🔴 <strong>Simulação ao Vivo Ativa (${state.sliderPct}% do fluxo):</strong> Passo <strong>${curWin.step ?? cutoffIdx + 1}</strong> · Versão do Modelo: <strong>v${curWin.model_version ?? 1}</strong>${batchMsg}</div>
        <div>Acurácia na Janela: <strong>${fmt(curWin.metrics?.accuracy, 4)}</strong></div>
      `;
    } else {
      liveBanner.hidden = true;
    }
  }

  // TAB 1: Prequential Chart & Overview
  function renderOverviewTab() {
    const scopeSel = document.getElementById('scope-selector');
    const scopes = Array.from(new Set((DATA.windows || []).map(w => w.scope || 'prequential')));
    if (!scopes.length) scopes.push('prequential');
    if (scopeSel.options.length !== scopes.length) {
      scopeSel.innerHTML = scopes.map(sc => `<option value="${sc}">${sc}</option>`).join('');
      if (scopes.includes(state.selectedScope)) scopeSel.value = state.selectedScope;
    }

    const allScopeWins = (DATA.windows || []).filter(w => (w.scope || 'prequential') === state.selectedScope);
    const maxCount = state.mode === 'live'
      ? Math.max(1, Math.round((state.sliderPct / 100) * allScopeWins.length))
      : allScopeWins.length;
    const wins = allScopeWins.slice(0, maxCount);

    const metricKeys = Array.from(new Set(wins.flatMap(w => Object.keys(w.metrics || {}))))
      .filter(k => !['tp', 'fp', 'tn', 'fn', 'support'].includes(k));
    const preferredOrder = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'brier', 'log_loss', 'ece'];
    metricKeys.sort((a, b) => {
      const ia = preferredOrder.indexOf(a);
      const ib = preferredOrder.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });

    const preqSeries = metricKeys.slice(0, 7).map((mk, idx) => {
      const rawPts = wins.map((w, i) => ({
        x: isFiniteNum(w.step) ? w.step : i + 1,
        y: w.metrics?.[mk] ?? null
      }));
      return {
        key: `preq:${mk}`,
        label: mk,
        color: PALETTE[idx % PALETTE.length],
        shape: MARKER_SHAPES[idx % MARKER_SHAPES.length],
        dash: DASH_STYLES[idx % DASH_STYLES.length],
        points: applySmoothing(rawPts, state.smoothStrategy, state.smoothWindow)
      };
    });

    const vLines = [];
    (DATA.drift?.alarms || []).forEach(a => {
      if (isFiniteNum(a.seq)) vLines.push({ x: a.seq, color: '#be123c', dash: '4,3', label: 'Alarme Drift' });
    });
    (DATA.updates || []).forEach(u => {
      if (isFiniteNum(u.seq)) vLines.push({ x: u.seq, color: '#0a7a5c', dash: '2,2', label: 'Retrain' });
    });

    renderLegendBar(document.getElementById('prequential-legend'), preqSeries, renderOverviewTab);
    drawInteractiveChart(
      document.getElementById('prequential-chart'),
      document.getElementById('prequential-tooltip'),
      preqSeries,
      { xLabel: 'Passo / Sequência (step)', yLabel: 'Valor da Métrica', vLines }
    );

    // UPF Fig 1 Preview in Overview
    renderUpfFig1Chart(
      document.getElementById('upf-fig1-overview-chart'),
      document.getElementById('upf-fig1-overview-tooltip'),
      document.getElementById('upf-fig1-overview-legend'),
      true
    );

    // Fading Accuracy & Probabilistic Calibration Series
    const fadingSeries = [];
    if (Array.isArray(DATA.fading_series) && DATA.fading_series.length) {
      const cut = state.mode === 'live' ? Math.max(1, Math.round((state.sliderPct / 100) * DATA.fading_series.length)) : DATA.fading_series.length;
      fadingSeries.push({
        key: 'fading_accuracy',
        label: 'fading_accuracy',
        color: PALETTE[0],
        shape: 'circle',
        dash: '',
        points: applySmoothing(DATA.fading_series.slice(0, cut).map(p => ({ x: p.step, y: p.value })), state.smoothStrategy, state.smoothWindow)
      });
    }
    ['brier', 'ece', 'log_loss', 'accuracy'].forEach((mk, idx) => {
      if (wins.some(w => isFiniteNum(w.metrics?.[mk]))) {
        fadingSeries.push({
          key: `fade:${mk}`,
          label: `window_${mk}`,
          color: PALETTE[(idx + 1) % PALETTE.length],
          shape: MARKER_SHAPES[(idx + 1) % MARKER_SHAPES.length],
          dash: DASH_STYLES[(idx + 1) % DASH_STYLES.length],
          points: applySmoothing(wins.map((w, i) => ({ x: w.step ?? i + 1, y: w.metrics?.[mk] ?? null })), state.smoothStrategy, state.smoothWindow)
        });
      }
    });
    renderLegendBar(document.getElementById('fading-legend'), fadingSeries, renderOverviewTab);
    drawInteractiveChart(
      document.getElementById('fading-chart'),
      document.getElementById('fading-tooltip'),
      fadingSeries,
      { xLabel: 'Passo / Janela', yLabel: 'Acurácia Fading / Calibração' }
    );
  }

  // UPF Fig. 1: Model Performance Decay (f_i evaluated on D_t) & Active Retraining Schedule Trajectory
  function renderUpfFig1Chart(svgEl, tooltipEl, legendEl, isPreview = false) {
    if (!svgEl) return;
    const ret = DATA.retraining;
    if (!ret || !ret.performance_matrix || !Array.isArray(ret.performance_matrix.err) || !ret.performance_matrix.err.length) {
      drawInteractiveChart(svgEl, tooltipEl, [], {
        emptyMessage: 'Matriz de desempenho por lote (UPF Fig. 1) disponível em execuções com análise de retreinamento em lotes.'
      });
      if (legendEl) legendEl.innerHTML = '';
      return;
    }

    const errMat = ret.performance_matrix.err;
    const B = errMat.length;
    const offline = ret.setup?.offline_batches ?? 0;
    const initial = ret.setup?.initial_model ?? Math.max(0, offline - 1);
    const useError = !isPreview && state.upfFig1Metric === 'error';
    const valTransform = v => isFiniteNum(v) ? (useError ? v : 1 - v) : null;

    const maxBatchLimit = state.mode === 'live'
      ? Math.max(offline, offline + Math.round((state.sliderPct / 100) * (B - offline)) - 1)
      : B - 1;

    const series = [];
    // Candidate model decay curves f_i across future batches t >= i
    const maxCandidateCurves = 12;
    const stride = Math.max(1, Math.ceil(B / maxCandidateCurves));
    for (let i = 0; i <= maxBatchLimit; i++) {
      if (i !== initial && i % stride !== 0 && i !== maxBatchLimit) continue;
      const pts = [];
      for (let t = Math.max(i, offline); t <= maxBatchLimit; t++) {
        const rawErr = errMat[i]?.[t];
        if (isFiniteNum(rawErr)) {
          pts.push({ x: t, y: valTransform(rawErr) });
        }
      }
      if (pts.length) {
        series.push({
          key: `model_f_${i}`,
          label: i === initial ? `Modelo Inicial f_${i}` : `f_${i} (treino D_${i})`,
          color: i === initial ? '#64748b' : '#94a3b8',
          shape: i === initial ? 'cross' : 'circle',
          dash: i === initial ? '4,3' : '2,2',
          width: i === initial ? 2.0 : 1.2,
          opacity: i === initial ? '0.85' : '0.45',
          showMarkers: i === initial,
          hideInTooltip: i !== initial && B > 10,
          points: applySmoothing(pts, state.smoothStrategy, state.smoothWindow)
        });
      }
    }

    // Overlaid Active Policy Schedule Trajectory
    const polRule = ret.rules?.[state.selectedPolicy];
    const curAlphaPt = polRule?.curve?.[state.alphaIndex] || polRule?.curve?.[0];
    const vLines = [];
    if (curAlphaPt) {
      const rSet = new Set(curAlphaPt.retrain_points || []);
      const servedBy = curAlphaPt.served_by || [];
      const trajPts = [];
      for (let t = offline; t <= maxBatchLimit; t++) {
        const modelIdx = servedBy[t - offline] ?? initial;
        const rawErr = errMat[modelIdx]?.[t];
        if (isFiniteNum(rawErr)) {
          const isRetrain = rSet.has(t);
          trajPts.push({
            x: t,
            y: valTransform(rawErr),
            highlight: isRetrain,
            shape: isRetrain ? 'star' : 'square',
            extra: `(f_${modelIdx}${isRetrain ? ' · RETRAIN ★' : ''})`
          });
          if (isRetrain) {
            vLines.push({ x: t, color: '#D55E00', dash: '3,3', label: `Retrain D_${t}` });
          }
        }
      }
      series.push({
        key: `traj_${state.selectedPolicy}`,
        label: `Trajetória Ativa: ${polRule.label || state.selectedPolicy} (α=${fmt(curAlphaPt.alpha, 3)}, r=${curAlphaPt.retrains})`,
        color: '#D55E00',
        shape: 'star',
        dash: '',
        width: 3.4,
        markerRadius: 4.5,
        maxMarkers: 60,
        points: applySmoothing(trajPts, state.smoothStrategy, state.smoothWindow)
      });
    }

    // Oracle Schedule Trajectory for comparison
    const oracleRule = ret.rules?.oracle;
    const oraclePt = oracleRule?.curve?.[state.alphaIndex];
    if (oraclePt && state.selectedPolicy !== 'oracle') {
      const oServed = oraclePt.served_by || [];
      const oPts = [];
      for (let t = offline; t <= maxBatchLimit; t++) {
        const mIdx = oServed[t - offline] ?? initial;
        const rawErr = errMat[mIdx]?.[t];
        if (isFiniteNum(rawErr)) {
          oPts.push({ x: t, y: valTransform(rawErr), extra: `(Oracle f_${mIdx})` });
        }
      }
      if (oPts.length) {
        series.push({
          key: 'traj_oracle',
          label: `Trajetória Oráculo (r=${oraclePt.retrains})`,
          color: '#0072B2',
          shape: 'diamond',
          dash: '6,3',
          width: 2.3,
          points: applySmoothing(oPts, state.smoothStrategy, state.smoothWindow)
        });
      }
    }

    const legendItems = series.filter(s => !s.hideInTooltip);
    renderLegendBar(legendEl, legendItems, () => renderUpfFig1Chart(svgEl, tooltipEl, legendEl, isPreview));
    drawInteractiveChart(svgEl, tooltipEl, series, {
      xLabel: 'Lote de Consulta Online D_t (t)',
      yLabel: useError ? 'Taxa de Erro pe_{i,t}' : 'Acurácia no Lote (1 - pe_{i,t})',
      vLines
    });
  }

  // CARA Figs 4-7 & UPF Figs 2, 4: 4-Panel Synchronized Curves vs Alpha
  function renderAlphaPanels() {
    const ret = DATA.retraining;
    if (!ret || !ret.rules) return;

    const rulesEntries = Object.entries(ret.rules).sort(
      (a, b) => getRuleAppearance(a[0], 0).order - getRuleAppearance(b[0], 0).order
    );
    const activeAlpha = ret.alphas?.[state.alphaIndex];

    // Proposition 3.1 vertical lines (r* <= 0, 1, 2)
    const propVLines = [];
    if (isFiniteNum(activeAlpha)) {
      propVLines.push({ x: activeAlpha, color: '#0f62fe', dash: '2,2', width: '1.8', label: `α=${fmt(activeAlpha, 3)}` });
    }
    const bAlphas = ret.setup?.bound_alphas || {};
    const bColors = { 'r<=0': '#be123c', 'r<=1': '#b45309', 'r<=2': '#6d28d9' };
    Object.entries(bAlphas).forEach(([k, val], idx) => {
      if (isFiniteNum(val)) {
        propVLines.push({
          x: val,
          color: bColors[k] || '#64748b',
          dash: '5,3',
          label: `Prop.3.1 ${k} (${fmt(val, 2)})`,
          labelOffsetY: 12 + idx * 12
        });
      }
    });

    const costKey = state.costMetricType === 'strategy_cost' ? 'strategy_cost' : 'cost';
    const costSeries = [];
    const retrainsSeries = [];
    const accSeries = [];
    const scpeSeries = [];

    rulesEntries.forEach(([rName, rInfo], idx) => {
      const app = getRuleAppearance(rName, idx);
      const curve = rInfo.curve || [];
      const baseMeta = {
        key: `alpha_rule_${rName}`,
        label: `${rInfo.label || rName}${isFiniteNum(rInfo.auc) ? ` (AUC=${fmt(rInfo.auc, 3)})` : ''}`,
        color: app.color,
        shape: app.shape,
        dash: app.dash
      };

      costSeries.push({
        ...baseMeta,
        points: applySmoothing(
          curve.map((c, i) => ({ x: c.alpha, y: c[costKey] ?? c.cost ?? null, highlight: i === state.alphaIndex })),
          state.smoothStrategy,
          state.smoothWindow
        )
      });
      retrainsSeries.push({
        ...baseMeta,
        points: applySmoothing(
          curve.map((c, i) => ({ x: c.alpha, y: c.retrains ?? null, highlight: i === state.alphaIndex })),
          state.smoothStrategy,
          state.smoothWindow
        )
      });
      accSeries.push({
        ...baseMeta,
        points: applySmoothing(
          curve.map((c, i) => ({ x: c.alpha, y: c.average_accuracy ?? null, highlight: i === state.alphaIndex })),
          state.smoothStrategy,
          state.smoothWindow
        )
      });
      if (rName !== 'oracle') {
        const scpeField = state.costMetricType === 'strategy_cost' ? 'strategy_scpe' : 'scpe';
        scpeSeries.push({
          ...baseMeta,
          points: applySmoothing(
            curve.map((c, i) => ({ x: c.alpha, y: c[scpeField] ?? c.scpe ?? null, highlight: i === state.alphaIndex })),
            state.smoothStrategy,
            state.smoothWindow
          )
        });
      }
    });

    // Add Proposition 3.1 theoretical upper bound curve on # retrains panel (UPF Fig. 4)
    const ubMap = ret.setup?.retrain_upper_bound || {};
    const L = ret.setup?.lipschitz_L;
    const T = ret.setup?.online_batches || 0;
    const ubPts = (ret.alphas || []).map(a => {
      const fromMap = ubMap[String(a)];
      const calc = isFiniteNum(L) && L > 0 ? Math.max(0, T - Math.sqrt(Math.max(0, a) / L)) : null;
      return { x: a, y: isFiniteNum(fromMap) ? fromMap : calc };
    }).filter(p => isFiniteNum(p.y));
    if (ubPts.length) {
      retrainsSeries.push({
        key: 'prop_3_1_bound',
        label: 'Limite Prop. 3.1: T - √(α/L)',
        color: '#be123c',
        shape: 'triangle-down',
        dash: '3,3',
        points: ubPts
      });
    }

    renderLegendBar(document.getElementById('alpha-panels-legend'), costSeries, renderAlphaPanels);
    drawInteractiveChart(
      document.getElementById('alpha-cost-chart'),
      document.getElementById('alpha-cost-tooltip'),
      costSeries,
      {
        xLabel: 'Custo de Retreinamento α (κ)',
        yLabel: state.costMetricType === 'strategy_cost' ? 'Strategy Cost S_κ(π)' : 'Custo Normalizado Ĉ_α(θ)',
        vLines: propVLines
      }
    );
    drawInteractiveChart(
      document.getElementById('alpha-retrains-chart'),
      document.getElementById('alpha-retrains-tooltip'),
      retrainsSeries,
      { xLabel: 'Custo de Retreinamento α (κ)', yLabel: '# Retreinamentos |R(θ)|', vLines: propVLines }
    );
    drawInteractiveChart(
      document.getElementById('alpha-accuracy-chart'),
      document.getElementById('alpha-accuracy-tooltip'),
      accSeries,
      { xLabel: 'Custo de Retreinamento α (κ)', yLabel: 'Query Accuracy Média', vLines: propVLines }
    );
    drawInteractiveChart(
      document.getElementById('alpha-scpe-chart'),
      document.getElementById('alpha-scpe-tooltip'),
      scpeSeries,
      { xLabel: 'Custo de Retreinamento α (κ)', yLabel: 'SCPE (%) vs Oráculo', vLines: propVLines }
    );
  }

  // CARA Fig. 3(a) & Fig. A.8(c,e): Upper-Triangular Staleness & Performance Heatmaps with Strategy Trajectory Overlay
  function drawTriangularMatrixWithTrajectory(svgEl, tooltipEl, matrix2D, options = {}) {
    if (!svgEl) return;
    svgEl.innerHTML = '';
    const ns = 'http://www.w3.org/2000/svg';
    const W = svgEl.clientWidth || 560;
    const H = svgEl.clientHeight || 350;
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);

    if (!Array.isArray(matrix2D) || !matrix2D.length) {
      const txt = document.createElementNS(ns, 'text');
      txt.setAttribute('x', W / 2);
      txt.setAttribute('y', H / 2);
      txt.setAttribute('text-anchor', 'middle');
      txt.setAttribute('fill', 'var(--text-muted)');
      txt.textContent = 'Matriz indisponível para este protocolo.';
      svgEl.appendChild(txt);
      return;
    }

    const B = matrix2D.length;
    const pad = { top: 28, right: 26, bottom: 42, left: 54 };
    const plotW = Math.max(100, W - pad.left - pad.right);
    const plotH = Math.max(100, H - pad.top - pad.bottom);
    const cellW = plotW / B;
    const cellH = plotH / B;

    const vals = [];
    for (let r = 0; r < B; r++) {
      for (let c = r; c < B; c++) {
        let v = matrix2D[r]?.[c];
        if (!isFiniteNum(v) && isFiniteNum(matrix2D[c]?.[r])) v = matrix2D[c][r];
        if (!isFiniteNum(v) && r === c && options.diagonalDefaultZero) v = 0.0;
        if (isFiniteNum(v)) vals.push(v);
      }
    }
    const vMin = vals.length ? Math.min(...vals) : 0;
    const vMax = vals.length ? Math.max(...vals) : 1;
    const span = vMax - vMin || 1;

    // Draw cells strictly upper-triangular (r <= c): row = trained batch t', col = evaluated batch t
    const cellsG = document.createElementNS(ns, 'g');
    for (let r = 0; r < B; r++) {
      for (let c = 0; c < B; c++) {
        if (r > c) continue;
        let val = matrix2D[r]?.[c];
        if (!isFiniteNum(val) && isFiniteNum(matrix2D[c]?.[r])) {
          val = matrix2D[c][r];
        }
        if (!isFiniteNum(val) && r === c && options.diagonalDefaultZero) {
          val = 0.0;
        }
        const x = pad.left + c * cellW;
        const y = pad.top + r * cellH;
        const rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', Math.max(1, cellW - 0.8));
        rect.setAttribute('height', Math.max(1, cellH - 0.8));

        if (isFiniteNum(val)) {
          const norm = Math.max(0, Math.min(1, (val - vMin) / span));
          const hue = options.colorScheme === 'warm'
            ? `rgba(213, 94, 0, ${(0.10 + norm * 0.78).toFixed(2)})`
            : `rgba(0, 114, 178, ${(0.10 + norm * 0.78).toFixed(2)})`;
          rect.setAttribute('fill', hue);
        } else {
          rect.setAttribute('fill', 'var(--surface-alt)');
        }
        rect.setAttribute('stroke', 'var(--border)');
        rect.setAttribute('stroke-width', '0.5');

        rect.addEventListener('mousemove', ev => {
          if (!tooltipEl) return;
          const box = svgEl.getBoundingClientRect();
          tooltipEl.innerHTML = `
            <div class="tt-title">Lote Treinado t' = D_${r} → Lote Avaliado t = D_${c}</div>
            <div class="tt-row"><span>${options.metricName || 'Valor'}</span><span>${fmt(val, 4)}</span></div>
          `;
          tooltipEl.hidden = false;
          tooltipEl.style.left = `${Math.min(box.width - 210, Math.max(6, ev.clientX - box.left + 12))}px`;
          tooltipEl.style.top = `${Math.max(6, ev.clientY - box.top - 20)}px`;
        });
        rect.addEventListener('mouseleave', () => { if (tooltipEl) tooltipEl.hidden = true; });
        cellsG.appendChild(rect);

        if (B <= 14 && isFiniteNum(val)) {
          const t = document.createElementNS(ns, 'text');
          t.setAttribute('x', x + cellW / 2);
          t.setAttribute('y', y + cellH / 2 + 3.5);
          t.setAttribute('text-anchor', 'middle');
          t.setAttribute('font-size', '9.5');
          t.setAttribute('font-family', 'var(--font-mono)');
          t.setAttribute('fill', 'var(--text)');
          t.setAttribute('pointer-events', 'none');
          t.textContent = fmt(val, 2);
          cellsG.appendChild(t);
        }
      }
    }
    svgEl.appendChild(cellsG);

    // Axis labels
    const stride = Math.max(1, Math.ceil(B / 12));
    for (let i = 0; i < B; i += stride) {
      const tx = document.createElementNS(ns, 'text');
      tx.setAttribute('x', pad.left + (i + 0.5) * cellW);
      tx.setAttribute('y', pad.top + plotH + 15);
      tx.setAttribute('text-anchor', 'middle');
      tx.setAttribute('font-size', '10');
      tx.setAttribute('fill', 'var(--text-muted)');
      tx.textContent = `D_${i}`;
      svgEl.appendChild(tx);

      const ty = document.createElementNS(ns, 'text');
      ty.setAttribute('x', pad.left - 6);
      ty.setAttribute('y', pad.top + (i + 0.5) * cellH + 3.5);
      ty.setAttribute('text-anchor', 'end');
      ty.setAttribute('font-size', '10');
      ty.setAttribute('fill', 'var(--text-muted)');
      ty.textContent = `f_${i}`;
      svgEl.appendChild(ty);
    }

    const xl = document.createElementNS(ns, 'text');
    xl.setAttribute('x', pad.left + plotW / 2);
    xl.setAttribute('y', H - 6);
    xl.setAttribute('text-anchor', 'middle');
    xl.setAttribute('font-size', '11');
    xl.setAttribute('font-weight', '600');
    xl.setAttribute('fill', 'var(--text-muted)');
    xl.textContent = 'Lote Avaliado t (Batch Evaluated D_t)';
    svgEl.appendChild(xl);

    const yl = document.createElementNS(ns, 'text');
    yl.setAttribute('transform', `translate(13, ${pad.top + plotH / 2}) rotate(-90)`);
    yl.setAttribute('text-anchor', 'middle');
    yl.setAttribute('font-size', '11');
    yl.setAttribute('font-weight', '600');
    yl.setAttribute('fill', 'var(--text-muted)');
    yl.textContent = "Modelo em Uso f_{t'} (Batch Trained t')";
    svgEl.appendChild(yl);

    // Offline/Online boundary line
    const offline = options.offlineBatches ?? 0;
    if (offline > 0 && offline < B) {
      const bx = pad.left + offline * cellW;
      const bl = document.createElementNS(ns, 'line');
      bl.setAttribute('x1', bx);
      bl.setAttribute('y1', pad.top);
      bl.setAttribute('x2', bx);
      bl.setAttribute('y2', pad.top + plotH);
      bl.setAttribute('stroke', 'var(--primary)');
      bl.setAttribute('stroke-width', '1.6');
      bl.setAttribute('stroke-dasharray', '4,3');
      svgEl.appendChild(bl);
    }

    // Overlay Strategy Step-Line Trajectory (Keep horizontal vs Retrain diagonal jump)
    const servedBy = options.servedBy || [];
    const retrainSet = new Set(options.retrainPoints || []);
    const maxOnlineSteps = state.mode === 'live'
      ? Math.max(1, Math.round((state.sliderPct / 100) * servedBy.length))
      : servedBy.length;

    if (servedBy.length && offline < B) {
      const trajG = document.createElementNS(ns, 'g');
      const coords = [];
      for (let idx = 0; idx < maxOnlineSteps; idx++) {
        const col = offline + idx;
        if (col >= B) break;
        const row = Math.max(0, Math.min(B - 1, servedBy[idx]));
        const cx = pad.left + (col + 0.5) * cellW;
        const cy = pad.top + (row + 0.5) * cellH;
        coords.push({ cx, cy, col, row, isRetrain: retrainSet.has(col) });
      }
      if (coords.length > 1) {
        const stepPath = document.createElementNS(ns, 'path');
        let dCmd = `M ${coords[0].cx.toFixed(1)} ${coords[0].cy.toFixed(1)}`;
        for (let i = 1; i < coords.length; i++) {
          // Step-line: horizontal first, then vertical jump to new model row
          dCmd += ` L ${coords[i].cx.toFixed(1)} ${coords[i - 1].cy.toFixed(1)} L ${coords[i].cx.toFixed(1)} ${coords[i].cy.toFixed(1)}`;
        }
        stepPath.setAttribute('d', dCmd);
        stepPath.setAttribute('fill', 'none');
        stepPath.setAttribute('stroke', '#111827');
        stepPath.setAttribute('stroke-width', '2.8');
        trajG.appendChild(stepPath);
      }
      coords.forEach(pt => {
        const m = markerSvgElement(
          pt.isRetrain ? 'star' : 'circle',
          pt.cx,
          pt.cy,
          pt.isRetrain ? 5.5 : 3.5,
          pt.isRetrain ? '#be123c' : '#111827',
          pt.isRetrain ? '#fef08a' : '#ffffff'
        );
        trajG.appendChild(m);
      });
      svgEl.appendChild(trajG);
    }
  }

  function renderMatricesAndTrajectories() {
    const ret = DATA.retraining;
    if (!ret) return;
    const polRule = ret.rules?.[state.selectedPolicy];
    const curPt = polRule?.curve?.[state.alphaIndex] || polRule?.curve?.[0] || {};
    const offline = ret.setup?.offline_batches ?? 0;

    const lbl = document.getElementById('staleness-traj-label');
    if (lbl) {
      lbl.textContent = `${polRule?.label || state.selectedPolicy} @ α=${fmt(curPt.alpha, 3)} (${curPt.retrains ?? 0} retrains)`;
    }

    drawTriangularMatrixWithTrajectory(
      document.getElementById('staleness-matrix-chart'),
      document.getElementById('staleness-matrix-tooltip'),
      ret.staleness_matrix || [],
      {
        metricName: 'Staleness Ψ_bar_{t,t\'}',
        colorScheme: 'warm',
        diagonalDefaultZero: true,
        offlineBatches: offline,
        servedBy: curPt.served_by || [],
        retrainPoints: curPt.retrain_points || []
      }
    );

    drawTriangularMatrixWithTrajectory(
      document.getElementById('performance-matrix-chart'),
      document.getElementById('performance-matrix-tooltip'),
      ret.performance_matrix?.err || [],
      {
        metricName: 'Erro pe_{i,j}',
        colorScheme: 'cool',
        offlineBatches: offline,
        servedBy: curPt.served_by || [],
        retrainPoints: curPt.retrain_points || []
      }
    );
  }

  // CARA Fig 3(b) Optimal Parameters & Fig B.10 Cumulative Strategy Cost & Fig E.14 Trade-off & UPF Fig 3 & Forecast
  function renderSecondaryPaperCharts() {
    const ret = DATA.retraining;
    if (!ret || !ret.rules) return;

    // 1. CARA Fig. 3(b): Optimal parameters (tau*, tau_cum*, phi*) vs alpha
    const paramSeries = [];
    const paramConfigs = [
      { rule: 'cara', paramKeys: ['tau', 'threshold', 'tau_star'], label: 'CARA τ* (Limiar de Staleness)', color: '#D55E00', shape: 'square', dash: '' },
      { rule: 'cara_cumulative', paramKeys: ['tau_cum', 'tau', 'threshold'], label: 'CARA-CT τ*_cum (Staleness Acumulado)', color: '#009E73', shape: 'triangle-up', dash: '6,3' },
      { rule: 'cara_periodic', paramKeys: ['phi', 'period', 'k'], label: 'CARA-P φ* (Período Ótimo de Lotes)', color: '#CC79A7', shape: 'diamond', dash: '3,3' }
    ];
    paramConfigs.forEach(cfg => {
      const r = ret.rules[cfg.rule];
      if (!r || !Array.isArray(r.curve)) return;
      const pts = [];
      r.curve.forEach(c => {
        const pObj = c.parameters || {};
        let val = null;
        for (const pk of cfg.paramKeys) {
          if (isFiniteNum(pObj[pk])) { val = pObj[pk]; break; }
        }
        if (val === null) {
          const firstFinite = Object.values(pObj).find(isFiniteNum);
          if (isFiniteNum(firstFinite)) val = firstFinite;
        }
        if (isFiniteNum(c.alpha) && isFiniteNum(val)) {
          pts.push({ x: c.alpha, y: val });
        }
      });
      if (pts.length) {
        paramSeries.push({
          key: `param_${cfg.rule}`,
          label: cfg.label,
          color: cfg.color,
          shape: cfg.shape,
          dash: cfg.dash,
          points: applySmoothing(pts, state.smoothStrategy, state.smoothWindow)
        });
      }
    });
    renderLegendBar(document.getElementById('cara-params-legend'), paramSeries, renderSecondaryPaperCharts);
    drawInteractiveChart(
      document.getElementById('cara-params-chart'),
      document.getElementById('cara-params-tooltip'),
      paramSeries,
      {
        xLabel: 'Custo de Retreinamento α (κ)',
        yLabel: 'Valor do Parâmetro Ótimo (τ*, τ*_cum, φ*)',
        emptyMessage: 'Parâmetros finitos (τ*, φ*) indisponíveis (quando nunca retreinar é ótimo, τ* = ∞).'
      }
    );

    // 2. CARA Fig. B.10: Cumulative Strategy Cost vs Online Batches t (dynamically evaluated at active alphaIndex)
    const cumObj = ret.cumulative_strategy_cost || {};
    const staleMat = Array.isArray(ret.staleness_matrix) ? ret.staleness_matrix : [];
    const offline = ret.setup?.offline_batches ?? 0;
    const activeAlphaVal = ret.alphas?.[state.alphaIndex] ?? ret.setup?.focus_alpha ?? 0;
    const cumSeries = [];
    const ruleEntriesForCum = Object.entries(ret.rules).sort(
      (a, b) => getRuleAppearance(a[0], 0).order - getRuleAppearance(b[0], 0).order
    );
    ruleEntriesForCum.forEach(([rName, rInfo], idx) => {
      const pt = rInfo.curve?.[state.alphaIndex] || rInfo.curve?.[0];
      let arr = [];
      if (staleMat.length && pt && Array.isArray(pt.served_by) && pt.served_by.length) {
        const rSet = new Set(pt.retrain_points || []);
        const alphaCost = isFiniteNum(pt.alpha) ? pt.alpha : activeAlphaVal;
        let running = 0.0;
        arr = pt.served_by.map((m, i) => {
          const t = offline + i;
          const rawStale = m === t ? 0.0 : (staleMat[t]?.[m] ?? staleMat[m]?.[t] ?? 0.0);
          const stale = isFiniteNum(rawStale) ? rawStale : 0.0;
          running += stale + (rSet.has(t) ? alphaCost : 0.0);
          return running;
        });
      } else if (Array.isArray(cumObj[rName])) {
        arr = cumObj[rName];
      }
      if (!arr.length) return;
      const app = getRuleAppearance(rName, idx);
      const cut = state.mode === 'live' ? Math.max(1, Math.round((state.sliderPct / 100) * arr.length)) : arr.length;
      const rLabel = rInfo.label || rName;
      cumSeries.push({
        key: `cum_${rName}`,
        label: `${rLabel} (r=${pt?.retrains ?? 0})`,
        color: app.color,
        shape: app.shape,
        dash: app.dash,
        points: applySmoothing(
          arr.slice(0, cut).map((v, i) => ({ x: offline + i, y: v })),
          state.smoothStrategy,
          state.smoothWindow
        )
      });
    });
    renderLegendBar(document.getElementById('cumulative-cost-legend'), cumSeries, renderSecondaryPaperCharts);
    drawInteractiveChart(
      document.getElementById('cumulative-cost-chart'),
      document.getElementById('cumulative-cost-tooltip'),
      cumSeries,
      { xLabel: `Lote Online t (D_t) @ α = ${fmt(activeAlphaVal, 4)}`, yLabel: 'Custo Estratégico Acumulado S_κ(t)' }
    );

    // 3. CARA Fig. E.14: Pareto Trade-off (Query Accuracy vs # Retrains across alpha)
    const paretoSeries = [];
    Object.entries(ret.rules).forEach(([rName, rInfo], idx) => {
      const app = getRuleAppearance(rName, idx);
      const pts = (rInfo.curve || [])
        .filter(c => isFiniteNum(c.retrains) && isFiniteNum(c.average_accuracy))
        .map(c => ({
          x: c.retrains,
          y: c.average_accuracy,
          extra: `(α=${fmt(c.alpha, 3)})`
        }))
        .sort((a, b) => a.x - b.x);
      if (pts.length) {
        paretoSeries.push({
          key: `pareto_${rName}`,
          label: rInfo.label || rName,
          color: app.color,
          shape: app.shape,
          dash: app.dash,
          markerRadius: 4.5,
          points: pts
        });
      }
    });
    renderLegendBar(document.getElementById('tradeoff-pareto-legend'), paretoSeries, renderSecondaryPaperCharts);
    drawInteractiveChart(
      document.getElementById('tradeoff-pareto-chart'),
      document.getElementById('tradeoff-pareto-tooltip'),
      paretoSeries,
      { xLabel: '# Decisões de Retreinamento |R(θ)|', yLabel: 'Acurácia Média de Consulta (Query Accuracy)' }
    );

    // 4. UPF Fig. 3 (Appendix A.2): Wrong-Alpha Sensitivity Heatmap
    const waGrid = ret.wrong_alpha?.[state.wrongAlphaRule] || Object.values(ret.wrong_alpha || {})[0];
    const waSvg = document.getElementById('wrong-alpha-heatmap');
    const waTt = document.getElementById('wrong-alpha-tooltip');
    if (waGrid && Array.isArray(waGrid.pct_increase) && waGrid.pct_increase.length) {
      drawWrongAlphaHeatmap(waSvg, waTt, waGrid);
    } else if (waSvg) {
      drawInteractiveChart(waSvg, waTt, [], { emptyMessage: 'Grade wrong_alpha indisponível para esta política.' });
    }

    // 5. UPF Figs. 5-8: Forecaster Quality Diagnostics
    const fc = ret.forecast;
    const fcBadges = document.getElementById('forecast-kpi-badges');
    if (fc && isFiniteNum(fc.mae)) {
      fcBadges.innerHTML = `
        <span class="pill badge-ok"><strong>MAE:</strong> ${fmt(fc.mae, 4)}</span>
        <span class="pill"><strong>Bias:</strong> ${fmt(fc.bias, 4)}</span>
        <span class="pill"><strong>Pares:</strong> ${fc.count ?? (fc.pairs?.length || 0)}</span>
      `;
      const pairs = Array.isArray(fc.pairs) ? fc.pairs : [];
      const scatterPts = pairs.map(p => ({ x: p[0], y: p[1] }));
      drawInteractiveChart(
        document.getElementById('forecast-scatter-chart'),
        document.getElementById('forecast-scatter-tooltip'),
        [{
          key: 'upf_forecast_pairs',
          label: 'Previsão UPF vs Realizado',
          color: '#0072B2',
          shape: 'circle',
          scatterOnly: true,
          maxMarkers: 250,
          markerRadius: 3.2,
          points: scatterPts
        }],
        { xLabel: 'Erro Previsto p̂e_{i,j}', yLabel: 'Erro Observado pe_{i,j}', diagonalLine: true }
      );

      // Residual Histogram
      const residuals = pairs.map(p => p[0] - p[1]).filter(isFiniteNum);
      drawHistogramSvg(
        document.getElementById('forecast-hist-chart'),
        document.getElementById('forecast-hist-tooltip'),
        residuals,
        { xLabel: 'Resíduo (Previsto - Observado)', yLabel: 'Frequência' }
      );
    } else {
      if (fcBadges) fcBadges.innerHTML = '<span class="pill">Previsor UPF não utilizado ou sem pares registrados</span>';
      drawInteractiveChart(document.getElementById('forecast-scatter-chart'), null, [], { emptyMessage: 'Sem pares de previsão UPF.' });
      drawInteractiveChart(document.getElementById('forecast-hist-chart'), null, [], { emptyMessage: 'Sem resíduos de previsão UPF.' });
    }
  }

  function drawWrongAlphaHeatmap(svgEl, tooltipEl, grid) {
    if (!svgEl) return;
    svgEl.innerHTML = '';
    const ns = 'http://www.w3.org/2000/svg';
    const W = svgEl.clientWidth || 540;
    const H = svgEl.clientHeight || 300;
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);

    const mat = grid.pct_increase || [];
    const prov = grid.provided || [];
    const real = grid.real || [];
    const R = mat.length;
    const C = (mat[0] || []).length;
    if (!R || !C) return;

    const pad = { top: 22, right: 20, bottom: 42, left: 58 };
    const plotW = Math.max(80, W - pad.left - pad.right);
    const plotH = Math.max(80, H - pad.top - pad.bottom);
    const cw = plotW / C;
    const ch = plotH / R;

    const allV = mat.flat().filter(isFiniteNum);
    const maxV = Math.max(1, ...allV);

    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        const val = mat[r]?.[c];
        const x = pad.left + c * cw;
        const y = pad.top + r * ch;
        const rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', Math.max(1, cw - 1));
        rect.setAttribute('height', Math.max(1, ch - 1));
        if (isFiniteNum(val)) {
          const ratio = Math.max(0, Math.min(1, val / maxV));
          rect.setAttribute('fill', r === c ? 'rgba(16, 185, 129, 0.22)' : `rgba(190, 18, 60, ${(0.08 + ratio * 0.78).toFixed(2)})`);
        } else {
          rect.setAttribute('fill', 'var(--surface-alt)');
        }
        rect.addEventListener('mousemove', ev => {
          if (!tooltipEl) return;
          const box = svgEl.getBoundingClientRect();
          tooltipEl.innerHTML = `
            <div class="tt-title">α Fornecido = ${fmt(prov[r], 3)} · α Real = ${fmt(real[c], 3)}</div>
            <div class="tt-row"><span>Aumento de Custo</span><span>+${fmt(val, 2)}%</span></div>
          `;
          tooltipEl.hidden = false;
          tooltipEl.style.left = `${Math.min(box.width - 210, Math.max(6, ev.clientX - box.left + 12))}px`;
          tooltipEl.style.top = `${Math.max(6, ev.clientY - box.top - 20)}px`;
        });
        rect.addEventListener('mouseleave', () => { if (tooltipEl) tooltipEl.hidden = true; });
        svgEl.appendChild(rect);

        if (R <= 12 && C <= 12 && isFiniteNum(val)) {
          const t = document.createElementNS(ns, 'text');
          t.setAttribute('x', x + cw / 2);
          t.setAttribute('y', y + ch / 2 + 3.5);
          t.setAttribute('text-anchor', 'middle');
          t.setAttribute('font-size', '9.5');
          t.setAttribute('font-family', 'var(--font-mono)');
          t.setAttribute('fill', 'var(--text)');
          t.setAttribute('pointer-events', 'none');
          t.textContent = `${fmt(val, 1)}%`;
          svgEl.appendChild(t);
        }
      }
    }

    const strideR = Math.max(1, Math.ceil(R / 8));
    for (let r = 0; r < R; r += strideR) {
      const ty = document.createElementNS(ns, 'text');
      ty.setAttribute('x', pad.left - 6);
      ty.setAttribute('y', pad.top + (r + 0.5) * ch + 3.5);
      ty.setAttribute('text-anchor', 'end');
      ty.setAttribute('font-size', '10');
      ty.setAttribute('fill', 'var(--text-muted)');
      ty.textContent = fmt(prov[r], 2);
      svgEl.appendChild(ty);
    }
    const strideC = Math.max(1, Math.ceil(C / 8));
    for (let c = 0; c < C; c += strideC) {
      const tx = document.createElementNS(ns, 'text');
      tx.setAttribute('x', pad.left + (c + 0.5) * cw);
      tx.setAttribute('y', pad.top + plotH + 15);
      tx.setAttribute('text-anchor', 'middle');
      tx.setAttribute('font-size', '10');
      tx.setAttribute('fill', 'var(--text-muted)');
      tx.textContent = fmt(real[c], 2);
      svgEl.appendChild(tx);
    }

    const xl = document.createElementNS(ns, 'text');
    xl.setAttribute('x', pad.left + plotW / 2);
    xl.setAttribute('y', H - 6);
    xl.setAttribute('text-anchor', 'middle');
    xl.setAttribute('font-size', '11');
    xl.setAttribute('font-weight', '600');
    xl.setAttribute('fill', 'var(--text-muted)');
    xl.textContent = 'Custo Real α_real';
    svgEl.appendChild(xl);

    const yl = document.createElementNS(ns, 'text');
    yl.setAttribute('transform', `translate(13, ${pad.top + plotH / 2}) rotate(-90)`);
    yl.setAttribute('text-anchor', 'middle');
    yl.setAttribute('font-size', '11');
    yl.setAttribute('font-weight', '600');
    yl.setAttribute('fill', 'var(--text-muted)');
    yl.textContent = 'Custo Informado α_fornecido';
    svgEl.appendChild(yl);
  }

  function drawHistogramSvg(svgEl, tooltipEl, values, options = {}) {
    if (!svgEl) return;
    if (!Array.isArray(values) || !values.length) {
      drawInteractiveChart(svgEl, tooltipEl, [], { emptyMessage: 'Sem dados para histograma.' });
      return;
    }
    const minV = Math.min(...values);
    const maxV = Math.max(...values);
    const binsCount = Math.min(18, Math.max(6, Math.round(Math.sqrt(values.length))));
    const span = (maxV - minV) || 0.1;
    const step = span / binsCount;
    const bins = Array.from({ length: binsCount }, (_, i) => ({
      x: minV + (i + 0.5) * step,
      y: 0
    }));
    values.forEach(v => {
      const idx = Math.min(binsCount - 1, Math.max(0, Math.floor((v - minV) / step)));
      bins[idx].y += 1;
    });
    drawInteractiveChart(svgEl, tooltipEl, [{
      key: 'hist_bins',
      label: 'Contagem por Faixa',
      color: '#009E73',
      shape: 'square',
      points: bins
    }], { xLabel: options.xLabel || 'Valor', yLabel: options.yLabel || 'Contagem', yMin: 0 });
  }

  // Paper Summary Tables (CARA Tables 2-4 & UPF Tables 1-2) with Best/Second-Best Highlighting
  function renderPaperSummaryTables() {
    const ret = DATA.retraining;
    const sumEl = document.getElementById('cara-summary-table');
    const regEl = document.getElementById('upf-regime-table');
    const aucSvg = document.getElementById('auc-bar-chart');
    if (!ret || !ret.rules) {
      if (sumEl) sumEl.innerHTML = '<p class="muted">Tabelas comparativas CARA/UPF disponíveis em execuções com análise de retreinamento.</p>';
      if (regEl) regEl.innerHTML = '';
      return;
    }

    const rows = Object.entries(ret.rules).map(([rKey, rInfo]) => {
      const curve = rInfo.curve || [];
      const accs = curve.map(c => c.average_accuracy).filter(isFiniteNum);
      const rets = curve.map(c => c.retrains).filter(isFiniteNum);
      const meanAcc = accs.length ? accs.reduce((a, b) => a + b, 0) / accs.length : null;
      const meanRet = rets.length ? rets.reduce((a, b) => a + b, 0) / rets.length : null;
      return {
        key: rKey,
        label: rInfo.label || rKey,
        isOracle: rKey.startsWith('oracle'),
        auc: rInfo.auc,
        mean_scpe: rInfo.mean_scpe,
        mean_strategy_scpe: rInfo.mean_strategy_scpe,
        mean_acc: meanAcc,
        mean_retrains: meanRet,
        c_hat_focus: curve[state.alphaIndex]?.cost ?? rInfo.c_hat,
        offline_s: rInfo.offline_seconds,
        online_s: rInfo.online_decision_seconds
      };
    });

    function rankStyles(items, field, lowerIsBetter = true) {
      const candidates = items.filter(it => !it.isOracle && isFiniteNum(it[field]));
      const sorted = [...candidates].sort((a, b) => lowerIsBetter ? a[field] - b[field] : b[field] - a[field]);
      const bestKey = sorted[0]?.key;
      const secondKey = sorted[1]?.key;
      return { bestKey, secondKey };
    }

    const rAuc = rankStyles(rows, 'auc', true);
    const rScpe = rankStyles(rows, 'mean_scpe', true);
    const rAcc = rankStyles(rows, 'mean_acc', false);
    const rFocus = rankStyles(rows, 'c_hat_focus', true);

    const cellCls = (rk, rankObj) => rk === rankObj.bestKey ? 'num best-val' : (rk === rankObj.secondKey ? 'num second-val' : 'num');

    let t1 = `<div class="table-wrapper"><table><thead><tr>
      <th>Política / Algoritmo</th>
      <th class="num">AUC Ĉ_α (↓)</th>
      <th class="num">Mean SCPE % (↓)</th>
      <th class="num">Mean Query Acc (↑)</th>
      <th class="num">Mean # Retrains</th>
      <th class="num">Ĉ_α no α Ativo (↓)</th>
      <th class="num">Offline (s)</th>
      <th class="num">Decisão Online (s)</th>
    </tr></thead><tbody>`;
    rows.forEach(r => {
      t1 += `<tr>
        <th>${r.label}${r.isOracle ? ' <span class="paper-tag">Ref</span>' : ''}</th>
        <td class="${cellCls(r.key, rAuc)}">${fmt(r.auc, 4)}</td>
        <td class="${cellCls(r.key, rScpe)}">${fmt(r.mean_scpe, 2)}%</td>
        <td class="${cellCls(r.key, rAcc)}">${fmt(r.mean_acc, 4)}</td>
        <td class="num">${fmt(r.mean_retrains, 2)}</td>
        <td class="${cellCls(r.key, rFocus)}">${fmt(r.c_hat_focus, 4)}</td>
        <td class="num">${fmt(r.offline_s, 4)}</td>
        <td class="num">${fmt(r.online_s, 5)}</td>
      </tr>`;
    });
    t1 += '</tbody></table></div>';
    if (sumEl) sumEl.innerHTML = t1;

    // AUC & SCPE Comparison Chart
    const aucPts = rows.filter(r => isFiniteNum(r.auc)).map((r, i) => ({
      x: i + 1,
      y: r.auc,
      extra: `(${r.label})`
    }));
    drawInteractiveChart(
      aucSvg,
      document.getElementById('auc-bar-tooltip'),
      [{
        key: 'auc_series',
        label: 'AUC de Ĉ_α(θ) por Política (menor = melhor)',
        color: '#0072B2',
        shape: 'diamond',
        markerRadius: 5,
        points: aucPts
      }],
      { xLabel: 'Índice da Política (1..' + aucPts.length + ')', yLabel: 'AUC de Ĉ_α(θ)', yMin: 0 }
    );

    // Table 2: Low / Focus / High Alpha Regimes (UPF Table 2)
    const tbl = ret.table || {};
    const lowMap = tbl.low || {};
    const highMap = tbl.high || {};
    let t2 = `<div class="table-wrapper"><table><thead><tr>
      <th>Política</th>
      <th class="num">Baixo α (0.1 α_max): Ĉ_α</th>
      <th class="num">Baixo α: # Retrains</th>
      <th class="num">Baixo α: Acc</th>
      <th class="num">Foco α (${fmt(ret.alphas?.[state.alphaIndex], 3)}): Ĉ_α</th>
      <th class="num">Foco α: # Retrains (Lotes)</th>
      <th class="num">Alto α (0.9 α_max): Ĉ_α</th>
      <th class="num">Alto α: # Retrains</th>
      <th class="num">Alto α: Acc</th>
    </tr></thead><tbody>`;
    rows.forEach(r => {
      const lCell = lowMap[r.key] || ret.rules[r.key]?.curve?.[1] || {};
      const fCell = ret.rules[r.key]?.curve?.[state.alphaIndex] || {};
      const hCell = highMap[r.key] || ret.rules[r.key]?.curve?.[Math.max(0, (ret.rules[r.key]?.curve?.length || 1) - 1)] || {};
      t2 += `<tr>
        <th>${r.label}</th>
        <td class="num">${fmt(lCell.cost, 4)}</td>
        <td class="num">${fmt(lCell.retrains)}</td>
        <td class="num">${fmt(lCell.average_accuracy, 4)}</td>
        <td class="num"><strong>${fmt(fCell.cost, 4)}</strong></td>
        <td class="num">${fmt(fCell.retrains)} <span class="muted">[${(fCell.retrain_points || []).join(', ')}]</span></td>
        <td class="num">${fmt(hCell.cost, 4)}</td>
        <td class="num">${fmt(hCell.retrains)}</td>
        <td class="num">${fmt(hCell.average_accuracy, 4)}</td>
      </tr>`;
    });
    t2 += '</tbody></table></div>';
    if (regEl) regEl.innerHTML = t2;
  }

  // TAB 3: Continual Learning & Historical Retention
  function renderRetentionTab() {
    const ret = DATA.retention || {};
    const checkpoints = ret.checkpoints || [];
    const matrix = ret.matrix || [];
    const periods = ret.periods || [];
    const series = [];

    // 1. Per-concept / per-period retention series (from ret.series if values array, or columns of ret.matrix)
    const rawRetSeries = Array.isArray(ret.series) ? ret.series : [];
    rawRetSeries.forEach((s, idx) => {
      if (s && Array.isArray(s.values)) {
        const pts = s.values.map((v, i) => ({ x: i + 1, y: v }));
        if (pts.some(p => isFiniteNum(p.y))) {
          series.push({
            key: `ret_concept_${s.period || idx}`,
            label: `Conceito ${s.period || idx + 1} (${ret.metric || 'accuracy'})`,
            color: PALETTE[idx % PALETTE.length],
            shape: MARKER_SHAPES[idx % MARKER_SHAPES.length],
            dash: DASH_STYLES[idx % DASH_STYLES.length],
            points: applySmoothing(pts, state.smoothStrategy, state.smoothWindow)
          });
        }
      }
    });

    // Extract checkpoint metric curves from ret.series objects ({step, absolute_retention, gem_bwt, post_learning_fm})
    if (rawRetSeries.length && typeof rawRetSeries[0] === 'object' && !Array.isArray(rawRetSeries[0].values)) {
      ['absolute_retention', 'gem_bwt', 'post_learning_fm'].forEach((mKey, mIdx) => {
        const pts = rawRetSeries
          .map((s, i) => ({ x: isFiniteNum(s.step) ? s.step : i + 1, y: s[mKey] ?? null }))
          .filter(p => isFiniteNum(p.y));
        if (pts.length) {
          const cIdx = series.length + mIdx;
          series.push({
            key: `ret_obj_${mKey}`,
            label: mKey,
            color: PALETTE[cIdx % PALETTE.length],
            shape: MARKER_SHAPES[cIdx % MARKER_SHAPES.length],
            dash: DASH_STYLES[cIdx % DASH_STYLES.length],
            points: applySmoothing(pts, state.smoothStrategy, state.smoothWindow)
          });
        }
      });
    }

    // Also extract per-period columns from ret.matrix if not already populated via s.values
    if (matrix.length && !rawRetSeries.some(s => s && Array.isArray(s.values))) {
      const numPeriods = Math.max(0, ...matrix.map(r => (r || []).length));
      for (let pIdx = 0; pIdx < Math.min(6, numPeriods); pIdx++) {
        const pts = matrix
          .map((row, cIdx) => ({ x: cIdx + 1, y: row?.[pIdx] ?? null }))
          .filter(p => isFiniteNum(p.y));
        if (pts.length > 1) {
          const cIdx = series.length;
          series.push({
            key: `ret_mat_period_${pIdx}`,
            label: `Período ${periods[pIdx] || `P${pIdx + 1}`}`,
            color: PALETTE[cIdx % PALETTE.length],
            shape: MARKER_SHAPES[cIdx % MARKER_SHAPES.length],
            dash: DASH_STYLES[cIdx % DASH_STYLES.length],
            points: applySmoothing(pts, state.smoothStrategy, state.smoothWindow)
          });
        }
      }
    }

    // 2. Tracking Continual Learning series (cl_*, retention_*)
    const tSeries = Object.assign({}, DATA.tracking_series || {}, ret.tracking_series || {});
    Object.entries(tSeries).forEach(([k, pts], idx) => {
      if (!k.startsWith('cl_') && !k.startsWith('retention_')) return;
      if (!Array.isArray(pts) || !pts.length) return;
      const cIdx = series.length + idx;
      series.push({
        key: `ts_${k}`,
        label: k,
        color: PALETTE[cIdx % PALETTE.length],
        shape: MARKER_SHAPES[cIdx % MARKER_SHAPES.length],
        dash: DASH_STYLES[cIdx % DASH_STYLES.length],
        points: applySmoothing(pts.map(p => ({ x: p.step, y: p.value })), state.smoothStrategy, state.smoothWindow)
      });
    });

    renderLegendBar(document.getElementById('retention-legend'), series, renderRetentionTab);
    drawInteractiveChart(
      document.getElementById('retention-chart'),
      document.getElementById('retention-tooltip'),
      series,
      { xLabel: 'Checkpoint / Passo de Avaliação', yLabel: 'Valor da Métrica de Retenção / CL' }
    );

    // Historical Retention Heatmap Matrix
    const matContainer = document.getElementById('retention-matrix-container');
    if (!matrix.length) {
      matContainer.innerHTML = '<p class="muted">Matriz de retenção histórica indisponível para este protocolo.</p>';
    } else {
      let html = `<div class="table-wrapper"><table><caption>Métrica: ${ret.metric || 'accuracy'} · absolute_retention: ${fmt(ret.absolute_retention, 4)} · gem_bwt: ${fmt(ret.gem_bwt, 4)} · post_learning_fm: ${fmt(ret.post_learning_fm, 4)}</caption><thead><tr><th>Checkpoint</th>`;
      periods.forEach(p => { html += `<th class="num">${p}</th>`; });
      html += '</tr></thead><tbody>';
      matrix.forEach((row, rIdx) => {
        html += `<tr><th>${checkpoints[rIdx] || `ckpt-${rIdx + 1}`}</th>`;
        (row || []).forEach(val => {
          if (!isFiniteNum(val)) {
            html += '<td class="matrix-cell">—</td>';
          } else {
            const alpha = Math.max(0.10, Math.min(0.75, val * 0.72));
            html += `<td class="matrix-cell" style="background:rgba(10,122,92,${alpha.toFixed(2)})">${fmt(val, 3)}</td>`;
          }
        });
        html += '</tr>';
      });
      html += '</tbody></table></div>';
      matContainer.innerHTML = html;
    }

    // Continual Task Matrix
    const clContainer = document.getElementById('continual-matrix-container');
    const clMat = ret.continual_matrix;
    const rowsList = Array.isArray(clMat) ? clMat : (Array.isArray(clMat?.accuracies) ? clMat.accuracies : []);
    if (!rowsList.length) {
      clContainer.innerHTML = '<p class="muted">Matriz de episódios/tarefas continual disponível nos protocolos multi-task ou baseados em episódios.</p>';
    } else if (typeof rowsList[0] === 'object' && !Array.isArray(rowsList[0]) && rowsList[0] !== null && rowsList[0].accuracies && typeof rowsList[0].accuracies === 'object') {
      const epKeys = Array.from(new Set(rowsList.flatMap(r => Object.keys(r.accuracies || {})))).sort((a, b) => Number(a) - Number(b));
      let html = '<div class="table-wrapper"><table><thead><tr><th>Task / Ep</th><th class="num">End Seq</th>';
      epKeys.forEach(ek => { html += `<th class="num">Ep ${ek}</th>`; });
      html += '<th class="num">Single Acc</th><th class="num">Joint Acc</th><th class="num">CL Avg Acc</th><th class="num">CL Forgetting</th></tr></thead><tbody>';
      rowsList.forEach((r, idx) => {
        const m = r.metrics || {};
        html += `<tr><th>Task ${r.task ?? idx}</th><td class="num">${r.end_seq ?? '—'}</td>`;
        epKeys.forEach(ek => {
          const val = r.accuracies?.[ek];
          if (!isFiniteNum(val)) {
            html += '<td class="matrix-cell">—</td>';
          } else {
            const bg = Math.max(0.08, Math.min(0.70, val * 0.68));
            html += `<td class="matrix-cell" style="background:rgba(0,114,178,${bg.toFixed(2)})">${fmt(val, 3)}</td>`;
          }
        });
        html += `<td class="num">${fmt(r.single_accuracy, 3)}</td>`;
        html += `<td class="num">${fmt(r.joint_accuracy, 3)}</td>`;
        html += `<td class="num"><strong>${fmt(m.cl_average_accuracy, 3)}</strong></td>`;
        html += `<td class="num">${fmt(m.cl_forgetting, 3)}</td></tr>`;
      });
      html += '</tbody></table></div>';
      clContainer.innerHTML = html;
    } else {
      let html = '<div class="table-wrapper"><table><thead><tr><th>Episódio / Checkpoint</th>';
      const cols = Array.isArray(rowsList[0]) ? rowsList[0].length : Object.keys(rowsList[0] || {}).length;
      for (let c = 0; c < cols; c++) html += `<th class="num">Task ${c + 1}</th>`;
      html += '</tr></thead><tbody>';
      rowsList.forEach((r, idx) => {
        const vals = Array.isArray(r) ? r : Object.values(r || {});
        html += `<tr><th>Ep ${idx + 1}</th>`;
        vals.forEach(v => {
          html += `<td class="matrix-cell">${isFiniteNum(v) ? fmt(v, 3) : '—'}</td>`;
        });
        html += '</tr>';
      });
      html += '</tbody></table></div>';
      clContainer.innerHTML = html;
    }
  }

  // TAB 4: Classification, ROC, PR & Confusion Matrix
  function renderClassificationTab() {
    const rcScopes = Object.keys(DATA.ranking_curves || {});
    const activeRc = DATA.ranking_curves?.[state.confusionScope] || DATA.ranking_curves?.[rcScopes[0]] || {};

    // ROC Curve
    const roc = activeRc.roc;
    if (roc && Array.isArray(roc.fpr) && roc.fpr.length) {
      const pts = roc.fpr.map((x, i) => ({ x, y: roc.tpr[i] }));
      drawInteractiveChart(
        document.getElementById('roc-chart'),
        document.getElementById('roc-tooltip'),
        [{
          key: 'roc_curve',
          label: `ROC (AUC=${fmt(roc.roc_auc, 4)}, Recall@1%FPR=${fmt(roc.recall_at_fpr_0_01, 4)})`,
          color: '#0072B2',
          shape: 'circle',
          points: pts
        }],
        { xLabel: 'False Positive Rate (FPR)', yLabel: 'True Positive Rate (TPR / Recall)', xMin: 0, xMax: 1, yMin: 0, yMax: 1, diagonalLine: true }
      );
    } else {
      drawInteractiveChart(document.getElementById('roc-chart'), null, [], { emptyMessage: 'Curva ROC indisponível para este escopo.' });
    }

    // PR Curve
    const pr = activeRc.pr;
    if (pr && Array.isArray(pr.recall) && pr.recall.length) {
      const pts = pr.recall.map((x, i) => ({ x, y: pr.precision[i] }));
      drawInteractiveChart(
        document.getElementById('pr-chart'),
        document.getElementById('pr-tooltip'),
        [{
          key: 'pr_curve',
          label: `PR (AP=${fmt(pr.average_precision, 4)}, Classe+=${pr.positive_class || '1'})`,
          color: '#D55E00',
          shape: 'square',
          points: pts
        }],
        { xLabel: 'Recall (Sensibilidade)', yLabel: 'Precision (Valor Preditivo Positivo)', xMin: 0, xMax: 1, yMin: 0, yMax: 1 }
      );
    } else {
      drawInteractiveChart(document.getElementById('pr-chart'), null, [], { emptyMessage: 'Curva Precision-Recall indisponível para este escopo.' });
    }

    // Confusion Matrix
    const confSel = document.getElementById('confusion-scope-selector');
    const confScopes = Object.keys(DATA.confusion || {});
    if (confScopes.length && confSel.options.length !== confScopes.length) {
      confSel.innerHTML = confScopes.map(s => `<option value="${s}">${s}</option>`).join('');
      if (confScopes.includes(state.confusionScope)) confSel.value = state.confusionScope;
      else state.confusionScope = confScopes[0];
    }
    const confObj = DATA.confusion?.[state.confusionScope] || DATA.confusion?.[confScopes[0]];
    const confEl = document.getElementById('confusion-matrix-container');
    if (!confObj || !Array.isArray(confObj.matrix) || !confObj.matrix.length) {
      confEl.innerHTML = '<p class="muted">Matriz de confusão indisponível.</p>';
    } else {
      const classes = confObj.classes?.length ? confObj.classes : confObj.matrix.map((_, i) => `Classe ${i}`);
      let html = '<div class="table-wrapper"><table><thead><tr><th>Real \\ Predito</th>';
      classes.forEach(c => { html += `<th class="num">${c}</th>`; });
      html += '<th class="num">Suporte Real</th><th class="num">Recall Classe</th></tr></thead><tbody>';
      confObj.matrix.forEach((row, rIdx) => {
        const rowSum = (row || []).reduce((acc, v) => acc + (isFiniteNum(v) ? v : 0), 0);
        const diag = row?.[rIdx] ?? 0;
        const rec = rowSum > 0 ? diag / rowSum : null;
        html += `<tr><th>${classes[rIdx] || rIdx}</th>`;
        (row || []).forEach((v, cIdx) => {
          const pct = rowSum > 0 && isFiniteNum(v) ? (100 * v / rowSum) : 0;
          const bg = rIdx === cIdx
            ? `rgba(10, 122, 92, ${(0.08 + 0.55 * (pct / 100)).toFixed(2)})`
            : `rgba(190, 18, 60, ${(0.04 + 0.45 * (pct / 100)).toFixed(2)})`;
          html += `<td class="matrix-cell" style="background:${bg}">${fmt(v, 0)} <span class="muted">(${fmt(pct, 1)}%)</span></td>`;
        });
        html += `<td class="num">${fmt(rowSum, 0)}</td><td class="num"><strong>${fmt(rec, 4)}</strong></td></tr>`;
      });
      html += '</tbody></table></div>';
      confEl.innerHTML = html;

      // Per-Class Support & Recall Chart
      const recPts = confObj.matrix.map((row, rIdx) => {
        const rowSum = (row || []).reduce((acc, v) => acc + (isFiniteNum(v) ? v : 0), 0);
        const rec = confObj.recall?.[rIdx] ?? (rowSum > 0 ? (row[rIdx] || 0) / rowSum : 0);
        return { x: rIdx, y: rec, extra: `(Classe ${classes[rIdx] || rIdx}, n=${rowSum})` };
      });
      drawInteractiveChart(
        document.getElementById('class-recall-chart'),
        document.getElementById('class-recall-tooltip'),
        [{
          key: 'class_recall',
          label: 'Recall por Classe',
          color: '#009E73',
          shape: 'triangle-up',
          markerRadius: 5,
          points: recPts
        }],
        { xLabel: 'Índice da Classe', yLabel: 'Recall (Sensibilidade)', yMin: 0, yMax: 1.05 }
      );
    }
  }

  // TAB 5: Drift, Training Effort, Costs & System Telemetry
  function renderDriftAndSystemTab() {
    // 1. Drift timeline & recovery table
    const drift = DATA.drift || {};
    const rec = drift.recovery || [];
    const delayPts = rec.filter(r => isFiniteNum(r.change_seq) && isFiniteNum(r.delay)).map(r => ({
      x: r.change_seq,
      y: r.delay,
      extra: `(Alarme em ${r.alarm_seq ?? '—'})`
    }));
    const vLines = (drift.alarms || []).map(a => ({ x: a.seq, color: '#be123c', dash: '3,3', label: 'Alarme' }));
    drawInteractiveChart(
      document.getElementById('drift-timeline-chart'),
      document.getElementById('drift-timeline-tooltip'),
      delayPts.length
        ? [{ key: 'drift_delays', label: 'Atraso de Detecção (passos)', color: '#be123c', shape: 'diamond', points: delayPts }]
        : (DATA.windows || []).slice(0, 1).map(() => ({
            key: 'alarm_markers',
            label: 'Fluxo Monitorado',
            color: '#0072B2',
            shape: 'circle',
            points: (DATA.windows || []).map((w, i) => ({ x: w.step ?? i + 1, y: w.metrics?.accuracy ?? 0 }))
          })),
      { xLabel: 'Sequência / Passo', yLabel: delayPts.length ? 'Atraso de Detecção (Delay)' : 'Acurácia + Alarmes', vLines }
    );

    const driftTbl = document.getElementById('drift-summary-table');
    if (driftTbl) {
      driftTbl.innerHTML = `<div class="table-wrapper"><table><thead><tr>
        <th>Alarmes</th><th>Changepoints</th><th>Precision</th><th>Recall</th><th>F1</th><th>Censurados</th>
      </tr></thead><tbody><tr>
        <td>${(drift.alarms || []).length}</td>
        <td>${(drift.changepoints || []).length}</td>
        <td>${fmt(drift.precision, 4)}</td>
        <td>${fmt(drift.recall, 4)}</td>
        <td>${fmt(drift.f1, 4)}</td>
        <td>${fmt(drift.censored_count, 0)}</td>
      </tr></tbody></table></div>`;
    }

    // 2. Per-fit Training Effort & Internal Learning Curve
    const fits = DATA.fits || [];
    const fitSel = document.getElementById('fit-selector');
    if (fits.length && fitSel.options.length !== fits.length) {
      fitSel.innerHTML = fits.map((f, i) => `<option value="${i}">${f.fit_id || `fit-${i + 1}`} (${f.action || 'fit'})</option>`).join('');
      fitSel.value = String(state.selectedFitIdx);
    }
    const effortSeries = [];
    if (fits.length) {
      effortSeries.push({
        key: 'fit_samples',
        label: 'Amostras / Apresentações de Treino',
        color: '#0072B2',
        shape: 'square',
        points: fits.map((f, i) => ({ x: i + 1, y: f.training_presentations ?? f.train_samples ?? 0, extra: `(${f.fit_id || i + 1})` }))
      });
    }
    drawInteractiveChart(
      document.getElementById('effort-fits-chart'),
      document.getElementById('effort-fits-tooltip'),
      effortSeries,
      { xLabel: 'Índice do Fit (1..N)', yLabel: 'Amostras / Apresentações', yMin: 0 }
    );

    const activeFit = fits[state.selectedFitIdx] || fits[0];
    const stepPts = activeFit?.steps || [];
    const stepKeys = Array.from(new Set(stepPts.flatMap(s => Object.keys(s.metrics || {})))).slice(0, 4);
    const internalSeries = stepKeys.map((k, idx) => ({
      key: `fit_step_${k}`,
      label: k,
      color: PALETTE[idx % PALETTE.length],
      shape: MARKER_SHAPES[idx % MARKER_SHAPES.length],
      points: stepPts.map((s, i) => ({ x: s.step ?? s.global_step ?? i + 1, y: s.metrics?.[k] ?? null }))
    }));
    drawInteractiveChart(
      document.getElementById('fit-internal-steps-chart'),
      document.getElementById('fit-internal-steps-tooltip'),
      internalSeries,
      { xLabel: `Passo Interno (${activeFit?.fit_id || 'fit'})`, yLabel: 'Métrica Interna do Fit', emptyMessage: 'Sem passos internos (observation_steps) registrados para este ajuste.' }
    );

    // 3. System Telemetry (RSS MB & CPU %)
    const sysSamples = DATA.system_metrics?.samples || [];
    const sysKeys = Array.from(new Set(sysSamples.flatMap(s => Object.keys(s.values || {})))).slice(0, 4);
    const sysSeries = sysKeys.map((k, idx) => ({
      key: `sys_${k}`,
      label: k,
      color: PALETTE[idx % PALETTE.length],
      shape: MARKER_SHAPES[idx % MARKER_SHAPES.length],
      points: sysSamples.map((s, i) => ({ x: s.t ?? i, y: s.values?.[k] ?? null }))
    }));
    drawInteractiveChart(
      document.getElementById('system-telemetry-chart'),
      document.getElementById('system-telemetry-tooltip'),
      sysSeries,
      { xLabel: 'Tempo Decorrido (s)', yLabel: 'Uso de Recursos (RSS / CPU)', emptyMessage: 'Amostras temporais de telemetria não registradas (veja sumário de custos ao lado).' }
    );

    // 4. Cost Breakdown & Fits/Updates tables
    const costsEl = document.getElementById('costs-breakdown-container');
    if (costsEl) {
      const cMetrics = Object.entries(DATA.costs?.metrics || {});
      const eMetrics = Object.entries(DATA.effort || {});
      const rMetrics = Object.entries(DATA.resources || {});
      const combined = [
        ...cMetrics.map(([k, v]) => ['Custo', k, v]),
        ...eMetrics.map(([k, v]) => ['Esforço', k, v]),
        ...rMetrics.map(([k, v]) => ['Recursos', k, v])
      ];
      if (!combined.length) {
        costsEl.innerHTML = '<p class="muted">Sem métricas detalhadas de custo.</p>';
      } else {
        let html = '<div class="table-wrapper"><table><thead><tr><th>Categoria</th><th>Métrica</th><th class="num">Valor</th></tr></thead><tbody>';
        combined.slice(0, 18).forEach(([cat, k, v]) => {
          html += `<tr><td>${cat}</td><th>${k}</th><td class="num">${fmt(v, 4)}</td></tr>`;
        });
        html += '</tbody></table></div>';
        costsEl.innerHTML = html;
      }
    }

    const fitsEl = document.getElementById('fits-table-container');
    if (fitsEl) {
      if (!fits.length) {
        fitsEl.innerHTML = '<p class="muted">Nenhum ajuste registrado.</p>';
      } else {
        let html = '<div class="table-wrapper"><table><thead><tr><th>Fit</th><th>Ação</th><th>Status</th><th>Seq</th><th class="num">Amostras</th><th class="num">Tempo (s)</th></tr></thead><tbody>';
        fits.forEach(f => {
          html += `<tr><th>${f.fit_id || '—'}</th><td>${f.action || '—'}</td><td>${f.promotion_status || '—'}</td><td>${f.start_seq ?? '—'}..${f.end_seq ?? '—'}</td><td class="num">${fmt(f.train_samples, 0)}</td><td class="num">${fmt(f.train_seconds, 4)}</td></tr>`;
        });
        html += '</tbody></table></div>';
        fitsEl.innerHTML = html;
      }
    }

    const updEl = document.getElementById('updates-table-container');
    if (updEl) {
      const upds = DATA.updates || [];
      if (!upds.length) {
        updEl.innerHTML = '<p class="muted">Nenhum evento pós-warmup registrado.</p>';
      } else {
        let html = '<div class="table-wrapper"><table><thead><tr><th>Seq / Lote</th><th>Ação</th><th>Motivo</th><th class="num">Amostras</th><th class="num">Tempo (s)</th></tr></thead><tbody>';
        upds.forEach(u => {
          html += `<tr><th>${u.seq ?? u.batch ?? '—'}</th><td>${u.action || 'retrain'}</td><td>${u.reason || '—'}</td><td class="num">${fmt(u.train_samples, 0)}</td><td class="num">${fmt(u.seconds, 4)}</td></tr>`;
        });
        html += '</tbody></table></div>';
        updEl.innerHTML = html;
      }
    }
  }

  // TAB 6: Complete Metrics Catalog
  function renderMetricsCatalog() {
    const groupSel = document.getElementById('metric-group');
    const groups = Array.from(new Set((DATA.all_metrics || []).map(m => m.group))).sort();
    if (groupSel.options.length <= 1 && groups.length) {
      groupSel.innerHTML = '<option value="all">Todos os grupos</option>' + groups.map(g => `<option value="${g}">${g}</option>`).join('');
    }
    const q = (document.getElementById('metric-search').value || '').toLowerCase().trim();
    const grp = groupSel.value;
    const st = document.getElementById('metric-status').value;

    const rows = (DATA.all_metrics || []).filter(m => {
      if (grp !== 'all' && m.group !== grp) return false;
      if (st !== 'all' && m.status !== st) return false;
      if (!q) return true;
      return `${m.group} ${m.name} ${m.reason || ''}`.toLowerCase().includes(q);
    });

    const el = document.getElementById('all-metrics-container');
    let html = `<div class="table-wrapper"><table><caption>Exibindo ${rows.length} de ${(DATA.all_metrics || []).length} métrica(s)</caption><thead><tr><th>Métrica</th><th>Grupo</th><th class="num">Valor</th><th>Status</th><th>Observação / Escopo</th></tr></thead><tbody>`;
    rows.forEach(m => {
      const cls = m.status === 'available' ? 'badge-ok' : 'badge-na';
      html += `<tr><th>${m.name}</th><td>${m.group}</td><td class="num">${fmt(m.value, 5)}</td><td class="${cls}">${m.status}</td><td>${m.reason || m.unit_or_scope || '—'}</td></tr>`;
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
  }

  // Wire UI Controls & Event Listeners
  function initControls() {
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
        const target = document.getElementById(btn.dataset.tab);
        if (target) target.classList.add('active');
        renderAll();
      });
    });

    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme');
      document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
      renderAll();
    });

    // Expand/collapse all formula & interpretability guides
    let guidesExpanded = false;
    document.getElementById('expand-all-guides').addEventListener('click', () => {
      guidesExpanded = !guidesExpanded;
      document.querySelectorAll('details.guide-box').forEach(d => { d.open = guidesExpanded; });
    });

    // Global vs Live Mode & Timeline Slider
    const modeGlobal = document.getElementById('mode-global');
    const modeLive = document.getElementById('mode-live');
    const slider = document.getElementById('timeline-slider');
    const stepLbl = document.getElementById('timeline-step-label');
    const playBtn = document.getElementById('timeline-play');

    function updateTimelineUI() {
      modeGlobal.classList.toggle('active', state.mode === 'global');
      modeLive.classList.toggle('active', state.mode === 'live');
      slider.value = String(state.sliderPct);
      stepLbl.textContent = `${state.sliderPct}%`;
      renderAll();
    }

    modeGlobal.addEventListener('click', () => {
      state.mode = 'global';
      state.sliderPct = 100;
      stopPlay();
      updateTimelineUI();
    });
    modeLive.addEventListener('click', () => {
      state.mode = 'live';
      if (state.sliderPct >= 100) state.sliderPct = 20;
      updateTimelineUI();
    });
    slider.addEventListener('input', () => {
      state.mode = 'live';
      state.sliderPct = Number(slider.value);
      updateTimelineUI();
    });
    document.getElementById('timeline-prev').addEventListener('click', () => {
      state.mode = 'live';
      state.sliderPct = Math.max(1, state.sliderPct - 5);
      updateTimelineUI();
    });
    document.getElementById('timeline-next').addEventListener('click', () => {
      state.mode = 'live';
      state.sliderPct = Math.min(100, state.sliderPct + 5);
      updateTimelineUI();
    });

    function stopPlay() {
      state.playing = false;
      if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
      playBtn.textContent = '▶ Play';
    }
    playBtn.addEventListener('click', () => {
      if (state.playing) {
        stopPlay();
        return;
      }
      state.mode = 'live';
      if (state.sliderPct >= 100) state.sliderPct = 5;
      state.playing = true;
      playBtn.textContent = '⏸ Pause';
      state.playTimer = setInterval(() => {
        if (state.sliderPct >= 100) {
          stopPlay();
          return;
        }
        state.sliderPct = Math.min(100, state.sliderPct + 5);
        updateTimelineUI();
      }, 380);
    });

    // Smoothing Toolbar
    const smoothStrat = document.getElementById('smooth-strategy');
    const smoothWin = document.getElementById('smooth-window');
    const smoothWinVal = document.getElementById('smooth-window-val');
    smoothStrat.addEventListener('change', () => {
      state.smoothStrategy = smoothStrat.value;
      renderAll();
    });
    smoothWin.addEventListener('input', () => {
      state.smoothWindow = Number(smoothWin.value);
      smoothWinVal.textContent = String(state.smoothWindow);
      renderAll();
    });

    // Policy & Alpha Operating Point Controls
    const polSel = document.getElementById('policy-selector');
    const alphaSlider = document.getElementById('alpha-slider');
    const alphaLbl = document.getElementById('alpha-value-label');
    if (DATA.retraining && DATA.retraining.rules) {
      const rules = Object.entries(DATA.retraining.rules);
      polSel.innerHTML = rules.map(([k, r]) => `<option value="${k}">${r.label || k}</option>`).join('');
      polSel.value = state.selectedPolicy;
      const alphas = DATA.retraining.alphas || [];
      alphaSlider.max = String(Math.max(0, alphas.length - 1));
      alphaSlider.value = String(state.alphaIndex);
      alphaLbl.textContent = fmt(alphas[state.alphaIndex], 4);

      polSel.addEventListener('change', () => {
        state.selectedPolicy = polSel.value;
        renderAll();
      });
      alphaSlider.addEventListener('input', () => {
        state.alphaIndex = Number(alphaSlider.value);
        alphaLbl.textContent = fmt(alphas[state.alphaIndex], 4);
        renderAll();
      });
    } else {
      document.getElementById('alpha-toolbar-group').style.display = 'none';
    }

    // Sub-chart selectors
    document.getElementById('cost-type-selector').addEventListener('change', e => {
      state.costMetricType = e.target.value;
      renderAlphaPanels();
    });
    document.getElementById('upf-fig1-metric').addEventListener('change', e => {
      state.upfFig1Metric = e.target.value;
      renderUpfFig1Chart(document.getElementById('upf-fig1-chart'), document.getElementById('upf-fig1-tooltip'), document.getElementById('upf-fig1-legend'), false);
    });
    document.getElementById('wrong-alpha-rule').addEventListener('change', e => {
      state.wrongAlphaRule = e.target.value;
      renderSecondaryPaperCharts();
    });
    document.getElementById('scope-selector').addEventListener('change', e => {
      state.selectedScope = e.target.value;
      renderOverviewTab();
    });
    document.getElementById('confusion-scope-selector').addEventListener('change', e => {
      state.confusionScope = e.target.value;
      renderClassificationTab();
    });
    document.getElementById('fit-selector').addEventListener('change', e => {
      state.selectedFitIdx = Number(e.target.value);
      renderDriftAndSystemTab();
    });

    // Catalog filters
    document.getElementById('metric-search').addEventListener('input', renderMetricsCatalog);
    document.getElementById('metric-group').addEventListener('change', renderMetricsCatalog);
    document.getElementById('metric-status').addEventListener('change', renderMetricsCatalog);

    window.addEventListener('resize', () => renderAll());
  }

  function renderAll() {
    renderHeaderAndInsights();
    renderOverviewTab();
    renderUpfFig1Chart(
      document.getElementById('upf-fig1-chart'),
      document.getElementById('upf-fig1-tooltip'),
      document.getElementById('upf-fig1-legend'),
      false
    );
    renderAlphaPanels();
    renderMatricesAndTrajectories();
    renderSecondaryPaperCharts();
    renderPaperSummaryTables();
    renderRetentionTab();
    renderClassificationTab();
    renderDriftAndSystemTab();
    renderMetricsCatalog();
  }

  initControls();
  renderAll();
})();
</script>
</body>
</html>
"""
