"""Detailed matplotlib figures for MLflow, logged with ``mlflow.log_figure()``.

These complement the dependency-free SVG artifacts in ``evaluation/artifacts/figures.py``: richer multi-panel
dashboards (drift markers, rolling bands, aggregated confusion matrix, FPR/FNR trade-off,
calibration quality, promotion gate, continual-learning matrix, cost breakdown, system
resources) rendered as PNG so the MLflow UI shows them in the run's artifact viewer.

Builders only read reports; they never change metrics, digests or decisions. matplotlib is
optional (``plots`` extra): without it every entry point returns ``[]``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager

MARKER_STYLE = {
    "alarm": ("#d62728", "--", "Alarme do detector"),
    "changepoint": ("#7f7f7f", ":", "Mudança de conceito"),
    "update": ("#2ca02c", "-.", "Adaptação promovida"),
    "rejected": ("#ff7f0e", "-.", "Adaptação rejeitada"),
}
FIGURE_DPI = 130
ROLLING_WINDOW = 5


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _xy(rows, key, x="end_seq"):
    pairs = [(r.get(x), _num(r.get(key))) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _rolling_mean(values, size=ROLLING_WINDOW):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - size + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _events(report) -> list[dict]:
    markers = [{"x": a["seq"], "kind": "alarm"} for a in report.get("alarms", [])]
    dataset = report.get("dataset")  # a dict in binary reports, a plain name in ZIA reports
    for point in (dataset.get("changepoints") if isinstance(dataset, dict) else None) or []:
        markers.append({"x": point, "kind": "changepoint"})
    for candidate in report.get("candidates", report.get("updates", [])):
        markers.append(
            {"x": candidate["seq"], "kind": "update" if candidate.get("promoted", True) else "rejected"}
        )
    return markers


def _draw_events(ax, markers, legend_seen: set) -> None:
    for marker in markers:
        color, style, label = MARKER_STYLE[marker["kind"]]
        ax.axvline(
            marker["x"],
            color=color,
            linestyle=style,
            linewidth=0.9,
            alpha=0.55,
            label=None if marker["kind"] in legend_seen else label,
        )
        legend_seen.add(marker["kind"])


def _plot_series(ax, windows, keys, *, markers=(), smooth=True, ylim=None, title="", ylabel="valor"):
    seen: set = set()
    for key in keys:
        xs, ys = _xy(windows, key)
        if not xs:
            continue
        (line,) = ax.plot(xs, ys, linewidth=1.2, alpha=0.45 if smooth else 1.0, label=key)
        if smooth and len(ys) > ROLLING_WINDOW:
            ax.plot(
                xs, _rolling_mean(ys), color=line.get_color(), linewidth=2.2, label=f"{key} (média móvel)"
            )
    _draw_events(ax, markers, seen)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("seq (fim da janela)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")


# ---- figure builders ----------------------------------------------------------------------------------
# Each builder returns (artifact_path, figure) pairs and skips silently when its data is missing.


def _performance_dashboard(plt, report):
    windows = report.get("windows") or []
    if not windows:
        return []
    markers = _events(report)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    _plot_series(
        axes[0][0],
        windows,
        ["accuracy", "balanced_accuracy"],
        markers=markers,
        ylim=(0, 1.02),
        title="Acurácia",
    )
    _plot_series(axes[0][1], windows, ["f1", "f1_macro"], markers=markers, ylim=(0, 1.02), title="F1")
    _plot_series(
        axes[1][0],
        windows,
        ["precision", "recall", "specificity"],
        markers=markers,
        ylim=(0, 1.02),
        title="Precisão, recall e especificidade",
    )
    _plot_series(
        axes[1][1],
        windows,
        ["mcc", "cohen_kappa"],
        markers=markers,
        ylim=(-0.05, 1.02),
        title="Concordância (MCC e kappa)",
    )
    fig.suptitle("Desempenho prequential ao longo do fluxo", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return [("plots/performance/dashboard.png", fig)]


def _confusion(plt, report):
    windows = report.get("windows") or []
    totals = {k: sum(w.get(k) or 0 for w in windows) for k in ("tp", "fp", "tn", "fn")}
    if not any(totals.values()):
        return []
    import numpy as np

    counts = np.array([[totals["tn"], totals["fp"]], [totals["fn"], totals["tp"]]], dtype=float)
    rows = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, title, fmt in (
        (axes[0], counts, "Matriz de confusão (contagens)", "{:,.0f}"),
        (axes[1], normalized, "Normalizada por classe real (recall)", "{:.1%}"),
    ):
        image = ax.imshow(data, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax.set_xticks([0, 1], ["negativo", "positivo"])
        ax.set_yticks([0, 1], ["negativo", "positivo"])
        ax.set_xlabel("predito")
        ax.set_ylabel("real")
        ax.set_title(title, fontsize=10, fontweight="bold")
        for (i, j), value in np.ndenumerate(data):
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                color="white" if value > data.max() / 2 else "black",
                fontsize=11,
            )
    fig.tight_layout()
    out = [("plots/performance/confusion_matrix.png", fig)]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    xs = [w["end_seq"] for w in windows]
    bottom = np.zeros(len(windows))
    for key, color in (("tp", "#2ca02c"), ("tn", "#1f77b4"), ("fp", "#ff7f0e"), ("fn", "#d62728")):
        values = np.array([w.get(key) or 0 for w in windows], dtype=float)
        ax.bar(range(len(xs)), values, bottom=bottom, label=key.upper(), color=color, width=0.85)
        bottom += values
    ax.set_xticks(range(len(xs))[:: max(1, len(xs) // 12)], xs[:: max(1, len(xs) // 12)])
    ax.set_xlabel("seq (fim da janela)")
    ax.set_ylabel("exemplos")
    ax.set_title("Composição da matriz de confusão por janela", fontsize=10, fontweight="bold")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    out.append(("plots/performance/confusion_timeline.png", fig))
    return out


def _error_tradeoff(plt, report):
    windows = report.get("windows") or []
    points = [
        (w["end_seq"], _num(w.get("false_positive_rate")), _num(w.get("false_negative_rate")))
        for w in windows
    ]
    points = [p for p in points if p[1] is not None and p[2] is not None]
    if len(points) < 2:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter = ax.scatter(
        [p[1] for p in points],
        [p[2] for p in points],
        c=[p[0] for p in points],
        cmap="viridis",
        s=45,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.plot([p[1] for p in points], [p[2] for p in points], color="gray", alpha=0.35, linewidth=0.8)
    fig.colorbar(scatter, ax=ax, label="seq (fim da janela)")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de falsos negativos")
    ax.set_title("Trade-off FPR × FNR por janela", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return [("plots/performance/error_tradeoff.png", fig)]


def _probabilistic(plt, report):
    windows = report.get("windows") or []
    if not any(_num(w.get("brier")) is not None for w in windows):
        return []
    markers = _events(report)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    _plot_series(axes[0][0], windows, ["brier"], markers=markers, title="Brier score (menor é melhor)")
    _plot_series(axes[0][1], windows, ["log_loss"], markers=markers, title="Log loss (menor é melhor)")
    _plot_series(axes[1][0], windows, ["ece", "mce"], markers=markers, title="Erro de calibração (ECE e MCE)")
    _plot_series(
        axes[1][1],
        windows,
        ["roc_auc", "average_precision"],
        markers=markers,
        ylim=(0, 1.02),
        title="Ranqueamento (ROC AUC e AP)",
    )
    fig.suptitle("Qualidade probabilística e calibração", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return [("plots/probabilistic/dashboard.png", fig)]


def _drift(plt, report):
    windows = report.get("windows") or []
    markers = _events(report)
    if not windows or not markers:
        return []
    fig, (top, lane) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    _plot_series(
        top,
        windows,
        ["accuracy", "error_rate"],
        markers=markers,
        ylim=(0, 1.02),
        title="Acurácia e erro com eventos de drift",
    )
    for row, kind in enumerate(MARKER_STYLE):
        xs = [m["x"] for m in markers if m["kind"] == kind]
        if xs:
            lane.scatter(xs, [row] * len(xs), color=MARKER_STYLE[kind][0], marker="|", s=240)
    lane.set_yticks(range(len(MARKER_STYLE)), [v[2] for v in MARKER_STYLE.values()], fontsize=8)
    lane.set_xlabel("seq")
    lane.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    out = [("plots/drift/timeline.png", fig)]
    recovery = (report.get("tracking") or {}).get("recovery") or []
    delays = [(r["change_seq"], _num(r.get("delay"))) for r in recovery]
    delays = [d for d in delays if d[1] is not None]
    if delays:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar([f"@{c}" for c, _ in delays], [d for _, d in delays], color="#9467bd")
        mean = sum(d for _, d in delays) / len(delays)
        ax.axhline(mean, color="black", linestyle="--", linewidth=1, label=f"média = {mean:,.0f}")
        ax.set_ylabel("amostras até recuperar")
        ax.set_xlabel("mudança de conceito (seq)")
        ax.set_title("Atraso de recuperação por mudança", fontsize=10, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        out.append(("plots/drift/recovery_delay.png", fig))
    return out


def _adaptation(plt, report):
    candidates = report.get("candidates", report.get("updates", []))
    if not candidates:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    total, xs, cumulative = 0, [], []
    for c in candidates:
        total += c.get("train_samples", 0)
        xs.append(c["seq"])
        cumulative.append(total)
    axes[0].step(xs, cumulative, where="post", color="#1f77b4", linewidth=2)
    promoted = [c for c in candidates if c.get("promoted", True)]
    rejected = [c for c in candidates if not c.get("promoted", True)]
    for group, color, label in ((promoted, "#2ca02c", "promovida"), (rejected, "#d62728", "rejeitada")):
        if group:
            axes[0].scatter(
                [c["seq"] for c in group],
                [cumulative[candidates.index(c)] for c in group],
                color=color,
                s=40,
                zorder=3,
                label=label,
            )
    budget = (report.get("training") or {}).get("budget")
    if _num(budget):
        axes[0].axhline(budget, color="black", linestyle="--", linewidth=1, label=f"orçamento = {budget:,}")
    axes[0].set_title("Apresentações de treino acumuladas", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("seq")
    axes[0].set_ylabel("apresentações")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    gated = [c for c in candidates if "candidate_accuracy" in c and "incumbent_accuracy" in c]
    if gated:
        cand = [c["candidate_accuracy"] for c in gated]
        inc = [c["incumbent_accuracy"] for c in gated]
        colors = ["#2ca02c" if c.get("promoted", True) else "#d62728" for c in gated]
        axes[1].scatter(inc, cand, c=colors, s=55, edgecolor="black", linewidth=0.4)
        low = min(cand + inc) - 0.02
        axes[1].plot([low, 1], [low, 1], color="gray", linestyle="--", linewidth=1, label="empate")
        axes[1].set_xlabel("acurácia do incumbente (holdout)")
        axes[1].set_ylabel("acurácia do candidato (holdout)")
        axes[1].set_title("Portão de promoção: acima da diagonal vence", fontsize=10, fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Sem holdout de promoção", ha="center", va="center")
    fig.tight_layout()
    return [("plots/adaptation/updates.png", fig)]


def _continual(plt, report):
    tracking = report.get("tracking") or {}
    matrix = (tracking.get("continual") or {}).get("matrix") or []
    if not matrix:
        return []
    import numpy as np

    episodes = sorted({int(e) for row in matrix for e in row.get("accuracies", {})})
    data = np.array(
        [
            [
                _num(row["accuracies"].get(str(e)))
                if _num(row["accuracies"].get(str(e))) is not None
                else np.nan
                for e in episodes
            ]
            for row in matrix
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(episodes), 1.0 + 0.7 * len(matrix)))
    image = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="acurácia")
    ax.set_xticks(range(len(episodes)), [f"e{e}" for e in episodes])
    ax.set_yticks(range(len(matrix)), [f"após t{row['task']}" for row in matrix])
    for (i, j), value in np.ndenumerate(data):
        if not np.isnan(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Matriz de acurácia do aprendizado contínuo", fontsize=10, fontweight="bold")
    ax.set_xlabel("episódio avaliado")
    fig.tight_layout()
    out = [("plots/continual/accuracy_matrix.png", fig)]
    series = tracking.get("series", {})
    names = [n for n in sorted(series) if n.startswith("cl_")]
    if names:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for name in names:
            ax.plot(
                [p["step"] for p in series[name]], [p["value"] for p in series[name]], marker="o", label=name
            )
        ax.set_xlabel("tarefa")
        ax.set_title("Métricas de aprendizado contínuo", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out.append(("plots/continual/cl_metrics.png", fig))
    return out


def _costs(plt, report):
    costs = (report.get("tracking") or {}).get("costs") or {}
    metrics = {k: v for k, v in (costs.get("metrics") or {}).items() if _num(v) is not None}
    sensitivity = (report.get("tracking") or {}).get("cost_sensitivity") or []
    if not metrics and not sensitivity:
        return []
    fig, axes = plt.subplots(1, 2 if sensitivity and metrics else 1, figsize=(13, 4.8), squeeze=False)
    axes = axes[0]
    slot = 0
    if metrics:
        names = sorted(metrics, key=lambda k: metrics[k])
        axes[slot].barh(names, [metrics[n] for n in names], color="#17becf")
        axes[slot].set_xlabel((costs.get("settings") or {}).get("unit", "custo"))
        axes[slot].set_title("Componentes do custo de cenário", fontsize=10, fontweight="bold")
        axes[slot].grid(alpha=0.25, axis="x")
        slot += 1
    if sensitivity:
        axes[slot].plot(
            [p["alpha"] for p in sensitivity], [p["value"] for p in sensitivity], marker="o", color="#e377c2"
        )
        axes[slot].set_xlabel("alpha")
        axes[slot].set_ylabel("custo")
        axes[slot].set_title("Sensibilidade do custo a alpha", fontsize=10, fontweight="bold")
        axes[slot].grid(alpha=0.25)
    fig.tight_layout()
    return [("plots/costs/breakdown.png", fig)]


def _summary(plt, report):
    metrics = report.get("metrics") or {}
    quality = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "precision",
        "recall",
        "specificity",
        "mcc",
        "cohen_kappa",
        "roc_auc",
        "average_precision",
    ]
    values = {k: _num(metrics.get(k)) for k in quality}
    values = {k: v for k, v in values.items() if v is not None}
    holdout = report.get("holdout") or {}
    if not values:
        return []
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(values) + 1.5))
    names = list(values)
    height = 0.38 if holdout else 0.7
    positions = range(len(names))
    ax.barh(
        [p + (height / 2 if holdout else 0) for p in positions],
        [values[n] for n in names],
        height=height,
        label="prequential",
        color="#1f77b4",
    )
    if holdout:
        held = [_num(holdout.get(n)) or 0 for n in names]
        ax.barh([p - height / 2 for p in positions], held, height=height, label="holdout", color="#ff7f0e")
    ax.set_yticks(list(positions), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title("Métricas finais de qualidade", fontsize=10, fontweight="bold")
    for p, n in zip(positions, names):
        ax.text(
            values[n] + 0.01, p + (height / 2 if holdout else 0), f"{values[n]:.3f}", va="center", fontsize=8
        )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    return [("plots/summary/final_metrics.png", fig)]


def _epochs(plt, report):
    epochs = report.get("epochs") or []
    if not epochs or "metrics" not in epochs[0]:
        return []
    names = sorted({n for e in epochs for n in e["metrics"] if _num(e["metrics"][n]) is not None})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for name in names:
        axes[0].plot(
            [e["epoch"] for e in epochs],
            [e["metrics"].get(name) for e in epochs],
            marker="o",
            markersize=3,
            label=name,
        )
    axes[0].set_title("Métricas no holdout por época", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("época")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.25)
    for key in ("training_seconds", "prediction_seconds"):
        if all(key in e for e in epochs):
            axes[1].plot([e["epoch"] for e in epochs], [e[key] for e in epochs], marker="o", label=key)
    axes[1].set_title("Tempo de treino e predição por época", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("época")
    axes[1].set_ylabel("s")
    axes[1].grid(alpha=0.25)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    return [("plots/performance/epochs.png", fig)]


def _system(plt, report):
    samples = (report.get("system_metrics") or {}).get("samples") or []
    if len(samples) < 2:
        return []
    panels = {
        "CPU (%)": (
            "container_cpu_utilization_percentage",
            "cpu_utilization_percentage",
            "process_cpu_utilization_percentage",
        ),
        "Memória (MiB)": (
            "container_memory_usage_megabytes",
            "process_memory_rss_megabytes",
            "system_memory_usage_megabytes",
        ),
        "Rede acumulada (MB)": ("network_receive_megabytes", "network_transmit_megabytes"),
        "Disco (%) / threads": ("disk_usage_percentage", "process_threads"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for ax, (title, names) in zip(axes.flat, panels.items()):
        for name in names:
            pts = [(s.get("elapsed_seconds", i), _num(s["values"].get(name))) for i, s in enumerate(samples)]
            pts = [p for p in pts if p[1] is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, linewidth=1.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("segundos desde o início")
    fig.suptitle("Recursos do ambiente de execução", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return [("plots/system/resources.png", fig)]


BUILDERS: tuple[Callable, ...] = (
    _performance_dashboard,
    _confusion,
    _error_tradeoff,
    _probabilistic,
    _drift,
    _adaptation,
    _continual,
    _costs,
    _summary,
    _epochs,
    _system,
)


def iter_report_figures(report: dict) -> Iterator[tuple[str, object]]:
    """Yield ``(artifact_path, Figure)`` one at a time, closing each after the consumer is done.

    A builder that fails is skipped: one broken plot never blocks the others or the run.
    """
    if not matplotlib_available():
        return
    plt = _pyplot()
    for builder in BUILDERS:
        try:
            pairs = builder(plt, report)
        except Exception:
            plt.close("all")
            continue
        for artifact_file, fig in pairs:
            try:
                yield artifact_file, fig
            finally:
                plt.close(fig)


@contextmanager
def _active_run(run_id: str | None):
    """Make ``run_id`` the active run for the fluent API without changing its status.

    ``mlflow.log_figure`` has no ``run_id`` argument, so an existing run is resumed and handed back
    as RUNNING: the caller still owns termination (and its completion tags).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    current = mlflow.active_run()
    if run_id is None or (current is not None and current.info.run_id == run_id):
        yield
        return
    if current is not None:
        raise RuntimeError(f"another MLflow run ({current.info.run_id}) is active")
    # start_run() refuses to resume a run whose experiment differs from the process-wide active one.
    mlflow.set_experiment(experiment_id=MlflowClient().get_run(run_id).info.experiment_id)
    mlflow.start_run(run_id=run_id)
    try:
        yield
    finally:
        mlflow.end_run(status="RUNNING")


def log_report_figures(
    report: dict, *, run_id: str | None = None, log_figure: Callable | None = None
) -> list[str]:
    """Log every figure with ``mlflow.log_figure`` (or an injected callable); return artifact paths."""
    logged: list[str] = []
    if log_figure is not None:
        for artifact_file, fig in iter_report_figures(report):
            log_figure(fig, artifact_file)
            logged.append(artifact_file)
        return logged
    import mlflow

    with _active_run(run_id):
        for artifact_file, fig in iter_report_figures(report):
            mlflow.log_figure(fig, artifact_file)
            logged.append(artifact_file)
    return logged


def save_report_figures(report: dict, directory) -> list[str]:
    """Write the same PNGs under ``<directory>/<artifact_path>`` (REST upload fallback, no mlflow)."""
    from pathlib import Path

    root = Path(directory)
    saved: list[str] = []
    for artifact_file, fig in iter_report_figures(report):
        target = root / artifact_file
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=FIGURE_DPI, bbox_inches="tight")
        saved.append(artifact_file)
    return saved
