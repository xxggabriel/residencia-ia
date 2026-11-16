"""Publication figures generated from the CARA and UPF result JSON files."""

from __future__ import annotations

import numpy as np

from .cara_replication import PAPER
from .retraining_replication import PAPER_AUC

INK = "#14233B"
MUTED = "#52627A"
GRID = "#DFE7F0"
CANVAS = "#F7F9FC"
REFERENCE = "#52627A"
COLORS = {
    "cara": "#246BD0", "cara_cumulative": "#007D74", "cara_periodic": "#824BB4",
    "upf": "#A54D26", "oracle": INK, "oracle_staleness": INK, "markov": "#AC580B",
    "frozen": "#697990", "always_retrain": "#B13E62", "adwin": "#397847", "ddm": "#995B40",
}
LABELS = {
    "cara": "CARA-T", "cara_cumulative": "CARA-CT", "cara_periodic": "CARA-P", "upf": "UPF",
    "oracle": "Oracle", "oracle_staleness": "Oracle", "markov": "Markov", "frozen": "Sem retreino",
    "always_retrain": "Retreinar sempre", "adwin": "ADWIN", "ddm": "DDM",
}
STYLES = {"oracle": "--", "oracle_staleness": "--", "frozen": ":", "cara_cumulative": "-.",
          "markov": (0, (5, 2)), "always_retrain": (0, (2, 2))}
DATASET_LABELS = {"covcon": "CovCon", "circle": "Circle", "circles": "Circles", "gauss": "Gauss"}


def _subplots(*args, figsize, **kwargs):
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    return figure, figure.subplots(*args, **kwargs)


def _dataset(name):
    base, sep, variant = name.partition("-")
    return DATASET_LABELS.get(base, base.capitalize()) + sep + variant.upper()


def _axes(ax, *, grid="y"):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("bottom", "left"))
        spine.set_color(GRID)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.tick_params(length=0, colors=MUTED, labelsize=9, pad=7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _header(fig, eyebrow, title, subtitle, *, left=0.065, top=0.72, **layout):
    fig.patch.set_facecolor(CANVAS)
    fig.subplots_adjust(left=left, top=top, **layout)
    fig.text(left, 0.955, eyebrow, color=COLORS["cara"], fontsize=9, fontweight="bold", va="top")
    fig.text(left, 0.907, title, color=INK, fontsize=21, fontweight="bold", va="top")
    fig.text(left, 0.835, subtitle, color=MUTED, fontsize=10, va="top")


def _legend(fig, handles, labels, *, y=0.795, columns=4, x=0.065):
    return fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y), ncol=columns,
                      frameon=False, labelcolor=INK, fontsize=9, handlelength=2.5,
                      borderaxespad=0, columnspacing=1.6)


def _plot_curve(ax, x, values, rule):
    ax.plot(x, values, color=COLORS.get(rule), linestyle=STYLES.get(rule, "-"),
            linewidth=2.05, label=LABELS.get(rule, rule), solid_capstyle="round")


def _percent(ax, *, horizontal=False):
    from matplotlib.ticker import PercentFormatter

    axis = ax.xaxis if horizontal else ax.yaxis
    axis.set_major_formatter(PercentFormatter(1, decimals=0))


def upf_figures(results):
    out = []
    for result in results:
        name, summary = result["dataset"], result["summary"]
        rules = [r for r in ("oracle", "upf", "cara", "cara_cumulative", "cara_periodic") if r in summary]
        fig, ax = _subplots(figsize=(8.8, 5.2))
        y = np.arange(len(rules))
        _axes(ax, grid="x")
        local = ax.barh(y - 0.17, [summary[r]["auc_mean"] for r in rules], 0.28,
                        xerr=[summary[r]["auc_std"] for r in rules], color=COLORS["upf"],
                        error_kw={"ecolor": INK, "capsize": 3, "elinewidth": 1.1})
        paper = PAPER_AUC[name]
        published = [np.nan if paper.get(r) is None else paper[r] for r in rules]
        reference = ax.barh(y + 0.17, published, 0.28, color="#D2DFED", edgecolor="#71849C")
        ax.set_yticks(y, [LABELS[r] for r in rules])
        ax.invert_yaxis()
        ax.set_xlabel("AUC do custo estimado · menor é melhor")
        _header(fig, "UPF  /  TABELA 1", f"Custo acumulado · {_dataset(name)}",
                f"{result['trials']} execuções · média ± desvio padrão", left=0.14, right=0.95, bottom=0.15)
        _legend(fig, [local, reference], ["Resultado local", "Artigo"], columns=2, x=0.14)
        out.append((f"upf_auc_{name}.png", fig))

        if result.get("curves"):
            fig, (top, bottom) = _subplots(2, 1, figsize=(8.8, 7.4), sharex=True)
            for rule, curve in result["curves"].items():
                _plot_curve(top, result["alphas"], curve["cost"], rule)
                _plot_curve(bottom, result["alphas"], curve["retrains"], rule)
            for ax in (top, bottom):
                _axes(ax)
            top.set_ylabel("Custo estimado")
            bottom.set_ylabel("Retreinos")
            bottom.set_xlabel("Custo de retreino (α)")
            _header(fig, "UPF  /  FIGURA 2", f"Custo e retreinos · {_dataset(name)}",
                    f"Sensibilidade a α · {result['trials']} execuções", left=0.11, top=0.725, right=0.96,
                    bottom=0.105, hspace=0.20)
            _legend(fig, *top.get_legend_handles_labels(), x=0.11)
            out.append((f"upf_curves_{name}.png", fig))
    return out


def _cara_summary(results):
    from matplotlib.lines import Line2D

    names = [r["dataset"] for r in results]
    labels = [_dataset(n) for n in names]
    rows = np.arange(len(names))
    height = max(5.2, 3.15 + len(rows) * 0.37)
    fig, axes = _subplots(1, 3, sharey=True, sharex=True, figsize=(11, height))
    for column, (rule, ax) in enumerate(zip(("cara", "cara_cumulative", "cara_periodic"), axes, strict=True)):
        _axes(ax, grid="x")
        ours = np.asarray([r["summary"][rule]["strategy_scpe"] for r in results], dtype=float)
        published = np.asarray([PAPER["scpe"][n][column] for n in names])
        ax.barh(rows, ours, height=0.55, color=COLORS[rule], alpha=0.88)
        ax.scatter(published, rows, marker="|", s=180, linewidths=2.2, color=INK, zorder=3)
        ax.set_title(LABELS[rule], loc="left", color=INK, fontweight="bold", fontsize=12, pad=10)
        ax.set_xlabel("SCPE (%)")
    axes[0].set_yticks(rows, labels)
    axes[0].invert_yaxis()
    _header(fig, "CARA  /  TABELA 2", "Distância ao Oracle",
            "Erro percentual do custo da estratégia · menor é melhor", left=0.12, top=0.69,
            right=0.97, bottom=0.13, wspace=0.18)
    _legend(fig, [Line2D([], [], color=COLORS["cara"], linewidth=7),
                  Line2D([], [], color=INK, linestyle="none", marker="|", markersize=11, markeredgewidth=2)],
            ["Resultado local · cor por variante", "Artigo"], x=0.12, columns=2)
    scpe = fig

    fig, ax = _subplots(figsize=(8.8, height))
    _axes(ax, grid="x")
    ours = np.asarray([r["summary"]["cara"]["accuracy"] for r in results], dtype=float)
    published = np.asarray([PAPER["accuracy_cara_t"][n] for n in names])
    ax.hlines(rows, ours, published, color="#A7B7CB", linewidth=2, zorder=2)
    local = ax.scatter(ours, rows, s=62, color=COLORS["cara"], edgecolor="white", linewidth=0.9, zorder=4)
    paper = ax.scatter(published, rows, s=77, facecolor="white", edgecolor=REFERENCE, linewidth=1.5,
                       marker="D", zorder=3)
    ax.set_yticks(rows, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Acurácia média")
    _percent(ax, horizontal=True)
    ax.set_xlim(max(0, min(np.nanmin(np.r_[ours, published]) - 0.045, 0.6)), 1.02)
    _header(fig, "CARA  /  TABELA 3", "Acurácia do CARA-T nas queries",
            "Cada linha conecta o resultado local ao do artigo para o mesmo dataset.",
            left=0.15, right=0.95, bottom=0.13)
    _legend(fig, [local, paper], ["Resultado local", "Artigo"], columns=2, x=0.15)
    return [("cara_scpe.png", scpe), ("cara_accuracy.png", fig)]


def cara_figures(results):
    out = _cara_summary(results) if results else []
    for result in results:
        if not result.get("curves"):
            continue
        fig, axes = _subplots(1, 3, figsize=(13.5, 5.3), sharex=True)
        panels = (("strategy_cost", "Custo da estratégia"), ("retrains", "Retreinos"),
                  ("average_accuracy", "Acurácia das queries"))
        for ax, (key, title) in zip(axes, panels, strict=True):
            for rule, curve in result["curves"].items():
                _plot_curve(ax, result["kappa"], curve[key], rule)
            _axes(ax)
            ax.set_title(title, loc="left", fontweight="bold", fontsize=12, color=INK, pad=10)
            ax.set_xlabel("Custo de retreino (κ)")
        _percent(axes[2])
        seeds = result["seeds"]
        _header(fig, "CARA  /  FIGURAS 4–7", f"Efeito do custo de retreino · {_dataset(result['dataset'])}",
                f"{seeds} {'semente' if seeds == 1 else 'sementes'} · métricas na fase online",
                top=0.66, right=0.97, bottom=0.16, wspace=0.25)
        _legend(fig, *axes[0].get_legend_handles_labels(), columns=5)
        out.append((f"cara_curves_{result['dataset']}.png", fig))
    return out


def schedule_figure(showcase):
    """Plot candidate models with the selected schedule emphasized."""
    from matplotlib import colormaps

    mean = np.asarray(showcase["accuracy_mean"], dtype=float)
    std = np.asarray(showcase["accuracy_std"], dtype=float)
    size, offline = showcase["batches"], showcase["offline"]
    fig, ax = _subplots(figsize=(11.5, 5.45))
    _axes(ax)
    ax.axvspan(offline - 0.5, size - 0.5, color="#ECF3FF", zorder=0)
    colors = colormaps["cividis"](np.linspace(0.12, 0.72, size - 1))
    for model in range(size - 1):
        xs = np.arange(model + 1, size)
        centre, spread = mean[model, xs], std[model, xs]
        ax.plot(xs, centre, marker="o", markersize=2.4, linewidth=1.05, alpha=0.8, color=colors[model],
                label=f"Modelo {model}")
        ax.fill_between(xs, centre - spread, centre + spread, color=colors[model], alpha=0.08)
    online = np.arange(offline, size)
    schedule = np.asarray(showcase["schedule_mean"], dtype=float)
    ax.plot(online, schedule, color=COLORS["cara"], linewidth=2.8, marker="o", markersize=5,
            markeredgecolor="white", label="Modelo selecionado", zorder=5)
    for batch in showcase["retrain_batches"]:
        level = schedule[batch - offline]
        ax.scatter(batch, level, s=72, marker="D", color="#AC580B", edgecolor="white", zorder=6)
        ax.annotate("Retreino", (batch, level), xytext=(0, 13 if level < 0.9 else -19),
                    textcoords="offset points", ha="center", color="#86470F", fontsize=8, zorder=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8})
    ax.set_xlim(-0.4, size - 0.55)
    ax.set_xticks(range(size))
    ax.set_xlabel("Lote avaliado")
    ax.set_ylabel("Acurácia")
    _percent(ax)
    _header(fig, "CARA  /  ESTRATÉGIA", f"Quando o modelo é retreinado · {_dataset(showcase['dataset'])}",
            "A faixa azul indica a fase online. As faixas finas mostram ±1 desvio padrão dos modelos.",
            left=0.1, top=0.68, right=0.96, bottom=0.14)
    handles, labels = ax.get_legend_handles_labels()
    _legend(fig, [handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]], x=0.1, columns=6)
    return f"cara_schedule_{showcase['dataset']}.png", fig
