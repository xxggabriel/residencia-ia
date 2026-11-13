"""Figures of the UPF and CARA replications: ours against the published numbers, plus the curves.

Input is the JSON written by ``scripts/replicate_upf.py`` / ``scripts/replicate_cara.py``.
Curves (cost, retrains, accuracy against alpha or kappa) exist only in results produced after curves
were recorded; older files still yield the comparison bars.
"""

from __future__ import annotations

from .cara_replication import PAPER
from .retraining_replication import PAPER_AUC

COLORS = {"cara": "#d62728", "cara_cumulative": "#ff7f0e", "cara_periodic": "#8c564b", "upf": "#1f77b4",
          "oracle": "#9467bd", "oracle_staleness": "#9467bd", "markov": "#2ca02c", "frozen": "#7f7f7f"}
STYLES = {"upf": ":", "cara": "-.", "cara_cumulative": "-.", "oracle": "--", "oracle_staleness": "--"}


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _bars(ax, labels, ours, paper, errors=None, ylabel=""):
    width = 0.38
    xs = list(range(len(labels)))
    ax.bar([x - width / 2 for x in xs], ours, width, yerr=errors, label="aqui", color="#1f77b4", capsize=3)
    ax.bar([x + width / 2 for x in xs], [p if p is not None else 0 for p in paper], width,
           label="artigo", color="#ff7f0e")
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def upf_figures(results):
    plt, out = _plt(), []
    for result in results:
        name, summary = result["dataset"], result["summary"]
        rules = [r for r in ("oracle", "upf", "cara", "cara_cumulative", "cara_periodic") if r in summary]
        fig, ax = plt.subplots(figsize=(7, 4))
        _bars(ax, rules, [summary[r]["auc_mean"] for r in rules], [PAPER_AUC[name].get(r) for r in rules],
              [summary[r]["auc_std"] for r in rules], "AUC de $\\hat{C}_\\alpha$ (menor é melhor)")
        ax.set_title(f"UPF Tabela 1 — {name} ({result['trials']} trials)", fontweight="bold")
        fig.tight_layout()
        out.append((f"upf_auc_{name}.png", fig))
        if result.get("curves"):
            fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)
            for rule, curve in result["curves"].items():
                style = {"color": COLORS.get(rule), "linestyle": STYLES.get(rule, "-"), "marker": "o",
                         "markersize": 3, "label": rule}
                top.plot(result["alphas"], curve["cost"], **style)
                bottom.plot(result["alphas"], curve["retrains"], **style)
            top.set_ylabel("$\\hat{C}_\\alpha(\\theta)$")
            bottom.set_ylabel("nº de retreinos")
            bottom.set_xlabel("α")
            top.set_title(f"UPF Fig. 2 — {name}", fontweight="bold")
            bottom.legend(fontsize=8)
            for ax in (top, bottom):
                ax.grid(alpha=0.3)
            fig.tight_layout()
            out.append((f"upf_curves_{name}.png", fig))
    return out


def cara_figures(results):
    plt, out = _plt(), []
    names = [r["dataset"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width, xs = 0.2, list(range(len(names)))
    for k, (rule, label, column) in enumerate((("cara", "CARA-T", 0), ("cara_cumulative", "CARA-CT", 1),
                                               ("cara_periodic", "CARA-P", 2))):
        ours = [r["summary"][rule]["strategy_scpe"] for r in results]
        ax.bar([x + (k - 1) * width for x in xs], ours, width, label=f"{label} (aqui)", color=COLORS[rule])
        ax.scatter([x + (k - 1) * width for x in xs], [PAPER["scpe"][n][column] for n in names], marker="_",
                   s=220, color="black", zorder=3, label="artigo" if k == 0 else None)
    ax.set_xticks(xs, names, rotation=20, ha="right")
    ax.set_ylabel("SCPE médio (%, menor é melhor)")
    ax.set_title("CARA Tabela 2 — SCPE sobre o Oracle (traço preto: artigo)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.append(("cara_scpe.png", fig))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _bars(ax, names, [r["summary"]["cara"]["accuracy"] for r in results],
          [PAPER["accuracy_cara_t"][n] for n in names], ylabel="acurácia média das queries")
    ax.set_title("CARA Tabela 3 — acurácia do CARA-T", fontweight="bold")
    fig.tight_layout()
    out.append(("cara_accuracy.png", fig))
    for result in results:
        if not result.get("curves"):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for rule, curve in result["curves"].items():
            style = {"color": COLORS.get(rule), "linestyle": STYLES.get(rule, "-"), "label": rule,
                     "marker": "o", "markersize": 3}
            for ax, key in zip(axes, ("strategy_cost", "retrains", "average_accuracy")):
                ax.plot(result["kappa"], curve[key], **style)
        for ax, title in zip(axes, ("Custo da estratégia", "# decisões de retreino", "Acurácia das queries")):
            ax.set_title(title)
            ax.set_xlabel("κ")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"CARA Figs. 4–7 — {result['dataset']}", fontweight="bold")
        fig.tight_layout()
        out.append((f"cara_curves_{result['dataset']}.png", fig))
    return out


def schedule_figure(showcase):
    """CARA Fig. 1: accuracy of the model trained on each batch against the batch it is evaluated on."""
    import numpy as np

    plt = _plt()
    mean = np.asarray(showcase["accuracy_mean"], dtype=float)
    std = np.asarray(showcase["accuracy_std"], dtype=float)
    size, offline = showcase["batches"], showcase["offline"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, size - 1))
    for model in range(size - 1):
        xs = list(range(model + 1, size))
        centre, spread = mean[model, model + 1:], std[model, model + 1:]
        label = f"$\\mathcal{{D}}_{{{model}}}$"
        ax.plot(xs, centre, marker="o", markersize=3, color=colors[model], label=label)
        ax.fill_between(xs, centre - spread, centre + spread, color=colors[model], alpha=0.18)
    online = list(range(offline, size))
    schedule = np.asarray(showcase["schedule_mean"], dtype=float)
    ax.plot(online, schedule, color="black", linestyle="-.", linewidth=2.2, marker="o", markersize=3,
            label="retraining schedule", zorder=5)
    top = float(np.nanmax(mean + std))
    for batch in showcase["retrain_batches"]:
        ax.annotate("retrain", (batch, schedule[batch - offline]), xytext=(batch, top + 0.012),
                    ha="center", color="#c0392b", fontsize=9,
                    bbox={"boxstyle": "round", "fc": "white", "ec": "#c0392b"},
                    arrowprops={"arrowstyle": "-", "color": "#c0392b", "alpha": 0.5})
    ax.set_xticks(range(size), [f"$\\mathcal{{D}}_{{{j}}}$" for j in range(size)])
    ax.set_xlabel("Evaluated on dataset")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.4)
    ax.legend(title="model $f$ trained on dataset:", ncol=1, fontsize=7, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"CARA Fig. 1 — {showcase['dataset']} ({showcase['rule']}, {showcase['seeds']} seeds)",
                 fontweight="bold")
    fig.tight_layout()
    return f"cara_schedule_{showcase['dataset']}.png", fig
