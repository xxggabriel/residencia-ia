"""Matplotlib replicas of the CARA and UPF paper figures, from a report's ``retraining`` analysis.

Figure map (paper -> artifact under ``plots/retraining/``):

* UPF Fig. 1  retraining problem            -> ``retraining_problem``
* UPF Fig. 2  C_hat and #retrains vs alpha  -> ``cost_and_retrains_vs_alpha``
* UPF Fig. 3  impact of a wrong alpha       -> ``wrong_alpha``
* UPF Fig. 4  Proposition 3.1 bounds        -> ``retrain_bounds``
* UPF Tab. 1/2 AUC and low/high alpha table -> ``auc_table``, ``low_high_alpha_table``
* UPF App. A.7 forecaster quality           -> ``forecast_quality``
* CARA Fig. 3 cost matrices + CARA params   -> ``cost_matrix``, ``cara_parameters``
* CARA Fig. 4-7 strategy cost/#retrains/acc -> ``strategy_retrains_accuracy``
* CARA Fig. B.10 cumulative strategy cost   -> ``cumulative_strategy_cost``

Builders only read the report. ``alpha`` is UPF's retraining cost ratio and CARA's ``kappa`` per query.
"""

from __future__ import annotations

import math

STYLE = {
    "upf": ("#1f77b4", ":", "o"), "cara": ("#d62728", "-.", "s"), "cara_cumulative": ("#ff7f0e", "-.", "D"),
    "cara_periodic": ("#8c564b", "-", "^"), "oracle": ("#9467bd", "--", "v"), "frozen": ("#7f7f7f", "-", "x"),
}


def _num(value):
    return None if value is None or not math.isfinite(value) else float(value)


def _style(name):
    return STYLE.get(name, ("#333333", "-", "o"))


def _curve(rule, key):
    pairs = [(c["alpha"], _num(c[key])) for c in rule["curve"]]
    return [p[0] for p in pairs if p[1] is not None], [p[1] for p in pairs if p[1] is not None]


def _draw(ax, analysis, key, names=None):
    for name, rule in analysis["rules"].items():
        if names and name not in names:
            continue
        color, line, marker = _style(name)
        xs, ys = _curve(rule, key)
        ax.plot(xs, ys, color=color, linestyle=line, marker=marker, markersize=4, linewidth=1.6,
                label=rule["label"])
    ax.grid(alpha=0.3)
    ax.set_xlabel("α")


def _retraining_problem(plt, report):
    analysis = report["retraining"]
    err = analysis["performance_matrix"]["err"]
    size = len(err)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    cmap = plt.get_cmap("Spectral")
    for i in range(size - 1):
        points = [(j, 1 - err[i][j]) for j in range(i + 1, size) if err[i][j] is not None]
        if points:
            ax.plot(*zip(*points), color=cmap(i / max(size - 2, 1)), marker=".", linewidth=1.1,
                    alpha=0.85, label=f"$D_{{{i}}}$")
    setup, deployed = analysis["setup"], analysis["deployed"]
    first = setup["offline_batches"]
    served = [(first + k, 1 - err[m][first + k]) for k, m in enumerate(deployed["served_by"])]
    ax.plot(*zip(*served), color="black", linestyle="-.", marker=".", linewidth=2.2, label="schedule do run")
    for n, t in enumerate(deployed["retrain_points"]):
        if len(deployed["retrain_points"]) <= 8:
            ax.annotate("retrain", (t + 1, 1 - err[t][t + 1]), xytext=(0, 26), textcoords="offset points",
                        ha="center", arrowprops={"arrowstyle": "-", "color": "#c44"}, color="#c44",
                        fontsize=8, bbox={"boxstyle": "round", "fc": "white", "ec": "#c44"})
        else:  # too many to label one by one: mark them on the schedule line
            ax.plot([t + 1], [1 - err[t][t + 1]], marker="v", color="#c44", markersize=7,
                    label="retreino" if n == 0 else None)
    ax.axvline(first - 0.5, color="#999", linestyle=":", linewidth=1)
    ax.set_xlabel("lote avaliado")
    ax.set_ylabel("acurácia")
    ax.set_title("O problema de retreino: modelo treinado em $D_i$ avaliado em $D_j$", fontweight="bold")
    ax.legend(title="modelo treinado em", ncol=3, fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return [("plots/retraining/retraining_problem.png", fig)]


def _cost_and_retrains(plt, report):
    analysis = report["retraining"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    _draw(top, analysis, "cost")
    top.set_ylabel("$\\hat{C}_\\alpha(\\theta)$")
    top.set_title("Custo e número de retreinos × α", fontweight="bold")
    _draw(bottom, analysis, "retrains")
    bottom.set_ylabel("nº de retreinos")
    bottom.legend(title="estratégia", fontsize=8)
    fig.tight_layout()
    return [("plots/retraining/cost_and_retrains_vs_alpha.png", fig)]


def _retrain_bounds(plt, report):
    analysis = report["retraining"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    _draw(left, analysis, "cost", {"upf", "oracle"})
    _draw(right, analysis, "retrains", {"upf", "oracle"})
    colors = {"r<=0": "#d62728", "r<=1": "#1f4e9d", "r<=2": "#2ca02c"}
    top_alpha = max(analysis["alphas"])
    for label, alpha in analysis["setup"]["bound_alphas"].items():
        if alpha is None or alpha > top_alpha * 1.6:
            continue
        for ax in (left, right):
            ax.axvline(alpha, color=colors[label], linewidth=1, label=f"$r^* \\leq {label[-1]}$"
                       if ax is right else None)
    left.set_ylabel("$\\hat{C}_\\alpha(\\theta)$")
    right.set_ylabel("nº de retreinos")
    right.legend(fontsize=8)
    fig.suptitle(f"Cotas da Proposição 3.1 (L = {analysis['setup']['lipschitz_L']:.3f})", fontweight="bold")
    fig.tight_layout()
    return [("plots/retraining/retrain_bounds.png", fig)]


def _wrong_alpha(plt, report):
    grids = report["retraining"]["wrong_alpha"]
    if not grids:
        return []
    fig, axes = plt.subplots(1, len(grids), figsize=(6.2 * len(grids), 4.8), squeeze=False)
    for ax, (name, grid) in zip(axes[0], grids.items()):
        matrix = [[math.nan if v is None else v for v in row] for row in grid["pct_increase"]]
        image = ax.imshow(matrix, cmap="YlOrBr", aspect="auto", vmin=0)
        ticks = list(range(len(grid["real"])))
        labels = [f"{a:.3g}" for a in grid["real"]]
        ax.set_xticks(ticks[:: max(1, len(ticks) // 6)], labels[:: max(1, len(ticks) // 6)], rotation=90)
        ax.set_yticks(ticks[:: max(1, len(ticks) // 6)], labels[:: max(1, len(ticks) // 6)])
        ax.set_xlabel("α real")
        ax.set_ylabel("α informado")
        ax.set_title(f"% de aumento de $\\hat{{C}}$ — {report['retraining']['rules'][name]['label']}")
        fig.colorbar(image, ax=ax)
    fig.suptitle("Robustez a um α especificado errado", fontweight="bold")
    fig.tight_layout()
    return [("plots/retraining/wrong_alpha.png", fig)]


def _tables(plt, report):
    analysis = report["retraining"]
    rules = analysis["rules"]
    labels = [rules[name]["label"] for name in rules]
    out = []
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = list(rules)
    auc = [rules[n]["auc"] for n in names]
    best = min((v for v in auc if v is not None), default=None)
    ax.bar(labels, [v or 0 for v in auc], color=[_style(n)[0] for n in names])
    for index, value in enumerate(auc):
        if value is not None:
            ax.text(index, value, f"{value:.4f}" + ("*" if value == best else ""), ha="center",
                    va="bottom", fontsize=8)
    ax.set_ylabel("AUC de $\\hat{C}_\\alpha(\\theta)$ (menor é melhor)")
    ax.set_title("AUC sobre α ∈ [0, α_max] (Tabela 1)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out.append(("plots/retraining/auc_table.png", fig))
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(names) + 1.8))
    ax.axis("off")
    rows = []
    for name in names:
        row = [rules[name]["label"]]
        for key in ("high", "low"):
            cell = analysis["table"][key].get(name)
            row += ["–"] * 3 if cell is None else [
                str(cell["retrains"]), f"{cell['average_accuracy']:.3f}", f"{cell['cost']:.3f}"]
        rows.append(row)
    top = analysis["setup"]["alpha_max"]
    header = ["", f"#retrain (α={0.9 * top:.3g})", "acurácia média", "Ĉ",
              f"#retrain (α={0.1 * top:.3g})", "acurácia média", "Ĉ"]
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Custo de retreino alto (0,9 α_max) × baixo (0,1 α_max) — Tabela 2", fontweight="bold")
    fig.tight_layout()
    out.append(("plots/retraining/low_high_alpha_table.png", fig))
    return out


def _forecast_quality(plt, report):
    forecast = report["retraining"].get("forecast")
    if not forecast or not forecast.get("pairs"):
        return []
    predicted, actual = zip(*forecast["pairs"])
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(actual, predicted, s=10, alpha=0.4, color="#1f77b4")
    top = max(max(actual), max(predicted), 1e-6)
    ax.plot([0, top], [0, top], color="#999", linestyle="--")
    ax.set_xlabel("erro observado $pe_{ij}$")
    ax.set_ylabel("erro previsto $\\mu_\\phi(r_{ij})$")
    ax.set_title(f"Previsor UPF — MAE {forecast['mae']:.4f}, viés {forecast['bias']:+.4f}", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return [("plots/retraining/forecast_quality.png", fig)]


def _cost_matrix(plt, report):
    analysis = report["retraining"]
    stale = [[math.nan if v is None else v for v in row] for row in analysis["staleness_matrix"]]
    size = len(stale)
    # heatmap[model trained, batch evaluated] = relative staleness
    grid = [[stale[t][m] if t > m else math.nan for t in range(size)] for m in range(size)]
    fig, ax = plt.subplots(figsize=(7.4, 6))
    image = ax.imshow(grid, cmap="Greys", aspect="auto")
    fig.colorbar(image, ax=ax, label="obsolescência relativa $\\bar\\Psi$")
    first = analysis["setup"]["offline_batches"]
    deployed = analysis["deployed"]
    path = [(first + k, m) for k, m in enumerate(deployed["served_by"])]
    ax.step([p[0] for p in path], [p[1] for p in path], where="post", color="#d62728", linewidth=1.6,
            label=f"{deployed['policy']} (run)")
    ax.plot(deployed["retrain_points"], deployed["retrain_points"], "o", color="#d62728", markersize=5)
    oracle = deployed["oracle_retrain_points"]
    ax.plot(oracle, oracle, "x", color="#9467bd", markersize=7, label="oracle")
    ax.axvline(first - 0.5, color="#999", linestyle=":")
    ax.set_xlabel("lote avaliado t")
    ax.set_ylabel("lote de treino t'")
    ax.invert_yaxis()
    ax.set_title("Matriz de custo (obsolescência) e estratégias de retreino", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return [("plots/retraining/cost_matrix.png", fig)]


def _cara_parameters(plt, report):
    rules = report["retraining"]["rules"]
    panels = (("cara", "tau", "$\\tau^*$"), ("cara_cumulative", "tau_cum", "$\\tau_{cum}^*$"),
              ("cara_periodic", "period", "$\\phi^*$"))
    available = [(n, k, t) for n, k, t in panels if n in rules]
    if not available:
        return []
    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 3.6), squeeze=False)
    for ax, (name, key, title) in zip(axes[0], available):
        points = [(c["alpha"], c["parameters"].get(key)) for c in rules[name]["curve"]]
        points = [(a, v) for a, v in points if v is not None]
        if points:
            ax.plot(*zip(*points), marker="o", color=_style(name)[0])
        ax.set_xlabel("κ = α")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Parâmetros ótimos de CARA otimizados offline", fontweight="bold")
    fig.tight_layout()
    return [("plots/retraining/cara_parameters.png", fig)]


def _strategy_retrains_accuracy(plt, report):
    analysis = report["retraining"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title in zip(axes, ("strategy_cost", "retrains", "average_accuracy"),
                              ("Custo da estratégia", "# decisões de retreino", "Acurácia das queries")):
        _draw(ax, analysis, key)
        ax.set_xlabel("κ = α")
        ax.set_title(title)
    oracle = analysis["rules"].get("oracle")
    if oracle is not None:
        xs = [c["alpha"] for c in oracle["curve"]]
        axes[0].plot(xs, [c["oracle_strategy_cost"] for c in oracle["curve"]], color="#2ca02c", linestyle="-",
                     linewidth=2.0, label="oracle (DP sobre Ψ̄)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Custo da estratégia, nº de retreinos e acurácia × custo de retreino", fontweight="bold")
    fig.tight_layout()
    return [("plots/retraining/strategy_retrains_accuracy.png", fig)]


def _cumulative(plt, report):
    analysis = report["retraining"]
    first = analysis["setup"]["offline_batches"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, series in analysis["cumulative_strategy_cost"].items():
        color, line, _ = _style(name)
        points = [(first + k, v) for k, v in enumerate(series) if v is not None]
        if points:
            ax.plot(*zip(*points), color=color, linestyle=line, linewidth=1.8,
                    label=analysis["rules"][name]["label"])
    ax.set_xlabel("lote online t")
    ax.set_ylabel("custo acumulado da estratégia")
    ax.set_title(f"Avaliação online (α = {analysis['setup']['focus_alpha']:.3g})", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return [("plots/retraining/cumulative_strategy_cost.png", fig)]


def _guard(builder):
    def wrapped(plt, report):
        return builder(plt, report) if report.get("retraining") else []
    wrapped.__name__ = builder.__name__
    return wrapped


RETRAINING_BUILDERS = tuple(_guard(b) for b in (
    _retraining_problem, _cost_and_retrains, _retrain_bounds, _wrong_alpha, _tables, _forecast_quality,
    _cost_matrix, _cara_parameters, _strategy_retrains_accuracy, _cumulative,
))
