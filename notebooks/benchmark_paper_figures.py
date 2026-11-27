"""Readable historical CARA/UPF figures and descriptive published comparisons.

Builders only read numeric evidence. They neither load learners nor replay training.
Global titles and footers can be disabled for insertion in editable documents.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from benchmark_plot_style import MUTED, apply_style, decorate, validate_figure
from benchmark_plot_style import POLICY_COLORS as COLORS
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

LABELS = {
    "cara": "CARA (CARA-T)",
    "upf": "UPF",
    "frozen": "Congelado",
    "oracle": "Oracle",
    "oracle_staleness": "Oracle",
    "markov": "Markov",
    "always_retrain": "Sempre retreinar",
}
DATASETS = ("electricity", "airlines")
DATASET_NAMES = {"electricity": "Electricity", "airlines": "Airlines"}
UPF_PAPER = "When to retrain a machine learning model (2025)"
UPF_URL = "https://proceedings.mlr.press/v267/regol25a.html"
UPF_DOI = "https://doi.org/10.48550/arXiv.2505.14903"
CARA_PAPER = "Cost-aware retraining for machine learning (2024)"
CARA_DOI = "https://doi.org/10.1016/j.knosys.2024.111610"
REPRESENTATIVE = (
    "CARA designa a variante CARA-T como representante visual. CARA-CT e CARA-P permanecem "
    "nos dados e nas tabelas; sua omissão nesta figura não afirma equivalência entre variantes."
)
UPF_LIMIT = (
    "Esta é uma execução histórica descritiva, anterior à rodada F109. As janelas sorteadas podem "
    "se sobrepor e a divisão local de treino e teste foi aleatória dentro do lote. CARA, adaptado "
    "à rota UPF histórica, também utilizou rótulos do teste corrente na obsolescência. Não é uma "
    "reprodução idêntica nem comprova avaliação prospectiva independente."
)
CARA_LIMIT = (
    "Esta é a execução histórica com Random Forest, anterior à rodada F109. O pré-processamento "
    "foi ajustado nos atributos do fluxo inteiro, uma convenção transdutiva. Versões e otimização "
    "diferem da referência. O cotejo é descritivo e não afirma reprodução idêntica."
)
METRICS = {
    "cost": ("Custo combinado", "Custo", False),
    "strategy_cost": ("Custo da estratégia", "Custo", False),
    "strategy_scpe": ("SCPE", "SCPE (%)", False),
    "average_accuracy": ("Acurácia média", "Acurácia", True),
    "retrains": ("Retreinamentos", "Retreinamentos", False),
    "combined_cost_auc": ("AUC do custo combinado", "AUC do custo", False),
}
MEANINGS = {
    "cost": "Soma das taxas de erro dos lotes mais α por retreinamento. Não é faturamento.",
    "strategy_cost": "Custo de retreinar e de manter um modelo obsoleto, nas unidades do objetivo CARA.",
    "strategy_scpe": (
        "Excesso percentual absoluto do custo da estratégia em relação ao Oracle de obsolescência. "
        "Não é erro de classificação; fica indefinido quando o custo do Oracle é zero."
    ),
    "average_accuracy": (
        "Média da fração de consultas corretas por lote. Não é F1 macro nem necessariamente a "
        "acurácia obtida reunindo todas as observações."
    ),
    "retrains": (
        "Média de decisões de retreinamento. Menos decisões, isoladamente, não significa melhor resultado."
    ),
    "combined_cost_auc": "Integral do custo combinado sobre o intervalo de α indicado. Não é AUROC nem AP.",
}
LINESTYLES = {
    "cara": "-",
    "upf": "--",
    "frozen": ":",
    "oracle": "-.",
    "oracle_staleness": "-.",
    "markov": "--",
    "always_retrain": ":",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(path):
    path = Path(path)
    return {"file": path.name, "sha256": _hash(path)}


def _archive_directory(root):
    root = Path(root)
    candidates = (root, root / "dados", root / "evidencias/semana-10/cara-upf/dados")
    matches = [p for p in candidates if (p / "upf.json.gz").is_file()]
    _require(bool(matches), "A pasta deve conter os arquivos históricos CARA/UPF.")
    return matches[0]


def _archive(root, name):
    path = root / f"{name}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    _require(isinstance(rows, list) and rows, f"Arquivo histórico vazio: {name}.")
    datasets = [row["dataset"] for row in rows]
    _require(len(datasets) == len(set(datasets)), f"Dataset repetido em {name}.")
    return {row["dataset"]: row for row in rows}, path


def _numbers(values, *, missing=False):
    result = []
    for value in values:
        if value is None and missing:
            result.append(np.nan)
        else:
            _require(value is not None and math.isfinite(float(value)), "Métrica ausente ou não finita.")
            result.append(float(value))
    return np.asarray(result)


def _plain(value, _position=None):
    return f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", "_").replace(".", ",").replace("_", ".")


def _percent(value, _position=None):
    return f"{100 * value:.0f}%"


def _value_label(metric, value, *, published=False):
    value = float(value)
    if metric == "average_accuracy":
        value *= 100
        places = 0 if published and math.isclose(value, round(value), abs_tol=1e-10) else 2
        return f"{value:.{places}f}%".replace(".", ",")
    places = {
        "combined_cost_auc": 4,
        "strategy_scpe": 2 if published else 3,
        "cost": 2 if published else 3,
        "retrains": 2,
    }.get(metric, 3)
    return f"{value:.{places}f}".replace(".", ",")


def _layout(panels, show_title, show_footer, *, left=0.16, height=6.5):
    fig, axes = plt.subplots(panels, 1, figsize=(6.2, height), dpi=300, squeeze=False)
    fig.subplots_adjust(
        left=left,
        right=0.96,
        top=0.83 if show_title else 0.955,
        bottom=0.245 if show_footer else 0.175,
        hspace=0.53,
    )
    return fig, list(axes[:, 0])


def _axes(ax, *, percent=False, logarithmic=False):
    ax.grid(axis="y", color="#DDE7EC", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent if percent else _plain))
    if logarithmic:
        ax.set_yscale("symlog", linthresh=10)
        ax.yaxis.set_major_formatter(FuncFormatter(_plain))


def _legend(fig, policies, show_footer, *, sources=False):
    if sources:
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=MUTED,
                markeredgecolor=MUTED,
                label="Histórico local",
                markersize=6,
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=MUTED,
                label="Publicado",
                markersize=6,
            ),
        ]
    else:
        handles = [
            Line2D([], [], color=COLORS[p], linestyle=LINESTYLES[p], linewidth=2, label=LABELS[p])
            for p in policies
        ]
    columns = 3 if len(handles) == 5 else (2 if len(handles) > 3 else len(handles))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.075 if show_footer else 0.018),
        ncols=columns,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.6,
        labelspacing=0.4,
    )


def _finish(
    fig, title, caption, *, show_title, show_footer, source_slot_id, source_files, metrics, footer, **metadata
):
    decorate(
        fig,
        title,
        caption,
        show_title=show_title,
        show_footer=show_footer,
        footer=footer,
        metric_descriptions={METRICS[m][0]: MEANINGS[m] for m in metrics},
        source_slot_id=source_slot_id,
        source_files=[_source(p) for p in source_files],
        represented_CARA="cara",
        omitted_display_variants=["cara_cumulative", "cara_periodic"],
        omission_means_equivalence=False,
        protocol_scope="historical_not_F109",
        **metadata,
    )
    validate_figure(fig)
    return fig


def _wilson(probabilities, counts):
    p, n = _numbers(probabilities), _numbers(counts)
    _require(np.all((0 <= p) & (p <= 1)) and np.all(n > 0), "Acurácia ou denominador inválido.")
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - width, center + width


def _trajectory(row, path, show_title, show_footer):
    dataset = row["dataset"]
    first = int(row["offline"]) - 1
    matrix = np.asarray(
        [[np.nan if x is None else float(x) for x in line] for line in row["accuracy_matrix"]]
    )
    _require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "Matriz de acurácia não quadrada.")
    trace = row["deployed_trace"]
    periods = np.asarray([int(x["period"]) for x in trace])
    accuracy = _numbers([x["accuracy"] for x in trace])
    _require(
        math.isclose(float(accuracy.mean()), row["average_accuracy"], abs_tol=1e-12),
        "A trajetória não confere com a acurácia arquivada.",
    )
    cost = len(row["retrains"]) * row["alpha"] + float(np.sum(1 - accuracy))
    _require(math.isclose(cost, row["cost"], abs_tol=1e-12), "O custo da trajetória não confere.")
    for record in trace:
        _require(
            math.isclose(
                matrix[int(record["model"]), int(record["period"])], record["accuracy"], abs_tol=1e-12
            ),
            "Modelo ativo divergente da matriz.",
        )
    fig, axes = _layout(2, show_title, show_footer, left=0.13)
    upper, lower = axes
    bounds = upper.get_position()
    upper.set_position([bounds.x0, bounds.y0, bounds.width - 0.1, bounds.height])
    shown = matrix[first:, first:]
    cmap = LinearSegmentedColormap.from_list("paper_accuracy", ["#F1F7FA", "#75AEC2", "#164D68"])
    cmap.set_bad("#F4F6F7")
    image = upper.imshow(
        np.ma.masked_invalid(shown),
        origin="upper",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=cmap,
        interpolation="nearest",
    )
    ticks = range(len(shown))
    upper.set_xticks(ticks, [f"D{x}" for x in range(first, len(matrix))])
    upper.set_yticks(ticks, [f"M{x}" for x in range(first, len(matrix))])
    upper.set_ylabel("Modelo candidato")
    upper.set_xlabel("Lote avaliado")
    colorbar_axis = fig.add_axes([0.89, bounds.y0, 0.024, bounds.height])
    colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=[0, 0.5, 1])
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    counts = [row["test_counts"][p] for p in periods]
    low, high = _wilson(accuracy, counts)
    color = COLORS[row["policy"]]
    lower.fill_between(periods, low, high, color=color, alpha=0.15, linewidth=0)
    lower.plot(periods, accuracy, color=color, linewidth=2.2, marker="o", markersize=4)
    retrains = set(row["retrains"])
    selected = [i for i, period in enumerate(periods) if period in retrains]
    lower.scatter(
        periods[selected],
        accuracy[selected],
        marker="s",
        s=45,
        facecolors="white",
        edgecolors=color,
        linewidths=1.7,
        zorder=4,
    )
    lower.set_xticks(periods, [f"D{p}" for p in periods])
    lower.set_xlabel("Lote atendido pelo modelo em uso")
    lower.set_ylabel("Acurácia em uso")
    lower.set_ylim(max(0, float(low.min()) - 0.04), min(1, float(high.max()) + 0.04))
    _axes(lower, percent=True)
    handles = [
        Line2D([], [], color=color, marker="o", label=LABELS[row["policy"]], linewidth=2),
        Line2D(
            [],
            [],
            color=color,
            marker="s",
            markerfacecolor="white",
            linestyle="none",
            label="Retreinamento",
            markersize=6,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.075 if show_footer else 0.018),
        ncols=2,
        frameon=False,
    )
    caption = (
        f"{DATASET_NAMES[dataset]}. No painel superior, cada linha representa um candidato e cada "
        "coluna representa um lote; células vazias não têm avaliação disponível. No painel inferior, "
        f"mostrei a acurácia do modelo em uso na seed {row['seed']}, com α igual a {row['alpha']}. "
        "Quadrados indicam retreinamentos com implantação no próprio lote. A faixa é o intervalo "
        "de Wilson de 95% para a acurácia de cada lote, usando seu denominador de consultas. É uma "
        "descrição binomial por lote de uma única seed, não dispersão entre seeds nem evidência de "
        "independência temporal. Os dados vêm do benchmark pareado histórico, sem reconstruir modelos. "
        + REPRESENTATIVE
    )
    return _finish(
        fig,
        f"{DATASET_NAMES[dataset]}: candidatos e modelo em uso",
        caption,
        show_title=show_title,
        show_footer=show_footer,
        source_slot_id=f"trajetoria-{dataset}",
        source_files=[path],
        metrics=["average_accuracy"],
        footer=f"Histórico pareado. Seed {row['seed']}. A faixa de Wilson é por lote.",
        dataset=dataset,
        seed=row["seed"],
        alpha=row["alpha"],
        interval="Wilson95_per_batch",
        panel_labels=["Candidatos", "Modelo em uso"],
        plotted_data=row,
    )


def _curve_pair(row, metrics, protocol, path, show_title, show_footer, part):
    cara = protocol == "cara"
    dataset = row["dataset"]
    policies = [
        p
        for p in (
            ["cara", "oracle_staleness", "frozen", "markov", "always_retrain"]
            if cara
            else ["cara", "upf", "oracle", "frozen"]
        )
        if p in row["curves"]
    ]
    grid = _numbers(row["kappa" if cara else "alphas"])
    fig, axes = _layout(len(metrics), show_title, show_footer, height=6.5 if len(metrics) > 1 else 4.9)
    fig.subplots_adjust(bottom=0.30 if show_footer else 0.22)
    plotted = []
    for axis, metric in zip(axes, metrics, strict=True):
        for policy in policies:
            raw = row["curves"][policy][metric]
            values = _numbers(raw, missing=metric == "strategy_scpe")
            _require(len(values) == len(grid), "Grade de preços e curva têm tamanhos diferentes.")
            axis.plot(
                grid,
                values,
                color=COLORS[policy],
                linestyle=LINESTYLES[policy],
                linewidth=2,
                marker=None if cara else "o",
                markersize=3,
            )
            plotted.append({"policy": policy, "metric": metric, "x": grid.tolist(), "value": raw})
        axis.set_ylabel(METRICS[metric][1])
        _axes(
            axis,
            percent=METRICS[metric][2],
            logarithmic=cara and metric in {"strategy_cost", "strategy_scpe"},
        )
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.xaxis.set_major_formatter(FuncFormatter(_plain))
        axis.set_xlim(float(grid[0]), float(grid[-1]))
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    axes[-1].set_xlabel("κ: preço do retreinamento" if cara else "α: preço relativo do retreinamento")
    _legend(fig, policies, show_footer)
    metrics_text = " e ".join(METRICS[m][0].lower() for m in metrics)
    caption = f"{DATASET_NAMES[dataset]}. {metrics_text.capitalize()} ao variar o preço de retreinamento. "
    if cara:
        caption += (
            "As curvas são médias de cinco seeds em cada um dos 100 preços de κ. SCPE fica "
            "indefinido em κ igual a zero, sem preencher a lacuna com zero. Os resumos agregados "
            "do arquivo, distintos destas curvas, excluem κ igual a zero nas três métricas. "
            "Custo e SCPE usam escala logarítmica simétrica, com trecho linear próximo de zero. "
            "Não desenhei barras por seed: os desvios não foram preservados. " + CARA_LIMIT
        )
    else:
        caption += (
            f"São médias de {row['trials']} repetições em cada preço de α; não são novas "
            "execuções nem os resultados F109. As curvas preservam a grade numérica original, "
            "com espaçamento proporcional aos preços. " + UPF_LIMIT
        )
    caption += " " + REPRESENTATIVE
    return _finish(
        fig,
        f"{DATASET_NAMES[dataset]}: protocolo {protocol.upper()}, parte {part}",
        caption,
        show_title=show_title,
        show_footer=show_footer,
        source_slot_id=f"metricas-protocolo-{protocol}",
        source_files=[path],
        metrics=metrics,
        footer="Médias históricas por preço. CARA representa CARA-T.",
        dataset=dataset,
        source_slot_part=f"{dataset}-{part}",
        panel_labels=[METRICS[m][0] for m in metrics],
        policies_shown=policies,
        plotted_data=plotted,
        dispersion="not_shown",
    )


def _pair_points(axis, records, policies, metric, *, log=False):
    positions = np.arange(len(policies))
    extents = []
    for index, policy in enumerate(policies):
        record = next(r for r in records if r["policy"] == policy)
        local, published = float(record["local_mean"]), float(record["published_value"])
        sd = record.get("local_sd")
        spread = record.get("published_spread")
        sd = None if sd in (None, "") else float(sd)
        spread = None if spread in (None, "") else float(spread)
        for value, error, offset, marker, filled in [
            (local, sd, -0.18, "o", True),
            (published, spread, 0.18, "D", False),
        ]:
            axis.errorbar(
                value,
                index + offset,
                xerr=error,
                fmt=marker,
                color=COLORS[policy],
                markerfacecolor=COLORS[policy] if filled else "white",
                markeredgewidth=1.3,
                markersize=6,
                capsize=3,
                elinewidth=1.3,
            )
            axis.text(
                1.04,
                index + offset,
                _value_label(metric, value, published=not filled),
                transform=axis.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=11,
                color=COLORS[policy],
                clip_on=False,
            )
            extents.extend([value - (error or 0), value + (error or 0)])
    axis.set_yticks(positions, [LABELS[p] for p in policies])
    axis.set_ylim(len(policies) - 0.5, -0.5)
    axis.grid(axis="x", color="#DDE7EC", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(length=3)
    if log:
        _require(min(extents) > 0, "Escala logarítmica exige SCPE positivo.")
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(_plain))
        axis.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        left, right = min(extents) / 1.3, max(extents) * 1.3
        axis.set_xlim(left, right)
        candidates = (
            [5, 8, 10, 15, 20, 30, 50, 100, 150, 200]
            if right / left < 5
            else [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        )
        axis.set_xticks([value for value in candidates if left <= value <= right])
    elif metric == "average_accuracy":
        axis.set_xlim(0, 1)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.xaxis.set_major_formatter(FuncFormatter(_percent))
    else:
        left, right = min(extents), max(extents)
        gap = max((right - left) * 0.1, 0.04)
        axis.set_xlim(left - gap, right + gap)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.xaxis.set_major_formatter(FuncFormatter(_plain))


def _auc(upf, path, show_title, show_footer):
    policies = ["cara", "upf", "oracle"]
    fig, axes = _layout(2, show_title, show_footer, left=0.25)
    fig.subplots_adjust(right=0.79)
    plotted = []
    for axis, dataset in zip(axes, DATASETS, strict=True):
        row = upf[dataset]
        records = [
            {
                "policy": p,
                "local_mean": row["summary"][p]["auc_mean"],
                "local_sd": row["summary"][p]["auc_std"],
                "published_value": row["summary"][p]["paper_auc"],
                "published_spread": None,
            }
            for p in policies
        ]
        _require(all(r["published_value"] is not None for r in records), "AUC publicada indisponível.")
        _pair_points(axis, records, policies, "combined_cost_auc")
        axis.set_title(DATASET_NAMES[dataset], loc="left", fontsize=11, fontweight="bold", pad=5)
        axis.set_xlabel(f"AUC do custo; α de 0 a {_plain(row['alpha_max'])}")
        plotted.append({"dataset": dataset, "alpha_min": 0, "alpha_max": row["alpha_max"], "rows": records})
    _legend(fig, [], show_footer, sources=True)
    caption = (
        "Electricity acima e Airlines abaixo. Comparei a AUC do custo combinado do arquivo histórico "
        f"com a Tabela 1 de {UPF_PAPER}, página 8 do PDF final. O DOI disponível identifica a versão "
        "preprint. Pontos cheios são as médias locais de dez seeds; suas barras são um desvio padrão "
        "populacional, com ddof igual a zero. Losangos vazios são valores publicados, sem dispersão "
        "informada nessa tabela. A integral cobre α entre 0 e 1 em Electricity e entre 0 e 0,7 em "
        "Airlines. AUC de custo menor é preferível dentro do mesmo protocolo e intervalo; não é "
        "AUROC nem autoriza comparar os dois datasets diretamente por esse número. "
        + UPF_LIMIT
        + " "
        + REPRESENTATIVE
    )
    return _finish(
        fig,
        "AUC do custo: referência publicada e execução histórica",
        caption,
        show_title=show_title,
        show_footer=show_footer,
        source_slot_id="auc-upf-publicado-local",
        source_files=[path],
        metrics=["combined_cost_auc"],
        footer="Cheio: histórico local. Vazio: publicado. Não é AUROC.",
        paper_title=UPF_PAPER,
        paper_url=UPF_URL,
        paper_doi=UPF_DOI,
        doi_version="preprint",
        published_table="1",
        published_pdf_page=8,
        local_sd_ddof=0,
        local_n=10,
        plotted_data=plotted,
    )


def _sensitivity(dataset, modes, paths, show_title, show_footer):
    policies = ["cara", "upf", "frozen"]
    fig, axes = _layout(2, show_title, show_footer)
    plotted = []
    for axis, mode in zip(axes, ("same-batch", "next-batch"), strict=True):
        row = modes[mode][dataset]
        grid = _numbers(row["alphas"])
        for policy in policies:
            values = _numbers(row["curves"][policy]["error_oracle_gap"])
            _require(len(values) == len(grid), "Grade de sensibilidade incompatível.")
            axis.plot(grid, values, color=COLORS[policy], linestyle=LINESTYLES[policy], linewidth=2)
            plotted.append(
                {"mode": mode, "policy": policy, "alpha": grid.tolist(), "difference": values.tolist()}
            )
        axis.axhline(0, color=COLORS["oracle"], linewidth=1, linestyle="-.")
        axis.set_title(
            "Implantação no mesmo lote" if mode == "same-batch" else "Implantação no lote seguinte",
            loc="left",
            fontsize=11,
            fontweight="bold",
            pad=5,
        )
        axis.set_ylabel("Excesso de custo")
        axis.set_xlim(float(grid[0]), float(grid[-1]))
        _axes(axis)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.xaxis.set_major_formatter(FuncFormatter(_plain))
    axes[0].tick_params(labelbottom=False)
    axes[-1].set_xlabel("α: preço relativo do retreinamento")
    _legend(fig, policies, show_footer)
    caption = (
        f"{DATASET_NAMES[dataset]}. Subtraí o custo combinado do Oracle de erro do custo de cada "
        "política nos mesmos candidatos e testes históricos. O painel superior usa implantação no "
        "próprio lote; o inferior usa implantação no seguinte. Cada curva é a média de dez seeds "
        "em cada preço. O eixo α respeita as distâncias numéricas entre os 20 preços, sem transformar "
        "uma grade irregular em colunas igualmente espaçadas. A linha zero é a referência Oracle. "
        "O custo soma taxas de erro e penalidades por retreinamento; não é SCPE nem faturamento. "
        "A comparação pareada separou a referência rotulada de treino das consultas de teste. "
        + REPRESENTATIVE
    )
    return _finish(
        fig,
        f"{DATASET_NAMES[dataset]}: excesso de custo para o Oracle",
        caption,
        show_title=show_title,
        show_footer=show_footer,
        source_slot_id="sensibilidade-oracle",
        source_files=paths,
        metrics=["cost"],
        footer="Benchmark histórico pareado. Linha zero: Oracle de erro.",
        dataset=dataset,
        source_slot_part=dataset,
        plotted_data=plotted,
        local_n=10,
        dispersion="not_shown",
    )


def build_historical_paper_figures(root, *, show_title=True, show_footer=True) -> dict:
    """Read archived EA numeric evidence and return thirteen native Matplotlib figures."""
    apply_style()
    root = _archive_directory(root)
    upf, upf_path = _archive(root, "upf")
    _require(set(upf) == set(DATASETS), "O protocolo histórico deve conter Electricity e Airlines.")
    path = root / "trajetorias.json"
    trajectories = json.loads(path.read_text(encoding="utf-8"))
    _require({r["dataset"] for r in trajectories} == set(DATASETS), "Trajetórias históricas incompletas.")
    figures = {
        f"historico-trajetoria-{r['dataset']}": _trajectory(r, path, show_title, show_footer)
        for r in trajectories
    }
    figures["historico-upf-auc-publicado-local"] = _auc(upf, upf_path, show_title, show_footer)
    for dataset in DATASETS:
        for part, metrics in enumerate((("cost", "average_accuracy"), ("retrains",)), 1):
            figures[f"historico-protocolo-upf-{dataset}-{part}"] = _curve_pair(
                upf[dataset], metrics, "upf", upf_path, show_title, show_footer, part
            )
        cara, cara_path = _archive(root, f"cara-{dataset}")
        for part, metrics in enumerate(
            (("strategy_cost", "strategy_scpe"), ("average_accuracy", "retrains")), 1
        ):
            figures[f"historico-protocolo-cara-{dataset}-{part}"] = _curve_pair(
                cara[dataset], metrics, "cara", cara_path, show_title, show_footer, part
            )
    modes, paths = {}, []
    for mode in ("same-batch", "next-batch"):
        modes[mode], path = _archive(root, mode)
        paths.append(path)
    for dataset in DATASETS:
        figures[f"historico-sensibilidade-oracle-{dataset}"] = _sensitivity(
            dataset, modes, paths, show_title, show_footer
        )
    _require(len(figures) == 13, "Número inesperado de figuras históricas.")
    return figures


def _comparison_records(papers_dir):
    root = Path(papers_dir)
    path = root / "paper-local-comparisons.csv"
    manifest = root / "paper-data-provenance.json"
    if manifest.is_file():
        package = json.loads(manifest.read_text(encoding="utf-8"))
        expected = package["files"][path.name]["sha256"]
        _require(_hash(path) == expected, "O CSV comparativo difere da proveniência fornecida.")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(rows and len({r["reference_id"] for r in rows}) == len(rows), "Comparativo vazio ou duplicado.")
    for row in rows:
        _numbers([row["local_mean"], row["published_value"]])
        _require(
            row["paper_id"] in {"cara_2024", "upf_2025"} and row["dataset"] in DATASETS,
            "Não há referência publicada selecionada para esse artigo ou dataset.",
        )
        expected = "cara_historical" if row["paper_id"] == "cara_2024" else "upf_historical"
        _require(
            row["local_protocol"] == expected,
            "O cotejo exige o protocolo histórico correspondente, não F109.",
        )
        _require(
            str(row["f109_comparable"]).lower() == "false", "Não comparar F109 numericamente com o artigo."
        )
        n = int(row["n_local"])
        seeds = [int(seed) for seed in row["local_seed_values"].split(";")]
        _require(n > 0 and len(seeds) == len(set(seeds)) == n, "Seeds locais não conferem com n_local.")
        if row["local_sd"]:
            ddof = int(row["local_sd_ddof"])
            _require(0 <= ddof < n and float(row["local_sd"]) >= 0, "Dispersão local incompatível com n.")
        else:
            _require(not row["local_sd_ddof"], "Um desvio ausente não pode receber divisor estatístico.")
    parameters = root / "comparison-parameters.csv"
    contracts = {}
    if parameters.is_file():
        with parameters.open(encoding="utf-8", newline="") as stream:
            contracts = {(r["paper_id"], r["dataset"]): r for r in csv.DictReader(stream)}
        for row in rows:
            contract = contracts.get((row["paper_id"], row["dataset"]))
            _require(
                contract is not None and int(contract["local_n_trials"]) == int(row["n_local"]),
                "O número de repetições difere do arquivo de parâmetros do protocolo.",
            )
    return rows, path, contracts


def _panel_contract(records):
    counts = {int(r["n_local"]) for r in records}
    seeds = {r["local_seed_values"] for r in records}
    divisors = {int(r["local_sd_ddof"]) if r["local_sd_ddof"] else None for r in records}
    _require(
        len(counts) == len(seeds) == len(divisors) == 1,
        "O painel mistura seeds, tamanhos de amostra ou definições de dispersão.",
    )
    return {
        "n": next(iter(counts)),
        "seeds": [int(x) for x in next(iter(seeds)).split(";")],
        "sd_ddof": next(iter(divisors)),
    }


def _spread_note(contracts):
    notes = []
    for label, contract in contracts.items():
        n, ddof = contract["n"], contract["sd_ddof"]
        if ddof is None:
            description = f"{n} seeds, sem desvio local disponível"
        else:
            kind = "populacional" if ddof == 0 else "amostral" if ddof == 1 else f"com ddof {ddof}"
            description = f"{n} seeds e barras de um desvio padrão {kind}, com ddof {ddof}"
        notes.append(f"{label}: {description}")
    return "; ".join(notes) + "."


def _published_pair(rows, path, parameters, paper, metric, show_title, show_footer):
    cara = paper == "cara_2024"
    selected = [
        r
        for r in rows
        if r["paper_id"] == paper
        and r["metric"] == metric
        and r["paper_table"] in ({"2", "3", "4"} if cara else {"2"})
    ]
    groups = DATASETS if cara else ("low", "high")
    policies = ["cara", "oracle_staleness", "frozen", "markov"] if cara else ["cara", "upf", "oracle"]
    if cara and metric == "strategy_scpe":
        policies.remove("oracle_staleness")
    fig, axes = _layout(2, show_title, show_footer, left=0.25)
    fig.subplots_adjust(right=0.79)
    plotted = []
    panel_contracts = {}
    for axis, group in zip(axes, groups, strict=True):
        records = [
            r for r in selected if r["dataset" if cara else "regime"] == group and r["policy"] in policies
        ]
        _require(
            {r["policy"] for r in records} == set(policies) and len(records) == len(policies),
            "Faltam valores publicados ou locais no painel comparativo.",
        )
        contract = _panel_contract(records)
        panel_contracts[
            DATASET_NAMES[group] if cara else f"α igual a {_plain(float(records[0]['alpha']))}"
        ] = contract
        _pair_points(axis, records, policies, metric, log=metric == "strategy_scpe")
        label = (
            DATASET_NAMES[group] if cara else f"Electricity: α igual a {_plain(float(records[0]['alpha']))}"
        )
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold", pad=5)
        xlabel = METRICS[metric][1] + ("; escala logarítmica" if metric == "strategy_scpe" else "")
        axis.set_xlabel(xlabel)
        plotted.append({"panel": group, "rows": records})
    _legend(fig, [], show_footer, sources=True)
    if cara:
        table = {"strategy_scpe": "2", "average_accuracy": "3", "retrains": "4"}[metric]
        caption = (
            f"Electricity acima e Airlines abaixo. Comparei {METRICS[metric][0].lower()} do arquivo "
            f"histórico com a Tabela {table} de {CARA_PAPER}, página 9 do PDF. As cores identificam "
            "as políticas, círculos cheios são médias locais e losangos vazios são os valores publicados. "
            "A coluna à direita informa as médias arredondadas, na mesma ordem dos marcadores. "
            "As tabelas publicadas não informam dispersão. "
            + _spread_note(panel_contracts)
            + " "
            + CARA_LIMIT
        )
        if parameters:
            prices = {
                d: int(parameters[(paper, d)]["local_positive_prices_in_CARA_summary"]) for d in DATASETS
            }
            caption += (
                " Preços positivos de κ nos três resumos: "
                + "; ".join(f"{DATASET_NAMES[d]}: {prices[d]}" for d in DATASETS)
                + ". Zero fica excluído."
            )
        if metric == "strategy_scpe":
            caption += " O eixo horizontal é logarítmico e não inclui um SCPE inventado para o Oracle."
        paper_title, paper_url, paper_doi = CARA_PAPER, CARA_DOI, CARA_DOI
        footer = "Cotejo histórico descritivo. Valores à direita seguem a ordem dos marcadores."
    else:
        table = "2"
        caption = (
            f"Electricity, {METRICS[metric][0].lower()}, com α igual a 0,1 acima e 0,9 abaixo. "
            f"Comparei os resultados históricos com a Tabela 2 de {UPF_PAPER}, página 8 do PDF "
            "final. As cores identificam políticas. Círculos cheios mostram médias locais. "
            + _spread_note(panel_contracts)
            + " Losangos "
            "vazios mostram médias publicadas, com o valor após ± transcrito como informado. "
            "O texto consultado não declara o tipo estatístico nem o divisor desse ±; ele não é "
            "rotulado como intervalo de confiança ou desvio padrão amostral. A coluna à direita "
            "informa as médias arredondadas, na mesma ordem dos marcadores. " + UPF_LIMIT
        )
        paper_title, paper_url, paper_doi = UPF_PAPER, UPF_URL, UPF_DOI
        footer = "Cotejo histórico descritivo. Consulte a legenda para a definição das barras."
    caption += " " + REPRESENTATIVE
    return _finish(
        fig,
        f"{METRICS[metric][0]}: referência {'CARA' if cara else 'UPF'} e histórico",
        caption,
        show_title=show_title,
        show_footer=show_footer,
        source_slot_id=f"comparativo-publicado-{'cara' if cara else 'upf'}",
        source_files=[path],
        metrics=[metric],
        footer=footer,
        source_slot_part=metric,
        paper_title=paper_title,
        paper_url=paper_url,
        paper_doi=paper_doi,
        doi_version="published_journal" if cara else "preprint",
        published_table=table,
        published_pdf_page=9 if cara else 8,
        local_panel_contracts=panel_contracts,
        published_spread_type="not_reported" if cara else "as_reported_type_and_ddof_unknown",
        plotted_data=plotted,
    )


def build_published_comparisons(papers_dir, *, show_title=True, show_footer=True) -> dict:
    """Return six descriptive historical comparisons, keeping all variants in the input CSV."""
    apply_style()
    rows, path, parameters = _comparison_records(papers_dir)
    figures = {}
    for paper, metrics in [
        ("cara_2024", ("strategy_scpe", "average_accuracy", "retrains")),
        ("upf_2025", ("cost", "average_accuracy", "retrains")),
    ]:
        for metric in metrics:
            name = f"comparativo-{paper}-{metric}"
            figures[name] = _published_pair(rows, path, parameters, paper, metric, show_title, show_footer)
    return figures
