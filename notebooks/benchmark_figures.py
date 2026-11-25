"""Readable historical INSECTS figures, calculated from the archived paired evidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from benchmark_figure_data import included_policies
from benchmark_plot_style import (
    BLUE,
    GRID,
    INK,
    MUTED,
    ORANGE,
    TEAL,
    apply_style,
    decorate,
    export_bundle,
)
from benchmark_plot_style import (
    POLICY_COLORS as CANONICAL_COLORS,
)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

POLICIES = included_policies("Logística")
MODELS = ("Hoeffding", "Logística")
DATASETS = ("Abrupto", "Recorrente")
POLICY_COLORS = dict(
    zip(
        POLICIES,
        (CANONICAL_COLORS[p] for p in ("frozen", "cara", "upf", "oracle", "adwin", "online")),
        strict=True,
    )
)
MARKERS = {"Hoeffding": "o", "Logística": "D"}
DELTA = LinearSegmentedColormap.from_list("historical_delta", [ORANGE, "#FAFBFC", TEAL])
SCOPE = "Histórico INSECTS: 66 runs completos, 22 configurações e seeds 42, 43 e 44. "
PAIR = "F1 macro e acurácia usam os mesmos eventos pareados por fluxo. "


def number(value, decimals=0):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _layout(show_title, show_footer, *, height=6.5, left=0.16, right=0.96, bottom=None):
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.2, height))
    fig.subplots_adjust(
        left=left,
        right=right,
        top=0.83 if show_title else 0.95,
        bottom=(0.24 if show_footer else 0.18) if bottom is None else bottom,
        hspace=0.68,
    )
    for ax, dataset in zip(axes, DATASETS, strict=True):
        ax.set_title(dataset, loc="left", fontweight="bold", pad=8)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, axes


def _policy_legend(fig, *, models=False, bottom=0.08):
    handles = [Line2D([], [], color=POLICY_COLORS[p], marker="o", linestyle="", label=p) for p in POLICIES]
    if models:
        handles += [Line2D([], [], color=INK, marker=MARKERS[m], linestyle="", label=m) for m in MODELS]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, bottom),
        ncol=3,
        frameon=False,
        columnspacing=1,
        handletextpad=0.4,
        labelspacing=0.45,
    )


def _model_legend(fig, bottom):
    fig.legend(
        handles=[
            Line2D([], [], color=c, marker=MARKERS[m], label=m)
            for m, c in zip(MODELS, (BLUE, ORANGE), strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, bottom),
        ncol=2,
        frameon=False,
    )


DEFINITIONS = {
    "regime_map": {
        "F1 macro": "Média do F1 das duas classes; maior é melhor.",
        "p.p.": "Diferença absoluta entre percentuais em relação ao Congelado do mesmo modelo.",
    },
    "detector_timeline": {
        "Acurácia": "Proporção de previsões corretas em cada janela; maior é melhor.",
        "Alarme": "Evento registrado pelo detector, sem equivalência automática com uma mudança real.",
        "Atualização": "Ação de atualização registrada; não é uma contagem de alarmes.",
    },
    "recovery_bars": {
        "Recuperação": (
            "Exemplos desde a fronteira anotada até duas janelas elegíveis consecutivas atingirem "
            "o alvo de acurácia."
        ),
        "n/N": (
            "Seeds que atingiram o alvo divididas pelas seeds avaliadas; valores não recuperados "
            "permanecem indisponíveis."
        ),
    },
    "quality_cost": {
        "F1 macro": "Média do F1 das duas classes nos eventos pareados; maior é melhor.",
        "Custo registrado": (
            "Estimativa registrada de CPU, RAM e armazenamento do run integral em milésimos de "
            "dólar, sem equivalência a fatura."
        ),
        "Atualizações": (
            "Online atualiza por exemplo; as demais políticas contam retreinamentos no run integral."
        ),
    },
    "paired_f1": {
        "F1 macro": "Média do F1 das duas classes na mesma coorte pareada; maior é melhor.",
        "Desvio padrão": (
            "Dispersão amostral entre as três seeds, com divisor n menos 1; não é intervalo de confiança."
        ),
    },
    "logistic_trajectories": {
        "F1 por janela": (
            "F1 macro recalculado em cada janela pareada e agregado entre seeds; não equivale ao "
            "F1 do fluxo inteiro."
        )
    },
    "quality_retention": {
        "F1 macro": "Média do F1 das duas classes nos eventos pareados; maior é melhor.",
        "Retenção": (
            "Acurácia média final nos probes de regimes anteriores do run integral; não é uma "
            "porcentagem de conhecimento preservado."
        ),
    },
    "paired_accuracy_f1": {
        "Acurácia": "Proporção de previsões corretas na coorte pareada.",
        "F1 macro": "Média do F1 das duas classes na mesma coorte; a diferença para acurácia é descritiva.",
    },
    "upf_class_recall": {
        "Recall": "Fração dos exemplos de uma classe classificados corretamente; maior é melhor.",
        "Recall negativo": (
            "Valor derivado de acurácia, prevalência e recall positivo da mesma coorte, sem "
            "reconstruir dispersão por seed."
        ),
    },
}


def _finish(fig, title, caption, slot, show_title, show_footer, **extra):
    descriptions = extra.pop("metric_descriptions", DEFINITIONS[slot])
    return decorate(
        fig,
        title,
        SCOPE + caption,
        show_title=show_title,
        show_footer=show_footer,
        footer="Três seeds históricas. Escopos e definições completos na legenda.",
        source_slot_id=slot,
        source_family="historical-insects",
        metric_descriptions=descriptions,
        **extra,
    )


def _dataset_rows(frame, dataset):
    return frame.loc[frame.dataset.eq(dataset)]


def regime_map(evidence, *, show_title=True, show_footer=True, dataset="Abrupto"):
    """A full-size heatmap for one stream, retaining model-specific configurations."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 6.5))
    fig.subplots_adjust(
        left=0.31, right=0.89, top=0.82 if show_title else 0.95, bottom=0.17 if show_footer else 0.13
    )
    rows = _dataset_rows(evidence.regimes, dataset)
    labels = [(m, p) for m in MODELS for p in included_policies(m)]
    regimes = sorted(rows.regime.unique())
    values = np.full((len(labels), len(regimes)), np.nan)
    absolute = values.copy()
    for i, (model, policy) in enumerate(labels):
        ref = rows.loc[rows.modelo.eq(model) & rows.politica.eq("Congelado")].set_index("regime")
        part = rows.loc[rows.modelo.eq(model) & rows.politica.eq(policy)].set_index("regime")
        for j, regime in enumerate(regimes):
            if regime in part.index and regime in ref.index:
                absolute[i, j] = part.loc[regime, "f1"] * 100
                values[i, j] = absolute[i, j] - ref.loc[regime, "f1"] * 100
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    image = ax.imshow(values, aspect="auto", cmap=DELTA, norm=norm)
    for i, (_, policy) in enumerate(labels):
        for j in range(len(regimes)):
            v = absolute[i, j] if policy == "Congelado" else values[i, j]
            text = "ND" if not np.isfinite(v) else number(v, 1) + ("%" if policy == "Congelado" else "")
            ax.text(j, i, text, ha="center", va="center", color="white" if abs(values[i, j]) > 42 else INK)
    ax.set_yticks(range(len(labels)), [f"{'H' if m == 'Hoeffding' else 'Log'} · {p}" for m, p in labels])
    ax.set_xticks(range(len(regimes)), [f"R{x}" for x in regimes])
    ax.set_xlabel("Regime anotado")
    ax.set_title(dataset, loc="left", fontweight="bold")
    cb = fig.colorbar(image, cax=fig.add_axes([0.915, 0.22, 0.022, 0.44]), ticks=[-60, 0, 60])
    cb.ax.set_title("p.p.", fontsize=11, pad=10)
    fractions = rows.groupby("regime")[["positivos", "avaliados"]].first()
    prevalence = "; ".join(
        f"R{r}: {number(v.positivos / v.avaliados * 100, 2)}%" for r, v in fractions.iterrows()
    )
    caption = (
        PAIR
        + (
            f"{dataset}. Congelado mostra F1 absoluto; as outras células mostram a diferença em pontos "
            "percentuais para o Congelado do mesmo modelo. H identifica Hoeffding e Log identifica "
            "regressão logística. Regimes com bases pareadas incompletas permanecem ND. "
        )
        + "Proporção positiva na coorte por regime: "
        + prevalence
        + "."
    )
    return _finish(
        fig,
        "F1 por regime em relação ao Congelado",
        caption,
        "regime_map",
        show_title,
        show_footer,
        metric_descriptions={
            "F1 macro": "Média do F1 das duas classes; maior é melhor.",
            "p.p.": "Diferença absoluta entre percentuais, sem dividir pela referência.",
        },
    )


def detector_timeline(evidence, *, show_title=True, show_footer=True):
    fig, axes = _layout(show_title, show_footer, bottom=0.20 if show_footer else 0.14)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for model, color in zip(MODELS, (BLUE, ORANGE), strict=True):
            records = evidence.detector.loc[
                evidence.detector.dataset.eq(dataset) & evidence.detector.modelo.eq(model)
            ]
            series = []
            for run in records.run:
                timeline = evidence.drift_artifacts[run]["detection_timeline"]
                markers = [p["x"] for p in timeline["markers"] if p["kind"] == "changepoint"]
                if markers != evidence.manifests[dataset]["changepoints"]:
                    raise ValueError("Marcos históricos divergentes do manifesto.")
                series.append(dict(next(s["points"] for s in timeline["series"] if s["name"] == "accuracy")))
            common = sorted(set.intersection(*(set(s) for s in series)))
            matrix = np.array([[s[x] for x in common] for s in series]) * 100
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1) if len(series) > 1 else np.zeros(len(common))
            ax.plot(np.array(common) / 1000, mean, color=color, lw=1.6, label=model)
            ax.fill_between(np.array(common) / 1000, mean - sd, mean + sd, color=color, alpha=0.12)
        for change in evidence.manifests[dataset]["changepoints"]:
            ax.axvline(change / 1000, color=MUTED, lw=0.7, ls=":")
        ax.set(ylabel="Acurácia (%)", xlabel="Posição no fluxo (mil exemplos)", ylim=(0, 103))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    _model_legend(fig, 0.075 if show_footer else 0.015)
    return _finish(
        fig,
        "Acurácia ao longo do fluxo com ADWIN",
        (
            "A linha é a média e a faixa indica um desvio padrão amostral nas posições presentes "
            "nas três seeds. Linhas verticais marcam fronteiras anotadas. A janela avalia a "
            "acurácia; não representa uma coorte de teste independente."
        ),
        "detector_timeline",
        show_title,
        show_footer,
    )


def detector_events(evidence, *, show_title=True, show_footer=True):
    fig, axes = _layout(show_title, show_footer, bottom=0.24 if show_footer else 0.18)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        manifest = evidence.manifests[dataset]
        bins = np.r_[np.arange(0, manifest["samples"], 5000), manifest["samples"]]
        for model, color in zip(MODELS, (BLUE, ORANGE), strict=True):
            runs = evidence.detector.loc[
                evidence.detector.dataset.eq(dataset) & evidence.detector.modelo.eq(model), "run"
            ]
            for event, style in (("alarm", "-"), ("update", "--")):
                counts = [
                    np.histogram(
                        [
                            x["x"]
                            for x in evidence.drift_artifacts[run]["detection_timeline"]["markers"]
                            if x["kind"] == event
                        ],
                        bins,
                    )[0]
                    for run in runs
                ]
                ax.step(bins[1:] / 1000, np.mean(counts, axis=0), where="pre", color=color, ls=style, lw=1.5)
        ax.set(xlabel="Posição no fluxo (mil exemplos)", ylabel="Eventos por 5 mil")
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    handles = [Line2D([], [], color=c, label=m) for m, c in zip(MODELS, (BLUE, ORANGE), strict=True)] + [
        Line2D([], [], color=INK, ls=s, label=label) for label, s in (("Alarme", "-"), ("Atualização", "--"))
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.075 if show_footer else 0.015),
        ncol=2,
        frameon=False,
    )
    return _finish(
        fig,
        "Alarmes e atualizações registrados pelo ADWIN",
        (
            "Contagem média das três seeds por bloco de 5 mil exemplos. O último bloco pode conter "
            "menos exemplos. Um alarme é um evento do detector e uma atualização é uma ação "
            "registrada. Neste histórico, posições e contagens de alarmes e atualizações coincidem, "
            "então as curvas se sobrepõem apesar do significado distinto dos eventos."
        ),
        "detector_timeline",
        show_title,
        show_footer,
    )


def recovery_bars(evidence, *, show_title=True, show_footer=True):
    fig, axes = _layout(show_title, show_footer, bottom=0.22 if show_footer else 0.16)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        changes = evidence.manifests[dataset]["changepoints"]
        for j, (model, color) in enumerate(zip(MODELS, (BLUE, ORANGE), strict=True)):
            runs = evidence.detector.loc[
                evidence.detector.dataset.eq(dataset) & evidence.detector.modelo.eq(model), "run"
            ]
            values = {p: [] for p in changes}
            for run in runs:
                recovery = evidence.drift_artifacts[run]["recovery"]
                points = [int(label.split("@")[1]) for label in recovery["labels"]]
                if points != changes:
                    raise ValueError("Fronteiras de recuperação divergentes.")
                for point, value in zip(points, recovery["values"], strict=True):
                    if value is not None:
                        values[point].append(value)
            for i, point in enumerate(changes):
                vals = values[point]
                y = i + (j - 0.5) * 0.28
                if vals:
                    ax.plot(np.mean(vals) / 1000, y, marker=MARKERS[model], color=color, ms=6)
                    ax.annotate(
                        f"{len(vals)}/{len(runs)}",
                        (np.mean(vals) / 1000, y),
                        xytext=(6, 0),
                        textcoords="offset points",
                        va="center",
                    )
                else:
                    ax.text(0, y, "ND", va="center")
        ax.set_yticks(range(len(changes)), [number(v) for v in changes])
        ax.invert_yaxis()
        ax.set(xlabel="Exemplos até recuperar (mil)", ylabel="Fronteira anotada")
        ax.set_xlim(0, max(ax.get_xlim()[1] * 1.28, 1))
    _model_legend(fig, 0.075 if show_footer else 0.015)
    return _finish(
        fig,
        "Recuperação da acurácia após fronteiras anotadas",
        (
            "Somente ADWIN. Alvo: duas janelas elegíveis consecutivas com acurácia pelo menos "
            "igual à última janela pré-mudança menos 5 p.p. O ponto é a média dos atrasos "
            "disponíveis e n/N explicita quantas seeds atingiram o alvo. O início da contagem é a "
            "fronteira anotada, sem exigir queda inicial ou alarme pareado."
        ),
        "recovery_bars",
        show_title,
        show_footer,
    )


def _scatter(evidence, kind, show_title, show_footer):
    fig, axes = _layout(show_title, show_footer, left=0.17, bottom=0.29 if show_footer else 0.23)
    merged = evidence.paired.merge(evidence.runs, on=["dataset", "modelo", "politica"], suffixes=("", "_run"))
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for row in _dataset_rows(merged, dataset).itertuples():
            y = row.custo_registrado * 1000 if kind == "cost" else row.retencao * 100
            if np.isfinite(y):
                if np.isfinite(row.f1_dp):
                    ax.errorbar(
                        row.f1 * 100,
                        y,
                        xerr=row.f1_dp * 100,
                        fmt="none",
                        color=POLICY_COLORS[row.politica],
                        capsize=2,
                        linewidth=1.1,
                        zorder=2,
                    )
                ax.scatter(
                    row.f1 * 100,
                    y,
                    c=POLICY_COLORS[row.politica],
                    marker=MARKERS[row.modelo],
                    s=49,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )
        ax.set_xlabel("F1 macro pareado (%)")
        if kind == "cost":
            ax.set_ylabel("Custo registrado (mUS$)")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 5, 10, 20, 50]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: number(v, 1)))
        else:
            ax.set_ylabel("Retenção final (%)")
            ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", color=GRID, lw=0.6)
    _policy_legend(fig, models=True, bottom=0.065 if show_footer else 0.005)
    title = (
        "Qualidade preditiva e custo registrado" if kind == "cost" else "Qualidade preditiva e retenção final"
    )
    caption = (
        PAIR
        + "Barras horizontais mostram um desvio padrão amostral do F1 entre três seeds. "
        + (
            (
                "O custo do run integral é a estimativa de CPU, RAM e armazenamento registrada em "
                "milésimos de US$. Escala vertical logarítmica. Valores de custo não são estimativas "
                "causais de uma única operação."
            )
            if kind == "cost"
            else (
                "A retenção é a média final de acurácia nos probes dos regimes anteriores do run "
                "integral. Não representa a proporção de conhecimento preservado nem uma medição na "
                "coorte pareada."
            )
        )
    )
    return _finish(
        fig,
        title,
        caption,
        "quality_cost" if kind == "cost" else "quality_retention",
        show_title,
        show_footer,
    )


def quality_cost(evidence, *, show_title=True, show_footer=True):
    return _scatter(evidence, "cost", show_title, show_footer)


def quality_retention(evidence, *, show_title=True, show_footer=True):
    return _scatter(evidence, "retention", show_title, show_footer)


def update_counts(evidence, *, show_title=True, show_footer=True):
    fig, axes = _layout(show_title, show_footer, left=0.23, bottom=0.22 if show_footer else 0.16)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for j, model in enumerate(MODELS):
            part = _dataset_rows(evidence.runs, dataset)
            part = part.loc[part.modelo.eq(model)].set_index("politica")
            for i, policy in enumerate(POLICIES):
                if policy in part.index:
                    ax.plot(
                        part.loc[policy, "atualizacoes"],
                        i + (j - 0.5) * 0.22,
                        marker=MARKERS[model],
                        color=POLICY_COLORS[policy],
                        ms=6,
                    )
        ax.set_yticks(range(len(POLICIES)), POLICIES)
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=1)
        maximum = _dataset_rows(evidence.runs, dataset).atualizacoes.max()
        ax.set_xlim(-0.15, maximum * 2)
        ax.xaxis.set_major_locator(FixedLocator([0, 1, 10, 1000, 100000]))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: number(v / 1000) + " mil" if v >= 1000 else number(v))
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Atualizações no run integral (log simétrica)")
    _model_legend(fig, 0.075 if show_footer else 0.015)
    return _finish(
        fig,
        "Número de atualizações por política",
        (
            "O Online atualiza por exemplo e só integra a regressão logística histórica. As demais "
            "políticas registram retreinamentos. Escala logarítmica simétrica preserva a contagem "
            "zero. A figura não inclui o Online na campanha F109."
        ),
        "quality_cost",
        show_title,
        show_footer,
    )


def paired_f1(evidence, *, show_title=True, show_footer=True):
    fig, axes = _layout(show_title, show_footer, left=0.23, bottom=0.22 if show_footer else 0.16)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for j, model in enumerate(MODELS):
            part = _dataset_rows(evidence.paired, dataset)
            part = part.loc[part.modelo.eq(model)].set_index("politica")
            for i, policy in enumerate(POLICIES):
                if policy in part.index:
                    row = part.loc[policy]
                    sd = row.f1_dp
                    ax.errorbar(
                        row.f1 * 100,
                        i + (j - 0.5) * 0.22,
                        xerr=sd * 100 if np.isfinite(sd) else None,
                        fmt=MARKERS[model],
                        color=POLICY_COLORS[policy],
                        ms=5,
                        capsize=2,
                        lw=1.2,
                    )
        ax.set_yticks(range(len(POLICIES)), POLICIES)
        ax.invert_yaxis()
        ax.set_xlabel("F1 macro pareado (%)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    _model_legend(fig, 0.075 if show_footer else 0.015)
    return _finish(
        fig,
        "F1 macro nos mesmos eventos de avaliação",
        PAIR
        + (
            "Pontos são médias e barras representam um desvio padrão amostral entre três seeds, "
            "quando disponível. Ausência de Online/Hoeffding é mantida. Barra de comprimento zero "
            "significa desvio zero observado."
        ),
        "paired_f1",
        show_title,
        show_footer,
    )


def logistic_trajectories(evidence, *, show_title=True, show_footer=True):
    fig, axes = _layout(show_title, show_footer, bottom=0.26 if show_footer else 0.20)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        rows = _dataset_rows(evidence.windows, dataset)
        rows = rows.loc[rows.modelo.eq("Logística")]
        for policy in POLICIES:
            part = rows.loc[rows.politica.eq(policy)].sort_values("passo")
            ax.plot(part.passo / 1000, part.f1 * 100, color=POLICY_COLORS[policy], lw=1.5, label=policy)
            ax.fill_between(
                part.passo / 1000,
                np.clip((part.f1 - part.dp) * 100, 0, 100),
                np.clip((part.f1 + part.dp) * 100, 0, 100),
                color=POLICY_COLORS[policy],
                alpha=0.13,
                linewidth=0,
            )
        for change in evidence.manifests[dataset]["changepoints"]:
            ax.axvline(change / 1000, color=MUTED, lw=0.7, ls=":")
        ax.set(xlabel="Posição no fluxo (mil exemplos)", ylabel="F1 macro por janela (%)", ylim=(0, 103))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    _policy_legend(fig, bottom=0.07 if show_footer else 0.01)
    return _finish(
        fig,
        "Trajetórias de F1 na regressão logística",
        (
            "Curvas de F1 macro calculadas nas janelas de cada política, nas posições comuns às três seeds. "
            "Faixas mostram um desvio padrão amostral, limitado ao intervalo de 0 a 100%. "
            "As linhas verticais indicam fronteiras anotadas, sem resumir todo o drift "
            "incremental do fluxo recorrente. O Online pertence apenas a este escopo histórico."
        ),
        "logistic_trajectories",
        show_title,
        show_footer,
    )


def paired_accuracy_f1(evidence, *, show_title=True, show_footer=True, dataset="Abrupto"):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 6.5))
    fig.subplots_adjust(
        left=0.32, right=0.96, top=0.82 if show_title else 0.95, bottom=0.22 if show_footer else 0.16
    )
    part = _dataset_rows(evidence.paired, dataset)
    for i, row in enumerate(part.sort_values(["modelo", "politica"]).itertuples()):
        ax.plot([row.f1 * 100, row.acuracia * 100], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter(row.acuracia * 100, i, color=BLUE, marker="o", s=38, zorder=2)
        ax.scatter(row.f1 * 100, i, color=ORANGE, marker="D", s=35, zorder=2)
    labels = [
        f"{'H' if r.modelo == 'Hoeffding' else 'Log'} · {r.politica}"
        for r in part.sort_values(["modelo", "politica"]).itertuples()
    ]
    prevalence = part.positivos.iloc[0] / part.avaliados.iloc[0]
    majority = max(prevalence, 1 - prevalence) * 100
    ax.axvline(majority, color=MUTED, linestyle=":", linewidth=1.4)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Resultado pareado (%)")
    ax.set_title(dataset, loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", ls="", label="Acurácia"),
            Line2D([], [], color=ORANGE, marker="D", ls="", label="F1 macro"),
            Line2D([], [], color=MUTED, ls=":", label="Classe majoritária"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.075 if show_footer else 0.015),
        ncol=2,
        frameon=False,
    )
    return _finish(
        fig,
        "Acurácia e F1 macro nos mesmos eventos",
        PAIR
        + (
            f"{dataset}. Cada segmento liga as duas métricas da mesma configuração. H significa "
            "Hoeffding e Log significa regressão logística. A distância entre pontos descreve a "
            "diferença entre acurácia e F1 macro, sem servir como teste estatístico. "
        )
        + f"A referência majoritária da mesma coorte é {number(majority, 6)}% de acurácia.",
        "paired_accuracy_f1",
        show_title,
        show_footer,
    )


def upf_class_recall(evidence, *, show_title=True, show_footer=True):
    fig, axes = _layout(show_title, show_footer, left=0.20, bottom=0.22 if show_footer else 0.16)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        part = _dataset_rows(evidence.paired, dataset)
        part = part.loc[part.politica.eq("UPF")]
        for j, row in enumerate(part.itertuples()):
            p = row.positivos / row.avaliados
            negative = (row.acuracia - p * row.recall) / (1 - p)
            ax.plot([row.recall * 100, negative * 100], [j, j], color=GRID, lw=3)
            ax.scatter(row.recall * 100, j, color=ORANGE, s=48, marker="D")
            ax.scatter(negative * 100, j, color=BLUE, s=48, marker="o")
        ax.set_yticks(range(len(part)), part.modelo)
        ax.invert_yaxis()
        ax.set_xlabel("Recall por classe (%)")
        ax.set_xlim(-3, 103)
    fig.legend(
        handles=[
            Line2D([], [], color=ORANGE, marker="D", ls="", label="Positiva"),
            Line2D([], [], color=BLUE, marker="o", ls="", label="Negativa"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.54, 0.075 if show_footer else 0.015),
        ncol=2,
        frameon=False,
    )
    return _finish(
        fig,
        "Recall por classe da política UPF",
        PAIR
        + (
            "Recall positivo é o valor arquivado. Recall negativo é derivado da identidade "
            "acurácia = prevalência × recall positivo + (1 − prevalência) × recall negativo, "
            "usando a prevalência da mesma coorte. Pontos descrevem classes diferentes, sem desvio "
            "padrão reconstruído."
        ),
        "upf_class_recall",
        show_title,
        show_footer,
    )


def build_historical_figures(evidence, *, show_title=True, show_footer=True):
    flags = {"show_title": show_title, "show_footer": show_footer}
    figures = {"historico-regimes-" + d.lower(): regime_map(evidence, dataset=d, **flags) for d in DATASETS}
    for name, function in (
        ("adwin-acuracia", detector_timeline),
        ("adwin-eventos", detector_events),
        ("adwin-recuperacao", recovery_bars),
        ("qualidade-custo", quality_cost),
        ("atualizacoes", update_counts),
        ("f1-pareado", paired_f1),
        ("trajetorias-logistica", logistic_trajectories),
        ("qualidade-retencao", quality_retention),
        ("upf-recall", upf_class_recall),
    ):
        figures["historico-" + name] = function(evidence, **flags)
    for d in DATASETS:
        figures["historico-acuracia-f1-" + d.lower()] = paired_accuracy_f1(evidence, dataset=d, **flags)
    return figures


def export_figures(figures, output):
    return export_bundle(figures, output)
