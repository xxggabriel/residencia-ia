"""Publication figures from validated benchmark tables, without changing raw evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from benchmark_athena_data import DATASETS, POLICIES, SEEDS, require
from benchmark_plot_style import (
    GRID,
    POLICY_COLORS,
    apply_style,
    decorate,
    export_bundle,
    validate_figure,
)
from matplotlib.ticker import FuncFormatter, MaxNLocator

SHORT = {
    "frozen": "Congelado",
    "drift": "Drift",
    "drift_adwin": "ADWIN",
    "drift_ddm": "DDM",
    "incremental": "Incremental",
    "periodic": "Periódica",
    "cara": "CARA",
    "cara_cumulative": "CARA cumulativa",
    "cara_periodic": "CARA periódica",
    "upf": "UPF",
    "oracle": "Oracle",
}
LABELS = SHORT
TITLES = {
    "electricity": "Electricity",
    "airlines": "Airlines",
    "insects_abrupt_imbalanced": "INSECTS abrupto",
    "insects_incremental_abrupt_reoccurring_imbalanced": "INSECTS incremental, abrupto e recorrente",
}
TEMPORAL_TITLES = TITLES
COLORS = POLICY_COLORS
FALLBACK_COLORS = ("#176B91", "#087F78", "#C55D32", "#608AAB", "#335671", "#718390")
HIDDEN_VARIANTS = frozenset(("cara_cumulative", "cara_periodic"))
ALIAS_TOLERANCE = 1e-12


def configure_arial():
    """Apply the common11-point publication style and return its font receipt."""
    return apply_style()


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _same(first, second):
    if not _finite(first) or not _finite(second):
        return not _finite(first) and not _finite(second)
    return math.isclose(float(first), float(second), rel_tol=0, abs_tol=ALIAS_TOLERANCE)


def _names(values):
    return tuple(row["name"] if isinstance(row, dict) else row for row in values)


def _dimensions(evidence, datasets=None, policies=None):
    provenance = evidence.provenance
    observed_datasets = tuple(evidence.primary["dataset"].drop_duplicates())
    observed_policies = tuple(evidence.primary["policy"].drop_duplicates())
    declared_datasets = provenance.get("datasets")
    declared_policies = provenance.get("policies")
    dataset_order = (
        _names(declared_datasets)
        if declared_datasets
        else tuple(
            [d for d in DATASETS if d in observed_datasets]
            + [d for d in observed_datasets if d not in DATASETS]
        )
    )
    policy_order = (
        _names(declared_policies)
        if declared_policies
        else tuple(
            [p for p in POLICIES if p in observed_policies]
            + [p for p in observed_policies if p not in POLICIES]
        )
    )
    chosen_datasets = _names(datasets) if datasets is not None else dataset_order
    source_policies = _names(policies) if policies is not None else policy_order
    require(chosen_datasets and set(chosen_datasets) <= set(observed_datasets), "Dataset sem evidência.")
    require(source_policies and set(source_policies) <= set(observed_policies), "Política sem evidência.")
    seeds = tuple(int(seed) for seed in provenance.get("seeds", SEEDS))
    require(seeds and len(set(seeds)) == len(seeds), "Seeds não declaradas ou repetidas.")
    visible = tuple(p for p in source_policies if p not in HIDDEN_VARIANTS)
    require(visible, "A seleção não contém políticas visíveis.")
    if set(source_policies) & HIDDEN_VARIANTS:
        require("cara" in source_policies, "As variantes CARA exigem a base CARA na comparação.")
    return chosen_datasets, source_policies, visible, seeds


def _dataset_label(evidence, dataset):
    metadata = evidence.provenance.get("dataset_metadata", {})
    if isinstance(metadata, list):
        metadata = {row["name"]: row for row in metadata}
    return metadata.get(dataset, {}).get("display_name", TITLES.get(dataset, dataset))


def _decimal(value, places=3):
    return f"{float(value):.{places}f}".replace(".", ",")


def _quality_equivalent(evidence, dataset, first, second, seeds):
    frame = evidence.primary
    for seed in seeds:
        values = []
        for policy in (first, second):
            selected = frame.loc[
                (frame["dataset"] == dataset) & (frame["policy"] == policy) & (frame["seed"] == seed)
            ]
            require(not selected.empty and not selected.metric.duplicated().any(), "Qualidade incompleta.")
            values.append(dict(zip(selected.metric, selected.value, strict=True)))
        if values[0].keys() != values[1].keys():
            return False
        if not all(_same(value, values[1][metric]) for metric, value in values[0].items()):
            return False
    return True


def _equivalent(series, dataset, first, second, seeds=SEEDS):
    """Compare every metric and window boundary separately in every declared seed."""
    for seed in seeds:
        a, b = series.get((dataset, first, seed)), series.get((dataset, second, seed))
        if a is None or b is None or len(a) != len(b):
            return False
        for left, right in zip(a, b, strict=True):
            if left.keys() != right.keys():
                return False
            if any(left[k] != right[k] for k in ("window", "start_seq", "end_seq", "n")):
                return False
            metrics = set(left) - {"window", "start_seq", "end_seq", "n"}
            if not all(_same(left[metric], right[metric]) for metric in metrics):
                return False
    return True


def _cara_equivalence(evidence, datasets, policies, seeds):
    proof = {}
    for dataset in datasets:
        proof[dataset] = {}
        for variant in sorted(set(policies) & HIDDEN_VARIANTS):
            quality = _quality_equivalent(evidence, dataset, "cara", variant, seeds)
            temporal = _equivalent(evidence.series, dataset, "cara", variant, seeds)
            require(quality and temporal, "A variante CARA difere da base; não pode ser ocultada.")
            proof[dataset][variant] = {"all_quality_metrics": quality, "all_temporal_metrics": temporal}
    return {"tolerance": ALIAS_TOLERANCE, "seeds": list(seeds), "datasets": proof}


def _ranking_note(evidence):
    if evidence.provenance.get("release") == "f1092ee0232a6d16504b3a2211a037c41a02068a":
        return "A AP positiva foi estimada por histograma de 4.096 bins."
    return "A definição e o estimador de AP são os declarados no contrato das tabelas."


def _metric_records(frame, dataset, metric, visible):
    """Use unavailable display rows for missing measures, without modifying source tables."""
    records = frame.loc[(frame.dataset == dataset) & (frame.metric == metric)].set_index("policy")
    require(records.index.is_unique, "Agregado de métrica repetido.")
    return records.reindex(visible)


def _dot_axes(show_title, show_footer):
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.5), dpi=300)
    fig.subplots_adjust(
        left=0.235,
        right=0.865,
        top=0.85 if show_title else 0.96,
        bottom=0.205 if show_footer else 0.115,
        hspace=0.60,
    )
    return fig, axes


def _dots(ax, records, visible, *, unit, upper=None, expected_n=3):
    require(records.index.is_unique and set(visible) <= set(records.index), "Agregado ausente ou repetido.")
    if upper is None:
        upper = max(
            (
                float(records.loc[p, "mean"])
                + (float(records.loc[p, "sample_sd"]) if _finite(records.loc[p, "sample_sd"]) else 0)
                for p in visible
                if _finite(records.loc[p, "mean"])
            ),
            default=1.0,
        )
        upper = max(upper * 1.08, 1.0)
    for ordinal, policy in enumerate(visible):
        row = records.loc[policy]
        color = COLORS.get(policy, FALLBACK_COLORS[ordinal % len(FALLBACK_COLORS)])
        if ordinal % 2 == 0:
            ax.axhspan(ordinal - 0.48, ordinal + 0.48, color="#F4F7F9", zorder=0)
        if not _finite(row["mean"]):
            text = "ND"
        else:
            mean = float(row["mean"])
            deviation = float(row["sample_sd"]) if _finite(row["sample_sd"]) else None
            ax.errorbar(
                mean,
                ordinal,
                xerr=deviation,
                fmt="o",
                color=color,
                markersize=5.5,
                capsize=3,
                elinewidth=1.4,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
            if unit == "score":
                text = _decimal(mean)
            elif unit == "count" and abs(mean - round(mean)) < 1e-10:
                text = str(round(mean))
            elif mean >= 1000:
                text = f"{mean:,.0f}".replace(",", ".")
            else:
                text = _decimal(mean, 2 if mean < 10 else 1)
            if int(row["n_defined"]) != expected_n:
                text += f" (n={int(row['n_defined'])})"
        ax.text(
            1.025,
            ordinal,
            text,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=11,
            color=color,
            fontweight="bold" if policy == "cara" else "normal",
            clip_on=False,
        )
    ax.set_yticks(np.arange(len(visible)), [SHORT.get(p, p) for p in visible], fontsize=11)
    ax.set_ylim(len(visible) - 0.55, -0.55)
    ax.set_xlim(-upper * 0.015 if unit != "score" else 0, upper)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=11, length=3)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    if unit == "score":
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _decimal(value, 2)))
    else:
        ticks = MaxNLocator(nbins=4, integer=unit == "count").tick_values(0, upper)
        ax.set_xticks([tick for tick in ticks if 0 <= tick <= upper])
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: (
                    f"{value:,.0f}".replace(",", ".")
                    if upper >= 10 or unit == "count"
                    else _decimal(value, 1)
                )
            )
        )


def quality_figures(evidence, *, show_title=True, show_footer=True, datasets=None, policies=None):
    """Show quality on the common cohort; raw11-policy evidence stays intact."""
    configure_arial()
    chosen, source, visible, seeds = _dimensions(evidence, datasets, policies)
    proof = _cara_equivalence(evidence, chosen, source, seeds)
    figures = {}
    for dataset in chosen:
        label = _dataset_label(evidence, dataset)
        fig, axes = _dot_axes(show_title, show_footer)
        for ax, metric, xlabel in zip(
            axes, ("macro_f1", "pr_auc"), ("F1 macro", "Precisão média (AP)"), strict=True
        ):
            records = _metric_records(evidence.quality, dataset, metric, visible)
            _dots(ax, records, visible, unit="score", upper=1, expected_n=len(seeds))
            ax.set_xlabel(xlabel, fontsize=11, labelpad=5)
        axes[1].set_xlabel("Precisão média (AP)", fontsize=11, labelpad=5)
        caption = (
            f"F1 macro e precisão média (AP) na coorte comum após calibração de {label}. "
            f"Cada ponto é a média de {len(seeds)} seeds; "
            "as barras representam DP amostral, com ddof 1, não intervalo de confiança. "
            "Os valores à direita são as médias; ND indica indisponibilidade. "
            + _ranking_note(evidence)
            + (
                " CARA representa somente a política base. As variantes cumulativa e "
                "periódica foram omitidas da visualização "
                "após conferir todas as métricas de qualidade e todas as janelas em "
                "cada seed, com tolerância de 1e-12."
                if set(source) & HIDDEN_VARIANTS
                else ""
            )
            + " As tabelas completas conservam todas as políticas e repetições. Se o "
            "DP for zero, a barra coincide com o ponto."
        )
        decorate(
            fig,
            f"{label}: qualidade preditiva",
            caption,
            show_title=show_title,
            show_footer=show_footer,
            footer=f"Média e DP amostral de {len(seeds)} seeds; DP não é intervalo de confiança.",
            metric_descriptions={
                "F1 macro": "Média não ponderada do F1 das classes.",
                "AP": "Precisão média da classe positiva; " + _ranking_note(evidence),
            },
            dataset=dataset,
            source_slot_id="a4-quality-" + dataset,
            figure_family="quality",
            visible_policies=list(visible),
            source_policies=list(source),
            seeds=list(seeds),
            quality_scope="common_postcalibration",
            cara_equivalence=proof,
            raw_evidence_modified=False,
        )
        validate_figure(fig)
        figures["a4-quality-" + dataset] = fig
    return figures


def resource_figures(evidence, *, show_title=True, show_footer=True, datasets=None, policies=None):
    """Show the base policy's own physical work; never pool cached variants."""
    configure_arial()
    chosen, source, visible, seeds = _dimensions(evidence, datasets, policies)
    figures = {}
    for dataset in chosen:
        label = _dataset_label(evidence, dataset)
        fig, axes = _dot_axes(show_title, show_footer)
        for ax, metric, xlabel, unit in zip(
            axes,
            ("actual_total_fit_seconds", "update_count"),
            ("Treino físico total (s)", "Atualizações após o modelo inicial"),
            ("seconds", "count"),
            strict=True,
        ):
            records = _metric_records(evidence.resources, dataset, metric, visible)
            _dots(ax, records, visible, unit=unit, expected_n=len(seeds))
            ax.set_xlabel(xlabel, fontsize=11, labelpad=5)
        variant_note = (
            " CARA representa estritamente a base: as variantes cumulativa e periódica não foram "
            "combinadas ou usadas como substitutas, pois seus recursos físicos não são equivalentes."
            if set(source) & HIDDEN_VARIANTS
            else ""
        )
        hardware = (
            " Nesta rodada, tarefas de 2 vCPU e 4 GiB executaram em modelos distintos de CPU."
            if evidence.provenance.get("release") == "f1092ee0232a6d16504b3a2211a037c41a02068a"
            else " Os tempos dependem do hardware e do ambiente declarados no contrato."
        )
        caption = (
            f"Treino físico total e atualizações implantadas na execução integral de {label}. "
            f"Pontos e valores à direita indicam médias de {len(seeds)} seeds; "
            "barras indicam DP amostral, ddof 1, não IC. "
            "O total físico exclui evidência reutilizada e já inclui "
            "materializações: esse subconjunto não deve ser somado novamente. "
            "O custo de execução própria de uma política cached não é seu custo "
            "autônomo, porque depende da construção CARA anterior."
            + variant_note
            + hardware
            + " Não se somam custos lógicos herdados, esforço físico e estimativas "
            "de referência; estimativa On-Demand não é faturamento observado."
        )
        decorate(
            fig,
            f"{label}: esforço físico e adaptação",
            caption,
            show_title=show_title,
            show_footer=show_footer,
            footer=(
                f"Execução integral; média e DP amostral de {len(seeds)} seeds, sem somar custos herdados."
            ),
            metric_descriptions={
                "Treino físico total": "Soma dos tempos dos ajustes realmente executados, incluindo "
                "materializações e excluindo evidência reutilizada.",
                "Atualizações": "Mudanças implantadas após o modelo inicial; não é a contagem de todos "
                "os ajustes candidatos.",
            },
            dataset=dataset,
            source_slot_id="a4-resources-" + dataset,
            figure_family="resources",
            visible_policies=list(visible),
            source_policies=list(source),
            seeds=list(seeds),
            resource_scope="full_evaluation",
            cara_resource_representative="cara" if "cara" in visible else None,
            cached_resource_costs_pooled=False,
            raw_evidence_modified=False,
        )
        validate_figure(fig)
        figures["a4-resources-" + dataset] = fig
    return figures


def aliases(series, *, datasets=None, policies=None, seeds=None):
    """Group only visible policies with equal per-seed metrics and aligned windows."""
    observed_datasets = tuple(dict.fromkeys(key[0] for key in series))
    observed_policies = tuple(dict.fromkeys(key[1] for key in series))
    chosen = (
        tuple(datasets)
        if datasets is not None
        else tuple(
            [d for d in DATASETS if d in observed_datasets]
            + [d for d in observed_datasets if d not in DATASETS]
        )
    )
    visible = tuple(
        p
        for p in (
            policies
            or tuple(
                [p for p in POLICIES if p in observed_policies]
                + [p for p in observed_policies if p not in POLICIES]
            )
        )
        if p not in HIDDEN_VARIANTS
    )
    declared_seeds = tuple(seeds) if seeds is not None else tuple(sorted({key[2] for key in series}))
    result = {}
    for dataset in chosen:
        groups = []
        for policy in visible:
            require(
                all((dataset, policy, seed) in series for seed in declared_seeds),
                "Série temporal incompleta.",
            )
            group = next(
                (
                    g
                    for g in groups
                    if all(_equivalent(series, dataset, member, policy, declared_seeds) for member in g)
                ),
                None,
            )
            if group is None:
                group = []
                groups.append(group)
            group.append(policy)
        require(
            {p for group in groups for p in group} == set(visible), "Política ausente da legenda temporal."
        )
        result[dataset] = groups
    return result


def _wrap(labels, limit=22):
    lines, current = [], ""
    for label in labels:
        addition = label if not current else " = " + label
        if current and len(current + addition) > limit:
            lines.append(current)
            current = "= " + label
        else:
            current += addition
    if current:
        lines.append(current)
    return "\n".join(lines)


def temporal_figures(evidence, *, show_title=True, show_footer=True, datasets=None, policies=None):
    """Show two aligned trajectories; aliases require every metric in every seed."""
    configure_arial()
    chosen, source, visible, seeds = _dimensions(evidence, datasets, policies)
    proof = _cara_equivalence(evidence, chosen, source, seeds)
    grouped = aliases(evidence.series, datasets=chosen, policies=visible, seeds=seeds)
    styles = ("-", "--", "-.", ":", "-", "--", "-.", ":", "-")
    markers = ("o", "s", "^", "D", "v", "P", "X", "h", "d")
    figures = {}
    for dataset in chosen:
        label = _dataset_label(evidence, dataset)
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.5), dpi=300)
        fig.subplots_adjust(
            left=0.135,
            right=0.965,
            top=0.85 if show_title else 0.96,
            bottom=0.365 if show_footer else 0.285,
            hspace=0.52,
        )
        handles, legend_labels = [], []
        for group in grouped[dataset]:
            representative = group[0]
            ordinal = visible.index(representative)
            color = COLORS.get(representative, FALLBACK_COLORS[ordinal % len(FALLBACK_COLORS)])
            final_values = []
            for ax, metric in zip(axes, ("macro_f1", "pr_auc"), strict=True):
                records = evidence.temporal.loc[
                    (evidence.temporal.dataset == dataset)
                    & (evidence.temporal.policy == representative)
                    & (evidence.temporal.metric == metric)
                ].sort_values("window")
                expected = evidence.series[dataset, representative, seeds[0]]
                if records.empty:
                    x = np.array([point["window"] for point in expected], dtype=float)
                    mean = np.full(len(expected), np.nan)
                    deviation = np.full(len(expected), np.nan)
                else:
                    require(len(records) == len(expected), "Agregado temporal sem todas as janelas.")
                    x = records["window"].to_numpy(dtype=float)
                    mean = records["mean"].to_numpy(dtype=float)
                    deviation = records["sample_sd"].to_numpy(dtype=float)
                (line,) = ax.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=styles[ordinal % len(styles)],
                    marker=markers[ordinal % len(markers)],
                    markersize=4,
                    markevery=max(1, len(x) // 8),
                    linewidth=1.65,
                )
                finite_band = np.isfinite(mean) & np.isfinite(deviation)
                if finite_band.any():
                    ax.fill_between(
                        x,
                        mean - deviation,
                        mean + deviation,
                        where=finite_band,
                        color=color,
                        alpha=0.11,
                        linewidth=0,
                    )
                final_values.append(_decimal(mean[-1], 2) if math.isfinite(mean[-1]) else "ND")
                if metric == "macro_f1":
                    handles.append(line)
            legend_labels.append(
                _wrap([SHORT.get(p, p) for p in group]) + f"\nF1 {final_values[0]}  AP {final_values[1]}"
            )
        windows = evidence.series[dataset, visible[0], seeds[0]]
        first, last = windows[0]["window"], windows[-1]["window"]
        for ax, axis_label in zip(axes, ("F1 macro", "Precisão média (AP)"), strict=True):
            ax.set_xlim(first - 0.3, last + 0.3)
            ax.set_ylim(-0.025, 1.025)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticks(sorted(set(int(v) for v in np.linspace(first, last, min(5, len(windows))))))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _decimal(value, 2)))
            ax.set_ylabel(axis_label, fontsize=11, labelpad=7)
            ax.tick_params(labelsize=11)
            ax.grid(axis="y", color=GRID, linewidth=0.7)
            ax.set_axisbelow(True)
        axes[0].tick_params(labelbottom=False)
        axes[1].set_xlabel("Janela da coorte comum", fontsize=11, labelpad=5)
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.080 if show_footer else 0.008),
            ncol=3,
            frameon=False,
            fontsize=11,
            handlelength=1.6,
            handletextpad=0.35,
            columnspacing=0.9,
            labelspacing=0.65,
            borderaxespad=0,
        )
        caption = (
            f"Evolução do F1 macro e da precisão média (AP) nas {len(windows)} janelas "
            f"alinhadas da coorte comum de {label}. "
            f"Linhas mostram médias de {len(seeds)} seeds; faixas, quando definidas, "
            "mostram DP amostral com ddof 1, não IC. "
            "A legenda informa os valores médios da última janela. O sinal = reúne "
            "somente políticas com todas as métricas, "
            "fronteiras e denominadores iguais em cada seed até a tolerância de 1e-12. "
            + _ranking_note(evidence)
            + (
                " As variantes CARA cumulativa e periódica foram conferidas e "
                "ocultadas; CARA identifica a base."
                if set(source) & HIDDEN_VARIANTS
                else ""
            )
            + " A igualdade das trajetórias não implica igualdade de esforço físico "
            "ou custo. Não se inferem fronteiras ou atrasos de drift."
        )
        decorate(
            fig,
            f"{label}: evolução por janela",
            caption,
            show_title=show_title,
            show_footer=show_footer,
            footer="Faixas: DP amostral, não IC. Legenda: médias da última janela; = "
            "indica séries equivalentes.",
            metric_descriptions={
                "F1 macro": "Média não ponderada do F1 das classes em cada janela comum.",
                "AP": "Precisão média positiva em cada janela; " + _ranking_note(evidence),
            },
            dataset=dataset,
            source_slot_id="temporal-" + dataset,
            figure_family="temporal",
            visible_policies=list(visible),
            source_policies=list(source),
            seeds=list(seeds),
            temporal_aliases=grouped[dataset],
            alias_tolerance=ALIAS_TOLERANCE,
            cara_equivalence=proof,
            quality_scope="common_postcalibration",
            raw_evidence_modified=False,
        )
        validate_figure(fig)
        figures["temporal-" + dataset] = fig
    return figures


def build_figures(evidence, *, show_title=True, show_footer=True, datasets=None, policies=None):
    """Build three panels per dataset under the shared, editable publication style."""
    font = configure_arial()
    chosen, source, visible, seeds = _dimensions(evidence, datasets, policies)
    figures = {
        **quality_figures(
            evidence, show_title=show_title, show_footer=show_footer, datasets=chosen, policies=source
        ),
        **resource_figures(
            evidence, show_title=show_title, show_footer=show_footer, datasets=chosen, policies=source
        ),
        **temporal_figures(
            evidence, show_title=show_title, show_footer=show_footer, datasets=chosen, policies=source
        ),
    }
    require(len(figures) == 3 * len(chosen), "Cada dataset deve ter qualidade, recursos e trajetórias.")
    for fig in figures.values():
        validate_figure(fig)
    return figures, font


def export_figures(figures, evidence, directory, font, *, show_title=None, show_footer=None, variant=None):
    """Export PNG300/PDF/SVG and external captions; preserve every raw policy and seed."""
    require(figures, "Nenhuma figura para exportar.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = getattr(next(iter(figures.values())), "_benchmark_metadata", {})
    title_flag = metadata.get("show_title", True) if show_title is None else bool(show_title)
    footer_flag = metadata.get("show_footer", True) if show_footer is None else bool(show_footer)
    temporary = None
    if any(
        getattr(fig, "_benchmark_metadata", {}).get("show_title") != title_flag
        or getattr(fig, "_benchmark_metadata", {}).get("show_footer") != footer_flag
        for fig in figures.values()
    ):
        chosen = tuple(
            dict.fromkeys(getattr(fig, "_benchmark_metadata", {}).get("dataset") for fig in figures.values())
        )
        temporary, font = build_figures(
            evidence,
            show_title=title_flag,
            show_footer=footer_flag,
            datasets=chosen,
            policies=metadata.get("source_policies"),
        )
        require(set(figures) <= set(temporary), "A reconstrução não preserva as figuras selecionadas.")
        selected = {name: temporary[name] for name in figures}
    else:
        selected = figures
    variant = variant or ("docs" if not title_flag and not footer_flag else "notebook")
    try:
        pdf_path = export_bundle(selected, directory, variant=variant)
    finally:
        if temporary is not None:
            for figure in temporary.values():
                plt.close(figure)
    for name, attribute in (
        ("qualidade-por-seed", "primary"),
        ("qualidade-media-dp", "quality"),
        ("recursos-por-seed", "resource_values"),
        ("recursos-media-dp", "resources"),
        ("janelas-por-seed", "temporal_values"),
        ("janelas-media-dp", "temporal"),
        ("holdout-media-dp", "holdout"),
        ("cobertura", "coverage"),
    ):
        getattr(evidence, attribute).to_csv(directory / (name + ".csv"), index=False)
    chosen, source, visible, seeds = _dimensions(
        evidence,
        metadata.get("dataset")
        and tuple(
            dict.fromkeys(getattr(fig, "_benchmark_metadata", {}).get("dataset") for fig in figures.values())
        ),
        metadata.get("source_policies"),
    )
    provenance = {
        **evidence.provenance,
        "font": font,
        "figures": len(selected),
        "visual_datasets": list(chosen),
        "sample_sd_ddof": 1,
        "visual_source_policies": list(source),
        "visual_policies": list(visible),
        "hidden_visual_variants": sorted(set(source) & HIDDEN_VARIANTS),
        "aliases": aliases(evidence.series, datasets=chosen, policies=visible, seeds=seeds),
        "cara_equivalence": _cara_equivalence(evidence, chosen, source, seeds),
        "alias_tolerance": ALIAS_TOLERANCE,
        "quality_scope": "common_postcalibration",
        "resource_scope": "full_evaluation",
        "holdout_scope": "separate_temporal_holdout_diagnostic",
        "show_title": title_flag,
        "show_footer": footer_flag,
        "models_loaded": False,
        "new_benchmarks_executed": False,
        "raw_evidence_modified": False,
        "limitations": [
            "Seeds repetem o mesmo fluxo; DP amostral não é intervalo de confiança.",
            "Hardware pode ser heterogêneo; reserva de recursos não torna as CPUs idênticas.",
            "As variantes cached dependem da construção CARA; seus recursos não foram combinados com a base.",
            "Materializações já integram os totais físicos e não devem ser somadas outra vez.",
            "Custos lógicos, unidades de cenário e estimativas de referência não são faturamento observado.",
            "AP usa o estimador declarado; nesta rodada F109 são 4.096 bins.",
        ],
    }
    (directory / f"proveniencia-{variant}.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    return pdf_path
