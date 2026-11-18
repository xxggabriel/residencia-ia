"""Render the seven final INSECTS figures from validated benchmark evidence."""

import matplotlib.pyplot as plt
import numpy as np
from benchmark_figure_data import included_policies
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

INK = "#172737"
MUTED = "#485D6E"
GRID = "#E3EBEF"
PENDING = "#EEF2F5"
TEAL = "#12675E"
ORANGE = "#9D472D"
BLUE = "#265C8A"
NAVY = "#24435A"
POLICIES = included_policies("Logística")
MODELS = ("Hoeffding", "Logística")
DATASETS = ("Abrupto", "Recorrente")
TITLES = {"Abrupto": "Abrupt Imbalanced", "Recorrente": "Incremental-Abrupt-Reoccurring Imbalanced"}
POLICY_COLORS = dict(zip(POLICIES, ("#70828F", BLUE, "#996348", "#5361A1", TEAL, "#8E4C76")))
DELTA = LinearSegmentedColormap.from_list("delta", [ORANGE, "#F8F9F7", TEAL])
ACCURACY = LinearSegmentedColormap.from_list("accuracy", ["#FACEAE", "#FEF8E9", "#90C8B8", "#124E51"])
COUNTS = LinearSegmentedColormap.from_list("counts", ["#F3F6F8", "#AFC7D9", "#24435A"])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "hatch.color": "#9CAEBB",
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
    }
)


def number(value, decimals=0):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def signed(value, decimals=1):
    value = 0 if abs(value) < 0.5 * 10**-decimals else value
    return ("+" if value > 0 else "") + number(value, decimals)


def seed_label(evidence):
    if isinstance(evidence.seed, tuple):
        return "Média das seeds " + ", ".join(str(seed) for seed in evidence.seed)
    return f"Seed {evidence.seed}"


def framed(title, subtitle, footnote, height=8.1):
    fig = plt.figure(figsize=(13.2, height))
    fig.text(0.065, 0.955, title, fontsize=24, fontweight="bold", va="top")
    fig.text(0.065, 0.882, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(0.065, 0.039, footnote, fontsize=9, color=MUTED, va="bottom", linespacing=1.55)
    return fig


def flat_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def tile(ax, bottom_left, width, height):
    ax.add_patch(
        Rectangle(
            bottom_left, width, height, facecolor=PENDING, edgecolor="#C1CCD4", hatch="//", linewidth=0.45
        )
    )


def missing_label(evidence, dataset, model, policy):
    pending = evidence.pending
    match = pending.dataset.eq(dataset) & pending.modelo.eq(model) & pending.politica.eq(policy)
    return "Pendente" if match.any() else "Sem run"


def absent_panel(ax, title, message):
    flat_axis(ax)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=MUTED)


def regime_map(evidence):
    regimes = evidence.regimes
    fig = framed(
        "F1 macro por regime em relação ao Congelado",
        f"INSECTS na AWS  /  Eventos pareados  /  {seed_label(evidence)}",
        "Fonte: predições do MLflow e marcos dos manifestos.\n"
        "Congelado mostra F1 absoluto. As demais linhas mostram a diferença em p.p.\n"
        "Sob cada regime, o percentual indica a proporção de positivos nos eventos pareados.",
    )
    outer = fig.add_gridspec(
        2, 2, left=0.16, right=0.87, top=0.80, bottom=0.20, width_ratios=[5, 2], wspace=0.32, hspace=0.57
    )
    refs = regimes.loc[regimes.politica.eq("Congelado"), ["dataset", "modelo", "regime", "f1"]]
    gaps = regimes.merge(refs, on=["dataset", "modelo", "regime"], suffixes=("", "_ref"))
    max_delta = (gaps.f1 - gaps.f1_ref).abs().max() * 100 if not gaps.empty else 0
    limit = max(45, 15 * np.ceil(max_delta / 15))
    scale = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for i, model in enumerate(MODELS):
        policies = included_policies(model)
        for j, dataset in enumerate(DATASETS):
            ax = fig.add_subplot(outer[i, j])
            data = regimes[regimes.modelo.eq(model) & regimes.dataset.eq(dataset)]
            ref = data[data.politica.eq("Congelado")].set_index("regime").sort_index()
            if ref.empty:
                absent_panel(ax, f"{dataset}  /  {model}", "Sem referência Congelado")
                continue
            for _, part in data.groupby("regime"):
                if part.avaliados.nunique() != 1 or part.positivos.nunique() != 1:
                    raise ValueError("As políticas de um regime devem usar a mesma base de avaliação.")
            scores = data.pivot(index="politica", columns="regime", values="f1")
            values = (scores.reindex(index=policies, columns=ref.index) - ref["f1"]) * 100
            for r, policy in enumerate(policies):
                for c, regime in enumerate(ref.index):
                    value = values.loc[policy, regime]
                    if np.isnan(value):
                        tile(ax, (c, r), 1, 1)
                        ax.text(
                            c + 0.5,
                            r + 0.5,
                            missing_label(evidence, dataset, model, policy),
                            ha="center",
                            va="center",
                            color=MUTED,
                            fontsize=8,
                        )
                    else:
                        ax.add_patch(Rectangle((c, r), 1, 1, fc=DELTA(scale(value)), ec="white", lw=1.4))
                        label = number(ref.loc[regime, "f1"] * 100, 1) + "% ref." if r == 0 else signed(value)
                        ax.text(
                            c + 0.5,
                            r + 0.5,
                            label,
                            ha="center",
                            va="center",
                            fontsize=9.7,
                            color="white" if abs(value) >= 0.64 * limit else INK,
                            fontweight="normal",
                        )
            ax.set(xlim=(0, len(ref)), ylim=(len(policies), 0))
            ax.set_title(f"{dataset}  /  {model}", loc="left", fontweight="bold", fontsize=12, pad=9)
            ax.set_xticks(
                np.arange(len(ref)) + 0.5,
                [f"R{k}  ·  {number(row.positivos / row.avaliados * 100, 1)}%" for k, row in ref.iterrows()],
                fontsize=9,
            )
            ax.set_yticks(np.arange(len(policies)) + 0.5, policies if not j else [])
            flat_axis(ax)
    cax = fig.add_axes([0.19, 0.126, 0.52, 0.019])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=scale, cmap=DELTA),
        cax=cax,
        orientation="horizontal",
        ticks=np.linspace(-limit, limit, 7),
    )
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: signed(x, 0)))
    cb.ax.tick_params(length=0, labelsize=8)
    fig.text(0.74, 0.126, "p.p. de F1 macro", fontsize=9, color=MUTED)
    return fig


def detector_timeline(evidence):
    detector_runs = evidence.detector.groupby(["dataset", "modelo"])
    manifest = evidence.manifests
    costs = evidence.drift_artifacts
    fig = framed(
        "Acurácia, alarmes e atualizações por posição no fluxo",
        f"INSECTS  /  ADWIN  /  Acurácia por janela e contagens por bloco  /  {seed_label(evidence)}",
        "Fonte: artefatos de drift do MLflow. Uma faixa cinza no início indica ausência de janela "
        "registrada em todas as seeds.\n"
        "Os regimes usam as fronteiras anotadas pelos autores. No Recorrente, também há drift "
        "incremental dentro de cada ciclo.",
        height=8.6,
    )
    outer = fig.add_gridspec(2, 1, left=0.218, right=0.875, bottom=0.205, top=0.805, hspace=0.49)
    accnorm = TwoSlopeNorm(vmin=0, vcenter=72, vmax=100)
    bins_by_dataset = {
        dataset: np.r_[np.arange(0, m["samples"], 5000), m["samples"]] for dataset, m in manifest.items()
    }
    event_counts = {}
    for row in evidence.detector.itertuples():
        markers = costs[row.run]["detection_timeline"]["markers"]
        block = bins_by_dataset[row.dataset]
        for event in ("alarm", "update"):
            event_counts[row.run, event] = np.histogram(
                [point["x"] for point in markers if point["kind"] == event], block
            )[0]
    max_events = max((values.max() for values in event_counts.values()), default=0)
    countnorm = Normalize(0, max(5, max_events))
    for i, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(outer[i, 0])
        if dataset not in manifest:
            absent_panel(ax, TITLES[dataset], "Sem dados do fluxo")
            continue
        m = manifest[dataset]
        ends = m["samples"]
        block = bins_by_dataset[dataset]
        for r, regime in enumerate(m["binary_distribution_per_regime"]):
            start, end = regime["start"], regime["end"]
            ax.add_patch(Rectangle((start, 0), end - start, 0.72, fc=["#D4E5E9", "#B3CED4"][r % 2]))
            ax.text((start + end) / 2, 0.36, f"R{regime['regime']}", ha="center", va="center", fontsize=9)
        ylabels = ["Regimes anotados"]
        for j, model in enumerate(MODELS):
            y = 1.05 + j * 2.8
            ylabels.extend([f"{model}  ·  Acurácia", "Alarmes", "Atualizações"])
            if (dataset, model) not in detector_runs.groups:
                tile(ax, (0, y), ends, 2.4)
                ax.text(ends / 2, y + 1.2, "Sem run ADWIN", ha="center", va="center", color=MUTED)
                continue
            data = detector_runs.get_group((dataset, model))
            series_by_run = []
            for row in data.itertuples():
                timeline = costs[row.run]["detection_timeline"]
                points = next(item["points"] for item in timeline["series"] if item["name"] == "accuracy")
                series_by_run.append(dict(points))
                if [p["x"] for p in timeline["markers"] if p["kind"] == "changepoint"] != m["changepoints"]:
                    raise ValueError(f"Marcos do ADWIN divergentes do manifesto em {row.run}.")
            ax.add_patch(Rectangle((0, y), ends, 0.74, fc=PENDING))
            seq = np.array(sorted(set.intersection(*(set(points) for points in series_by_run))))
            if seq.size:
                scores = np.mean([[points[x] for x in seq] for points in series_by_run], axis=0)
                width = seq[1] - seq[0] if len(seq) > 1 else 1
                edges = np.r_[max(0, seq[0] - width), seq]
                ax.pcolormesh(edges, [y, y + 0.74], [scores * 100], cmap=ACCURACY, norm=accnorm)
            for k, event in enumerate(("alarm", "update"), start=1):
                counts = np.mean([event_counts[run, event] for run in data.run], axis=0)
                ax.pcolormesh(
                    block, [y + 0.83 * k, y + 0.83 * k + 0.74], [counts], cmap=COUNTS, norm=countnorm
                )
                ax.text(
                    1.012,
                    y + 0.83 * k + 0.37,
                    number(counts.sum(), 0 if np.isclose(counts.sum() % 1, 0) else 1),
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
        for point in m["changepoints"]:
            ax.axvline(point, color="#2A3C45", linewidth=0.8, alpha=0.85)
        ax.text(
            1.012,
            -0.52,
            "Total médio" if isinstance(evidence.seed, tuple) else "Total",
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            color=MUTED,
        )
        ax.set_xlim(0, ends)
        ax.set_ylim(6.3, -0.08)
        ax.set_yticks([0.36, 1.42, 2.25, 3.08, 4.22, 5.05, 5.88], ylabels, fontsize=9)
        ticks = np.arange(0, ends, 50000)
        ax.set_xticks(ticks, [number(t / 1000) for t in ticks], fontsize=9)
        ax.set_title(TITLES[dataset], loc="left", fontweight="bold", fontsize=12, pad=10)
        flat_axis(ax)
    fig.text(0.565, 0.161, "Posição no fluxo (mil exemplos)", ha="center", color=MUTED, fontsize=9)
    for x, cmap, norm, ticks, caption in (
        (0.22, ACCURACY, accnorm, [0, 25, 50, 75, 100], "Acurácia (%)"),
        (0.60, COUNTS, countnorm, MaxNLocator(nbins=5, integer=True), "Eventos por bloco de 5 mil exemplos"),
    ):
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=fig.add_axes([x, 0.105, 0.25, 0.015]),
            orientation="horizontal",
            ticks=ticks,
        )
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8)
        fig.text(x, 0.127, caption, fontsize=8.7, color=MUTED)
    return fig


def recovery_bars(evidence):
    detector_runs = evidence.detector.groupby(["dataset", "modelo"])
    costs = evidence.drift_artifacts
    manifest = evidence.manifests
    fig = framed(
        "Exemplos até recuperar a acurácia",
        f"ADWIN  /  {seed_label(evidence)}  /  Somente runs que atingiram o alvo",
        "Recuperação: duas janelas elegíveis consecutivas com acurácia ≥ a da última janela "
        "pré-mudança menos 5 p.p.\n"
        "A contagem começa no drift anotado. O alvo pode ser atingido mesmo sem queda inicial "
        "ou alarme pareado.",
        height=6.6,
    )
    outer = fig.add_gridspec(
        1, 2, left=0.098, right=0.948, top=0.76, bottom=0.225, width_ratios=[5, 2], wspace=0.16
    )
    colors = [NAVY, "#A7542C"]
    max_delay = max(
        (
            value
            for a in costs.values()
            if "recovery" in a
            for value in a["recovery"]["values"]
            if value is not None
        ),
        default=0,
    )
    limit = max(24500, max_delay * 1.12)
    for i, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(outer[0, i])
        title = f"{dataset}  /  versão desbalanceada"
        if dataset not in manifest:
            absent_panel(ax, title, "Sem dados do fluxo")
            continue
        change = manifest[dataset]["changepoints"]
        for j, model in enumerate(MODELS):
            values = {point: [] for point in change}
            if (dataset, model) in detector_runs.groups:
                for run in detector_runs.get_group((dataset, model)).run:
                    series = costs[run]["recovery"]
                    points = [int(label.split("@")[1]) for label in series["labels"]]
                    if points != change:
                        raise ValueError(f"Marcos de recuperação divergentes no run {run}.")
                    for point, value in zip(points, series["values"], strict=True):
                        if value is not None:
                            values[point].append(value)
            pos = np.arange(len(change)) + (j - 0.5) * 0.39
            for x, delays in zip(pos, values.values(), strict=True):
                value = float(np.mean(delays)) if delays else None
                if value is None:
                    tile(ax, (x - 0.175, 0), 0.35, limit * 0.045)
                else:
                    bars = ax.bar(x, value, width=0.35, color=colors[j])
                    label = number(value, 0 if value.is_integer() else 1)
                    if isinstance(evidence.seed, tuple) and len(delays) < len(evidence.seed):
                        label += f" ({len(delays)}/{len(evidence.seed)})"
                    ax.bar_label(bars, labels=[label], padding=4, fontsize=8.5)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=9)
        ax.set_ylim(0, limit)
        ax.set_xticks(np.arange(len(change)), [number(v) for v in change], fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{number(x / 1000)} mil" if x else "0"))
        ax.tick_params(axis="y", labelleft=not i)
        flat_axis(ax)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color=GRID, linewidth=0.8)
    fig.text(
        0.515,
        0.165,
        "Posição exata do drift anotado (índice do exemplo)",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    handles = [Patch(fc=c, label=m) for c, m in zip(colors, MODELS, strict=True)]
    if len(detector_runs) < 4 or any(
        None in a["recovery"]["values"] for a in costs.values() if "recovery" in a
    ):
        handles.append(Patch(fc=PENDING, ec="#C1CCD4", hatch="//", label="Sem valor registrado"))
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=len(handles),
        loc="lower right",
        bbox_to_anchor=(0.95, 0.10),
        fontsize=9,
    )
    return fig


def quality_cost(evidence):
    runs, paired = evidence.runs, evidence.paired
    max_cost = runs.custo_registrado.max()
    cost_limit = max(21, np.ceil(max_cost * 1100)) if np.isfinite(max_cost) else 21
    fig = framed(
        "F1 macro, custo e número de atualizações",
        f"INSECTS  /  Eventos pareados e custo do run integral  /  {seed_label(evidence)}",
        "O custo é a estimativa de infraestrutura registrada (CPU, RAM e armazenamento), em "
        "milésimos de US$.\n"
        "O número ao lado da barra de custo indica atualizações. O Online atualiza por exemplo e as "
        "demais políticas registram retreinamentos.",
        height=9.8,
    )
    outer = fig.add_gridspec(2, 2, left=0.145, right=0.945, top=0.80, bottom=0.155, hspace=0.54, wspace=0.37)
    for i, (dataset, model) in enumerate((d, m) for d in DATASETS for m in MODELS):
        policies = included_policies(model)
        grid = outer[i // 2, i % 2].subgridspec(1, 2, width_ratios=[1.12, 1], wspace=0.19)
        ax, cost_ax = [fig.add_subplot(grid[0, j]) for j in range(2)]
        pair = (
            paired[paired.dataset.eq(dataset) & paired.modelo.eq(model)]
            .merge(runs[["run", "custo_registrado", "atualizacoes"]], on="run", validate="one_to_one")
            .set_index("politica")
        )
        for j, policy in enumerate(policies):
            if policy not in pair.index:
                for a, width in ((ax, 100), (cost_ax, cost_limit)):
                    tile(a, (0, j - 0.27), width, 0.54)
                ax.text(
                    50,
                    j,
                    missing_label(evidence, dataset, model, policy),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                continue
            row = pair.loc[policy]
            cost = row.custo_registrado * 1000
            color = POLICY_COLORS[policy]
            ax.barh(j, row.f1 * 100, color=color, height=0.52)
            if row.f1 > 0.92:
                ax.text(
                    row.f1 * 100 - 1.4,
                    j,
                    number(row.f1 * 100, 1),
                    ha="right",
                    va="center",
                    color="white",
                    fontsize=8,
                )
            else:
                ax.text(row.f1 * 100 + 1.6, j, number(row.f1 * 100, 1), va="center", fontsize=8)
            count_label = number(row.atualizacoes, 0 if float(row.atualizacoes).is_integer() else 1)
            if np.isfinite(cost):
                cost_ax.barh(j, cost, color=color, height=0.52)
                label = f"{number(cost, 3)}  ·  {count_label}"
                margin = cost_limit * 0.017
                if cost > 0.57 * cost_limit:
                    cost_ax.text(cost - margin, j, label, ha="right", va="center", color="white", fontsize=8)
                else:
                    cost_ax.text(cost + margin, j, label, va="center", fontsize=8)
            else:
                cost_ax.text(
                    0.02 * cost_limit,
                    j,
                    f"Sem custo  ·  {count_label}",
                    va="center",
                    fontsize=8,
                )
        ax.text(
            0,
            1.135,
            f"{dataset}  /  {model}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
        ax.set_title("F1 macro (%)", loc="left", fontsize=9, color=MUTED, pad=7)
        cost_ax.set_title("mUSD  ·  atualizações", loc="left", fontsize=9, color=MUTED, pad=7)
        for a, limit, ticks, labels in [
            (ax, 100, [0, 25, 50, 75, 100], policies),
            (cost_ax, cost_limit, None, []),
        ]:
            a.set(xlim=(0, limit), ylim=(len(policies) - 0.5, -0.5))
            if ticks is None:
                a.xaxis.set_major_locator(MaxNLocator(nbins=5))
            else:
                a.set_xticks(ticks)
            a.set_yticks(np.arange(len(policies)), labels, fontsize=9)
            flat_axis(a)
            a.tick_params(axis="x", labelsize=8)
            a.set_axisbelow(True)
            a.xaxis.grid(color=GRID, linewidth=0.8)
    return fig


def paired_f1(evidence):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.5), layout="constrained")
    fig.suptitle(
        f"INSECTS · F1 macro nos eventos comuns · {seed_label(evidence)}", fontsize=19, fontweight="bold"
    )
    for i, dataset in enumerate(DATASETS):
        source = evidence.paired.loc[evidence.paired.dataset.eq(dataset)]
        if source.empty:
            for ax in axes[i]:
                absent_panel(ax, dataset, "Sem eventos pareados")
            continue
        if source.avaliados.nunique() != 1:
            raise ValueError(f"Cardinalidades diferentes no pareamento de {dataset}.")
        for ax, model in zip(axes[i], MODELS, strict=True):
            rows = source.loc[source.modelo.eq(model)].set_index("politica")
            policies = included_policies(model)
            rows = rows.reindex(policies).dropna(subset=["f1"])

            spread = rows.get("f1_dp")
            bars = ax.barh(
                rows.index,
                100 * rows.f1,
                color=[POLICY_COLORS[p] for p in rows.index],
                xerr=None if spread is None else 100 * spread.fillna(0),
                capsize=3,
            )
            ax.bar_label(bars, labels=[number(v * 100, 2) + "%" for v in rows.f1], padding=5)
            ax.set_title(
                f"{dataset} · {model} · {number(source.avaliados.iloc[0])} eventos",
                loc="left",
                fontweight="bold",
                fontsize=11,
            )
            ax.set(xlim=(0, 100), xlabel="F1 macro (%)" if spread is None else "F1 macro (%) · média ± DP")
            ax.invert_yaxis()
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=GRID)
            flat_axis(ax)
    return fig


def logistic_trajectories(evidence):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7), layout="constrained", sharey=True)
    fig.suptitle(
        f"INSECTS · F1 macro por janela da regressão logística · {seed_label(evidence)}",
        fontsize=17,
        fontweight="bold",
    )
    data = evidence.windows.loc[evidence.windows.modelo.eq("Logística")]
    routes = (("Por evento", ("Congelado", "Online", "ADWIN")), ("Por bloco", ("CARA", "UPF", "Oracle")))
    for i, dataset in enumerate(DATASETS):
        manifest = evidence.manifests.get(dataset)
        for ax, (route, policies) in zip(axes[i], routes, strict=True):
            if manifest is None:
                absent_panel(ax, f"{dataset} · {route}", "Sem dados do fluxo")
                continue
            series = data.loc[data.dataset.eq(dataset)]
            spacings = []
            for policy in policies:
                points = series.loc[series.politica.eq(policy)].sort_values("passo")
                if points.empty:
                    continue
                if points.passo.duplicated().any():
                    raise ValueError(f"Janelas duplicadas em {dataset}, {policy}.")
                if len(points) > 1:
                    spacings.append(np.median(np.diff(points.passo)))
                if "dp" in points and points.dp.notna().any():
                    ax.fill_between(
                        points.passo / 1000,
                        100 * (points.f1 - points.dp).clip(0, 1),
                        100 * (points.f1 + points.dp).clip(0, 1),
                        color=POLICY_COLORS[policy],
                        alpha=0.15,
                        linewidth=0,
                    )
                ax.plot(
                    points.passo / 1000,
                    points.f1 * 100,
                    color=POLICY_COLORS[policy],
                    lw=1.3,
                    marker="o" if route == "Por bloco" else None,
                    ms=3,
                    label=policy,
                )
            for position in manifest["changepoints"]:
                ax.axvline(position / 1000, color="#74828A", ls=":", lw=1)
            spacing = (
                f" · intervalo típico de {number(np.median(spacings) / 1000, 1)} mil" if spacings else ""
            )
            ax.set_title(f"{dataset} · {route}{spacing}", loc="left", fontweight="bold", fontsize=10)
            ax.set(
                xlim=(0, manifest["samples"] / 1000),
                ylim=(0, 100),
                xlabel=(
                    "Posição no fluxo (mil exemplos) · faixa ±1 DP"
                    if isinstance(evidence.seed, tuple)
                    else "Posição no fluxo (mil exemplos) · pontilhado marca drift"
                ),
            )
            if ax in axes[:, 0]:
                ax.set_ylabel("F1 macro da janela (%)")
            ax.yaxis.set_major_locator(MultipleLocator(20))
            if ax.lines and spacings:
                ax.legend(loc="lower right", ncols=3, fontsize=8, frameon=False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=GRID)
            flat_axis(ax)
    return fig


def quality_retention(evidence):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), layout="constrained", sharex=True, sharey=True)
    fig.suptitle(
        f"INSECTS · F1 no fluxo e retenção histórica · {seed_label(evidence)}", fontsize=18, fontweight="bold"
    )
    data = evidence.paired.merge(evidence.runs[["run", "retencao"]], on="run", validate="one_to_one")
    marks = {"Hoeffding": "^", "Logística": "o"}
    for ax, dataset in zip(axes, DATASETS, strict=True):
        rows = data.loc[data.dataset.eq(dataset)].dropna(subset=["retencao"])
        for r in rows.itertuples():
            ax.scatter(
                r.f1 * 100,
                r.retencao * 100,
                marker=marks[r.modelo],
                color=POLICY_COLORS[r.politica],
                s=65,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xlabel("F1 macro nos eventos comuns (%)")
        ax.grid(color=GRID)
        flat_axis(ax)
    axes[0].set_ylabel("Acurácia nos probes de regimes anteriores (%)")
    fig.legend(
        handles=[
            *[Patch(facecolor=POLICY_COLORS[p], label=p) for p in POLICIES],
            *[Line2D([], [], marker=marks[m], color=INK, linestyle="none", label=m) for m in MODELS],
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    return fig


def export_figures(figures, output):
    """Save individual PNGs and a single multipage PDF, keeping figures available for Jupyter."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "figuras").mkdir(exist_ok=True)
    pdf_path = output / "figuras-finais-insects.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Figuras finais do INSECTS na Semana 10"}) as pdf:
        for name, fig in figures.items():
            fig.savefig(output / "figuras" / f"{name}.png", dpi=205)
            pdf.savefig(fig)
    return pdf_path
