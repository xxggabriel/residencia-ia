"""Generate SVG figures for the updated CARA (2024) and UPF (2025) benchmark reports across 4 streams."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/Users/gabrieloliveira/residencia-ia")
EVIDENCE_DIR = ROOT / "evidencias/homelab"
PUB_CSV = ROOT / "evidencias/aws-benchmarks/papers/published-reference-values.csv"
OUT_FIG_DIR = ROOT / "docs_v2/relatorios/figuras/metodologia_artigos"

sys.path.insert(0, str(ROOT / "notebooks"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from benchmark_plot_style import BLUE, INK, MUTED, ORANGE, POLICY_COLORS, TEAL, apply_style  # noqa: E402


def _find_latest_run(prefix: str) -> Path:
    matches = sorted(p for p in EVIDENCE_DIR.iterdir() if p.is_dir() and p.name.startswith(f"{prefix}_"))
    if not matches:
        raise FileNotFoundError(f"No run directory found for prefix {prefix!r} in {EVIDENCE_DIR}")
    return matches[-1]


def _load_summary(prefix: str) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    run_dir = _find_latest_run(prefix)
    data = json.loads((run_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    for row in data:
        key = (str(row["policy"]), str(row["regime"]), str(row["metric"]))
        out[key] = (float(row["mean"]), float(row["std"]), int(row["n_seeds"]))
    return out


def _load_published() -> dict[tuple[str, str, str, str, str], float]:
    out: dict[tuple[str, str, str, str, str], float] = {}
    with PUB_CSV.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            out[
                (
                    row["paper_id"],
                    row["dataset"],
                    row["policy"],
                    row["metric"],
                    row["regime"],
                )
            ] = float(row["published_value"])
    return out


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)


def build_cara_insects_comparativo(
    abrupt: dict[tuple[str, str, str], tuple[float, float, int]],
    reocc: dict[tuple[str, str, str], tuple[float, float, int]],
) -> plt.Figure:
    policies = [
        ("oracle_staleness", "Oracle"),
        ("cara", "CARA-T"),
        ("cara_cumulative", "CARA-CT"),
        ("cara_periodic", "CARA-P"),
        ("markov", "Markov"),
        ("ddm", "DDM"),
        ("adwin", "ADWIN"),
        ("frozen", "Congelado"),
    ]
    labels = [lbl for _, lbl in policies]
    y = np.arange(len(policies))
    h = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)

    # Panel A: SCPE (%)
    scpe_pols = policies[1:]  # Oracle SCPE is 0
    y_scpe = np.arange(len(scpe_pols))
    ab_scpe = [abrupt[(k, "price_grid", "strategy_scpe")][0] for k, _ in scpe_pols]
    re_scpe = [reocc[(k, "price_grid", "strategy_scpe")][0] for k, _ in scpe_pols]
    b1 = ax1.barh(y_scpe - h / 2, ab_scpe, height=h, color=BLUE, label="INSECTS Abrupto")
    b2 = ax1.barh(y_scpe + h / 2, re_scpe, height=h, color=ORANGE, label="INSECTS Recorrente")
    for bars, vals in ((b1, ab_scpe), (b2, re_scpe)):
        for bar, val in zip(bars, vals, strict=True):
            ax1.text(
                val + 5.0,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%".replace(".", ","),
                va="center",
                ha="left",
                fontsize=7.8,
                color=INK,
            )
    ax1.set_yticks(y_scpe, [lbl for _, lbl in scpe_pols], fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(*ab_scpe, *re_scpe) * 1.18)
    ax1.set_title("SCPE vs. Oracle (%, ↓ melhor)", fontsize=10.8, fontweight="bold", color=INK, loc="left")
    ax1.set_xlabel("Strategy Cost Percentage Error (%)", fontsize=9.2, color=INK)
    _style_axis(ax1)

    # Panel B: Average Accuracy (all 8 policies)
    fig.delaxes(ax2)
    ax2 = fig.add_subplot(1, 2, 2)
    ab_acc = [abrupt[(k, "price_grid", "average_accuracy")][0] for k, _ in policies]
    re_acc = [reocc[(k, "price_grid", "average_accuracy")][0] for k, _ in policies]
    a1 = ax2.barh(y - h / 2, ab_acc, height=h, color=BLUE, label="INSECTS Abrupto")
    a2 = ax2.barh(y + h / 2, re_acc, height=h, color=ORANGE, label="INSECTS Recorrente")
    for bars, vals in ((a1, ab_acc), (a2, re_acc)):
        for bar, val in zip(bars, vals, strict=True):
            ax2.text(
                val + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}".replace(".", ","),
                va="center",
                ha="left",
                fontsize=7.8,
                color=INK,
            )
    ax2.set_yticks(y, labels, fontsize=9.2)
    ax2.invert_yaxis()
    ax2.set_xlim(0.65, 0.96)
    ax2.set_title(
        "Acurácia Média por Lote (↑ melhor)", fontsize=10.8, fontweight="bold", color=INK, loc="left"
    )
    ax2.set_xlabel("Acurácia média (75 lotes online, 5 seeds)", fontsize=9.2, color=INK)
    _style_axis(ax2)

    handles, leg_labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.89),
        ncol=2,
        frameon=False,
        fontsize=9.2,
    )
    fig.suptitle(
        "Protocolo CARA (2024) em INSECTS: Abrupt Imbalanced vs. Incremental Abrupt Reoccurring",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=12.6,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.02,
        "Fonte: homelab_insects (Random Forest n_estimators=100, 100 lotes 25/75, "
        "split 90/10, gamma=1/33, 99 kappas, 5 seeds).",
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.79, bottom=0.14, wspace=0.28)
    return fig


def build_upf_insects_auc_e_regimes(
    abrupt: dict[tuple[str, str, str], tuple[float, float, int]],
    reocc: dict[tuple[str, str, str], tuple[float, float, int]],
) -> plt.Figure:
    policies = [
        ("oracle", "Oracle"),
        ("upf", "UPF"),
        ("cara", "CARA-T"),
        ("cara_cumulative", "CARA-CT"),
        ("cara_periodic", "CARA-P"),
        ("adwin_5", "ADWIN-5%"),
        ("fhddm_5", "FHDDM-5%"),
        ("kswin_5", "KSWIN-5%"),
        ("fhddm_50", "FHDDM-50%"),
        ("adwin_50", "ADWIN-50%"),
        ("kswin_50", "KSWIN-50%"),
    ]
    labels = [lbl for _, lbl in policies]
    y = np.arange(len(policies))
    h = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8))

    ab_auc = [abrupt[(k, "price_grid", "combined_cost_auc")][0] for k, _ in policies]
    re_auc = [reocc[(k, "price_grid", "combined_cost_auc")][0] for k, _ in policies]
    b1 = ax1.barh(y - h / 2, ab_auc, height=h, color=BLUE, label="INSECTS Abrupto (α_max=1,1035)")
    b2 = ax1.barh(y + h / 2, re_auc, height=h, color=TEAL, label="INSECTS Recorrente (α_max=0,6425)")
    for bars, vals in ((b1, ab_auc), (b2, re_auc)):
        for bar, val in zip(bars, vals, strict=True):
            ax1.text(
                val + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}".replace(".", ","),
                va="center",
                ha="left",
                fontsize=7.6,
                color=INK,
            )
    ax1.set_yticks(y, labels, fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(*ab_auc, *re_auc) * 1.18)
    ax1.set_title(
        "AUC do Custo Combinado (↓ melhor)", fontsize=10.8, fontweight="bold", color=INK, loc="left"
    )
    ax1.set_xlabel("Integral trapezoidal de C_α(θ) sobre [0, α_max]", fontsize=9.2, color=INK)
    _style_axis(ax1)

    # Panel B: Retrains in low vs high regime (adaptação ao custo)
    reg_pols = [
        ("oracle", "Oracle"),
        ("upf", "UPF"),
        ("cara", "CARA-T"),
        ("cara_cumulative", "CARA-CT"),
        ("cara_periodic", "CARA-P"),
        ("adwin_5", "ADWIN-5%"),
        ("fhddm_5", "FHDDM-5%"),
        ("kswin_5", "KSWIN-5%"),
    ]
    yr = np.arange(len(reg_pols))
    ab_low = [abrupt[(k, "low", "retrains")][0] for k, _ in reg_pols]
    ab_high = [abrupt[(k, "high", "retrains")][0] for k, _ in reg_pols]
    re_low = [reocc[(k, "low", "retrains")][0] for k, _ in reg_pols]
    re_high = [reocc[(k, "high", "retrains")][0] for k, _ in reg_pols]

    hw = 0.2
    ax2.barh(yr - 1.5 * hw, ab_low, height=hw, color=BLUE, label="Abrupto · Low (0,1·α_max)")
    ax2.barh(yr - 0.5 * hw, ab_high, height=hw, color="#93c5fd", label="Abrupto · High (0,9·α_max)")
    ax2.barh(yr + 0.5 * hw, re_low, height=hw, color=TEAL, label="Recorrente · Low (0,1·α_max)")
    ax2.barh(yr + 1.5 * hw, re_high, height=hw, color="#99f6e4", label="Recorrente · High (0,9·α_max)")
    ax2.set_yticks(yr, [lbl for _, lbl in reg_pols], fontsize=8.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(*ab_low, *ab_high, *re_low, *re_high) + 1.0)
    ax2.set_title(
        "Retreinamentos: Regime Barato vs. Caro", fontsize=10.8, fontweight="bold", color=INK, loc="left"
    )
    ax2.set_xlabel("Média de retreinamentos em T = 8 lotes online (10 repetições)", fontsize=9.2, color=INK)
    _style_axis(ax2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper left", bbox_to_anchor=(0.07, 0.90), ncol=1, frameon=False, fontsize=8.6)
    fig.legend(h2, l2, loc="upper left", bbox_to_anchor=(0.53, 0.90), ncol=2, frameon=False, fontsize=8.4)
    fig.suptitle(
        "Protocolo UPF (2025) em INSECTS: AUC do Custo Combinado e Sensibilidade ao Custo",
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=12.6,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.02,
        "Fonte: homelab_insects (XGBoost eta=0,3, |D|=10.000, w=7 offline + T=8 online, "
        "split 80/20, α_max dinâmico por Oracle).",
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.77, bottom=0.13, wspace=0.28)
    return fig


def build_tradeoff_4streams(
    cara_runs: dict[str, dict[tuple[str, str, str], tuple[float, float, int]]],
) -> plt.Figure:
    titles = {
        "electricity": "Electricity (CARA · γ=20)",
        "airlines": "Airlines (CARA · 7.198 atributos)",
        "insects_abrupt_imbalanced": "INSECTS Abrupt Imbalanced (33 atrib.)",
        "insects_incremental_abrupt_reoccurring_imbalanced": "INSECTS Incremental Reoccurring (33 atrib.)",
    }
    pols = [
        ("oracle_staleness", "Oracle", "D"),
        ("cara", "CARA-T", "o"),
        ("cara_cumulative", "CARA-CT", "s"),
        ("cara_periodic", "CARA-P", "^"),
        ("markov", "Markov", "P"),
        ("adwin", "ADWIN", "X"),
        ("ddm", "DDM", "v"),
        ("frozen", "Congelado", "o"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    for ax, (ds_key, title) in zip(axes.flat, titles.items(), strict=True):
        summary = cara_runs[ds_key]
        for pkey, plabel, marker in pols:
            acc = summary[(pkey, "price_grid", "average_accuracy")][0]
            ret = summary[(pkey, "price_grid", "retrains")][0]
            scpe = summary[(pkey, "price_grid", "strategy_scpe")][0] if pkey != "oracle_staleness" else 0.0
            color = POLICY_COLORS.get(pkey, INK)
            ax.scatter(
                ret, acc, s=82, color=color, marker=marker, zorder=4, edgecolors="white", linewidths=0.7
            )
            lbl = plabel if pkey == "oracle_staleness" else f"{plabel} ({scpe:.0f}%)"
            dx = 0.6 if ret < 22 else -0.6
            ha = "left" if ret < 22 else "right"
            ax.annotate(
                lbl,
                (ret, acc),
                xytext=(dx * 5, 3),
                textcoords="offset points",
                fontsize=7.6,
                color=INK,
                ha=ha,
            )
        ax.set_title(title, fontsize=10.4, fontweight="bold", color=INK, loc="left")
        ax.set_xlabel("Média de retreinamentos (75 lotes online)", fontsize=8.8, color=INK)
        ax.set_ylabel("Acurácia média", fontsize=8.8, color=INK)
        ax.grid(True, linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Fronteira Custo vs. Acurácia no Protocolo CARA (2024) nos 4 Streams",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=12.8,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.015,
        "Rótulos entre parênteses indicam SCPE (%) médio vs. Oracle "
        "sobre 99 valores de κ (média de 5 seeds no homelab).",
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.09, hspace=0.32, wspace=0.22)
    return fig


def build_reproducao_papers(
    cara_elec: dict[tuple[str, str, str], tuple[float, float, int]],
    cara_air: dict[tuple[str, str, str], tuple[float, float, int]],
    upf_elec: dict[tuple[str, str, str], tuple[float, float, int]],
    upf_air: dict[tuple[str, str, str], tuple[float, float, int]],
    pub: dict[tuple[str, str, str, str, str], float],
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.4))

    # Left: CARA 2024 Electricity & Airlines SCPE (Published vs Observed)
    c_pols = [
        ("cara", "CARA-T"),
        ("cara_cumulative", "CARA-CT"),
        ("cara_periodic", "CARA-P"),
        ("markov", "Markov"),
    ]
    y1 = np.arange(len(c_pols))
    h = 0.2
    pub_e = [pub[("cara_2024", "electricity", k, "strategy_scpe", "price_grid")] for k, _ in c_pols]
    obs_e = [cara_elec[(k, "price_grid", "strategy_scpe")][0] for k, _ in c_pols]
    pub_a = [pub[("cara_2024", "airlines", k, "strategy_scpe", "price_grid")] for k, _ in c_pols]
    obs_a = [cara_air[(k, "price_grid", "strategy_scpe")][0] for k, _ in c_pols]

    ax1.barh(y1 - 1.5 * h, pub_e, height=h, color=BLUE, label="CARA Paper · Electricity")
    ax1.barh(y1 - 0.5 * h, obs_e, height=h, color="#93c5fd", label="Homelab · Electricity")
    ax1.barh(y1 + 0.5 * h, pub_a, height=h, color=ORANGE, label="CARA Paper · Airlines")
    ax1.barh(y1 + 1.5 * h, obs_a, height=h, color="#fdba74", label="Homelab · Airlines")
    ax1.set_yticks(y1, [lbl for _, lbl in c_pols], fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_title(
        "Reprodução CARA (2024): SCPE (%, ↓ melhor)", fontsize=10.6, fontweight="bold", color=INK, loc="left"
    )
    ax1.set_xlabel("SCPE (%) sobre 99 valores de κ", fontsize=9.0, color=INK)
    _style_axis(ax1)

    # Right: UPF 2025 Electricity & Airlines Combined Cost AUC
    u_pols = [
        ("oracle", "Oracle"),
        ("upf", "UPF"),
        ("cara", "CARA-T"),
        ("cara_periodic", "CARA-P"),
        ("adwin_5", "ADWIN-5%"),
        ("fhddm_5", "FHDDM-5%"),
    ]
    y2 = np.arange(len(u_pols))
    pub_ue = [pub[("upf_2025", "electricity", k, "combined_cost_auc", "price_grid")] for k, _ in u_pols]
    obs_ue = [upf_elec[(k, "price_grid", "combined_cost_auc")][0] for k, _ in u_pols]
    pub_ua = [pub[("upf_2025", "airlines", k, "combined_cost_auc", "price_grid")] for k, _ in u_pols]
    obs_ua = [upf_air[(k, "price_grid", "combined_cost_auc")][0] for k, _ in u_pols]

    ax2.barh(y2 - 1.5 * h, pub_ue, height=h, color=BLUE, label="UPF Paper · Electricity")
    ax2.barh(y2 - 0.5 * h, obs_ue, height=h, color="#93c5fd", label="Homelab · Electricity")
    ax2.barh(y2 + 0.5 * h, pub_ua, height=h, color=TEAL, label="UPF Paper · Airlines")
    ax2.barh(y2 + 1.5 * h, obs_ua, height=h, color="#99f6e4", label="Homelab · Airlines")
    ax2.set_yticks(y2, [lbl for _, lbl in u_pols], fontsize=9.0)
    ax2.invert_yaxis()
    ax2.set_title(
        "Reprodução UPF (2025): Combined Cost AUC (↓)",
        fontsize=10.6,
        fontweight="bold",
        color=INK,
        loc="left",
    )
    ax2.set_xlabel("AUC de C_α(θ) (T = 8 online, 10 repetições)", fontsize=9.0, color=INK)
    _style_axis(ax2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper left", bbox_to_anchor=(0.07, 0.90), ncol=2, frameon=False, fontsize=8.4)
    fig.legend(h2, l2, loc="upper left", bbox_to_anchor=(0.54, 0.90), ncol=2, frameon=False, fontsize=8.4)
    fig.suptitle(
        "Validação da Metodologia Atualizada frente aos Artigos Originais (Electricity & Airlines)",
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=12.6,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.02,
        "Fonte: published-reference-values.csv e evidencias/homelab "
        "(sem checkpoints em disco, scikit-multiflow 0.5.3).",
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.13, wspace=0.25)
    return fig


def main() -> None:
    apply_style()
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    cara_elec = _load_summary("cara-electricity")
    cara_air = _load_summary("cara-airlines")
    cara_in_ab = _load_summary("cara-insects-abrupt")
    cara_in_re = _load_summary("cara-insects-reoccurring")
    upf_elec = _load_summary("upf-electricity")
    upf_air = _load_summary("upf-airlines")
    upf_in_ab = _load_summary("upf-insects-abrupt")
    upf_in_re = _load_summary("upf-insects-reoccurring")
    pub = _load_published()

    figs = {
        "cara-insects-comparativo.svg": build_cara_insects_comparativo(cara_in_ab, cara_in_re),
        "upf-insects-auc-e-regimes.svg": build_upf_insects_auc_e_regimes(upf_in_ab, upf_in_re),
        "tradeoff-acuracia-retreinos-4streams.svg": build_tradeoff_4streams(
            {
                "electricity": cara_elec,
                "airlines": cara_air,
                "insects_abrupt_imbalanced": cara_in_ab,
                "insects_incremental_abrupt_reoccurring_imbalanced": cara_in_re,
            }
        ),
        "reproducao-papers-4streams.svg": build_reproducao_papers(
            cara_elec, cara_air, upf_elec, upf_air, pub
        ),
    }

    for name, fig in figs.items():
        path = OUT_FIG_DIR / name
        fig.savefig(path, format="svg")
        plt.close(fig)
        print("Wrote:", path)


if __name__ == "__main__":
    main()
