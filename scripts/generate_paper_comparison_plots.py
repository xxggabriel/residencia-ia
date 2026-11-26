"""Generate paper comparison SVGs using 3-seed AWS benchmark means (seeds 42, 43, 44)."""

from __future__ import annotations

import csv
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/gabrieloliveira/residencia-ia")
OUT_FIG_ROOT = ROOT / "docs_v2/relatorios/figuras/papers"
TMP_OUT = Path("/tmp/papers-svg-build")

sys.path.insert(0, str(ROOT / "notebooks"))
import matplotlib.pyplot as plt
import numpy as np
from benchmark_plot_style import BLUE, INK, MUTED, ORANGE, apply_style


def load_aws_3seed_means() -> dict[tuple[str, str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with open(ROOT / "evidencias/aws-benchmarks/metrics-by-seed.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (
                r["dataset"] in ("electricity", "airlines")
                and r["metric_scope"] == "prequential"
                and r["metric"] in ("cost_scpe", "upf_paper_auc", "accuracy", "retrain_count")
            ):
                grouped[(r["metric"], r["dataset"], r["policy"])].append(float(r["value"]))
    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for k, vals in grouped.items():
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[k] = (m, sd)
    return out


def build_papers_vs_aws(aws_means: dict[tuple[str, str, str], tuple[float, float]]) -> plt.Figure:
    pub_rows = list(
        csv.DictReader(
            open(ROOT / "evidencias/aws-benchmarks/papers/published-reference-values.csv", encoding="utf-8")
        )
    )

    # Panel A: SCPE (%) - CARA 2024 Paper vs AWS (3-seed mean)
    cara_pols = [("CARA-T", "cara"), ("CARA-CT", "cara_cumulative"), ("CARA-P", "cara_periodic")]
    labels_a, pub_elec_a, aws_elec_a, pub_air_a, aws_air_a = [], [], [], [], []
    for lbl, pol_key in cara_pols:
        labels_a.append(lbl)
        pub_elec_a.append(
            float(
                [
                    r["published_value"]
                    for r in pub_rows
                    if r["paper_id"] == "cara_2024"
                    and r["metric"] == "strategy_scpe"
                    and r["dataset"] == "electricity"
                    and r["policy_label"] == lbl
                ][0]
            )
        )
        pub_air_a.append(
            float(
                [
                    r["published_value"]
                    for r in pub_rows
                    if r["paper_id"] == "cara_2024"
                    and r["metric"] == "strategy_scpe"
                    and r["dataset"] == "airlines"
                    and r["policy_label"] == lbl
                ][0]
            )
        )
        aws_elec_a.append(aws_means[("cost_scpe", "electricity", pol_key)][0])
        aws_air_a.append(aws_means[("cost_scpe", "airlines", pol_key)][0])

    # Panel B: Combined Cost AUC - UPF 2025 Paper vs AWS (3-seed mean)
    upf_pols = [
        ("UPF", "upf"),
        ("CARA-T", "cara"),
        ("CARA-CT", "cara_cumulative"),
        ("CARA-P", "cara_periodic"),
        ("Oracle de erro", "oracle"),
    ]
    labels_b, pub_elec_b, aws_elec_b, pub_air_b, aws_air_b = [], [], [], [], []
    for lbl, pol_key in upf_pols:
        labels_b.append("Oracle" if lbl == "Oracle de erro" else lbl)
        pub_elec_b.append(
            float(
                [
                    r["published_value"]
                    for r in pub_rows
                    if r["paper_id"] == "upf_2025"
                    and r["metric"] == "combined_cost_auc"
                    and r["dataset"] == "electricity"
                    and r["policy_label"] == lbl
                ][0]
            )
        )
        pub_air_b.append(
            float(
                [
                    r["published_value"]
                    for r in pub_rows
                    if r["paper_id"] == "upf_2025"
                    and r["metric"] == "combined_cost_auc"
                    and r["dataset"] == "airlines"
                    and r["policy_label"] == lbl
                ][0]
            )
        )
        aws_elec_b.append(aws_means[("upf_paper_auc", "electricity", pol_key)][0])
        aws_air_b.append(aws_means[("upf_paper_auc", "airlines", pol_key)][0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.5), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.73, bottom=0.20, wspace=0.34)

    # Panel A
    y1 = np.arange(len(labels_a))
    h = 0.19
    b1 = ax1.barh(y1 - 1.5 * h, pub_elec_a, height=h, color=BLUE, label="Artigo · Electricity")
    b2 = ax1.barh(
        y1 - 0.5 * h, aws_elec_a, height=h, color="#93c5fd", label="AWS (média 3 seeds) · Electricity"
    )
    b3 = ax1.barh(y1 + 0.5 * h, pub_air_a, height=h, color=ORANGE, label="Artigo · Airlines")
    b4 = ax1.barh(y1 + 1.5 * h, aws_air_a, height=h, color="#fdba74", label="AWS (média 3 seeds) · Airlines")
    for bars, vals in ((b1, pub_elec_a), (b2, aws_elec_a), (b3, pub_air_a), (b4, aws_air_a)):
        for bar, val in zip(bars, vals, strict=True):
            ax1.text(
                val + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%".replace(".", ","),
                va="center",
                ha="left",
                fontsize=8.0,
                color=INK,
            )
    ax1.set_yticks(y1, labels_a, fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 82)
    ax1.set_title("CARA (2024) vs. AWS · SCPE (% ↓)", fontsize=10.8, fontweight="bold", color=INK, loc="left")
    ax1.set_xlabel("Excesso de custo relativo ao Oracle (%)", fontsize=9.2, color=INK)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B
    y2 = np.arange(len(labels_b))
    c1 = ax2.barh(y2 - 1.5 * h, pub_elec_b, height=h, color=BLUE)
    c2 = ax2.barh(y2 - 0.5 * h, aws_elec_b, height=h, color="#93c5fd")
    c3 = ax2.barh(y2 + 0.5 * h, pub_air_b, height=h, color=ORANGE)
    c4 = ax2.barh(y2 + 1.5 * h, aws_air_b, height=h, color="#fdba74")
    for bars, vals in ((c1, pub_elec_b), (c2, aws_elec_b), (c3, pub_air_b), (c4, aws_air_b)):
        for bar, val in zip(bars, vals, strict=True):
            ax2.text(
                val + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}".replace(".", ","),
                va="center",
                ha="left",
                fontsize=8.0,
                color=INK,
            )
    ax2.set_yticks(y2, labels_b, fontsize=9.2)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 5.1)
    ax2.set_title(
        "UPF (2025) vs. AWS · AUC de custo (↓)", fontsize=10.8, fontweight="bold", color=INK, loc="left"
    )
    ax2.set_xlabel("AUC do custo combinado", fontsize=9.2, color=INK)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.10, 0.86), ncol=2, frameon=False, fontsize=8.8
    )
    fig.suptitle(
        "Valores publicados nos artigos base vs. médias na AWS (3 seeds)",
        x=0.11,
        y=0.95,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.11,
        0.02,
        "Fonte: published-reference-values.csv (artigos) e metrics-by-seed.csv (campanha na AWS, média das seeds 42, 43 e 44).\n"
        "Cotejo estritamente descritivo: horizontes (T=16 na AWS vs. T=8 no UPF), modelos (RF vs. XGBoost) e grades de custo diferem.",
        fontsize=8.6,
        color=MUTED,
        va="bottom",
    )
    return fig


def build_accuracy_and_retrains_vs_cara(
    aws_means: dict[tuple[str, str, str], tuple[float, float]],
) -> plt.Figure:
    pub_rows = list(
        csv.DictReader(
            open(ROOT / "evidencias/aws-benchmarks/papers/published-reference-values.csv", encoding="utf-8")
        )
    )

    pols = [
        ("Congelado", "Congelado", "frozen"),
        ("CARA-T", "CARA-T", "cara"),
        ("CARA-CT", "CARA-CT", "cara_cumulative"),
        ("CARA-P", "CARA-P", "cara_periodic"),
        ("Oracle", "Oracle de obsolescência", "oracle"),
        ("ADWIN", "ADWIN", "drift_adwin"),
        ("DDM", "DDM", "drift_ddm"),
    ]
    labels = [p[0] for p in pols]

    def get_cara_val(metric: str, ds: str, pub_lbl: str) -> float:
        return float(
            [
                r["published_value"]
                for r in pub_rows
                if r["paper_id"] == "cara_2024"
                and r["metric"] == metric
                and r["dataset"] == ds
                and r["policy_label"] == pub_lbl
            ][0]
        )

    pub_acc_elec = [get_cara_val("average_accuracy", "electricity", p[1]) for p in pols]
    aws_acc_elec = [aws_means[("accuracy", "electricity", p[2])][0] for p in pols]
    pub_acc_air = [get_cara_val("average_accuracy", "airlines", p[1]) for p in pols]
    aws_acc_air = [aws_means[("accuracy", "airlines", p[2])][0] for p in pols]

    pub_ret_elec = [get_cara_val("retrains", "electricity", p[1]) for p in pols]
    aws_ret_elec = [aws_means[("retrain_count", "electricity", p[2])][0] for p in pols]
    pub_ret_air = [get_cara_val("retrains", "airlines", p[1]) for p in pols]
    aws_ret_air = [aws_means[("retrain_count", "airlines", p[2])][0] for p in pols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.8), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.73, bottom=0.19, wspace=0.32)

    y = np.arange(len(labels))
    h = 0.19

    # Panel A: Accuracy
    a1 = ax1.barh(y - 1.5 * h, pub_acc_elec, height=h, color=BLUE, label="CARA (2024) · Electricity")
    a2 = ax1.barh(
        y - 0.5 * h, aws_acc_elec, height=h, color="#93c5fd", label="AWS (média 3 seeds) · Electricity"
    )
    a3 = ax1.barh(y + 0.5 * h, pub_acc_air, height=h, color=ORANGE, label="CARA (2024) · Airlines")
    a4 = ax1.barh(y + 1.5 * h, aws_acc_air, height=h, color="#fdba74", label="AWS (média 3 seeds) · Airlines")
    for bars, vals in ((a1, pub_acc_elec), (a2, aws_acc_elec), (a3, pub_acc_air), (a4, aws_acc_air)):
        for bar, val in zip(bars, vals, strict=True):
            ax1.text(
                val + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}".replace(".", ","),
                va="center",
                ha="left",
                fontsize=7.6,
                color=INK,
            )
    ax1.set_yticks(y, labels, fontsize=9.0)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.92)
    ax1.set_title(
        "Acurácia média por lote / coorte comum (↑)", fontsize=10.6, fontweight="bold", color=INK, loc="left"
    )
    ax1.set_xlabel("Acurácia (fração 0 a 1)", fontsize=9.2, color=INK)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Retrain count
    r1 = ax2.barh(y - 1.5 * h, pub_ret_elec, height=h, color=BLUE)
    r2 = ax2.barh(y - 0.5 * h, aws_ret_elec, height=h, color="#93c5fd")
    r3 = ax2.barh(y + 0.5 * h, pub_ret_air, height=h, color=ORANGE)
    r4 = ax2.barh(y + 1.5 * h, aws_ret_air, height=h, color="#fdba74")
    for bars, vals in ((r1, pub_ret_elec), (r2, aws_ret_elec), (r3, pub_ret_air), (r4, aws_ret_air)):
        for bar, val in zip(bars, vals, strict=True):
            ax2.text(
                val + 0.45,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}".replace(".", ","),
                va="center",
                ha="left",
                fontsize=7.6,
                color=INK,
            )
    ax2.set_yticks(y, labels, fontsize=9.0)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 34.0)
    ax2.set_title(
        "Decisões de retreinamento (média de seeds)", fontsize=10.6, fontweight="bold", color=INK, loc="left"
    )
    ax2.set_xlabel("Contagem de retreinamentos", fontsize=9.2, color=INK)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    handles, leg_labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.86),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )
    fig.suptitle(
        "Acurácia e frequência de retreinamento: CARA (2024) vs. médias na AWS",
        x=0.11,
        y=0.95,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.11,
        0.02,
        "Fonte: published-reference-values.csv (Tabelas 3 e 4 do CARA, n=5 seeds) e metrics-by-seed.csv (AWS, média das seeds 42, 43 e 44).\n"
        "Na AWS, políticas de custo operam em alpha único (1,0), realizando 1 retreinamento vs. até 27,7 no CARA (média sobre 99 valores de kappa).",
        fontsize=8.6,
        color=MUTED,
        va="bottom",
    )
    return fig


def main() -> None:
    apply_style()
    if TMP_OUT.exists():
        shutil.rmtree(TMP_OUT)
    TMP_OUT.mkdir(parents=True, exist_ok=True)
    OUT_FIG_ROOT.mkdir(parents=True, exist_ok=True)

    aws_means = load_aws_3seed_means()

    fig1 = build_papers_vs_aws(aws_means)
    fig1.savefig(TMP_OUT / "papers-valores-publicados-referencia.svg", format="svg")
    plt.close(fig1)

    fig2 = build_accuracy_and_retrains_vs_cara(aws_means)
    fig2.savefig(TMP_OUT / "papers-acuracia-e-retreinos-referencia.svg", format="svg")
    plt.close(fig2)

    for svg in TMP_OUT.glob("*.svg"):
        shutil.copy2(svg, OUT_FIG_ROOT / svg.name)
        print("Copied", OUT_FIG_ROOT / svg.name)


if __name__ == "__main__":
    main()
