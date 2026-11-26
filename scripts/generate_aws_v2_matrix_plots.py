"""Fix aws-versoes-e-retreinos-timeline.svg & papers-valores-publicados-referencia.svg,
and generate 5 new SVG plots from evidencias/aws-benchmarks-v2/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("/Users/gabrieloliveira/residencia-ia")
EV2 = ROOT / "evidencias/aws-benchmarks-v2"
OUT_FIG_ROOT = ROOT / "docs_v2/relatorios/figuras"

sys.path.insert(0, str(ROOT / "notebooks"))
import matplotlib.pyplot as plt
import numpy as np
from benchmark_athena_figures import SHORT
from benchmark_plot_style import BLUE, INK, MUTED, ORANGE, TEAL, apply_style
from matplotlib.patches import Patch

PURPLE = "#7C3AED"
RED = "#DC2626"

DATASET_LABELS = {
    "electricity": "Electricity",
    "airlines": "Airlines",
    "insects_abrupt_imbalanced": "INSECTS Abrupto",
    "insects_incremental_abrupt_reoccurring_imbalanced": "INSECTS Incr. Abr. Recorr.",
}

AWS_CHANGEPOINTS = {
    "insects_abrupt_imbalanced": [83859, 128651, 182320, 242883, 268380],
    "insects_incremental_abrupt_reoccurring_imbalanced": [150683, 301365],
}


def save_svg(fig: plt.Figure, rel_path: str):
    out_path = OUT_FIG_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


# 1. Fixed aws-versoes-e-retreinos-timeline.svg: clean discrete block-action grid (28 blocks x 8 policies)
def build_versoes_timeline_v2() -> plt.Figure:
    mvs = list(csv.DictReader(open(EV2 / "athena/f109_model_versions.csv", encoding="utf-8")))
    seed42_runs = {
        r["run_id"]
        for r in csv.DictReader(open(EV2 / "athena/f109_runs_uris.csv", encoding="utf-8"))
        if r["seed"] == "42"
    }
    mvs42 = [r for r in mvs if r["run_id"] in seed42_runs]

    # Get exact block end_seqs per dataset from windows_full_28blocks.csv
    win_rows = list(csv.DictReader(open(EV2 / "s3-reports/windows_full_28blocks.csv", encoding="utf-8")))
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]
    act_code = {"none": 0, "initial": 1, "reset_retrain": 2, "continue": 3}
    act_colors = {0: "#E2E8F0", 1: PURPLE, 2: BLUE, 3: TEAL}

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.79, bottom=0.14, wspace=0.28, hspace=0.44)

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        # 27 evaluation blocks (blocks 1..27 after warmup block 0)
        ds_wins = sorted({int(w["end_seq"]) for w in win_rows if w["dataset"] == ds})
        # Map initial warmup end_seq + 27 blocks
        all_ends = sorted({int(r["end_seq"]) for r in mvs42 if r["dataset"] == ds} | set(ds_wins))
        # Use uniform 28 block indices 0..27 sorted by end_seq
        end_to_idx = {es: idx for idx, es in enumerate(all_ends)}
        n_blk = len(all_ends)

        grid = np.zeros((len(policies), n_blk, 3))
        for y_idx, pol in enumerate(policies):
            pol_map = {
                end_to_idx[int(r["end_seq"])]: act_code[r["action"]]
                for r in mvs42
                if r["dataset"] == ds and r["policy"] == pol
            }
            for x_idx in range(n_blk):
                code = pol_map.get(x_idx, 0)
                hex_c = act_colors[code].lstrip("#")
                grid[y_idx, x_idx] = [int(hex_c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

        ax.imshow(grid, aspect="auto", interpolation="nearest")
        # Subtle white grid lines between cells
        ax.set_xticks(np.arange(-0.5, n_blk, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(policies), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        if ds in AWS_CHANGEPOINTS:
            for cp in AWS_CHANGEPOINTS[ds]:
                # Find first block whose end_seq >= cp
                blk_idx = next((i for i, es in enumerate(all_ends) if es >= cp), n_blk - 1)
                ax.axvline(blk_idx - 0.5, color=RED, ls="--", lw=1.6, zorder=5)

        ax.set_yticks(range(len(policies)), [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.set_xticks([0, 6, 13, 20, n_blk - 1], [f"B{i}" for i in (0, 6, 13, 20, n_blk - 1)], fontsize=8.8)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Bloco do stream (B0 = aquecimento inicial)", fontsize=9.2, color=INK)

    legend_items = [
        Patch(facecolor=PURPLE, label="Treino inicial (B0)"),
        Patch(facecolor=BLUE, label="Retreino do zero (reset_retrain)"),
        Patch(facecolor=TEAL, label="Atualização incremental (continue)"),
        Patch(facecolor="#E2E8F0", edgecolor="#cbd5e1", label="Sem atualização"),
        Patch(facecolor="none", edgecolor=RED, linestyle="--", linewidth=1.5, label="Drift real (INSECTS)"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(0.13, 0.90),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )
    fig.suptitle(
        "Mapa de ações por bloco: treino inicial, retreino do zero e incremental (F109)",
        x=0.14,
        y=0.965,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.14,
        0.02,
        "Fonte: Athena continual_bench.model_versions (885 versões promovidas na F109; seed 42).\n"
        "Cada célula representa um bloco do stream; linhas tracejadas vermelhas marcam blocos com fronteiras reais de drift nos INSECTS.",
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    return fig


# 2. Fixed papers-valores-publicados-referencia.svg: add exact value labels at the end of every bar
def build_papers_ref_v2() -> plt.Figure:
    rows = list(
        csv.DictReader(
            open(ROOT / "evidencias/aws-benchmarks/papers/published-reference-values.csv", encoding="utf-8")
        )
    )
    cara_pols = ["CARA-T", "CARA-CT", "CARA-P", "Markov", "Congelado"]
    elec_scpe = [
        float(
            [
                r["published_value"]
                for r in rows
                if r["paper_id"] == "cara_2024"
                and r["metric"] == "strategy_scpe"
                and r["dataset"] == "electricity"
                and r["policy_label"] == p
            ][0]
        )
        for p in cara_pols
    ]
    air_scpe = [
        float(
            [
                r["published_value"]
                for r in rows
                if r["paper_id"] == "cara_2024"
                and r["metric"] == "strategy_scpe"
                and r["dataset"] == "airlines"
                and r["policy_label"] == p
            ][0]
        )
        for p in cara_pols
    ]

    upf_pols = ["UPF", "CARA-T", "CARA-CT", "CARA-P", "Oracle de erro"]
    elec_auc = [
        float(
            [
                r["published_value"]
                for r in rows
                if r["paper_id"] == "upf_2025"
                and r["metric"] == "combined_cost_auc"
                and r["dataset"] == "electricity"
                and r["policy_label"] == p
            ][0]
        )
        for p in upf_pols
    ]
    air_auc = [
        float(
            [
                r["published_value"]
                for r in rows
                if r["paper_id"] == "upf_2025"
                and r["metric"] == "combined_cost_auc"
                and r["dataset"] == "airlines"
                and r["policy_label"] == p
            ][0]
        )
        for p in upf_pols
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.8), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.76, bottom=0.22, wspace=0.34)

    y1 = np.arange(len(cara_pols))
    h = 0.34
    b1e = ax1.barh(y1 - h / 2, elec_scpe, height=h, color=BLUE, label="Electricity")
    b1a = ax1.barh(y1 + h / 2, air_scpe, height=h, color=ORANGE, label="Airlines")
    for b, v in list(zip(b1e, elec_scpe, strict=True)) + list(zip(b1a, air_scpe, strict=True)):
        ax1.text(
            v * 1.12,
            b.get_y() + b.get_height() / 2,
            f"{v:.2f}%".replace(".", ","),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
        )
    ax1.set_yticks(y1, cara_pols, fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_xlim(3.5, 420.0)
    ax1.set_title("CARA (2024) · SCPE publicado (% ↓)", fontsize=11, fontweight="bold", color=INK, loc="left")
    ax1.set_xlabel("Excesso de custo relativo ao Oracle (%, log)", fontsize=9.2, color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    y2 = np.arange(len(upf_pols))
    b2e = ax2.barh(y2 - h / 2, elec_auc, height=h, color=BLUE, label="Electricity")
    b2a = ax2.barh(y2 + h / 2, air_auc, height=h, color=ORANGE, label="Airlines")
    for b, v in list(zip(b2e, elec_auc, strict=True)) + list(zip(b2a, air_auc, strict=True)):
        ax2.text(
            v + 0.015,
            b.get_y() + b.get_height() / 2,
            f"{v:.3f}".replace(".", ","),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
        )
    ax2.set_yticks(y2, upf_pols, fontsize=9.2)
    ax2.invert_yaxis()
    ax2.set_xlim(2.0, 3.18)
    ax2.set_title(
        "UPF (2025) · AUC de custo publicado (↓)", fontsize=11, fontweight="bold", color=INK, loc="left"
    )
    ax2.set_xlabel("AUC do custo combinado", fontsize=9.2, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Valores publicados de referência nos artigos base (CARA 2024 e UPF 2025)",
        x=0.11,
        y=0.93,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.11,
        0.02,
        "Fonte: evidencias/aws-benchmarks/papers/published-reference-values.csv. Métricas e protocolos próprios dos artigos\n"
        "(Random Forest no CARA; XGBoost eta=0,3 em 15 lotes no UPF), não comparáveis numericamente à campanha AWS F109.",
        fontsize=8.8,
        color=MUTED,
        va="bottom",
    )
    return fig


# 3. Confusion matrices (prequential and holdout) across 4 streams x 4 key policies
def build_confusion_grid(scope: str, title: str, subtitle: str) -> plt.Figure:
    rows = [
        r
        for r in csv.DictReader(open(EV2 / "s3-reports/confusion_matrices.csv", encoding="utf-8"))
        if r["seed"] == "42" and r["scope"] == scope
    ]
    cm_map = {(r["dataset"], r["policy"]): r for r in rows}
    datasets = list(DATASET_LABELS.keys())
    pols = ["frozen", "drift_adwin", "incremental", "cara"]

    fig, axes = plt.subplots(4, 4, figsize=(9.2, 8.6), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.09, wspace=0.22, hspace=0.32)

    for i, ds in enumerate(datasets):
        for j, pol in enumerate(pols):
            ax = axes[i, j]
            r = cm_map[(ds, pol)]
            tn, fp, fn, tp = int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"])
            mat = np.array([[tn, fp], [fn, tp]], dtype=float)
            row_sums = mat.sum(axis=1, keepdims=True)
            pct = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0) * 100.0
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
            for r_i in range(2):
                for c_i in range(2):
                    val_cnt = int(mat[r_i, c_i])
                    val_pct = pct[r_i, c_i]
                    txt_col = "white" if val_pct > 55 else INK
                    ax.text(
                        c_i,
                        r_i,
                        f"{val_pct:.1f}%\n({val_cnt:,})".replace(",", "."),
                        ha="center",
                        va="center",
                        fontsize=8.0,
                        color=txt_col,
                        fontweight="bold",
                    )
            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"] if i == 3 else ["", ""], fontsize=8.2)
            ax.set_yticks([0, 1], ["Real 0", "Real 1"] if j == 0 else ["", ""], fontsize=8.2)
            if i == 0:
                ax.set_title(SHORT.get(pol, pol), fontsize=10.5, fontweight="bold", color=INK, pad=6)
            if j == 0:
                ax.set_ylabel(DATASET_LABELS[ds], fontsize=9.5, fontweight="bold", color=INK, labelpad=6)

    fig.suptitle(title, x=0.11, y=0.95, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.11, 0.015, subtitle, fontsize=8.8, color=MUTED, va="bottom")
    return fig


# 4. Per-class precision, recall, F1 (Class 0 vs Class 1)
def build_class_metrics_figure() -> plt.Figure:
    rows = [
        r
        for r in csv.DictReader(open(EV2 / "s3-reports/class_metrics.csv", encoding="utf-8"))
        if r["seed"] == "42" and r["scope"] == "prequential"
    ]
    cmap = {(r["dataset"], r["policy"], r["class_label"]): r for r in rows}
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.0), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.79, bottom=0.14, wspace=0.30, hspace=0.44)
    y = np.arange(len(policies))
    h = 0.25

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        f1_0 = [float(cmap[(ds, p, "0")]["f1"]) for p in policies]
        f1_1 = [float(cmap[(ds, p, "1")]["f1"]) for p in policies]
        rec_1 = [float(cmap[(ds, p, "1")]["recall"]) for p in policies]

        ax.barh(y - h, f1_0, height=h, color=BLUE, label="F1 Classe 0 (negativa)")
        ax.barh(y, f1_1, height=h, color=ORANGE, label="F1 Classe 1 (positiva)")
        ax.barh(y + h, rec_1, height=h, color=TEAL, label="Revocação Classe 1 (positiva)")
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Pontuação (0 a 1)", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=3, frameon=False, fontsize=9.3
    )
    fig.suptitle(
        "Desempenho por classe (F1 Classe 0 vs. F1 e Revocação Classe 1 · F109)",
        x=0.14,
        y=0.96,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.14,
        0.02,
        "Fonte: evidencias/aws-benchmarks-v2/s3-reports/class_metrics.csv (escopo prequencial completo).\n"
        "Em streams desbalanceados (Airlines e INSECTS), a classe positiva (1) concentra a maior sensibilidade às atualizações.",
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    return fig


# 5. ROC-AUC vs PR-AUC across policies and streams
def build_roc_vs_pr_auc() -> plt.Figure:
    roc = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(
            open(ROOT / "evidencias/aws-benchmarks/quality-summary.csv", encoding="utf-8")
        )
        if r["metric"] == "roc_auc"
    }
    pr = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(
            open(ROOT / "evidencias/aws-benchmarks/quality-summary.csv", encoding="utf-8")
        )
        if r["metric"] == "pr_auc"
    }
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.30, hspace=0.44)
    y = np.arange(len(policies))
    h = 0.34
    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        ax.barh(y - h / 2, [roc[(ds, p)] for p in policies], height=h, color=BLUE, label="ROC-AUC (↑)")
        ax.barh(y + h / 2, [pr[(ds, p)] for p in policies], height=h, color=ORANGE, label="PR-AUC / AP (↑)")
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_xlim(0.2, 1.0)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Área sob a curva", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=2, frameon=False, fontsize=10
    )
    fig.suptitle(
        "ROC-AUC vs. PR-AUC na coorte comum pós-calibração (F109)",
        x=0.14,
        y=0.96,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.14,
        0.02,
        "Fonte: evidencias/aws-benchmarks/quality-summary.csv. Em INSECTS Incr. Abr. Recorr.,\n"
        "a PR-AUC cai fortemente no Congelado (0,279) mesmo quando a ROC-AUC permanece acima de 0,58.",
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    return fig


# 6. Model size (KB) and training seconds per version from Athena model_versions
def build_model_size_and_train_time() -> plt.Figure:
    mvs = list(csv.DictReader(open(EV2 / "athena/f109_model_versions.csv", encoding="utf-8")))
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.32, hspace=0.44)
    y = np.arange(len(policies))

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        kb_means = []
        ts_means = []
        for pol in policies:
            sub = [r for r in mvs if r["dataset"] == ds and r["policy"] == pol]
            kb_means.append(float(np.mean([float(r["bytes"]) / 1024.0 for r in sub])))
            ts_means.append(float(np.mean([float(r["train_seconds"]) for r in sub])))

        ax.barh(y, kb_means, height=0.58, color=PURPLE, alpha=0.82, label="Tamanho médio por versão (KB)")
        for idx, (kb, ts) in enumerate(zip(kb_means, ts_means, strict=True)):
            ax.text(
                kb + max(kb_means) * 0.02,
                idx,
                f"{kb:.0f} KB ({ts:.1f}s/ver)",
                va="center",
                fontsize=8.4,
                color=INK,
            )
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_xlim(0, max(kb_means) * 1.48)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Tamanho serializado médio por versão (KB)", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Tamanho serializado por versão e tempo médio de ajuste (Athena F109)",
        x=0.14,
        y=0.95,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.14,
        0.02,
        "Fonte: Athena continual_bench.model_versions (885 versões promovidas). Em Incremental, o modelo XGBoost\n"
        "acumula rodadas de boosting ao longo dos 23 ajustes, crescendo de ~200 KB para mais de 4 MB na versão final.",
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    return fig


def main():
    apply_style()
    save_svg(build_versoes_timeline_v2(), "drift-e-retreinos/aws-versoes-e-retreinos-timeline.svg")
    save_svg(build_papers_ref_v2(), "papers/papers-valores-publicados-referencia.svg")
    save_svg(
        build_confusion_grid(
            "prequential",
            "Matrizes de confusão normalizadas no fluxo prequencial completo (F109)",
            "Fonte: evidencias/aws-benchmarks-v2/s3-reports/confusion_matrices.csv (seed 42). Percentuais normalizados por linha (classe real).",
        ),
        "qualidade/aws-matriz-confusao-coorte.svg",
    )
    save_svg(
        build_confusion_grid(
            "holdout",
            "Matrizes de confusão normalizadas no holdout final de 10% (F109)",
            "Fonte: evidencias/aws-benchmarks-v2/s3-reports/confusion_matrices.csv (seed 42, escopo holdout). Percentuais normalizados por linha.",
        ),
        "qualidade/aws-matriz-confusao-holdout.svg",
    )
    save_svg(build_class_metrics_figure(), "qualidade/aws-metricas-por-classe.svg")
    save_svg(build_roc_vs_pr_auc(), "qualidade/aws-roc-e-pr-auc.svg")
    save_svg(build_model_size_and_train_time(), "recursos-e-custos/aws-tamanho-modelo-e-tempo-treino.svg")
    print("Generated/updated 7 SVG figures successfully")


if __name__ == "__main__":
    main()
