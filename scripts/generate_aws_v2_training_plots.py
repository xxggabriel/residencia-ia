"""Generate AWS-v2 plots matching the 3 attached reference styles:
1) Per-batch candidate model curves + 95% Wilson CI + deployed trajectory + Retreino lines (4 streams)
2) Stream-position strip heatmap of accuracy, alarms, and updates for INSECTS Abrupto & Recorrente.
Uses ONLY evidencias/aws-benchmarks-v2/.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/gabrieloliveira/residencia-ia")
V2_DIR = ROOT / "evidencias/aws-benchmarks-v2"
OUT_REGIMES = ROOT / "docs_v2/relatorios/figuras/regimes-e-matrizes"
OUT_DRIFT = ROOT / "docs_v2/relatorios/figuras/drift-e-retreinos"

sys.path.insert(0, str(ROOT / "notebooks"))
from benchmark_plot_style import INK, MUTED, apply_style


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return p, p
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_block_sizes() -> dict[tuple[str, int], int]:
    sizes: dict[tuple[str, int], int] = {}
    with open(V2_DIR / "s3-reports/windows_full_28blocks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["policy"] == "frozen" and r["seed"] == "42":
                sizes[(r["dataset"], int(r["block"]))] = int(r["n"])
    return sizes


def build_batch_accuracy_plot(
    dataset: str,
    title_prefix: str,
    policy_key: str,
    deployed_label: str,
    retrain_blocks: list[int],
    block_sizes: dict[tuple[str, int], int],
    decisions: dict,
) -> plt.Figure:
    mat_data = json.load(open(V2_DIR / f"s3-reports/retraining_matrices/{dataset}__{policy_key}__accuracy_matrix.json", encoding="utf-8"))
    mat = mat_data["matrix"]  # 24 x 24
    dep = decisions[f"{dataset}__{policy_key}"]
    served_by = dep["served_by"]  # 16 values for online blocks 8..23

    # Show candidate models M7..M23 across batches D7..D23 (17 batches)
    start_b, end_b = 7, 23
    batches = list(range(start_b, end_b + 1))
    x_labels = [f"D{b}" for b in batches]
    cand_models = [7, 9, 11, 13, 15, 16, 17, 19, 21, 23]
    if dataset == "insects_incremental_abrupt_reoccurring_imbalanced" and 16 not in cand_models:
        cand_models.append(16)
    cand_models = sorted(set(cand_models))

    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(1, len(cand_models) - 1)) for i in range(len(cand_models))]

    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.23)

    y_min, y_max = 1.0, 0.0
    for idx, m_idx in enumerate(cand_models):
        xs, ys, lo_ys, hi_ys = [], [], [], []
        for b in batches:
            val = mat[m_idx][b]
            if val is not None:
                n_b = block_sizes.get((dataset, b), 1000)
                lo, hi = wilson_ci(float(val), n_b)
                xs.append(b)
                ys.append(float(val) * 100.0)
                lo_ys.append(lo * 100.0)
                hi_ys.append(hi * 100.0)
        if xs:
            y_min = min(y_min, min(lo_ys) / 100.0)
            y_max = max(y_max, max(hi_ys) / 100.0)
            ax.fill_between(xs, lo_ys, hi_ys, color=colors[idx], alpha=0.10, linewidth=0)
            ax.plot(xs, ys, marker="o", markersize=3.3, linewidth=1.35, color=colors[idx], label=f"Modelo D{m_idx}")

    # Deployed "Em uso" trajectory: D7 served by M7 (or M0 on D7 -> M7 from D8..D23; include D7=mat[0][7] or mat[7][8])
    dep_xs = list(range(8, 24))
    dep_ys = [float(mat[served_by[i]][8 + i]) * 100.0 for i in range(len(served_by))]
    ax.plot(
        dep_xs,
        dep_ys,
        linestyle="-.",
        linewidth=2.3,
        color="#0f172a",
        marker="o",
        markersize=5.0,
        label=f"Em uso · {deployed_label}",
        zorder=10,
    )

    y_lo = max(0.0, math.floor(y_min * 100.0 / 5.0) * 5.0 - 3.0)
    y_hi = min(100.0, math.ceil(y_max * 100.0 / 5.0) * 5.0 + 5.0)
    ax.set_ylim(y_lo, y_hi)

    for i_r, rb in enumerate(retrain_blocks):
        ax.axvline(rb, color="#475569", linestyle=":", linewidth=1.1, zorder=5)
        y_tag = y_hi - 2.2 - (i_r % 2) * 3.5
        ax.text(
            rb,
            y_tag,
            "Retreino",
            ha="center",
            va="top",
            fontsize=8.0,
            color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#fca5a5", linewidth=0.8),
            zorder=12,
        )

    ax.set_xticks(batches, x_labels, fontsize=8.4)
    ax.set_ylabel("Acurácia (%)", fontsize=9.4, color=INK)
    ax.set_xlabel("Lote avaliado · intervalos de Wilson de 95% para a acurácia em cada lote", fontsize=8.8, color=MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(True, axis="y", linestyle="-", alpha=0.22, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=6,
        frameon=False,
        fontsize=8.0,
        handlelength=1.8,
    )

    n_med = int(statistics.median([block_sizes.get((dataset, b), 0) for b in batches]))
    fig.suptitle(f"{title_prefix} · acurácia por lote e retreinamento", x=0.08, y=0.95, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.08,
        0.875,
        f"Matriz 24 x 24 da campanha na AWS · blocos D7 a D23 · mediana de ~{n_med:,} registros por lote".replace(",", "."),
        fontsize=9.0,
        color=MUTED,
    )
    return fig


def build_insects_strip_heatmap() -> plt.Figure:
    # Load 28-block accuracy (3-seed mean) from windows_full_28blocks.csv
    win_acc: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    win_bounds: dict[tuple[str, int], tuple[int, int]] = {}
    with open(V2_DIR / "s3-reports/windows_full_28blocks.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for ds in ("insects_abrupt_imbalanced", "insects_incremental_abrupt_reoccurring_imbalanced"):
        sub_f = sorted([r for r in rows if r["dataset"] == ds and r["policy"] == "frozen" and r["seed"] == "42"], key=lambda x: int(x["block"]))
        prev_end = 0
        for r in sub_f:
            b = int(r["block"])
            end_s = int(r["end_seq"])
            win_bounds[(ds, b)] = (prev_end, end_s)
            prev_end = end_s + 1

    for r in rows:
        if r["dataset"] in ("insects_abrupt_imbalanced", "insects_incremental_abrupt_reoccurring_imbalanced"):
            win_acc[(r["dataset"], r["policy"], int(r["block"]))].append(float(r["accuracy"]) * 100.0)

    # Load alarms_by_run.csv
    alarms: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    with open(V2_DIR / "s3-reports/alarms_by_run.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in ("insects_abrupt_imbalanced", "insects_incremental_abrupt_reoccurring_imbalanced"):
                alarms[(r["dataset"], r["policy"], r["seed"])].append(int(r["seq"]))

    # Load updates from f109_model_versions.csv (exclude initial v0)
    updates: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    with open(V2_DIR / "athena/f109_model_versions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in ("insects_abrupt_imbalanced", "insects_incremental_abrupt_reoccurring_imbalanced") and r["action"] != "initial":
                updates[(r["dataset"], r["policy"], r["run_id"])].append(int(r["end_seq"]))

    # Map run_id -> seed using f109_runs_uris.csv
    run_to_seed: dict[str, str] = {}
    with open(V2_DIR / "athena/f109_runs_uris.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            run_to_seed[r["run_id"]] = r["seed"]

    updates_by_seed: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for (ds, pol, rid), seqs in updates.items():
        s = run_to_seed.get(rid, "42")
        updates_by_seed[(ds, pol, s)].extend(seqs)

    # Custom diverging colormap (orange -> pale -> teal) matching Image 3
    cmap_acc = mcolors.LinearSegmentedColormap.from_list("acc_div", ["#f4a582", "#f7f7f7", "#92c5de", "#01665e"])
    cmap_evt = mcolors.LinearSegmentedColormap.from_list("evt_seq", ["#f1f5f9", "#94a3b8", "#334155", "#0f172a"])

    fig = plt.figure(figsize=(10.0, 6.8), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42, left=0.17, right=0.92, top=0.83, bottom=0.14)

    configs = [
        (
            "insects_abrupt_imbalanced",
            "Abrupt Imbalanced",
            355275,
            [("R1", 0, 83859), ("R2", 83859, 128651), ("R3", 128651, 182320), ("R4", 182320, 242883), ("R5", 242883, 268380), ("R6", 268380, 355275)],
            [("ADWIN", "drift_adwin"), ("DDM", "drift_ddm"), ("Incremental", "incremental")],
        ),
        (
            "insects_incremental_abrupt_reoccurring_imbalanced",
            "Incremental-Abrupt-Reoccurring Imbalanced",
            452044,
            [("R1", 0, 150683), ("R2", 150683, 301365), ("R3", 301365, 452044)],
            [("ADWIN", "drift_adwin"), ("DDM", "drift_ddm"), ("UPF", "upf")],
        ),
    ]

    for panel_idx, (ds, ds_title, max_seq, regimes, pol_list) in enumerate(configs):
        ax = fig.add_subplot(gs[panel_idx])
        max_k = max_seq / 1000.0
        bin_w = 5.0  # 5 mil exemplos per bin
        n_bins = int(math.ceil(max_k / bin_w))

        # Rows: top is Regimes, then for each policy: Acuracia, Alarmes, Atualizacoes
        y_rows = []
        row_labels = ["Regimes anotados"]
        row_totals = [""]

        # Plot regime bar at top
        total_rows = 1 + len(pol_list) * 3
        y_top = total_rows - 1
        for r_lbl, r_s, r_e in regimes:
            x0, x1 = r_s / 1000.0, r_e / 1000.0
            rect = mpatches.Rectangle((x0, y_top + 0.12), x1 - x0, 0.76, facecolor="#cbd5e1", edgecolor="white", linewidth=1.0)
            ax.add_patch(rect)
            ax.text((x0 + x1) / 2.0, y_top + 0.50, r_lbl, ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK)

        # Boundary lines
        for _, r_s, _ in regimes[1:]:
            ax.axvline(r_s / 1000.0, color="#334155", linewidth=0.95, zorder=8)

        cur_y = y_top - 1
        for p_lbl, p_key in pol_list:
            # 1) Accuracy strip across 28 blocks (1-indexed 1..28)
            for b in range(1, 29):
                s_seq, e_seq = win_bounds[(ds, b)]
                vals = win_acc[(ds, p_key, b)]
                acc_m = statistics.mean(vals) if vals else 50.0
                rect = mpatches.Rectangle((s_seq / 1000.0, cur_y + 0.08), max(0.8, (e_seq - s_seq) / 1000.0), 0.84, facecolor=cmap_acc(acc_m / 100.0), edgecolor="none")
                ax.add_patch(rect)
            row_labels.append(f"{p_lbl} · Acurácia")
            row_totals.append("")
            cur_y -= 1

            # 2) Alarms strip (5k bins, mean across 3 seeds)
            al_bins = np.zeros(n_bins)
            al_counts = []
            for seed in ("42", "43", "44"):
                seqs = alarms.get((ds, p_key, seed), [])
                al_counts.append(len(seqs))
                for s in seqs:
                    bi = min(n_bins - 1, int((s / 1000.0) // bin_w))
                    al_bins[bi] += 1.0 / 3.0
            for bi in range(n_bins):
                rect = mpatches.Rectangle((bi * bin_w, cur_y + 0.08), bin_w, 0.84, facecolor=cmap_evt(min(1.0, al_bins[bi] / 5.0)), edgecolor="none")
                ax.add_patch(rect)
            row_labels.append("Alarmes")
            row_totals.append(f"{statistics.mean(al_counts):.0f}" if sum(al_counts) > 0 else "0")
            cur_y -= 1

            # 3) Updates strip (5k bins, mean across 3 seeds)
            up_bins = np.zeros(n_bins)
            up_counts = []
            for seed in ("42", "43", "44"):
                seqs = updates_by_seed.get((ds, p_key, seed), [])
                up_counts.append(len(seqs))
                for s in seqs:
                    bi = min(n_bins - 1, int((s / 1000.0) // bin_w))
                    up_bins[bi] += 1.0 / 3.0
            for bi in range(n_bins):
                rect = mpatches.Rectangle((bi * bin_w, cur_y + 0.08), bin_w, 0.84, facecolor=cmap_evt(min(1.0, up_bins[bi] / 5.0)), edgecolor="none")
                ax.add_patch(rect)
            row_labels.append("Atualizações")
            row_totals.append(f"{statistics.mean(up_counts):.0f}")
            cur_y -= 1

        ax.set_xlim(0, max_k)
        ax.set_ylim(0, total_rows)
        yticks = [total_rows - 1 - i + 0.5 for i in range(total_rows)]
        ax.set_yticks(yticks, row_labels, fontsize=8.0, color=INK)
        for i_t, tot in enumerate(row_totals):
            if tot:
                ax.text(max_k + max_k * 0.015, yticks[i_t], tot, va="center", ha="left", fontsize=8.2, fontweight="bold", color=INK)
        ax.text(max_k + max_k * 0.015, total_rows + 0.15, "Total médio", ha="left", va="bottom", fontsize=7.8, color=MUTED)
        ax.set_title(ds_title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if panel_idx == 1:
            ax.set_xlabel("Posição no fluxo (mil exemplos)", fontsize=9.0, color=INK)

    # Colorbars at bottom
    cax1 = fig.add_axes([0.17, 0.06, 0.25, 0.018])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 100), cmap=cmap_acc), cax=cax1, orientation="horizontal")
    cb1.set_ticks([0, 25, 50, 75, 100])
    cb1.ax.tick_params(labelsize=7.5)
    cax1.set_title("Acurácia (%)", fontsize=8.0, color=INK, loc="left", pad=3)

    cax2 = fig.add_axes([0.55, 0.06, 0.25, 0.018])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 5), cmap=cmap_evt), cax=cax2, orientation="horizontal")
    cb2.set_ticks([0, 1, 2, 3, 4, 5])
    cb2.ax.tick_params(labelsize=7.5)
    cax2.set_title("Eventos por bloco de 5 mil exemplos", fontsize=8.0, color=INK, loc="left", pad=3)

    fig.suptitle("Acurácia, alarmes e atualizações por posição no fluxo", x=0.17, y=0.955, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.17, 0.905, "INSECTS / ADWIN, DDM, Incremental e UPF / Média das seeds 42, 43 e 44", fontsize=9.0, color=MUTED)
    fig.text(
        0.17,
        0.008,
        "Fonte: evidencias/aws-benchmarks-v2 (janelas por bloco, alarmes por execução e versões de modelos do Athena).",
        fontsize=8.2,
        color=MUTED,
    )
    return fig


def main() -> None:
    apply_style()
    OUT_REGIMES.mkdir(parents=True, exist_ok=True)
    OUT_DRIFT.mkdir(parents=True, exist_ok=True)

    block_sizes = load_block_sizes()
    decisions = json.load(open(V2_DIR / "s3-reports/retraining_decisions.json", encoding="utf-8"))

    specs = [
        ("airlines", "Airlines", "cara", "CARA", [7], "aws-acuracia-por-lote-modelos-airlines.svg"),
        ("electricity", "Electricity", "upf", "UPF", [7], "aws-acuracia-por-lote-modelos-electricity.svg"),
        ("insects_abrupt_imbalanced", "INSECTS Abrupto", "cara", "CARA", [7], "aws-acuracia-por-lote-modelos-insects-abrupto.svg"),
        ("insects_incremental_abrupt_reoccurring_imbalanced", "INSECTS Recorrente", "upf", "UPF", [7, 16], "aws-acuracia-por-lote-modelos-insects-recorrente.svg"),
    ]
    for ds, title, pol, lbl, rbs, fname in specs:
        fig = build_batch_accuracy_plot(ds, title, pol, lbl, rbs, block_sizes, decisions)
        out = OUT_REGIMES / fname
        fig.savefig(out, format="svg")
        plt.close(fig)
        print("Saved", out)

    fig_strip = build_insects_strip_heatmap()
    out_strip = OUT_DRIFT / "aws-acuracia-alarmes-atualizacoes-insects.svg"
    fig_strip.savefig(out_strip, format="svg")
    plt.close(fig_strip)
    print("Saved", out_strip)


if __name__ == "__main__":
    main()
