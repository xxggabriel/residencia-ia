"""Generate 4 pure-AWS regime/matrix SVGs from evidencias/aws-benchmarks-v2 and scrub F109/prequencial from all SVGs."""

from __future__ import annotations

import csv
import glob
import json
import re
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
from matplotlib.colors import LinearSegmentedColormap

# Exact changepoints from src/continual_bench/datasets/real_streams.py
CHANGEPOINTS = {
    "insects_abrupt_imbalanced": [83859, 128651, 182320, 242883, 268380],
    "insects_incremental_abrupt_reoccurring_imbalanced": [150683, 301365],
}


def build_aws_regime_heatmap(ds: str, title: str) -> plt.Figure:
    """Heatmap of frozen F1 (%) baseline and p.p. gain/loss by annotated regime from windows_full_28blocks.csv."""
    rows = [
        r
        for r in csv.DictReader(open(EV2 / "s3-reports/windows_full_28blocks.csv", encoding="utf-8"))
        if r["dataset"] == ds
    ]
    cps = CHANGEPOINTS[ds]

    # Assign each block to a regime R1..R{len(cps)+1} based on block midpoint/end_seq
    def regime_of(end_seq: int) -> int:
        for idx, cp in enumerate(cps, 1):
            if end_seq < cp:
                return idx
        return len(cps) + 1

    # Exclude warmup R1 if only 1 block or keep all regimes with data >= R2
    regimes = sorted({regime_of(int(r["end_seq"])) for r in rows if regime_of(int(r["end_seq"])) >= 2})
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]

    # Compute mean macro_f1 (%) per (policy, regime)
    agg: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        reg = regime_of(int(r["end_seq"]))
        if reg in regimes:
            agg.setdefault((r["policy"], reg), []).append(float(r["macro_f1"]) * 100.0)
    means = {k: float(np.mean(v)) for k, v in agg.items()}

    mat = np.zeros((len(policies), len(regimes)))
    ann = [["" for _ in regimes] for _ in policies]
    for i, pol in enumerate(policies):
        for j, reg in enumerate(regimes):
            val = means[(pol, reg)]
            if pol == "frozen":
                mat[i, j] = 0.0
                ann[i][j] = f"{val:.1f}%"
            else:
                diff = val - means[("frozen", reg)]
                mat[i, j] = diff
                ann[i][j] = f"{diff:+.1f}"

    cmap = LinearSegmentedColormap.from_list("div", ["#C2592B", "#F8FAFC", TEAL])
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=300)
    fig.subplots_adjust(left=0.20, right=0.86, top=0.80, bottom=0.18)

    im = ax.imshow(mat, cmap=cmap, vmin=-45, vmax=45, aspect="auto")
    for i in range(len(policies)):
        for j in range(len(regimes)):
            txt_col = "white" if abs(mat[i, j]) > 26 else INK
            ax.text(
                j,
                i,
                ann[i][j].replace(".", ","),
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold" if policies[i] == "frozen" else "normal",
                color=txt_col,
            )

    ax.set_xticks(range(len(regimes)), [f"R{r}" for r in regimes], fontsize=10)
    ax.set_yticks(range(len(policies)), [SHORT.get(p, p) for p in policies], fontsize=9.8)
    ax.set_xlabel("Regime anotado do stream", fontsize=10, color=INK)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Diferença vs. Congelado (p.p.)", fontsize=9.5, color=INK)

    fig.text(
        0.20,
        0.02,
        "Fonte: evidencias/aws-benchmarks-v2/s3-reports/windows_full_28blocks.csv (XGBoost).\n"
        "A linha Congelado mostra o F1 macro absoluto (%) no regime; as demais mostram a diferença em pontos percentuais (p.p.).",
        fontsize=8.8,
        color=MUTED,
        va="bottom",
    )
    return fig


def build_aws_matrix_and_trace(ds: str, pol: str, title: str, color: str) -> plt.Figure:
    """Plot 24x24 candidate accuracy matrix from S3 + deployed accuracy trace and retrain markers."""
    m = json.loads(
        (EV2 / f"s3-reports/retraining_matrices/{ds}__{pol}__accuracy_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    mat = np.array([[np.nan if v is None else float(v) * 100.0 for v in row] for row in m["matrix"]])

    # Get model_versions retrains for (ds, pol, seed=42)
    seed42_runs = {
        r["run_id"]
        for r in csv.DictReader(open(EV2 / "athena/f109_runs_uris.csv", encoding="utf-8"))
        if r["seed"] == "42"
    }
    mvs = [
        r
        for r in csv.DictReader(open(EV2 / "athena/f109_model_versions.csv", encoding="utf-8"))
        if r["run_id"] in seed42_runs and r["dataset"] == ds and r["policy"] == pol
    ]
    retrain_seqs = {int(r["end_seq"]) for r in mvs if r["action"] == "reset_retrain"}

    # Get 27 evaluation blocks (B1..B27) accuracy from windows_full_28blocks.csv
    wins = sorted(
        [
            r
            for r in csv.DictReader(open(EV2 / "s3-reports/windows_full_28blocks.csv", encoding="utf-8"))
            if r["dataset"] == ds and r["policy"] == pol
        ],
        key=lambda x: int(x["block"]),
    )
    xs = [int(w["block"]) for w in wins]
    ys = [float(w["accuracy"]) * 100.0 for w in wins]
    ret_pts = [
        (int(w["block"]), float(w["accuracy"]) * 100.0) for w in wins if int(w["end_seq"]) in retrain_seqs
    ]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 6.8), dpi=300, gridspec_kw={"height_ratios": [1.35, 1.0]}
    )
    fig.subplots_adjust(left=0.11, right=0.90, top=0.84, bottom=0.14, hspace=0.38)

    im = ax1.imshow(mat, cmap="Blues", vmin=40, vmax=95, aspect="auto")
    ax1.set_title(
        "Matriz de acurácia: modelo candidato (linha) × lote avaliado (coluna)",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=6,
    )
    ax1.set_ylabel("Modelo candidato (M0..M23)", fontsize=9.5, color=INK)
    ax1.set_xlabel("Lote avaliado na matriz (D0..D23)", fontsize=9.5, color=INK)
    ax1.set_xticks([0, 7, 15, 23], ["D0", "D7", "D15", "D23"], fontsize=8.8)
    ax1.set_yticks([0, 7, 15, 23], ["M0", "M7", "M15", "M23"], fontsize=8.8)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("Acurácia (%)", fontsize=9, color=INK)

    ax2.plot(
        xs, ys, color=color, lw=2.0, marker="o", ms=3.8, label=f"{SHORT.get(pol, pol)} (acurácia por bloco)"
    )
    if ret_pts:
        ax2.scatter(
            [p[0] for p in ret_pts],
            [p[1] for p in ret_pts],
            facecolors="white",
            edgecolors=color,
            marker="s",
            s=65,
            linewidths=2.0,
            zorder=5,
            label="Retreinamento (reset_retrain)",
        )
    ax2.set_title(
        "Acurácia em uso ao longo dos blocos e pontos de retreinamento",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=6,
    )
    ax2.set_xlabel("Bloco do stream", fontsize=9.5, color=INK)
    ax2.set_ylabel("Acurácia (%)", fontsize=9.5, color=INK)
    ax2.legend(frameon=False, fontsize=9.2, loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.35, lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(title, x=0.11, y=0.95, ha="left", fontsize=13.2, fontweight="bold", color=INK)
    fig.text(
        0.11,
        0.02,
        "Fonte: evidencias/aws-benchmarks-v2/s3-reports/retraining_matrices/ e windows_full_28blocks.csv.\n"
        "Quadrados vazados na curva inferior marcam os blocos onde o modelo foi retreinado do zero.",
        fontsize=8.8,
        color=MUTED,
        va="bottom",
    )
    return fig


def scrub_all_svgs():
    """Remove any 'F109'/'f109' and 'prequencial' text from all SVGs in docs_v2/relatorios/figuras/."""
    for p in sorted(glob.glob(str(OUT_FIG_ROOT / "*/*.svg"))):
        txt = Path(p).read_text(encoding="utf-8")
        orig = txt
        txt = re.sub(r"\s*[·(]?\s*AWS\s+F109\s*[)]?", " (AWS)", txt, flags=re.I)
        txt = re.sub(r"\s*[·(]?\s*F109\s*[)]?", "", txt, flags=re.I)
        txt = re.sub(r"fluxo\s+prequencial\s+completo", "fluxo contínuo completo", txt, flags=re.I)
        txt = re.sub(r"escopo\s+prequencial", "escopo contínuo", txt, flags=re.I)
        txt = re.sub(r"prequencial", "contínuo", txt, flags=re.I)
        if txt != orig:
            Path(p).write_text(txt, encoding="utf-8")


def main():
    apply_style()
    dest = OUT_FIG_ROOT / "regimes-e-matrizes"
    dest.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "aws-regimes-insects-abrupto.svg",
            lambda: build_aws_regime_heatmap(
                "insects_abrupt_imbalanced", "Ganho por regime anotado · INSECTS Abrupto"
            ),
        ),
        (
            "aws-regimes-insects-recorrente.svg",
            lambda: build_aws_regime_heatmap(
                "insects_incremental_abrupt_reoccurring_imbalanced",
                "Ganho por regime anotado · INSECTS Incr. Abr. Recorrente",
            ),
        ),
        (
            "aws-matriz-trajetoria-electricity.svg",
            lambda: build_aws_matrix_and_trace(
                "electricity",
                "upf",
                "Matriz de decisão (24×24) e trajetória em uso · Electricity (UPF)",
                ORANGE,
            ),
        ),
        (
            "aws-matriz-trajetoria-airlines.svg",
            lambda: build_aws_matrix_and_trace(
                "airlines", "cara", "Matriz de decisão (24×24) e trajetória em uso · Airlines (CARA)", BLUE
            ),
        ),
    ]
    for fname, fn in plots:
        fig = fn()
        fig.savefig(dest / fname, format="svg")
        plt.close(fig)
        print("Generated:", dest / fname)

    scrub_all_svgs()
    print("Scrubbed F109 and prequencial from all SVGs in", OUT_FIG_ROOT)


if __name__ == "__main__":
    main()
