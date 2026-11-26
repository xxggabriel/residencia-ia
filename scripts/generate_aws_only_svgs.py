"""Generate pure-AWS SVG figures from Athena (model_versions, cost_estimates, cur2) and S3 report.json files.
Zero data from evidencias/semana-10 is used.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/Users/gabrieloliveira/residencia-ia")
ATHENA_DIR = Path("/Users/gabrieloliveira/.gemini/antigravity/brain/eaba6c5a-00df-4e63-9ed2-042f6b5d344f/scratch/athena_data")
REPORTS_DIR = ATHENA_DIR / "reports_f109"
OUT_FIG_ROOT = ROOT / "docs_v2/relatorios/figuras"
TMP_OUT = Path("/tmp/aws-only-svg-build")

sys.path.insert(0, str(ROOT / "notebooks"))
import matplotlib.pyplot as plt
import numpy as np
from benchmark_athena_data import load_athena
from benchmark_athena_figures import SHORT, build_figures
from benchmark_plot_style import BLUE, INK, MUTED, ORANGE, TEAL, apply_style, export_bundle
from matplotlib.lines import Line2D

PURPLE = "#7C3AED"
RED = "#DC2626"
SLATE = "#64748b"

DATASET_LABELS = {
    "electricity": "electricity",
    "airlines": "airlines",
    "insects_abrupt_imbalanced": "INSECTS Abrupto",
    "insects_incremental_abrupt_reoccurring_imbalanced": "INSECTS Incr. Abr. Recorr.",
}

AWS_CHANGEPOINTS = {
    "insects_abrupt_imbalanced": [83859, 128651, 182320, 242883, 268380],
    "insects_incremental_abrupt_reoccurring_imbalanced": [150683, 301365],
}

POLICY_COLORS = {
    "frozen": "#64748b",
    "periodic": ORANGE,
    "drift_adwin": "#0284c7",
    "drift_ddm": "#0f766e",
    "incremental": TEAL,
    "cara": BLUE,
    "upf": PURPLE,
    "oracle": "#111827",
}

POLICY_MARKERS = {
    "frozen": "o",
    "periodic": "s",
    "drift_adwin": "^",
    "drift_ddm": "v",
    "incremental": "P",
    "cara": "D",
    "upf": "X",
    "oracle": "*",
}


def save_svg(fig: plt.Figure, rel_path: str):
    out_path = OUT_FIG_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def load_report(ds: str, pol: str, seed: int = 42) -> dict:
    return json.loads((REPORTS_DIR / f"{ds}__{pol}__{seed}.json").read_text(encoding="utf-8"))


# 1. Qualidade: cohort vs holdout + calibracao (brier & log_loss)
def build_aws_cohort_vs_holdout() -> plt.Figure:
    q_rows = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/quality-summary.csv", encoding="utf-8"))
        if r["metric"] == "macro_f1"
    }
    h_rows = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/holdout-summary.csv", encoding="utf-8"))
        if r["metric"] == "macro_f1"
    }
    policies = ["frozen", "periodic", "drift_adwin", "drift_ddm", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.30, hspace=0.42)
    y_pos = np.arange(len(policies))
    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        c_vals = [q_rows[(ds, p)] for p in policies]
        h_vals = [h_rows[(ds, p)] for p in policies]
        ax.hlines(y_pos, c_vals, h_vals, color="#cbd5e1", lw=2.0, zorder=2)
        ax.scatter(c_vals, y_pos, color=BLUE, s=42, label="Coorte comum pós-calibração", zorder=3)
        ax.scatter(h_vals, y_pos, color=ORANGE, marker="s", s=38, label="Holdout final (10%)", zorder=4)
        ax.set_yticks(y_pos, [SHORT.get(p, p) for p in policies], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0.35, 0.90)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("F1 macro", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("F1 macro: coorte comum pós-calibração vs. holdout final (10%)", x=0.14, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.14, 0.02,
        "Fonte: AWS F109 (quality-summary.csv e holdout-summary.csv). Coorte comum avalia os 60% centrais;\n"
        "holdout avalia os 10% finais do stream sem novas atualizações de modelo.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_calibracao() -> plt.Figure:
    brier = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/quality-summary.csv", encoding="utf-8"))
        if r["metric"] == "brier"
    }
    logloss = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/quality-summary.csv", encoding="utf-8"))
        if r["metric"] == "log_loss"
    }
    policies = ["frozen", "periodic", "drift_adwin", "drift_ddm", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.30, hspace=0.42)
    y = np.arange(len(policies))
    h = 0.34
    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        ax.barh(y - h / 2, [brier[(ds, p)] for p in policies], height=h, color=TEAL, label="Brier score (↓)")
        ax.barh(y + h / 2, [logloss[(ds, p)] for p in policies], height=h, color=PURPLE, alpha=0.85, label="Log-loss (↓)")
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Erro probabilístico (menor é melhor)", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Calibração probabilística na coorte comum (Brier e Log-loss · F109)", x=0.14, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.14, 0.02,
        "Fonte: AWS F109 (quality-summary.csv). Brier e Log-loss medem a qualidade das probabilidades preditas;\n"
        "valores menores indicam probabilidades melhor calibradas. CARA-CT, CARA-P e Oracle coincidem com CARA.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


# 2. Evolução temporal: todos os 28 blocos do S3 report.json (novos dados vs congelado)
def build_aws_evolucao_vs_congelado() -> plt.Figure:
    focus = [
        ("frozen", "Congelado (sem atualização)", SLATE, "--", 2.0),
        ("incremental", "Incremental (23 atualizações)", TEAL, "-", 2.0),
        ("drift_adwin", "ADWIN (retreino por alarme)", "#0284c7", "-", 1.7),
        ("periodic", "Periódica (5 retreinos)", ORANGE, "-", 1.6),
        ("cara", "CARA (1 retreino por custo)", BLUE, "-.", 1.7),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.15, wspace=0.22, hspace=0.44)

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        all_y = []
        for pol, lbl, col, ls, lw in focus:
            rep = load_report(ds, pol, 42)
            wins = sorted(rep["windows"], key=lambda w: w["end_seq"])
            xs = [w["end_seq"] / 1000.0 for w in wins]
            ys = [w["macro_f1"] for w in wins]
            all_y.extend(ys)
            ax.plot(xs, ys, color=col, ls=ls, lw=lw, label=lbl, zorder=4 if pol in ("frozen", "incremental") else 3)

        if ds in AWS_CHANGEPOINTS:
            for idx_cp, cp in enumerate(AWS_CHANGEPOINTS[ds]):
                ax.axvline(cp / 1000.0, color=RED, ls=":", lw=1.4, zorder=2,
                           label="Drift real (fronteira anotada)" if (ds == "insects_abrupt_imbalanced" and idx_cp == 0) else None)

        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Posição no stream (mil exemplos)", fontsize=9.5, color=INK)
        ax.set_ylabel("F1 macro no bloco", fontsize=9.5, color=INK)
        ax.set_ylim(max(0.20, min(all_y) - 0.05), min(0.95, max(all_y) + 0.05))
        ax.grid(True, linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes.flat[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.91), ncol=3, frameon=False, fontsize=9.2, columnspacing=1.0)
    fig.suptitle("Evolução completa por bloco de novos dados vs. modelo congelado (F109)", x=0.09, y=0.965, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.09, 0.02,
        "Fonte: 28 blocos completos extraídos dos report.json no S3 (seed 42; F109 determinístico entre seeds em qualidade).\n"
        "Linhas verticais vermelhas marcam fronteiras reais nos streams INSECTS (Electricity e Airlines não têm fronteiras anotadas).",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_ganho_sobre_congelado() -> plt.Figure:
    focus = [
        ("incremental", "Incremental", TEAL, "-"),
        ("drift_adwin", "ADWIN", "#0284c7", "-"),
        ("drift_ddm", "DDM", "#0f766e", "--"),
        ("periodic", "Periódica", ORANGE, "-"),
        ("cara", "CARA", BLUE, "-."),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.0), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.14, wspace=0.22, hspace=0.44)

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        base = {w["end_seq"]: w["macro_f1"] for w in load_report(ds, "frozen", 42)["windows"]}
        for pol, lbl, col, ls in focus:
            wins = sorted(load_report(ds, pol, 42)["windows"], key=lambda w: w["end_seq"])
            xs = [w["end_seq"] / 1000.0 for w in wins]
            diffs = [(w["macro_f1"] - base[w["end_seq"]]) * 100.0 for w in wins]
            ax.plot(xs, diffs, color=col, ls=ls, lw=1.7, label=lbl)
        ax.axhline(0.0, color=SLATE, ls=":", lw=1.1)
        if ds in AWS_CHANGEPOINTS:
            for cp in AWS_CHANGEPOINTS[ds]:
                ax.axvline(cp / 1000.0, color=RED, ls=":", lw=1.1, alpha=0.7)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Posição no stream (mil exemplos)", fontsize=9.5, color=INK)
        ax.set_ylabel("Ganho vs. Congelado (p.p.)", fontsize=9.5, color=INK)
        ax.grid(True, linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.90), ncol=5, frameon=False, fontsize=9.5)
    fig.suptitle("Ganho de F1 macro em pontos percentuais sobre o modelo congelado (F109)", x=0.09, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.09, 0.02,
        "Fonte: blocos completos dos report.json no S3 (AWS F109). Valores positivos indicam ganho ao atualizar;\n"
        "valores negativos indicam degradação em relação ao modelo mantido congelado após o aquecimento.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


# 3. Drift e Retreinos: timelines exatas com alarmes + reset_retrain + changepoints do S3/Athena!
def build_drift_timeline_insects(ds: str, title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.14, hspace=0.36)

    rep_frz = load_report(ds, "frozen", 42)
    wins_frz = sorted(rep_frz["windows"], key=lambda w: w["end_seq"])
    xs_f = [w["end_seq"] / 1000.0 for w in wins_frz]
    ys_f = [w["macro_f1"] for w in wins_frz]

    for ax, (pol, pname, col) in zip(axes, [("drift_adwin", "ADWIN", "#0284c7"), ("drift_ddm", "DDM", TEAL)], strict=True):
        rep = load_report(ds, pol, 42)
        wins = sorted(rep["windows"], key=lambda w: w["end_seq"])
        xs = [w["end_seq"] / 1000.0 for w in wins]
        ys = [w["macro_f1"] for w in wins]
        ax.plot(xs_f, ys_f, color=SLATE, ls="--", lw=1.4, label="Congelado (referência)", zorder=2)
        ax.plot(xs, ys, color=col, lw=2.0, label=f"{pname} (F1 macro por bloco)", zorder=3)

        for i, cp in enumerate(AWS_CHANGEPOINTS[ds]):
            ax.axvline(cp / 1000.0, color=RED, ls=":", lw=1.5, zorder=1, label="Drift real (fronteira anotada)" if i == 0 else None)

        alarms = [a["seq"] / 1000.0 for a in rep.get("alarms", [])]
        if alarms:
            ax.scatter(alarms, [min(ys_f + ys) - 0.03] * len(alarms), marker="^", s=45, color=ORANGE, zorder=4, label="Alarme detectado")

        retrains = [mv["end_seq"] / 1000.0 for mv in rep.get("model_versions", []) if mv["action"] == "reset_retrain"]
        initial = [mv["end_seq"] / 1000.0 for mv in rep.get("model_versions", []) if mv["action"] == "initial"]
        if initial:
            ax.scatter(initial, [max(ys_f + ys) + 0.02] * len(initial), marker="o", s=50, color=PURPLE, zorder=5, label="Treino inicial (warmup)")
        if retrains:
            ax.scatter(retrains, [max(ys_f + ys) + 0.02] * len(retrains), marker="s", s=40, color=BLUE, zorder=5, label="Retreino do zero (reset_retrain)")

        ax.set_title(f"{title} · Detector {pname} ({len(alarms)} alarmes, {len(retrains)} retreinos)", fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_ylabel("F1 macro", fontsize=9.5, color=INK)
        ax.set_ylim(max(0.15, min(ys_f + ys) - 0.08), min(1.0, max(ys_f + ys) + 0.08))
        ax.grid(True, linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].set_xlabel("Posição no stream (mil exemplos)", fontsize=9.5, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.91), ncol=3, frameon=False, fontsize=9.0)
    fig.suptitle(f"Drift real, alarmes detectados e retreinos do zero · {title} (F109)", x=0.09, y=0.965, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.09, 0.02,
        "Fonte: S3 report.json (alarms, model_versions e windows) e Athena continual_bench.model_versions (AWS F109).\n"
        "Triângulos laranja mostram cada alarme emitido; quadrados azuis mostram cada reset_retrain executado no fim do bloco.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_drift_timeline_elec_air() -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.14, hspace=0.45)

    for ax, ds in zip(axes, ["electricity", "airlines"], strict=True):
        rep_frz = load_report(ds, "frozen", 42)
        wins_frz = sorted(rep_frz["windows"], key=lambda w: w["end_seq"])
        xs_f = [w["end_seq"] / 1000.0 for w in wins_frz]
        ys_f = [w["macro_f1"] for w in wins_frz]
        ax.plot(xs_f, ys_f, color=SLATE, ls="--", lw=1.4, label="Congelado", zorder=2)

        rep_adw = load_report(ds, "drift_adwin", 42)
        wins_a = sorted(rep_adw["windows"], key=lambda w: w["end_seq"])
        ax.plot([w["end_seq"] / 1000.0 for w in wins_a], [w["macro_f1"] for w in wins_a], color="#0284c7", lw=1.8, label="ADWIN", zorder=3)

        rep_ddm = load_report(ds, "drift_ddm", 42)
        wins_d = sorted(rep_ddm["windows"], key=lambda w: w["end_seq"])
        ax.plot([w["end_seq"] / 1000.0 for w in wins_d], [w["macro_f1"] for w in wins_d], color=TEAL, lw=1.6, label="DDM", zorder=3)

        alarms_a = [a["seq"] / 1000.0 for a in rep_adw.get("alarms", [])]
        retrains_a = [mv["end_seq"] / 1000.0 for mv in rep_adw.get("model_versions", []) if mv["action"] == "reset_retrain"]
        y_lo = min(ys_f + [w["macro_f1"] for w in wins_a] + [w["macro_f1"] for w in wins_d]) - 0.03
        y_hi = max(ys_f + [w["macro_f1"] for w in wins_a] + [w["macro_f1"] for w in wins_d]) + 0.02
        if alarms_a:
            ax.scatter(alarms_a, [y_lo] * len(alarms_a), marker="^", s=38, color=ORANGE, zorder=4, label="Alarme ADWIN")
        if retrains_a:
            ax.scatter(retrains_a, [y_hi] * len(retrains_a), marker="s", s=34, color=BLUE, zorder=5, label="Retreino ADWIN (reset)")

        ax.set_title(f"{DATASET_LABELS[ds]} · ADWIN ({len(alarms_a)} alarmes, {len(retrains_a)} retreinos) vs. DDM", fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Posição no stream (mil exemplos)", fontsize=9.5, color=INK)
        ax.set_ylabel("F1 macro", fontsize=9.5, color=INK)
        ax.set_ylim(max(0.25, y_lo - 0.04), min(0.95, y_hi + 0.05))
        ax.grid(True, linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.91), ncol=5, frameon=False, fontsize=9.0)
    fig.suptitle("Alarmes detectados e retreinos do zero em Electricity e Airlines (F109)", x=0.09, y=0.965, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.09, 0.02,
        "Fonte: S3 report.json (alarms, model_versions e windows · AWS F109). Electricity e Airlines não possuem fronteiras reais anotadas;\n"
        "os marcadores mostram todos os alarmes do ADWIN e os retreinos do zero disparados ao fim de cada bloco.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_versoes_timeline() -> plt.Figure:
    """Plot exact model_versions (initial, reset_retrain, continue) from Athena continual_bench.model_versions."""
    mvs = list(csv.DictReader(open(ATHENA_DIR / "f109_model_versions.csv", encoding="utf-8")))
    # Filter seed 42 runs by matching run_id from f109_runs_uris.csv
    seed42_runs = {
        r["run_id"]
        for r in csv.DictReader(open(ATHENA_DIR / "f109_runs_uris.csv", encoding="utf-8"))
        if r["seed"] == "42"
    }
    mvs42 = [r for r in mvs if r["run_id"] in seed42_runs]

    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]
    action_styles = {
        "initial": (PURPLE, "o", "Treino inicial"),
        "reset_retrain": (BLUE, "s", "Retreino do zero (reset_retrain)"),
        "continue": (TEAL, "^", "Atualização incremental (continue)"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.28, hspace=0.44)

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        for y_idx, pol in enumerate(policies):
            sub = [r for r in mvs42 if r["dataset"] == ds and r["policy"] == pol]
            for act, (col, mk, _) in action_styles.items():
                xs = [int(r["end_seq"]) / 1000.0 for r in sub if r["action"] == act]
                if xs:
                    ax.scatter(xs, [y_idx] * len(xs), color=col, marker=mk, s=34, zorder=4)
        if ds in AWS_CHANGEPOINTS:
            for cp in AWS_CHANGEPOINTS[ds]:
                ax.axvline(cp / 1000.0, color=RED, ls=":", lw=1.2, zorder=1)
        ax.set_yticks(range(len(policies)), [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Posição no stream (mil exemplos)", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [Line2D([], [], color=c, marker=m, linestyle="None", markersize=6.5, label=lbl) for c, m, lbl in action_styles.values()]
    handles.append(Line2D([], [], color=RED, ls=":", lw=1.4, label="Drift real (INSECTS)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Linha do tempo das versões de modelo (inicial, retreino do zero e incremental)", x=0.14, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.14, 0.02,
        "Fonte: Athena continual_bench.model_versions (885 versões promovidas na F109; exibida seed 42).\n"
        "CARA-CT, CARA-P e Oracle seguem o mesmo ponto de retreino do CARA; upf executa 2 retreinos em INSECTS Recorrente.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_drift_alarmes_e_retreinos() -> plt.Figure:
    m_rows = {
        (r["dataset"], r["policy"], r["metric"]): float(r["mean"])
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/resources-summary.csv", encoding="utf-8"))
        if r["metric"] in ("drift_alarm_count", "retrain_count", "update_count")
    }
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.15, wspace=0.32, hspace=0.44)
    y = np.arange(len(policies))
    h = 0.26
    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        ax.barh(y - h, [m_rows[(ds, p, "drift_alarm_count")] for p in policies], height=h, color=RED, alpha=0.85, label="Alarmes de drift")
        ax.barh(y, [m_rows[(ds, p, "retrain_count")] for p in policies], height=h, color=BLUE, label="Retreinos do zero")
        ax.barh(y + h, [m_rows[(ds, p, "update_count")] for p in policies], height=h, color=TEAL, label="Atualizações totais (incl. incremental)")
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Contagem média por execução", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Alarmes de drift, retreinos do zero e atualizações por política (F109)", x=0.14, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.14, 0.02,
        "Fonte: AWS F109 (resources-summary.csv). Apenas ADWIN e DDM emitem alarmes; incremental soma 23 atualizações sem retreino do zero.\n"
        "CARA-CT, CARA-P e Oracle coincidem com CARA (1 retreino). A política drift permaneceu com 0 alarmes.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


# 4. Esquecimento / Retenção na F109 (extraído dos 132 report.json no S3!)
def build_aws_retencao_absoluta() -> plt.Figure:
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.30, hspace=0.42)
    y = np.arange(len(policies))

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        means = []
        for pol in policies:
            vals = [load_report(ds, pol, s)["tracking_observations"]["retention"]["absolute_retention"] for s in (42, 43, 44)]
            means.append(float(np.mean(vals)))
        bars = ax.barh(y, means, color=[POLICY_COLORS[p] for p in policies], height=0.62)
        for b, v in zip(bars, means, strict=True):
            ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}".replace(".", ","), va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.02)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Retenção absoluta (↑)", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Retenção absoluta nos probes de regimes anteriores (AWS F109)", x=0.14, y=0.95, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.14, 0.02,
        "Fonte: S3 report.json (tracking_observations.retention.absolute_retention · AWS F109, média de 3 seeds).\n"
        "gem_bwt e post_learning_fm estão marcados como indisponíveis no F109 (no_historical_post_learning_baselines).",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_plasticidade_vs_estabilidade() -> plt.Figure:
    q_rows = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/quality-summary.csv", encoding="utf-8"))
        if r["metric"] == "macro_f1"
    }
    policies = ["frozen", "periodic", "drift_adwin", "drift_ddm", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.14, wspace=0.22, hspace=0.44)

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        xs, ys = [], []
        for pol in policies:
            ret = float(np.mean([load_report(ds, pol, s)["tracking_observations"]["retention"]["absolute_retention"] for s in (42, 43, 44)]))
            f1 = q_rows[(ds, pol)]
            xs.append(ret)
            ys.append(f1)
            ax.scatter(
                ret, f1, s=95 if pol == "oracle" else 55,
                color=POLICY_COLORS[pol], marker=POLICY_MARKERS[pol],
                zorder=4, edgecolors="white", linewidths=0.7,
            )
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Retenção absoluta (estabilidade, ↑)", fontsize=9.5, color=INK)
        ax.set_ylabel("F1 macro (plasticidade, ↑)", fontsize=9.5, color=INK)
        ax.set_xlim(max(0.20, min(xs) - 0.08), min(1.0, max(xs) + 0.08))
        ax.set_ylim(max(0.35, min(ys) - 0.06), min(0.90, max(ys) + 0.06))
        ax.grid(True, linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        Line2D([], [], color=POLICY_COLORS[p], marker=POLICY_MARKERS[p], linestyle="None",
               markersize=8 if p == "oracle" else 6.5, label=SHORT.get(p, p))
        for p in policies
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.90), ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle("Trade-off plasticidade (F1 macro) vs. estabilidade (retenção) · F109", x=0.09, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.09, 0.02,
        "Fonte: quality-summary.csv e S3 report.json (tracking_observations.retention.absolute_retention · AWS F109).\n"
        "Em Electricity, Incremental lidera F1 macro (0,729), mas apresenta a menor retenção (0,539).",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


# 5. Recursos e Custos (incluindo Batch/Fargate reais do Athena e CUR 2.0 real!)
def build_aws_quality_vs_cost() -> plt.Figure:
    q_rows = {
        (r["dataset"], r["policy"]): float(r["mean"])
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/quality-summary.csv", encoding="utf-8"))
        if r["metric"] == "macro_f1"
    }
    c_rows = {
        (r["dataset"], r["policy"]): float(r["mean"]) * 1000.0
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/resources-summary.csv", encoding="utf-8"))
        if r["metric"] == "aws_evaluation_cost_total_usd"
    }
    policies = ["frozen", "periodic", "drift_adwin", "drift_ddm", "incremental", "cara", "upf", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.14, wspace=0.22, hspace=0.44)
    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        xs, ys = [], []
        for pol in policies:
            x, y = c_rows[(ds, pol)], q_rows[(ds, pol)]
            xs.append(x)
            ys.append(y)
            ax.scatter(
                x, y, s=95 if pol == "oracle" else 55,
                color=POLICY_COLORS[pol], marker=POLICY_MARKERS[pol],
                zorder=4, edgecolors="white", linewidths=0.7,
            )
        ax.axvline(1.645667, color=MUTED, ls=":", lw=1.0, zorder=1)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Custo de referência da avaliação (mUS$)", fontsize=9.5, color=INK)
        ax.set_ylabel("F1 macro (coorte comum)", fontsize=9.5, color=INK)
        ax.set_ylim(max(0.35, min(ys) - 0.06), min(0.90, max(ys) + 0.06))
        ax.set_xlim(left=0.5, right=max(xs) * 1.15)
        ax.grid(True, linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [
        Line2D([], [], color=POLICY_COLORS[p], marker=POLICY_MARKERS[p], linestyle="None",
               markersize=8 if p == "oracle" else 6.5, label=SHORT.get(p, p))
        for p in policies
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.90), ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle("Qualidade preditiva vs. custo de referência na avaliação AWS", x=0.09, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.09, 0.02,
        "Linha pontilhada: piso de 60 s On-Demand (1,65 mUS$). Estimativa de referência (não é fatura AWS nem preço Spot).\n"
        "CARA-CT, CARA-P, UPF e Oracle reutilizam a matriz construída fisicamente por CARA; frozen e drift coincidem.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_decomposicao_custos() -> plt.Figure:
    c_log = {
        (r["dataset"], r["policy"]): float(r["mean"]) * 1000.0
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/resources-summary.csv", encoding="utf-8"))
        if r["metric"] == "aws_cost_total_usd"
    }
    c_ev = {
        (r["dataset"], r["policy"]): float(r["mean"]) * 1000.0
        for r in csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/resources-summary.csv", encoding="utf-8"))
        if r["metric"] == "aws_evaluation_cost_total_usd"
    }
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "cara", "upf", "oracle", "incremental"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.30, hspace=0.44)
    y = np.arange(len(policies))
    h = 0.34
    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        ax.barh(y - h / 2, [c_log[(ds, p)] for p in policies], height=h, color=BLUE, label="Custo aprendiz promovido (aws_cost_total)")
        ax.barh(y + h / 2, [c_ev[(ds, p)] for p in policies], height=h, color=ORANGE, label="Custo avaliação total (aws_evaluation_cost_total)")
        ax.axvline(1.645667, color=MUTED, ls=":", lw=1.0)
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Custo de referência (mUS$)", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=2, frameon=False, fontsize=9.3)
    fig.suptitle("Escopos de custo na AWS: aprendiz promovido vs. avaliação total (F109)", x=0.14, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.14, 0.02,
        "Linha pontilhada: piso de 60 s On-Demand (1,65 mUS$). CARA inclui construção física da matriz em aws_evaluation_cost_total;\n"
        "UPF e Oracle reutilizam a matriz em cache. Estimativas On-Demand em tasks Spot não representam fatura AWS.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_batch_duracao_e_fila() -> plt.Figure:
    """Plot real Batch/Fargate execution seconds vs queue seconds from Athena continual_bench.cost_estimates."""
    rows = list(csv.DictReader(open(ATHENA_DIR / "f109_cost_estimates.csv", encoding="utf-8")))
    policies = ["frozen", "periodic", "drift_ddm", "drift_adwin", "cara", "upf", "oracle", "incremental"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14, wspace=0.30, hspace=0.44)
    y = np.arange(len(policies))

    for ax, ds in zip(axes.flat, DATASET_LABELS.keys(), strict=True):
        exec_m, queue_m = [], []
        for pol in policies:
            sub = [r for r in rows if r["dataset"] == ds and r["policy"] == pol and r["status"] == "SUCCEEDED"]
            exec_m.append(float(np.mean([float(r["seconds"]) for r in sub])))
            queue_m.append(float(np.mean([float(r["queue_seconds"]) for r in sub])))
        ax.barh(y, exec_m, color=TEAL, height=0.58, label="Execução Fargate (started -> stopped)")
        ax.barh(y, queue_m, left=exec_m, color="#94a3b8", height=0.58, label="Fila Batch (created -> started)")
        ax.set_yticks(y, [SHORT.get(p, p) for p in policies], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold", color=INK, loc="left", pad=6)
        ax.set_xlabel("Duração média por job Batch (segundos)", fontsize=9.5, color=INK)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.13, 0.90), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Tempo real de execução Fargate e fila no AWS Batch (Athena F109)", x=0.14, y=0.96, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(
        0.14, 0.02,
        "Fonte: Athena continual_bench.cost_estimates (133 tentativas da campanha F109).\n"
        "Inclui inicialização do contêiner, avaliação e publicação de artefatos no S3/MLflow.",
        fontsize=9, color=MUTED, va="bottom",
    )
    return fig


def build_aws_cur2_faturamento() -> plt.Figure:
    """Plot real observed AWS CUR 2.0 billing for 2026-11 from continual_bench_billing.cur2."""
    rows = list(csv.DictReader(open(ATHENA_DIR / "cur2_2026_11_by_service.csv", encoding="utf-8")))
    by_svc: dict[str, float] = {}
    fargate_spot_usd, fargate_spot_hrs = 0.0, 0.0
    fargate_od_usd, fargate_od_hrs = 0.0, 0.0
    for r in rows:
        svc = r["line_item_product_code"]
        usd = float(r["unblended_usd"])
        by_svc[svc] = by_svc.get(svc, 0.0) + usd
        ut = r["line_item_usage_type"]
        if "SpotUsage-Fargate-vCPU-Hours" in ut:
            fargate_spot_usd += usd
            fargate_spot_hrs += float(r["usage_amount"])
        elif "Fargate-vCPU-Hours" in ut:
            fargate_od_usd += usd
            fargate_od_hrs += float(r["usage_amount"])

    top_svcs = sorted(by_svc.items(), key=lambda kv: kv[1], reverse=True)[:8]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.4), dpi=300)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.78, bottom=0.20, wspace=0.38)

    y = np.arange(len(top_svcs))
    ax1.barh(y, [v for _, v in top_svcs], color=BLUE, height=0.6)
    for idx, (_, v) in enumerate(top_svcs):
        ax1.text(v + 0.3, idx, f"${v:.2f}", va="center", fontsize=9, color=INK)
    ax1.set_yticks(y, [k.replace("Amazon", "") for k, _ in top_svcs], fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(v for _, v in top_svcs) * 1.25)
    ax1.set_title("Gasto real por serviço (nov/2026)", fontsize=11, fontweight="bold", color=INK, loc="left")
    ax1.set_xlabel("Custo observado (USD)", fontsize=9.5, color=INK)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    spot_rate = fargate_spot_usd / fargate_spot_hrs if fargate_spot_hrs else 0.0
    od_rate = fargate_od_usd / fargate_od_hrs if fargate_od_hrs else 0.04048
    rates = [spot_rate, od_rate]
    bars2 = ax2.bar(["Fargate Spot\n(CUR observado)", "Fargate On-Demand\n(tabela/CUR)"], rates, color=[TEAL, ORANGE], width=0.5)
    for b, v in zip(bars2, rates, strict=True):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"${v:.4f}/vCPU-h", ha="center", fontsize=9.5, color=INK)
    ax2.set_ylim(0, max(rates) * 1.25)
    ax2.set_title("Tarifa efetiva Fargate vCPU/hora", fontsize=11, fontweight="bold", color=INK, loc="left")
    ax2.set_ylabel("USD por vCPU-hora", fontsize=9.5, color=INK)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35, lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Faturamento observado no AWS CUR 2.0 (conta · nov/2026 · total US$ 51,12)", x=0.17, y=0.94, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(
        0.17, 0.02,
        "Fonte: Athena continual_bench_billing.cur2 (billing_period = 2026-11). Escopo da conta inteira no mês (inclui S3, VPC Endpoints e outras campanhas).\n"
        "A tarifa efetiva Spot observada ($0,0148/vCPU-h) é ~2,7x menor que a tarifa On-Demand ($0,0405/vCPU-h) usada nas estimativas de referência.",
        fontsize=8.8, color=MUTED, va="bottom",
    )
    return fig


# 6. Papers: valores publicados de referência (SEM evidencias/semana-10!)
def build_papers_referencia_publicada() -> plt.Figure:
    rows = list(csv.DictReader(open(ROOT / "evidencias/aws-benchmarks/papers/published-reference-values.csv", encoding="utf-8")))
    # Panel A: CARA 2024 Table 2 SCPE (%) on Electricity and Airlines
    cara_pols = ["CARA-T", "CARA-CT", "CARA-P", "Markov", "Congelado"]
    elec_scpe, air_scpe = [], []
    for p in cara_pols:
        e = [float(r["published_value"]) for r in rows if r["paper_id"] == "cara_2024" and r["metric"] == "strategy_scpe" and r["dataset"] == "electricity" and r["policy_label"] == p]
        a = [float(r["published_value"]) for r in rows if r["paper_id"] == "cara_2024" and r["metric"] == "strategy_scpe" and r["dataset"] == "airlines" and r["policy_label"] == p]
        elec_scpe.append(e[0] if e else np.nan)
        air_scpe.append(a[0] if a else np.nan)

    # Panel B: UPF 2025 Table 1 Combined Cost AUC on Electricity and Airplanes
    upf_pols = ["UPF", "CARA-T", "CARA-CT", "CARA-P", "Oracle"]
    elec_auc, air_auc = [], []
    for p in upf_pols:
        e = [float(r["published_value"]) for r in rows if r["paper_id"] == "upf_2025" and r["metric"] == "combined_cost_auc" and r["dataset"] == "electricity" and r["policy_label"] == p]
        a = [float(r["published_value"]) for r in rows if r["paper_id"] == "upf_2025" and r["metric"] == "combined_cost_auc" and r["dataset"] == "airlines" and r["policy_label"] == p]
        elec_auc.append(e[0] if e else np.nan)
        air_auc.append(a[0] if a else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.6), dpi=300)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.22, wspace=0.32)

    y1 = np.arange(len(cara_pols))
    h = 0.34
    ax1.barh(y1 - h / 2, elec_scpe, height=h, color=BLUE, label="electricity")
    ax1.barh(y1 + h / 2, air_scpe, height=h, color=ORANGE, label="airlines")
    ax1.set_yticks(y1, cara_pols, fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_title("CARA (2024) · SCPE publicado (% ↓)", fontsize=11, fontweight="bold", color=INK, loc="left")
    ax1.set_xlabel("Excesso de custo relativo ao Oracle (%, log)", fontsize=9.2, color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    y2 = np.arange(len(upf_pols))
    ax2.barh(y2 - h / 2, elec_auc, height=h, color=BLUE, label="electricity")
    ax2.barh(y2 + h / 2, air_auc, height=h, color=ORANGE, label="airlines")
    ax2.set_yticks(y2, upf_pols, fontsize=9.2)
    ax2.invert_yaxis()
    ax2.set_xlim(2.0, 3.1)
    ax2.set_title("UPF (2025) · AUC de custo publicado (↓)", fontsize=11, fontweight="bold", color=INK, loc="left")
    ax2.set_xlabel("AUC do custo combinado", fontsize=9.2, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35, lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Valores publicados de referência nos artigos base (CARA 2024 e UPF 2025)", x=0.12, y=0.93, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(
        0.12, 0.02,
        "Fonte: evidencias/aws-benchmarks/papers/published-reference-values.csv. Métricas e protocolos próprios dos artigos\n"
        "(Random Forest no CARA; XGBoost eta=0,3 em 15 lotes no UPF), não comparáveis numericamente à campanha AWS F109.",
        fontsize=8.8, color=MUTED, va="bottom",
    )
    return fig


def main():
    if OUT_FIG_ROOT.exists():
        shutil.rmtree(OUT_FIG_ROOT)
    OUT_FIG_ROOT.mkdir(parents=True, exist_ok=True)
    if TMP_OUT.exists():
        shutil.rmtree(TMP_OUT)
    TMP_OUT.mkdir(parents=True)

    apply_style()

    # 1. Generate 'atual' (12 AWS F109 SVGs) from notebooks/
    ev = load_athena(ROOT / "evidencias/aws-benchmarks")
    figs_atual, _ = build_figures(ev, show_title=True, show_footer=True)
    export_bundle(figs_atual, TMP_OUT / "atual", variant="notebook")

    routing = {
        "qualidade": "a4-quality-*.svg",
        "evolucao-temporal": "temporal-*.svg",
        "recursos-e-custos": "a4-resources-*.svg",
    }
    total = 0
    for cat, pat in routing.items():
        dest = OUT_FIG_ROOT / cat
        dest.mkdir(parents=True, exist_ok=True)
        for svg in sorted((TMP_OUT / "atual/notebook/figuras").glob(pat)):
            shutil.copy2(svg, dest / svg.name)
            total += 1

    # 2. Generate 14 new pure-AWS + published-papers SVGs
    new_plots = [
        ("qualidade/aws-cohort-vs-holdout.svg", build_aws_cohort_vs_holdout),
        ("qualidade/aws-calibracao.svg", build_aws_calibracao),
        ("evolucao-temporal/aws-evolucao-vs-congelado.svg", build_aws_evolucao_vs_congelado),
        ("evolucao-temporal/aws-ganho-sobre-congelado.svg", build_aws_ganho_sobre_congelado),
        ("drift-e-retreinos/aws-drift-timeline-insects-abrupto.svg", lambda: build_drift_timeline_insects("insects_abrupt_imbalanced", "INSECTS Abrupto")),
        ("drift-e-retreinos/aws-drift-timeline-insects-recorrente.svg", lambda: build_drift_timeline_insects("insects_incremental_abrupt_reoccurring_imbalanced", "INSECTS Incr. Abr. Recorr.")),
        ("drift-e-retreinos/aws-drift-timeline-electricity-airlines.svg", build_drift_timeline_elec_air),
        ("drift-e-retreinos/aws-versoes-e-retreinos-timeline.svg", build_aws_versoes_timeline),
        ("drift-e-retreinos/aws-drift-alarmes-e-retreinos.svg", build_aws_drift_alarmes_e_retreinos),
        ("esquecimento/aws-retencao-absoluta.svg", build_aws_retencao_absoluta),
        ("esquecimento/aws-plasticidade-vs-estabilidade.svg", build_aws_plasticidade_vs_estabilidade),
        ("recursos-e-custos/aws-quality-vs-cost.svg", build_aws_quality_vs_cost),
        ("recursos-e-custos/aws-decomposicao-custos.svg", build_aws_decomposicao_custos),
        ("recursos-e-custos/aws-batch-duracao-e-fila.svg", build_aws_batch_duracao_e_fila),
        ("recursos-e-custos/aws-cur2-faturamento-real.svg", build_aws_cur2_faturamento),
        ("papers/papers-valores-publicados-referencia.svg", build_papers_referencia_publicada),
    ]
    for rel_path, fn in new_plots:
        save_svg(fn(), rel_path)
        total += 1

    print(f"Total pure-AWS SVGs generated: {total}")
    for sub in sorted(OUT_FIG_ROOT.iterdir()):
        if sub.is_dir():
            files = sorted(p.name for p in sub.glob("*.svg"))
            print(f"  {sub.name}/ ({len(files)}): {', '.join(files)}")


if __name__ == "__main__":
    main()
