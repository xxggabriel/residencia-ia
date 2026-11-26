"""Render the Week 10 CARA and UPF audit figures from archived benchmark data."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

DEFAULT = Path(__file__).resolve().parents[1] / "evidencias/semana-10/cara-upf"
LABELS = {
    "cara": "CARA-T",
    "cara_cumulative": "CARA-CT",
    "cara_periodic": "CARA-P",
    "upf": "UPF",
    "frozen": "Congelado",
    "oracle": "Oracle",
    "oracle_staleness": "Oracle de obsolescência",
    "adwin": "ADWIN",
    "ddm": "DDM",
    "markov": "Markov",
}
COLORS = dict(
    zip(
        LABELS,
        [
            "#2564a0",
            "#178178",
            "#7959a2",
            "#bc532d",
            "#83939e",
            "#172838",
            "#b1812a",
            "#537b2c",
            "#a64c70",
            "#5e767a",
        ],
        strict=True,
    )
)


def archive(root, name):
    with gzip.open(root / "dados" / f"{name}.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def replay_trajectories(root, inputs):
    import xgboost

    from continual_bench.evaluation.retraining_replication import OFFLINE, XGBoostLearner, trial_batches
    from continual_bench.retraining import build_paper_matrix

    rows = []
    for item in archive(root, "same-batch"):
        dataset = item["dataset"]
        policy = {"electricity": "upf", "airlines": "cara"}[dataset]
        metadata = item["comparison"]
        expected = metadata["source"]["sha256"]
        with inputs[dataset].open("rb") as stream:
            input_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if input_hash != expected or xgboost.__version__ != metadata["learner"]["version"]:
            raise ValueError(f"Dados ou versão de XGBoost divergentes em {dataset}")
        seed = 0
        batch = trial_batches(
            dataset,
            seed,
            electricity_path=inputs["electricity"],
            airlines_path=inputs["airlines"],
            size=metadata["batch_size"],
            airlines_disk_cache=False,
        )
        learner = XGBoostLearner(seed)
        matrix, candidates = build_paper_matrix(batch, learner, OFFLINE, seed=seed, separate_query_split=True)
        saved = next(t for t in item["trials_detail"] if t["seed"] == seed)["table"]["low"][policy]
        retrains = saved["retrain_points"]
        deployed = matrix.initial
        trace = []
        for period in range(matrix.offline, matrix.batches):
            if period in retrains:
                deployed = period
            trace.append({"period": period, "model": deployed, "accuracy": 1 - matrix.err[deployed, period]})
        mean_accuracy = float(np.mean([row["accuracy"] for row in trace]))
        implied_cost = len(retrains) * saved["alpha"] + len(trace) * (1 - mean_accuracy)
        if not np.isclose(mean_accuracy, saved["average_accuracy"]) or not np.isclose(
            implied_cost, saved["cost"]
        ):
            raise ValueError(f"Trajetória não confere com acurácia e custo arquivados em {dataset}")
        retained = item["candidate_tracking"][seed]["training_history"]
        hashes = [
            hashlib.sha256(learner.observation_checkpoint(candidate)[0]).hexdigest()
            for candidate in candidates
        ]
        recorded = [fit["checkpoint"]["sha256"] for fit in retained]
        if hashes != recorded:
            raise ValueError(f"Checkpoints reconstituídos divergentes em {dataset}")
        scores = 1 - matrix.err
        initial_test, initial_labels = batch[matrix.initial][2:]
        initial_predictions, _ = learner.observation_predict(candidates[matrix.initial], initial_test, 1)
        scores[matrix.initial, matrix.initial] = np.mean(initial_predictions == initial_labels)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "policy": policy,
                "alpha": saved["alpha"],
                "mode": "same-batch",
                "offline": matrix.offline,
                "source_sha256": input_hash,
                "test_counts": [len(x[3]) for x in batch],
                "candidate_sha256": hashes,
                "accuracy_matrix": [[float(v) if np.isfinite(v) else None for v in line] for line in scores],
                "retrains": retrains,
                "deployed_trace": trace,
                "average_accuracy": mean_accuracy,
                "cost": float(implied_cost),
            }
        )
        print(f"{dataset}: {len(hashes)} checkpoints, acurácia e custo conferidos", flush=True)
    (root / "dados" / "trajetorias.json").write_text(json.dumps(rows, indent=2) + "\n")


def save(root, fig, name):
    fig.savefig(root / "figuras" / f"{name}.png", dpi=175)
    plt.close(fig)


def setup_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.19)
    ax.set_axisbelow(True)


def wilson(p, n):
    z = 1.959964
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - width, center + width


def plot_trajectory(root, row):
    fig, ax = plt.subplots(figsize=(12, 5.3), layout="constrained")
    fig.suptitle(
        f"{row['dataset'].title()} · acurácia por lote e retreinamento",
        fontsize=18,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    matrix = np.array(row["accuracy_matrix"], dtype=float)
    first = row["offline"] - 1
    palettes = plt.get_cmap("turbo")(np.linspace(0.10, 0.90, len(matrix) - first))
    sizes = np.array(row["test_counts"])
    for i in range(first, len(matrix)):
        period = np.arange(i, len(matrix))
        scores = matrix[i, period]
        a, b = wilson(scores, sizes[period])
        color = palettes[i - first]
        ax.plot(period, scores, color=color, marker="o", ms=3.5, lw=1.5, label=f"Modelo D{i}")
        ax.fill_between(period, a, b, color=color, alpha=0.085, linewidth=0)
    trace = row["deployed_trace"]
    ax.plot(
        [r["period"] for r in trace],
        [r["accuracy"] for r in trace],
        color="#162331",
        lw=2.6,
        linestyle="-.",
        marker="o",
        label=f"Em uso · {LABELS[row['policy']]}",
    )
    for index, period in enumerate(row["retrains"]):
        ax.axvline(period, color="#526170", linestyle=":", linewidth=1)
        ax.text(
            period,
            0.97 if index % 2 == 0 else 0.88,
            "Retreino",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#8f382d",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#e4b7ae", "boxstyle": "round,pad=.24"},
        )
    ax.set_title(
        "Benchmark pareado no mesmo lote · seed 0 · α = 0,1 · 2.000 registros por lote, 400 no teste",
        loc="left",
        fontsize=10,
    )
    ax.set_xticks(range(first, len(matrix)), [f"D{i}" for i in range(first, len(matrix))])
    ax.set_xlabel("Lote avaliado · intervalos de Wilson de 95% para a acurácia em cada lote")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(top=min(1.05, ax.get_ylim()[1] + 0.065))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    setup_axis(ax)
    ax.legend(ncols=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    save(root, fig, f"trajetoria-{row['dataset']}")


def plot_curves(root, rows, protocol):
    cara = protocol == "cara"
    columns = (
        [
            ("strategy_cost", "Custo da estratégia", False),
            ("strategy_scpe", "SCPE (%) · κ > 0", False),
            ("average_accuracy", "Acurácia", True),
            ("retrains", "Retreinamentos", False),
        ]
        if cara
        else [
            ("cost", "Custo de erro e retreino", False),
            ("average_accuracy", "Acurácia", True),
            ("retrains", "Retreinamentos", False),
        ]
    )
    fig, axes = plt.subplots(2, len(columns), figsize=(15, 7.8), layout="constrained")
    fig.suptitle(
        f"Electricity e Airlines · protocolo {protocol.upper()} · média de {5 if cara else 10} seeds",
        fontsize=18,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    for i, row in enumerate(rows):
        grid = row["kappa" if cara else "alphas"]
        for j, (metric, name, percent) in enumerate(columns):
            ax = axes[i, j]
            for p, curves in row["curves"].items():
                ax.plot(
                    grid,
                    curves[metric],
                    color="#222222" if p == "always_retrain" else COLORS.get(p),
                    linestyle="--" if p == "always_retrain" else "-",
                    marker="o" if not cara else None,
                    markersize=2.5,
                    lw=1.55,
                    label=LABELS.get(p, "Sempre retreinar" if p == "always_retrain" else p),
                )
            ax.set_title(f"{row['dataset'].title()} · {name}", loc="left", fontsize=10)
            ax.set_xlabel("κ · preço do retreino" if cara else "α · preço relativo do retreino")
            if cara and metric in {"strategy_cost", "strategy_scpe"}:
                ax.set_yscale("symlog", linthresh=10)
                ax.set_ylabel("Escala log simétrica", fontsize=8)
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            setup_axis(ax)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=7, frameon=False, fontsize=9)
    save(root, fig, f"metricas-protocolo-{protocol}")


def plot_auc(root, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    fig.suptitle(
        "Electricity e Airlines · AUC do custo combinado no protocolo UPF",
        fontsize=16,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    for ax, row in zip(axes, rows, strict=True):
        data = [(p, s) for p, s in row["summary"].items() if s["paper_auc"] is not None]
        y = np.arange(len(data))
        ax.barh(
            y - 0.16,
            [s["auc_mean"] for _, s in data],
            height=0.29,
            color="#2564a0",
            xerr=[s["auc_std"] for _, s in data],
            capsize=3,
            label="Local · média ± 1 desvio padrão (10 seeds)",
        )
        ax.barh(y + 0.16, [s["paper_auc"] for _, s in data], height=0.29, color="#a1b1bb", label="Publicado")
        ax.set_yticks(y, [LABELS[p] for p, _ in data])
        ax.invert_yaxis()
        ax.set_title(row["dataset"].title(), loc="left", fontweight="bold")
        alpha_top = str(row["alphas"][-1]).replace(".", ",")
        ax.set_xlabel(f"AUC do custo combinado · α de 0 a {alpha_top}")
        setup_axis(ax)
    handles, names = axes[0].get_legend_handles_labels()
    right = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, right)
    fig.legend(handles, names, loc="outside lower center", ncols=2, frameon=False)
    save(root, fig, "auc-upf-publicado-local")


def plot_gap(root):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.7), layout="constrained")
    fig.suptitle(
        "Electricity e Airlines · diferença de custo combinado para o Oracle de erro",
        fontsize=17,
        fontweight="bold",
    )
    modes = ("next-batch", "same-batch")
    mode_labels = ("Implantação no lote seguinte", "Implantação no mesmo lote")
    policies = ("frozen", "cara", "cara_cumulative", "cara_periodic", "upf")
    all_rows = [archive(root, mode) for mode in modes]

    def gaps(row):
        return np.array([row["curves"][p]["error_oracle_gap"] for p in policies])

    norm = matplotlib.colors.PowerNorm(
        0.65, vmin=0, vmax=max(gaps(x).max() for rows in all_rows for x in rows)
    )
    for i, rows in enumerate(all_rows):
        for j, row in enumerate(rows):
            ax = axes[i, j]
            values = gaps(row)
            image = ax.imshow(values, cmap="YlOrBr", norm=norm, aspect="auto", interpolation="none")
            ticks = [
                k
                for k, a in enumerate(row["alphas"])
                if a in (0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1)
            ]
            ax.set_xticks(ticks, [str(row["alphas"][k]).replace(".", ",") for k in ticks], fontsize=8)
            ax.set_yticks(range(len(policies)), [LABELS[p] for p in policies])
            ax.set_title(f"{row['dataset'].title()} · {mode_labels[i]}", loc="left")
            ax.set_xlabel("α · valores avaliados em colunas de mesma largura")
    fig.colorbar(image, ax=axes, label="Diferença de custo combinado", fraction=0.037)
    save(root, fig, "sensibilidade-oracle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--electricity", type=Path)
    parser.add_argument("--airlines", type=Path)
    args = parser.parse_args()
    if bool(args.electricity) != bool(args.airlines):
        parser.error("informe os dois arquivos de entrada para refazer as trajetórias")
    if args.electricity:
        replay_trajectories(args.root, {"electricity": args.electricity, "airlines": args.airlines})
    (args.root / "figuras").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#182938", "axes.labelcolor": "#304351"})
    for row in json.loads((args.root / "dados" / "trajetorias.json").read_text()):
        plot_trajectory(args.root, row)
    upf = archive(args.root, "upf")
    plot_auc(args.root, upf)
    plot_curves(args.root, upf, "upf")
    plot_curves(args.root, [archive(args.root, f"cara-{d}")[0] for d in ("electricity", "airlines")], "cara")
    plot_gap(args.root)


if __name__ == "__main__":
    main()
