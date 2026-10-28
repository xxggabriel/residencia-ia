"""Render only complete, frozen nonlinear-v5 evidence; never train or tune on results."""

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ("logistic", "hoeffding_tree", "knn")
DATASETS = ("sea", "agrawal", "stagger", "electricity", "airlines")
MODES = ("streaming", "incremental")
METRICS = ("accuracy", "f1", "brier", "log_loss")
NAMES = {"logistic": "Logística", "hoeffding_tree": "Hoeffding Tree", "knn": "kNN"}
MODE_NAMES = {"streaming": "streaming", "incremental": "incremental", "offline": "offline"}


def describe(values):
    assert values and all(math.isfinite(value) for value in values), "invalid numeric observations"
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def arm(report):
    cfg = report["config"]
    name = cfg["detector"]["name"] if cfg["policy"]["name"] == "drift" else cfg["policy"]["name"]
    return f"{name}@{cfg['training_budget']}" if name in {"periodic", "adwin", "ddm"} else name


def stat_text(stat, *, percent=False):
    factor, digits = (100, 2) if percent else (1, 4)
    text = f"{stat['mean'] * factor:.{digits}f}"
    if stat["sd"] is not None:
        text += f" ± {stat['sd'] * factor:.{digits}f}"
    return text


def direction_text(delta, metric):
    if delta == 0:
        return "empata"
    improved = delta > 0 if metric in {"accuracy", "f1"} else delta < 0
    return "melhora" if improved else "piora"


def _confusion(report, reference):
    assert reference["dataset_sha256"] == report["dataset"]["sha256"], "reference dataset differs"
    n, positives = reference["evaluation_n"], reference["evaluation_positive"]
    metrics = report["metrics"]
    assert metrics["n"] == n and 0 <= positives <= n, "reference evaluation population differs"
    accuracy, f1 = metrics["accuracy"], metrics["f1"]
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in (accuracy, f1))
    correct_float = accuracy * n
    correct = round(correct_float)
    assert math.isclose(correct, correct_float, rel_tol=0, abs_tol=1e-6), "nonintegral correct count"
    errors = n - correct
    tp_float = f1 * errors / (2 * (1 - f1)) if f1 < 1 else positives
    tp = round(tp_float)
    assert math.isclose(tp, tp_float, rel_tol=0, abs_tol=1e-6), "nonintegral true-positive count"
    fn, fp = positives - tp, errors - (positives - tp)
    tn = n - tp - fn - fp
    assert min(tp, fp, tn, fn) >= 0 and tp + tn == correct, "inconsistent confusion matrix"
    reconstructed = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    assert math.isclose(reconstructed, f1, rel_tol=0, abs_tol=1e-12), "F1 reconstruction differs"
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / positives if positives else 0,
    }


def _storage(records):
    """Measured payload bytes only: distinguish references from content-addressed journal objects."""
    by_model = defaultdict(list)
    for record in records:
        by_model[record["report"]["config"]["model"]["name"]].append(record)

    def measure(rows):
        sums = Counter()
        known = Counter()
        journals = {}
        for record in rows:
            sizes = record.get("artifact_sizes") or {}
            for name in ("model_states_bytes", "model_index_bytes", "final_model_bytes"):
                value = sizes.get(name)
                if value is not None:
                    assert isinstance(value, int) and value >= 0, "invalid artifact byte count"
                    sums[name] += value
                    known[name] += 1
            if sizes.get("model_states_bytes") is not None:
                uri = record["receipt"]["model_states_uri"]
                if uri in journals:
                    assert journals[uri] == sizes["model_states_bytes"], "same journal URI has two sizes"
                journals[uri] = sizes["model_states_bytes"]
        return {
            "runs": len(rows),
            "measured_records": dict(known),
            "reference_bytes": {
                name: sums[name] if known[name] else None
                for name in ("model_states_bytes", "model_index_bytes", "final_model_bytes")
            },
            "unique_journal_objects": len(journals),
            "unique_journal_bytes": sum(journals.values()) if journals else None,
        }

    return {
        "all": measure(records),
        "by_model": {model: measure(rows) for model, rows in by_model.items()},
        "scope": "Payload measurements when present; not total disk usage, replicas, logs or CPU cost",
    }


def validate(comparison, current, references, manifest, *, current_sha256, manifest_sha256):
    assert current["complete_campaign"] and current["runs_verified"] == 867, "requires complete 867-run audit"
    assert comparison["complete"] and comparison["runs"] == 867, "comparison is incomplete"
    assert manifest["campaign"] == comparison["campaign"] == current["campaign"] == "nonlinear-v5"
    assert manifest["expected_runs"] == 867 and manifest["parallel_tasks"] == 8
    assert comparison["cases_per_model"] == 289 and set(comparison["models"]) == set(MODELS)
    assert comparison["input_sha256"]["current"] == current_sha256, "comparison uses another current input"
    assert comparison["input_sha256"].get("manifest") == manifest_sha256, "comparison uses another manifest"
    source = manifest["source_sha256"]
    assert comparison["source_sha256"] == current["source_sha256"] == source, "frozen source differs"
    expected = {row["run_key"]: row for row in manifest["configurations"]}
    records = current["records"]
    assert len(expected) == len(records) == 867
    assert {row["report"]["run_key"] for row in records} == set(expected), "manifest run coverage differs"
    ref_index = {(r["dataset"], r["seed"], r["mode"]): r for r in references["references"]}
    assert len(ref_index) == len(references["references"]) == 34
    matrices, grouped_matrices, raw_index = {}, defaultdict(list), {}
    model_counts = Counter()
    source_counts = Counter()
    for record in records:
        report = record["report"]
        key, cfg = report["run_key"], report["config"]
        assert record["publication_verified"] and report["complete"], "unverified run"
        assert cfg == expected[key]["config"] and cfg["name"] == "nonlinear-v5"
        expected_source = expected[key].get("source_sha256", source)
        assert isinstance(expected_source, str) and expected_source, "invalid per-run source override"
        assert report["environment"]["code_sha256"] == expected_source, "unauthorized per-run source"
        source_counts[expected_source] += 1
        model, dataset, mode, seed = (
            cfg["model"]["name"],
            cfg["dataset"]["name"],
            cfg["mode"],
            cfg["dataset"]["seed"],
        )
        assert model in MODELS and dataset in DATASETS and mode in (*MODES, "offline")
        reference = ref_index[(dataset, seed, "offline" if mode == "offline" else "prequential")]
        matrices[key] = _confusion(report, reference)
        group = (model, dataset, mode, arm(report))
        grouped_matrices[group].append(matrices[key])
        raw_key = (*group, seed)
        assert raw_key not in raw_index, "duplicate model/context/seed"
        raw_index[raw_key] = report
        model_counts[model] += 1
    assert model_counts == Counter({model: 289 for model in MODELS})
    assert comparison["source_counts"] == dict(source_counts), "comparison source counts differ"
    assert comparison["source_revisions"] == sorted(source_counts), "comparison source revisions differ"
    if "source_counts" in current:
        assert current["source_counts"] == dict(source_counts), "audit source counts differ"
    if "source_revisions" in current:
        assert current["source_revisions"] == sorted(source_counts), "audit source revisions differ"
    groups = comparison["groups"]
    assert len(groups) == 255 and len(comparison["paired_model_comparisons"]) == 170
    lookup = {}
    for group in groups:
        key = (group["model"], group["dataset"], group["mode"], group["strategy"])
        assert key not in lookup and key in grouped_matrices
        n = 1 if group["dataset"] in {"electricity", "airlines"} else 5
        assert len(grouped_matrices[key]) == group["metrics"]["accuracy"]["n"] == n
        lookup[key] = group
    assert set(lookup) == set(grouped_matrices)
    return {
        "lookup": lookup,
        "raw_index": raw_index,
        "matrices": matrices,
        "matrix_groups": grouped_matrices,
        "reference_index": ref_index,
    }


def summarize(comparison, current, references, manifest, checked):
    lookup, raw_index = checked["lookup"], checked["raw_index"]
    diagnostics = {
        key: {metric: describe([row[metric] for row in rows]) for metric in ("precision", "recall")}
        for key, rows in checked["matrix_groups"].items()
    }
    paired = {
        (p["model"], p["dataset"], p["mode"], p["strategy"]): p
        for p in comparison["paired_model_comparisons"]
    }
    online, deltas, rankings, budget_pairs = [], [], [], []
    for dataset in DATASETS:
        seeds = [42] if dataset in {"electricity", "airlines"} else list(range(42, 47))
        for mode in MODES:
            for model in MODELS:
                key = (model, dataset, mode, "online")
                group = lookup[key]
                online.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "mode": mode,
                        "metrics": group["metrics"],
                        **diagnostics[key],
                        "training_used": group["training_used"],
                        "run_keys": group["run_keys"],
                    }
                )
                if model != "logistic":
                    control = lookup[("logistic", dataset, mode, "online")]
                    mean_deltas = {}
                    for metric in METRICS:
                        delta = group["metrics"][metric]["mean"] - control["metrics"][metric]["mean"]
                        reported = paired[key]["metrics"][metric]
                        assert math.isclose(delta, reported["mean"], rel_tol=0, abs_tol=1e-12)
                        assert reported["higher_is_better"] == (metric in {"accuracy", "f1"})
                        if abs(delta) > 1e-12:
                            assert (delta > 0) == (reported["mean"] > 0), "plot delta sign mismatch"
                        mean_deltas[metric] = delta
                    deltas.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "mode": mode,
                            "delta_definition": "nonlinear online minus logistic online",
                            "metric_deltas": mean_deltas,
                            "paired_statistics": paired[key]["metrics"],
                        }
                    )
                for cap in (2900, 5800):
                    strategies = ("none", "online", f"periodic@{cap}", f"adwin@{cap}", f"ddm@{cap}")
                    candidates = [lookup[(model, dataset, mode, strategy)] for strategy in strategies]
                    best = max(row["metrics"]["accuracy"]["mean"] for row in candidates)
                    winners = [
                        row
                        for row in candidates
                        if math.isclose(row["metrics"]["accuracy"]["mean"], best, rel_tol=0, abs_tol=1e-12)
                    ]
                    rankings.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "mode": mode,
                            "cap": cap,
                            "best_accuracy_mean": best,
                            "best_strategies": [row["strategy"] for row in winners],
                            "selection": "Post-hoc benchmark selection; not independent validation",
                        }
                    )
                for strategy in ("periodic", "adwin", "ddm"):
                    low = [raw_index[(model, dataset, mode, f"{strategy}@2900", seed)] for seed in seeds]
                    high = [raw_index[(model, dataset, mode, f"{strategy}@5800", seed)] for seed in seeds]
                    available = [
                        bool(a.get("prediction_digest")) and bool(b.get("prediction_digest"))
                        for a, b in zip(low, high)
                    ]
                    budget_pairs.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "mode": mode,
                            "strategy": strategy,
                            "pairs": len(seeds),
                            "available_digest_pairs": sum(available),
                            "identical_prediction_pairs": sum(
                                ok and a["prediction_digest"] == b["prediction_digest"]
                                for ok, a, b in zip(available, low, high)
                            ),
                            "low_training_used": describe(
                                [sum(u["train_samples"] for u in a["updates"]) for a in low]
                            ),
                            "high_training_used": describe(
                                [sum(u["train_samples"] for u in a["updates"]) for a in high]
                            ),
                            "high_minus_low": {
                                metric: describe(
                                    [b["metrics"][metric] - a["metrics"][metric] for a, b in zip(low, high)]
                                )
                                for metric in METRICS
                            },
                        }
                    )
    reference_groups = defaultdict(list)
    for reference in references["references"]:
        reference_groups[(reference["dataset"], reference["mode"])].append(reference)
    reference_summary = [
        {
            "dataset": dataset,
            "mode": mode,
            "positive_rate": describe([r["evaluation_positive_rate"] for r in rows]),
            **{
                name: {metric: describe([r[name][metric] for r in rows]) for metric in ("accuracy", "f1")}
                for name in ("always_positive", "always_negative")
            },
        }
        for (dataset, mode), rows in sorted(reference_groups.items())
    ]
    return {
        "campaign": "nonlinear-v5",
        "complete": True,
        "runs": 867,
        "source_sha256": manifest["source_sha256"],
        "source_revisions": comparison["source_revisions"],
        "source_counts": comparison["source_counts"],
        "online": online,
        "online_deltas": deltas,
        "descriptive_accuracy_rankings": rankings,
        "budget_pairs": budget_pairs,
        "confusion_matrices": checked["matrices"],
        "label_references": reference_summary,
        "storage": _storage(current["records"]),
        "logistic_v4_comparison": comparison["logistic_v4_comparison"],
        "versioning": {
            key: current.get(key)
            for key in (
                "dataset_versions",
                "registered_final_versions",
                "registered_model_families",
                "snapshot_references",
                "unique_snapshot_ids",
                "unique_state_hashes",
            )
        },
        "limitations": comparison["limitations"],
        "reference_scope": references["scope"],
    }


def plot(summary, base):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(r["model"], r["dataset"], r["mode"]): r for r in summary["online_deltas"]}
    contexts = [(dataset, mode) for dataset in DATASETS for mode in MODES]
    models = ("hoeffding_tree", "knn")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.2), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "f1"), ("Acurácia", "F1")):
        values = np.array(
            [
                [100 * lookup[(model, dataset, mode)]["metric_deltas"][metric] for model in models]
                for dataset, mode in contexts
            ]
        )
        assert np.isfinite(values).all()
        limit = max(1, math.ceil(float(np.abs(values).max())))
        shown = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(2), [NAMES[model] for model in models])
        ax.set_yticks(
            range(len(contexts)),
            [
                dataset.upper() if dataset in {"sea", "stagger"} else dataset.title()
                for dataset, _ in contexts
            ],
        )
        ax.set_yticklabels(
            [
                (dataset.upper() if dataset in {"sea", "stagger"} else dataset.title())
                + (" · evento" if mode == "streaming" else " · lote")
                for dataset, mode in contexts
            ]
        )
        ax.set_title(f"Online: diferença de {title} vs. logística (p.p.)", fontsize=11)
        for y in range(len(contexts)):
            for x in range(2):
                value = float(values[y, x])
                display = 0.0 if abs(value) < 0.05 else value
                ax.text(
                    x,
                    y,
                    f"{display:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if abs(value) > 0.6 * limit else "#17283b",
                )
        fig.colorbar(shown, ax=ax, shrink=0.75, label="pontos percentuais; positivo favorece não linear")
    fig.suptitle("Campanha nonlinear-v5 · mesmo braço online e 2.900 apresentações pós-warmup", fontsize=13)
    fig.supxlabel(
        "Sintéticos: média de 5 sementes (SD nas tabelas). Reais: um prefixo de 3.000 exemplos, n=1.",
        fontsize=9,
    )
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def render_report(
    comparison, summary, environment=None, airflow=None, recovery=None, execution=None, diagnostic=None
):
    online_index = {(row["model"], row["dataset"], row["mode"]): row for row in summary["online"]}
    bridge = summary["logistic_v4_comparison"]
    text = [
        "# Campanha nonlinear-v5 — comparação de modelos e políticas\n",
        "A campanha contém 867 runs verificadas: 289 casos pareados para cada modelo. "
        "Os dados, sementes, protocolos e tetos de treino foram preservados em relação à v4. "
        "A tabela principal mantém a política online comum; escolher a melhor política depois de "
        "ver os resultados é uma análise descritiva, não validação independente.\n",
    ]
    for dataset in ("agrawal", "airlines"):
        rows = [online_index[(model, dataset, "streaming")] for model in MODELS]
        values = "; ".join(
            f"{NAMES[row['model']]}: acurácia {row['metrics']['accuracy']['mean'] * 100:.2f}%, "
            f"F1 {row['metrics']['f1']['mean'] * 100:.2f}%"
            for row in rows
        )
        text.append(f"**{dataset.title()}, online por evento:** {values}.\n")
    text.append(
        "Em Agrawal, acurácia e F1 melhoram com ambos os modelos não lineares. "
        "Em Airlines, o F1 sobe, mas a acurácia "
        "e o log loss pioram: não houve melhora geral. A árvore também piora o log loss em Agrawal; "
        "o kNN melhora essa métrica nesse dataset. As tabelas abaixo separam protocolos e dispersão.\n"
    )
    versions = summary["versioning"]
    text.append(
        f"A auditoria verificou {versions['dataset_versions']} versões de conteúdo dos datasets, "
        f"{versions['registered_final_versions']} versões finais no Registry, "
        f"{versions['registered_model_families']} famílias e "
        f"{versions['snapshot_references']} referências a estados treinados.\n"
    )
    if airflow is not None:
        text.append(
            f"Estado Airflow registrado: `{airflow.get('state')}`; "
            f"tarefas por estado: `{json.dumps(airflow.get('task_states', {}), sort_keys=True)}`; "
            f"tentativas adicionais registradas: {airflow.get('retry_attempts', 'n/d')}, "
            "incluindo as duas recuperações manuais.\n"
        )
    if execution is not None:
        text.append(
            f"A execução confirmou pico de {execution['peak_evaluate_and_publish']} tarefas "
            f"`evaluate_and_publish` e {execution['peak_worker_active']} processos ativos; "
            f"{execution['elapsed_wall_seconds_including_recovery'] / 60:.2f} minutos entre o início "
            "original e a consolidação final, incluindo a recuperação. "
            "[Registro operacional](nonlinear-v5-execution.json), "
            "[primeira tentativa](nonlinear-v5-first-attempt-airflow.json), "
            "[estado final Airflow](nonlinear-v5-airflow.json). "
            "O Airflow redefine a data de início ao limpar tarefas; a duração usa a data preservada.\n"
        )
    text.extend(
        [
            "## Revisões autorizadas e recuperação\n",
            f"Fonte principal do manifesto: `{summary['source_sha256']}`. "
            "Cada run deve corresponder ao SHA padrão ou a um override explícito na sua própria linha "
            "do manifesto. Uma lista global de revisões não autoriza fontes adicionais.\n",
            "| Revisão SHA-256 | Runs autorizadas e verificadas |",
            "|---|---:|",
        ]
    )
    for revision, count in sorted(summary["source_counts"].items()):
        text.append(f"| `{revision}` | {count} |")
    if len(summary["source_revisions"]) > 1:
        text.append(
            "\nA recuperação trata um cache de inferência da árvore: o splitter nominal criava dicionários "
            "vazios para classes não observadas durante predict. A correção remove apenas esses caches "
            "novos/vazios. As configurações, dados, parâmetros dos modelos, protocolos e tetos continuam "
            "iguais aos casos do manifesto. Isso não substitui a conferência de replay/paridade dos casos "
            "reexecutados, nem cria réplicas independentes.\n"
        )
    if recovery is not None:
        recovered = recovery["runs"]
        assert recovery["recovered_runs"] == len(recovered) == 2
        assert all(all(row["parity"].values()) for row in recovered)
        text.append(
            "As duas execuções recuperadas reproduziram exatamente métricas, digest de predições, "
            "alarmes, atualizações sem os tempos, ledger de treino, dados e configuração. "
            "As tentativas anteriores foram preservadas. "
            "[Evidência completa de recuperação e replay](nonlinear-v5-recovery.json).\n"
        )
    elif len(summary["source_revisions"]) > 1:
        text.append(
            "Evidência de replay/paridade ainda não anexada a este renderer. "
            "Não se declara paridade dessa recuperação sem o arquivo correspondente.\n"
        )
    text.extend(
        [
            "## Comparação online, sem selecionar política pelo resultado\n",
            "Acurácia, F1, precisão e revocação em %. Sintéticos: média ± desvio-padrão amostral "
            "de cinco sementes; reais: um valor, sem estimativa de variância. Brier e log loss: "
            "menor é melhor. Não são intervalos de confiança nem testes de significância.\n",
            "| Dataset | Protocolo | Modelo | Acurácia | F1 | Brier | Log loss | Precisão | Revocação |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["online"]:
        metrics = row["metrics"]
        text.append(
            f"| {row['dataset'].title()} | {MODE_NAMES[row['mode']]} | {NAMES[row['model']]} | "
            f"{stat_text(metrics['accuracy'], percent=True)} | {stat_text(metrics['f1'], percent=True)} | "
            f"{stat_text(metrics['brier'])} | {stat_text(metrics['log_loss'])} | "
            f"{stat_text(row['precision'], percent=True)} | {stat_text(row['recall'], percent=True)} |"
        )
    text.extend(
        [
            "\n![Diferenças online contra logística](nonlinear-v5-online-comparison.png)\n",
            "## Resultados observados, com foco em Agrawal e Airlines\n",
        ]
    )
    for model in ("hoeffding_tree", "knn"):
        selected = [row for row in summary["online_deltas"] if row["model"] == model]
        counts = {
            metric: sum(
                row["metric_deltas"][metric] > 0
                if metric in {"accuracy", "f1"}
                else row["metric_deltas"][metric] < 0
                for row in selected
            )
            for metric in METRICS
        }
        text.append(
            f"**{NAMES[model]}:** teve média melhor que a logística em {counts['accuracy']}/10 "
            f"cenários online de acurácia, {counts['f1']}/10 de F1, {counts['brier']}/10 de Brier "
            f"e {counts['log_loss']}/10 de log loss. Cada cenário é dataset/protocolo, "
            "com dados compartilhados; essas contagens não são réplicas independentes.\n"
        )
    for dataset in ("agrawal", "airlines"):
        for mode in MODES:
            base = online_index[("logistic", dataset, mode)]
            text.append(
                f"**{dataset.title()} — {MODE_NAMES[mode]}:** logística online tem acurácia "
                f"{base['metrics']['accuracy']['mean'] * 100:.2f}%, F1 "
                f"{base['metrics']['f1']['mean'] * 100:.2f}% e revocação "
                f"{base['recall']['mean'] * 100:.2f}%. "
            )
            for model in ("hoeffding_tree", "knn"):
                row = online_index[(model, dataset, mode)]
                deltas = next(
                    d
                    for d in summary["online_deltas"]
                    if d["model"] == model and d["dataset"] == dataset and d["mode"] == mode
                )["metric_deltas"]
                text.append(
                    f"{NAMES[model]}: acurácia {row['metrics']['accuracy']['mean'] * 100:.2f}% "
                    f"({deltas['accuracy'] * 100:+.2f} p.p.; "
                    f"{direction_text(deltas['accuracy'], 'accuracy')}), "
                    f"F1 {row['metrics']['f1']['mean'] * 100:.2f}% "
                    f"({deltas['f1'] * 100:+.2f} p.p.; {direction_text(deltas['f1'], 'f1')}), "
                    f"revocação {row['recall']['mean'] * 100:.2f}%. "
                )
            text.append("\n")
        reference = next(
            row
            for row in summary["label_references"]
            if row["dataset"] == dataset and row["mode"] == "prequential"
        )
        text.append(
            f"Referências constantes calculadas a posteriori para {dataset.title()}: "
            f"sempre positivo tem acurácia {reference['always_positive']['accuracy']['mean'] * 100:.2f}% "
            f"e F1 {reference['always_positive']['f1']['mean'] * 100:.2f}%; sempre negativo tem "
            f"acurácia {reference['always_negative']['accuracy']['mean'] * 100:.2f}% e F1 zero. "
            "São controles de interpretação, não novos braços treinados ou seleção independente.\n"
        )
    if diagnostic is not None:
        checked = diagnostic["measured_summary"]
        assert diagnostic["campaign"] == "nonlinear-v5" and checked["runs"] == 10
        assert checked["all_final_trees_one_node"] and checked["all_final_grids_constant_negative"]
        text.append(
            "**Diagnóstico da árvore no STAGGER:** os dez casos online terminaram com uma única "
            "folha e nenhuma divisão. Receberam as 3000 amostras, incluindo 2900 atualizações após "
            "o aquecimento; a folha NBA escolheu a maioria e previu negativo nos 27 padrões possíveis. "
            "As condições de split desse perfil permaneceram conservadoras dentro do horizonte, "
            "sem bloqueio de crescimento por memória. Isso explica o F1 zero observado nesse braço; "
            "não demonstra que outra configuração teria bom resultado. "
            "[Diagnóstico medido e limites da inferência](nonlinear-v5-stagger-tree-diagnostic.json).\n"
        )
    text.extend(
        [
            "## Políticas: melhor acurácia observada por teto\n",
            "Seleção **a posteriori** pela acurácia média no próprio benchmark. Empates de média "
            "são preservados. Esses vencedores não foram selecionados em validação independente. "
            "Online usa 2900 apresentações nos dois tetos; congelado usa zero após o warmup.\n",
            "| Dataset | Protocolo | Modelo | Teto | Política(s) com maior média | Acurácia (%) |",
            "|---|---|---|---:|---|---:|",
        ]
    )
    for row in summary["descriptive_accuracy_rankings"]:
        text.append(
            f"| {row['dataset'].title()} | {row['mode']} | {NAMES[row['model']]} | {row['cap']} | "
            f"{' / '.join(row['best_strategies'])} | {row['best_accuracy_mean'] * 100:.2f} |"
        )
    group_index = {
        (row["model"], row["dataset"], row["mode"], row["strategy"]): row
        for row in comparison["groups"]
    }
    tree_adwin = group_index[("hoeffding_tree", "agrawal", "streaming", "adwin@2900")]
    tree_ddm = group_index[("hoeffding_tree", "agrawal", "streaming", "ddm@2900")]
    knn_ddm = group_index[("knn", "airlines", "streaming", "ddm@2900")]
    tree_online = online_index[("hoeffding_tree", "agrawal", "streaming")]
    knn_online = online_index[("knn", "airlines", "streaming")]
    text.extend(
        [
            "\n## Detectores e interpretação das políticas\n",
            "Os detectores recebem o erro do modelo: mudar o estimador altera também a sequência "
            "de alarmes. Os resultados avaliam a combinação modelo/detector/política; não isolam "
            "a qualidade intrínseca do detector. Não há vencedor único em todas as métricas.\n",
            f"Em Agrawal/streaming, árvore+ADWIN tem acurácia média "
            f"{tree_adwin['metrics']['accuracy']['mean'] * 100:.2f}% e árvore+DDM "
            f"{tree_ddm['metrics']['accuracy']['mean'] * 100:.2f}% no teto2900, abaixo da árvore online "
            f"({tree_online['metrics']['accuracy']['mean'] * 100:.2f}%). "
            "Isso não implica ordenação igual por F1, probabilidades ou consumo.\n",
            f"Em Airlines/streaming, kNN+DDM alcança acurácia "
            f"{knn_ddm['metrics']['accuracy']['mean'] * 100:.2f}%, mas F1 "
            f"{knn_ddm['metrics']['f1']['mean'] * 100:.2f}%, contra "
            f"{knn_online['metrics']['accuracy']['mean'] * 100:.2f}% "
            f"e {knn_online['metrics']['f1']['mean'] * 100:.2f}% "
            "do kNN online. Escolher apenas pela acurácia esconderia a pior detecção da classe positiva.\n",
        ]
    )
    for detector in ("adwin", "ddm"):
        pairs = [row for row in summary["budget_pairs"] if row["strategy"] == detector]
        identical = sum(row["identical_prediction_pairs"] for row in pairs)
        total = sum(row["available_digest_pairs"] for row in pairs)
        text.append(
            f"**{detector.upper()}:** {identical}/{total} pares modelo/dataset/semente/protocolo "
            "mantiveram o digest de predições ao elevar o teto de2900 para5800. "
            "O teto maior só pode ajudar quando muda as atualizações efetivamente executadas; "
            "não se deve inferir consumo idêntico entre políticas.\n"
        )
    text.extend(
        [
            "\n## Consumo efetivo e mudança de orçamento\n",
            "Treino é contagem de apresentações após o aquecimento, incluindo replays. "
            "Não equivale a segundos de CPU, tempo total, memória ou custo de aquisição de rótulos. "
            "A tabela do apêndice mostra o consumo por braço; os pares abaixo preservam dataset, "
            "protocolo, modelo e semente, alterando apenas o teto.\n",
        ]
    )
    for model in MODELS:
        pairs = [row for row in summary["budget_pairs"] if row["model"] == model]
        n = sum(row["pairs"] for row in pairs)
        available = sum(row["available_digest_pairs"] for row in pairs)
        identical = sum(row["identical_prediction_pairs"] for row in pairs)
        text.append(
            f"**{NAMES[model]}:** {identical}/{available} pares com digest disponível mantiveram "
            f"predições idênticas ao elevar o teto de 2900 para 5800; {n - available} pares "
            f"sem comparação de digest, de {n} pares totais.\n"
        )
    text.extend(
        [
            "| Modelo | Dataset | Protocolo | Braço | Pares | Treino teto 2900 | Treino teto 5800 | "
            "Δ acurácia (p.p.) | Predições idênticas |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["budget_pairs"]:
        text.append(
            f"| {NAMES[row['model']]} | {row['dataset']} | {row['mode']} | {row['strategy']} | "
            f"{row['pairs']} | {stat_text(row['low_training_used'])} | "
            f"{stat_text(row['high_training_used'])} | "
            f"{stat_text(row['high_minus_low']['accuracy'], percent=True)} | "
            f"{row['identical_prediction_pairs']}/{row['available_digest_pairs']} |"
        )
    text.extend(
        [
            "\n## Comparação da logística com a campanha v4\n",
            "Resultados de reprodução são observados, não assumidos. Mudanças de hashes de código "
            "e parâmetros explícitos do modelo não substituem a conferência de métricas e predições.\n",
            "| Escopo | Casos | Métricas idênticas | Digests prequential idênticos | "
            "Cortes/janelas offline idênticos |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("all", "synthetic", "real"):
        row = bridge[name]
        text.append(
            f"| {name} | {row['cases']} | {row['metrics_exact_matches']} | "
            f"{row['prediction_digest_exact_matches']}/{row['prequential_cases']} | "
            f"{row['offline_split_and_window_exact_matches']}/{row['offline_cases']} |"
        )
    if bridge["differences"]:
        text.append(
            f"\nForam registradas {len(bridge['differences'])} diferenças por caso. "
            "Consultar `logistic_v4_comparison.differences` no JSON para valores anteriores/atuais "
            "e digests/cortes/janelas; não se declara reprodução exata desses casos.\n"
        )
    else:
        text.append("\nNão foram registradas diferenças nos campos conferidos da logística v5 contra v4.\n")
    text.extend(
        [
            "## Estados, arquivos e custo de persistência\n",
            "Os tamanhos abaixo são payloads medidos pelo auditor, quando presentes; não incluem "
            "todos os checkpoints, metadados, logs, cópias locais, réplicas ou overhead do sistema "
            "de arquivos. Diários por referência e objetos S3 únicos são contagens diferentes.\n",
            "| Escopo | Runs | Diários com medida | GiB de diários por referência | "
            "Objetos únicos por URI | GiB únicos de diários |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    storage = summary["storage"]
    for name, row in [("Todos", storage["all"]), *[(NAMES[m], storage["by_model"][m]) for m in MODELS]]:
        referenced_bytes = row["reference_bytes"]["model_states_bytes"]
        unique_bytes = row["unique_journal_bytes"]
        referenced_gib = f"{referenced_bytes / 2**30:.3f}" if referenced_bytes is not None else "n/d"
        unique_gib = f"{unique_bytes / 2**30:.3f}" if unique_bytes is not None else "n/d"
        text.append(
            f"| {name} | {row['runs']} | {row['measured_records'].get('model_states_bytes', 0)} | "
            f"{referenced_gib} | {row['unique_journal_objects']} | {unique_gib} |"
        )
    text.extend(
        [
            "\n## Limitações e interpretação\n",
            "- Cinco sementes sintéticas; um único prefixo real. As 867 configurações compartilham "
            "dados e não são 867 réplicas independentes. SD é descritivo; não há alegação de significância "
            "ou intervalos de confiança. Reexecutar sementes anteriores não cria amostras independentes.\n",
            "- Online é o braço comum principal. O ranking de políticas é selecionado depois dos resultados "
            "e não representa validação independente ou promessa de melhora em Agrawal/Airlines.\n",
            "- kNN retém no máximo 256 vizinhos, inclusive no offline. Seu scaler evolui e os vizinhos "
            "preservam a codificação de admissão, podendo misturar sistemas de coordenadas.\n",
            "- A árvore usa atributos originais e categorias nativas; elevel/car/zipcode do Agrawal e "
            "size/color/shape do STAGGER são nominais explícitos. Logística/kNN têm outro tratamento "
            "de features. Uma melhoria não isola somente capacidade não linear.\n",
            "- Um perfil de hiperparâmetros por família, sem busca ou validação independente de parâmetros "
            "ou limiar 0.5. Capacidade, memória, representação e retenção de histórico diferem.\n",
            "- O relógio e atraso de rótulo são simulados; os dados reais são prefixos de 3000 exemplos "
            "na ordem publicada. Não há pontos reais de drift anotados; Electricity já vem pré-processado "
            "com escopo de ajuste desconhecido.\n",
            "- Periódico/drift reiniciam modelo/preprocessador numa janela de até 256 rótulos; online "
            "continua aprendendo. Alarmes refletem o erro do modelo e não medem qualquer drift. "
            "Ausência de alarme não prova estabilidade.\n",
            "- Offline treina 1797 exemplos disponíveis e avalia 1200 finais; prequential avalia 2900 "
            "após warmup de 100. Os protocolos não são agregados num ranking único.\n",
            "- Precisão/revocação foram reconstruídas das métricas exatas e contagens dos rótulos para "
            "os 867 casos, conferindo contagens inteiras, acurácia e F1. Não são novas runs MLflow.\n",
            "- Compute/wall/RSS incluem efeitos do processo e de persistência/versionamento. Timers de "
            "avaliação não são CPU pura, nem duração completa da orquestração/publicação.\n",
            "## Proveniência e reprodução\n",
            f"SHA principal do manifesto congelado: `{summary['source_sha256']}`; "
            "as revisões autorizadas por run estão detalhadas acima. "
            "Configurações: [manifesto](../configs/campaign-nonlinear-v5.json). "
            "[Ambiente e imagens](nonlinear-v5-environment.json).\n",
        ]
    )
    if environment is not None:
        images = [
            (row.get("service", "serviço"), row.get("image_id", "n/d"))
            for row in environment.get("limits", [])
        ]
        if images:
            text.append(
                "Imagens registradas: " + "; ".join(f"`{name}` → `{image}`" for name, image in images) + ".\n"
            )
    text.extend(
        [
            "[Comparação completa](nonlinear-v5-comparison.json) · "
            "[Resumo e matrizes reconstruídas](nonlinear-v5-summary.json) · "
            "[Referências constantes](versioned-v4-reference-baselines.json) · "
            "[Figura PNG](nonlinear-v5-online-comparison.png) / "
            "[PDF](nonlinear-v5-online-comparison.pdf).\n",
            "Validação: [144 testes anteriores à campanha](nonlinear-v5-unit-tests.txt), "
            "[12 testes com serviços reais](nonlinear-v5-service-tests.txt), "
            "[156 testes da correção](nonlinear-v5-recovery-unit-tests.txt), "
            "[auditoria de publicação](nonlinear-v5-verification.log) e "
            "[saúde final autenticada](nonlinear-v5-final-health.json).\n",
            "`python scripts/compare_nonlinear_campaign.py` calcula os pares; "
            "`python scripts/render_nonlinear_campaign.py` gera este relatório somente com "
            "867 runs completas e o código/configurações iguais ao manifesto congelado. "
            "Nenhum modelo foi treinado, ajustado ou selecionado em validação independente pelo renderer.\n",
            "## Apêndice — todos os 255 grupos\n",
            "Sintéticos: média ± SD; reais: n=1. Treino é pós-warmup; zero no offline não significa "
            "ausência do ajuste inicial. Tempos em segundos, RSS em bytes.\n",
            "| Modelo | Dataset | Protocolo | Braço | n | Acurácia (%) | F1 (%) | Brier | Log loss | "
            "Treino | Alarmes | Atualizações | Compute s | Wall s | Pico RSS |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["groups"]:
        metrics = row["metrics"]
        text.append(
            f"| {NAMES[row['model']]} | {row['dataset']} | {row['mode']} | {row['strategy']} | "
            f"{metrics['accuracy']['n']} | {stat_text(metrics['accuracy'], percent=True)} | "
            f"{stat_text(metrics['f1'], percent=True)} | {stat_text(metrics['brier'])} | "
            f"{stat_text(metrics['log_loss'])} | {stat_text(row['training_used'])} | "
            f"{stat_text(row['alarms'])} | {stat_text(row['updates'])} | "
            f"{stat_text(row['compute_seconds'])} | {stat_text(row['wall_seconds'])} | "
            f"{stat_text(row['peak_rss_bytes'])} |"
        )
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="reports/nonlinear-v5-comparison.json")
    parser.add_argument("--current", default="reports/generated/nonlinear-v5-inputs.json")
    parser.add_argument("--references", default="reports/versioned-v4-reference-baselines.json")
    parser.add_argument("--manifest", default="configs/campaign-nonlinear-v5.json")
    parser.add_argument("--output-directory", default="reports")
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)) for name in ("comparison", "current", "references", "manifest")}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    inputs = {name: json.loads(contents) for name, contents in raw.items()}
    current_sha = hashlib.sha256(raw["current"]).hexdigest()
    checked = validate(
        inputs["comparison"],
        inputs["current"],
        inputs["references"],
        inputs["manifest"],
        current_sha256=current_sha,
        manifest_sha256=hashlib.sha256(raw["manifest"]).hexdigest(),
    )
    summary = summarize(
        inputs["comparison"], inputs["current"], inputs["references"], inputs["manifest"], checked
    )
    summary["input_sha256"] = {name: hashlib.sha256(contents).hexdigest() for name, contents in raw.items()}
    output = Path(args.output_directory)
    environment_path, airflow_path = (
        output / "nonlinear-v5-environment.json",
        output / "nonlinear-v5-airflow.json",
    )
    recovery_path = output / "nonlinear-v5-recovery.json"
    recovery_raw = recovery_path.read_bytes() if recovery_path.exists() else None
    recovery = json.loads(recovery_raw) if recovery_raw is not None else None
    summary["recovery_evidence"] = {
        "available": recovery_raw is not None,
        "path": recovery_path.name,
        "sha256": hashlib.sha256(recovery_raw).hexdigest() if recovery_raw is not None else None,
    }
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else None
    airflow = json.loads(airflow_path.read_text()) if airflow_path.exists() else None
    if environment is not None:
        assert environment["source_sha256"] in summary["source_revisions"], "environment source is unlisted"
    if airflow is not None:
        assert airflow["dag_id"] == "benchmark_nonlinear", "Airflow evidence is for another DAG"
    execution_path = output / "nonlinear-v5-execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    diagnostic_path = output / "nonlinear-v5-stagger-tree-diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    report = render_report(
        inputs["comparison"], summary, environment, airflow, recovery, execution, diagnostic
    )
    # All validation/derivation completes before any report or figure is written.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nonlinear-render-", dir=output) as temporary:
        stage = Path(temporary)
        plot(summary, stage / "nonlinear-v5-online-comparison")
        (stage / "nonlinear-v5.md").write_text(report)
        (stage / "nonlinear-v5-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for name in (
            "nonlinear-v5.md",
            "nonlinear-v5-summary.json",
            "nonlinear-v5-online-comparison.png",
            "nonlinear-v5-online-comparison.pdf",
        ):
            (stage / name).replace(output / name)
    print(
        json.dumps(
            {
                "runs": summary["runs"],
                "online_rows": len(summary["online"]),
                "confusion_matrices": len(summary["confusion_matrices"]),
                "report": str(output / "nonlinear-v5.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
