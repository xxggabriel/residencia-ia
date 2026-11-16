"""Figure builders: turn a finished report into segmented plots plus their data.

Each `Figure` carries chart-ready data; the writer renders it to SVG and saves
the same data as JSON, so every plot can be re-drawn or analysed elsewhere.
Builders only read reports: they never change metrics, digests or decisions.
"""

import math
from dataclasses import dataclass, field

from . import svg


@dataclass
class Figure:
    group: str
    name: str
    title: str
    kind: str  # "line", "bar" or "heatmap"
    data: dict = field(default_factory=dict)

    def render(self) -> str:
        options = {
            k: v
            for k, v in self.data.items()
            if k in {"x_label", "y_label", "markers", "y_range", "value_range"}
        }
        if self.kind == "line":
            return svg.line_chart(self.title, self.data["series"], **options)
        if self.kind == "bar":
            return svg.bar_chart(
                self.title, self.data["labels"], self.data["values"], x_label=self.data.get("x_label", "")
            )
        if self.kind == "heatmap":
            return svg.heatmap(
                self.title, self.data["rows"], self.data["columns"], self.data["matrix"], **options
            )
        raise ValueError(f"unknown figure kind {self.kind}")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _line(group, name, title, series, **data):
    return Figure(group, name, title, "line", {"series": series, **data})


def _bar(group, name, title, mapping, **data):
    labels = list(mapping)
    return Figure(
        group,
        name,
        title,
        "bar",
        {"labels": labels, "values": [mapping[k] if _finite(mapping[k]) else None for k in labels], **data},
    )


# ---- binary prequential / holdout -------------------------------------------------------------------

WINDOW_FIGURES = {
    "performance": {
        "accuracy": ("Acurácia por janela", ["accuracy", "balanced_accuracy"]),
        "f1": ("F1 por janela", ["f1", "f1_macro", "f1_weighted"]),
        "precision_recall": (
            "Precisão, recall e especificidade",
            ["precision", "recall", "specificity", "negative_predictive_value"],
        ),
        "error_rates": ("Taxas de erro", ["error_rate", "false_positive_rate", "false_negative_rate"]),
        "agreement": ("Concordância (MCC e kappa)", ["mcc", "cohen_kappa"]),
        "class_balance": ("Taxa de positivos real e predita", ["positive_rate", "predicted_positive_rate"]),
        "confusion": ("Matriz de confusão por janela", ["tp", "fp", "tn", "fn"]),
    },
    "probabilistic": {
        "losses": ("Brier e log loss por janela", ["brier", "log_loss"]),
        "calibration": ("Calibração (ECE e MCE)", ["ece", "mce"]),
        "ranking": ("Ranqueamento (ROC AUC e AP)", ["roc_auc", "average_precision"]),
    },
}
SUMMARY_FAMILIES = {
    "quality": (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "f1_weighted",
        "precision",
        "recall",
        "specificity",
        "mcc",
        "cohen_kappa",
        "roc_auc",
        "average_precision",
        "fading_accuracy",
        "mean_window_accuracy",
        "brier",
        "log_loss",
        "ece",
        "mce",
    ),
    "drift": "drift_",
    "recovery": "recovery_",
    "adaptation": (
        "adaptation_",
        "update_count",
        "retrain_count",
        "updates_per_1000_evaluated",
        "realized_adaptation_gain",
        "training_",
    ),
    "continual": ("cl_", "retention_"),
    "costs": ("cost_", "scenario_total_cost", "upf_", "posthoc_", "scpe"),
}


def _aws_figures(costs):
    aws = costs.get("aws") or {}
    if not aws:
        return []
    return [_bar("costs", "aws_usd", "Estimativa de infraestrutura AWS", aws["metrics"], x_label="USD")]


def _events(report):
    """Detector alarms, synthetic changepoints and promoted/rejected adaptations as x markers."""
    markers = [
        {"x": a["seq"], "kind": "alarm", "label": f"alarm seq={a['seq']}"} for a in report.get("alarms", [])
    ]
    for point in (report.get("dataset") or {}).get("changepoints") or []:
        markers.append({"x": point, "kind": "changepoint", "label": f"changepoint seq={point}"})
    for candidate in report.get("candidates", report.get("updates", [])):
        kind = "update" if candidate.get("promoted", True) else "rejected"
        markers.append(
            {
                "x": candidate["seq"],
                "kind": kind,
                "label": f"{candidate.get('action')} seq={candidate['seq']}",
            }
        )
    return markers


def _summary_family(metrics, rule):
    prefixes = (rule,) if isinstance(rule, str) else rule
    return {
        k: v
        for k, v in sorted(metrics.items())
        if any(k == p or (p.endswith("_") and k.startswith(p)) for p in prefixes)
    }


def binary_figures(report) -> list[Figure]:
    windows = report.get("windows", [])
    tracking = report.get("tracking") or {}
    figures = []
    accuracy_markers = _events(report)
    for group, entries in WINDOW_FIGURES.items():
        for name, (title, keys) in entries.items():
            series = [{"name": key, "points": [[w["end_seq"], w.get(key)] for w in windows]} for key in keys]
            markers = accuracy_markers if name == "accuracy" else []
            figures.append(
                _line(
                    group,
                    name,
                    title,
                    series,
                    x_label="seq (fim da janela)",
                    y_label="valor",
                    markers=markers,
                )
            )
    tracked = tracking.get("series", {})
    if "fading_accuracy" in tracked:
        figures.append(
            _line(
                "performance",
                "fading_accuracy",
                "Acurácia com esquecimento (fading)",
                [
                    {
                        "name": "fading_accuracy",
                        "points": [[p["step"], p["value"]] for p in tracked["fading_accuracy"]],
                    }
                ],
                x_label="seq",
                y_label="acurácia",
                markers=accuracy_markers,
            )
        )
    figures.append(
        _line(
            "drift",
            "timeline",
            "Alarmes, mudanças de conceito e adaptações",
            [
                {"name": "accuracy", "points": [[w["end_seq"], w.get("accuracy")] for w in windows]},
                {"name": "error_rate", "points": [[w["end_seq"], w.get("error_rate")] for w in windows]},
            ],
            x_label="seq",
            y_label="valor",
            markers=accuracy_markers,
            y_range=(0.0, 1.0),
        )
    )
    metrics = tracking.get("metrics", {})
    for family, rule in SUMMARY_FAMILIES.items():
        selected = _summary_family(metrics, rule)
        if selected:
            figures.append(_bar("summary", family, f"Resumo: {family}", selected, x_label="valor final"))
    recovery = tracking.get("recovery") or []
    if recovery:
        figures.append(
            _bar(
                "drift",
                "recovery_delay",
                "Atraso de recuperação por mudança (amostras)",
                {f"change@{r['change_seq']}": r.get("delay") for r in recovery},
                x_label="amostras",
            )
        )
    candidates = report.get("candidates", report.get("updates", []))
    cumulative, promoted_total = [], 0
    for candidate in candidates:
        promoted_total += candidate["train_samples"]
        cumulative.append([candidate["seq"], promoted_total])
    figures.append(
        _line(
            "adaptation",
            "training_presentations",
            "Apresentações de treino acumuladas pós-warmup",
            [{"name": "train_samples", "points": cumulative}],
            x_label="seq",
            y_label="apresentações",
            markers=accuracy_markers,
        )
    )
    gated = [c for c in candidates if "candidate_accuracy" in c]
    if gated:
        figures.append(
            _line(
                "adaptation",
                "promotion_holdout",
                "Holdout de promoção: candidato x incumbente",
                [
                    {"name": "candidate", "points": [[c["seq"], c["candidate_accuracy"]] for c in gated]},
                    {"name": "incumbent", "points": [[c["seq"], c["incumbent_accuracy"]] for c in gated]},
                ],
                x_label="seq",
                y_label="acurácia",
            )
        )
    training = report.get("training") or {}
    figures.append(
        _bar(
            "adaptation",
            "budget",
            "Orçamento de treino",
            {
                "budget": training.get("budget"),
                "used": training.get("used"),
                "blocked_decisions": training.get("blocked_decisions"),
                "updates": len(report.get("updates", [])),
                "alarms": len(report.get("alarms", [])),
            },
        )
    )
    figures.extend(_continual_figures(tracking))
    near = (report.get("task_evaluation") or {}).get("near_future") or {}
    if near.get("windows"):
        figures.append(
            _bar(
                "near_future",
                "windows",
                "Acurácia no futuro próximo por janela (modelo congelado)",
                {f"start@{w.get('start_seq')}": w.get("near_future_accuracy") for w in near["windows"]},
                x_label="acurácia",
            )
        )
    costs = (tracking.get("costs") or {}).get("metrics") or {}
    if costs:
        figures.append(
            _bar(
                "costs",
                "components",
                "Componentes do custo de cenário",
                costs,
                x_label=(tracking.get("costs") or {}).get("settings", {}).get("unit", ""),
            )
        )
    figures.extend(_aws_figures(tracking.get("costs") or {}))
    if tracking.get("cost_sensitivity"):
        figures.append(
            _line(
                "costs",
                "sensitivity",
                "Sensibilidade do custo a alpha",
                [
                    {
                        "name": "cost",
                        "points": [[p["alpha"], p["value"]] for p in tracking["cost_sensitivity"]],
                    }
                ],
                x_label="alpha",
                y_label="custo",
            )
        )
    return figures


def _continual_figures(tracking):
    figures = []
    series = tracking.get("series", {})
    cl = [name for name in sorted(series) if name.startswith("cl_")]
    if cl:
        figures.append(
            _line(
                "continual",
                "cl_metrics",
                "Métricas de aprendizado contínuo por tarefa",
                [{"name": n, "points": [[p["step"], p["value"]] for p in series[n]]} for n in cl],
                x_label="tarefa",
                y_label="valor",
            )
        )
    for prefix, title in (
        ("retention_accuracy/", "Acurácia de retenção por conceito"),
        ("retention_drop/", "Queda de retenção por conceito"),
    ):
        names = [n for n in sorted(series) if n.startswith(prefix)]
        if names:
            figures.append(
                _line(
                    "continual",
                    prefix.rstrip("/"),
                    title,
                    [
                        {"name": n.split("/", 1)[1], "points": [[p["step"], p["value"]] for p in series[n]]}
                        for n in names
                    ],
                    x_label="seq",
                    y_label="valor",
                )
            )
    matrix = (tracking.get("continual") or {}).get("matrix") or []
    if matrix:
        episodes = sorted({int(e) for row in matrix for e in row.get("accuracies", {})})
        figures.append(
            Figure(
                "continual",
                "accuracy_matrix",
                "Matriz de acurácia (linha: após tarefa)",
                "heatmap",
                {
                    "rows": [f"t{row['task']}" for row in matrix],
                    "columns": [f"e{e}" for e in episodes],
                    "matrix": [[row["accuracies"].get(str(e)) for e in episodes] for row in matrix],
                    "x_label": "episódio avaliado",
                    "y_label": "após tarefa",
                },
            )
        )
    return figures


# ---- typed prequential ------------------------------------------------------------------------------


def _rolling(rows, size, score):
    points, window = [], []
    for seq, value in rows:
        window.append(value)
        if len(window) > size:
            window.pop(0)
        points.append([seq, score(window)])
    return points


def typed_figures(report, timeline) -> list[Figure]:
    """timeline: [(seq, split, target, prediction)] in arrival order."""
    task = report["config"]["task"]
    figures = []
    per_split = {
        f"{split}/{name}": value
        for split, metrics in report["metrics"].items()
        for name, value in metrics.items()
        if name != "n"
    }
    figures.append(_bar("performance", "metrics_by_split", "Métricas por partição", per_split))
    per_task = {
        key: next((v for k, v in entry.items() if k != "n"), None)
        for key, entry in report["per_task"].items()
    }
    if per_task:
        figures.append(
            _bar("performance", "per_task", "Métrica principal por tarefa (prequential)", per_task)
        )
    size = max(10, len(timeline) // 20)
    for split in sorted({row[1] for row in timeline}):
        rows = [
            (seq, (abs(y - p) if task == "regression" else float(y != p)))
            for seq, s, y, p in timeline
            if s == split
        ]
        figures.append(
            _line(
                "performance",
                f"rolling_{split}",
                f"{'MAE' if task == 'regression' else 'Erro'} móvel ({size} amostras) — {split}",
                [
                    {
                        "name": "mae" if task == "regression" else "error_rate",
                        "points": _rolling(rows, size, lambda w: sum(w) / len(w)),
                    }
                ],
                x_label="seq",
                y_label="valor",
                markers=[{"x": u["seq"], "kind": "update", "label": u["action"]} for u in report["updates"]][
                    :500
                ],
            )
        )
    cumulative, total = [], report["training"]["warmup_samples"]
    for update in report["updates"]:
        total += update["train_samples"]
        cumulative.append([update["seq"], total])
    figures.append(
        _line(
            "adaptation",
            "training_presentations",
            "Apresentações de treino acumuladas",
            [{"name": "train_samples", "points": cumulative}],
            x_label="seq",
            y_label="apresentações",
        )
    )
    figures.append(_bar("costs", "components", "Componentes do custo de cenário", report["costs"]["metrics"]))
    figures.extend(_aws_figures(report["costs"]))
    return figures


# ---- repeated epochs --------------------------------------------------------------------------------


def epoch_figures(report) -> list[Figure]:
    epochs = report["epochs"]
    names = sorted({name for epoch in epochs for name in epoch["metrics"]})
    figures = [
        _line(
            "performance",
            "metrics_by_epoch",
            "Métricas no holdout por época",
            [{"name": n, "points": [[e["epoch"], e["metrics"].get(n)] for e in epochs]} for n in names],
            x_label="época",
            y_label="valor",
        )
    ]
    costs = sorted({name for epoch in epochs for name in epoch["costs"]["metrics"]})
    figures.append(
        _line(
            "costs",
            "by_epoch",
            "Custo de cenário por época",
            [
                {"name": n, "points": [[e["epoch"], e["costs"]["metrics"].get(n)] for e in epochs]}
                for n in costs
            ],
            x_label="época",
            y_label="custo",
        )
    )
    figures.append(_bar("costs", "total", "Custo de cenário total", report["costs"]["metrics"]))
    figures.extend(_aws_figures(report["costs"]))
    figures.append(
        _line(
            "compute",
            "seconds_by_epoch",
            "Tempo de treino e predição por época",
            [
                {"name": key, "points": [[e["epoch"], e[key]] for e in epochs]}
                for key in ("training_seconds", "prediction_seconds")
            ],
            x_label="época",
            y_label="s",
        )
    )
    return figures


# ---- system metrics ---------------------------------------------------------------------------------

SYSTEM_FIGURES = {
    "cpu": ("CPU (%)", ("cpu_utilization_percentage", "process_cpu_utilization_percentage"), "%"),
    "container_cpu_percentage": (
        "CPU do container (% da cota)", ("container_cpu_utilization_percentage",), "%",
    ),
    "container_cpu_vcpus": (
        "CPU do container (vCPU)", ("container_cpu_usage_vcpus", "container_cpu_limit_vcpus"), "vCPU",
    ),
    "container_memory_percentage": (
        "RAM do container (% do limite)", ("container_memory_usage_percentage",), "%",
    ),
    "container_memory_megabytes": (
        "RAM do container (MiB)",
        ("container_memory_usage_megabytes", "container_memory_limit_megabytes"),
        "MiB",
    ),
    "memory_percentage": (
        "Memória e disco (%)",
        ("system_memory_usage_percentage", "disk_usage_percentage"),
        "%",
    ),
    "memory_megabytes": (
        "Memória (MB)",
        ("system_memory_usage_megabytes", "process_memory_rss_megabytes"),
        "MB",
    ),
    "disk_megabytes": ("Disco (MB)", ("disk_usage_megabytes", "disk_available_megabytes"), "MB"),
    "network": ("Rede acumulada (MB)", ("network_receive_megabytes", "network_transmit_megabytes"), "MB"),
    "threads": ("Threads do processo", ("process_threads",), "threads"),
}


def system_figures(system) -> list[Figure]:
    if not system or not system.get("samples"):
        return []
    samples = system["samples"]
    figures = []

    def series(names):
        return [
            {"name": n, "points": [[round(s["elapsed_seconds"], 3), s["values"].get(n)] for s in samples]}
            for n in names
            if any(n in s["values"] for s in samples)
        ]

    for name, (title, keys, unit) in SYSTEM_FIGURES.items():
        selected = series(keys)
        if selected:
            figures.append(
                _line("system", name, title, selected, x_label="segundos desde o início", y_label=unit)
            )
    gpu = sorted({n for s in samples for n in s["values"] if n.startswith("gpu_")})
    if gpu:
        figures.append(
            _line("system", "gpu", "GPU", series(gpu), x_label="segundos desde o início", y_label="valor")
        )
    return figures


# ---- cost-aware retraining (CARA, UPF, Oracle) -------------------------------------------------------


def retraining_figures(analysis) -> list[Figure]:
    """Paper-style plots of the cost/performance trade-off; dependency-free SVG companions of the PNGs."""
    rules, alphas, setup = analysis["rules"], analysis["alphas"], analysis["setup"]

    def curve(key):
        return [{"name": rule["label"], "points": [[c["alpha"], c[key]] for c in rule["curve"]]}
                for rule in rules.values()]

    figures = [
        _line("retraining", "cost_vs_alpha", "Custo Ĉα(θ) × α", curve("cost"),
              x_label="α (custo de retreino relativo)", y_label="Ĉα(θ)"),
        _line("retraining", "retrains_vs_alpha", "Número de retreinos × α", curve("retrains"),
              x_label="α", y_label="retreinos"),
        _line("retraining", "accuracy_vs_alpha", "Acurácia média online × α", curve("average_accuracy"),
              x_label="α", y_label="acurácia média"),
        _line("retraining", "scpe_vs_alpha", "SCPE (% sobre o Oracle) × α", curve("scpe"),
              x_label="α", y_label="SCPE %"),
        _line("retraining", "strategy_cost_vs_alpha", "Custo da estratégia (CARA) × κ=α",
              curve("strategy_cost"), x_label="κ = α", y_label="custo da estratégia"),
        _line("retraining", "retrain_bound", "Cotas de retreino (Proposição 3.1)",
              [s for s in curve("cost") if s["name"] in {"UPF", "oracle"}],
              x_label="α", y_label="Ĉα(θ)",
              markers=[{"x": a, "kind": "bound", "label": label}
                       for label, a in setup["bound_alphas"].items()
                       if a is not None and a <= max(alphas) * 1.5]),
        _line("retraining", "cumulative_strategy_cost", "Custo acumulado da estratégia online",
              [{"name": rules[name]["label"],
                "points": [[setup["offline_batches"] + k, v] for k, v in enumerate(series)]}
               for name, series in analysis["cumulative_strategy_cost"].items()],
              x_label="lote online", y_label="custo acumulado"),
    ]
    auc = {rule["label"]: rule["auc"] for rule in rules.values()}
    figures.append(_bar("retraining", "auc_by_policy", "AUC de Ĉα(θ) (Tabela 1)", auc))
    scpe = {rule["label"]: rule["mean_scpe"] for rule in rules.values() if rule["label"] != "oracle"}
    figures.append(_bar("retraining", "mean_scpe_by_policy", "SCPE médio sobre a grade de α", scpe))
    matrix = analysis["performance_matrix"]["err"]
    accuracy = [[None if v is None else 1 - v for v in row] for row in matrix]
    labels = [str(i) for i in range(len(matrix))]
    figures.append(Figure("retraining", "accuracy_matrix", "Acurácia do modelo treinado no lote i no lote j",
                          "heatmap", {"rows": labels, "columns": labels, "matrix": accuracy,
                                      "x_label": "lote avaliado j", "y_label": "lote de treino i",
                                      "value_range": (0, 1)}))
    stale = analysis["staleness_matrix"]
    values = [v for row in stale for v in row if v is not None]
    if values:
        figures.append(Figure("retraining", "staleness_matrix", "Obsolescência Ψ̄ (CARA)", "heatmap", {
            "rows": labels, "columns": labels,
            "matrix": [list(column) for column in zip(*stale)],
            "x_label": "lote avaliado t", "y_label": "lote de treino t'",
            "value_range": (min(values), max(values))}))
    for name, grid in analysis["wrong_alpha"].items():
        figures.append(Figure("retraining", f"wrong_alpha_{name}", f"Aumento % de Ĉ com α errado ({name})",
                              "heatmap", {"rows": [f"{a:.3g}" for a in grid["provided"]],
                                          "columns": [f"{a:.3g}" for a in grid["real"]],
                                          "matrix": grid["pct_increase"], "x_label": "α real",
                                          "y_label": "α informado",
                                          "value_range": (0, max((v for r in grid["pct_increase"] for v in r
                                                                  if v is not None), default=1) or 1)}))
    return figures
