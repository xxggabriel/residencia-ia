"""Reproducible descriptive comparison of the 289 verified v4 runs; no pooled protocols."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("accuracy", "f1", "brier", "log_loss")
DATASETS = ("sea", "agrawal", "stagger", "electricity", "airlines")
MODES = ("streaming", "incremental")


def arm(report):
    cfg = report["config"]
    name = cfg["detector"]["name"] if cfg["policy"]["name"] == "drift" else cfg["policy"]["name"]
    return f"{name}@{cfg['training_budget']}" if name in ("periodic", "adwin", "ddm") else name


def describe(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def analyze(payload, manifest):
    assert payload["complete_campaign"] and payload["runs_verified"] == 289
    expected = {row["run_key"]: row["config"] for row in manifest["configurations"]}
    reports = [row["report"] for row in payload["records"]]
    assert len(reports) == len(expected) == 289
    assert {r["run_key"] for r in reports} == set(expected)
    groups = defaultdict(list)
    index = {}
    for record in payload["records"]:
        r = record["report"]
        assert record["publication_verified"] and r["complete"]
        cfg = r["config"]
        assert cfg == expected[r["run_key"]]
        assert r["environment"]["code_sha256"] == manifest["source_sha256"]
        assert r["metrics"]["n"] == (1200 if cfg["mode"] == "offline" else 2900)
        key = (cfg["dataset"]["name"], cfg["mode"], arm(r))
        groups[key].append(r)
        index[(*key, cfg["dataset"]["seed"])] = r
    aggregates = []
    for (dataset, mode, strategy), rows in sorted(groups.items()):
        n = 1 if dataset in ("electricity", "airlines") else 5
        assert len(rows) == n
        aggregates.append(
            {
                "dataset": dataset,
                "mode": mode,
                "strategy": strategy,
                "seeds": [r["config"]["dataset"]["seed"] for r in rows],
                "metrics": {m: describe([r["metrics"][m] for r in rows]) for m in METRICS},
                "training_used": describe([sum(u["train_samples"] for u in r["updates"]) for r in rows]),
                "updates": describe([len(r["updates"]) for r in rows]),
                "alarms": describe([len(r["alarms"]) for r in rows]),
                "alarms_before_first_change": None
                if dataset in ("electricity", "airlines")
                else sum(a["seq"] < 1000 for r in rows for a in r["alarms"]),
                "run_keys": [r["run_key"] for r in rows],
            }
        )
    assert len(aggregates) == 85
    paired = []
    budgets = []
    detectors = []
    for dataset in DATASETS:
        seeds = [42] if dataset in ("electricity", "airlines") else list(range(42, 47))
        for mode in MODES:
            for comparator in (
                "none",
                "periodic@2900",
                "adwin@2900",
                "ddm@2900",
                "periodic@5800",
                "adwin@5800",
                "ddm@5800",
            ):
                row = {"dataset": dataset, "mode": mode, "comparison": "online - " + comparator}
                for metric in METRICS:
                    diffs = [
                        index[(dataset, mode, "online", s)]["metrics"][metric]
                        - index[(dataset, mode, comparator, s)]["metrics"][metric]
                        for s in seeds
                    ]
                    sign = 1 if metric in ("accuracy", "f1") else -1
                    row[metric] = {
                        **describe(diffs),
                        "online_wins": sum(sign * d > 0 for d in diffs),
                        "ties": sum(d == 0 for d in diffs),
                    }
                paired.append(row)
            for strategy in ("periodic", "adwin", "ddm"):
                low = [index[(dataset, mode, strategy + "@2900", s)] for s in seeds]
                high = [index[(dataset, mode, strategy + "@5800", s)] for s in seeds]
                budgets.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "strategy": strategy,
                        "pairs": len(seeds),
                        "identical_prediction_pairs": sum(
                            a["prediction_digest"] == b["prediction_digest"] for a, b in zip(low, high)
                        ),
                        "high_minus_low": {
                            m: describe([b["metrics"][m] - a["metrics"][m] for a, b in zip(low, high)])
                            for m in METRICS
                        },
                    }
                )
            # Descriptive alarm coverage after known synthetic changes. Not detector recall.
            if dataset in ("electricity", "airlines"):
                continue
            for strategy in ("adwin", "ddm"):
                delays = []
                hits = 0
                pre = 0
                for seed in seeds:
                    r = index[(dataset, mode, strategy + "@2900", seed)]
                    pre += sum(a["seq"] < 1000 for a in r["alarms"])
                    for point in (1000, 2000):
                        candidates = [
                            a["seq"] - point for a in r["alarms"] if point <= a["seq"] < point + 500
                        ]
                        if candidates:
                            hits += 1
                            delays.append(min(candidates))
                detectors.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "strategy": strategy,
                        "post_change_windows": 2 * len(seeds),
                        "windows_with_alarm": hits,
                        "first_alarm_delay_in_hit_windows": describe(delays) if delays else None,
                        "alarms_before_first_change": pre,
                        "interpretation": "Post-hoc 500-example window coverage; not precision/recall",
                    }
                )
    return {
        "campaign": "versioned-v4",
        "runs": 289,
        "groups": aggregates,
        "paired_online_comparisons": paired,
        "budget_comparisons": budgets,
        "synthetic_alarm_windows": detectors,
        "uncertainty": "Sample SD across five synthetic seeds; real prefixes n=1. No significance claim.",
    }


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(r["dataset"], r["mode"], r["strategy"]): r for r in result["groups"]}
    strategies = ("online", "periodic@2900", "adwin@2900", "ddm@2900")
    contexts = [(d, mode) for d in DATASETS for mode in MODES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "f1"), ("Ganho de acurácia", "Ganho de F1")):
        values = np.array(
            [
                [
                    100
                    * (
                        lookup[(d, mode, s)]["metrics"][metric]["mean"]
                        - lookup[(d, mode, "none")]["metrics"][metric]["mean"]
                    )
                    for s in strategies
                ]
                for d, mode in contexts
            ]
        )
        limit = float(np.max(np.abs(values)))
        shown = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(4), ["Online", "Periódico", "ADWIN", "DDM"])
        ax.set_yticks(
            range(10), [d.title() + (" · evento" if m == "streaming" else " · lote") for d, m in contexts]
        )
        ax.set_title(title + " vs. congelado (p.p.)", fontsize=12)
        for y in range(10):
            for x in range(4):
                value = values[y, x]
                ax.text(
                    x,
                    y,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > limit * 0.6 else "#17283b",
                    fontsize=10,
                )
        fig.colorbar(shown, ax=ax, shrink=0.7, label="pontos percentuais")
    fig.suptitle("Campanha v4 · teto pós-warmup de 2.900 apresentações", fontsize=15)
    fig.supxlabel(
        "Sintéticos: médias de 5 sementes. Electricity/Airlines: um prefixo fixo de 3.000 exemplos.",
        fontsize=10,
    )
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/generated/versioned-v4-inputs.json")
    parser.add_argument("--output", default="reports/versioned-v4-analysis.json")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    raw = Path(args.input).read_bytes()
    payload = json.loads(raw)
    result = analyze(payload, json.loads(Path("configs/campaign-versioned-v4.json").read_text()))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    reference_path = Path("reports/versioned-v4-reference-baselines.json")
    reference_raw = reference_path.read_bytes()
    references = json.loads(reference_raw)
    result["reference_evidence_sha256"] = hashlib.sha256(reference_raw).hexdigest()
    result["reference_scope"] = references["scope"]
    reference_groups = defaultdict(list)
    for row in references["references"]:
        reference_groups[(row["dataset"], row["mode"])].append(row)
    result["label_references"] = [
        {
            "dataset": dataset,
            "mode": mode,
            "positive_rate": describe([r["evaluation_positive_rate"] for r in rows]),
            **{
                name: {metric: describe([r[name][metric] for r in rows]) for metric in rows[0][name]}
                for name in ("always_positive", "always_negative")
            },
        }
        for (dataset, mode), rows in sorted(reference_groups.items())
    ]
    reference_index = {(r["dataset"], r["seed"], r["mode"]): r for r in references["references"]}
    confusion_groups = defaultdict(list)
    confusion = []
    for record in payload["records"]:
        r = record["report"]
        cfg = r["config"]
        mode = "offline" if cfg["mode"] == "offline" else "prequential"
        reference = reference_index[(cfg["dataset"]["name"], cfg["dataset"]["seed"], mode)]
        assert reference["dataset_sha256"] == r["dataset"]["sha256"]
        n, positive = reference["evaluation_n"], reference["evaluation_positive"]
        assert r["metrics"]["n"] == n
        correct = round(r["metrics"]["accuracy"] * n)
        assert math.isclose(correct, r["metrics"]["accuracy"] * n, abs_tol=1e-8)
        errors, f1 = n - correct, r["metrics"]["f1"]
        tp_float = f1 * errors / (2 * (1 - f1)) if f1 < 1 else positive
        tp = round(tp_float)
        assert math.isclose(tp, tp_float, abs_tol=1e-8)
        fn, fp = positive - tp, errors - (positive - tp)
        tn = n - tp - fn - fp
        assert min(tp, fn, fp, tn) >= 0 and tp + tn == correct
        assert math.isclose(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, f1)
        row = {
            "run_key": r["run_key"],
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / positive if positive else 0,
        }
        confusion.append(row)
        confusion_groups[(cfg["dataset"]["name"], cfg["mode"], arm(r))].append(row)
    result["derived_confusion_matrices"] = confusion
    for group in result["groups"]:
        rows = confusion_groups[(group["dataset"], group["mode"], group["strategy"])]
        group["derived_precision"] = describe([r["precision"] for r in rows])
        group["derived_recall"] = describe([r["recall"] for r in rows])
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        plot(result, Path("reports/versioned-v4-policy-comparison"))
    print(json.dumps({"runs": result["runs"], "groups": len(result["groups"])}))


if __name__ == "__main__":
    main()
