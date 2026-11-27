import math
import random

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("mlflow")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from continual_bench.infrastructure.mlflow_figures import (  # noqa: E402
    log_report_figures,
    save_report_figures,
)


def sample_report():
    rng = random.Random(7)
    windows = []
    for i in range(1, 21):
        tp, fp, tn, fn = rng.randint(30, 60), rng.randint(2, 12), rng.randint(60, 120), rng.randint(2, 15)
        n = tp + fp + tn + fn
        windows.append(
            {
                "end_seq": i * 100,
                "n": n,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "accuracy": (tp + tn) / n,
                "balanced_accuracy": 0.8,
                "f1": 0.7,
                "f1_macro": 0.75,
                "precision": 0.7,
                "recall": 0.7,
                "specificity": 0.9,
                "mcc": 0.5,
                "cohen_kappa": 0.45,
                "error_rate": (fp + fn) / n,
                "false_positive_rate": fp / (fp + tn),
                "false_negative_rate": fn / (fn + tp),
                "brier": 0.1 + rng.random() / 10,
                "log_loss": 0.4,
                "ece": 0.05,
                "mce": 0.2,
                "roc_auc": 0.85,
                "average_precision": 0.7,
            }
        )
    return {
        "windows": windows,
        "alarms": [{"seq": 700}, {"seq": 1500}],
        "dataset": {"changepoints": [650, 1400]},
        "candidates": [
            {
                "seq": 800,
                "action": "retrain",
                "train_samples": 400,
                "promoted": True,
                "candidate_accuracy": 0.82,
                "incumbent_accuracy": 0.7,
            },
            {
                "seq": 1600,
                "action": "retrain",
                "train_samples": 300,
                "promoted": False,
                "candidate_accuracy": 0.6,
                "incumbent_accuracy": 0.75,
            },
        ],
        "training": {"budget": 1000, "used": 700},
        "metrics": {"accuracy": 0.8, "f1": 0.7, "mcc": 0.5, "roc_auc": 0.85},
        "holdout": {"accuracy": 0.78, "f1": 0.66},
        "tracking": {
            "recovery": [{"change_seq": 650, "delay": 120}, {"change_seq": 1400, "delay": 90}],
            "costs": {"metrics": {"cost_fp": 12.0, "cost_fn": 39.5}, "settings": {"unit": "units"}},
            "cost_sensitivity": [{"alpha": a, "value": 10 + a} for a in (0.0, 0.1, 1.0)],
            "continual": {
                "matrix": [
                    {"task": 1, "accuracies": {"0": 0.9}},
                    {"task": 2, "accuracies": {"0": 0.8, "1": 0.85}},
                ]
            },
            "series": {"cl_forgetting": [{"step": 1, "value": 0.0}, {"step": 2, "value": 0.1}]},
        },
        "system_metrics": {
            "samples": [
                {
                    "elapsed_seconds": float(t),
                    "timestamp_ms": 1_700_000_000_000 + t * 500,
                    "values": {
                        "container_cpu_utilization_percentage": 40.0 + t,
                        "process_threads": 4.0,
                        "container_memory_usage_megabytes": 300.0 + t,
                    },
                }
                for t in range(6)
            ]
        },
    }


EXPECTED = {
    "plots/performance/dashboard.png",
    "plots/performance/confusion_matrix.png",
    "plots/performance/confusion_timeline.png",
    "plots/performance/error_tradeoff.png",
    "plots/probabilistic/dashboard.png",
    "plots/drift/timeline.png",
    "plots/drift/recovery_delay.png",
    "plots/adaptation/updates.png",
    "plots/continual/accuracy_matrix.png",
    "plots/continual/cl_metrics.png",
    "plots/costs/breakdown.png",
    "plots/summary/final_metrics.png",
    "plots/system/resources.png",
}


def test_figures_are_logged_with_mlflow_log_figure_to_the_resumed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # skinny build has no SQL backend
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    client = MlflowClient()
    experiment = client.create_experiment("figures")
    run = client.create_run(experiment)
    logged = log_report_figures(sample_report(), run_id=run.info.run_id)
    assert set(logged) == EXPECTED
    files = {
        a.path
        for folder in (
            "performance",
            "probabilistic",
            "drift",
            "adaptation",
            "continual",
            "costs",
            "summary",
            "system",
        )
        for a in client.list_artifacts(run.info.run_id, f"plots/{folder}")
    }
    assert {f for f in EXPECTED} <= files
    # The caller still owns termination: the run must be handed back as RUNNING, with no run left active.
    assert client.get_run(run.info.run_id).info.status == "RUNNING"
    assert mlflow.active_run() is None


def test_sparse_report_skips_missing_data_without_failing():
    calls = []
    logged = log_report_figures(
        {"windows": [], "metrics": {}}, log_figure=lambda fig, path: calls.append(path)
    )
    assert logged == calls == []


def test_save_report_figures_writes_png_files(tmp_path):
    saved = save_report_figures(sample_report(), tmp_path)
    assert set(saved) == EXPECTED
    assert all((tmp_path / path).stat().st_size > 1000 for path in saved)
    assert not math.isnan(len(saved))
