"""Validate the local typed flow in the real stack's client environment."""

import json
import os
from pathlib import Path

import pytest

from continual_bench.application.readiness import ensure_tracking_ready
from continual_bench.datasets.catalog import load_dataset
from continual_bench.evaluation.typed import run_typed


@pytest.mark.services
@pytest.mark.parametrize("dataset", ["sea", "friedman_drift"])
@pytest.mark.parametrize("policy", ["none", "online", "periodic"])
@pytest.mark.parametrize("priced_time", [False, True])
def test_typed_costs_in_service_environment(dataset, policy, priced_time):
    ensure_tracking_ready()
    root = Path(os.environ["SERVICE_TEST_STATE"]) / "typed"
    source = load_dataset(dataset, os.environ["BENCH_DATASET_CACHE"], examples=100)
    kwargs = {"policy": policy, "warmup": 10, "every": 20}
    if priced_time:
        kwargs["costs"] = {"regression_scale": 10, "regression_error": 2, "classification_error": 3,
                           "train_sample": 0.02, "update": 0.3, "train_second": 2, "predict_second": 3}
    report = run_typed(source, root, **kwargs)
    costs = report["costs"]
    config = costs["settings"]
    if dataset == "sea":
        errors = sum(m["n"] * (1 - m["accuracy"]) for m in report["metrics"].values() if m["n"])
        expected = errors * config["classification_error"]
    else:
        errors = sum(m["n"] * m["mae"] for m in report["metrics"].values() if m["n"])
        expected = errors / config["regression_scale"] * config["regression_error"]
    assert costs["metrics"]["cost_prediction_errors"] == pytest.approx(expected)
    assert costs["metrics"]["cost_training_presentations"] == pytest.approx(
        (10 + report["training"]["postwarmup_samples"]) * config["train_sample"]
    )
    assert costs["metrics"]["cost_training_time"] == pytest.approx(
        report["compute"]["training_seconds"] * config["train_second"]
    )
    assert costs["metrics"]["cost_total"] == pytest.approx(sum(
        v for k, v in costs["metrics"].items() if k != "cost_total"
    ))
    assert run_typed(source, root, **kwargs) == report
    directory = root / report["identity"]
    assert json.loads((directory / "report.json").read_text())["costs"] == costs
    (directory / "typed-cost-evidence.json").write_text(json.dumps({
        "run_key": report["run_key"], "dataset": dataset, "policy": policy,
        "costs": costs, "publication": "local_report_only_no_mlflow_publisher",
        "replay_idempotent": True,
    }, indent=2))
