"""Experiment matrices expand, filter and execute datasets × models × strategies."""

import json

import pytest
from pydantic import ValidationError

from continual_bench.application.matrix import ExperimentMatrix, case_key, execute_matrix
from continual_bench.cli import main
from continual_bench.config import Config, EpochConfig, TypedConfig

SMALL = {"warmup": 30, "metric_window": 30, "replay_window": 60,
         "tracking": {"probe_samples": 4, "continual_references": False}}
STRATEGIES = [
    {"detector": {"name": "none"}, "policy": {"name": "none"}},
    {"detector": {"name": "adwin"}, "policy": {"name": "drift", "cooldown": 10, "min_samples": 30,
                                                 "min_window": 15}},
]


def binary_matrix(**changes):
    return ExperimentMatrix.model_validate({
        "name": "unit", "datasets": [{"name": "sea", "samples": 150}, {"name": "friedman_drift"}],
        "models": [{"name": "logistic"}, {"name": "linear_regression"}],
        "strategies": STRATEGIES, "modes": ["streaming", "offline"], "base": SMALL, **changes})


def test_binary_plan_skips_incompatible_pairs_and_deduplicates_offline():
    plan = binary_matrix().plan()
    assert all(isinstance(case, Config) for case in plan["cases"])
    # sea × logistic: 2 streaming strategies + one frozen offline holdout.
    assert sorted((c.mode, c.policy.name) for c in plan["cases"]) == [
        ("offline", "none"), ("streaming", "drift"), ("streaming", "none")]
    reasons = {(s["dataset"], s["model"]): s["reason"] for s in plan["skipped"]}
    assert "binary engine requires binary" in reasons[("friedman_drift", "logistic")]
    assert "does not support binary" in reasons[("sea", "linear_regression")]
    assert len({case_key(c) for c in plan["cases"]}) == len(plan["cases"])
    with pytest.raises(ValueError):
        binary_matrix(on_incompatible="error").plan()


def test_typed_and_epoch_plans_adapt_binary_streams_and_respect_model_tasks():
    typed = ExperimentMatrix.model_validate({
        "name": "typed", "evaluator": "typed", "typed_policies": ["online", "periodic"],
        "datasets": [{"name": "sea"}, {"name": "friedman_drift"}],
        "models": [{"name": "logistic"}, {"name": "knn"}]}).plan()
    assert all(isinstance(case, TypedConfig) for case in typed["cases"])
    pairs = {(c.dataset.name, c.model.name) for c in typed["cases"]}
    assert pairs == {("sea", "logistic"), ("sea", "knn"), ("friedman_drift", "knn")}
    assert typed["skipped"] == [{"dataset": "friedman_drift", "model": "logistic", "seed": 42,
                                 "reason": "model logistic does not support regression"}]
    epochs = ExperimentMatrix.model_validate({
        "name": "epochs", "evaluator": "epochs", "datasets": [{"name": "stagger", "samples": 40}],
        "models": [{"name": "hoeffding_tree"}], "seeds": [1, 2]}).plan()
    assert [type(c) for c in epochs["cases"]] == [EpochConfig, EpochConfig]
    assert [c.seed for c in epochs["cases"]] == [1, 2] and epochs["cases"][0].examples == 40


def test_matrix_size_is_bounded_before_expansion():
    with pytest.raises(ValidationError, match="limit"):
        ExperimentMatrix.model_validate({"name": "huge", "datasets": [{"name": "sea"}] * 100,
                                         "models": [{"name": "logistic"}] * 50, "seeds": list(range(2))})


def test_execute_matrix_journals_and_resumes(tmp_path):
    matrix = ExperimentMatrix.model_validate({
        "name": "run", "datasets": [{"name": "sea", "samples": 150}], "models": [{"name": "logistic"}],
        "strategies": STRATEGIES, "modes": ["streaming"], "base": SMALL})
    journal = execute_matrix(matrix, tmp_path)
    assert journal["complete"] and len(journal["cases"]) == 2
    assert {row["policy"] for row in journal["cases"].values()} == {"none", "drift"}
    summary = json.loads((tmp_path / matrix.identity() / "summary.json").read_text())
    assert all(0 <= row["metrics"]["accuracy"] <= 1 for row in summary)
    assert execute_matrix(matrix, tmp_path)["cases"] == journal["cases"]


def test_cli_lists_components_and_dry_runs_a_matrix(tmp_path, capsys):
    main(["components"])
    listing = json.loads(capsys.readouterr().out)
    assert {"name": "knn", "tasks": ["binary", "multiclass", "regression"]} in listing["models"]
    friedman = {"name": "friedman_drift", "task": "regression", "family": "river_regression"}
    assert friedman in listing["datasets"]
    path = tmp_path / "matrix.yaml"
    path.write_text("name: dry\nevaluator: typed\ndatasets: [{name: friedman_drift}]\n"
                    "models: [{name: knn}, {name: logistic}]\n")
    main(["matrix", str(path), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == 1 and plan["planned"][0]["model"] == "knn" and len(plan["skipped"]) == 1


@pytest.mark.parametrize("name", ["matrix-example.yaml", "matrix-any-model.yaml"])
def test_versioned_example_matrices_plan_without_errors(name):
    from pathlib import Path

    import yaml

    data = yaml.safe_load((Path(__file__).parents[1] / "configs" / name).read_text())
    plan = ExperimentMatrix.model_validate(data).plan()
    assert plan["cases"] and all(s["reason"] for s in plan["skipped"])
