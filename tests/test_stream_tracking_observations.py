from pathlib import Path

import pytest
from sklearn.metrics import average_precision_score, f1_score, r2_score, root_mean_squared_error

from continual_bench.application.runner import evaluate, prepare
from continual_bench.config import Config
from continual_bench.core.contracts import TaskSample
from continual_bench.datasets.arrays import ListDataset
from continual_bench.datasets.synthetic import FriedmanDriftSource, SyntheticSource
from continual_bench.evaluation.iterative import run_iterative
from continual_bench.evaluation.iterative_scores import ClassificationScorer, RegressionScorer
from continual_bench.evaluation.typed import run_typed
from continual_bench.infrastructure.training_tracking import training_checkpoint_paths


def test_iterative_binary_ranking_is_explicit_and_macro_uses_the_complete_class_universe():
    observations = [("0", {"0": .5, "1": .5}), ("1", {"1": .5, "0": .5})]
    scorer = ClassificationScorer()
    for label, distribution in observations:
        scorer.update(label, distribution)
    assert scorer.result()["pr_auc"] == .5
    assert scorer.result()["f1_macro"] == 1
    canonical = scorer.quality(("0", "1", "2"))
    assert canonical["average_precision"] is None
    reference = f1_score([0, 1], [0, 1], labels=[0, 1, 2], average="macro", zero_division=0)
    assert canonical["macro_f1"] == pytest.approx(reference)
    assert canonical["classes"] == ["0", "1", "2", "__untrained__"]
    assert canonical["class_support"] == [1, 1, 0, 0]
    assert canonical["class_recall"] == [1., 1., None, None]

    scorer = ClassificationScorer(observed_positive_class="1")
    scorer.update("1", {})  # Missing score remains a classification error outside the ranking denominator.
    for label, distribution in observations:
        scorer.update(label, distribution)
    binary = scorer.quality(("0", "1"), curves=True)
    assert binary["evaluated_examples"] == 3 and binary["ranking_evaluated_examples"] == 2
    assert binary["average_precision"] == pytest.approx(average_precision_score([0, 1], [.5, .5]))
    assert binary["roc_curve"]["tpr"] == [0, 1]


def test_iterative_r2_handles_high_offset_targets_and_preserves_degenerate_convention():
    actual = [1e9, 1e9 + 1, 1e9 + 2]
    predicted = [1e9, 1e9 + 1, 1e9 + 1]
    old_variance = sum(value**2 for value in actual) / 3 - (sum(actual) / 3)**2
    assert old_variance == 0
    scorer = RegressionScorer(retain_residuals=True)
    for y, p in zip(actual, predicted, strict=True):
        scorer.update(y, p)
    assert scorer.result()["r2"] == pytest.approx(r2_score(actual, predicted)) == .5
    assert scorer.result()["rmse"] == pytest.approx(root_mean_squared_error(actual, predicted))
    assert scorer.quality("unit")["residuals"]["error"] == [0, 0, 1]

    constant = RegressionScorer()
    constant.update(1, 1)
    constant.update(1, 1)
    assert constant.result()["r2"] == 0


def test_binary_delayed_resume_has_distinct_rejected_checkpoints_and_released_scope(tmp_path):
    config = Config.model_validate({
        "mode": "streaming", "dataset": {"name": "sea", "samples": 100, "label_delay": 4},
        "warmup": 10, "replay_window": 10, "metric_window": 8, "checkpoint_every": 7,
        "policy": {"name": "periodic", "every": 5, "cooldown": 0, "min_window": 2, "min_samples": 10},
        "training_budget": 60, "promotion": {"validation_samples": 2, "min_improvement": 1},
        "tracking": {"probe_samples": 3, "continual_references": False},
    })
    manifest = prepare(config, tmp_path)
    assert evaluate(config, manifest, tmp_path, stop_after=33)["complete"] is False
    report = evaluate(config, manifest, tmp_path)
    assert evaluate(config, manifest, tmp_path) == report
    binary = report["tracking_observations"]
    assert binary["scopes"]["prequential"]["evaluated_examples"] == 90
    assert binary["effort"]["updates"] == 0
    assert binary["effort"]["rejected_fits"] >= 2
    assert binary["effort"]["epochs"] is None
    assert binary["resources"]["inference_examples"] == 100
    fits = report["training_history"]
    initial = fits[0]
    assert len(initial["observation_steps"]) == 10
    assert len(initial["steps"]) == 1
    assert initial["observation_steps"][-1]["metrics"]["training/presentations_cumulative"] == 10
    assert all(fit["promotion_status"] == "rejected" for fit in fits[1:])
    assert all(fit["deployed_version"] is None for fit in fits[1:])
    assert all(fit["observation_scopes"]["promotion_validation/candidate"]["evaluated_examples"] == 2
               for fit in fits[1:])
    assert len(training_checkpoint_paths(tmp_path / config.identity(), report)) == len(fits)
    assert binary["windows"][-1]["quality"]["evaluated_examples"] == 2
    assert all(window["model_version"] == report["model_versions"]["final_version_id"]
               for window in binary["windows"] if window["start_seq"] >= 20)


def test_typed_reuses_river_scoring_and_keeps_empty_partitions_distinct(tmp_path):
    regression = run_typed(FriedmanDriftSource(40, 42, label_delay=3), tmp_path / "regression", warmup=5)
    collected = regression["tracking_observations"]
    quality = collected["scopes"]["prequential"]
    assert quality["rmse"] == pytest.approx(regression["metrics"]["train"]["rmse"])
    assert len(quality["residuals"]["error"]) == 35
    assert collected["scopes"]["validation"]["rmse"] is None
    assert "rmse" in collected["scopes"]["validation"]["availability"]
    assert collected["effort"]["epochs"] is None
    online = regression["training_history"][1]
    assert online["observation_steps"][-1]["metrics"]["training/presentations_cumulative"] == 35
    assert len(online["observation_steps"]) == 35
    assert len(online["observation_scopes"]["prequential_before_learning"]["residuals"]["error"]) == 35

    binary = run_typed(SyntheticSource("sea", 40, 42, 3, 1000), tmp_path / "binary", warmup=5)
    collected = binary["tracking_observations"]
    assert collected["task"] == "binary"
    quality = collected["scopes"]["prequential"]
    assert quality["positive_class"] == "1"
    assert quality["ranking_evaluated_examples"] > 0
    assert quality["average_precision"] is not None


def test_typed_prefix_preserves_declared_class_order_with_sparse_confusion_windows(tmp_path):
    class Prefix:
        def manifest(self):
            return {"name": "prefix", "version": "v1", "task": "multiclass",
                    "class_order": list(range(100)), "labels": ["0"]}

        def samples_iter(self):
            for seq in range(12):
                yield TaskSample(str(seq), seq, seq, seq, {"x": float(seq)}, "0")

    report = run_typed(Prefix(), tmp_path, policy="none", warmup=2, window=1)
    observed = report["tracking_observations"]
    complete = observed["scopes"]["prequential"]
    assert complete["classes"] == [str(label) for label in range(100)] + ["__untrained__"]
    assert complete["macro_f1"] == pytest.approx(.01)
    assert complete["class_support"] == [10] + [0] * 100
    assert len(complete["confusion_matrix"]) == 101
    for window in observed["windows"]:
        quality = window["quality"]
        assert "confusion_matrix" not in quality
        assert quality["confusion_counts_sparse"] == [[0, 0, 1]]
        assert quality["macro_f1"] == pytest.approx(.01)


def test_iterative_steps_include_partial_groups_and_best_checkpoint_matches_epoch(tmp_path: Path):
    total = 600
    data = ListDataset("regression-fixture", "regression", [{"x": 0.} for _ in range(total)],
                       [0.] * total, [0] * 580 + [1] * 20, {"target_unit": "fixture-units"})
    report = run_iterative(data, tmp_path, max_epochs=3, patience=1, log=lambda _: None)
    first, update = report["training_history"]
    assert first["summary"]["best_epoch"] == 1
    assert first["observation_epochs"][-1]["early_stopped"]
    assert [row["metrics"]["training/batch_size"] for row in first["observation_steps"]] == [512, 10, 512, 10]
    assert [(row["first_step"], row["last_step"]) for row in first["observation_epochs"]] == [(1, 2), (3, 4)]
    assert first["observation_scopes"]["validation"]["target_unit"] == "fixture-units"
    assert first["observation_scopes"]["validation"]["evaluated_examples"] == 58
    assert len(first["observation_scopes"]["validation"]["residuals"]["error"]) == 58
    assert update["source_model_version"] == first["deployed_version"]
    pre_initial, pre_update = report["tracking_observations"]["windows"]
    assert pre_initial["fit_id"] is None and pre_initial["model_version"] is None
    assert pre_update["fit_id"] == first["fit_id"]
    assert pre_update["model_version"] == first["deployed_version"]
    assert report["tracking_observations"]["effort"]["unique_training_examples"] == 540
    assert report["tracking_observations"]["effort"]["training_presentations"] == 1080
    assert report["tracking_observations"]["effort"]["replay_presentations"] == 0
    assert report["tracking_observations"]["effort"]["repeated_epoch_presentations"] == 540
