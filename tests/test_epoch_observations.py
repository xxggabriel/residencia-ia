from math import sqrt

import pytest

from continual_bench.config import EpochConfig
from continual_bench.core.contracts import TaskSample
from continual_bench.evaluation.epochs import run_epochs
from continual_bench.evaluation.retraining_observations import reserved_retention


class _Source:
    def __init__(self, holdout, **manifest):
        self.holdout = holdout
        self.metadata = manifest

    def manifest(self):
        return {"version": "observations-fixture", **self.metadata}

    def samples_iter(self):
        for index in range(8):
            yield TaskSample(str(index), index, index, index, {"x": float(index)}, self.holdout[0],
                             split="train")
        for index, y in enumerate(self.holdout, 8):
            yield TaskSample(str(index), index, index, index, {"x": float(index)}, y, split="test")


class _ConstantClassifier:
    def learn_one(self, x, y):
        pass

    def predict_one(self, x):
        return "a"


class _RankedClassifier(_ConstantClassifier):
    def predict_one(self, x):
        return "0"

    def predict_proba_one(self, x):
        return {"0": 0.5, "1": 0.5}


class _ConstantRegressor(_ConstantClassifier):
    def predict_one(self, x):
        return 0.0


def test_epoch_observation_counts_and_fixed_macro_universe(tmp_path):
    config = EpochConfig(dataset="sea", examples=10, epochs=2)
    source = _Source(["a", "a"], task="multiclass", class_order=["a", "b", "c"])
    report = run_epochs(config, tmp_path, tmp_path, source=source, model_factory=_ConstantClassifier)
    common = report["tracking_observations"]
    assert report["epochs"][-1]["metrics"]["macro_f1"] == 1.0
    assert common["scopes"]["holdout"]["macro_f1"] == pytest.approx(1 / 3)
    assert common["scopes"]["holdout"]["class_support"] == [2, 0, 0]
    assert common["scopes"]["holdout"]["class_recall"] == [1.0, None, None]
    assert common["effort"]["evaluated_examples"] == 4
    assert common["effort"]["unique_evaluated_examples"] == 2
    assert common["effort"]["unique_training_examples"] == 8
    assert common["effort"]["training_presentations"] == 16
    assert common["effort"]["updates"] == 0
    assert common["resources"]["cost_per_1000_evaluated"] == 250 * common["resources"]["cost_total"]
    [fit] = report["training_history"]
    assert fit["observed_step_kind"] == "learn_one"
    assert [row["step"] for row in fit["steps"]] == [1, 2]  # Existing epoch coordinate.
    assert [row["step"] for row in fit["observation_steps"]] == list(range(1, 17))
    assert [row["metrics"]["training/presentations_cumulative"] for row in fit["observation_steps"]] == list(
        range(1, 17),
    )
    assert [(row["first_step"], row["last_step"]) for row in fit["observation_epochs"]] == [(1, 8), (9, 16)]
    assert fit["observation_steps"][-1]["metrics"]["training/native_loss"] is None
    assert common["lineage"]["checkpoint"] == fit["checkpoint"]


def test_binary_epoch_scores_ties_and_regression_keeps_target_units(tmp_path):
    classifier = _Source(
        ["1", "0"], task="multiclass", labels=["0", "1"],
        target_conversion="explicit binary boolean to class strings, no information loss",
    )
    config = EpochConfig(dataset="sea", examples=10, epochs=1)
    report = run_epochs(config, tmp_path, tmp_path, source=classifier, model_factory=_RankedClassifier)
    common = report["tracking_observations"]
    assert common["task"] == "binary"
    assert common["scopes"]["holdout"]["average_precision"] == 0.5
    assert common["scopes"]["holdout"]["positive_class"] == "1"
    assert common["scopes"]["holdout"]["positive_recall"] == 0.0
    assert common["scopes"]["holdout"]["recall_at_fpr_0_01"] is None

    regressor = _Source([1.0, 3.0], task="regression", target_unit="fixture_unit")
    report = run_epochs(config, tmp_path, tmp_path, source=regressor, model_factory=_ConstantRegressor)
    quality = report["tracking_observations"]["scopes"]["holdout"]
    assert quality["mae"] == 2.0
    assert quality["rmse"] == pytest.approx(sqrt(5.0))
    assert quality["target_unit"] == "fixture_unit"
    assert quality["residuals"] == {"prediction": [0.0, 0.0], "error": [1.0, 3.0]}


def test_retention_scores_share_fixed_heldout_cohorts():
    retention = reserved_retention(
        [[0.8, None, None], [0.9, 0.4, None], [0.7, 0.5, 0.9]], ["first", "middle", "last"],
    )
    assert retention["series"][0]["gem_bwt"] is None
    assert retention["absolute_retention"] == pytest.approx(0.6)
    assert retention["gem_bwt"] == pytest.approx(0.0)
    assert retention["post_learning_fm"] == pytest.approx(0.05)
