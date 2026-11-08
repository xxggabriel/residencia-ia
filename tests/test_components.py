"""Catalog-driven decoupling: any dataset with any model family supporting its target type."""

import hashlib
import pickle
from collections import OrderedDict

import pytest
from pydantic import ValidationError

from continual_bench.application.components import default_registry, make_source
from continual_bench.config import Component, Config, DatasetConfig, EpochConfig
from continual_bench.core.policies import (
    DriftTriggeredRetraining,
    NoRetraining,
    PeriodicRetraining,
    PolicyStrategy,
)
from continual_bench.core.tasks import BINARY, MULTICLASS, REGRESSION, task_kind
from continual_bench.datasets.catalog import CATALOG, DATASETS, datasets_for
from continual_bench.evaluation.epochs import run_epochs
from continual_bench.evaluation.typed import run_typed
from continual_bench.infrastructure.storage import LocalCheckpoints
from continual_bench.models import MODELS, IncompatibleComponent, compatible_models, default_model

SAMPLES = {
    BINARY: ({"a": 1.0, "b": "x"}, True),
    MULTICLASS: ({"a": 1.0, "b": "x"}, "red"),
    REGRESSION: ({"a": 1.0, "c": 2.0}, 3.5),
}


@pytest.mark.parametrize("family,task", [(m.name, t) for m in MODELS for t in m.tasks])
def test_every_declared_model_task_builds_learns_and_predicts(family, task):
    model = MODELS.get(family).build(task)
    x, y = SAMPLES[task]
    if task == BINARY:
        assert 0 <= model.predict(x) <= 1
        model.learn(x, y)
        assert 0 <= model.predict(x) <= 1
    else:
        model.learn_one(x, y)
        prediction = task_kind(task).normalize_prediction(model.predict_one(x))
        assert isinstance(prediction, float if task == REGRESSION else str)
    pickle.loads(pickle.dumps(model))  # Checkpoints and model versions pickle every learner.


def test_compatibility_is_explicit_and_catalog_declares_every_task():
    assert set(compatible_models(REGRESSION)) == {"linear_regression", "hoeffding_tree", "knn"}
    assert "linear_regression" not in compatible_models(BINARY)
    with pytest.raises(IncompatibleComponent, match="does not support binary"):
        MODELS.get("linear_regression").build(BINARY)
    assert default_model(MULTICLASS) == "hoeffding_tree" and default_model(REGRESSION) == "linear_regression"
    assert len(DATASETS) == 17 and "stream51" not in CATALOG
    assert "friedman_drift" in datasets_for(REGRESSION)
    assert all(CATALOG.get(name).task in (BINARY, MULTICLASS, REGRESSION) for name in DATASETS)


def test_binary_engine_config_rejects_incompatible_dataset_or_model():
    with pytest.raises(ValidationError, match="regression targets"):
        Config.model_validate({"dataset": {"name": "friedman_drift"}})
    with pytest.raises(ValidationError, match="does not support binary"):
        Config.model_validate({"model": {"name": "linear_regression"}})


def test_plugin_registers_models_for_explicit_tasks():
    registry = default_registry()
    registry.register("model", "constant", lambda: _Constant(), tasks=(REGRESSION,))
    assert registry.models.get("constant").supports(REGRESSION)
    with pytest.raises(IncompatibleComponent):
        registry.model_factory("constant", BINARY)
    registry.ensure_compatible("friedman_drift", "constant")
    with pytest.raises(IncompatibleComponent):
        registry.ensure_compatible("sea", "constant")
    assert "constant" not in MODELS  # Plugins never mutate the global catalog.


class _Constant:
    def predict_one(self, x):
        return 0.0

    def learn_one(self, x, y):
        pass


def test_policy_factory_returns_one_strategy_per_kind():
    assert isinstance(PolicyStrategy("none"), NoRetraining)
    assert isinstance(PolicyStrategy("periodic", every=5), PeriodicRetraining)
    assert isinstance(PolicyStrategy("drift"), DriftTriggeredRetraining)
    with pytest.raises(ValueError, match="unknown policy"):
        PolicyStrategy("sometimes")
    with pytest.raises(ValueError, match="matching policy"):
        PolicyStrategy("periodic", params={"error_threshold": 0.2})


@pytest.mark.parametrize("family", ["linear_regression", "hoeffding_tree", "knn"])
def test_typed_regression_runs_with_any_regression_family(tmp_path, family):
    source = make_source(default_registry(), DatasetConfig(name="friedman_drift", samples=200, label_delay=2))
    report = run_typed(source, tmp_path, policy="periodic", every=40, max_samples=200,
                       model={"name": family, "params": {}})
    assert report["complete"] and report["config"]["model"]["family"] == family
    assert report["metrics"]["train"]["n"] == 180 and report["metrics"]["train"]["mae"] >= 0


@pytest.mark.parametrize("family", ["logistic", "hoeffding_tree", "knn"])
def test_binary_stream_feeds_any_multiclass_family_off_engine(tmp_path, family):
    source = make_source(default_registry(), DatasetConfig(name="sea", samples=200, label_delay=1))
    report = run_typed(source, tmp_path, max_samples=200, model={"name": family})
    assert report["config"]["task"] == MULTICLASS and report["config"]["source"]["labels"] == ["0", "1"]
    assert 0 <= report["metrics"]["train"]["accuracy"] <= 1
    with pytest.raises(IncompatibleComponent):
        run_typed(source, tmp_path, max_samples=200, model={"name": "linear_regression"})


def test_epochs_accept_a_model_family_without_changing_default_dump(tmp_path):
    default = EpochConfig(dataset="friedman_drift", examples=60)
    assert "model" not in default.model_dump()
    knn = EpochConfig(dataset="friedman_drift", examples=60, model=Component(name="knn"))
    report = run_epochs(knn, tmp_path, tmp_path / "cache")
    assert report["contract"]["config"]["model"]["name"] == "knn"
    assert report["epochs"][-1]["metrics"]["rmse"] >= 0
    assert report["run_key"] != run_epochs(default, tmp_path, tmp_path / "cache")["run_key"]


def test_checkpoint_from_an_older_package_layout_is_a_contract_mismatch(tmp_path):
    checkpoints = LocalCheckpoints(tmp_path)
    checkpoints.save("old", {"state": OrderedDict()})
    path = checkpoints.path("old")
    blob = path.read_bytes()
    payload = blob[68:].replace(b"collections", b"missing_mod")  # Same length, unresolvable module.

    path.write_bytes(blob[:4] + hashlib.sha256(payload).hexdigest().encode() + payload)
    with pytest.raises(ValueError, match="incompatible code version"):
        checkpoints.load("old")


def test_example_plugin_regression_model_runs_in_the_typed_evaluator(tmp_path):
    registry = default_registry(("examples.custom_plugin:register",))
    source = make_source(registry, DatasetConfig(name="friedman_drift", samples=120))
    report = run_typed(source, tmp_path, max_samples=120, model={"name": "running_mean"},
                       models=registry.models)
    assert report["config"]["model"]["family"] == "running_mean" and report["metrics"]["train"]["mae"] > 0
