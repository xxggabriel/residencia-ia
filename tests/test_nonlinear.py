"""Contracts for bounded nonlinear models; no external datasets or services."""

import math
import pickle

import pytest

from continual_bench.adapters.nonlinear import HoeffdingTreeModel, KNNModel
from continual_bench.adapters.river import RiverModel


@pytest.mark.parametrize("factory", [HoeffdingTreeModel, KNNModel])
def test_nonlinear_prediction_is_read_only_for_known_and_unseen_features(factory):
    model = factory()
    empty = pickle.dumps(model, protocol=5)
    assert model.predict({"value": 1.0, "category": "new"}) == 0.5
    assert pickle.dumps(model, protocol=5) == empty
    for i in range(100):
        model.learn({"value": float(i % 7), "category": "a" if i % 2 else "b"}, bool(i % 2))
    before = pickle.dumps(model, protocol=5)
    for features in (
        {"value": 1.0, "category": "a"},
        {"value": 50.0, "category": "unseen"},
        {"unknown_numeric": 2.0, "unknown_nominal": "z"},
        {},
    ):
        probability = model.predict(features)
        assert math.isfinite(probability) and 0 <= probability <= 1
        assert pickle.dumps(model, protocol=5) == before


@pytest.mark.parametrize("factory", [HoeffdingTreeModel, KNNModel])
def test_single_negative_class_probability_is_zero(factory):
    model = factory()
    model.learn({"x": 1.0, "color": "red"}, False)
    assert model.predict({"x": 1.0, "color": "red"}) == 0.0


@pytest.mark.parametrize("factory", [HoeffdingTreeModel, KNNModel])
def test_nonlinear_models_pickle_and_resume_predictions_exactly(factory):
    model = factory()
    for i in range(80):
        model.learn({"x": float(i % 5), "cat": str(i % 3)}, i % 2 == 0)
    restored = pickle.loads(pickle.dumps(model, protocol=5))
    for i in range(80, 140):
        features, label = {"x": float(i % 5), "cat": str(i % 3)}, i % 2 == 0
        assert restored.predict(features) == model.predict(features)
        restored.learn(features, label)
        model.learn(features, label)
    for x in range(5):
        for category in ("0", "1", "2", "unseen"):
            assert restored.predict({"x": float(x), "cat": category}) == model.predict(
                {"x": float(x), "cat": category}
            )
    if isinstance(model, KNNModel):
        assert list(restored.classifier._nn.window) == list(model.classifier._nn.window)
        assert dict(restored.scaler.means) == dict(model.scaler.means)
        assert dict(restored.scaler.vars) == dict(model.scaler.vars)
    else:
        assert restored.classifier._train_weight_seen_by_model == model.classifier._train_weight_seen_by_model


def test_tree_learns_native_categorical_rule_without_numeric_scaler():
    model = HoeffdingTreeModel()
    for i in range(200):
        color = ("red", "green", "blue")[i % 3]
        model.learn({"color": color, "noise": float(i % 7)}, color == "red")
    assert model.predict({"color": "red", "noise": 2.0}) >= 0.5
    assert model.predict({"color": "green", "noise": 2.0}) < 0.5
    assert model.predict({"color": "blue", "noise": 2.0}) < 0.5
    assert not hasattr(model, "scaler")
    assert model.classifier.max_depth == 8 and model.classifier.max_size == 8.0


def test_knn_learns_categorical_xor_with_bounded_neighbor_window():
    model = KNNModel()
    examples = [({"a": a, "b": b}, (a == "on") != (b == "on")) for a in ("off", "on") for b in ("off", "on")]
    for _ in range(100):
        for features, label in examples:
            model.learn(features, label)
    assert all((model.predict(x) >= 0.5) == y for x, y in examples)
    assert len(model.classifier._nn.window) == 256
    assert model.classifier._nn.window.maxlen == 256


def test_knn_encoder_matches_logistic_encoder_for_known_features():
    model, reference = KNNModel(), RiverModel()
    for i in range(20):
        features = {"value": float(i), "category": str(i % 3)}
        assert model.encode(features, learn=True) == reference.encode(features, learn=True)
    query = {"value": 13.0, "category": "unseen"}
    assert model.encode(query, learn=False) == reference.encode(query, learn=False)


@pytest.mark.parametrize(
    "factory,params",
    [
        (KNNModel, {"n_neighbors": 0}),
        (KNNModel, {"window_size": 0}),
        (KNNModel, {"n_neighbors": 10, "window_size": 5}),
        (HoeffdingTreeModel, {"grace_period": 0}),
        (HoeffdingTreeModel, {"max_depth": 0}),
        (HoeffdingTreeModel, {"max_size": 0}),
        (HoeffdingTreeModel, {"memory_estimate_period": 0}),
        (HoeffdingTreeModel, {"leaf_prediction": "invalid"}),
        (HoeffdingTreeModel, {"nominal_attributes": "category"}),
        (HoeffdingTreeModel, {"nominal_attributes": ["category", "category"]}),
    ],
)
def test_invalid_nonlinear_limits_fail_explicitly(factory, params):
    with pytest.raises(ValueError):
        factory(**params)


def test_tree_can_treat_integer_category_codes_as_nominal():
    model = HoeffdingTreeModel(nominal_attributes=["code"])
    model.learn({"code": 2, "value": 2.0}, True)
    assert model.classifier.nominal_attributes == ["code"]
    assert type(model.classifier._root.splitters["code"]).__name__ == "NominalSplitterClassif"
    assert type(model.classifier._root.splitters["value"]).__name__ == "GaussianSplitter"


@pytest.mark.parametrize("factory", [HoeffdingTreeModel, KNNModel])
def test_frozen_engine_checkpoint_resume_preserves_prediction_digest(factory):
    from continual_bench.adapters.river import NullDetector
    from continual_bench.domain.contracts import Event
    from continual_bench.domain.engine import Engine
    from continual_bench.domain.policies import PolicyStrategy

    reference = Engine(factory, NullDetector, PolicyStrategy("none"), warmup=30, window=64, metric_window=20)
    events = [
        Event(str(i), i, i, i, {"x": float(i % 5), "cat": str(i % 3)}, bool(i % 2), None) for i in range(120)
    ]
    for event in events[:60]:
        reference.observe_batch([event])
    restored = pickle.loads(pickle.dumps(reference, protocol=5))
    for event in events[60:]:
        reference.observe_batch([event])
        restored.observe_batch([event])
    reference.finish()
    restored.finish()
    assert reference.report()["prediction_digest"] == restored.report()["prediction_digest"]
    assert reference.report()["metrics"] == restored.report()["metrics"]
    assert reference.report()["updates"] == restored.report()["updates"] == []


def tree_with_partial_nominal_observer(*, existing_empty=False):
    from river.tree.nodes.htc_nodes import LeafNaiveBayesAdaptive
    from river.tree.splitter.nominal_splitter_classif import NominalSplitterClassif

    model = HoeffdingTreeModel()
    model.classifier.classes = {False, True}
    leaf = LeafNaiveBayesAdaptive({False: 1.0, True: 1.0}, 0, model.classifier.splitter)
    observer = NominalSplitterClassif()
    observer.update("known", False, 1.0)
    if existing_empty:
        observer._att_dist_per_class[True] = {}
    leaf.splitters["category"] = observer
    model.classifier._root = leaf
    return model, observer


@pytest.mark.parametrize("existing_empty", [False, True])
@pytest.mark.parametrize("category", ["known", "unknown"])
def test_tree_nominal_prediction_preserves_partial_observer_and_sdk_probabilities(existing_empty, category):
    import copy

    model, observer = tree_with_partial_nominal_observer(existing_empty=existing_empty)
    sdk = copy.deepcopy(model.classifier)
    query = {"category": category}
    expected = sdk.predict_proba_one(query).get(True, 0.0)
    assert True in sdk._root.splitters["category"]._att_dist_per_class
    before = pickle.dumps(model, protocol=5)
    assert model.predict(query) == expected
    assert pickle.dumps(model, protocol=5) == before
    assert (True in observer._att_dist_per_class) is existing_empty
    assert model.predict(query) == expected
    assert pickle.dumps(model, protocol=5) == before


def test_tree_nominal_prediction_rolls_back_empty_entries_when_sdk_raises(monkeypatch):
    from river import tree

    model, observer = tree_with_partial_nominal_observer()
    before = pickle.dumps(model, protocol=5)
    original = tree.HoeffdingTreeClassifier.predict_proba_one

    def interrupted(classifier, features):
        original(classifier, features)
        assert True in observer._att_dist_per_class
        raise ValueError("prediction interrupted")

    monkeypatch.setattr(tree.HoeffdingTreeClassifier, "predict_proba_one", interrupted)
    with pytest.raises(ValueError, match="prediction interrupted"):
        model.predict({"category": "known"})
    assert True not in observer._att_dist_per_class
    assert pickle.dumps(model, protocol=5) == before


def test_tree_nominal_prediction_rejects_nonempty_new_statistics_without_deleting_them(monkeypatch):
    from river import tree

    model, observer = tree_with_partial_nominal_observer()

    def unexpectedly_learning(classifier, features):
        observer._att_dist_per_class[True]["new"] = 1.0
        return {False: 0.5, True: 0.5}

    monkeypatch.setattr(tree.HoeffdingTreeClassifier, "predict_proba_one", unexpectedly_learning)
    with pytest.raises(RuntimeError, match="nonempty nominal class statistics"):
        model.predict({"category": "known"})
    assert observer._att_dist_per_class[True] == {"new": 1.0}
