"""Exact class-only batch scoring and bounded native decoder reuse."""

import copy
import hashlib
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
from river import linear_model, preprocessing

from continual_bench.core.contracts import Event
from continual_bench.evaluation.binary_retraining import (
    BinaryLearner,
    FeatureCodec,
    _dense_logistic_classes,
)
from continual_bench.models.linear import RiverModel
from continual_bench.models.tree import HoeffdingTreeModel


def events_for(rows):
    return [Event(str(i), i, i, i, row, bool(i % 2)) for i, row in enumerate(rows)]


def fixture(seed=0, count=500, dimensions=33):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, dimensions))
    rows = [{f"f{j}": float(value) for j, value in enumerate(row)} for row in x]
    events = events_for(rows)
    codec = FeatureCodec(events)
    encoded, labels = codec.encode(events)
    model = RiverModel()
    for row, target in zip(rows[:200], labels[:200], strict=True):
        model.learn(row, bool(target))
    return codec, encoded, labels, model


def scalar_losses(model, codec, x, y):
    return np.asarray([model.predict(row) >= 0.5 for row in codec.decode(x)], dtype=bool) != y.astype(bool)


@pytest.mark.parametrize("seed", [0, 42, 43, 44, 2026])
def test_dense_numeric_logistic_losses_equal_scalar_for_three_seeds_and_chunk_boundaries(seed):
    codec, x, y, model = fixture(seed, count=8_205)
    before = x.copy()
    state_before = hashlib.sha256(pickle.dumps(model)).hexdigest()
    learner = BinaryLearner(RiverModel, codec, 2048)
    assert _dense_logistic_classes(model, x, codec) is not None
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(model, codec, x, y))
    np.testing.assert_array_equal(x, before)
    assert hashlib.sha256(pickle.dumps(model)).hexdigest() == state_before


@pytest.mark.parametrize("trained", [False, True])
def test_untrained_and_zero_variance_scalers_have_identical_classes(trained):
    events = events_for([{"constant": 7.0, "flag": bool(i % 2)} for i in range(20)])
    codec = FeatureCodec(events)
    x, y = codec.encode(events)
    model = RiverModel()
    if trained:
        for event in events:
            model.learn(event.x, event.y)
    assert model.scaler.vars.get("constant", 0.0) == 0.0
    reference = copy.deepcopy(model)
    before = hashlib.sha256(pickle.dumps(model)).hexdigest()
    assert (_dense_logistic_classes(model, x, codec) is not None) is trained
    assert hashlib.sha256(pickle.dumps(model)).hexdigest() == before
    learner = BinaryLearner(RiverModel, codec, 20)
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(reference, codec, x, y))
    assert dict(model.scaler.means) == dict(reference.scaler.means)
    assert dict(model.scaler.vars) == dict(reference.scaler.vars)
    assert (
        hashlib.sha256(pickle.dumps(model)).hexdigest() == hashlib.sha256(pickle.dumps(reference)).hexdigest()
    )
    if trained:
        assert hashlib.sha256(pickle.dumps(model)).hexdigest() == before


def test_unseen_scaler_keys_preserve_scalar_state_changes_and_model_hash():
    _, _, _, model = fixture(count=30)
    events = events_for([{**{f"f{i}": float(i) for i in range(33)}, "unseen": 1.0}])
    codec = FeatureCodec(events)
    x, y = codec.encode(events)
    reference = copy.deepcopy(model)
    before = hashlib.sha256(pickle.dumps(model)).hexdigest()
    assert "unseen" not in model.scaler.vars and "unseen" not in model.scaler.means
    assert _dense_logistic_classes(model, x, codec) is None
    assert hashlib.sha256(pickle.dumps(model)).hexdigest() == before
    learner = BinaryLearner(RiverModel, codec, 20)
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(reference, codec, x, y))
    assert dict(model.scaler.means) == dict(reference.scaler.means)
    assert dict(model.scaler.vars) == dict(reference.scaler.vars)
    assert (
        hashlib.sha256(pickle.dumps(model)).hexdigest() == hashlib.sha256(pickle.dumps(reference)).hexdigest()
    )


@pytest.mark.parametrize("intercept", [0.0, -1e-18, 1e-18, -1e-14, 1e-14])
def test_near_zero_logits_use_scalar_sigmoid_ties_exactly(intercept):
    codec, x, y, model = fixture(count=30)
    model.classifier.intercept = intercept
    for name in tuple(model.classifier.weights):
        model.classifier._weights[name] = 0.0
    learner = BinaryLearner(RiverModel, codec, 20)
    bulk = _dense_logistic_classes(model, x, codec)
    assert bulk is not None
    if intercept != 0.0:
        assert len(bulk[1]) == len(x)
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(model, codec, x, y))
    if intercept == -1e-18:
        # exp rounds this tiny negative logit to a probability of exactly 0.5.
        assert model.predict(codec.decode(x[:1])[0]) == 0.5


def test_cancellation_and_large_finite_values_retain_scalar_classes():
    events = events_for([{"a": value, "b": value} for value in [1e150, -1e150, 1.0, -1.0, 0.0]])
    codec = FeatureCodec(events)
    x, y = codec.encode(events)
    model = RiverModel()
    model.scaler.means.update(a=0.0, b=0.0)
    model.scaler.vars.update(a=1.0, b=1.0)
    model.classifier._weights.update({"a": 1.0, "b": -1.0})
    learner = BinaryLearner(RiverModel, codec, 20)
    bulk = _dense_logistic_classes(model, x, codec)
    assert bulk is not None and len(bulk[1]) == 4
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(model, codec, x, y))
    # Overflow in a finite-input weighted reduction also falls back rather than
    # classifying an approximate or invalid bulk logit.
    model.classifier._weights.update({"a": 1e200, "b": -1e200})
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(model, codec, x, y))


def test_all_uncertain_rows_use_bounded_scalar_fallback_chunks():
    events = events_for([{"a": float(i + 1), "b": float(i + 1)} for i in range(9_001)])
    codec = FeatureCodec(events)
    x, y = codec.encode(events)
    model = RiverModel()
    model.scaler.means.update(a=0.0, b=0.0)
    model.scaler.vars.update(a=1.0, b=1.0)
    model.classifier._weights.update({"a": 1.0, "b": -1.0})
    learner = BinaryLearner(RiverModel, codec, 20)
    decode = Mock(wraps=codec.decode)
    codec.decode = decode
    np.testing.assert_array_equal(learner.losses(model, x, y), np.logical_not(y.astype(bool)))
    sizes = [len(call.args[0]) for call in decode.call_args_list]
    assert sizes == [4096, 4096, 809]
    assert learner._decoded_cache is None


@pytest.mark.parametrize(
    "rows",
    [
        [{"x": 1.0}, {"x": None}, {}],
        [{"x": "AA"}, {"x": "DL"}, {"x": "AA"}],
        [{"x": True}, {"x": False}, {"x": True}],
    ],
)
def test_missing_nominal_and_boolean_features_preserve_decoder_semantics(rows):
    events = events_for(rows)
    codec = FeatureCodec(events)
    x, y = codec.encode(events)
    model = RiverModel()
    for event in events:
        model.learn(event.x, event.y)
    learner = BinaryLearner(RiverModel, codec, 20)
    if codec.cat_to_id or not np.isfinite(x).all():
        assert _dense_logistic_classes(model, x, codec) is None
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(model, codec, x, y))
    if codec.bool_cols:
        x[0, 0] = 0.1
        assert _dense_logistic_classes(model, x, codec) is None
        np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(model, codec, x, y))


@pytest.mark.parametrize("variant", ["subclass", "classifier_subclass", "scaler_subclass", "no_std"])
def test_nonstandard_models_keep_scalar_path(variant):
    codec, x, y, model = fixture(count=30)
    if variant == "subclass":

        class Custom(RiverModel):
            def predict(self, x):
                return 0.1

        model = Custom()
    elif variant == "classifier_subclass":

        class CustomClassifier(linear_model.LogisticRegression):
            pass

        model.classifier = CustomClassifier()
    elif variant == "scaler_subclass":

        class CustomScaler(preprocessing.StandardScaler):
            pass

        model.scaler = CustomScaler()
    else:
        model.scaler.with_std = False
    assert _dense_logistic_classes(model, x, codec) is None
    learner = BinaryLearner(RiverModel, codec, 20)
    np.testing.assert_array_equal(learner.losses(model, x, y), scalar_losses(model, codec, x, y))


def test_native_tree_decode_cache_reuses_one_array_and_detects_in_place_mutation():
    codec, x, y, _ = fixture(count=30)
    tree = HoeffdingTreeModel()
    for row, target in zip(codec.decode(x[:20]), y[:20], strict=True):
        tree.learn(row, bool(target))
    original = copy.deepcopy(x)
    learner = BinaryLearner(HoeffdingTreeModel, codec, 20)
    decode = Mock(wraps=codec.decode)
    codec.decode = decode
    first = learner.losses(tree, x, y)
    second = learner.losses(tree, x, y)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(x, original)
    assert learner._decoded_cache[2] == codec.decode(x)
    assert decode.call_count == 2
    x[0, 0] += 1.0
    learner.losses(tree, x, y)
    assert decode.call_count == 3
    other = x[:10].copy()
    learner.losses(tree, other, y[:10])
    assert learner._decoded_cache[0] is other
    learner.losses(tree, x, y)
    assert decode.call_count == 5  # the first array was evicted, not kept in a second cache entry


def test_mutating_custom_predictor_never_receives_shared_cached_dictionaries():
    codec, x, y, _ = fixture(count=15)
    before = x.copy()

    class MutatingModel:
        def predict(self, row):
            assert "injected" not in row
            row["injected"] = 1.0
            return 0.6

    learner = BinaryLearner(MutatingModel, codec, 20)
    model = MutatingModel()
    expected = np.logical_not(y.astype(bool))
    np.testing.assert_array_equal(learner.losses(model, x, y), expected)
    np.testing.assert_array_equal(learner.losses(model, x, y), expected)
    np.testing.assert_array_equal(x, before)
    assert learner._decoded_cache is None


def test_object_array_bypasses_cache_and_observes_mutation_inside_same_object():
    class MutableFloat:
        def __init__(self, value):
            self.value = value

        def __float__(self):
            return self.value

    codec = FeatureCodec(events_for([{"x": 0.0}]))
    value = MutableFloat(-1.0)
    x = np.asarray([[value]], dtype=object)
    y = np.array([False])
    model = RiverModel()
    model.scaler.means["x"] = 0.0
    model.scaler.vars["x"] = 1.0
    model.classifier._weights["x"] = 1.0
    learner = BinaryLearner(RiverModel, codec, 20)
    assert _dense_logistic_classes(model, x, codec) is None
    assert learner.losses(model, x, y).tolist() == [False]
    value.value = 1.0
    assert learner.losses(model, x, y).tolist() == [True]
    assert learner._decoded_cache is None
