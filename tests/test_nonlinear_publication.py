"""Versioning/serving regressions for nonlinear state, including frozen checkpoint recovery."""

import hashlib
import json
import pickle
from functools import partial

import pytest

from continual_bench.adapters.catalog import model_family_name
from continual_bench.adapters.nonlinear import HoeffdingTreeModel, KNNModel
from continual_bench.adapters.river import NullDetector
from continual_bench.adapters.versioning import ModelVersions, _canonical_state
from continual_bench.application.config import Config
from continual_bench.application.runner import run
from continual_bench.domain.contracts import Event
from continual_bench.domain.engine import Engine
from continual_bench.domain.policies import PolicyStrategy


@pytest.mark.parametrize("model_class", [HoeffdingTreeModel, KNNModel])
def test_complete_frozen_state_survives_checkpoint_without_new_training(tmp_path, model_class):
    model = model_class()
    for i in range(100):
        model.learn({"x": float(i % 5), "cat": str(i % 3)}, bool(i % 2))
    versions = ModelVersions(tmp_path, "d" * 64, "c" * 64)
    version_id = versions.record(model, {"action": "initial_fit", "seq_range": [0, 100]})
    recorded = versions.journal.read_bytes()
    model, versions = pickle.loads(pickle.dumps((model, versions)))
    metadata = versions.finish(model, {"x": 2.0, "cat": "1"})
    assert metadata["count"] == 1 and metadata["final_version_id"] == version_id
    assert versions.journal.read_bytes() == recorded
    assert metadata["state_sha256"] == hashlib.sha256(recorded).hexdigest()
    model.learn({"x": 200.0, "cat": "different"}, True)
    with pytest.raises(ValueError, match="not versioned"):
        versions.finish(model, {"x": 2.0, "cat": "1"})


@pytest.mark.parametrize("model_class", [HoeffdingTreeModel, KNNModel])
def test_engine_checkpoint_preserves_prediction_digest_and_version_count(tmp_path, model_class):
    def engine(directory):
        return Engine(
            partial(model_class),
            NullDetector,
            PolicyStrategy("none"),
            warmup=30,
            window=64,
            metric_window=30,
            reset_detector=True,
            versions=ModelVersions(directory, "d" * 64, "c" * 64),
        )

    events = [
        Event(str(i), i, i, i + 3, {"x": float(i % 5), "cat": str(i % 3)}, bool(i % 2), None)
        for i in range(180)
    ]
    uninterrupted, interrupted = engine(tmp_path / "one"), engine(tmp_path / "two")
    for event in events:
        uninterrupted.observe_batch([event])
    for event in events[:90]:
        interrupted.observe_batch([event])
    interrupted = pickle.loads(pickle.dumps(interrupted))
    for event in events[90:]:
        interrupted.observe_batch([event])
    for item in (uninterrupted, interrupted):
        item.finish()
        assert item.versions.finish(item.model, events[0].x)["count"] == 1
    assert uninterrupted.report()["metrics"] == interrupted.report()["metrics"]
    assert uninterrupted.report()["prediction_digest"] == interrupted.report()["prediction_digest"]


def test_canonical_verification_preserves_mutable_aliases_and_dict_order():
    shared = [1, 2]
    assert _canonical_state({"a": shared, "b": shared}) != _canonical_state({"a": [1, 2], "b": [1, 2]})
    assert _canonical_state({"a": 1, "b": 2}) != _canonical_state({"b": 2, "a": 1})
    value = {"set": {"0", "1", "2"}, "bytes": b"value"}
    assert _canonical_state(value) == _canonical_state(pickle.loads(pickle.dumps(value)))


def test_registry_separates_models_and_preserves_logistic_names():
    base = {
        "dataset": {"name": "airlines"},
        "mode": "streaming",
        "policy": {"name": "online"},
        "detector": {"name": "none"},
    }
    assert model_family_name({**base, "model": {"name": "logistic"}}) == (
        "continual-bench-airlines-streaming-online-none"
    )
    for name in ("hoeffding_tree", "knn"):
        assert model_family_name({**base, "model": {"name": name}}) == (
            f"continual-bench-airlines-{name}-streaming-online-none"
        )


@pytest.mark.parametrize("name", ["hoeffding_tree", "knn"])
def test_nonlinear_model_packages_load_in_fresh_process(tmp_path, name):
    pytest.importorskip("mlflow.pyfunc", exc_type=ImportError)
    import subprocess
    import sys

    from continual_bench.adapters.catalog import prepare_mlflow_model

    cfg = Config.model_validate(
        {
            "name": "nonlinear-package",
            "dataset": {"name": "stagger", "samples": 180},
            "warmup": 30,
            "model": {"name": name},
            "detector": {"name": "none"},
            "policy": {"name": "online", "action": "learn_one"},
        }
    )
    report = run(cfg, tmp_path)
    path = prepare_mlflow_model(tmp_path / cfg.identity())
    request = {
        "uri": str(path),
        "dataset": "stagger",
        "model": name,
        "probe": report["model_versions"]["inference_probe"],
    }
    from pathlib import Path

    checked = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/registry_inference_probe.py")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(checked.stdout)
    assert result["fresh_process"] and result["model"] == name
    assert "/code/continual_bench/adapters/nonlinear.py" in result["bundled_model_code"]


@pytest.mark.parametrize("name", ["logistic", "hoeffding_tree", "knn"])
def test_numeric_signature_accepts_fractional_values_after_first_integer_row(tmp_path, name):
    pytest.importorskip("mlflow.pyfunc", exc_type=ImportError)
    import mlflow.pyfunc

    from continual_bench.adapters.catalog import prepare_mlflow_model
    from continual_bench.adapters.model_loader import RiverPredictor, inference_frame

    cfg = Config.model_validate(
        {
            "name": "numeric-signature",
            "dataset": {"name": "agrawal", "samples": 180},
            "model": {"name": name},
            "policy": {"name": "none"},
            "warmup": 30,
        }
    )
    report = run(cfg, tmp_path)
    directory = tmp_path / cfg.identity()
    model = mlflow.pyfunc.load_model(str(prepare_mlflow_model(directory)))
    probe = dict(report["model_versions"]["inference_probe"]["features"])
    assert probe["commission"] == 0
    probe["commission"] = 1234.5
    frame = inference_frame(probe)
    assert model.predict(frame).tolist() == RiverPredictor(directory).predict(frame).tolist()
