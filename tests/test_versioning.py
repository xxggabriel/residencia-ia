import hashlib
import json
import pickle

import pytest

from continual_bench.adapters.river import RiverModel
from continual_bench.adapters.versioning import ModelVersions, load_version
from continual_bench.application.config import Config
from continual_bench.application.runner import run


def test_version_journal_recovers_uncommitted_tail_and_reloads_states(tmp_path):
    model = RiverModel()
    versions = ModelVersions(tmp_path, "d" * 64, "c" * 64)
    model.learn({"x": 1.0}, True)
    first = versions.record(model, {"seq_range": [0, 1], "action": "initial_fit"})
    saved = pickle.dumps((model, versions))
    model.learn({"x": 2.0}, False)
    orphan = versions.record(model, {"seq_range": [1, 2], "action": "learn_one"})
    model, versions = pickle.loads(saved)
    model.learn({"x": 3.0}, True)
    final = versions.record(model, {"seq_range": [2, 3], "action": "learn_one"})
    info = versions.finish(model, {"x": 4.0})
    index = json.loads((tmp_path / "model-versions.json").read_text())
    assert len(index["versions"]) == info["count"] == 2
    assert index["versions"][1]["parent_version"] == first
    assert orphan not in [v["version_id"] for v in index["versions"]]
    assert load_version(tmp_path, final).predict({"x": 4.0}) == model.predict({"x": 4.0})
    with (tmp_path / "models.bin").open("r+b") as stream:
        stream.seek(index["versions"][-1]["offset"])
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_version(tmp_path, final)


@pytest.mark.parametrize("with_metadata", [True, False])
def test_loading_version_rejects_lineage_corruption_with_unchanged_weights(tmp_path, with_metadata):
    model = RiverModel()
    model.learn({"x": 1.0}, True)
    versions = ModelVersions(tmp_path, "d" * 64, "c" * 64)
    version = versions.record(model, {"seq_range": [0, 1], "action": "initial_fit"})
    versions.finish(model, {"x": 2.0})
    path = tmp_path / "model-versions.json"
    index = json.loads(path.read_text())
    index["versions"][0]["training"]["seq_range"] = [100, 101]
    path.write_text(json.dumps(index))
    if not with_metadata:
        (tmp_path / "model-metadata.json").unlink()
    with pytest.raises(ValueError, match="checksum"):
        load_version(tmp_path, version)


@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_each_trained_model_links_data_and_roundtrips(mode, tmp_path):
    config = Config.model_validate({"name": "version-unit", "mode": mode,
                                    "dataset": {"samples": 180, "label_delay": 3}, "warmup": 20,
                                    "policy": {"name": "none" if mode == "offline" else "online",
                                               "action": "learn_one"}})
    report = run(config, tmp_path)
    directory = tmp_path / config.identity()
    metadata = report["model_versions"]
    assert metadata["count"] == (1 if mode == "offline" else 161)
    entries = json.loads((directory / "model-versions.json").read_text())["versions"]
    assert all(e["dataset_version"] == report["dataset"]["sha256"] for e in entries)
    assert metadata["state_sha256"] == hashlib.sha256((directory / "model.pkl").read_bytes()).hexdigest()


def test_mlflow_model_flavor_loads_complete_preprocessing(tmp_path):
    pytest.importorskip("mlflow.pyfunc", exc_type=ImportError)
    import mlflow.pyfunc
    import pandas as pd

    from continual_bench.adapters.catalog import prepare_mlflow_model

    config = Config.model_validate({"dataset": {"samples": 120}, "warmup": 20})
    report = run(config, tmp_path)
    path = prepare_mlflow_model(tmp_path / config.identity())
    loaded = mlflow.pyfunc.load_model(str(path))
    probe = report["model_versions"]["inference_probe"]
    assert loaded.predict(pd.DataFrame([probe["features"]]))[0] == probe["probability"]


def test_mlflow_packaging_recovers_after_metadata_written_before_code(tmp_path, monkeypatch):
    pytest.importorskip("mlflow.pyfunc", exc_type=ImportError)
    from pathlib import Path

    import mlflow.pyfunc

    from continual_bench.adapters.catalog import prepare_mlflow_model

    config = Config.model_validate({"dataset": {"samples": 120}, "warmup": 20})
    run(config, tmp_path)
    directory = tmp_path / config.identity()
    save = mlflow.pyfunc.save_model

    def interrupted(**kwargs):
        target = Path(kwargs["path"])
        target.mkdir()
        (target / "MLmodel").write_text("partial package")
        raise RuntimeError("interrupted before inference code and dependencies")

    monkeypatch.setattr(mlflow.pyfunc, "save_model", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_mlflow_model(directory)
    assert not (directory / "mlflow-model" / ".ready.json").exists()
    monkeypatch.setattr(mlflow.pyfunc, "save_model", save)
    target = prepare_mlflow_model(directory)
    assert (target / ".ready.json").is_file()
    assert (target / "requirements.txt").is_file()
    assert (target / "code/continual_bench/adapters/model_loader.py").is_file()
    assert prepare_mlflow_model(directory) == target
