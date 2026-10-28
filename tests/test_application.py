import hashlib
import pickle
from pathlib import Path

import pytest
from pydantic import ValidationError

from continual_bench.adapters.local import LocalCheckpoints, run_lock
from continual_bench.adapters.river import RiverDetector, RiverModel
from continual_bench.application.config import Config
from continual_bench.application.registry import default_registry, make_source
from continual_bench.application.runner import evaluate, prepare, read_events, run


def config(mode="streaming", dataset="sea", policy="periodic"):
    return Config.model_validate(
        {
            "mode": mode,
            "warmup": 20,
            "metric_window": 20,
            "dataset": {"name": dataset, "samples": 180, "label_delay": 5},
            "policy": {
                "name": "none" if mode == "offline" else policy,
                "action": "learn_one" if policy == "online" else "reset_retrain",
                "every": 30,
                "min_samples": 20,
                "min_window": 10,
                "cooldown": 10,
            },
        }
    )


@pytest.mark.parametrize("dataset", ["sea", "agrawal", "stagger"])
def test_source_contract_reproducibility_ids_time_and_probe_separation(dataset, tmp_path):
    cfg = config(dataset=dataset)
    manifest = prepare(cfg, tmp_path)
    events = list(read_events(manifest["path"]))
    assert len(events) == 180
    assert [e.seq for e in events] == list(range(180))
    assert len({e.event_id for e in events}) == 180
    assert all(e.available_at - e.timestamp == 5 for e in events)
    assert prepare(cfg, tmp_path)["sha256"] == manifest["sha256"]
    assert hashlib.sha256(Path(manifest["path"]).read_bytes()).hexdigest() == manifest["sha256"]
    source = make_source(default_registry(), cfg.dataset)
    probes = source.probes()
    assert probes == source.probes()
    assert manifest["changepoints"] == [60, 120]
    assert len(list(source.events())) == 180


@pytest.mark.parametrize("dataset", ["sea", "agrawal", "stagger"])
@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_nine_dataset_mode_smokes(dataset, mode, tmp_path):
    cfg = config(mode, dataset)
    result = run(cfg, tmp_path)
    assert result["complete"]
    assert result["observed"] == 180
    expected = 72 if mode == "offline" else 160
    assert result["metrics"]["n"] == expected
    assert sum(w["n"] for w in result["windows"]) == expected
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert run(cfg, tmp_path) == result
    if mode == "offline":
        assert result["split"]["train"] == [0, 103]
        assert result["split"]["purged_label_delay"] == [103, 108]
        assert not result["updates"] and not result["alarms"]


@pytest.mark.parametrize("mode", ["streaming", "incremental"])
@pytest.mark.parametrize("policy", ["none", "periodic", "drift", "online"])
def test_checkpoint_resume_matches_uninterrupted_prediction_and_update_history(mode, policy, tmp_path):
    cfg = config(mode=mode, policy=policy)
    if policy == "online":
        cfg = Config.model_validate(
            {**cfg.model_dump(), "policy": {**cfg.policy.model_dump(), "action": "learn_one"}}
        )
    fresh, interrupted = tmp_path / "fresh", tmp_path / "interrupted"
    expected = run(cfg, fresh)
    manifest = prepare(cfg, interrupted)
    assert not evaluate(cfg, manifest, interrupted, stop_after=73)["complete"]
    result = evaluate(cfg, manifest, interrupted)
    assert expected["prediction_digest"] == result["prediction_digest"]
    assert expected["metrics"] == result["metrics"]
    assert expected["windows"] == result["windows"]

    def compact(report):
        return [{k: v for k, v in u.items() if k != "seconds"} for u in report["updates"]]

    assert compact(expected) == compact(result)
    assert expected["retention_probes"] == result["retention_probes"]


def test_frozen_model_predictions_do_not_update_preprocessing():
    model = RiverModel()
    model.learn({"n": 2, "cat": "a"}, True)
    before = pickle.dumps(model)
    for i in range(5):
        assert 0 <= model.predict({"n": 10000, "cat": "unseen"}) <= 1
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("name", ["adwin", "ddm"])
def test_mature_river_detector_contract(name):
    detector = RiverDetector(name)
    alarms = [detector.update(error) for error in [0] * 500 + [1] * 500]
    assert all(isinstance(value, bool) for value in alarms)
    assert any(alarms)


def test_checkpoint_corruption_lock_and_resume_manifest_fail_closed(tmp_path):
    cfg = config()
    manifest = prepare(cfg, tmp_path)
    evaluate(cfg, manifest, tmp_path, stop_after=40)
    with run_lock(tmp_path, cfg.identity()):
        with pytest.raises(RuntimeError, match="already active"):
            with run_lock(tmp_path, cfg.identity()):
                pass
    checkpoints = LocalCheckpoints(tmp_path)
    checkpoints.path(cfg.identity()).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        evaluate(cfg, manifest, tmp_path)
    Path(manifest["path"]).write_text("bad data")
    with pytest.raises(ValueError, match="checksum"):
        evaluate(cfg, manifest, tmp_path)


def test_registry_rejects_unknown_and_duplicate_components():
    registry = default_registry()
    with pytest.raises(ValueError, match="unknown"):
        registry.factory("detector", "missing")
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("model", "logistic", RiverModel)


def test_budget_resume_and_cap_preserve_prequential_metrics(tmp_path):
    cfg = Config.model_validate({**config().model_dump(), "training_budget": 80})
    fresh = run(cfg, tmp_path / "fresh")
    manifest = prepare(cfg, tmp_path / "resume")
    evaluate(cfg, manifest, tmp_path / "resume", stop_after=73)
    resumed = evaluate(cfg, manifest, tmp_path / "resume")
    assert fresh["prediction_digest"] == resumed["prediction_digest"]
    assert fresh["metrics"] == resumed["metrics"]
    assert fresh["training"] == resumed["training"]
    assert fresh["training"]["used"] <= 80
    assert fresh["training"]["blocked_decisions"] > 0


def test_online_rejects_budget_that_would_skip_available_labels():
    cfg = config(policy="online").model_dump()
    with pytest.raises(ValidationError, match="every post-warmup"):
        Config.model_validate({**cfg, "training_budget": 159})


@pytest.mark.parametrize(
    "change",
    [
        {"typo": True},
        {"mode": "offiline"},
        {"warmup": 10000},
        {"mode": "offline"},
        {"dataset": {"samples": 1_000_000}},
        {"policy": {"min_window": 10000}},
        {"plugins": ["os.system"]},
        {"mode": "offline", "policy": {"name": "none"}, "dataset": {"label_delay": 10000}},
    ],
)
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValidationError):
        Config.model_validate(change)
