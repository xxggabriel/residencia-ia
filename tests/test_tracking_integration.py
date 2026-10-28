"""Real local runs exercise checkpoint, prediction, lineage and instrumentation parity."""

import hashlib
import json
from pathlib import Path

import pytest

from continual_bench.adapters.local import LocalCheckpoints
from continual_bench.adapters.river import NullDetector, RiverModel, SyntheticSource
from continual_bench.adapters.versioning import load_version
from continual_bench.application.config import Config
from continual_bench.application.runner import evaluate, prepare, run
from continual_bench.domain.engine import Engine
from continual_bench.domain.observability import OnlineObservability
from continual_bench.domain.policies import PolicyStrategy


def small_config(mode="streaming", tracking=None, policy="online"):
    return Config.model_validate({
        "name": "tracking-regression",
        "mode": mode,
        "dataset": {"name": "sea", "samples": 120, "label_delay": 3},
        "warmup": 20,
        "metric_window": 20,
        "batch_size": 8,
        "replay_window": 30,
        "checkpoint_every": 20,
        "training_budget": None if mode == "offline" else 100,
        "detector": {"name": "none"},
        "policy": {"name": "none" if mode == "offline" else policy,
                   "action": "learn_one" if policy == "online" and mode != "offline" else "reset_retrain",
                   "every": 20, "cooldown": 0, "min_samples": 20, "min_window": 10},
        "tracking": tracking,
    })


def without_runtime(value):
    if isinstance(value, dict):
        return {key: without_runtime(item) for key, item in value.items()
                if "seconds" not in key and key not in {"instrumentation", "memory_samples"}}
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def prediction_evidence(directory):
    fields = ("event_id", "seq", "timestamp", "available_at", "released_at", "probability", "label",
              "trained_labels_at_prediction", "comparison_id", "previous_probability")
    return [{key: row[key] for key in fields}
            for row in map(json.loads, (directory / "predictions.jsonl").read_text().splitlines())]


def test_all_867_frozen_configuration_dumps_and_identities_remain_identical():
    manifest = json.loads((Path(__file__).parents[1] / "configs/campaign-nonlinear-v5.json").read_text())
    assert len(manifest["configurations"]) == 867
    for row in manifest["configurations"]:
        config = Config.model_validate(row["config"])
        assert config.tracking is None
        assert config.model_dump(mode="json") == row["config"]
        assert json.loads(config.model_dump_json()) == row["config"]
        assert config.identity() == row["run_key"]
        assert "tracking" not in config.model_dump()


def test_explicit_tracking_creates_new_identity_without_changing_default_dump():
    historical = small_config()
    explicit = Config.model_validate({**historical.model_dump(mode="json"), "tracking": {}})
    assert historical.model_dump(mode="json") == Config.model_validate(
        {**historical.model_dump(mode="json"), "tracking": None}
    ).model_dump(mode="json")
    assert explicit.identity() != historical.identity()
    assert explicit.model_dump(mode="json")["tracking"]["version"] == "residency-metrics-v1"


@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_tracking_local_run_has_exact_predictions_counts_and_reference_lineage(mode, tmp_path):
    config = small_config(mode, {"probe_samples": 8, "adaptation_shadows": True})
    report = run(config, tmp_path)
    directory = tmp_path / config.identity()
    tracking = report["tracking"]
    ledger = prediction_evidence(directory)
    expected = 48 if mode == "offline" else 100
    assert report["metrics"]["n"] == len(ledger) == expected
    assert tracking["prediction_ledger"]["rows"] == expected
    assert hashlib.sha256((directory / "predictions.jsonl").read_bytes()).hexdigest() == (
        tracking["prediction_ledger"]["sha256"]
    )
    assert all(row["released_at"] >= row["available_at"] for row in ledger)
    assert [row["seq"] for row in ledger] == list(range(72 if mode == "offline" else 20, 120))
    metrics = tracking["metrics"]
    assert metrics["training_presentations"] == (69 if mode == "offline" else 120)
    assert metrics["training_unique_examples"] == metrics["training_presentations"]
    assert metrics["training_repeated_presentations"] == 0
    if mode == "offline":
        assert metrics["training_used_post_warmup"] == 0
        assert tracking["continual"] is None
        assert metrics["cl_average_accuracy"] is None
        assert not (directory / "references").exists()
        assert report["split"]["train"] == [0, 69]
    else:
        assert metrics["training_used_post_warmup"] == 100
        assert all(row["trained_labels_at_prediction"] <= row["seq"] - 3 for row in ledger)
        continual = tracking["continual"]
        assert [(row["task"], row["end_seq"]) for row in continual["matrix"]] == [
            (0, 39), (1, 79), (2, 119),
        ]
        assert continual["released_labels"] == 120
        assert len(tracking["reference_models"]) == 6
        for reference in tracking["reference_models"]:
            path = directory / "references" / f"{reference['name']}-{reference['task']}"
            assert (path / "model.pkl").exists()
            index = json.loads((path / "model-versions.json").read_text())
            entry = index["versions"][0]
            assert entry["dataset_version"] == report["dataset"]["sha256"]
            assert entry["code_sha256"] == report["environment"]["code_sha256"]
            assert entry["training"]["seq_range"] == reference["seq_range"]
            restored = load_version(path, reference["metadata"]["final_version_id"])
            assert 0 <= restored.predict({"0": 1, "1": 2, "2": 3}) <= 1
    assert run(config, tmp_path) == report
    json.dumps(tracking, allow_nan=False)


@pytest.mark.parametrize("mode", ["streaming", "incremental"])
def test_interrupted_tracking_run_matches_core_metrics_predictions_and_cl_matrix(mode, tmp_path):
    config = small_config(mode, {"probe_samples": 8, "adaptation_shadows": True})
    fresh_root, resume_root = tmp_path / "fresh", tmp_path / "resumed"
    fresh = run(config, fresh_root)
    manifest = prepare(config, resume_root)
    progress = evaluate(config, manifest, resume_root, stop_after=51)
    assert progress["complete"] is False
    snapshot = LocalCheckpoints(resume_root).load(config.identity())
    assert snapshot["state"].observer.continual.next_seq == snapshot["state"].labeled
    resumed = evaluate(config, manifest, resume_root)
    assert fresh["prediction_digest"] == resumed["prediction_digest"]
    assert fresh["metrics"] == resumed["metrics"]
    assert fresh["windows"] == resumed["windows"]
    assert fresh["training"] == resumed["training"]
    assert without_runtime(fresh["tracking"]["continual"]) == without_runtime(
        resumed["tracking"]["continual"]
    )
    assert prediction_evidence(fresh_root / config.identity()) == prediction_evidence(
        resume_root / config.identity()
    )
    assert without_runtime(fresh["tracking"]["adaptation"]) == without_runtime(
        resumed["tracking"]["adaptation"]
    )


@pytest.mark.parametrize("mode", ["streaming", "incremental"])
def test_reference_and_shadow_toggles_do_not_change_main_predictions_or_budget(mode, tmp_path):
    disabled = small_config(mode, {"probe_samples": 8, "continual_references": False,
                                   "adaptation_shadows": False})
    enabled = small_config(mode, {"probe_samples": 8, "continual_references": True,
                                  "adaptation_shadows": True})
    before, after = run(disabled, tmp_path), run(enabled, tmp_path)
    assert before["dataset"]["sha256"] == after["dataset"]["sha256"]
    assert before["prediction_digest"] == after["prediction_digest"]
    assert before["metrics"] == after["metrics"]
    assert before["windows"] == after["windows"]
    assert before["training"] == after["training"]
    assert without_runtime(before["updates"]) == without_runtime(after["updates"])
    assert before["model_versions"]["state_sha256"] == after["model_versions"]["state_sha256"]
    assert before["tracking"]["metrics"]["cl_intransigence"] is None
    assert after["tracking"]["metrics"]["cl_intransigence"] is not None
    assert before["tracking"].get("reference_models") == []
    assert len(after["tracking"]["reference_models"]) == 6
    assert before["tracking"]["adaptation"] == []
    assert after["tracking"]["adaptation"]


@pytest.mark.parametrize("size", [1, 8])
def test_engine_observer_itself_does_not_change_policy_or_learning(size):
    from continual_bench.application.config import TrackingConfig

    settings = TrackingConfig(adaptation_shadows=True, continual_references=False).model_dump(mode="json")
    source = SyntheticSource("sea", 120, 42, 3, 1000)
    events = list(source.events())

    def engine(observer):
        return Engine(RiverModel, NullDetector,
                      PolicyStrategy("periodic", every=20, cooldown=0, min_samples=20, min_window=10),
                      warmup=20, window=30, metric_window=20, training_budget=80, observer=observer)

    baseline, observed = engine(None), engine(OnlineObservability(settings))
    for start in range(0, len(events), size):
        baseline.observe_batch(events[start:start + size])
        observed.observe_batch(events[start:start + size])
    baseline.finish()
    observed.finish()
    for field in ("metrics", "windows", "training", "prediction_digest"):
        assert baseline.report()[field] == observed.report()[field]
    assert without_runtime(baseline.updates) == without_runtime(observed.updates)
    assert observed.training_used <= 80
    assert len(observed.observer.predictions) == 100
    assert observed.observer.warmup_train_samples == 20


def test_episode_probe_evaluation_occurs_after_the_boundary_label_update(tmp_path):
    config = small_config(tracking={"probe_samples": 8})
    report = run(config, tmp_path)
    state = LocalCheckpoints(tmp_path).load(config.identity())["state"]
    final = report["tracking"]["continual"]["matrix"][-1]
    for task, samples in state.observer.continual.probes.items():
        accuracy = sum((state.model.predict(dict(x)) >= 0.5) == y for x, y in samples) / len(samples)
        assert final["accuracies"][str(task)] == accuracy
    assert state.observer.continual.reference_models[-1]["seq_range"] == [0, 120]


def test_offline_drain_ledger_has_no_training_suffix_leakage(tmp_path):
    config = small_config("offline", {"probe_samples": 8})
    report = run(config, tmp_path)
    ledger = prediction_evidence(tmp_path / config.identity())
    assert {row["trained_labels_at_prediction"] for row in ledger} == {69}
    assert all(row["comparison_id"] is None for row in ledger)
    assert report["tracking"]["metrics"]["training_presentations"] == 69
    assert report["tracking"]["metrics"]["training_budget_blocked_decisions"] is None


def test_config_reference_budget_does_not_limit_the_main_online_training(tmp_path):
    config = small_config(tracking={"probe_samples": 8, "reference_budget": 0})
    report = run(config, tmp_path)
    assert report["training"]["used"] == 100
    tracking = report["tracking"]
    assert tracking["reference_models"] == []
    assert tracking["metrics"]["cl_average_accuracy"] is not None
    assert tracking["metrics"]["cl_forward_transfer"] is None
    assert tracking["availability"]["cl_forward_transfer"]["reason"] == "reference_budget_exceeded"
