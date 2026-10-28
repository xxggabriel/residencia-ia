"""Matrix and publication contracts without requiring the Airflow SDK locally."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from continual_bench.application.config import Config


def jobs_module():
    spec = importlib.util.spec_from_file_location(
        "bench_jobs", Path(__file__).parents[1] / "dags/bench_jobs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("suite,count", [
    ("full", 33), ("streaming", 15), ("incremental", 15), ("offline", 3), ("strategies", 30),
])
def test_matrix_validates_and_covers_strategies(suite, count):
    configs = [Config.model_validate(c) for c in jobs_module().configurations(
        {"samples": 3000, "seeds": [42], "label_delay": 3}, suite
    )]
    assert len(configs) == count
    assert len({c.identity() for c in configs}) == count
    for config in configs:
        if config.mode == "offline":
            assert config.policy.name == config.detector.name == "none"
        if config.policy.name == "drift":
            assert config.detector.name in {"adwin", "ddm"}
        if config.policy.name == "online":
            assert config.policy.action == "learn_one"


def test_job_requires_complete_publication(monkeypatch):
    jobs = jobs_module()
    monkeypatch.setattr(jobs, "request_json", lambda *args: {
        "run_key": "fixture", "status": "succeeded", "receipt": {},
    })
    with pytest.raises(RuntimeError, match="Incomplete publication"):
        jobs.execute_job({})


def test_consolidation_rejects_duplicates_and_retains_protocols():
    jobs = jobs_module()
    result = {"run_key": "fixture", "summary": {"complete": True, "protocol": "temporal_holdout"},
              "receipt": {"mlflow_run_id": "tracked"}}
    assert jobs.consolidate([result])["results"][0]["protocol"] == "temporal_holdout"
    with pytest.raises(ValueError):
        jobs.consolidate([result, result])


def test_controlled_campaign_is_unique_and_preserves_online_budget():
    configs = [Config.model_validate(c) for c in jobs_module().configurations(
        {"samples": 3000, "seeds": [42, 43, 44, 45, 46], "label_delay": 3}, "controlled"
    )]
    assert len(configs) == len({c.identity() for c in configs}) == 261
    online = [c for c in configs if c.policy.name == "online"]
    assert len(online) == 30
    assert all(c.policy.cooldown == 0 and c.training_budget == 2900 for c in online)
    assert len([c for c in configs if c.name.endswith("bridge")]) == 6


def test_versioned_campaign_covers_real_data_without_duplicate_seed_trials():
    configs = [Config.model_validate(c) for c in jobs_module().configurations(
        {"samples": 3000, "seeds": [42, 43, 44, 45, 46], "label_delay": 3}, "versioned"
    )]
    assert len(configs) == len({c.identity() for c in configs}) == 289
    assert {c.dataset.name for c in configs} == {"sea", "agrawal", "stagger", "electricity", "airlines"}
    real = [c for c in configs if c.dataset.name in {"electricity", "airlines"}]
    assert len(real) == 34
    assert {c.dataset.seed for c in real} == {42}


def test_nonlinear_campaign_pairs_all_models_with_v4_data_protocols_and_budgets():
    jobs = jobs_module()
    params = {"samples": 3000, "seeds": [42, 43, 44, 45, 46], "label_delay": 3}
    previous = [Config.model_validate(c) for c in jobs.configurations(params, "versioned")]
    configs = [Config.model_validate(c) for c in jobs.configurations(params, "nonlinear")]
    assert len(configs) == len({c.identity() for c in configs}) == 867
    assert {c.name for c in configs} == {"nonlinear-v5"}
    models = ("logistic", "hoeffding_tree", "knn")

    def protocol_key(config):
        return json.dumps(config.model_dump(mode="json", exclude={"name", "model"}), sort_keys=True)

    expected_protocols = Counter(protocol_key(c) for c in previous)
    for model in models:
        selected = [c for c in configs if c.model.name == model]
        assert len(selected) == 289
        assert Counter(protocol_key(c) for c in selected) == expected_protocols
        for dataset in ("electricity", "airlines"):
            real = [c for c in selected if c.dataset.name == dataset]
            assert len(real) == 17
            assert {c.dataset.seed for c in real} == {42}
    for start in range(0, len(configs), 3):
        batch = configs[start:start + 3]
        assert tuple(c.model.name for c in batch) == models
        assert len({protocol_key(c) for c in batch}) == 1


def test_nonlinear_campaign_records_model_parameters_and_keeps_budget_semantics():
    configs = [Config.model_validate(c) for c in jobs_module().configurations(
        {"samples": 3000, "seeds": [42, 43, 44, 45, 46], "label_delay": 3}, "nonlinear"
    )]
    expected = {
        "logistic": {"learning_rate": 0.05},
        "hoeffding_tree": {"grace_period": 50, "max_depth": 8, "max_size": 8,
                           "memory_estimate_period": 100, "leaf_prediction": "nba"},
        "knn": {"n_neighbors": 11, "window_size": 256, "weighted": True},
    }
    for config in configs:
        params = dict(expected[config.model.name])
        nominal = {"agrawal": ["elevel", "car", "zipcode"], "stagger": ["size", "color", "shape"]}
        if config.model.name == "hoeffding_tree" and config.dataset.name in nominal:
            params["nominal_attributes"] = nominal[config.dataset.name]
        assert config.model.params == params
        assert config.dataset.samples == 3000 and config.dataset.label_delay == 3
        if config.mode == "offline":
            assert config.policy.name == config.detector.name == "none"
            assert config.training_budget is None
        elif config.policy.name == "none":
            assert config.training_budget == 0
        elif config.policy.name == "online":
            assert config.policy.action == "learn_one"
            assert config.policy.cooldown == 0 and config.training_budget == 2900
        else:
            assert config.policy.action == "reset_retrain"
            assert config.training_budget in (2900, 5800)


def test_nonlinear_campaign_fails_early_above_airflow_mapping_limit():
    jobs = jobs_module()
    params = {"samples": 3000, "seeds": [42, 43, 44, 45, 46, 47], "label_delay": 3}
    configs = [Config.model_validate(c) for c in jobs.configurations(params, "nonlinear")]
    assert len(configs) == len({c.identity() for c in configs}) == 1020
    with pytest.raises(ValueError, match="1173 configurations; Airflow max_map_length=1024"):
        jobs.configurations({**params, "seeds": [*params["seeds"], 48]}, "nonlinear")
