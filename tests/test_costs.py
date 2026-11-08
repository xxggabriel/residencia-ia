import pytest

from continual_bench.evaluation.costs import account, error_units, profile


def test_normalized_regression_and_classification():
    costs = profile({"regression_scale": 10, "regression_error": 2, "classification_error": 3})
    assert error_units("regression", 15, 5, costs) == 2
    assert error_units("multiclass", "a", "b", costs) == 3
    assert error_units("multiclass", "a", "a", costs) == 0


def test_missing_priced_timer_is_unavailable():
    costs = account(profile({"train_second": 2}), errors=1, presentations=10, updates=1,
                    train_seconds=None, predict_seconds=0)
    assert costs["metrics"]["cost_total"] is None
    assert costs["consumption"]["train_seconds"] is None


@pytest.mark.parametrize("values", [{"regression_scale": 0}, {"train_sample": -1},
                                    {"classification_error": float("nan")}])
def test_invalid_profiles_are_rejected(values):
    with pytest.raises(ValueError):
        profile(values)


@pytest.mark.parametrize("policy", ["none", "online", "periodic", "drift", "performance", "cost_heuristic"])
def test_main_flow_records_default_costs_for_every_policy(tmp_path, policy):
    from continual_bench.application.runner import run
    from continual_bench.config import Config

    action = "learn_one" if policy == "online" else "reset_retrain"
    config = Config.model_validate({"dataset": {"samples": 100}, "warmup": 10,
                                   "policy": {"name": policy,
                                              "action": action}})
    report = run(config, tmp_path)
    costs = report["tracking"]["costs"]
    assert costs["metrics"]["cost_total"] is not None
    assert costs["settings"]["train_sample"] == 0.001
    assert costs["consumption"]["training_presentations"] >= 10
    assert report["tracking"]["metrics"]["cost_total"] == costs["metrics"]["cost_total"]
