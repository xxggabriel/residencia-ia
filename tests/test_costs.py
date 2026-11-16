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


@pytest.mark.parametrize("policy", ["none", "online", "periodic", "drift", "performance"])
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
    directory = tmp_path / report["run_key"]
    stored_bytes = (directory / "models.bin").stat().st_size + (directory / "model.pkl").stat().st_size
    aws = costs["aws"]
    assert aws["consumption"]["model_storage_bytes"] == stored_bytes
    assert (directory / "artifacts/plots/costs/aws_usd.svg").is_file()
    assert aws["metrics"]["cost_model_storage_usd"] == pytest.approx(stored_bytes / 1024**3 * 0.023)
    assert report["tracking"]["metrics"]["aws_cost_total_usd"] == aws["metrics"]["cost_total_usd"]
    assert aws["consumption"]["processing_seconds"] == pytest.approx(
        costs["consumption"]["train_seconds"] + costs["consumption"]["prediction_seconds"])
    assert costs["settings"]["train_sample"] == 0.001
    assert costs["consumption"]["training_presentations"] >= 10
    assert report["tracking"]["metrics"]["cost_total"] == costs["metrics"]["cost_total"]


@pytest.mark.parametrize("seconds,retention,expected_billed", [(0.1, 30, 60), (3600, 15, 3600),
                                                            (60.1, 60, 61)])
def test_aws_prices_provisioned_cpu_ram_and_actual_storage(seconds, retention, expected_billed):
    settings = profile({"aws": {"vcpus": 2, "memory_gb": 4, "retention_days": retention}})
    result = account(settings, errors=99, presentations=10, updates=1,
                     train_seconds=seconds / 2, predict_seconds=seconds / 2,
                     model_storage_bytes=3 * 1024**3)
    aws = result["aws"]
    assert aws["consumption"]["billed_compute_seconds"] == expected_billed
    assert aws["metrics"]["cost_cpu_usd"] == pytest.approx(2 * expected_billed / 3600 * 0.04048)
    assert aws["metrics"]["cost_ram_usd"] == pytest.approx(4 * expected_billed / 3600 * 0.004445)
    assert aws["metrics"]["cost_model_storage_usd"] == pytest.approx(3 * retention / 30 * 0.023)
    assert aws["metrics"]["cost_total_usd"] == pytest.approx(
        2 * expected_billed / 3600 * 0.04048 + 4 * expected_billed / 3600 * 0.004445
        + 3 * retention / 30 * 0.023)
    assert result["metrics"]["cost_total"] == 99  # Scenario error units stay separate from USD.
    assert aws["settings"]["price_date"] == "2026-11-08"


@pytest.mark.parametrize("train,predict,storage,missing", [
    (None, 1, 0, "cost_cpu_usd"), (1, None, 0, "cost_ram_usd"),
    (1, 1, None, "cost_model_storage_usd"),
])
def test_aws_missing_consumption_never_becomes_free(train, predict, storage, missing):
    result = account(profile(), errors=0, presentations=0, updates=0,
                     train_seconds=train, predict_seconds=predict, model_storage_bytes=storage)
    assert result["aws"]["metrics"][missing] is None
    assert result["aws"]["metrics"]["cost_total_usd"] is None


@pytest.mark.parametrize("aws", [{"memory_gb": -1}, {"vcpus": 0}, {"retention_days": -1},
                                 {"storage_gb_month_usd": float("nan")}, {"memory_gb": 1},
                                 {"billing_month_days": 0}, {"region": "sa-east-1"}])
def test_aws_invalid_profiles_are_rejected(aws):
    with pytest.raises(ValueError):
        profile({"aws": aws})


def test_aws_can_be_disabled_without_changing_scenario_formula():
    result = account(profile({"aws": None, "train_sample": 2, "update": 3}),
                     errors=1, presentations=10, updates=2, train_seconds=1, predict_seconds=1)
    assert "aws" not in result
    assert result["metrics"]["cost_total"] == 27


def test_aws_configuration_changes_run_identity():
    from continual_bench.config import Config

    first = Config.model_validate({"tracking": {"costs": {"aws": {"retention_days": 30}}}})
    second = Config.model_validate({"tracking": {"costs": {"aws": {"retention_days": 60}}}})
    assert first.identity() != second.identity()
