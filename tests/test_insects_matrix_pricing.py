"""Tests for Fargate On-Demand pricing, alpha=lambda*custo_retreino_usd, and INSECTS configs."""

from pathlib import Path

import pytest

from continual_bench.application.matrix import ExperimentMatrix
from continual_bench.application.runner import run
from continual_bench.config import Config, load_model
from continual_bench.config.benchmark import AwsCostConfig
from continual_bench.core.policies import PolicyStrategy
from continual_bench.evaluation.paired import compare_paired_ledgers, compare_run_directories
from continual_bench.infrastructure.observation_tracking import parent_observation_points
from continual_bench.retraining.pricing import (
    fargate_hourly_rate_usd,
    fargate_retrain_cost_usd,
    resolve_retrain_alpha,
    summarize_fargate_retraining_cost,
)

CONFIGS = Path(__file__).parents[1] / "configs"


def test_aws_cost_config_aligns_with_documented_fargate_worker():
    cfg = AwsCostConfig()
    assert cfg.region == "us-east-1"
    assert cfg.os_family == "LINUX"
    assert cfg.cpu_architecture == "X86_64"
    assert cfg.vcpus == 2
    assert cfg.memory_gb == 4
    assert cfg.ephemeral_storage_gb == 50
    assert cfg.ephemeral_free_gb == 20
    assert cfg.vcpu_hour_usd == pytest.approx(0.04048)
    assert cfg.memory_gb_hour_usd == pytest.approx(0.004445)
    assert cfg.ephemeral_gb_hour_usd == pytest.approx(0.000111)
    assert cfg.pricing_source == "https://aws.amazon.com/fargate/pricing/"

    hourly_rate, breakdown = fargate_hourly_rate_usd(cfg)
    # 2 * 0.04048 + 4 * 0.004445 + (50 - 20) * 0.000111 = 0.10207 USD/h
    assert hourly_rate == pytest.approx(0.10207)
    assert breakdown["billable_ephemeral_gb"] == pytest.approx(30.0)


def test_fargate_on_demand_dedicated_and_marginal_pricing():
    dedicated_short = fargate_retrain_cost_usd(12.3, billing_mode="dedicated_task")
    assert dedicated_short["billed_seconds"] == 60.0
    assert dedicated_short["custo_retreino_usd"] == pytest.approx(0.10207 * 60.0 / 3600.0)

    dedicated_long = fargate_retrain_cost_usd(60.1, billing_mode="dedicated_task")
    assert dedicated_long["billed_seconds"] == 61.0
    assert dedicated_long["custo_retreino_usd"] == pytest.approx(0.10207 * 61.0 / 3600.0)

    marginal = fargate_retrain_cost_usd(12.5, billing_mode="marginal_in_task")
    assert marginal["billed_seconds"] == pytest.approx(12.5)
    assert marginal["custo_retreino_usd"] == pytest.approx(0.10207 * 12.5 / 3600.0)


def test_cara_upf_and_oracle_share_identical_lambda_pricing_rule():
    params = {"lambda": 500.0, "retrain_duration_seconds": 30.0, "fargate_billing_mode": "dedicated_task"}
    cara = PolicyStrategy("cara", params=params)
    upf = PolicyStrategy("upf", params=params)
    oracle = PolicyStrategy("oracle", params=params)

    expected_cost = 0.10207 * 60.0 / 3600.0
    expected_alpha = 500.0 * expected_cost
    for policy in (cara, upf, oracle):
        assert policy.custo_retreino_usd == pytest.approx(expected_cost)
        assert policy.cost_lambda == pytest.approx(500.0)
        assert policy.alpha == pytest.approx(expected_alpha)

    assert not cara.is_retrospective_reference
    assert not upf.is_retrospective_reference
    assert oracle.is_retrospective_reference is True
    assert "future" in (oracle.future_information_used or "")

    with pytest.raises(ValueError, match="conflicts"):
        resolve_retrain_alpha({"lambda": 100.0, "alpha": 999.0})


def test_zero_retrains_records_zero_estimated_cost_and_preserves_reference():
    resolution = resolve_retrain_alpha({"lambda": 100.0, "fargate_billing_mode": "dedicated_task"})
    summary = summarize_fargate_retraining_cost(
        resolution,
        retrain_count=0,
        operational_execution_mode="batch-fargate-spot",
        operational_aws_cost_usd=0.005,
    )
    assert summary["retrain_count"] == 0
    assert summary["estimated_retraining_cost_usd"] == 0.0
    assert summary["reference_cost_per_action_usd"] == pytest.approx(0.10207 * 60.0 / 3600.0)
    assert summary["operational_execution_mode"] == "batch-fargate-spot"
    assert summary["operational_aws_cost_usd"] == pytest.approx(0.005)


def test_insects_72_and_pilot_configs_validate_and_expand():
    matrix_72 = load_model(ExperimentMatrix, CONFIGS / "matrices/insects-72.yaml")
    plan_72 = matrix_72.plan()
    assert len(plan_72["cases"]) == 72
    assert plan_72["skipped"] == []
    assert {c.dataset.name for c in plan_72["cases"]} == {
        "insects_abrupt_imbalanced",
        "insects_incremental_abrupt_reoccurring_imbalanced",
    }
    assert {c.model.name for c in plan_72["cases"]} == {"logistic", "hoeffding_tree"}
    assert {c.policy.name for c in plan_72["cases"]} == {"none", "online", "drift", "cara", "upf", "oracle"}
    assert {c.dataset.seed for c in plan_72["cases"]} == {42, 43, 44}
    assert {c.mode for c in plan_72["cases"]} == {"incremental"}
    assert all(c.tracking.costs.aws.vcpus == 2 for c in plan_72["cases"])
    assert all(c.tracking.costs.aws.memory_gb == 4 for c in plan_72["cases"])
    assert all(c.tracking.costs.aws.ephemeral_storage_gb == 50 for c in plan_72["cases"])

    pilot = load_model(ExperimentMatrix, CONFIGS / "matrices/insects-pilot.yaml")
    pilot_plan = pilot.plan()
    streaming_cases = [c for c in pilot_plan["cases"] if c.mode == "streaming"]
    incremental_cases = [c for c in pilot_plan["cases"] if c.mode == "incremental"]
    assert {c.policy.name for c in streaming_cases} == {"none", "online", "drift"}
    assert {c.policy.name for c in incremental_cases} == {"none", "online", "drift", "cara", "upf", "oracle"}
    assert all(c.dataset.samples == 100000 for c in pilot_plan["cases"])


def test_end_to_end_run_observations_pricing_regimes_and_paired_comparison(tmp_path):
    base_cfg = {
        "mode": "incremental",
        "dataset": {"name": "sea", "samples": 240, "seed": 42, "label_delay": 0},
        "warmup": 20,
        "batch_size": 20,
        "replay_window": 120,
        "metric_window": 20,
        "tracking": {
            "continual_references": True,
            "adaptation_shadows": True,
            "probe_samples": 8,
            "fargate_billing_mode": "dedicated_task",
            "fargate_retrain_duration_seconds": 10.0,
        },
    }
    none_cfg = Config.model_validate({**base_cfg, "policy": {"name": "none"}})
    oracle_cfg = Config.model_validate({**base_cfg, "policy": {"name": "oracle", "params": {"lambda": 10.0}}})
    none_report = run(none_cfg, tmp_path / "none")
    oracle_report = run(oracle_cfg, tmp_path / "oracle")

    # Zero-retrain policy records 0.0 estimated cost while preserving reference cost per action
    none_fargate = none_report["tracking"]["fargate_retraining"]
    assert none_fargate["retrain_count"] == 0
    assert none_fargate["estimated_retraining_cost_usd"] == 0.0
    assert none_fargate["reference_cost_per_action_usd"] == pytest.approx(0.10207 * 60.0 / 3600.0)
    assert none_report["tracking"]["oracle_reference"]["is_retrospective_reference"] is False

    # Oracle explicitly marked as retrospective reference with future info documented
    oracle_ref = oracle_report["tracking"]["oracle_reference"]
    assert oracle_ref["is_retrospective_reference"] is True
    assert oracle_ref["deployable_online"] is False
    assert "future" in oracle_ref["future_information_used"]

    # Observations_v1 coverage: regimes, prediction_ledger, MLflow resource scalars, unavailable reasons
    assert none_report["tracking"]["prediction_ledger"]["event_id_tagged"] is True
    obs = oracle_report["tracking_observations"]
    assert obs["drift"]["regimes"]["available"] is True
    assert len(obs["drift"]["regimes"]["regimes"]) >= 1
    assert obs["resources"]["fargate_retrain_lambda"] == pytest.approx(10.0)
    assert obs["resources"]["fargate_retrain_alpha"] == pytest.approx(10.0 * 0.10207 * 60.0 / 3600.0)
    # Without lambda on `none`, availability records explicit reason instead of fabricating 0
    assert none_fargate["lambda"] is None
    assert none_fargate["lambda_unavailable_reason"] == "lambda_not_configured"
    assert none_report["tracking_observations"]["resources"]["fargate_retrain_lambda"] is None
    assert "fargate_retrain_lambda" in none_report["tracking_observations"]["resources"]["availability"]

    mlflow_points = dict((name, value) for name, _step, value in parent_observation_points(oracle_report))
    assert "observations_v1/resources/fargate_retrain_alpha" in mlflow_points
    assert "observations_v1/resources/fargate_retrain_reference_cost_usd" in mlflow_points

    # Paired common-event comparison across run directories
    none_dir = tmp_path / "none" / none_report["run_key"]
    oracle_dir = tmp_path / "oracle" / oracle_report["run_key"]
    paired = compare_run_directories({"none": none_dir, "oracle": oracle_dir}, reference="none")
    assert paired["common_events"] == oracle_report["tracking"]["prediction_ledger"]["rows"]
    assert "oracle" in paired["pairwise"]

    # Order or label mismatch fails closed
    bad_ledgers = {
        "a": [{"event_id": "e1", "seq": 0, "y": True, "probability": 0.9}],
        "b": [{"event_id": "e1", "seq": 0, "y": False, "probability": 0.9}],
    }
    with pytest.raises(ValueError, match="label mismatch"):
        compare_paired_ledgers(bad_ledgers)
