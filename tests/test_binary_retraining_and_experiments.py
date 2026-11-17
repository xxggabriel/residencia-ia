"""Tests for CARA/UPF/Oracle execution, metric tracking, online streaming, and experiment separation."""

from unittest.mock import Mock

import pytest

from continual_bench.application.cloud import planning
from continual_bench.application.cloud.contracts import BenchmarkWorkload, CampaignRequest, RunUnit
from continual_bench.application.cloud.fixtures import publish_fixture_release
from continual_bench.application.matrix import ExperimentMatrix
from continual_bench.application.runner import evaluate, prepare
from continual_bench.application.runner import run as run_binary
from continual_bench.config import Config
from continual_bench.core.contracts import Event
from continual_bench.evaluation.binary_retraining import FeatureCodec
from continual_bench.infrastructure.aws.benchmark_tracking import aws_metric_points
from continual_bench.infrastructure.aws.matrix_tracking import MatrixMlflowRun
from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker
from continual_bench.infrastructure.object_store import LocalObjectStore


@pytest.mark.parametrize("policy_name", ["cara", "upf", "oracle"])
def test_binary_runner_executes_batch_retraining_policies(policy_name, tmp_path):
    config = Config.model_validate(
        {
            "mode": "incremental",
            "warmup": 20,
            "metric_window": 20,
            "batch_size": 25,
            "replay_window": 100,
            "dataset": {"name": "sea", "samples": 200, "seed": 7},
            "model": {"name": "logistic"},
            "detector": {"name": "none"},
            "policy": {"name": policy_name},
        }
    )
    report = run_binary(config, str(tmp_path))
    assert report["complete"] is True
    assert "retraining" in report
    assert {"cara", "upf", "oracle"} <= set(report["retraining"]["rules"])
    metrics = report["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    for key in (
        "retraining_c_hat",
        "retraining_oracle_c_hat",
        "retraining_cara_auc",
        "retraining_upf_auc",
        "retraining_oracle_auc",
    ):
        assert metrics[key] is not None


def test_batch_policy_requires_incremental_mode():
    with pytest.raises(ValueError, match="requires incremental mode"):
        Config.model_validate(
            {
                "mode": "streaming",
                "dataset": {"name": "sea", "samples": 200},
                "policy": {"name": "cara"},
            }
        )


@pytest.mark.parametrize(
    "mode,policy",
    [
        ("streaming", {"name": "online", "action": "learn_one"}),
        ("incremental", {"name": "drift"}),
        ("incremental", {"name": "periodic", "every": 40, "min_samples": 20, "min_window": 20}),
    ],
)
def test_non_batch_policies_still_track_cara_upf_oracle_metrics(mode, policy, tmp_path):
    config = Config.model_validate(
        {
            "mode": mode,
            "warmup": 20,
            "metric_window": 20,
            "batch_size": 25,
            "replay_window": 100,
            "dataset": {"name": "sea", "samples": 200, "seed": 11},
            "model": {"name": "logistic"},
            "detector": {"name": "adwin"} if policy["name"] == "drift" else {"name": "none"},
            "policy": policy,
        }
    )
    report = run_binary(config, str(tmp_path))
    assert "retraining" in report
    metrics = report["metrics"]
    for prefix in ("cara", "upf", "oracle"):
        assert metrics[f"retraining_{prefix}_auc"] is not None
        assert metrics[f"retraining_{prefix}_c_hat"] is not None
        assert metrics[f"retraining_{prefix}_retrains"] is not None
    mlflow_keys = {item["key"] for item in aws_metric_points(report)}
    assert "summary/retraining_cara_auc" in mlflow_keys
    assert "summary/retraining_upf_auc" in mlflow_keys
    assert "summary/retraining_oracle_auc" in mlflow_keys


def test_feature_codec_preserves_categorical_and_missing_values():
    events = [
        Event(
            event_id="e0",
            seq=0,
            timestamp=0,
            available_at=0,
            x={"airline": "AA", "delay": 5.0, "flag": True},
            y=False,
        ),
        Event(
            event_id="e1",
            seq=1,
            timestamp=1,
            available_at=1,
            x={"airline": "DL", "delay": None, "flag": False},
            y=True,
        ),
    ]
    codec = FeatureCodec(events)
    x_arr, y_arr = codec.encode(events)
    decoded = codec.decode(x_arr)
    assert y_arr.tolist() == [0, 1]
    assert decoded[0] == {"airline": "AA", "delay": 5.0, "flag": True}
    assert decoded[1] == {"airline": "DL", "flag": False}


def test_benchmark_workload_online_streaming_needs_no_policy_and_incremental_runs_policies(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id="cara-upf-oracle-test",
        matrix=BenchmarkWorkload(
            evaluator="binary",
            datasets=("sea",),
            models=("logistic",),
            policies=("cara", "upf", "oracle", "drift"),
            detectors=("adwin", "ddm"),
            modes=("streaming", "incremental"),
            samples=300,
            seeds=(42,),
        ),
    )
    plan = planning.build_plan(request, release, project_root=".")
    units = plan["units"]
    streaming_units = [u for u in units if u["labels"]["mode"] == "streaming"]
    incremental_units = [u for u in units if u["labels"]["mode"] == "incremental"]

    # Online streaming execution: single unit per (dataset, model, seed) with no retraining policy.
    assert len(streaming_units) == 1
    assert streaming_units[0]["labels"]["policy"] == "online"
    assert streaming_units[0]["labels"]["detector"] == "none"

    # Incremental execution: cara, upf, oracle (detector=none) + drift (adwin, ddm).
    inc_pairs = sorted((u["labels"]["policy"], u["labels"]["detector"]) for u in incremental_units)
    assert inc_pairs == [
        ("cara", "none"),
        ("drift", "adwin"),
        ("drift", "ddm"),
        ("oracle", "none"),
        ("upf", "none"),
    ]


def test_experiment_matrix_skips_batch_policies_in_streaming_mode():
    plan = ExperimentMatrix.model_validate(
        {
            "name": "batch-skip",
            "datasets": [{"name": "sea", "samples": 200}],
            "models": [{"name": "logistic"}],
            "strategies": [
                {"detector": {"name": "none"}, "policy": {"name": "online", "action": "learn_one"}},
                {"detector": {"name": "none"}, "policy": {"name": "cara"}},
                {"detector": {"name": "none"}, "policy": {"name": "upf"}},
                {"detector": {"name": "none"}, "policy": {"name": "oracle"}},
            ],
            "modes": ["streaming", "incremental"],
        }
    ).plan()
    assert sorted((c.mode, c.policy.name) for c in plan["cases"]) == [
        ("incremental", "cara"),
        ("incremental", "online"),
        ("incremental", "oracle"),
        ("incremental", "upf"),
        ("streaming", "online"),
    ]


def test_mlflow_experiments_are_logically_separated_by_policy():
    git_sha = "075a88cd957f3ab64ee930b88b3753798d9889a6"
    experiments = set()
    for policy in ("cara", "upf", "oracle", "online", "drift"):
        unit = RunUnit(
            run_id="1424bfc8ed2cbd966ae1",
            campaign_id="energy-airlines",
            kind="matrix_case",
            params={"evaluator": "binary"},
            contract={"release": {"git_sha": git_sha}},
            contract_sha256="a" * 64,
            job_name=f"cb-energy-airlines-{policy}",
            unit_key=f"campaigns/energy-airlines/units/{policy}.json",
            labels={"dataset": "electricity", "model": "logistic", "mode": "incremental", "policy": policy},
        )
        run = MatrixMlflowRun(unit, "job-1", "job", "http://fake", log=Mock())
        assert run.experiment_name == f"075a88c-continual-bench-matrix-{policy}"
        experiments.add(run.experiment_name)
    assert len(experiments) == 5

    zia_tracker = SageMakerMlflowTracker(
        tracking_uri="http://fake",
        dataset="fraud",
        policy="cara",
        model_name="mlp",
        run_id="run-1",
        git_sha=git_sha,
    )
    assert zia_tracker.experiment_name == "075a88c-continual-bench-zia-cara"


def test_benchmark_workload_matrix_id_reflects_incremental_policies(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    release = publish_fixture_release(store)
    req_cara = CampaignRequest(
        campaign_id="matrix-id-test",
        matrix=BenchmarkWorkload(
            evaluator="binary",
            datasets=("sea",),
            models=("logistic",),
            policies=("cara",),
            modes=("streaming", "incremental"),
            samples=200,
            seeds=(42,),
        ),
    )
    req_upf = CampaignRequest(
        campaign_id="matrix-id-test",
        matrix=BenchmarkWorkload(
            evaluator="binary",
            datasets=("sea",),
            models=("logistic",),
            policies=("upf",),
            modes=("streaming", "incremental"),
            samples=200,
            seeds=(42,),
        ),
    )
    plan_cara = planning.build_plan(req_cara, release, project_root=".")
    plan_upf = planning.build_plan(req_upf, release, project_root=".")
    assert plan_cara["units"][0]["contract"]["matrix"] != plan_upf["units"][0]["contract"]["matrix"]


def test_batch_retraining_stop_after_returns_incomplete_then_resumes_full_run(tmp_path):
    config = Config.model_validate(
        {
            "mode": "incremental",
            "warmup": 20,
            "metric_window": 20,
            "batch_size": 25,
            "replay_window": 100,
            "dataset": {"name": "sea", "samples": 200, "seed": 7},
            "model": {"name": "logistic"},
            "detector": {"name": "none"},
            "policy": {"name": "cara"},
        }
    )
    manifest = prepare(config, str(tmp_path))
    partial = evaluate(config, manifest, str(tmp_path), stop_after=100)
    assert partial == {"complete": False, "observed": 100, "run_key": config.identity()}

    full = evaluate(config, manifest, str(tmp_path))
    assert full["complete"] is True
    assert full["observed"] == 200
    assert full["metrics"]["retraining_cara_auc"] is not None
