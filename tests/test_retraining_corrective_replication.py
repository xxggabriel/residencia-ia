"""Causal held-out decisions and complete per-repetition policy evidence."""

import copy
import json

import numpy as np
import pytest

from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.evaluation import cara_replication as cara
from continual_bench.evaluation import retraining_replication as upf
from continual_bench.evaluation.retraining_policy_tracking import policy_report, policy_tracking
from continual_bench.infrastructure.metric_tracking import metric_points


class LightweightLearner:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, index, x, y):
        return float(x[:, 0].mean())

    def losses(self, cutoff, x, y):
        return (x[:, 0] > cutoff) != y


def candidate_stub(name, seed, *args, **kwargs):
    return {
        "run_key": f"{name}-{seed}-{kwargs['protocol']}",
        "contract": {"environment": {"fixture": True}},
        "artifact_mode": kwargs["artifact_mode"],
    }


@pytest.mark.parametrize("mode", ["individual", "same-batch", "next-batch"])
def test_trial_heldout_label_flip_preserves_cara_decisions(monkeypatch, mode):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(upf, "XGBoostLearner", LightweightLearner)
    monkeypatch.setattr(upf, "candidate_report", candidate_stub)
    rng = np.random.default_rng(5)
    batches = [data.split_batch(x, y, rng) for x, y in data.upf_gauss(15, size=48, seed=7)]
    flipped = [(x, y, xt, 1 - yt) for x, y, xt, yt in batches]
    def selected_batches(*args, **kwargs):
        assert kwargs["airlines_disk_cache"] is False
        return batches

    monkeypatch.setattr(upf, "trial_batches", selected_batches)
    options = {
        "compare_oracles": mode != "individual",
        "decision_timing": "next-batch" if mode == "next-batch" else "same-batch",
        "rules": ("cara", "cara_cumulative", "cara_periodic"),
        "metrics_only": True,
    }
    original = upf.run_trial("electricity", 0, **options)
    monkeypatch.setattr(upf, "trial_batches", lambda *a, **k: flipped)
    modified = upf.run_trial("electricity", 0, **options)
    np.testing.assert_allclose(
        np.asarray(original["staleness_matrix"], dtype=float),
        np.asarray(modified["staleness_matrix"], dtype=float),
    )
    for rule in ("cara", "cara_cumulative", "cara_periodic"):
        for old, new in zip(original["rules"][rule]["curve"], modified["rules"][rule]["curve"], strict=True):
            assert old["retrain_points"] == new["retrain_points"]
            assert old["parameters"] == new["parameters"]
            assert old["average_accuracy"] == pytest.approx(1 - new["average_accuracy"])
    assert original["candidate_tracking"]["artifact_mode"] == "metrics_only"
    assert original["policy_tracking"]["contract"]["settings"]["separate_query_split"]


def test_individual_export_contains_every_seed_and_recomputable_statistics(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(upf, "XGBoostLearner", LightweightLearner)
    monkeypatch.setattr(upf, "candidate_report", candidate_stub)
    rng = np.random.default_rng(5)
    batches = [data.split_batch(x, y, rng) for x, y in data.upf_gauss(15, size=48, seed=7)]
    monkeypatch.setattr(upf, "trial_batches", lambda *a, **k: batches)
    report = upf.replicate("gauss", trials=2, rules=("cara",), metrics_only=True)
    details = report["trials_detail"]
    assert [trial["seed"] for trial in details] == [0, 1]
    assert report["dispersion"] == {"n": 2, "ddof": 0, "divisor": 2, "kind": "population_std"}
    for rule, summary in report["summary"].items():
        values = [trial["summary"][rule]["auc"] for trial in details]
        assert summary["auc_mean"] == pytest.approx(np.mean(values))
        assert summary["auc_std"] == pytest.approx(np.std(values, ddof=0))
        for trial in details:
            assert trial["curves"][rule][0]["alpha"] == 0
            assert isinstance(trial["curves"][rule][0]["retrain_points"], list)
            assert trial["performance_matrix"]["batches"] == 15
            assert trial["policy_tracking"]["recorded_metric_points"] > 0
    json.dumps(report, allow_nan=False)


def test_cara_export_keeps_seed_schedules_and_parameters(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(cara, "OFFLINE", 2)
    monkeypatch.setattr(cara, "batches_for", lambda *a, **k: ([None] * 6, None, 20.0))

    def phase(batches, learner, *args, **kwargs):
        size = len(batches)
        return cara.Phase(
            np.zeros((size, size)),
            np.full((size, size), 0.75),
            [],
            {"run_key": f"candidate-{learner.seed}-{size}", "contract": {"environment": {}}},
        )

    def evaluate(offline, online, kappa, seed, *args):
        return {
            rule: {
                "strategy_cost": seed + kappa,
                "strategy_scpe": seed * 2 + kappa,
                "average_accuracy": 0.7 + seed / 10,
                "retrains": seed,
                "retrain_points": [0, 2] if seed else [0],
                "served_by": [0, 0, 0, 2] if seed else [0, 0, 0, 0],
                "parameters": {"threshold": seed + kappa},
            }
            for rule in cara.RULES
        }

    monkeypatch.setattr(cara, "build_phase", phase)
    monkeypatch.setattr(cara, "evaluate_kappa", evaluate)
    report = cara.replicate("electricity", seeds=2, kappa_points=3, kappa_max=1, metrics_only=True)
    assert report["protocol"] == "CARA-authors-historical-v1"
    assert len(report["trials_detail"]) == 2
    first, second = [row["curves_by_policy"]["cara"][1] for row in report["trials_detail"]]
    assert first["retrain_points"] == [0] and second["retrain_points"] == [0, 2]
    assert (
        report["curves"]["cara"]["strategy_cost"][1] == (first["strategy_cost"] + second["strategy_cost"]) / 2
    )
    assert second["parameters"]["threshold"] == 1.5
    assert second["served_by"] == [0, 0, 0, 2]
    json.dumps(report, allow_nan=False)


def test_policy_tracker_publishes_analysis_after_evaluation(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "stub")
    monkeypatch.setenv("BENCH_STATE", str(tmp_path))
    analysis = {
        "setup": {"offline_batches": 2, "batches": 4},
        "alphas": [0.0, 0.5],
        "rules": {
            "cara": {
                "auc": 0.4,
                "curve": [
                    {"alpha": 0.0, "cost": 1.0, "scpe": None, "retrain_points": [2]},
                    {"alpha": 0.5, "cost": 0.6, "scpe": 2.0, "retrain_points": []},
                ],
            }
        },
    }
    captured = []

    def publish(directory, report, **kwargs):
        captured.append(copy.deepcopy(report))
        assert json.loads((directory / "report.json").read_text()) == report
        return {"complete": True, "mlflow_run_id": "policy-run"}

    monkeypatch.setattr("continual_bench.infrastructure.benchmark_tracking.publish_local_benchmark", publish)
    result = policy_report(
        "fixture",
        0,
        protocol="test",
        analysis=analysis,
        candidate_reports=[{"run_key": "candidate", "contract": {"environment": {}}}],
        settings={"separate_query_split": True},
        artifact_mode="metrics_only",
    )
    assert result["mlflow"]["mlflow_run_id"] == "policy-run"
    assert captured[0]["analysis"] == analysis
    assert captured[0]["artifact_mode"] == "metrics_only"
    points = metric_points(policy_tracking(analysis))
    assert ("tracking_v1/policy/cara/cost", 1, 0.6) in points
    assert ("tracking_v1/policy/cara/schedule/alpha_0", 2, 1) in points
    assert not list(tmp_path.rglob("*.pkl")) and not list(tmp_path.rglob("*.ubj"))


@pytest.mark.parametrize("failure_mode", ["exception", "incomplete_receipt"])
def test_metrics_only_publication_failure_is_explicit_and_retryable(monkeypatch, tmp_path, failure_mode):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "stub")
    monkeypatch.setenv("BENCH_STATE", str(tmp_path))

    def fail(*args, **kwargs):
        if failure_mode == "incomplete_receipt":
            return {"complete": False}
        raise OSError("private://credentials@example")

    monkeypatch.setattr("continual_bench.infrastructure.benchmark_tracking.publish_local_benchmark", fail)
    arguments = {
        "protocol": "test",
        "analysis": {
            "setup": {"offline_batches": 2, "batches": 4},
            "alphas": [0.0],
            "rules": {"cara": {"auc": 0.1, "curve": [{"cost": 0.1, "retrain_points": []}]}},
        },
        "candidate_reports": [{"run_key": "candidate", "contract": {"environment": {}}}],
        "settings": {},
        "artifact_mode": "metrics_only",
    }
    with pytest.raises(RuntimeError, match="publication incomplete") as failure:
        policy_report("fixture", 0, **arguments)
    assert "credentials" not in str(failure.value)
    assert len(list(tmp_path.rglob("report.json"))) == 1
    monkeypatch.setattr(
        "continual_bench.infrastructure.benchmark_tracking.publish_local_benchmark",
        lambda *a, **k: {"complete": True, "mlflow_run_id": "retry"},
    )
    assert policy_report("fixture", 0, **arguments)["mlflow"]["mlflow_run_id"] == "retry"
