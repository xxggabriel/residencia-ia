"""Causal held-out decisions and complete per-repetition policy evidence."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from continual_bench.core.metrics import Metrics
from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.evaluation import cara_replication as cara
from continual_bench.evaluation import retraining_replication as upf
from continual_bench.evaluation.retraining_policy_tracking import policy_report, policy_tracking
from continual_bench.infrastructure.metric_tracking import metric_points


def real_source_fixture(directory, dataset, count=107):
    path = directory / f"{dataset}.arff"
    if dataset == "electricity":
        text = "@relation fixture\n@attribute a numeric\n@data\n"
        text += "\n".join(
            f"0,1,{i},{i + 1},0.1,0.2,0.3,0.4,{'UP' if i % 2 else 'DOWN'}" for i in range(count)
        )
        coordinate = 0
    else:
        fields = (
            ("Airline", "{AA,BB}"),
            ("Flight", "numeric"),
            ("AirportFrom", "{X,Y}"),
            ("AirportTo", "{Y,Z}"),
            ("DayOfWeek", "{1,2}"),
            ("Time", "numeric"),
            ("Length", "numeric"),
            ("Delay", "{0,1}"),
        )
        text = (
            "@relation fixture\n" + "".join(f"@attribute {key} {kind}\n" for key, kind in fields) + "@data\n"
        )
        text += "\n".join(f"AA,{i},X,Y,1,{i},200,{i % 2}" for i in range(count))
        coordinate = 1
    path.write_text(text + "\n")
    return path, coordinate


@pytest.mark.parametrize("dataset", ["electricity", "airlines"])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_full_source_includes_every_row_once_and_preserves_tail(tmp_path, dataset, seed):
    path, coordinate = real_source_fixture(tmp_path, dataset)
    selected = upf.trial_batches(
        dataset,
        seed,
        electricity_path=path,
        airlines_path=path,
        airlines_disk_cache=False,
        full_source=True,
    )
    repeated = upf.trial_batches(
        dataset,
        seed,
        electricity_path=path,
        airlines_path=path,
        airlines_disk_cache=False,
        full_source=True,
    )
    coverage = selected.coverage
    assert coverage["source_rows"] == coverage["selected_rows"] == 107
    assert coverage["excluded_rows"] == 0
    assert coverage["batch_rows"] == [8, 8] + [7] * 13
    assert coverage["offline_rows"] + coverage["online_rows"] == 107
    assert coverage["included_row_ranges"][0][0] == 0
    assert coverage["included_row_ranges"][-1][1] == 107
    assert selected.source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    selected_indices = []
    for split, again, span in zip(selected, repeated, coverage["source_row_ranges"], strict=True):
        train_x, train_y, holdout_x, holdout_y = split
        indices = np.concatenate((train_x[:, coordinate], holdout_x[:, coordinate])).astype(int)
        np.testing.assert_array_equal(np.sort(indices), np.arange(span["start"], span["stop"]))
        assert not set(train_x[:, coordinate]) & set(holdout_x[:, coordinate])
        assert len(train_y) == span["train_rows"] and len(holdout_y) == span["holdout_rows"]
        for array, repeated_array in zip(split, again, strict=True):
            np.testing.assert_array_equal(array, repeated_array)
        selected_indices.extend(indices.tolist())
    assert sorted(selected_indices) == list(range(107))


@pytest.mark.parametrize("dataset", ["electricity", "airlines"])
def test_legacy_windows_remain_explicitly_distinct_from_full_source(tmp_path, dataset):
    path, _ = real_source_fixture(tmp_path, dataset)
    historical = upf.trial_batches(
        dataset,
        42,
        electricity_path=path,
        airlines_path=path,
        size=4,
        airlines_disk_cache=False,
    )
    assert historical.coverage["selection"] == "random_contiguous_window"
    assert historical.coverage["selected_rows"] == 60
    assert historical.coverage["excluded_rows"] == 47
    assert historical.coverage["full_source"] is False
    with pytest.raises(ValueError, match="fixed batch size"):
        upf.trial_batches(dataset, 42, electricity_path=path, airlines_path=path, size=4, full_source=True)


@pytest.mark.parametrize("dataset", ["gauss", "circles"])
def test_full_synthetic_sources_keep_all_default_examples(dataset):
    selected = upf.trial_batches(dataset, 42, full_source=True)
    repeated = upf.trial_batches(dataset, 42, full_source=True)
    other_seed = upf.trial_batches(dataset, 43, full_source=True)
    assert selected.coverage["source_rows"] == selected.coverage["selected_rows"] == 75_000
    assert selected.coverage["batch_rows"] == [5000] * 15
    for split, again in zip(selected, repeated, strict=True):
        for array, repeated_array in zip(split, again, strict=True):
            np.testing.assert_array_equal(array, repeated_array)
    assert not np.array_equal(selected[0][0], other_seed[0][0])


def test_full_source_defaults_to_three_explicit_seeds_and_exports_file_hash(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(upf, "XGBoostLearner", LightweightLearner)
    monkeypatch.setattr(upf, "candidate_report", candidate_stub)
    path, _ = real_source_fixture(tmp_path, "electricity")
    report = upf.replicate(
        "electricity",
        electricity_path=path,
        full_source=True,
        rules=("cara",),
        metrics_only=True,
    )
    assert report["seeds"] == [42, 43, 44]
    assert report["trials"] == report["dispersion"]["n"] == 3
    assert report["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "v3-full-source" in report["protocol"]
    assert [trial["seed"] for trial in report["trials_detail"]] == [42, 43, 44]
    assert all(row["coverage"]["all_source_rows_included"] for row in report["trials_detail"])
    assert report["evaluation"]["comparable_to_paper"] is False
    assert report["paper_reference"]["comparable_to_paper"] is False
    assert report["paper_reference"]["reason"] == "full_source_partition_changes_the_evaluation_protocol"
    assert all(row["paper_auc"] is None for row in report["summary"].values())
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("seeds", [[], [42, 42], [-1, 43, 44], 0, True])
def test_invalid_seed_identities_are_rejected(seeds):
    with pytest.raises(ValueError, match="seeds"):
        upf.replicate("gauss", seeds=seeds)


def test_explicit_seeds_and_legacy_count_cannot_disagree():
    with pytest.raises(ValueError, match="count differs"):
        upf.replicate("gauss", trials=10, seeds=[42, 43, 44])


class LightweightLearner:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, index, x, y):
        return float(x[:, 0].mean())

    def losses(self, cutoff, x, y):
        return (x[:, 0] > cutoff) != y

    def observation_predict(self, cutoff, x, positive):
        scores = (x[:, 0] > cutoff).astype(float)
        return scores.astype(int), scores


def candidate_stub(name, seed, *args, **kwargs):
    return {
        "run_key": f"{name}-{seed}-{kwargs['protocol']}",
        "contract": {"environment": {"fixture": True}},
        "artifact_mode": kwargs["artifact_mode"],
    }


@pytest.mark.parametrize("mode", ["individual", "same-batch", "next-batch"])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_trial_heldout_label_flip_preserves_cara_decisions(monkeypatch, mode, seed):
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
        "full_source": True,
    }
    original = upf.run_trial("electricity", seed, **options)
    monkeypatch.setattr(upf, "trial_batches", lambda *a, **k: flipped)
    modified = upf.run_trial("electricity", seed, **options)
    np.testing.assert_allclose(
        np.asarray(original["staleness_matrix"], dtype=float),
        np.asarray(modified["staleness_matrix"], dtype=float),
    )
    for rule in ("cara", "cara_cumulative", "cara_periodic"):
        for old, new in zip(original["rules"][rule]["curve"], modified["rules"][rule]["curve"], strict=True):
            assert old["retrain_points"] == new["retrain_points"]
            assert old["parameters"] == new["parameters"]
            assert old["average_accuracy"] == pytest.approx(1 - new["average_accuracy"])
            assert old["quality"]["accuracy"] == pytest.approx(1 - new["quality"]["accuracy"])
            assert old["quality"]["n"] == new["quality"]["n"]
    assert original["candidate_tracking"]["artifact_mode"] == "metrics_only"
    assert original["policy_tracking"]["contract"]["settings"]["separate_query_split"]
    assert original["evaluation_quality"]["used_for_decisions"] is False


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("same_batch", [True, False])
def test_policy_quality_matches_manual_core_metrics_and_caches_schedules(seed, same_batch):
    rng = np.random.default_rng(seed)
    probabilities = [np.array([0.9, 0.7]), np.array([0.9, 0.8, 0.1, 0.2, 0.3])]
    labels = [np.array([1, 0]), np.array([1, 1, 0, 0, 0])]
    for p, y in zip(probabilities, labels, strict=True):
        order = rng.permutation(len(y))
        p[:] = p[order]
        y[:] = y[order]
    batches = [(np.zeros((1, 1)), np.zeros(1), np.zeros((1, 1)), np.zeros(1))] * 2
    batches += [(np.zeros((1, 1)), np.zeros(1), p[:, None], y) for p, y in zip(probabilities, labels)]

    class Learner:
        def __init__(self):
            self.calls = []

        def observation_predict(self, model, x, positive):
            self.calls.append((model, len(x)))
            scores = x[:, 0] if model == 2 else np.full(len(x), 0.3)
            return (scores >= 0.5).astype(int), scores

    # Both the operating-point table and the second policy repeat an existing schedule.
    changed = {"retrain_points": [2], "average_accuracy": 0.75}
    frozen = {"retrain_points": [], "average_accuracy": 0.55}
    analysis = {
        "rules": {"cara": {"curve": [changed, frozen]}, "upf": {"curve": [copy.deepcopy(changed)]}},
        "table": {"low": {"cara": copy.deepcopy(changed)}},
    }
    before = copy.deepcopy(analysis)
    learner = Learner()
    matrix = SimpleNamespace(offline=2, batches=4, initial=1, paper_timing=same_batch)
    upf.add_policy_quality(analysis, batches, [0, 1, 2, 3], learner, matrix)
    expected = Metrics()
    served = [2, 2] if same_batch else [1, 2]
    for model, p, y in zip(served, probabilities, labels, strict=True):
        for probability, target in zip(p if model == 2 else np.full(len(p), 0.3), y, strict=True):
            expected.add(bool(target), float(probability))
    assert changed["quality"] == expected.values()
    assert changed["served_by"] == served
    assert changed["quality"]["accuracy"] != changed["average_accuracy"]
    assert changed["retrain_points"] == before["rules"]["cara"]["curve"][0]["retrain_points"]
    assert analysis["evaluation_quality"]["unique_schedules"] == 2
    assert analysis["evaluation_quality"]["probability_cache_pairs"] == len(learner.calls)
    assert len(learner.calls) == (4 if same_batch else 3)
    points = metric_points(
        policy_tracking(
            {
                **analysis,
                "setup": {"offline_batches": 2, "batches": 4},
                "alphas": [0.0, 0.5],
            }
        )
    )
    for name, value in expected.values().items():
        assert (f"tracking_v1/policy/cara/quality/{name}", 0, value) in points
    json.dumps(analysis, allow_nan=False)


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


@pytest.mark.parametrize("seed_specification", [2, [42, 43]])
def test_cara_export_keeps_seed_schedules_and_parameters(monkeypatch, seed_specification):
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
        seed = 0 if seed in (0, 42) else 1
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
    report = cara.replicate(
        "electricity", seeds=seed_specification, kappa_points=3, kappa_max=1, metrics_only=True
    )
    assert report["protocol"] == "CARA-authors-historical-v1"
    assert len(report["trials_detail"]) == 2
    expected_seeds = [0, 1] if seed_specification == 2 else seed_specification
    assert report["seed_values"] == [trial["seed"] for trial in report["trials_detail"]] == expected_seeds
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
