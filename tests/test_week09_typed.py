import json
from dataclasses import replace

import pytest

from continual_bench.adapters.week09_sources import FriedmanDriftSource, TaskSample
from continual_bench.application.typed_benchmark import run_typed


class FixtureSource:
    def __init__(self, records, task="multiclass"):
        self.records, self.task = records, task

    def samples_iter(self):
        yield from self.records

    def manifest(self):
        return {"name": "synthetic_fixture", "version": "v1", "task": self.task, "seed": 42}


def records(n=30, delay=0):
    return [TaskSample(str(i), i, i, i + delay, {"x": float(i % 3)}, str(i % 3),
                       task_id=i // 10, split="test" if i % 5 == 4 else "train") for i in range(n)]


def test_friedman_runs_native_regression_and_complete_replay_is_idempotent(tmp_path):
    source = FriedmanDriftSource(90, 42, label_delay=3)
    first = run_typed(source, tmp_path, warmup=10)
    second = run_typed(source, tmp_path, warmup=10)
    assert first == second
    assert first["observed"] == 90
    assert first["metrics"]["train"]["n"] == 80
    assert first["metrics"]["train"]["mae"] >= 0
    assert first["metrics"]["train"]["rmse"] >= 0
    assert first["training"]["postwarmup_samples"] == 80
    assert (tmp_path / first["identity"] / "samples.jsonl").is_file()
    assert first["config"]["source"]["task"] == "regression"


def test_multiclass_metrics_keep_three_classes_and_exclude_heldout_training(tmp_path):
    report = run_typed(FixtureSource(records()), tmp_path, warmup=3)
    assert report["metrics"]["train"]["n"] == 21
    assert report["metrics"]["test"]["n"] == 6
    assert 0 <= report["metrics"]["test"]["macro_f1"] <= 1
    assert report["training"]["postwarmup_samples"] == 21
    assert report["metrics"]["validation"]["accuracy"] is None
    assert report["protocol"]["heldout_training"] is False
    assert "test:0" in report["per_task"]


@pytest.mark.parametrize("policy", ["none", "online", "periodic"])
def test_typed_baselines_obey_training_budgets_and_heldout_split(tmp_path, policy):
    report = run_typed(FixtureSource(records()), tmp_path, policy=policy, warmup=3,
                       every=5, window=4, training_budget=7)
    assert report["training"]["postwarmup_samples"] <= 7
    if policy == "none":
        assert report["updates"] == []
    elif policy == "periodic":
        assert all(u["action"] == "reset_retrain" and u["train_samples"] == 4 for u in report["updates"])
    else:
        assert report["training"]["blocked_decisions"] > 0


def test_typed_predicts_before_release_and_never_trains_test_labels(tmp_path, monkeypatch):
    import continual_bench.application.typed_benchmark as module

    learned = []

    class AuditModel:
        def predict_one(self, x):
            return "0"

        def learn_one(self, x, y):
            learned.append((x["x"], y))

    monkeypatch.setattr(module, "_model", lambda task: AuditModel())
    data = [TaskSample(str(i), i, i, i + 2, {"x": float(i)}, str(i % 2),
                       split="test" if i == 4 else "train") for i in range(10)]
    report = run_typed(FixtureSource(data), tmp_path, warmup=2)
    predictions = json.loads((tmp_path / report["identity"] / "predictions.json").read_text())
    assert [r["training_labels_available"] for r in predictions[:4]] == [0, 0, 0, 1]
    assert len(learned) == 9 and all(x != 4.0 for x, _ in learned)


def test_typed_rejects_overlap_order_nonfinite_and_unsupported_policy(tmp_path):
    source = FixtureSource(records())
    with pytest.raises(ValueError, match="unsupported"):
        run_typed(source, tmp_path, policy="cost", warmup=2)
    invalid = records()
    invalid[1] = replace(invalid[1], sample_id=invalid[0].sample_id)
    with pytest.raises(ValueError, match="duplicate"):
        run_typed(FixtureSource(invalid), tmp_path, warmup=2)
    invalid = records()
    invalid[1] = replace(invalid[1], available_at=0)
    with pytest.raises(ValueError, match="ordered"):
        run_typed(FixtureSource(invalid), tmp_path, warmup=2)
    data = [TaskSample(str(i), i, i, i, {"x": float(i)}, float(i)) for i in range(10)]
    data[2] = replace(data[2], y=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        run_typed(FixtureSource(data, "regression"), tmp_path, warmup=2)


def test_typed_failure_leaves_no_completion_and_retry_restarts_cleanly(tmp_path, monkeypatch):
    import continual_bench.application.typed_benchmark as module

    factory = module._model

    class BrokenModel:
        def predict_one(self, x):
            return "0"

        def learn_one(self, x, y):
            raise RuntimeError("injected training failure")

    monkeypatch.setattr(module, "_model", lambda task: BrokenModel())
    with pytest.raises(RuntimeError, match="injected"):
        run_typed(FixtureSource(records()), tmp_path, warmup=2)
    assert not list(tmp_path.rglob("report.json"))
    monkeypatch.setattr(module, "_model", factory)
    report = run_typed(FixtureSource(records()), tmp_path, warmup=2)
    assert report["observed"] == 30
    assert len(list(tmp_path.rglob("report.json"))) == 1


def test_typed_prefix_budget_records_selection_and_rejects_insufficient_warmup(tmp_path):
    report = run_typed(FixtureSource(records()), tmp_path, warmup=2, max_samples=10)
    assert report["observed"] == 10
    assert report["protocol"]["selection"] == "bounded published prefix"
    with pytest.raises(ValueError, match="warmup"):
        run_typed(FixtureSource(records()), tmp_path, warmup=10, max_samples=10)
