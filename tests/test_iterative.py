import numpy as np

from continual_bench.datasets.arrays import ImageDataset, ListDataset
from continual_bench.evaluation.iterative import (
    MemoryBudgetExceeded,
    MemoryGuard,
    run_iterative,
    update_bounds,
)


def test_update_bounds_use_tasks_or_ten_blocks():
    assert update_bounds([0, 0, 1, 1, 2], 5) == [(0, 2), (2, 4), (4, 5)]
    assert len(update_bounds([None] * 95, 95)) == 10


def test_run_iterative_early_stops_and_reports(tmp_path):
    rng = np.random.default_rng(0)
    xs = [{"a": float(v), "b": float(-v)} for v in rng.normal(size=300)]
    ys = ["1" if x["a"] > 0 else "0" for x in xs]
    data = ListDataset("toy", "multiclass", xs, ys, [None] * 300, {})
    report = run_iterative(data, tmp_path, max_epochs=100, patience=3, log=lambda _: None)
    assert report["complete"] and len(report["updates"]) == 10
    assert all(u["epochs_run"] <= 100 and u["patience_consumed"] <= 3 for u in report["updates"])
    assert any(u["early_stopped"] for u in report["updates"])
    assert (tmp_path / "result.json").exists()


def test_image_dataset_builds_rows_lazily():
    x = np.arange(12, dtype=np.uint8).reshape(3, 4)
    data = ImageDataset("img", x, ["0", "1", "0"], [0, 0, 1], {})
    row, label = data.row(1)
    assert len(row) == 4 and label == "1" and abs(row["pixel_0000"] - 4 / 255) < 1e-9


def test_memory_guard_raises(monkeypatch):
    guard = MemoryGuard(budget_bytes=1000, fraction=1.0)
    monkeypatch.setattr(MemoryGuard, "usage", staticmethod(lambda: 10_000))
    try:
        guard.check()
    except MemoryBudgetExceeded:
        return
    raise AssertionError("guard did not trigger")


class FakeTracker:
    def __init__(self):
        self.epochs, self.rounds, self.published = [], [], None
        self.errors = []

    def log_epoch(self, metrics):
        self.epochs.append(metrics)

    def finalize_round(self, summary):
        self.rounds.append(summary)

    def publish(self, report, directory):
        self.published = report
        return {"mlflow_run_id": "fake"}

    def fail(self, message):
        raise AssertionError(message)


def test_tracker_receives_sagemaker_metric_contract(tmp_path):
    rng = np.random.default_rng(1)
    xs = [{"a": float(v)} for v in rng.normal(size=200)]
    ys = ["1" if x["a"] > 0 else "0" for x in xs]
    data = ListDataset("toy", "multiclass", xs, ys, [None] * 200, {})
    tracker = FakeTracker()
    report = run_iterative(data, tmp_path, max_epochs=3, patience=2, log=lambda _: None, tracker=tracker)
    expected = {"epoch", "train:log_loss", "train:accuracy", "validation:log_loss", "validation:accuracy",
                "validation:f1_macro", "validation:balanced_accuracy", "early_stopped",
                "best_epoch", "patience_consumed", "best_validation_loss", "peak_rss_mb"}
    assert expected <= set(tracker.epochs[0])
    assert any("validation:pr_auc" in row and "train:pr_auc" in row for row in tracker.epochs)
    assert len(tracker.rounds) == len(report["updates"]) == 10
    assert report["mlflow"]["mlflow_run_id"] == "fake"
    published = set(tracker.published)
    assert {"summary", "windows", "class_metrics", "model_versions", "costs", "system_metrics"} <= published
    assert (tmp_path / "artifacts" / "data" / "index.json").exists()
