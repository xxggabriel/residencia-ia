import hashlib
import json
import pickle

import pytest

from continual_bench.application.runner import run
from continual_bench.config import Config
from continual_bench.core.contracts import TaskSample
from continual_bench.evaluation.stream_training_history import stream_training_history
from continual_bench.evaluation.typed import run_typed


def _assert_history(directory, report):
    fits = report["training_history"]
    assert json.loads((directory / "training-history.json").read_text()) == {"version": 1, "fits": fits}
    index = json.loads((directory / "artifacts/data/index.json").read_text())
    assert "data/training/history.json" in index["data"]
    steps = [step for fit in fits for step in fit["steps"]]
    assert [row["global_step"] for row in steps] == list(range(1, len(steps) + 1))
    for fit in fits:
        assert [row["step"] for row in fit["steps"]] == list(range(1, len(fit["steps"]) + 1))
        assert all(row["kind"] == "update" for row in fit["steps"])
        presentations = sum(row["metrics"]["training_presentations"] for row in fit["steps"])
        assert presentations == fit["training_presentations"]
        assert any(f"plots/training/{fit['fit_id']}/" in fig["plot"] for fig in index["figures"])
        checkpoint = fit["checkpoint"]
        if checkpoint:
            path = directory / checkpoint["file"]
            with path.open("rb") as stream:
                stream.seek(checkpoint.get("offset", 0))
                state = stream.read(checkpoint["bytes"])
            assert hashlib.sha256(state).hexdigest() == checkpoint["sha256"]
    return fits


@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_binary_protocols_record_real_training_and_shared_journal(tmp_path, mode):
    config = Config.model_validate({
        "mode": mode, "dataset": {"name": "sea", "samples": 150, "label_delay": 3}, "warmup": 20,
        "metric_window": 33,
        "policy": {"name": "none" if mode == "offline" else "online", "action": "learn_one"},
    })
    report = run(config, tmp_path)
    directory = tmp_path / config.identity()
    fits = _assert_history(directory, report)
    assert fits[0]["action"] == "initial_fit"
    if mode == "offline":
        assert len(fits) == 1
        assert [fits[0]["start_seq"], fits[0]["end_seq"]] == report["split"]["train"]
    else:
        assert len(fits) == 2
        online = fits[1]
        assert online["training_presentations"] == report["training"]["used"]
        assert [row["metrics"]["learning_calls"] for row in online["steps"]] == [33, 33, 33, 31]
        assert online["end_seq"] == 150
    assert fits[-1]["checkpoint"]["sha256"] == report["model_versions"]["state_sha256"]
    assert stream_training_history(report, directory) == fits


def test_historical_gate_preserves_rejected_fit_and_original_validation_scope():
    report = {"config": {"warmup": 2, "metric_window": 5}, "labeled": 6, "updates": [], "candidates": [
        {"seq": 5, "action": "reset_retrain", "train_samples": 3, "seconds": .1, "promoted": False,
         "train_seq_range": [2, 5], "validation_samples": 1, "validation_seq_range": [5, 6],
         "candidate_accuracy": 0., "incumbent_accuracy": 1., "accuracy_gain": -1.},
    ]}
    initial, rejected = stream_training_history(report)
    assert initial["start_seq"] == 0
    assert initial["steps"][0]["metrics"]["training_seconds"] is None
    assert rejected["start_seq"] == 2 and rejected["end_seq"] == 5
    assert rejected["validation_samples"] == 1 and rejected["checkpoint"] is None
    assert rejected["steps"][0]["metrics"]["promotion/accepted"] == 0
    assert rejected["steps"][0]["metrics"]["promotion/candidate/accuracy"] == 0


class _TypedSource:
    def samples_iter(self):
        yield from (
            TaskSample(str(i), i, i, i + 2, {"x": float(i)}, str(i % 2),
                       split="validation" if i % 5 == 1 else "train")
            for i in range(16)
        )

    def manifest(self):
        return {"name": "sample", "version": "v1", "task": "multiclass"}


def test_typed_periodic_keeps_one_reloadable_checkpoint_per_fit(tmp_path):
    report = run_typed(_TypedSource(), tmp_path, warmup=2, every=3, window=4, policy="periodic")
    directory = tmp_path / report["run_key"]
    fits = _assert_history(directory, report)
    assert len(fits) == len(report["updates"]) + 1
    assert fits[0]["training_presentations"] == 2 and fits[0]["end_seq"] == 3
    for fit, update in zip(fits[1:], report["updates"], strict=True):
        assert [fit["start_seq"], fit["end_seq"]] == update["train_seq_range"]
        assert fit["training_presentations"] == update["train_samples"]
        assert fit["validation_samples"] == 0
    with (directory / fits[-1]["checkpoint"]["file"]).open("rb") as stream:
        assert pickle.load(stream).predict_one({"x": 2.}) in ("0", "1")

    old_updates = [{k: v for k, v in row.items() if k != "train_seq_range"} for row in report["updates"]]
    old_report = {**report, "training": {"warmup_samples": 2}, "updates": old_updates}
    derived = stream_training_history(old_report, directory)
    assert [f["start_seq"] for f in derived] == [f["start_seq"] for f in fits]
    assert derived[0]["train_seconds"] is None
