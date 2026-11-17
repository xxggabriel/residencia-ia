import copy
import hashlib
import json

import pytest

from continual_bench.evaluation.artifacts import Figure, write_artifacts
from continual_bench.evaluation.training_evidence import write_training_artifacts
from continual_bench.infrastructure.benchmark_tracking import publish_local_benchmark
from continual_bench.infrastructure.training_tracking import training_checkpoint_paths, training_round_mlflow


def fit(name, kind, start, checkpoint=None):
    return {
        "fit_id": name,
        "action": "reset" if checkpoint else "online",
        "start_seq": None,
        "end_seq": 10,
        "checkpoint": checkpoint,
        "steps": [
            {
                "step": i,
                "global_step": start + i - 1,
                "kind": kind,
                "metrics": {"holdout:mae": 1 / i, "train_presentations": i, "missing": None},
            }
            for i in (1, 2)
        ],
        "summary": {},
    }


def test_fit_plots_preserve_existing_index_and_missing_values(tmp_path):
    write_artifacts(tmp_path, [Figure("performance", "quality", "Quality", "line", {"series": []})], {})
    report = {"training_history": [fit("fit-01", "epoch", 1), fit("fit-02", "update", 3)]}
    write_training_artifacts(tmp_path, report)
    result = write_training_artifacts(tmp_path, report)
    assert result["figures"] == 5
    directory = tmp_path / "artifacts/data"
    assert (directory / "performance/quality.json").is_file()
    curve = json.loads((directory / "training/fit-02/loss.json").read_text())
    assert curve["global_steps"] == [[1, 3], [2, 4]]
    assert curve["step_kind"] == "update"
    assert curve["series"][0]["points"] == [[1, 1.0], [2, 0.5]]
    assert json.loads((tmp_path / "training-history.json").read_text())["fits"] == report["training_history"]
    with pytest.raises(ValueError, match="global step"):
        list(training_round_mlflow("parent", {"training_history": [fit("another", "epoch", 0)]}))


def test_local_benchmark_groups_rounds_checks_checkpoints_and_reuses_parent(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = MlflowClient(uri)
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "model-versions.bin").write_bytes(b"prefixmodel")
    checkpoint = {
        "file": "model-versions.bin",
        "offset": 6,
        "bytes": 5,
        "sha256": hashlib.sha256(b"model").hexdigest(),
    }
    report = {
        "complete": True,
        "run_key": "fixture",
        "metrics": {"test": {"mae": 0.1}},
        "training_history": [fit("fit-01", "epoch", 1, checkpoint), fit("fit-02", "update", 3)],
    }
    (directory / "report.json").write_text(json.dumps(report))
    write_training_artifacts(directory, report)
    first = publish_local_benchmark(directory, report, experiment_name="local", client=client)
    second = publish_local_benchmark(directory, report, experiment_name="local", client=client)
    assert second == first
    parent = first["mlflow_run_id"]
    assert client.get_run(parent).data.metrics["summary/test/mae"] == 0.1
    assert {x.path for x in client.list_artifacts(parent)} >= {
        "training-history.json",
        "model-versions.bin",
        "plots",
        "data",
    }
    one, two = [entry["run_id"] for entry in first["training_rounds"]]
    assert client.get_run(one).data.params["checkpoint_offset"] == "6"
    assert client.get_run(two).data.tags["cb-step-kind"] == "update"
    assert [(m.step, m.value) for m in client.get_metric_history(two, "global_step")] == [(1, 3), (2, 4)]
    assert [(m.step, m.value) for m in client.get_metric_history(one, "holdout/mae")] == [(1, 1), (2, 0.5)]
    changed = copy.deepcopy(report)
    changed["training_history"][0]["checkpoint"]["offset"] = 0
    with pytest.raises(ValueError, match="checkpoint checksum"):
        training_checkpoint_paths(directory, changed)


def test_matrix_services_publishes_completed_case_without_training_again(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from continual_bench.application import matrix

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    config = matrix.ExperimentMatrix.model_validate(
        {
            "name": "typed-smoke",
            "evaluator": "typed",
            "datasets": [{"name": "sea", "samples": 100}],
            "models": [{"name": "hoeffding_tree"}],
            "base": {"warmup": 10},
            "on_incompatible": "error",
        }
    )
    state = tmp_path / "state"
    [case] = matrix.execute_matrix(config, state)["cases"].values()
    assert case["status"] == "passed" and "tracking" not in case

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("publication reran a completed evaluation")

    monkeypatch.setattr(matrix, "_execute_case", unexpected_evaluation)
    [published] = matrix.execute_matrix(config, state, services=True)["cases"].values()
    assert published["run_key"] == case["run_key"]
    assert published["tracking"]["complete"] is True
    assert published["tracking"]["training_rounds"]
