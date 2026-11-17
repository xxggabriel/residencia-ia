import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.backfill_training_history import BACKFILL_SHA, BACKFILL_STATUS, main

RUN_ID = "a" * 32


def _epoch_report():
    return {
        "complete": True, "version": "semana09-epochs-v1", "run_key": "b" * 24,
        "epochs": [
            {"epoch": epoch, "train_presentations": 5, "test_examples": 2,
             "metrics": {"mae": mae}, "training_seconds": 0.01, "prediction_seconds": 0.02,
             "costs": {"metrics": {"errors": 1}}}
            for epoch, mae in ((1, 0.6), (2, 0.5))
        ],
    }


def test_local_historical_inspection_is_read_only_and_writes_disjoint_output_on_apply(tmp_path):
    path = tmp_path / "run" / "epoch-report.json"
    path.parent.mkdir()
    source = json.dumps(_epoch_report(), indent=2).encode()
    path.write_bytes(source)
    output = tmp_path / "backfills"
    [plan] = main([str(path.parent)])
    assert plan["availability"] == "partial" and plan["groups"] == 1 and plan["steps"] == 2
    assert plan["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert not output.exists()
    [receipt] = main([str(path), "--apply", "--output-dir", str(output)])
    destination = Path(receipt["output"])
    assert path.read_bytes() == source
    assert list(path.parent.iterdir()) == [path]
    history = json.loads((destination / "training-history.json").read_text())
    [fit] = history["fits"]
    assert fit["checkpoint"] is None and fit["training_presentations"] == 10
    assert [(row["step"], row["global_step"], row["metrics"]["holdout:mae"]) for row in fit["steps"]] == [
        (1, 1, 0.6), (2, 2, 0.5),
    ]
    assert not any("validation:mae" in row["metrics"] for row in fit["steps"])
    assert (destination / "artifacts" / "data" / "index.json").is_file()
    assert (destination / "backfill-provenance.json").is_file()


def test_legacy_zia_preserves_updates_but_does_not_turn_iterations_into_epochs(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "complete": True, "protocol": "batch_prequential_with_final_temporal_holdout",
        "model_versions": [
            {"action": "initial", "start_seq": 0, "end_seq": 5, "sha256": "a" * 64,
             "file": "models/0000.ubj", "iterations": 9, "train_samples": 5},
            {"action": "online", "start_seq": 5, "end_seq": 10, "sha256": "b" * 64,
             "file": "models/0001.ubj", "iterations": 800, "train_samples": 5},
        ],
    }))
    output = tmp_path.parent / (tmp_path.name + "-derived")
    [result] = main([str(path), "--apply", "--output-dir", str(output)])
    assert result["groups"] == 2 and result["steps"] == 2 and result["availability"] == "partial"
    history = json.loads((Path(result["output"]) / "training-history.json").read_text())
    for fit in history["fits"]:
        [step] = fit["steps"]
        assert step["kind"] == "update" and step["step"] == 1
        assert "iterations" not in step["metrics"]


def test_legacy_iterative_groups_recorded_epochs_and_recovers_regression_metric_names(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "complete": True, "config": {"eval_metric": "validation:mae"},
        "updates": [{"update": 1, "range": [0, 4], "train": 3, "epochs_run": 2},
                    {"update": 2, "range": [4, 8], "train": 3, "epochs_run": 1}],
        "epoch_history": [
            {"update": 1, "epoch": 1, "global_epoch": 1, "validation:log_loss": 0.7},
            {"update": 1, "epoch": 2, "global_epoch": 2, "validation:log_loss": 0.6},
            {"update": 2, "epoch": 1, "global_epoch": 3, "validation:log_loss": 0.5},
        ],
    }))
    [row] = main([str(path), "--apply", "--output-dir", str(tmp_path.parent / (tmp_path.name + "-derived"))])
    fits = json.loads((Path(row["output"]) / "training-history.json").read_text())["fits"]
    assert [(fit["start_seq"], fit["end_seq"]) for fit in fits] == [(0, 4), (4, 8)]
    assert [[(step["step"], step["global_step"], step["metrics"]["validation:mae"]) for step in fit["steps"]]
            for fit in fits] == [[(1, 1, 0.7), (2, 2, 0.6)], [(1, 3, 0.5)]]
    assert all("validation:log_loss" not in step["metrics"] for fit in fits for step in fit["steps"])


def test_s3_final_resolves_only_the_winning_attempt_and_binds_its_parent():
    base = "s3://private/campaigns/matrix/runs/unit"
    final_uri = f"{base}/final.json"
    prefix = f"{base}/attempts/job-2"
    run_key = "b" * 24
    report_uri = f"{prefix}/epochs/{run_key}/epoch-report.json"
    report_raw = json.dumps(_epoch_report()).encode()
    final = {
        "kind": "matrix_case", "version": "run-final-v1", "outputs_uri": prefix,
        "links": {"mlflow_run_id": RUN_ID}, "run_key": run_key,
        "matrix_coordinates": {"mode": "epochs"}, "report_sha256": hashlib.sha256(report_raw).hexdigest(),
    }
    calls = []

    def read(uri):
        calls.append(uri)
        return {final_uri: json.dumps(final).encode(), report_uri: report_raw}[uri]

    [row] = main([final_uri], read_s3=read)
    assert calls == [final_uri, report_uri]
    assert row["run_id"] == RUN_ID and row["source"] == report_uri and row["steps"] == 2
    final["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        main([final_uri], read_s3=read)


class _Mlflow:
    def __init__(self, report):
        self.report = report
        self.tags = {}
        self.children = []
        self.uploaded = []

    def get_run(self, run_id):
        assert run_id == RUN_ID
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, status="FINISHED", experiment_id="17"),
            data=SimpleNamespace(tags=self.tags),
        )

    def list_artifacts(self, run_id):
        return [SimpleNamespace(path="epoch-report.json", is_dir=False)]

    def download_artifacts(self, run_id, path, destination):
        dest = Path(destination) / path
        dest.write_bytes(self.report)
        return str(dest)

    def search_runs(self, experiment_ids, filter_string, max_results):
        assert experiment_ids == ["17"] and RUN_ID in filter_string and max_results == 1
        return self.children

    def set_tag(self, run_id, key, value):
        assert run_id == RUN_ID
        self.tags[key] = value

    def log_artifacts(self, run_id, directory, artifact_path):
        assert run_id == RUN_ID and artifact_path.startswith("training-backfill/v1/")
        self.uploaded.append((artifact_path, sorted(p.name for p in Path(directory).iterdir())))


def test_publication_is_explicit_and_reuses_exact_parent_without_touching_existing_report(
    tmp_path, monkeypatch,
):
    from continual_bench.infrastructure import training_tracking

    client = _Mlflow(json.dumps(_epoch_report()).encode())

    def children(tracker, experiment_id, run_id, report):
        assert tracker is client and experiment_id == "17" and run_id == RUN_ID
        assert report["training_history"][0]["steps"][1]["global_step"] == 2
        client.children.append("derived-child")

    monkeypatch.setattr(training_tracking, "publish_sdk_training_rounds", children)
    [plan] = main(["--run-id", RUN_ID], mlflow_client=client)
    assert plan["steps"] == 2 and not client.tags and not client.uploaded
    args = ["--run-id", RUN_ID, "--apply", "--output-dir", str(tmp_path)]
    original = client.report
    [done] = main(args, mlflow_client=client)
    assert done["publication"] == "applied" and client.tags[BACKFILL_STATUS] == "complete"
    assert client.tags[BACKFILL_SHA] in client.uploaded[0][0]
    assert "backfill-provenance.json" in client.uploaded[0][1]
    [again] = main(args, mlflow_client=client)
    assert again["publication"] == "already_applied"
    assert client.report == original and len(client.uploaded) == len(client.children) == 1


def test_native_legacy_parent_can_recover_only_real_epoch_points():
    class Native(_Mlflow):
        def list_artifacts(self, run_id):
            return []

        def get_run(self, run_id):
            run = super().get_run(run_id)
            run.data.metrics = {"epoch": 2, "train/log_loss": 0.5, "training_round/epochs_trained": 2}
            return run

        def get_metric_history(self, run_id, name):
            values = {"epoch": [1, 2], "train/log_loss": [0.7, 0.5]}[name]
            return [SimpleNamespace(value=value, step=i) for i, value in enumerate(values, 1)]

    client = Native(None)
    client.tags["cb-native-stage"] = "training"
    [result] = main(["--run-id", RUN_ID], mlflow_client=client)
    assert result["scheme"] == "native_epochs" and result["steps"] == 2
    assert result["availability"] == "partial" and result["source"].endswith("#metrics")
    assert not client.children and not client.uploaded
    client.tags["cb-native-stage"] = "processing"
    [excluded] = main(["--run-id", RUN_ID], mlflow_client=client)
    assert excluded["availability"] == "unavailable" and excluded["steps"] == 0
