import json
import signal
from shutil import copytree

import numpy as np
import pytest

from continual_bench.application.cloud import layout, planning
from continual_bench.application.cloud.contracts import CampaignRequest
from continual_bench.application.cloud.fixtures import (
    fixture_parquet,
    publish_fixture_release,
    publish_fixture_snapshot,
)
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.application.cloud.worker import EXIT_FAILED, run_unit
from continual_bench.config import TrackingConfig
from continual_bench.datasets.zia import sha256_file
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json


@pytest.mark.parametrize("dataset", ["social", "fraud"])
def test_incremental_tracks_single_row_before_holdout(tmp_path, dataset):
    if dataset == "fraud":
        pytest.importorskip("xgboost")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from continual_bench.evaluation.zia import run
    from continual_bench.evaluation.zia_predictions import prediction_manifest
    from continual_bench.evaluation.zia_publication import replay_predictions

    path = tmp_path / dataset / "smoke.parquet"
    manifest = fixture_parquet(path, dataset)
    data = pq.read_table(path)
    data = data.set_column(0, "event_time", pa.array(np.arange(data.num_rows, dtype=np.int64)))
    pq.write_table(data, path)
    manifest.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    path.with_suffix(".json").write_text(json.dumps(manifest))

    output = tmp_path / "run"
    report = run(
        tmp_path,
        dataset,
        "smoke",
        "incremental",
        output,
        TrackingConfig().model_dump(),
        iterative_config={"max_epochs": 2},
    )
    fit = report["training_history"][-1]
    assert (fit["start_seq"], fit["end_seq"]) == (8999, 9000)
    assert fit["train_samples"] == 1
    assert fit["validation_samples"] == 0
    assert fit["training_presentations"] == 2
    assert fit["summary"]["selection_scope"] == "train"
    assert report["contract"]["holdout_start"] == 9000
    replay = tmp_path / "replay"
    copytree(output / "models", replay / "models")
    replay_predictions(replay, report, tmp_path)
    assert prediction_manifest(replay, report)["parts"] == prediction_manifest(output, report)["parts"]


def test_interrupted_fit_persists_history_without_a_deployed_checkpoint(tmp_path):
    pytest.importorskip("pyarrow")
    from continual_bench.evaluation.zia import run

    class Tracker:
        interrupted = False

        def log_epoch_metrics(self, metrics, *, step):
            self.interrupted = True

    fixture_parquet(tmp_path / "social" / "smoke.parquet")
    tracker = Tracker()
    output = tmp_path / "run"
    with pytest.raises(InterruptedError, match="fit was interrupted"):
        run(
            tmp_path,
            "social",
            "smoke",
            "incremental",
            output,
            TrackingConfig().model_dump(),
            tracker=tracker,
            iterative_config={"max_epochs": 2},
            stop_requested=lambda: tracker.interrupted,
        )
    fit = json.loads((output / "training-fits-partial" / "fit-000001.json").read_text())
    assert fit["status"] == "interrupted" and fit["steps"][0]["step"] == 1
    assert fit["checkpoint"] is None and fit["deployed_version"] is None
    assert not (output / "report.json").exists()
    assert not (output / "models").exists()


def test_batch_sigterm_preserves_interrupted_fit(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from continual_bench.models.zia import MultinomialLogistic

    store = LocalObjectStore(tmp_path / "objects")
    snapshot = publish_fixture_snapshot(store, tmp_path, dataset="social")
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id="interrupted-zia-fit",
        snapshot=snapshot.model_dump(),
        matrix={"kind": "zia", "policies": ["frozen"]},
    )
    plan = planning.build_plan(request, release, snapshot_manifest=verify_snapshot(store, snapshot))
    planning.publish_plan(store, plan)
    [unit] = plan["units"]
    train_epoch = MultinomialLogistic.train_epoch

    def interrupt_after_train(model, batch):
        metrics = train_epoch(model, batch)
        signal.raise_signal(signal.SIGTERM)
        return metrics

    monkeypatch.setattr(MultinomialLogistic, "train_epoch", interrupt_after_train)
    env = {"AWS_BATCH_JOB_ID": "interrupted", "AWS_BATCH_JOB_ATTEMPT": "1"}
    assert run_unit(store, unit["unit_key"], tmp_path / "work", env=env) == EXIT_FAILED
    attempt = layout.attempt_key_for("interrupted", "1")
    status = get_json(store, layout.attempt_status_key(request.campaign_id, unit["run_id"], attempt))
    assert status["state"] == "INTERRUPTED"
    prefix = layout.attempt_prefix(request.campaign_id, unit["run_id"], attempt)
    fit = get_json(store, f"{prefix}/partial/training-fits-partial/fit-000001.json")
    assert fit["status"] == "interrupted"
    assert len(fit["steps"]) == 1
    assert store.head(layout.final_key(request.campaign_id, unit["run_id"])) is None
