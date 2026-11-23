"""Complete metric evidence uses one deterministic archive and no redundant metric reads."""

import hashlib
import json
import os
import zipfile

import pytest

from continual_bench.evaluation.tracking_observations import observations
from continual_bench.infrastructure.metric_evidence_bundle import (
    ARCHIVE_NAME,
    MANIFEST_NAME,
    build_metric_evidence_bundle,
)
from continual_bench.infrastructure.observation_tracking import publish_sdk_observations


def test_bundle_keeps_complete_text_evidence_and_rejects_raw_inputs_and_images(tmp_path):
    retained = {
        "report.json": b'{"metrics":{"accuracy":0.8},"inference_probe":{"features":{"x":1}}}',
        "training-history.json": b'{"fits":[]}',
        "predictions.jsonl": b'{"seq":1,"label":true,"probability":0.7}\n',
        "artifacts/data/series.json": b'{"points":[1,null,3]}',
        "tracking-observations/mlflow-lineage.json": b'{"parent":"id"}',
        "tracking-observations/observations.json": b'{"all_metrics":true}',
        "dashboard.html": b"<html>Complete local dashboard</html>",
    }
    for name, payload in retained.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for name in ("events.jsonl", "samples.jsonl", "continual-probes.json", "plot.png", "plot.svg", ".lock"):
        (tmp_path / name).write_text("excluded")
    receipt = build_metric_evidence_bundle(tmp_path)
    archive_path = tmp_path / receipt["artifact_path"]
    assert archive_path.name == ARCHIVE_NAME
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == receipt["sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(retained) | {MANIFEST_NAME}
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        manifest_raw = archive.read(MANIFEST_NAME)
        assert hashlib.sha256(manifest_raw).hexdigest() == receipt["manifest"]["sha256"]
        manifest = json.loads(manifest_raw)
        for member in manifest["members"]:
            payload = archive.read(member["path"])
            assert payload == retained[member["path"]]
            assert len(payload) == member["bytes"]
            assert hashlib.sha256(payload).hexdigest() == member["sha256"]
    assert receipt["evidence_file_count"] == len(retained)
    assert receipt["member_count"] == len(retained) + 1
    # Neither source timestamps nor the publication's own receipt rebind the archive.
    os.utime(tmp_path / "report.json", (1_700_000_000, 1_700_000_000))
    (tmp_path / "benchmark-receipt.json").write_text("publication receipt")
    assert build_metric_evidence_bundle(tmp_path) == receipt


@pytest.mark.parametrize("filename", ["model.pkl", "checkpoint.bin", "candidate.ubj"])
def test_bundle_fails_closed_before_packaging_serialized_models(tmp_path, filename):
    (tmp_path / "report.json").write_text("{}")
    (tmp_path / filename).write_bytes(b"model-state")
    with pytest.raises(ValueError, match="model or checkpoint"):
        build_metric_evidence_bundle(tmp_path)
    assert not (tmp_path / ARCHIVE_NAME).exists()


def test_sdk_observations_can_skip_only_the_points_already_logged_by_its_caller(tmp_path, monkeypatch):
    class Client:
        def __init__(self):
            self.uploads = []

        def log_batch(self, *args, **kwargs):
            pass

        def log_artifact(self, *args, **kwargs):
            self.uploads.append(args)

        def log_artifacts(self, *args, **kwargs):
            self.uploads.append(args)

    root = tmp_path / "tracking-observations"
    root.mkdir()
    (root / "observations.json").write_text("{}")
    report = {
        "artifact_mode": "metrics_only",
        "tracking_observations": observations(
            "binary",
            {},
            windows=[{"step": i, "scope": "fixture", "quality": {"accuracy": i / 600}} for i in range(600)],
        ),
    }
    calls = []
    monkeypatch.setattr(
        "continual_bench.infrastructure.observation_tracking.write_observation_evidence", lambda *args: root
    )
    monkeypatch.setattr(
        "continual_bench.infrastructure.metric_tracking.log_metric_points",
        lambda client, run_id, points, **kwargs: calls.append(list(points)),
    )
    client = Client()
    publish_sdk_observations(
        client, "run", tmp_path, report, [], max_points_per_key=None, upload_artifacts=False
    )
    assert len(calls) == 1 and len(calls[0]) == 600
    assert client.uploads == []
    publish_sdk_observations(
        client,
        "run",
        tmp_path,
        report,
        [],
        max_points_per_key=None,
        metrics_already_logged=True,
        upload_artifacts=False,
    )
    assert len(calls) == 1  # No duplicate history reads or writes for the same observation points.
    publish_sdk_observations(client, "run", tmp_path, report, [])
    assert len(calls) == 2 and client.uploads  # Default publication is unchanged.


def test_metrics_only_run_artifacts_uploads_one_bundle_without_repeating_observation_history(
    tmp_path, monkeypatch
):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from continual_bench.infrastructure.metric_tracking import log_metric_points
    from continual_bench.infrastructure.run_artifacts import publish_run_artifacts

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = MlflowClient(uri)
    experiment = client.create_experiment("metric-evidence-bundle")
    run_id = client.create_run(experiment).info.run_id
    points = [("observations_v1/windows/fixture/accuracy", i, i / 600) for i in range(600)]
    log_metric_points(client, run_id, points, max_points_per_key=None)

    def redundant_write(*args, **kwargs):
        pytest.fail("already logged observation series was reconciled or uploaded again")

    monkeypatch.setattr("continual_bench.infrastructure.metric_tracking.log_metric_points", redundant_write)
    monkeypatch.setattr("continual_bench.infrastructure.run_artifacts.log_detailed_figures", redundant_write)
    monkeypatch.setattr(client, "log_artifacts", redundant_write)
    upload = client.log_artifact
    uploaded = []

    def record_upload(run, path, **kwargs):
        uploaded.append(path)
        return upload(run, path, **kwargs)

    monkeypatch.setattr(client, "log_artifact", record_upload)
    report = {
        "artifact_mode": "metrics_only",
        "complete": True,
        "tracking_observations": observations(
            "binary",
            {},
            windows=[{"step": i, "scope": "fixture", "quality": {"accuracy": i / 600}} for i in range(600)],
        ),
        "system_metrics": {
            "samples": [{"timestamp_ms": 1000 + i, "values": {"cpu": float(i)}} for i in range(600)]
        },
    }
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps(report))
    result = publish_run_artifacts(client, run_id, directory, report, observations_already_logged=True)
    assert [os.path.basename(path) for path in uploaded] == [ARCHIVE_NAME]
    assert result["artifact_files"] == 1
    assert result["metric_evidence_bundle"]["artifact_path"] == ARCHIVE_NAME
    assert "observation_plots" not in result
    assert len(client.get_metric_history(run_id, points[0][0])) == 600
    assert len(client.get_metric_history(run_id, "system/cpu")) == 600
    with zipfile.ZipFile(directory / ARCHIVE_NAME) as archive:
        assert len(json.loads(archive.read("report.json"))["tracking_observations"]["windows"]) == 600
        assert "dashboard.html" in archive.namelist()
        assert "tracking-observations/mlflow-lineage.json" in archive.namelist()
