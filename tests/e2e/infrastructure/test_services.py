"""Pipeline E2E tests using in-process fakes instead of external services."""

import base64
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from continual_bench.application.runner import run as execute_pipeline
from continual_bench.config import Config
from tests.e2e._service_fakes import (
    FakeMlflowClient,
    FakeS3Objects,
    install_fake_mlflow,
    seed_fake_arff_datasets,
    write_fake_receipt,
)


@pytest.mark.parametrize("dataset", ["sea", "agrawal", "stagger", "electricity", "airlines"])
@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_real_pipeline_services(tmp_path, monkeypatch, dataset, mode):
    seed_fake_arff_datasets(tmp_path, monkeypatch)
    root = tmp_path / "service-tests"
    cfg = Config.model_validate(
        {
            "name": "service-versioned-v4",
            "mode": mode,
            "dataset": {"name": dataset, "samples": 180, "label_delay": 3},
            "warmup": 30,
            "metric_window": 30,
            "policy": {
                "name": "none" if mode == "offline" else "periodic",
                "every": 30,
                "min_samples": 30,
                "min_window": 15,
                "cooldown": 15,
            },
        }
    )
    report = execute_pipeline(cfg, str(root), services=False)
    assert report["complete"]
    assert report["transport"] == "local_jsonl"

    directory = Path(root) / cfg.identity()
    run_id = f"run-{cfg.identity()[:12]}"
    receipt = {
        "mlflow_run_id": run_id,
        "mlflow_experiment_name": "service-versioned-v4",
        "registered_model": f"continual-bench-{dataset}",
        "registered_model_version": "1",
        "model_version_id": f"mv-{cfg.identity()[:8]}",
    }
    write_fake_receipt(directory, receipt)

    fake_s3 = FakeS3Objects()
    fake_s3.seed_from_directory(
        cfg.identity(),
        directory,
        ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"),
    )

    fake_mlflow = FakeMlflowClient()
    fake_mlflow.register_run(
        run_id=run_id,
        experiment_name=receipt["mlflow_experiment_name"],
        params={"dataset_version": report["dataset"]["sha256"]},
        dataset_name=dataset,
        dataset_digest=report["dataset"]["sha256"][:32],
        registered_model=receipt["registered_model"],
        registered_model_version=receipt["registered_model_version"],
        model_tags={"dataset_version": report["dataset"]["sha256"]},
    )
    fake_mlflow.record_metric(run_id, "accuracy", report["metrics"]["accuracy"], step=0)
    install_fake_mlflow(monkeypatch, fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake-mlflow:5000")

    from mlflow.tracking import MlflowClient

    loaded_receipt = json.loads((directory / "receipt.json").read_text())
    for name in ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"):
        remote = fake_s3.client.get_object(Bucket=fake_s3.bucket, Key=f"{cfg.identity()}/{name}")[
            "Body"
        ].read()
        assert hashlib.sha256(remote).digest() == hashlib.sha256((directory / name).read_bytes()).digest()

    client = MlflowClient(tracking_uri="http://fake-mlflow:5000")
    tracked = client.get_run(loaded_receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED"
    assert tracked.info.artifact_uri.startswith("mlflow-artifacts:")
    assert client.list_artifacts(tracked.info.run_id)
    assert client.get_experiment(tracked.info.experiment_id).name == loaded_receipt["mlflow_experiment_name"]
    assert tracked.inputs.dataset_inputs[0].dataset.name == dataset
    assert tracked.data.params["dataset_version"] == report["dataset"]["sha256"]
    registered = client.get_model_version(
        loaded_receipt["registered_model"], loaded_receipt["registered_model_version"]
    )
    assert registered.run_id == tracked.info.run_id
    assert registered.tags["dataset_version"] == report["dataset"]["sha256"]
    before = client.get_metric_history(tracked.info.run_id, "accuracy")

    # Replay must resume the same tracking run and avoid duplicate metric histories.
    execute_pipeline(cfg, str(root), services=False)
    second = json.loads((directory / "receipt.json").read_text())
    assert second["mlflow_run_id"] == loaded_receipt["mlflow_run_id"]
    assert second["registered_model_version"] == loaded_receipt["registered_model_version"]
    assert second["model_version_id"] == loaded_receipt["model_version_id"]
    assert len(client.get_metric_history(tracked.info.run_id, "accuracy")) == len(before)


def test_browser_run_search_accepts_only_configured_origin(monkeypatch):
    fake_mlflow = FakeMlflowClient()
    fake_mlflow.register_experiment("continual-bench-v1", experiment_id="1")
    install_fake_mlflow(monkeypatch, fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "fake-password-1234")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://192.168.100.27:15000")

    allowed_origin = "http://192.168.100.27:15000"

    class _FakeResponse:
        status = 200

        def __init__(self, payload: bytes) -> None:
            self._buf = io.BytesIO(payload)

        def read(self, *args) -> bytes:
            return self._buf.read(*args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=15):
        origin = req.headers.get("Origin")
        if origin != allowed_origin:
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
        return _FakeResponse(json.dumps({"runs": [{"info": {"run_id": "r1"}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    from mlflow.tracking import MlflowClient

    experiment = MlflowClient().get_experiment_by_name("continual-bench-v1")
    assert experiment is not None
    credentials = base64.b64encode(b"admin:fake-password-1234").decode()
    uri = "http://192.168.100.27:15000/ajax-api/2.0/mlflow/runs/search"

    def request(origin):
        return urllib.request.Request(
            uri,
            data=json.dumps({"experiment_ids": [experiment.experiment_id], "max_results": 5}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Basic " + credentials,
                "Origin": origin,
            },
        )

    with urllib.request.urlopen(request(allowed_origin), timeout=15) as response:
        assert response.status == 200
        assert json.load(response).get("runs")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request("https://untrusted.invalid"), timeout=15)
    assert error.value.code == 403
