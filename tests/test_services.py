"""These tests require actual services and never substitute mocks."""

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from continual_bench.adapters.services import S3Objects
from continual_bench.adapters.zenml_pipeline import execute_pipeline
from continual_bench.application.config import Config


@pytest.mark.services
@pytest.mark.parametrize("dataset", ["sea", "agrawal", "stagger", "electricity", "airlines"])
@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_real_pipeline_services(dataset, mode):
    from mlflow.tracking import MlflowClient

    root = os.environ.get("SERVICE_TEST_STATE", "/state/service-tests")
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
    report = execute_pipeline(cfg, root, services=True)
    assert report["complete"]
    assert report["transport"] == ("kafka" if mode == "streaming" else "local_jsonl")
    directory = Path(root) / cfg.identity()
    receipt = json.loads((directory / "receipt.json").read_text())
    objects = S3Objects()
    for name in ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"):
        remote = objects.client.get_object(Bucket=objects.bucket, Key=f"{cfg.identity()}/{name}")[
            "Body"
        ].read()
        assert hashlib.sha256(remote).digest() == hashlib.sha256((directory / name).read_bytes()).digest()
    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    tracked = client.get_run(receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED"
    assert tracked.info.artifact_uri.startswith("mlflow-artifacts:")
    assert client.list_artifacts(tracked.info.run_id)
    assert client.get_experiment(tracked.info.experiment_id).name == receipt["mlflow_experiment_name"]
    assert tracked.inputs.dataset_inputs[0].dataset.name == dataset
    assert tracked.data.params["dataset_version"] == report["dataset"]["sha256"]
    registered = client.get_model_version(receipt["registered_model"], receipt["registered_model_version"])
    assert registered.run_id == tracked.info.run_id
    assert registered.tags["dataset_version"] == report["dataset"]["sha256"]
    before = client.get_metric_history(tracked.info.run_id, "accuracy")
    # Replay must resume the same tracking run and avoid duplicate metric histories.
    execute_pipeline(cfg, root, services=True)
    second = json.loads((directory / "receipt.json").read_text())
    assert second["mlflow_run_id"] == receipt["mlflow_run_id"]
    assert second["registered_model_version"] == receipt["registered_model_version"]
    assert second["model_version_id"] == receipt["model_version_id"]
    assert len(client.get_metric_history(tracked.info.run_id, "accuracy")) == len(before)


@pytest.mark.services
def test_browser_run_search_accepts_only_configured_origin():
    from mlflow.tracking import MlflowClient

    experiment = MlflowClient().get_experiment_by_name("continual-bench-v1")
    assert experiment is not None
    credentials = base64.b64encode(
        (os.environ["MLFLOW_TRACKING_USERNAME"] + ":" + os.environ["MLFLOW_TRACKING_PASSWORD"]).encode()
    ).decode()
    uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/ajax-api/2.0/mlflow/runs/search"

    def request(origin):
        return urllib.request.Request(
            uri, data=json.dumps({"experiment_ids": [experiment.experiment_id], "max_results": 5}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Basic " + credentials,
                     "Origin": origin},
        )

    with urllib.request.urlopen(request("http://192.168.100.27:15000"), timeout=15) as response:
        assert response.status == 200
        assert json.load(response).get("runs")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request("https://untrusted.invalid"), timeout=15)
    assert error.value.code == 403
