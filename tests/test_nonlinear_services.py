"""Real-service smoke runs for the new model families, without altering historical runs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from continual_bench.adapters.zenml_pipeline import execute_pipeline
from continual_bench.application.config import Config


@pytest.mark.services
@pytest.mark.parametrize("model", ["hoeffding_tree", "knn"])
@pytest.mark.parametrize("dataset", ["agrawal", "airlines"])
@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_nonlinear_actual_services_roundtrip(model, dataset, mode):
    from mlflow.tracking import MlflowClient

    params = (
        {
            "grace_period": 50,
            "max_depth": 8,
            "max_size": 8,
            "memory_estimate_period": 100,
            "leaf_prediction": "nba",
        }
        if model == "hoeffding_tree"
        else {"n_neighbors": 11, "window_size": 256, "weighted": True}
    )
    if model == "hoeffding_tree" and dataset == "agrawal":
        params["nominal_attributes"] = ["elevel", "car", "zipcode"]
    cfg = Config.model_validate(
        {
            "name": "nonlinear-v5-services",
            "mode": mode,
            "dataset": {"name": dataset, "samples": 180, "label_delay": 3},
            "warmup": 30,
            "model": {"name": model, "params": params},
            "detector": {"name": "none"},
            "policy": {
                "name": "none" if mode == "offline" else "online",
                "action": "reset_retrain" if mode == "offline" else "learn_one",
            },
            "training_budget": None if mode == "offline" else 150,
        }
    )
    root = Path(os.environ.get("SERVICE_TEST_STATE", "/state/nonlinear-tests-v5"))
    report = execute_pipeline(cfg, str(root), services=True)
    receipt = json.loads((root / cfg.identity() / "receipt.json").read_text())
    client = MlflowClient()
    tracked = client.get_run(receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED" and tracked.data.params["model"] == model
    assert json.loads(tracked.data.params["model_params"]) == params
    assert tracked.inputs.dataset_inputs[0].dataset.digest == report["dataset"]["sha256"][:32]
    assert report["model_versions"]["count"] == (1 if mode == "offline" else 151)
    registered = client.get_model_version(receipt["registered_model"], receipt["registered_model_version"])
    assert registered.tags["model"] == model and registered.run_id == tracked.info.run_id
    request = {
        "dataset": dataset,
        "model": model,
        "uri": receipt["model_uri"],
        "probe": report["model_versions"]["inference_probe"],
    }
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/registry_inference_probe.py")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(child.stdout)["prediction_identical"]
    execute_pipeline(cfg, str(root), services=True)
    replay = json.loads((root / cfg.identity() / "receipt.json").read_text())
    assert replay["mlflow_run_id"] == receipt["mlflow_run_id"]
    assert replay["registered_model_version"] == receipt["registered_model_version"]
