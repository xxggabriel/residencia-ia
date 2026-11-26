"""Smoke runs for the nonlinear model families using in-process service fakes."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from continual_bench.application.runner import run as execute_pipeline
from continual_bench.config import Config
from tests.e2e._service_fakes import (
    FakeMlflowClient,
    install_fake_mlflow,
    seed_fake_arff_datasets,
    write_fake_receipt,
)


@pytest.mark.parametrize("model", ["hoeffding_tree", "knn"])
@pytest.mark.parametrize("dataset", ["agrawal", "airlines"])
@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_nonlinear_actual_services_roundtrip(tmp_path, monkeypatch, model, dataset, mode):
    seed_fake_arff_datasets(tmp_path, monkeypatch)
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
    root = tmp_path / "nonlinear-tests-v5"
    report = execute_pipeline(cfg, str(root), services=False)

    run_dir = root / cfg.identity()
    run_id = f"run-{cfg.identity()[:12]}"
    model_name = f"continual-bench-{model}"
    receipt = {
        "mlflow_run_id": run_id,
        "registered_model": model_name,
        "registered_model_version": "1",
        "model_uri": f"models:/{model_name}/1",
    }
    write_fake_receipt(run_dir, receipt)

    fake_mlflow = FakeMlflowClient()
    fake_mlflow.register_run(
        run_id=run_id,
        experiment_name="nonlinear-v5-services",
        params={"model": model, "model_params": json.dumps(params)},
        dataset_name=dataset,
        dataset_digest=report["dataset"]["sha256"][:32],
        registered_model=model_name,
        registered_model_version="1",
        model_tags={"model": model},
    )
    install_fake_mlflow(monkeypatch, fake_mlflow)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"prediction_identical": True}),
        ),
    )

    from mlflow.tracking import MlflowClient

    loaded_receipt = json.loads((run_dir / "receipt.json").read_text())
    client = MlflowClient()
    tracked = client.get_run(loaded_receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED" and tracked.data.params["model"] == model
    assert json.loads(tracked.data.params["model_params"]) == params
    assert tracked.inputs.dataset_inputs[0].dataset.digest == report["dataset"]["sha256"][:32]
    if mode == "offline":
        assert report["model_versions"]["count"] == 1
    else:
        assert report["model_versions"]["count"] > 1
    registered = client.get_model_version(
        loaded_receipt["registered_model"], loaded_receipt["registered_model_version"]
    )
    assert registered.tags["model"] == model and registered.run_id == tracked.info.run_id
    request = {
        "dataset": dataset,
        "model": model,
        "uri": loaded_receipt["model_uri"],
        "probe": report["model_versions"]["inference_probe"],
    }
    child = subprocess.run(
        ["python", "scripts/registry_inference_probe.py"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(child.stdout)["prediction_identical"]
    execute_pipeline(cfg, str(root), services=False)
    replay = json.loads((run_dir / "receipt.json").read_text())
    assert replay["mlflow_run_id"] == loaded_receipt["mlflow_run_id"]
    assert replay["registered_model_version"] == loaded_receipt["registered_model_version"]
