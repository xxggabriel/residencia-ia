"""Real homelab epoch pipeline, immutable artifact readback and retry evidence."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from continual_bench.adapters.epoch_pipeline import execute_epoch_pipeline
from continual_bench.adapters.services import S3Objects
from continual_bench.application.epoch_benchmark import EpochConfig


@pytest.mark.services
def test_epoch_pipeline_artifacts_and_publication_retry():
    from mlflow.tracking import MlflowClient
    from zenml.client import Client

    root = Path(os.environ["SERVICE_TEST_STATE"])
    config = EpochConfig(dataset="sea", examples=100, epochs=2)
    result = execute_epoch_pipeline(
        config, str(root), os.environ["BENCH_DATASET_CACHE"],
        real_directory=os.environ["BENCH_DATASETS"],
    )
    report, receipt = result["summary"], result["receipt"]
    assert report["unique_examples"] == 100 and len(report["epochs"]) == 2
    assert receipt["complete"]
    directory = root / report["run_key"]
    objects = S3Objects()
    for filename, entry in receipt["files"].items():
        uri = urlsplit(entry["uri"])
        assert uri.scheme == "s3" and uri.netloc == objects.bucket
        raw = objects.client.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))["Body"].read()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert raw == (directory / filename).read_bytes()

    mlflow = MlflowClient()
    tracked = mlflow.get_run(receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED"
    assert tracked.data.params["sample_sha256"] == report["contract"]["samples_sha256"]
    assert {a.path for a in mlflow.list_artifacts(tracked.info.run_id)} >= set(receipt["files"])
    history = mlflow.get_metric_history(tracked.info.run_id, "accuracy")
    assert {point.step for point in history} == {1, 2}
    assert len(history) == 2

    zenml = Client()
    run = zenml.list_pipeline_runs(project=os.environ["BENCH_ZENML_PROJECT"], size=1, hydrate=True).items[0]
    assert str(run.status) == "completed"
    steps = zenml.list_run_steps(pipeline_run_id=run.id, hydrate=True).items
    assert {step.name for step in steps} == {"epoch_evaluate", "epoch_publish"}
    assert all(str(step.status) == "completed" and step.outputs for step in steps)

    repeated = execute_epoch_pipeline(
        config, str(root), os.environ["BENCH_DATASET_CACHE"],
        real_directory=os.environ["BENCH_DATASETS"],
    )
    assert repeated["receipt"] == receipt
    assert len(mlflow.get_metric_history(tracked.info.run_id, "accuracy")) == len(history)
    evidence = {"run_key": report["run_key"], "mlflow_run_id": tracked.info.run_id,
                "zenml_run_id": str(run.id), "zenml_steps_verified": len(steps),
                "objects_sha256_verified": True, "publication_replay_idempotent": True,
                "code_sha256": report["contract"]["environment"]["code_sha256"]}
    (directory / "epoch-service-evidence.json").write_text(json.dumps(evidence, indent=2))
