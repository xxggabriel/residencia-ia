"""E2E epoch pipeline, immutable artifact readback, and retry evidence using in-process fakes."""

import hashlib
import json
from urllib.parse import urlsplit

import pytest

from continual_bench.evaluation.epochs import EpochConfig
from continual_bench.orchestration import epoch_pipeline
from tests.e2e._service_fakes import (
    FakeMlflowClient,
    FakeS3Objects,
    FakeZenmlClient,
    install_fake_mlflow,
    install_fake_zenml,
)


@pytest.mark.parametrize("priced_time", [False, True])
def test_epoch_pipeline_artifacts_and_publication_retry(tmp_path, monkeypatch, priced_time):
    root = tmp_path / "epoch-state"
    cache = tmp_path / "cache"
    real_dir = tmp_path / "real"
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    config = EpochConfig(
        dataset="sea",
        examples=100,
        epochs=2,
        **(
            {"costs": {"train_sample": 0.02, "update": 0.3, "train_second": 2, "predict_second": 3}}
            if priced_time
            else {}
        ),
    )

    fake_s3 = FakeS3Objects()
    fake_mlflow = FakeMlflowClient()
    monkeypatch.setattr(
        "continual_bench.application.readiness.ensure_tracking_ready",
        lambda *args, **kwargs: None,
    )

    def fake_epoch_publish(report: dict, root_str: str) -> dict:
        run_dir = root / report["run_key"]
        receipt_path = run_dir / "epoch-receipt.json"
        if receipt_path.exists():
            return json.loads(receipt_path.read_text())

        run_id = f"epoch-{report['run_key'][:12]}"
        files = {}
        artifacts = {}
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name != "epoch-receipt.json":
                raw = path.read_bytes()
                key = f"{report['run_key']}/{path.name}"
                fake_s3.client.put_object(Bucket=fake_s3.bucket, Key=key, Body=raw)
                files[path.name] = {
                    "uri": f"s3://{fake_s3.bucket}/{key}",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                artifacts[path.name] = raw

        fake_mlflow.register_run(
            run_id=run_id,
            experiment_name="epoch-pipeline",
            params={"sample_sha256": report["contract"]["samples_sha256"]},
            artifacts=artifacts,
        )
        for key, value in report["costs"]["metrics"].items():
            fake_mlflow.record_metric(run_id, "costs/total/" + key, value, step=0)
        for epoch in report["epochs"]:
            fake_mlflow.record_metric(run_id, "accuracy", epoch["metrics"]["accuracy"], step=epoch["epoch"])
            for key, value in epoch["costs"]["metrics"].items():
                fake_mlflow.record_metric(run_id, "costs/" + key, value, step=epoch["epoch"])

        receipt = {"complete": True, "mlflow_run_id": run_id, "files": files}
        receipt_path.write_text(json.dumps(receipt, indent=2))
        return receipt

    monkeypatch.setattr(epoch_pipeline, "epoch_publish", fake_epoch_publish)
    install_fake_mlflow(monkeypatch, fake_mlflow)
    install_fake_zenml(monkeypatch, FakeZenmlClient(("epoch_evaluate", "epoch_publish")))

    result = epoch_pipeline.execute_epoch_pipeline(
        config,
        str(root),
        str(cache),
        real_directory=str(real_dir),
    )
    report, receipt = result["summary"], result["receipt"]
    assert report["unique_examples"] == 100 and len(report["epochs"]) == 2
    assert receipt["complete"]
    directory = root / report["run_key"]
    for filename, entry in receipt["files"].items():
        uri = urlsplit(entry["uri"])
        assert uri.scheme == "s3" and uri.netloc == fake_s3.bucket
        raw = fake_s3.client.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))["Body"].read()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert raw == (directory / filename).read_bytes()

    from mlflow.tracking import MlflowClient
    from zenml.client import Client

    mlflow = MlflowClient()
    tracked = mlflow.get_run(receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED"
    assert tracked.data.params["sample_sha256"] == report["contract"]["samples_sha256"]
    assert {a.path for a in mlflow.list_artifacts(tracked.info.run_id)} >= set(receipt["files"])
    assert tracked.data.metrics["costs/total/cost_total"] == report["costs"]["metrics"]["cost_total"]
    for key, value in report["costs"]["metrics"].items():
        assert tracked.data.metrics["costs/total/" + key] == pytest.approx(value)
        cost_history = mlflow.get_metric_history(tracked.info.run_id, "costs/" + key)
        assert len(cost_history) == 2
        assert {point.step: point.value for point in cost_history} == pytest.approx(
            {epoch["epoch"]: epoch["costs"]["metrics"][key] for epoch in report["epochs"]}
        )
    assert report["costs"]["settings"] == config.costs.model_dump()
    history = mlflow.get_metric_history(tracked.info.run_id, "accuracy")
    assert {point.step for point in history} == {1, 2}
    assert len(history) == 2

    zenml = Client()
    run = zenml.list_pipeline_runs(project="fake-project", size=1, hydrate=True).items[0]
    assert str(run.status) == "completed"
    steps = zenml.list_run_steps(pipeline_run_id=run.id, hydrate=True).items
    assert {step.name for step in steps} == {"epoch_evaluate", "epoch_publish"}
    assert all(str(step.status) == "completed" and step.outputs for step in steps)

    repeated = epoch_pipeline.execute_epoch_pipeline(
        config,
        str(root),
        str(cache),
        real_directory=str(real_dir),
    )
    assert repeated["receipt"] == receipt
    assert len(mlflow.get_metric_history(tracked.info.run_id, "accuracy")) == len(history)
    evidence = {
        "run_key": report["run_key"],
        "mlflow_run_id": tracked.info.run_id,
        "zenml_run_id": str(run.id),
        "zenml_steps_verified": len(steps),
        "objects_sha256_verified": True,
        "publication_replay_idempotent": True,
        "costs": report["costs"],
        "mlflow_costs_verified": True,
        "code_sha256": report["contract"]["environment"]["code_sha256"],
    }
    (directory / "epoch-service-evidence.json").write_text(json.dumps(evidence, indent=2))
