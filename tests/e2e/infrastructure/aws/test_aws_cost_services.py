"""AWS cost and artifact E2E test using in-process service fakes."""

import hashlib
import json
from pathlib import Path

import pytest

from continual_bench.application.runner import run as execute_pipeline
from continual_bench.config import Config
from tests.e2e._service_fakes import (
    FakeMlflowClient,
    FakeS3Objects,
    install_fake_kafka,
    install_fake_mlflow,
    write_fake_receipt,
)


def single_run_config():
    return Config.model_validate(
        {
            "name": "aws-cost-single-validation",
            "mode": "streaming",
            "dataset": {"name": "sea", "samples": 180, "label_delay": 3},
            "warmup": 30,
            "metric_window": 30,
            "replay_window": 60,
            "max_seconds": 60,
            "policy": {"name": "periodic", "every": 30, "min_samples": 30, "min_window": 15, "cooldown": 15},
            "tracking": {
                "probe_samples": 4,
                "continual_references": False,
                "adaptation_shadows": False,
                "costs": {
                    "train_sample": 0.001,
                    "update": 0.05,
                    "aws": {"vcpus": 1, "memory_gb": 2, "retention_days": 30},
                },
            },
        }
    )


def test_single_run_publishes_aws_costs_to_mlflow(tmp_path, monkeypatch):
    root = tmp_path / "aws-cost-state"
    root.mkdir(parents=True, exist_ok=True)
    config = single_run_config()
    directory = root / config.identity()
    assert not (directory / "report.json").exists()
    report = execute_pipeline(config, str(root), services=False)

    run_id = f"run-{config.identity()[:12]}"
    receipt = {
        "mlflow_run_id": run_id,
        "mlflow_experiment_name": "aws-cost-single-validation",
        "registered_model": "continual-bench-sea",
        "registered_model_version": "1",
    }
    write_fake_receipt(directory, receipt)

    fake_s3 = FakeS3Objects()
    fake_s3.seed_from_directory(
        config.identity(),
        directory,
        ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"),
    )

    costs = report["tracking"]["costs"]["aws"]
    mlflow_artifacts = {
        "report.json": (directory / "report.json").read_bytes(),
        "plots/costs/aws_usd.svg": (directory / "artifacts/plots/costs/aws_usd.svg").read_bytes(),
        "data/costs/aws_usd.json": (directory / "artifacts/data/costs/aws_usd.json").read_bytes(),
    }
    fake_mlflow = FakeMlflowClient()
    fake_mlflow.register_run(
        run_id=run_id,
        experiment_name=receipt["mlflow_experiment_name"],
        tags={"run_key": config.identity()},
        registered_model=receipt["registered_model"],
        registered_model_version=receipt["registered_model_version"],
        artifacts=mlflow_artifacts,
    )
    for name, value in costs["metrics"].items():
        fake_mlflow.record_metric(run_id, "tracking_v1/summary/aws_" + name, value, step=0)

    install_fake_mlflow(monkeypatch, fake_mlflow)
    install_fake_kafka(monkeypatch, config.dataset.samples)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "fake-kafka:9092")

    verify_published_run(config, directory, report, fake_s3)


def verify_published_run(config, directory: Path, report: dict, objects: FakeS3Objects) -> None:
    import os

    from confluent_kafka import Consumer, TopicPartition
    from mlflow.tracking import MlflowClient

    assert report["complete"] and report["observed"] == 180
    receipt = json.loads((directory / "receipt.json").read_text())
    costs = report["tracking"]["costs"]["aws"]
    stored = (directory / "models.bin").stat().st_size + (directory / "model.pkl").stat().st_size
    assert costs["consumption"]["model_storage_bytes"] == stored > 0
    assert costs["settings"] == config.tracking.costs.aws.model_dump()
    billed = costs["consumption"]["billed_compute_seconds"]
    expected = {
        "cost_cpu_usd": billed / 3600 * 0.04048,
        "cost_ram_usd": 2 * billed / 3600 * 0.004445,
        "cost_model_storage_usd": stored / 1024**3 * 0.023,
    }
    expected["cost_total_usd"] = sum(expected.values())
    assert costs["metrics"] == pytest.approx(expected, rel=1e-12, abs=1e-15)

    object_hashes = {}
    for name in ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"):
        raw = objects.client.get_object(Bucket=objects.bucket, Key=f"{config.identity()}/{name}")[
            "Body"
        ].read()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == hashlib.sha256((directory / name).read_bytes()).digest().hex()
        object_hashes[name] = digest

    mlflow = MlflowClient()
    run_id = receipt["mlflow_run_id"]
    tracked = mlflow.get_run(run_id)
    assert tracked.info.status == "FINISHED"
    matches = mlflow.search_runs(
        [tracked.info.experiment_id], filter_string=f"tags.run_key = '{config.identity()}'"
    )
    assert len(matches) == 1 and matches[0].info.run_id == run_id
    verified = {}
    for name, value in expected.items():
        key = "tracking_v1/summary/aws_" + name
        history = mlflow.get_metric_history(run_id, key)
        assert len(history) == 1 and history[0].step == 0
        assert history[0].value == pytest.approx(value, rel=1e-12, abs=1e-15)
        assert tracked.data.metrics[key] == pytest.approx(value, rel=1e-12, abs=1e-15)
        verified[key] = history[0].value
    assert (
        mlflow.get_model_version(receipt["registered_model"], receipt["registered_model_version"]).run_id
        == run_id
    )
    download = directory / "mlflow-readback"
    download.mkdir(exist_ok=True)
    artifact_hashes = {}
    for remote, local in (
        ("report.json", directory / "report.json"),
        ("plots/costs/aws_usd.svg", directory / "artifacts/plots/costs/aws_usd.svg"),
        ("data/costs/aws_usd.json", directory / "artifacts/data/costs/aws_usd.json"),
    ):
        retrieved = Path(mlflow.download_artifacts(run_id, remote, str(download)))
        digest = hashlib.sha256(retrieved.read_bytes()).hexdigest()
        assert digest == hashlib.sha256(local.read_bytes()).hexdigest()
        artifact_hashes[remote] = digest

    consumer = Consumer(
        {
            "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
            "group.id": "aws-cost-readback-" + config.identity(),
            "enable.auto.commit": False,
        }
    )
    try:
        topic = "bench-" + config.identity()
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, 0), timeout=10)
        assert high == config.dataset.samples and 0 <= low <= high
    finally:
        consumer.close()

    evidence = {
        "run_key": config.identity(),
        "mlflow_run_id": run_id,
        "mlflow_experiment_id": tracked.info.experiment_id,
        "mlflow_experiment_name": receipt["mlflow_experiment_name"],
        "benchmark_runs_executed": 1,
        "publication_replayed": False,
        "aws_costs": costs,
        "mlflow_metrics_verified": verified,
        "s3_objects_sha256": object_hashes,
        "mlflow_artifacts_sha256": artifact_hashes,
        "kafka": {"topic": topic, "low": low, "high": high, "retained_records": high - low},
        "code_sha256": report["environment"]["code_sha256"],
        "registered_model": receipt["registered_model"],
        "registered_model_version": receipt["registered_model_version"],
    }
    (directory / "aws-service-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
