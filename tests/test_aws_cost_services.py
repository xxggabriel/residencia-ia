"""One real homelab run, followed only by readback of its AWS costs and artifacts."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from continual_bench.config import Config
from continual_bench.infrastructure.services import S3Objects
from continual_bench.orchestration.zenml_pipeline import execute_pipeline


def single_run_config():
    return Config.model_validate({
        "name": os.environ.get("SERVICE_TEST_RUN_NAME", "aws-cost-single-validation"),
        "mode": "streaming",
        "dataset": {"name": "sea", "samples": 180, "label_delay": 3},
        "warmup": 30, "metric_window": 30, "replay_window": 60, "max_seconds": 60,
        "policy": {"name": "periodic", "every": 30, "min_samples": 30,
                   "min_window": 15, "cooldown": 15},
        "tracking": {"probe_samples": 4, "continual_references": False,
                     "adaptation_shadows": False,
                     "costs": {"train_sample": 0.001, "update": 0.05,
                               "aws": {"vcpus": 1, "memory_gb": 2, "retention_days": 30}}},
    })


@pytest.mark.services
def test_single_run_publishes_aws_costs_to_mlflow():
    root = Path(os.environ["SERVICE_TEST_STATE"])
    config = single_run_config()
    directory = root / config.identity()
    # Refuse reuse: this test must submit exactly one newly evaluated benchmark.
    assert not (directory / "report.json").exists()
    report = execute_pipeline(config, str(root), services=True)
    verify_published_run(config, directory, report)


@pytest.mark.services
@pytest.mark.skipif(not os.environ.get("SERVICE_TEST_RUN_KEY"),
                    reason="readback requires an explicitly selected existing run")
def test_existing_run_aws_cost_readback():
    """Explicitly selected verifier: never evaluate or publish another run."""
    config = single_run_config()
    assert os.environ["SERVICE_TEST_RUN_KEY"] == config.identity()
    directory = Path(os.environ["SERVICE_TEST_STATE"]) / config.identity()
    report = json.loads((directory / "report.json").read_text())
    verify_published_run(config, directory, report)


def verify_published_run(config, directory, report):
    from confluent_kafka import Consumer, TopicPartition
    from mlflow.tracking import MlflowClient
    from zenml.client import Client

    assert report["complete"] and report["observed"] == 180
    assert report["transport"] == "kafka"
    receipt = json.loads((directory / "receipt.json").read_text())
    costs = report["tracking"]["costs"]["aws"]
    stored = (directory / "models.bin").stat().st_size + (directory / "model.pkl").stat().st_size
    assert costs["consumption"]["model_storage_bytes"] == stored > 0
    assert costs["settings"] == config.tracking.costs.aws.model_dump()
    billed = costs["consumption"]["billed_compute_seconds"]
    expected = {"cost_cpu_usd": billed / 3600 * 0.04048,
                "cost_ram_usd": 2 * billed / 3600 * 0.004445,
                "cost_model_storage_usd": stored / 1024**3 * 0.023}
    expected["cost_total_usd"] = sum(expected.values())
    assert costs["metrics"] == pytest.approx(expected, rel=1e-12, abs=1e-15)

    objects = S3Objects()
    object_hashes = {}
    for name in ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"):
        raw = objects.client.get_object(Bucket=objects.bucket,
                                        Key=f"{config.identity()}/{name}")["Body"].read()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == hashlib.sha256((directory / name).read_bytes()).hexdigest()
        object_hashes[name] = digest

    mlflow = MlflowClient()
    run_id = receipt["mlflow_run_id"]
    tracked = mlflow.get_run(run_id)
    assert tracked.info.status == "FINISHED"
    matches = mlflow.search_runs([tracked.info.experiment_id],
                                filter_string=f"tags.run_key = '{config.identity()}'")
    assert len(matches) == 1 and matches[0].info.run_id == run_id
    verified = {}
    for name, value in expected.items():
        key = "tracking_v1/summary/aws_" + name
        history = mlflow.get_metric_history(run_id, key)
        assert len(history) == 1 and history[0].step == 0
        assert history[0].value == pytest.approx(value, rel=1e-12, abs=1e-15)
        assert tracked.data.metrics[key] == pytest.approx(value, rel=1e-12, abs=1e-15)
        verified[key] = history[0].value
    assert mlflow.get_model_version(receipt["registered_model"],
                                    receipt["registered_model_version"]).run_id == run_id
    download = directory / "mlflow-readback"
    download.mkdir(exist_ok=True)
    artifact_hashes = {}
    for remote, local in (("report.json", directory / "report.json"),
                          ("plots/costs/aws_usd.svg", directory / "artifacts/plots/costs/aws_usd.svg"),
                          ("data/costs/aws_usd.json", directory / "artifacts/data/costs/aws_usd.json")):
        retrieved = Path(mlflow.download_artifacts(run_id, remote, str(download)))
        digest = hashlib.sha256(retrieved.read_bytes()).hexdigest()
        assert digest == hashlib.sha256(local.read_bytes()).hexdigest()
        artifact_hashes[remote] = digest

    zenml = Client()
    runs = zenml.list_pipeline_runs(project=os.environ["BENCH_ZENML_PROJECT"], size=2, hydrate=True)
    assert len(runs.items) == 1
    pipeline = runs.items[0]
    assert str(pipeline.status) == "completed"
    steps = zenml.list_run_steps(pipeline_run_id=pipeline.id, hydrate=True).items
    assert {step.name for step in steps} == {
        "materialize", "benchmark", "audit_drift_and_retraining", "persist"}

    assert all(str(step.status) == "completed" and step.outputs for step in steps)
    consumer = Consumer({"bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
                         "group.id": "aws-cost-readback-" + config.identity(), "enable.auto.commit": False})
    try:
        topic = "bench-" + config.identity()
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, 0), timeout=10)
        # Synthetic event timestamps can expire under the topic's retention policy.
        # The high watermark still records all appended offsets for this fresh topic.
        assert high == config.dataset.samples and 0 <= low <= high
    finally:
        consumer.close()
    evidence = {"run_key": config.identity(), "mlflow_run_id": run_id,
                "mlflow_experiment_id": tracked.info.experiment_id,
                "mlflow_experiment_name": receipt["mlflow_experiment_name"],
                "zenml_run_id": str(pipeline.id), "zenml_steps_verified": len(steps),
                "benchmark_runs_executed": 1, "publication_replayed": False,
                "aws_costs": costs, "mlflow_metrics_verified": verified,
                "s3_objects_sha256": object_hashes, "mlflow_artifacts_sha256": artifact_hashes,
                "kafka": {"topic": topic, "low": low, "high": high, "retained_records": high - low},
                "code_sha256": report["environment"]["code_sha256"],
                "registered_model": receipt["registered_model"],
                "registered_model_version": receipt["registered_model_version"]}
    (directory / "aws-service-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"run_key": config.identity(), "mlflow_run_id": run_id,
                      "mlflow_experiment_id": tracked.info.experiment_id, "aws_costs": costs["metrics"]}))
