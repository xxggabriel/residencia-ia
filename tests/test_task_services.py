"""Small real-service checks. Run only on the authorized homelab client container."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from continual_bench.adapters.services import S3Objects
from continual_bench.adapters.zenml_pipeline import execute_pipeline
from continual_bench.application.config import Config
from continual_bench.application.runner import environment


@pytest.mark.services
@pytest.mark.parametrize("priced_time", [False, True])
@pytest.mark.parametrize("source,mode,policy", [
    ("sea", "streaming", "cost_heuristic"),
    ("hyperplane", "incremental", "performance"),
    ("stagger", "offline", "none"),
])
def test_task_evaluation_real_service_evidence(source, mode, policy, priced_time):
    from mlflow.tracking import MlflowClient
    from zenml.client import Client

    root = Path(os.environ["SERVICE_TEST_STATE"])
    cfg = Config.model_validate({
        "name": os.environ.get("SERVICE_TEST_RUN_NAME", "semana09-" + environment()["code_sha256"][:12]),
        "mode": mode,
        "dataset": {"name": source, "samples": 120, "label_delay": 3},
        "warmup": 20, "metric_window": 20, "replay_window": 40, "max_seconds": 60,
        "policy": {"name": policy, "min_samples": 20, "min_window": 8, "cooldown": 10,
                   **({"params": {"monitor_window": 10}} if mode != "offline" else {})},
        "tracking": {"probe_samples": 4, "continual_references": False,
                     "task_evaluation": {"horizon": 8, "gap": 1, "max_windows": 2},
                     **({"costs": {"unit": "custom_scenario", "train_sample": 0.02, "update": 0.3,
                                   "train_second": 2, "predict_second": 3}} if priced_time else {})},
        **({"promotion": {"validation_samples": 8}} if mode == "incremental" else {}),
    })
    report = execute_pipeline(cfg, str(root), services=True)
    assert report["complete"] and report["observed"] == 120
    assert report["transport"] == ("kafka" if mode == "streaming" else "local_jsonl")
    assert report["task_evaluation"]["near_future"]["completed_windows"] == 2
    directory = root / cfg.identity()
    receipt = json.loads((directory / "receipt.json").read_text())
    objects = S3Objects()
    for name in ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"):
        raw = objects.client.get_object(Bucket=objects.bucket, Key=f"{cfg.identity()}/{name}")["Body"].read()
        assert hashlib.sha256(raw).digest() == hashlib.sha256((directory / name).read_bytes()).digest()

    mlflow = MlflowClient()
    tracked = mlflow.get_run(receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED"
    costs = report["tracking"]["costs"]
    assert costs["metrics"]["cost_total"] is not None
    assert costs["metrics"]["cost_total"] == pytest.approx(sum(
        value for key, value in costs["metrics"].items() if key != "cost_total"
    ))
    for key, value in costs["metrics"].items():
        history_cost = mlflow.get_metric_history(tracked.info.run_id, "tracking_v1/summary/" + key)
        assert len(history_cost) == 1
        assert history_cost[0].value == pytest.approx(value)
    assert costs["settings"] == cfg.tracking.costs.model_dump()

    assert tracked.data.params["dataset_sha256"] == report["dataset"]["sha256"]
    assert mlflow.get_model_version(receipt["registered_model"], receipt["registered_model_version"]).run_id \
        == tracked.info.run_id
    assert mlflow.list_artifacts(tracked.info.run_id)
    history = mlflow.get_metric_history(tracked.info.run_id, "accuracy")
    assert history

    zenml = Client()
    runs = zenml.list_pipeline_runs(project=os.environ["BENCH_ZENML_PROJECT"], size=1, hydrate=True)
    latest = runs.items[0]
    assert str(latest.status) == "completed"
    steps = zenml.list_run_steps(pipeline_run_id=latest.id, hydrate=True).items
    assert {step.name for step in steps} == {
        "materialize", "benchmark", "audit_drift_and_retraining", "persist",
    }
    assert all(str(step.status) == "completed" and step.outputs for step in steps)

    kafka_evidence = None
    if mode == "streaming":
        from confluent_kafka import Consumer, TopicPartition

        consumer = Consumer({"bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
                             "group.id": "semana09-verify-" + cfg.identity(), "enable.auto.commit": False})
        try:
            low, high = consumer.get_watermark_offsets(
                TopicPartition("bench-" + cfg.identity(), 0), timeout=10,
            )
            assert high - low >= cfg.dataset.samples
            kafka_evidence = {"topic": "bench-" + cfg.identity(), "low": low, "high": high}
        finally:
            consumer.close()

    execute_pipeline(cfg, str(root), services=True)
    repeated = json.loads((directory / "receipt.json").read_text())
    for key in ("mlflow_run_id", "registered_model_version", "model_version_id"):
        assert repeated[key] == receipt[key]
    assert len(mlflow.get_metric_history(tracked.info.run_id, "accuracy")) == len(history)
    evidence = {"run_key": cfg.identity(), "source": source, "mode": mode, "policy": policy,
                "mlflow_run_id": tracked.info.run_id, "zenml_run_id": str(latest.id),
                "costs": costs, "mlflow_costs_verified": True,
                "steps_verified": len(steps), "kafka": kafka_evidence,
                "objects_sha256_verified": True, "publication_replay_idempotent": True}
    (directory / "service-evidence.json").write_text(json.dumps(evidence, indent=2))
