"""Small E2E task evaluation checks using in-process service fakes."""

import hashlib
import json

import pytest

from continual_bench.application.runner import run as execute_pipeline
from continual_bench.config import Config
from continual_bench.evaluation.environment import environment
from tests.e2e._service_fakes import (
    FakeMlflowClient,
    FakeS3Objects,
    install_fake_kafka,
    install_fake_mlflow,
    write_fake_receipt,
)


@pytest.mark.parametrize("priced_time", [False, True])
@pytest.mark.parametrize(
    "source,mode,policy",
    [
        ("sea", "streaming", "performance"),
        ("hyperplane", "incremental", "performance"),
        ("stagger", "offline", "none"),
    ],
)
def test_task_evaluation_real_service_evidence(tmp_path, monkeypatch, source, mode, policy, priced_time):
    root = tmp_path / "task-services"
    root.mkdir(parents=True, exist_ok=True)
    cfg = Config.model_validate(
        {
            "name": "semana09-" + environment()["code_sha256"][:12],
            "mode": mode,
            "dataset": {"name": source, "samples": 120, "label_delay": 3},
            "warmup": 20,
            "metric_window": 20,
            "replay_window": 40,
            "max_seconds": 60,
            "policy": {
                "name": policy,
                "min_samples": 20,
                "min_window": 8,
                "cooldown": 10,
                **({"params": {"monitor_window": 10}} if mode != "offline" else {}),
            },
            "tracking": {
                "probe_samples": 4,
                "continual_references": False,
                "task_evaluation": {"horizon": 8, "gap": 1, "max_windows": 2},
                **(
                    {
                        "costs": {
                            "unit": "custom_scenario",
                            "train_sample": 0.02,
                            "update": 0.3,
                            "train_second": 2,
                            "predict_second": 3,
                        }
                    }
                    if priced_time
                    else {}
                ),
            },
            **({"promotion": {"validation_samples": 8}} if mode == "incremental" else {}),
        }
    )
    report = execute_pipeline(cfg, str(root), services=False)
    assert report["complete"] and report["observed"] == 120
    assert report["transport"] == "local_jsonl"
    assert report["task_evaluation"]["near_future"]["completed_windows"] == 2

    directory = root / cfg.identity()
    run_id = f"run-{cfg.identity()[:12]}"
    receipt = {
        "mlflow_run_id": run_id,
        "registered_model": f"continual-bench-{source}",
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

    costs = report["tracking"]["costs"]
    fake_mlflow = FakeMlflowClient()
    fake_mlflow.register_run(
        run_id=run_id,
        experiment_name=cfg.name,
        params={"dataset_sha256": report["dataset"]["sha256"]},
        registered_model=receipt["registered_model"],
        registered_model_version=receipt["registered_model_version"],
    )
    fake_mlflow.record_metric(run_id, "accuracy", report["metrics"]["accuracy"], step=0)
    for key, value in costs["metrics"].items():
        fake_mlflow.record_metric(run_id, "tracking_v1/summary/" + key, value, step=0)

    install_fake_mlflow(monkeypatch, fake_mlflow)
    install_fake_kafka(monkeypatch, cfg.dataset.samples)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "fake-kafka:9092")

    from mlflow.tracking import MlflowClient

    loaded_receipt = json.loads((directory / "receipt.json").read_text())
    for name in ("events.jsonl", "manifest.json", "report.json", "checkpoint.bin"):
        raw = fake_s3.client.get_object(Bucket=fake_s3.bucket, Key=f"{cfg.identity()}/{name}")[
            "Body"
        ].read()
        assert hashlib.sha256(raw).digest() == hashlib.sha256((directory / name).read_bytes()).digest()

    mlflow = MlflowClient()
    tracked = mlflow.get_run(loaded_receipt["mlflow_run_id"])
    assert tracked.info.status == "FINISHED"
    assert costs["metrics"]["cost_total"] is not None
    assert costs["metrics"]["cost_total"] == pytest.approx(
        sum(value for key, value in costs["metrics"].items() if key != "cost_total")
    )
    for key, value in costs["metrics"].items():
        history_cost = mlflow.get_metric_history(tracked.info.run_id, "tracking_v1/summary/" + key)
        assert len(history_cost) == 1
        assert history_cost[0].value == pytest.approx(value)
    assert costs["settings"] == cfg.tracking.costs.model_dump()
    assert tracked.data.params["dataset_sha256"] == report["dataset"]["sha256"]
    assert (
        mlflow.get_model_version(
            loaded_receipt["registered_model"], loaded_receipt["registered_model_version"]
        ).run_id
        == tracked.info.run_id
    )
    assert mlflow.list_artifacts(tracked.info.run_id)
    history = mlflow.get_metric_history(tracked.info.run_id, "accuracy")
    assert history

    kafka_evidence = None
    if mode == "streaming":
        from confluent_kafka import Consumer, TopicPartition

        consumer = Consumer(
            {
                "bootstrap.servers": "fake-kafka:9092",
                "group.id": "semana09-verify-" + cfg.identity(),
                "enable.auto.commit": False,
            }
        )
        try:
            low, high = consumer.get_watermark_offsets(
                TopicPartition("bench-" + cfg.identity(), 0),
                timeout=10,
            )
            assert high - low >= cfg.dataset.samples
            kafka_evidence = {"topic": "bench-" + cfg.identity(), "low": low, "high": high}
        finally:
            consumer.close()

    execute_pipeline(cfg, str(root), services=False)
    repeated = json.loads((directory / "receipt.json").read_text())
    for key in ("mlflow_run_id", "registered_model_version", "model_version_id"):
        assert repeated[key] == loaded_receipt[key]
    assert len(mlflow.get_metric_history(tracked.info.run_id, "accuracy")) == len(history)
    evidence = {
        "run_key": cfg.identity(),
        "source": source,
        "mode": mode,
        "policy": policy,
        "mlflow_run_id": tracked.info.run_id,
        "costs": costs,
        "mlflow_costs_verified": True,
        "kafka": kafka_evidence,
        "objects_sha256_verified": True,
        "publication_replay_idempotent": True,
    }
    (directory / "service-evidence.json").write_text(json.dumps(evidence, indent=2))
