"""Real adapters: errors propagate; no local fallback masquerades as a service test."""

import json
import os
import time
from pathlib import Path

from continual_bench.domain.contracts import Event

from .catalog import publish_catalog, register_final_model, strategy_name
from .metric_tracking import log_metric_points, publish_tracking


class KafkaTransport:
    def __init__(self):
        self.bootstrap = os.environ["KAFKA_BOOTSTRAP_SERVERS"]

    def contract(self):
        return {"name": "kafka", "version": 1, "bootstrap": self.bootstrap}

    def events(self, manifest, source, deadline):
        from confluent_kafka import OFFSET_BEGINNING, Consumer, Producer, TopicPartition
        from confluent_kafka.admin import AdminClient, NewTopic

        topic = "bench-" + manifest["run_key"]
        admin = AdminClient({"bootstrap.servers": self.bootstrap})
        if topic not in admin.list_topics(timeout=15).topics:
            future = admin.create_topics(
                [
                    NewTopic(
                        topic,
                        num_partitions=1,
                        replication_factor=1,
                        config={"retention.ms": "86400000", "retention.bytes": "268435456"},
                    )
                ]
            )[topic]
            future.result(20)
        producer = Producer(
            {
                "bootstrap.servers": self.bootstrap,
                "enable.idempotence": True,
                "acks": "all",
                "message.timeout.ms": 30000,
                "queue.buffering.max.kbytes": 16384,
            }
        )
        errors = []

        def delivered(error, _message):
            if error:
                errors.append(str(error))

        began = time.monotonic()
        for event in source:
            if time.monotonic() - began > deadline:
                raise TimeoutError("Kafka publishing deadline exceeded")
            while True:
                try:
                    from continual_bench.adapters.river import event_json

                    producer.produce(
                        topic,
                        partition=0,
                        key=event.event_id,
                        value=event_json(event),
                        timestamp=event.timestamp * 1000,
                        on_delivery=delivered,
                    )
                    break
                except BufferError:
                    producer.poll(0.05)
            producer.poll(0)
        remaining = producer.flush(30)
        if remaining or errors:
            raise RuntimeError(f"Kafka delivery failed: pending={remaining} errors={errors[:3]}")
        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap,
                "group.id": "replay-" + manifest["run_key"],
                "enable.auto.commit": False,
                "isolation.level": "read_committed",
            }
        )
        consumer.assign([TopicPartition(topic, 0, OFFSET_BEGINNING)])
        next_seq = 0
        expected_file = Path(manifest["path"]).open()
        last_message = time.monotonic()
        try:
            while next_seq < manifest["samples"]:
                if time.monotonic() - began > deadline or time.monotonic() - last_message > 30:
                    raise TimeoutError(f"Kafka consumption stalled at {next_seq}")
                message = consumer.poll(1)
                if message is None:
                    continue
                if message.error():
                    raise RuntimeError(str(message.error()))
                last_message = time.monotonic()
                event = Event(**json.loads(message.value()))
                if message.key().decode() != event.event_id:
                    raise ValueError("Kafka key/event ID mismatch")
                if event.seq < next_seq:
                    continue
                if event.seq != next_seq:
                    raise ValueError("Kafka event order broken")
                from continual_bench.adapters.river import event_json

                if event_json(event) != next(expected_file).strip():
                    raise ValueError("Kafka payload differs from immutable source manifest")
                next_seq += 1
                yield event
        finally:
            consumer.close()
            expected_file.close()


class S3Objects:
    def __init__(self):
        import boto3
        from botocore.config import Config

        self.bucket = os.environ.get("BENCH_BUCKET", "benchmarks")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            config=Config(
                connect_timeout=10,
                read_timeout=30,
                retries={"max_attempts": 3},
                s3={"addressing_style": "path"},
            ),
        )

    def upload(self, key, path):
        self.client.upload_file(str(path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"


class MLflowTracker:
    def publish(self, identity, config, report, files, catalog, objects=None):
        import mlflow
        from mlflow.entities import Param, RunTag
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = MlflowClient()
        name = strategy_name(config)
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            from mlflow.exceptions import MlflowException

            try:
                experiment_id = client.create_experiment(name)
            except MlflowException as error:
                if error.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
                experiment_id = client.get_experiment_by_name(name).experiment_id
        else:
            experiment_id = experiment.experiment_id
        matches = client.search_runs([experiment_id], f"tags.run_key = '{identity}'")
        if len(matches) > 1:
            raise RuntimeError("duplicate remote MLflow run_key; manual reconciliation required")
        run = (
            matches[0]
            if matches
            else client.create_run(
                experiment_id,
                tags={
                    "mlflow.runName": "-".join((config["dataset"]["name"], config["model"]["name"],
                                                config["mode"], config["detector"]["name"],
                                                config["policy"]["name"],
                                                str(config["dataset"]["seed"]))),
                    "run_key": identity,
                    "protocol": report["protocol"],
                    "dataset": config["dataset"]["name"],
                    "strategy": name,
                },
            )
        )
        run_id = run.info.run_id
        # Deterministic metric keys/steps; inspect histories before append to tolerate partial publication.
        params = {"summary_" + key: value for key, value in report["metrics"].items() if value is not None}
        params.update({key: config[key] for key in ("mode", "name", "protocol_version")})
        params.update({
            "seed": config["dataset"]["seed"],
            "policy": config["policy"]["name"],
            "detector": config["detector"]["name"],
            "dataset_sha256": report["dataset"]["sha256"],
            "code_sha256": report["environment"]["code_sha256"],
            "samples": config["dataset"]["samples"],
            "training_budget": config.get("training_budget"),
            "policy_every": config["policy"]["every"],
            "policy_cooldown": config["policy"]["cooldown"],
            "dataset": config["dataset"]["name"],
            "dataset_version": catalog["dataset_version"],
            "dataset_source": report["dataset"]["source"],
            "dataset_source_version": report["dataset"]["version"],
            "dataset_license": report["dataset"]["license"],
            "model_version_id": catalog["model_version_id"],
            "model_state_sha256": catalog["model_state_sha256"],
            "model_versions_count": catalog["model_versions_count"],
            "model": config["model"]["name"],
            "model_params": json.dumps(config["model"]["params"], sort_keys=True),
        })
        client.log_batch(run_id, params=[Param(key, str(value)) for key, value in params.items()],
                         synchronous=True)
        metrics = [(key, window["end_seq"], value)
                   for window in report["windows"] for key, value in window.items()
                   if key != "end_seq" and value is not None]
        log_metric_points(client, run_id, metrics)
        for path in files:
            client.log_artifact(run_id, path)
        from mlflow.entities import Dataset, DatasetInput, InputTag

        if run.inputs is None or not run.inputs.dataset_inputs:
            from mlflow.models import infer_signature

            from .model_loader import inference_frame

            schema = infer_signature(inference_frame(report["model_versions"]["inference_probe"]["features"]))
            dataset = Dataset(name=config["dataset"]["name"], digest=catalog["dataset_version"][:32],
                              source_type="s3", source=json.dumps({"uri": catalog["dataset_uri"]}),
                              schema=schema.inputs.to_json(),
                              profile=json.dumps({"num_rows": config["dataset"]["samples"],
                                                  "num_features": len(schema.inputs.inputs)}))
            client.log_inputs(run_id, datasets=[DatasetInput(dataset, tags=[
                InputTag("mlflow.data.context", report["protocol"]),
                InputTag("dataset_sha256", catalog["dataset_version"]),
                InputTag("label_delay", str(config["dataset"]["label_delay"])),
            ])])
        registry = register_final_model(client, run, config, report, Path(files[0]).parent, catalog)
        summaries = [("summary_" + key, 0, value)
                     for key, value in report["metrics"].items() if value is not None]
        used = sum(update["train_samples"] for update in report["updates"])
        summaries.append(("training_used", 0, used))
        log_metric_points(client, run_id, summaries)
        client.log_batch(run_id, tags=[RunTag("update_count", str(len(report["updates"]))),
                               RunTag("alarm_count", str(len(report["alarms"]))),
                               RunTag("campaign", config["name"]),
                               RunTag("registered_model", registry["registered_model"]),
                               RunTag("registered_model_version", registry["registered_model_version"])],
                         synchronous=True)
        tracking = publish_tracking(client, run_id, report, Path(files[0]).parent, objects=objects)
        client.set_terminated(run_id, "FINISHED")
        return {"mlflow_run_id": run_id, "mlflow_experiment_id": experiment_id,
                "mlflow_experiment_name": name, **registry, **({"tracking": tracking} if tracking else {})}


def publish(config, report, root):
    from continual_bench.adapters.local import atomic_write, run_lock

    identity = config.identity()
    directory = Path(root) / identity
    # Same lock across both sinks; a single worker replica owns publication.
    with run_lock(root, identity):
        files = [
            str(directory / file)
            for file in ("report.json", "manifest.json", "checkpoint.bin", "events.jsonl")
        ]
        objects = S3Objects()
        catalog = publish_catalog(objects, report, directory)
        atomic_write(directory / "catalog-links.json", json.dumps(catalog, indent=2).encode())
        uris = [objects.upload(f"{identity}/{Path(path).name}", path) for path in files]
        metadata_files = [directory / name for name in ("model-versions.json", "model-metadata.json",
                                                       "dataset-catalog.json", "catalog-links.json")]
        atomic_write(directory / "config.json", config.model_dump_json(indent=2).encode())
        atomic_write(directory / "environment.json", json.dumps(report["environment"], indent=2).encode())
        metadata_files += [directory / "config.json", directory / "environment.json"]
        tracked = MLflowTracker().publish(identity, config.model_dump(mode="json"), report,
                                          files[:2] + [str(p) for p in metadata_files], catalog,
                                          objects=objects)
        receipt = {"run_key": identity, **tracked, **catalog, "objects": uris}
        atomic_write(directory / "receipt.json", json.dumps(receipt, indent=2).encode())
        return receipt
