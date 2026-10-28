"""Importable ZenML definitions required by SDK source resolution."""

import logging

from zenml import pipeline, step

from continual_bench.application.config import Config
from continual_bench.application.runner import evaluate, prepare

logger = logging.getLogger(__name__)


@step(enable_cache=False, enable_step_logs=True)
def materialize(config_data: dict, directory: str) -> dict:
    config = Config.model_validate(config_data)
    logger.info("Materializing dataset=%s mode=%s samples=%d", config.dataset.name, config.mode,
                config.dataset.samples)
    manifest = prepare(config, directory)
    logger.info("Materialization complete; event manifest persisted")
    return manifest


@step(enable_cache=False, enable_step_logs=True)
def benchmark(manifest: dict, config_data: dict, directory: str, use_services: bool) -> dict:
    validated = Config.model_validate(config_data)
    transport = None
    if use_services and validated.mode == "streaming":
        from continual_bench.adapters.services import KafkaTransport

        transport = KafkaTransport()
    logger.info("Benchmark started; mode=%s transport=%s", validated.mode,
                "kafka" if transport is not None else "local_jsonl")
    report = evaluate(validated, manifest, directory, transport)
    logger.info("Benchmark completed; complete=%s", report["complete"])
    return report


@step(enable_cache=False, enable_step_logs=True)
def audit_drift_and_retraining(report: dict, config_data: dict) -> dict:
    """Audit actual engine decisions; never use synthetic change points to trigger training."""
    config = Config.model_validate(config_data)
    alarms = {alarm["seq"] for alarm in report["alarms"]}
    updates = report["updates"]
    if not report["complete"]:
        raise ValueError("Cannot publish an incomplete benchmark")
    if config.policy.name == "none" and updates:
        raise ValueError("Frozen policy unexpectedly updated the model")
    if config.policy.name == "drift" and any(update["seq"] not in alarms for update in updates):
        raise ValueError("Drift policy retrained without a detector alarm")
    if config.policy.name != "online" and any(
        b["labeled"] - a["labeled"] < config.policy.cooldown for a, b in zip(updates, updates[1:])
    ):
        raise ValueError("Retraining violated the configured cooldown")
    used = sum(update["train_samples"] for update in updates)
    if config.training_budget is not None and used > config.training_budget:
        raise ValueError("Training exceeded the configured budget")
    if config.policy.name == "online" and len(updates) != config.dataset.samples - config.warmup:
        raise ValueError("Online policy did not update every post-warmup label")
    logger.info("Strategy audit: detector=%s policy=%s alarms=%d updates=%d; guards passed",
                config.detector.name, config.policy.name, len(alarms), len(updates))
    return report


@step(enable_cache=False, enable_step_logs=True)
def persist(report: dict, config_data: dict, directory: str, use_services: bool) -> dict:
    if use_services:
        from continual_bench.adapters.services import publish

        logger.info("Publishing report to the configured tracking and object stores")
        receipt = publish(Config.model_validate(config_data), report, directory)
        logger.info("Publication complete")
        return receipt
    logger.info("Report retained locally; service publication was not requested")
    return {"run_key": report["run_key"], "publication": "local_only"}


@pipeline(enable_cache=False, enable_pipeline_logs=True)
def finite_benchmark(config_data: dict, directory: str, use_services: bool):
    manifest = materialize(config_data, directory)
    report = benchmark(manifest, config_data, directory, use_services)
    audited = audit_drift_and_retraining(report, config_data)
    persist(audited, config_data, directory, use_services)
