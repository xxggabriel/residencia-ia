"""Binary benchmark use case: materialize, evaluate with resumable checkpoints, publish.

The runner orchestrates; `evaluation.binary.protocol_for(mode)` performs the
protocol. Service I/O (Kafka transport, publication) is injected or opt-in.
"""

import hashlib
import json
import time
from pathlib import Path

from continual_bench.config import Config, TrackingConfig
from continual_bench.core.contracts import event_json
from continual_bench.evaluation.binary import BinaryRun, Progress, protocol_for, read_events
from continual_bench.evaluation.environment import environment, peak_rss_bytes
from continual_bench.evaluation.tracking import metric_catalog, save_tracking_files
from continual_bench.infrastructure.storage import LocalCheckpoints, atomic_write, run_lock
from continual_bench.infrastructure.versioning import ModelVersions

from .components import default_registry, make_source
from .plugin_identity import package_identity

__all__ = ["environment", "evaluate", "prepare", "read_events", "run"]
MATERIALIZATION_BUDGET = 256 * 1024**2


def prepare(config: Config, root: str) -> dict:
    registry = default_registry(config.plugins)
    source = make_source(registry, config.dataset)
    directory = Path(root) / config.identity()
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "events.jsonl"
    # Deterministic materialization is bounded by validated samples <= 100k.
    payload = "".join(event_json(e) + "\n" for e in source.events()).encode()
    if len(payload) > MATERIALIZATION_BUDGET:
        raise ValueError("dataset exceeds 256 MiB materialization budget")
    digest = hashlib.sha256(payload).hexdigest()
    if data_path.exists() and hashlib.sha256(data_path.read_bytes()).hexdigest() != digest:
        raise ValueError("source changed for an existing run; choose a new config/name")
    atomic_write(data_path, payload)
    manifest = {
        **source.manifest(),
        "sha256": digest,
        "path": str(data_path),
        "run_key": config.identity(),
        "protocol": config.mode,
        "dataset_version": digest,
    }
    atomic_write(directory / "manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


def resume_contract(config: Config, manifest: dict, env: dict, transport) -> dict:
    """Everything that must match for a checkpoint to be resumed or a report reused."""
    return {
        "config": config.model_dump(mode="json"),
        "source": manifest["sha256"],
        "python": env["python"],
        "packages": env["packages"],
        "code_sha256": env["code_sha256"],
        "python_hash_seed": env["python_hash_seed"],
        "metric_definitions_sha256": hashlib.sha256(metric_catalog()[1]).hexdigest(),
        "plugins": package_identity(config.plugins, config.plugin_packages),
        "transport": transport.contract()
        if config.mode == "streaming" and transport
        else {"name": "local_jsonl", "version": 1},
    }


def evaluate(config: Config, manifest: dict, root: str, transport=None, *, stop_after=None) -> dict:
    """Returns complete report or bounded progress; service I/O is injected."""
    registry = default_registry(config.plugins)
    source = make_source(registry, config.dataset)
    identity = config.identity()
    directory = Path(root) / identity
    checkpoints = LocalCheckpoints(root)
    env = environment()
    contract = resume_contract(config, manifest, env, transport)
    events_path = Path(manifest["path"])
    if hashlib.sha256(events_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("dataset manifest checksum mismatch")
    with run_lock(root, identity):
        snapshot = checkpoints.load(identity)
        if snapshot and snapshot["contract"] != contract:
            raise ValueError("resume environment/config/dataset/transport/plugin mismatch")
        report_path = directory / "report.json"
        if snapshot and snapshot.get("complete"):
            return json.loads(report_path.read_text())
        started = time.monotonic()
        settings = (config.tracking or TrackingConfig()).model_dump(mode="json")
        protocol = protocol_for(config.mode, config.batch_size)
        run = BinaryRun(
            config=config,
            source=source,
            events_path=events_path,
            model_factory=registry.factory("model", config.model.name, **config.model.params),
            detector_factory=registry.factory("detector", config.detector.name, **config.detector.params),
            policy_factory=registry.factory("policy", config.policy.name,
                                            **config.policy.model_dump(exclude={"name"})),
            versions=ModelVersions(directory, manifest["sha256"], env["code_sha256"]),
            settings=settings,
            started=started,
            checkpoint=lambda state: checkpoints.save(
                identity, {"contract": contract, "state": state, "complete": False}),
            snapshot=snapshot,
            transport=transport,
            manifest=manifest,
            stop_after=stop_after,
        )
        outcome = protocol.execute(run)
        if isinstance(outcome, Progress):
            return {"complete": False, "observed": outcome.observed, "run_key": identity}
        report = outcome.report
        probe = next(read_events(events_path)).x
        report["model_versions"] = outcome.versions.finish(outcome.model, probe)
        report.update(
            {
                "complete": True,
                "run_key": identity,
                "config": config.model_dump(mode="json"),
                "protocol": protocol.name,
                "transport": contract["transport"]["name"],
                "environment": env,
                "dataset": manifest,
                "wall_seconds_this_attempt": time.monotonic() - started,
                "peak_rss_bytes_process": peak_rss_bytes(),
            }
        )
        save_tracking_files(directory, report, settings, outcome.observer)
        if settings.get("task_evaluation") is not None:
            from continual_bench.evaluation.reporting import supplemental_report

            report["task_evaluation"] = supplemental_report(
                report, directory, near_future=outcome.observer.near_future
            )
        # Report first, then complete checkpoint: either crash point is safe to recompute/reuse.
        atomic_write(report_path, json.dumps(report, indent=2).encode())
        checkpoints.save(identity, {"contract": contract, "state": outcome.state, "complete": True})
        return report


def run(config: Config, root="state", *, services=False):
    if services:
        from .readiness import ensure_tracking_ready

        ensure_tracking_ready()
    manifest = prepare(config, root)
    transport = None
    if services and config.mode == "streaming":
        from continual_bench.infrastructure.services import KafkaTransport

        transport = KafkaTransport()
    report = evaluate(config, manifest, root, transport)
    if services:
        from continual_bench.infrastructure.services import publish

        publish(config, report, root)
    return report
