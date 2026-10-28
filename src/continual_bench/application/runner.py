import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import resource
import time
from pathlib import Path

from continual_bench.adapters.local import LocalCheckpoints, atomic_write, run_lock
from continual_bench.adapters.river import event_json
from continual_bench.adapters.versioning import ModelVersions
from continual_bench.domain.continual import ContinualEvaluation
from continual_bench.domain.contracts import Event
from continual_bench.domain.engine import Engine
from continual_bench.domain.metrics import Metrics
from continual_bench.domain.observability import OnlineObservability

from .config import Config, TrackingConfig
from .plugin_identity import package_identity
from .registry import default_registry, make_source
from .tracking import metric_catalog, save_tracking_files


def current_rss_bytes():
    """Linux current RSS; peak RSS is not substituted for an integral elsewhere."""
    path = Path("/proc/self/statm")
    if not path.exists():
        return None
    try:
        return int(path.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def continual_evaluation(source, factory, settings):
    if not hasattr(source, "generator") or not hasattr(source, "boundaries"):
        return None
    probes = {}
    for task, function in enumerate(source.functions):
        generator = source.generator(function, source.seed + 2_000_000 + task)
        probes[task] = [({str(k): v for k, v in x.items()}, bool(y))
                        for x, y in itertools.islice(generator, settings["probe_samples"])]
    return ContinualEvaluation(factory, probes, source.boundaries + [source.samples],
                               reference_budget=settings["reference_budget"],
                               references_enabled=settings["continual_references"])


def read_events(path):
    with Path(path).open() as f:
        for line in f:
            yield Event(**json.loads(line))


def prepare(config: Config, root: str) -> dict:
    registry = default_registry(config.plugins)
    source = make_source(registry, config.dataset)
    directory = Path(root) / config.identity()
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "events.jsonl"
    # Deterministic materialization is bounded by validated samples <= 100k.
    payload = "".join(event_json(e) + "\n" for e in source.events()).encode()
    if len(payload) > 256 * 1024**2:
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


def environment():
    versions = {}
    for package in ("continual-bench", "river", "pydantic", "zenml", "mlflow-skinny", "confluent-kafka",
                    "pandas", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(sources.rglob("*.py")):
        digest.update(str(path.relative_to(sources)).encode())
        digest.update(path.read_bytes())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "code_sha256": digest.hexdigest(),
        "git_revision": os.environ.get("BENCH_GIT_REVISION", "uncommitted"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
    }


def evaluate(config: Config, manifest: dict, root: str, transport=None, *, stop_after=None) -> dict:
    """Returns complete report or bounded progress; service I/O is injected."""
    registry = default_registry(config.plugins)
    source = make_source(registry, config.dataset)
    identity = config.identity()
    directory = Path(root) / identity
    checkpoints = LocalCheckpoints(root)
    env = environment()
    versions = ModelVersions(directory, manifest["sha256"], env["code_sha256"])
    # Platform is recorded, package/Python versions guard resume compatibility.
    contract = {
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
        if config.mode == "offline":
            # Offline has a temporal training prefix and a frozen suffix, no online detector/policy cycle.
            model = registry.factory("model", config.model.name, **config.model.params)()
            split = int(config.dataset.samples * config.offline_train_fraction)
            test_start = next(itertools.islice(read_events(events_path), split, split + 1)).timestamp
            metrics, current, windows = Metrics(), Metrics(), []
            observer = OnlineObservability(settings, memory_sampler=current_rss_bytes)
            observer.sample_memory()
            trained = 0
            for event in read_events(events_path):
                if time.monotonic() - started > config.max_seconds:
                    raise TimeoutError("offline deadline exceeded")
                if event.seq < split:
                    if event.available_at < test_start:
                        train_started = time.perf_counter()
                        model.learn(dict(event.x), event.y)
                        observer.warmup_train_seconds += time.perf_counter() - train_started
                        observer.warmup_train_samples += 1
                        trained += 1
                else:
                    prediction_started = time.perf_counter()
                    p = model.predict(dict(event.x))
                    elapsed = time.perf_counter() - prediction_started
                    trace = observer.before_prediction(event, model, trained, elapsed)
                    observer.after_label(event, p, trace, event.available_at, model, True)
                    metrics.add(event.y, p)
                    current.add(event.y, p)
                    if current.n == config.metric_window:
                        windows.append({"end_seq": event.seq, **current.values()})
                        current = Metrics()
                if event.seq % config.metric_window == 0:
                    observer.sample_memory()
            if current.n:
                windows.append({"end_seq": config.dataset.samples - 1, **current.values()})
            if trained == 0:
                raise ValueError("offline prefix has no labels available before test start")
            report = {
                "observed": config.dataset.samples,
                "labeled": config.dataset.samples,
                "metrics": metrics.values(),
                "windows": windows,
                "updates": [],
                "alarms": [],
                "retention_probes": [],
                "split": {
                    "train": [0, trained],
                    "purged_label_delay": [trained, split],
                    "test": [split, config.dataset.samples],
                    "intervals": "half-open",
                },
                "compute_seconds": time.monotonic() - started,
            }
            state = {"model": model, "split": split}
            versions.record(model, {"action": "initial_fit", "seq_range": [0, trained],
                                    "label_cutoff": test_start})
            observer.sample_memory()
        else:
            model_factory = registry.factory("model", config.model.name, **config.model.params)
            detector_factory = registry.factory("detector", config.detector.name, **config.detector.params)
            policy_args = config.policy.model_dump(exclude={"name"})
            engine = (
                snapshot["state"]
                if snapshot
                else Engine(
                    model_factory,
                    detector_factory,
                    registry.factory("policy", config.policy.name, **policy_args)(),
                    warmup=config.warmup,
                    window=config.replay_window,
                    metric_window=config.metric_window,
                    reset_detector=config.reset_detector_on_retrain,
                    training_budget=config.training_budget,
                    min_retrain_window=config.policy.min_window,
                    versions=versions,
                    observer=OnlineObservability(
                        settings, continual_evaluation(source, model_factory, settings), current_rss_bytes
                    ),
                )
            )
            if config.mode == "streaming" and transport:
                events = transport.events(manifest, read_events(events_path), config.max_seconds)
            else:
                events = read_events(events_path)
            size = config.batch_size if config.mode == "incremental" else 1
            events = iter(events)
            next_checkpoint = engine.next_seq + config.checkpoint_every
            probes = source.probes() if hasattr(source, "probes") else {}
            for batch in iter(lambda: list(itertools.islice(events, size)), []):
                if time.monotonic() - started > config.max_seconds:
                    raise TimeoutError("benchmark deadline exceeded; resume from last checkpoint")
                before = len(engine.windows)
                engine.observe_batch(batch)
                if len(engine.windows) > before:
                    engine.probe(probes)
                if engine.next_seq >= next_checkpoint:
                    engine.versions.flush()
                    checkpoints.save(identity, {"contract": contract, "state": engine, "complete": False})
                    next_checkpoint = engine.next_seq + config.checkpoint_every
                if stop_after is not None and engine.next_seq >= stop_after:
                    engine.versions.flush()
                    checkpoints.save(identity, {"contract": contract, "state": engine, "complete": False})
                    return {"complete": False, "observed": engine.next_seq, "run_key": identity}
            if engine.next_seq != config.dataset.samples:
                raise ValueError(f"source ended early at {engine.next_seq}")
            engine.finish()
            engine.observer.sample_memory()
            engine.probe(probes)
            report = engine.report()
            state = engine
            versions = engine.versions
            observer = engine.observer
        probe = next(read_events(events_path)).x
        report["model_versions"] = versions.finish(model if config.mode == "offline" else engine.model, probe)
        report.update(
            {
                "complete": True,
                "run_key": identity,
                "config": config.model_dump(mode="json"),
                "protocol": "temporal_holdout"
                if config.mode == "offline"
                else "batch_prequential"
                if config.mode == "incremental"
                else "event_prequential",
                "transport": contract["transport"]["name"],
                "environment": env,
                "dataset": manifest,
                "wall_seconds_this_attempt": time.monotonic() - started,
                "peak_rss_bytes_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
            }
        )
        save_tracking_files(directory, report, settings, observer)
        # Report first, then complete checkpoint: either crash point is safe to recompute/reuse.
        atomic_write(report_path, json.dumps(report, indent=2).encode())
        checkpoints.save(identity, {"contract": contract, "state": state, "complete": True})
        return report


def run(config: Config, root="state", *, services=False):
    if services:
        from .readiness import ensure_tracking_ready

        ensure_tracking_ready()
    manifest = prepare(config, root)
    transport = None
    if services and config.mode == "streaming":
        from continual_bench.adapters.services import KafkaTransport

        transport = KafkaTransport()
    report = evaluate(config, manifest, root, transport)
    if services:
        from continual_bench.adapters.services import publish

        publish(config, report, root)
    return report
