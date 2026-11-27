"""Internal-only finite-job endpoint; Airflow alone decides when to submit work."""

import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from continual_bench.application.readiness import StackNotReady, ensure_tracking_ready
from continual_bench.config import Config
from continual_bench.infrastructure.storage import atomic_write, publication_lock, run_lock
from continual_bench.observability.events import emit_event, task_context

ROOT = Path(os.environ.get("BENCH_STATE", "/state/runs"))
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "1"))
EXECUTOR = ThreadPoolExecutor(max_workers=CONCURRENCY)
SLOTS = threading.BoundedSemaphore(CONCURRENCY)
JOBS = {}
ACTIVE = set()
PEAK_ACTIVE = 0
GUARD = threading.Lock()


def perform(config):
    identity = config.identity()
    directory = ROOT / identity
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "submitted.json"
    atomic_write(config_path, config.model_dump_json(indent=2).encode())
    status = {"run_key": identity, "status": "running", "started_at": time.time()}
    atomic_write(directory / "job.json", json.dumps(status).encode())
    try:
        environment = os.environ.copy()
        with (directory / "worker.log").open("a") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "continual_bench.cli",
                    "run",
                    str(config_path),
                    "--state",
                    str(ROOT),
                    "--services",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config.max_seconds + 180,
                env=environment,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"benchmark exited {result.returncode}; inspect worker.log")
        status["status"] = "succeeded"
        status["receipt"] = json.loads((directory / "receipt.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        if report.get("tracking"):
            from mlflow.tracking import MlflowClient

            from continual_bench.infrastructure.metric_tracking import publish_execution
            from continual_bench.infrastructure.services import S3Objects

            with run_lock(ROOT, identity):
                publish_execution(
                    MlflowClient(),
                    status["receipt"]["mlflow_run_id"],
                    report,
                    directory,
                    time.time() - status["started_at"],
                    objects=S3Objects(),
                )
        status["summary"] = {
            "complete": report["complete"],
            "protocol": report["protocol"],
            "dataset": config.dataset.name,
            "seed": config.dataset.seed,
            "model": config.model.name,
            "model_params": config.model.params,
            "samples": config.dataset.samples,
            "policy": config.policy.name,
            "detector": config.detector.name,
            "metrics": report["metrics"],
            "alarm_count": len(report["alarms"]),
            "update_count": len(report["updates"]),
            "transport": report["transport"],
            "training_budget": config.training_budget,
            "training_used": sum(u["train_samples"] for u in report["updates"]),
            "dataset_version": report["dataset"]["sha256"],
            "model_versions_count": report["model_versions"]["count"],
            "tracking_version": report.get("tracking", {}).get("version"),
        }
    except Exception as error:
        status.update(status="failed", error=str(error))
    finally:
        status["finished_at"] = time.time()
        atomic_write(directory / "job.json", json.dumps(status).encode())


def campaign_identity(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:24]


def _resolve_campaign_config(payload: dict):
    from continual_bench.protocols.base import CampaignConfig
    from continual_bench.protocols.campaign import load_campaign_config

    config_path = payload.get("config_path")
    if config_path:
        cfg = load_campaign_config(config_path)
        updates: dict = {}
        if payload.get("campaign"):
            updates["name"] = str(payload["campaign"])
        if payload.get("policies"):
            updates["policies"] = list(payload["policies"])
        if payload.get("seeds"):
            updates["seeds"] = [int(s) for s in payload["seeds"]]
        if updates:
            cfg = cfg.model_copy(update=updates)
        return cfg

    protocol = str(payload.get("protocol", "cara")).strip().lower()
    dataset_name = str(payload.get("dataset", "electricity")).strip()
    campaign_name = str(payload.get("campaign") or f"{protocol}-{dataset_name}")
    model_name = str(payload.get("model") or "logistic")
    policies = list(payload.get("policies") or [])
    seeds = [int(s) for s in (payload.get("seeds") or [42, 43, 44, 45, 46])]
    return CampaignConfig(
        name=campaign_name,
        protocol=protocol,  # type: ignore[arg-type]
        dataset={"name": dataset_name, "protocol": protocol},  # type: ignore[arg-type]
        model={"name": model_name},  # type: ignore[arg-type]
        seeds=seeds,
        policies=policies,
    )


def perform_campaign(payload: dict) -> None:
    from continual_bench.protocols.campaign import run_campaign

    identity = campaign_identity(payload)
    campaign_root = ROOT / "campaigns"
    directory = campaign_root / identity
    directory.mkdir(parents=True, exist_ok=True)
    status: dict = {"campaign_key": identity, "status": "running", "started_at": time.time()}
    atomic_write(directory / "job.json", json.dumps(status).encode())
    try:
        env_name = payload.get("env") or os.environ.get("BENCH_ENV", "docker")
        save_checkpoints = payload.get("save_checkpoints")
        cfg = _resolve_campaign_config(payload)
        result = run_campaign(
            cfg,
            env_name=env_name,
            save_model_checkpoints=save_checkpoints,
        )
        status.update(
            status="succeeded",
            protocol=result.protocol,
            dataset=result.dataset,
            aggregate_summary=result.aggregate_summary,
            artifacts=result.artifacts,
            mlflow_receipt=result.mlflow_receipt,
        )
    except Exception as error:
        status.update(status="failed", error=str(error))
    finally:
        status["finished_at"] = time.time()
        atomic_write(directory / "job.json", json.dumps(status, default=str).encode())


def _open_campaign_tracker(config, payload, *, recover_evaluations=False):
    from continual_bench.protocols.campaign import create_campaign_tracker, resolve_campaign_settings

    settings = resolve_campaign_settings(
        config,
        env_name=payload.get("env") or os.environ.get("BENCH_ENV", "docker"),
        save_model_checkpoints=payload.get("save_checkpoints"),
    )
    batch_key = campaign_identity({"campaign_key": payload["campaign_key"]})
    run_id = f"campaign-{batch_key}"
    backend, tracker, run_id = create_campaign_tracker(config, settings, run_id=run_id)
    with publication_lock(ROOT, run_id):
        tracker.start_campaign(resume=True, recover_evaluations=recover_evaluations)
    return backend, tracker, run_id, settings


def perform_campaign_seed(payload: dict, *, progress=None, execution=None) -> None:
    import pickle

    from continual_bench.protocols.base import CampaignConfig
    from continual_bench.protocols.cara import CaraProtocol
    from continual_bench.protocols.upf import UpfProtocol

    identity = campaign_identity(payload)
    seed_root = ROOT / "campaign_seeds"
    directory = seed_root / identity
    directory.mkdir(parents=True, exist_ok=True)
    tracker = None
    status: dict = {"seed_key": identity, "status": "running", "started_at": time.time()}
    if execution:
        status["worker_execution"] = execution
    atomic_write(directory / "job.json", json.dumps(status).encode())
    try:
        cfg = CampaignConfig(**payload["config"])
        seed = int(payload["seed"])
        batch_key = str(payload["campaign_key"])
        _, tracker, _, _ = _open_campaign_tracker(cfg, payload)
        protocol = CaraProtocol() if cfg.protocol == "cara" else UpfProtocol()
        if progress:
            progress("seed.matrix_started", "Building the seed matrix")
        seed_data = protocol.run_seed_matrix(cfg, seed, tracker)
        if progress:
            progress("seed.matrix_completed", "Seed matrix computed; publishing result")
        cache_dir = ROOT / "campaign_cache" / batch_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        seed_path = cache_dir / f"seed_{seed}.pkl"
        atomic_write(seed_path, pickle.dumps(seed_data))
        status.update(status="succeeded", seed=seed, seed_path=str(seed_path))
    except Exception as error:
        if progress:
            progress(
                "seed.execution_failed",
                f"Seed execution failed ({type(error).__name__})",
                level=logging.ERROR,
            )
        if tracker is not None:
            tracker.fail_campaign()
        status.update(status="failed", error=str(error))
    finally:
        status["finished_at"] = time.time()
        atomic_write(directory / "job.json", json.dumps(status, default=str).encode())


def perform_campaign_consolidate(payload: dict) -> None:
    import pickle
    import shutil

    from continual_bench.protocols.base import CampaignConfig
    from continual_bench.protocols.campaign import finalize_campaign_result
    from continual_bench.protocols.cara import CaraProtocol
    from continual_bench.protocols.upf import UpfProtocol

    identity = campaign_identity(payload)
    cons_root = ROOT / "campaign_consolidate"
    directory = cons_root / identity
    directory.mkdir(parents=True, exist_ok=True)
    tracker = None
    status: dict = {"consolidate_key": identity, "status": "running", "started_at": time.time()}
    atomic_write(directory / "job.json", json.dumps(status).encode())
    try:
        cfg = CampaignConfig(**payload["config"])
        backend, tracker, run_id, settings = _open_campaign_tracker(cfg, payload, recover_evaluations=True)
        seed_paths = [Path(p) for p in payload["seed_paths"]]
        seed_matrices = [pickle.loads(p.read_bytes()) for p in seed_paths]
        if cfg.protocol == "cara":
            proto = CaraProtocol()
            ctx = proto._prepare_context(cfg)
            result = proto.score_and_finalize(
                cfg, seed_matrices, backend, tracker, gamma=ctx[2], offline=ctx[1]
            )
        else:
            result = UpfProtocol().score_and_finalize(cfg, seed_matrices, backend, tracker)
        result = finalize_campaign_result(cfg, result, backend, tracker, settings, run_id)

        batch_key = str(payload["campaign_key"])
        shutil.rmtree(ROOT / "campaign_cache" / batch_key, ignore_errors=True)
        status.update(
            status="succeeded",
            protocol=result.protocol,
            dataset=result.dataset,
            aggregate_summary=result.aggregate_summary,
            artifacts=result.artifacts,
            mlflow_receipt=result.mlflow_receipt,
        )
    except Exception as error:
        if tracker is not None:
            tracker.fail_campaign()
        status.update(status="failed", error=str(error))
    finally:
        status["finished_at"] = time.time()
        atomic_write(directory / "job.json", json.dumps(status, default=str).encode())


def health_snapshot():
    with GUARD:
        return {
            "status": "ok",
            "queue": len(JOBS),
            "active": len(ACTIVE),
            "peak_active": PEAK_ACTIVE,
            "concurrency": CONCURRENCY,
        }


def _seed_fields(config, identity):
    campaign_key = config["campaign_key"]
    return {
        "campaign_key": campaign_key,
        "seed_key": identity,
        "run_id": f"campaign-{campaign_identity({'campaign_key': campaign_key})}",
        "seed": config["seed"],
    }


def _associate(status, observer, config):
    if not observer:
        return status
    origin = status.get("worker_execution") or {}
    execution_id = origin.get("job_attempt_id")
    owns_execution = origin.get("task_attempt_key") == observer["task_attempt_key"]
    association = {
        "job_key": status["seed_key"],
        "job_attempt_id": execution_id,
        "association_role": "origin" if owns_execution else "observer",
        "origin_task_attempt_key": origin.get("task_attempt_key"),
    }
    event_id = hashlib.sha256(
        f"{observer['task_attempt_key']}:{status['seed_key']}:{execution_id}".encode()
    ).hexdigest()[:32]
    emit_event(
        "seed.task_associated",
        f"Task associated with seed job {status['seed_key']} ({association['association_role']})",
        event_id=event_id,
        worker_host=socket.gethostname(),
        **observer,
        **association,
        **_seed_fields(config, status["seed_key"]),
    )
    return {**status, "task_association": association}


def submit_job(root, identity, config, perform_job, *, key_field="run_key", observer=None):
    """One slot per backend; shared-volume claims prevent duplicate jobs across replicas."""
    root = Path(root)
    path = root / identity / "job.json"
    claim = run_lock(root / ".job-claims", identity)
    try:
        claim.__enter__()
    except RuntimeError:
        status = {key_field: identity, "status": "queued_or_running"}
        if key_field == "seed_key" and path.exists():
            status["worker_execution"] = json.loads(path.read_text()).get("worker_execution")
        return 202, _associate(status, observer, config)
    try:
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["status"] == "succeeded":
                return 200, _associate(previous, observer, config)
        if not SLOTS.acquire(blocking=False):
            return 429, {"error": "worker_busy"}
        token = str(root), identity
        status = {key_field: identity, "status": "queued"}
        execution = None
        try:
            if key_field == "seed_key":
                execution = {
                    **(observer or {}),
                    "worker_host": socket.gethostname(),
                    **_seed_fields(config, identity),
                    "job_key": identity,
                    "job_attempt_id": uuid4().hex,
                }
                status["worker_execution"] = execution
            atomic_write(path, json.dumps(status).encode())
            with GUARD:
                JOBS[token] = True
            EXECUTOR.submit(_run_claimed, token, config, perform_job, claim, execution)
        except Exception:
            with GUARD:
                JOBS.pop(token, None)
            SLOTS.release()
            raise
        claim = None  # The executor now owns the lease until publication finishes.
        return 202, _associate(status, observer, config)
    finally:
        if claim is not None:
            claim.__exit__(None, None, None)


def _run_claimed(token, config, perform_job, claim, execution=None):
    global PEAK_ACTIVE
    with GUARD:
        ACTIVE.add(token)
        PEAK_ACTIVE = max(PEAK_ACTIVE, len(ACTIVE))
    path = Path(token[0]) / token[1] / "job.json"
    sequence = 0

    def progress(name, message, *, level=logging.INFO):
        nonlocal sequence
        sequence += 1
        emit_event(name, message, level=level, event_sequence=sequence, **execution)

    terminal = "failed"
    try:
        if execution:
            progress("seed.started", f"Running seed {execution['seed']}")
            perform_job(config, progress=progress, execution=execution)
            terminal = json.loads(path.read_text())["status"]
        else:
            perform_job(config)
    except Exception as error:
        status = json.loads(path.read_text())
        status.update(status="failed", error=type(error).__name__, finished_at=time.time())
        atomic_write(path, json.dumps(status).encode())
    finally:
        if execution:
            progress(
                "seed.finished", f"Seed {execution['seed']} finished: {terminal}",
                level=logging.INFO if terminal == "succeeded" else logging.ERROR,
            )
        with GUARD:
            ACTIVE.discard(token)
            JOBS.pop(token, None)
        claim.__exit__(None, None, None)
        SLOTS.release()


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, value):
        payload = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/ready":
            try:
                self.respond(200, ensure_tracking_ready())
            except StackNotReady as error:
                self.respond(503, {"ready": False, "error": str(error)})
            return
        if self.path == "/health":
            self.respond(200, health_snapshot())
            return
        prefix, _, identity = self.path.rpartition("/")
        if prefix in ("/runs", "/campaigns", "/campaign-seeds", "/campaign-consolidate"):
            if len(identity) == 24 and all(c in "0123456789abcdef" for c in identity):
                root = ROOT
                if prefix == "/campaigns":
                    root = ROOT / "campaigns"
                elif prefix == "/campaign-seeds":
                    root = ROOT / "campaign_seeds"
                elif prefix == "/campaign-consolidate":
                    root = ROOT / "campaign_consolidate"
                path = root / identity / "job.json"
                if path.exists():
                    self.respond(200, json.loads(path.read_text()))
                    return
        self.respond(404, {"error": "not_found"})

    def do_POST(self):
        if self.path not in (
            "/runs",
            "/campaigns",
            "/campaign-plan",
            "/campaign-seeds",
            "/campaign-consolidate",
        ):
            self.respond(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("body size must be 1..65536")
            payload = self.rfile.read(length)
            if self.path == "/campaign-plan":
                plan_payload = json.loads(payload)
                cfg = _resolve_campaign_config(plan_payload)
                cfg_dict = cfg.model_dump(mode="json", exclude_unset=True)
                c_key = campaign_identity({
                    "config": cfg_dict,
                    "dag_run_id": plan_payload.get("dag_run_id"),
                    "env": plan_payload.get("env", "docker"),
                    "save_checkpoints": bool(plan_payload.get("save_checkpoints", False)),
                })
                seed_jobs = [
                    {
                        "campaign_key": c_key,
                        "seed": int(s),
                        "config": cfg_dict,
                        "env": plan_payload.get("env", "docker"),
                        "save_checkpoints": bool(plan_payload.get("save_checkpoints", False)),
                    }
                    for s in cfg.seeds
                ]
                self.respond(200, {"campaign_key": c_key, "config": cfg_dict, "seed_jobs": seed_jobs})
                return
            if self.path == "/campaign-seeds":
                seed_payload = json.loads(payload)
                observer = task_context(seed_payload.pop("airflow_task", None))
                seed_key = campaign_identity(seed_payload)
                seed_root = ROOT / "campaign_seeds"
                # Serialize admission and origin publication across execution containers.
                with publication_lock(seed_root, seed_key):
                    code, status = submit_job(
                        seed_root, seed_key, seed_payload, perform_campaign_seed,
                        key_field="seed_key", observer=observer,
                    )
                self.respond(code, status)
                return
            if self.path == "/campaign-consolidate":
                cons_payload = json.loads(payload)
                code, status = submit_job(
                    ROOT / "campaign_consolidate",
                    campaign_identity(cons_payload),
                    cons_payload,
                    perform_campaign_consolidate,
                    key_field="consolidate_key",
                )
                self.respond(code, status)
                return
            if self.path == "/campaigns":
                campaign_payload = json.loads(payload)
                if not isinstance(campaign_payload, dict):
                    raise ValueError("campaign payload must be a JSON object")
                code, status = submit_job(
                    ROOT / "campaigns",
                    campaign_identity(campaign_payload),
                    campaign_payload,
                    perform_campaign,
                    key_field="campaign_key",
                )
                self.respond(code, status)
                return
            config = Config.model_validate_json(payload)
            # Plugins run only from the trusted CLI/config deployment path.
            if config.plugins:
                raise ValueError("plugins must be installed/configured by operator, not HTTP requests")
            root, identity, handler = ROOT, config.identity(), perform
        except Exception as error:
            self.respond(400, {"error": str(error)})
            return
        try:
            ensure_tracking_ready()
        except StackNotReady as error:
            self.respond(503, {"ready": False, "error": str(error)})
            return
        code, status = submit_job(root, identity, config, handler)
        self.respond(code, status)


def serve():
    ROOT.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", 8088), Handler).serve_forever()
