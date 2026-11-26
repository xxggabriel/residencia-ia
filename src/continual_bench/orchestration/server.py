"""Internal-only finite-job endpoint; Airflow alone decides when to submit work."""

import hashlib
import json
import logging
import os
import shutil
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
from continual_bench.infrastructure.storage import atomic_write, run_lock
from continual_bench.observability.events import emit_event

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
        # SDK active-stack/project files are mutable: each subprocess owns a copy.
        environment = os.environ.copy()
        if environment.get("ZENML_CONFIG_PATH"):
            isolated = directory / "zen-config"
            shutil.copytree(environment["ZENML_CONFIG_PATH"], isolated, dirs_exist_ok=True)
            environment["ZENML_CONFIG_PATH"] = str(isolated)
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
                    "--zenml",
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


def health_snapshot():
    with GUARD:
        return {
            "status": "ok",
            "queue": len(JOBS),
            "active": len(ACTIVE),
            "peak_active": PEAK_ACTIVE,
            "concurrency": CONCURRENCY,
        }


def campaign_identity(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:24]


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
        if prefix in ("/runs", "/zia-runs"):
            if len(identity) == 24 and all(c in "0123456789abcdef" for c in identity):
                root = ROOT
                if prefix == "/zia-runs":
                    from .zia_jobs import JOB_ROOT

                    root = JOB_ROOT
                path = root / identity / "job.json"
                if path.exists():
                    self.respond(200, json.loads(path.read_text()))
                    return
        self.respond(404, {"error": "not_found"})

    def do_POST(self):
        if self.path not in ("/runs", "/zia-runs"):
            self.respond(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("body size must be 1..65536")
            payload = self.rfile.read(length)
            if self.path == "/zia-runs":
                from . import zia_jobs

                config = zia_jobs.Job.model_validate_json(payload)
                root, identity, handler = zia_jobs.JOB_ROOT, zia_jobs.identity(config), zia_jobs.perform
            else:
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
