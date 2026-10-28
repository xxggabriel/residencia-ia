"""Internal-only finite-job endpoint; Airflow alone decides when to submit work."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from continual_bench.adapters.local import atomic_write, run_lock

from .config import Config
from .readiness import StackNotReady, ensure_tracking_ready

ROOT = Path(os.environ.get("BENCH_STATE", "/state/runs"))
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "1"))
EXECUTOR = ThreadPoolExecutor(max_workers=CONCURRENCY)
JOBS = {}
ACTIVE = set()
PEAK_ACTIVE = 0
GUARD = threading.Lock()


def perform(config):
    global PEAK_ACTIVE
    identity = config.identity()
    directory = ROOT / identity
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "submitted.json"
    atomic_write(config_path, config.model_dump_json(indent=2).encode())
    with GUARD:
        ACTIVE.add(identity)
        PEAK_ACTIVE = max(PEAK_ACTIVE, len(ACTIVE))
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

            from continual_bench.adapters.metric_tracking import publish_execution
            from continual_bench.adapters.services import S3Objects

            with run_lock(ROOT, identity):
                publish_execution(MlflowClient(), status["receipt"]["mlflow_run_id"], report, directory,
                                  time.time() - status["started_at"], objects=S3Objects())
        status["summary"] = {
            "complete": report["complete"], "protocol": report["protocol"],
            "dataset": config.dataset.name, "seed": config.dataset.seed,
            "model": config.model.name, "model_params": config.model.params,
            "samples": config.dataset.samples, "policy": config.policy.name,
            "detector": config.detector.name, "metrics": report["metrics"],
            "alarm_count": len(report["alarms"]), "update_count": len(report["updates"]),
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
        with GUARD:
            ACTIVE.discard(identity)
            JOBS.pop(identity, None)


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
            with GUARD:
                snapshot = {"status": "ok", "queue": len(JOBS), "active": len(ACTIVE),
                            "peak_active": PEAK_ACTIVE, "concurrency": CONCURRENCY}
            self.respond(200, snapshot)
            return
        if self.path.startswith("/runs/"):
            identity = self.path.removeprefix("/runs/")
            if len(identity) == 24 and all(c in "0123456789abcdef" for c in identity):
                path = ROOT / identity / "job.json"
                if path.exists():
                    self.respond(200, json.loads(path.read_text()))
                    return
        self.respond(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/runs":
            self.respond(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("body size must be 1..65536")
            config = Config.model_validate_json(self.rfile.read(length))
            # Internal API accepts built-ins only. Plugins run from the trusted CLI/config deployment path.
            if config.plugins:
                raise ValueError("plugins must be installed/configured by operator, not HTTP requests")
        except Exception as error:
            self.respond(400, {"error": str(error)})
            return
        identity = config.identity()
        try:
            ensure_tracking_ready()
        except StackNotReady as error:
            self.respond(503, {"ready": False, "error": str(error)})
            return
        with GUARD:
            if identity in JOBS:
                self.respond(202, {"run_key": identity, "status": "queued_or_running"})
                return
            previous = ROOT / identity / "job.json"
            if previous.exists():
                status = json.loads(previous.read_text())
                if status["status"] == "succeeded":
                    self.respond(200, status)
                    return
            if len(JOBS) >= 16:
                self.respond(429, {"error": "queue_full"})
                return
            JOBS[identity] = True
            (ROOT / identity).mkdir(parents=True, exist_ok=True)
            atomic_write(previous, json.dumps({"run_key": identity, "status": "queued"}).encode())
            EXECUTOR.submit(perform, config)
        self.respond(202, {"run_key": identity, "status": "queued"})


def serve():
    ROOT.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", 8088), Handler).serve_forever()
