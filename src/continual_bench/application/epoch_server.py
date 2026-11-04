"""Isolated internal worker; legacy binary jobs and new dataset jobs share one queue."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from continual_bench.adapters.dataset_loading import DATASETS
from continual_bench.adapters.local import atomic_write

from . import server
from .epoch_benchmark import EpochConfig, dataset_fingerprint
from .runner import environment

EPOCH_ROOT = Path(os.environ.get("BENCH_EPOCH_ROOT", "/state/epoch-jobs"))
SMOKE_ROOT = Path(os.environ.get("BENCH_DATASET_SMOKES", "/state/dataset-smokes"))
EPOCH_ACTIVE = set()
EPOCH_GUARD = threading.Lock()


def request_key(config):
    value = {"config": config.model_dump(), "runtime_fingerprint": dataset_fingerprint()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def dataset_readiness():
    path = SMOKE_ROOT / "dataset-smoke-complete.json"
    if not path.exists():
        raise ValueError("all-dataset 100-example two-epoch smoke receipt is absent")
    receipt = json.loads(path.read_text())
    rows = receipt.get("datasets", [])
    if (not receipt.get("complete") or len(rows) != len(DATASETS)
            or {r["dataset"] for r in rows} != set(DATASETS)
            or any(not r.get("complete") or r.get("unique_examples") != 100 or r.get("epochs") != 2
                   or r.get("code_sha256") != environment()["code_sha256"]
                   or r.get("runtime_fingerprint") != dataset_fingerprint() for r in rows)):
        raise ValueError("dataset smoke receipt is incomplete, stale or mismatched")
    return {"ready": True, "datasets": len(rows), "code_sha256": environment()["code_sha256"]}


def perform_epoch(config):
    key = request_key(config)
    directory = EPOCH_ROOT / key
    directory.mkdir(parents=True, exist_ok=True)
    submitted = directory / "submitted.json"
    atomic_write(submitted, config.model_dump_json().encode())
    status = {"request_key": key, "status": "running", "started_at": time.time()}
    atomic_write(directory / "job.json", json.dumps(status).encode())
    try:
        with (directory / "worker.log").open("a") as output:
            command = [sys.executable, "-m", "continual_bench.cli", "epoch-job", str(submitted),
                       "--state", str(directory / "runs"), "--cache", os.environ["BENCH_DATASET_CACHE"],
                       "--real-directory", os.environ["BENCH_DATASETS"], "--services"]
            if os.environ.get("BENCH_DATASET_DOWNLOAD") == "true":
                command.append("--download")
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True,
                           timeout=config.max_seconds + 180)
        paths = list((directory / "runs").glob("*/epoch-receipt.json"))
        if len(paths) != 1:
            raise ValueError("epoch publication did not produce exactly one receipt")
        receipt = json.loads(paths[0].read_text())
        if not receipt.get("complete") or not receipt.get("mlflow_run_id"):
            raise ValueError("incomplete epoch publication")
        status.update(status="succeeded", receipt=receipt)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__)
    finally:
        status["finished_at"] = time.time()
        atomic_write(directory / "job.json", json.dumps(status).encode())
        with EPOCH_GUARD:
            EPOCH_ACTIVE.discard(key)


class Handler(server.Handler):
    def do_GET(self):
        if self.path == "/datasets-ready":
            try:
                self.respond(200, dataset_readiness())
            except (ValueError, OSError) as error:
                self.respond(503, {"ready": False, "error": str(error)})
            return
        if self.path.startswith("/epoch-runs/"):
            key = self.path.removeprefix("/epoch-runs/")
            if len(key) == 24 and all(c in "0123456789abcdef" for c in key):
                path = EPOCH_ROOT / key / "job.json"
                if path.exists():
                    self.respond(200, json.loads(path.read_text()))
                    return
            self.respond(404, {"error": "not_found"})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/epoch-runs":
            super().do_POST()
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("invalid body size")
            config = EpochConfig.model_validate_json(self.rfile.read(size))
            if config.dataset not in DATASETS:
                raise ValueError("unknown dataset")
            dataset_readiness()
        except Exception as error:
            self.respond(400, {"error": str(error)})
            return
        key = request_key(config)
        path = EPOCH_ROOT / key / "job.json"
        with EPOCH_GUARD:
            if key in EPOCH_ACTIVE:
                self.respond(202, {"request_key": key, "status": "queued_or_running"})
                return
            if path.exists() and json.loads(path.read_text()).get("status") == "succeeded":
                self.respond(200, json.loads(path.read_text()))
                return
            if len(EPOCH_ACTIVE) >= 16:
                self.respond(429, {"error": "queue_full"})
                return
            EPOCH_ACTIVE.add(key)
            atomic_write(path, json.dumps({"request_key": key, "status": "queued"}).encode())
            server.EXECUTOR.submit(perform_epoch, config)
        self.respond(202, {"request_key": key, "status": "queued"})


def serve():
    from continual_bench.adapters.zenml_pipeline import configure_local_stack

    configure_local_stack(str(server.ROOT))
    ThreadingHTTPServer(("0.0.0.0", 8088), Handler).serve_forever()
