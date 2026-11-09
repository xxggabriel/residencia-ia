"""Submit bounded-memory Zia jobs through the existing independent worker slots."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from continual_bench.datasets.zia import load_manifest, sha256_file
from continual_bench.evaluation.environment import encoded, environment
from continual_bench.infrastructure.storage import atomic_write

DATA_ROOT = Path(os.environ.get("BENCH_ZIA_DATA", "/zia-data"))
JOB_ROOT = Path(os.environ.get("BENCH_STATE", "/state/runs")).parent / "zia-runs"
TIMEOUT = 2 * 24 * 3600


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Literal["fraud", "social", "ecommerce"]
    scope: Literal["smoke", "full"]
    policy: Literal["frozen", "periodic", "drift", "incremental"]
    campaign: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


def identity(job):
    _, manifest = load_manifest(DATA_ROOT, job.dataset, job.scope)
    return hashlib.sha256(encoded({
        "job": job.model_dump(), "input_sha256": manifest["sha256"],
        "tracking_sha256": sha256_file(Path("configs/base/tracking.yaml")),
        "code_sha256": environment()["code_sha256"],
    })).hexdigest()[:24]


def perform(job):
    key = identity(job)
    directory = JOB_ROOT / key
    started = time.time()
    status = {"run_key": key, "status": "running", "started_at": started, **job.model_dump()}
    atomic_write(directory / "job.json", encoded(status))
    try:
        with (directory / "worker.log").open("a") as log:
            result = subprocess.run([
                sys.executable, "-m", "continual_bench.evaluation.zia",
                str(DATA_ROOT), job.dataset, job.scope, job.policy, str(directory),
                "--campaign", job.campaign,
            ], stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"benchmark exited {result.returncode}; inspect {key}/worker.log")
        report = json.loads((directory / "report.json").read_text())
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        elapsed = time.time() - started
        from mlflow.tracking import MlflowClient

        from continual_bench.evaluation.zia_artifacts import tracking_report
        from continual_bench.evaluation.zia_predictions import prediction_manifest
        from continual_bench.infrastructure.metric_tracking import publish_execution

        receipt["tracking"] = publish_execution(
            MlflowClient(), receipt["mlflow_run_id"],
            tracking_report(report, prediction_manifest(directory, report)), directory, elapsed,
        )
        atomic_write(receipt_path, encoded(receipt))
        status.update(status="succeeded", receipt=receipt, summary={
            "complete": report["complete"], "protocol": report["protocol"],
            "dataset": job.dataset, "policy": job.policy, "model": report["model"],
            "samples": report["metrics"]["dataset_rows"], "metrics": report["metrics"],
        })
    except Exception as error:
        status.update(status="failed", error=str(error))
    finally:
        status["finished_at"] = time.time()
        atomic_write(directory / "job.json", encoded(status))
