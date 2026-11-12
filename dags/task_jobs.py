"""Full-project matrix including the admitted dataset catalog and existing model/policy suites."""

import json
import time
import urllib.request

from continual_bench.application.project_suite import project_configurations as configurations  # noqa: F401

WORKER = "http://dataset-worker:8088"


def request(path, payload=None):
    call = urllib.request.Request(WORKER + path,
                                  data=None if payload is None else json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=30) as response:
        return json.load(response)


def readiness():
    basic, datasets = request("/ready"), request("/datasets-ready")
    if not basic.get("ready") or not datasets.get("ready"):
        raise RuntimeError("dataset worker is not admitted")
    return datasets


def execute(case):
    epochs = case["kind"] == "epochs"
    endpoint = "/epoch-runs" if epochs else "/runs"
    submitted = request(endpoint, case["config"])
    key = submitted["request_key" if epochs else "run_key"]
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline:
        status = request(endpoint + "/" + key)
        if status["status"] == "succeeded":
            if not status.get("receipt", {}).get("mlflow_run_id"):
                raise RuntimeError("benchmark finished without publication evidence")
            return {"kind": case["kind"], "key": key, "receipt": status["receipt"]}
        if status["status"] == "failed":
            raise RuntimeError(f"benchmark {key} failed; inspect its isolated worker log")
        time.sleep(3)
    raise TimeoutError(f"benchmark {key} did not finish before the DAG task deadline")
