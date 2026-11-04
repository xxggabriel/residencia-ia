"""Full-project matrix including the admitted dataset catalog and existing model/policy suites."""

import json
import time
import urllib.request
from copy import deepcopy

WORKER = "http://dataset-worker:8088"
DATASETS = (
    "sea", "agrawal", "stagger", "hyperplane", "rbf_drift", "friedman_drift",
    "electricity", "airlines", "insects", "gas", "split_mnist", "permuted_mnist",
    "split_cifar10", "split_cifar100", "core50", "stream51", "clear10", "clear100",
)


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


def configurations(params, code):
    from bench_jobs import nonlinear_configurations

    if len(params["seeds"]) != 1:
        raise ValueError("this full-project protocol requires exactly one paired seed")
    # Retain all prior paired models, modes and budgets; use the new isolated worker/code.
    base = nonlinear_configurations({"samples": params["samples"], "seeds": params["seeds"],
                                     "label_delay": params["label_delay"]})
    campaign = "semana09-full-" + code[:12]
    legacy = [{"kind": "binary", "config": {**c, "name": campaign,
               "tracking": {"probe_samples": 16, "continual_references": False,
                            "week09": {"horizon": 16, "gap": 1, "max_windows": 4}}}} for c in base]
    models = {json.dumps(c["model"], sort_keys=True) for c in base
              if "nominal_attributes" not in c["model"]["params"]}
    extra = []
    for name in ("sea", "agrawal", "stagger", "hyperplane", "rbf_drift"):
        for model in sorted(models):
            selected_model = json.loads(model)
            if selected_model["name"] == "hoeffding_tree" and name in {"agrawal", "stagger"}:
                selected_model["params"]["nominal_attributes"] = (
                    ["elevel", "car", "zipcode"] if name == "agrawal" else ["size", "color", "shape"]
                )
            for mode in ("streaming", "incremental"):
                variants = [("none", "performance"), ("none", "cost_heuristic")]
                variants += [(d, "drift") for d in ("eddm", "hddm_a", "hddm_w", "page_hinkley", "kswin")]
                if name in {"hyperplane", "rbf_drift"}:
                    variants += [("none", "none"), ("none", "periodic"), ("none", "online"),
                                 ("adwin", "drift"), ("ddm", "drift")]
                for detector, policy in variants:
                    config = {"name": campaign, "mode": mode, "model": deepcopy(selected_model),
                              "dataset": {"name": name, "samples": params["samples"],
                                          "seed": params["seeds"][0], "label_delay": params["label_delay"]},
                              "policy": {"name": policy, "action": "learn_one" if policy == "online"
                                         else "reset_retrain"}, "detector": {"name": detector},
                              "tracking": deepcopy(legacy[0]["config"]["tracking"])}
                    extra.append({"kind": "binary", "config": config})
            if name in {"hyperplane", "rbf_drift"}:
                extra.append({"kind": "binary", "config": {
                    "name": campaign, "mode": "offline", "model": deepcopy(selected_model),
                    "dataset": {"name": name, "samples": params["samples"],
                                "seed": params["seeds"][0], "label_delay": params["label_delay"]},
                    "policy": {"name": "none"}, "detector": {"name": "none"},
                    "tracking": deepcopy(legacy[0]["config"]["tracking"]),
                }})
    # Native multiclass/regression target types follow the explicit epoch/holdout contract.
    datasets = [{"kind": "epochs", "config": {"dataset": name, "examples": params["dataset_examples"],
                 "epochs": params["epochs"], "seed": params["seeds"][0], "max_seconds": 900}}
                for name in DATASETS]
    result = legacy + extra + datasets
    # Model nominal-feature variants can repeat across sources; only exact requests are deduplicated.
    unique = {json.dumps(item, sort_keys=True): item for item in result}
    if len(unique) > 1024:
        raise ValueError("Semana 09 full matrix exceeds Airflow mapping budget")
    return list(unique.values())


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
