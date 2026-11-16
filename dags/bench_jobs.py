"""Finite worker jobs and reproducible matrices; no per-event scheduler tasks."""

import json
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

from continual_bench.application.project_suite import (
    controlled_configurations,
    nonlinear_configurations,
    versioned_configurations,
)


def tracking_profile():
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs/base/tracking.yaml"
    return yaml.safe_load(path.read_text())


def configurations(params, suite):
    if suite == "policies":
        return policy_configurations(params)
    if suite in ("metrics", "metrics_smoke"):
        cases = nonlinear_configurations(params)
        if suite == "metrics_smoke":
            selected = {
                ("agrawal", "hoeffding_tree", "streaming", "online", "none"),
                ("stagger", "knn", "incremental", "online", "none"),
                ("sea", "logistic", "streaming", "drift", "ddm"),
                ("airlines", "knn", "streaming", "periodic", "none"),
                ("electricity", "hoeffding_tree", "incremental", "online", "none"),
                ("airlines", "logistic", "offline", "none", "none"),
                ("agrawal", "hoeffding_tree", "offline", "none", "none"),
                ("stagger", "knn", "offline", "none", "none"),
            }
            cases = [c for c in cases if (c["dataset"]["name"], c["model"]["name"], c["mode"],
                                          c["policy"]["name"], c["detector"]["name"]) in selected
                     and c["dataset"]["seed"] == params["seeds"][0]
                     and c.get("training_budget") in (0, None, params["samples"] - 100)]
            # Periodic/drift have two caps: keep only the lower cap in this bounded check.
            assert len(cases) == 8
        profile = params.get("tracking", tracking_profile())
        if suite == "metrics_smoke":
            profile = {**profile, "detection_window": max(1, params["samples"] // 6)}
        campaign = params.get("campaign", "metrics-v6" if suite == "metrics" else "metrics-v6-smoke")
        return [{**c, "name": campaign, "tracking": deepcopy(profile), "max_seconds": 1800} for c in cases]
    if suite == "nonlinear":
        return nonlinear_configurations(params)
    if suite == "versioned":
        return versioned_configurations(params)
    if suite == "controlled":
        return controlled_configurations(params)
    modes = ("streaming", "incremental", "offline") if suite == "full" else (
        ("streaming", "incremental") if suite == "strategies" else (suite,)
    )
    result = []
    for seed in params["seeds"]:
        for dataset in ("sea", "agrawal", "stagger"):
            for mode in modes:
                variants = [("none", "none")]
                if mode != "offline":
                    variants += [("none", "periodic"), ("none", "online"),
                                 ("adwin", "drift"), ("ddm", "drift")]
                for detector, policy in variants:
                    result.append({
                        "name": "benchmark-v2", "mode": mode,
                        "dataset": {"name": dataset, "samples": params["samples"],
                                    "seed": seed, "label_delay": params["label_delay"]},
                        "detector": {"name": detector},
                        "policy": {"name": policy,
                                   "action": "learn_one" if policy == "online" else "reset_retrain"},
                    })
    return result


POLICY_MODELS = {
    "logistic": {"learning_rate": 0.05},
    "hoeffding_tree": {"grace_period": 50, "max_depth": 8, "max_size": 8, "memory_estimate_period": 100,
                       "leaf_prediction": "nba"},
    "knn": {"n_neighbors": 11, "window_size": 256, "weighted": True},
}
# Binary streams accepted by the drift/retraining engine (multiclass/regression use other evaluators).
POLICY_DATASETS = ("electricity", "airlines", "sea", "agrawal", "stagger", "hyperplane", "rbf_drift")
# River generators encode these categorical features as integers; the tree must treat them as nominal.
NOMINAL_FEATURES = {"agrawal": ["elevel", "car", "zipcode"], "stagger": ["size", "color", "shape"]}
DRIFT_DETECTORS = ("adwin", "ddm", "eddm", "hddm_a", "hddm_w", "page_hinkley", "kswin")


def policy_strategies():
    """Every retraining policy; drift-triggered retraining once per River detector."""
    guards = {"every": 500, "cooldown": 100, "min_samples": 100, "min_window": 50, "action": "reset_retrain"}
    strategies = [("none", {"name": "none"}),
                  ("none", {"name": "online", "action": "learn_one"}),
                  ("none", {"name": "periodic", **guards})]
    strategies += [(detector, {"name": "drift", **guards}) for detector in DRIFT_DETECTORS]
    strategies.append(("none", {"name": "performance", **guards,
                                "params": {"monitor_window": 200, "error_threshold": 0.3}}))
    return strategies


def policy_configurations(params):
    """One real dataset x models x all retraining policies; offline frozen holdout as reference.

    Real prefixes are deterministic, so a single seed is used. Every case omits `tracking`,
    so the worker collects the full default metric profile, system metrics and plots.
    """
    dataset = params["dataset"]
    if dataset not in POLICY_DATASETS:
        raise ValueError(f"policies suite needs a binary dataset; got {dataset}")
    campaign = params.get("campaign") or f"policies-{dataset}-v1"
    experiment = params.get("experiment") or f"continual-bench/{campaign}"
    models = params.get("models") or list(POLICY_MODELS)
    data = {"name": dataset, "samples": params["samples"], "seed": params["seeds"][0],
            "label_delay": params["label_delay"]}
    common = {"name": campaign, "mlflow_experiment": experiment, "warmup": 100, "metric_window": 100,
              "replay_window": 256, "max_seconds": 1800}
    result = []
    for model in models:
        component = {"name": model, "params": deepcopy(POLICY_MODELS[model])}
        if model == "hoeffding_tree" and dataset in NOMINAL_FEATURES:
            component["params"]["nominal_attributes"] = NOMINAL_FEATURES[dataset]
        for mode in ("streaming", "incremental"):
            for detector, policy in policy_strategies():
                result.append({**common, "mode": mode, "dataset": dict(data), "model": deepcopy(component),
                               "detector": {"name": detector}, "policy": deepcopy(policy)})
        result.append({**common, "mode": "offline", "dataset": dict(data), "model": deepcopy(component),
                       "detector": {"name": "none"}, "policy": {"name": "none"}})
    return result


def request_json(path, payload=None):
    request = urllib.request.Request(
        "http://worker:8088" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def execute_job(config):
    identity = request_json("/runs", config)["run_key"]
    started = time.monotonic()
    deadline = max(1200, config.get("max_seconds", 600) + 240)
    while time.monotonic() - started < deadline:
        status = request_json(f"/runs/{identity}")
        if status["status"] == "succeeded":
            if not status.get("summary", {}).get("complete") or not status.get("receipt"):
                raise RuntimeError(f"Incomplete publication for {identity}")
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"{identity}: {status['error']}")
        time.sleep(3)
    raise TimeoutError(f"job still incomplete: {identity}")


def consolidate(results):
    rows = []
    for result in results:
        summary = result["summary"]
        if not summary["complete"] or not result["receipt"]["mlflow_run_id"]:
            raise RuntimeError("Cannot consolidate an incomplete benchmark")
        rows.append({"run_key": result["run_key"], **summary,
                     "mlflow_run_id": result["receipt"]["mlflow_run_id"]})
    if not rows or len({r["run_key"] for r in rows}) != len(rows):
        raise ValueError("Expected a nonempty matrix of unique runs")
    # Protocol is retained per row: temporal holdout is never pooled with prequential scores.
    return {"complete": True, "runs": len(rows), "results": rows}
