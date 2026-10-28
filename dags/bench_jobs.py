"""Finite worker jobs and reproducible matrices; no per-event scheduler tasks."""

import json
import time
import urllib.request
from copy import deepcopy
from pathlib import Path


def tracking_profile():
    path = Path(__file__).resolve().parents[1] / "configs/tracking-metrics-profile.json"
    return json.loads(path.read_text())


def configurations(params, suite):
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


def nonlinear_configurations(params):
    """Paired v4 protocols and budgets, with model families interleaved per case."""
    models = (
        {"name": "logistic", "params": {"learning_rate": 0.05}},
        {"name": "hoeffding_tree", "params": {
            "grace_period": 50, "max_depth": 8, "max_size": 8,
            "memory_estimate_period": 100, "leaf_prediction": "nba",
        }},
        {"name": "knn", "params": {"n_neighbors": 11, "window_size": 256, "weighted": True}},
    )
    template = versioned_configurations(params)
    # The deployed Airflow mapping limit is 1024; fail in plan, before task expansion.
    case_count = len(template) * len(models)
    if case_count > 1024:
        raise ValueError(
            f"nonlinear campaign has {case_count} configurations; Airflow max_map_length=1024 "
            "allows at most six seeds"
        )

    def model_for_source(config, model):
        model = deepcopy(model)
        nominal = {
            "agrawal": ["elevel", "car", "zipcode"],
            "stagger": ["size", "color", "shape"],
        }
        if model["name"] == "hoeffding_tree" and config["dataset"]["name"] in nominal:
            # River generators encode these categorical features as integers.
            model["params"]["nominal_attributes"] = nominal[config["dataset"]["name"]]
        return model

    return [
        {**deepcopy(config), "name": "nonlinear-v5", "model": model_for_source(config, model)}
        for config in template
        for model in models
    ]


def versioned_configurations(params):
    """All five datasets; deterministic real prefixes are not fake independent seed trials."""
    synthetic = controlled_configurations(params)
    result = [{**c, "name": "versioned-v4"} for c in synthetic if not c["name"].endswith("bridge")]
    # Reuse the same strategy matrix for each real dataset, once (seed has no effect on these prefixes).
    first_seed = {**params, "seeds": [params["seeds"][0]]}
    template = [c for c in controlled_configurations(first_seed)
                if c["dataset"]["name"] == "sea" and not c["name"].endswith("bridge")]
    for dataset in ("electricity", "airlines"):
        result += [{**c, "name": "versioned-v4", "dataset": {**c["dataset"], "name": dataset}}
                   for c in template]
    return result


def controlled_configurations(params):
    """Five paired seeds; shared sample-visit ceilings, not forced equal consumption."""
    result = []
    labels = params["samples"] - 100
    for seed in params["seeds"]:
        for dataset in ("sea", "agrawal", "stagger"):
            for mode in ("streaming", "incremental"):
                variants = [("none", "none", 0), ("none", "online", labels)]
                variants += [(detector, policy, budget)
                             for budget in (labels, 2 * labels)
                             for detector, policy in (("none", "periodic"),
                                                      ("adwin", "drift"), ("ddm", "drift"))]
                # Bridge reproduces the old 'online' behavior, only for the historical seed.
                if seed == 42:
                    variants += [("none", "sparse_bridge", None)]
                for detector, policy, budget in variants:
                    sparse = policy == "sparse_bridge"
                    result.append({
                        "name": "controlled-v3-bridge" if sparse else "controlled-v3",
                        "mode": mode, "dataset": {"name": dataset, "samples": params["samples"],
                                                   "seed": seed, "label_delay": params["label_delay"]},
                        "detector": {"name": detector}, "training_budget": budget,
                        "policy": {"name": "periodic" if sparse else policy,
                                   "every": 100 if sparse else 200,
                                   "cooldown": 0 if policy == "online" else 100,
                                   "action": "learn_one" if policy == "online" or sparse
                                   else "reset_retrain"},
                    })
            result.append({"name": "controlled-v3", "mode": "offline",
                           "dataset": {"name": dataset, "samples": params["samples"], "seed": seed,
                                       "label_delay": params["label_delay"]},
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
