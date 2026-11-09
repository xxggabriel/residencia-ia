"""Shared, service-independent benchmark definitions for Airflow and AWS Batch."""

import json
from copy import deepcopy

DATASETS = (
    "sea", "agrawal", "stagger", "hyperplane", "rbf_drift", "friedman_drift",
    "electricity", "airlines", "insects", "gas", "split_mnist", "permuted_mnist",
    "split_cifar10", "split_cifar100", "core50", "clear10", "clear100",
)


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


def project_configurations(params, code):
    if len(params["seeds"]) != 1:
        raise ValueError("this full-project protocol requires exactly one paired seed")
    # Retain all prior paired models, modes and budgets; use the new isolated worker/code.
    base = nonlinear_configurations({"samples": params["samples"], "seeds": params["seeds"],
                                     "label_delay": params["label_delay"]})
    campaign = "semana09-full-" + code[:12]
    legacy = [{"kind": "binary", "config": {**c, "name": campaign,
               "tracking": {"probe_samples": 16, "continual_references": False,
                            "task_evaluation": {"horizon": 16, "gap": 1, "max_windows": 4}}}} for c in base]
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
