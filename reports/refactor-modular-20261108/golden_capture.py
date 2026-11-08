"""Capture deterministic benchmark outputs (no timings/code hashes) for refactor equivalence."""

import json
import re
import sys
import tempfile
from pathlib import Path

try:
    from continual_bench.application.config import Config, DatasetConfig
except ImportError:  # refactored layout
    from continual_bench.config import Config, DatasetConfig
from continual_bench.application.runner import run

try:
    from continual_bench.application.typed_benchmark import run_typed
    from continual_bench.application.registry import default_registry, make_source
except ImportError:  # refactored layout
    from continual_bench.evaluation.typed import run_typed
    from continual_bench.application.components import default_registry, make_source
try:
    from continual_bench.application.epoch_benchmark import EpochConfig, run_epochs
except ImportError:
    from continual_bench.evaluation.epochs import EpochConfig, run_epochs

VOLATILE = re.compile(r"second|latency|throughput|rss|memory|byte|environment|code_sha|elapsed|wall|"
                      r"model_versions|digest_of_code|runtime_fingerprint|contract|path|cache|"
                      r"identity|run_key|request_key|report|^key$|python|packages|platform|samples_sha|"
                      r"cost_training_time|cost_prediction_time|cost_total|instrumentation|"
                      r"version_id|state_sha|final_version|code_uri|dataset_version|^config$|^dataset$")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in sorted(value.items()) if not VOLATILE.search(str(k))}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return round(value, 10)
    return value


def binary_cases():
    models = [{"name": "logistic"}, {"name": "hoeffding_tree"}, {"name": "knn"}]
    for dataset in ("sea", "agrawal", "stagger", "hyperplane", "rbf_drift"):
        for model in models:
            for mode, policy, detector, extra in (
                ("streaming", "drift", "adwin", {}),
                ("streaming", "online", "none", {"action": "learn_one"}),
                ("incremental", "periodic", "ddm", {}),
                ("incremental", "performance", "none", {"params": {"error_threshold": 0.3}}),
                ("streaming", "cost_heuristic", "none", {}),
                ("offline", "none", "none", {}),
            ):
                yield {
                    "name": "golden", "mode": mode, "model": model,
                    "dataset": {"name": dataset, "samples": 600, "label_delay": 3},
                    "warmup": 50, "metric_window": 50, "detector": {"name": detector},
                    "policy": {"name": policy, "every": 100, "cooldown": 50, "min_samples": 50,
                               "min_window": 25, **extra},
                    "tracking": {"probe_samples": 8, "continual_references": dataset == "sea",
                                 "task_evaluation": {"horizon": 8, "gap": 1, "max_windows": 2}},
                }
    yield {"name": "golden-promo", "mode": "streaming",
           "dataset": {"name": "sea", "samples": 600, "label_delay": 3}, "warmup": 50,
           "policy": {"name": "periodic", "every": 100, "cooldown": 50, "min_samples": 50, "min_window": 25},
           "promotion": {"validation_samples": 16}, "training_budget": 300}


def main(output):
    results = {}
    with tempfile.TemporaryDirectory() as root:
        for index, data in enumerate(binary_cases()):
            config = Config.model_validate(data)
            report = run(config, root)
            tracking = report.get("tracking", {})
            results[f"binary-{index}"] = {"identity": config.identity(),
                                          "digest": report["prediction_digest"] if "prediction_digest" in report
                                          else None, "report": clean(report), "tracking": clean(tracking)}
        registry = default_registry()
        for policy in ("none", "online", "periodic"):
            source = make_source(registry, DatasetConfig(name="friedman_drift", samples=400, label_delay=2))
            report = run_typed(source, root + "/typed", policy=policy, max_samples=400)
            results[f"typed-friedman-{policy}"] = clean(report) | {"digest": report["prediction_digest"]}
        for name in ("sea", "agrawal", "stagger", "hyperplane", "rbf_drift", "friedman_drift"):
            report = run_epochs(EpochConfig(dataset=name, examples=200, epochs=2), root + "/epochs",
                                root + "/cache")
            results[f"epochs-{name}"] = clean(report) | {
                "digests": [e["prediction_digest"] for e in report["epochs"]]}
    Path(output).write_text(json.dumps(results, indent=1, sort_keys=True))
    print(len(results), "cases")


if __name__ == "__main__":
    main(sys.argv[1])
