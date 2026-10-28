"""One complete real-data pipeline after the recovery/integrity fixes are deployed."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

config_path = Path("/state/zen-validation-v4-final")
if not config_path.exists():
    shutil.copytree("/state/zen-config", config_path)
os.environ["ZENML_CONFIG_PATH"] = str(config_path)

from continual_bench.adapters.zenml_pipeline import execute_pipeline  # noqa: E402
from continual_bench.application.config import Config  # noqa: E402
from continual_bench.application.runner import environment  # noqa: E402

cfg = Config.model_validate(
    {
        "name": "versioned-v4-recovery-validation",
        "mode": "streaming",
        "dataset": {"name": "electricity", "samples": 180, "label_delay": 3},
        "warmup": 30,
        "metric_window": 30,
        "policy": {"name": "online", "action": "learn_one"},
        "training_budget": 150,
    }
)
root = Path("/state/final-probe-v4")
report = execute_pipeline(cfg, str(root), services=True)
directory = root / cfg.identity()
receipt = json.loads((directory / "receipt.json").read_text())
assert report["complete"] and report["transport"] == "kafka"
assert report["model_versions"]["count"] == 151 and len(report["updates"]) == 150
assert (directory / "mlflow-model/.ready.json").is_file()
assert report["environment"]["code_sha256"] == environment()["code_sha256"]
request = {"dataset": "electricity", "uri": receipt["model_uri"],
           "probe": report["model_versions"]["inference_probe"]}
checked = subprocess.run(
    [sys.executable, str(Path(__file__).with_name("registry_inference_probe.py"))],
    input=json.dumps(request), text=True, capture_output=True, check=True,
)
result = {"complete": True, "config": cfg.model_dump(mode="json"), "receipt": receipt,
          "model_versions": report["model_versions"], "metrics": report["metrics"],
          "source_sha256": environment()["code_sha256"], "registry_probe": json.loads(checked.stdout)}
(root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print("Final deployed pipeline and Registry inference verified:", receipt["model_uri"])
