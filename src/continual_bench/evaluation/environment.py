"""Runtime identity: package versions, code digest, locks fingerprint and memory probes."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from continual_bench.core.observability import peak_rss_bytes as peak_rss_bytes

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_LOCKS = ("image-datasets.lock.json", "real-streams.lock.json", "temporal-vision.lock.json")


def environment():
    versions = {}
    for package in (
        "continual-bench",
        "river",
        "pydantic",
        "zenml",
        "mlflow-skinny",
        "confluent-kafka",
        "pandas",
        "numpy",
        "scipy",
        "pillow",
        "pyarrow",
        "scikit-learn",
        "matplotlib",
        "xgboost",
        "psutil",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        digest.update(str(path.relative_to(PACKAGE_ROOT)).encode())
        digest.update(path.read_bytes())
    execution = os.environ.get("BENCH_EXECUTION_ENV")
    if not execution:
        if os.environ.get("SM_TRAINING_ENV"):
            execution = "aws_sagemaker"
        elif job := os.environ.get("AWS_BATCH_JOB_ID"):
            execution = "local_batch" if job.startswith("local-") else "aws_batch"
        else:
            execution = "local"
    return {
        "execution": execution,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "code_sha256": digest.hexdigest(),
        "git_revision": os.environ.get("BENCH_GIT_REVISION", "uncommitted"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
    }


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def dataset_fingerprint(env=None):
    """Code, dependency and dataset-lock identity that admits a full dataset suite."""
    env = environment() if env is None else env
    locks = {}
    for name in DATASET_LOCKS:
        candidates = (PACKAGE_ROOT.parents[1] / "configs" / "contracts" / name, PACKAGE_ROOT / name)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"dataset contract must accompany the runtime: {name}")
        locks[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(
        encoded(
            {
                "locks": locks,
                "code": env["code_sha256"],
                "python": env.get("python"),
                "packages": env.get("packages"),
            }
        )
    ).hexdigest()


def current_rss_bytes():
    """Linux current RSS; peak RSS is not substituted for an integral elsewhere."""
    path = Path("/proc/self/statm")
    if not path.exists():
        return None
    try:
        return int(path.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None
