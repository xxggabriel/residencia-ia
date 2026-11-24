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


def runtime_resources():
    """Reproduction metadata without hostnames, private addresses or credentials.

    Host totals and the process/container limits are separate observations. A shared
    machine's memory and CPU count are not treated as a reservation for this run.
    """
    cpu_model = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    memory = None
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = None
    limits = {}
    roots = [Path("/sys/fs/cgroup")]
    try:
        line = next(x for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
        roots.insert(0, roots[0] / line[3:].lstrip("/"))
    except (OSError, StopIteration):
        pass
    for filename in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        for root in roots:
            try:
                limits[filename] = (root / filename).read_text().strip()
                break
            except OSError:
                continue
    return {
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity_count,
        "host_memory_bytes": memory,
        "cgroup_v2_limits": limits,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "scope": "observed_host_and_process_limits_not_dedicated_resource_reservation",
    }


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
        "xgboost-cpu",
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
        "resources": runtime_resources(),
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
