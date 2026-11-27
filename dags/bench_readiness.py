"""Finite read-only checks used by the Airflow admission task."""

import json
import os
import urllib.error
import urllib.request

HTTP_DEPENDENCIES = {
    "mlflow": os.environ.get("BENCH_MLFLOW_HEALTH_URL", "http://mlflow:5000/health"),
    "minio": os.environ.get("BENCH_MINIO_HEALTH_URL", "http://minio:9000/minio/health/live"),
    "worker": os.environ.get("BENCH_WORKER_HEALTH_URL", "http://worker:8088/ready"),
}


def require_dependencies(opener=None, *, include_worker: bool = True):
    failures = []
    deps = {
        k: v
        for k, v in HTTP_DEPENDENCIES.items()
        if include_worker or k != "worker"
    }
    for name, url in deps.items():
        try:
            with (opener or urllib.request.urlopen)(url, timeout=4) as response:
                if response.status != 200:
                    failures.append(f"{name}: HTTP {response.status}")
        except urllib.error.HTTPError as error:
            detail = f"HTTP {error.code}"
            if name == "worker":
                try:
                    detail += ": " + json.loads(error.read(4096)).get("error", "not ready")[:500]
                except (ValueError, TypeError, AttributeError):
                    pass
            failures.append(f"{name}: {detail}")
        except (urllib.error.URLError, OSError, TimeoutError):
            failures.append(f"{name}: unreachable")
    if failures:
        raise RuntimeError(
            "Benchmark stack is not ready: "
            + "; ".join(failures)
            + ". No benchmark was submitted. Check deployment readiness before rerunning."
        )
    return {"ready": True, "checked": list(deps)}
