"""Finite read-only checks used by the Airflow admission task."""

import json
import urllib.error
import urllib.request

HTTP_DEPENDENCIES = {
    "mlflow": "http://mlflow:5000/health",
    "minio": "http://minio:9000/minio/health/live",
    "zenml": "http://zenml:8080/health",
    "worker": "http://worker:8088/ready",
}


def require_dependencies(opener=None):
    failures = []
    for name, url in HTTP_DEPENDENCIES.items():
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
    return {"ready": True, "checked": list(HTTP_DEPENDENCIES)}
