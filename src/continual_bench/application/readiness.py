"""Read-only tracking precondition, before jobs or benchmark artifacts are created."""

import base64
import json
import os
import urllib.error
import urllib.request


class StackNotReady(RuntimeError):
    pass


def ensure_tracking_ready(environ=None, opener=None):
    environment = os.environ if environ is None else environ
    password = environment.get("MLFLOW_TRACKING_PASSWORD", "")
    if len(password) < 12:
        raise StackNotReady(
            "MLflow 3.17.2 requires an admin password of at least 12 characters. "
            "An approved MLFLOW_PASSWORD is still required; no benchmark was submitted."
        )
    uri = environment.get("MLFLOW_TRACKING_URI", "")
    username = environment.get("MLFLOW_TRACKING_USERNAME", "")
    if not uri.startswith(("http://", "https://")) or not username:
        raise StackNotReady("Explicit MLflow URI and username are required; no benchmark was submitted.")
    authorization = base64.b64encode(f"{username}:{password}".encode()).decode()
    request = urllib.request.Request(
        uri.rstrip("/") + "/api/2.0/mlflow/experiments/search?max_results=1",
        headers={"Authorization": "Basic " + authorization},
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=4) as response:
            if response.status != 200 or not isinstance(json.loads(response.read(65536)), dict):
                raise StackNotReady(
                    "MLflow returned an invalid readiness response; no benchmark was submitted."
                )
    except urllib.error.HTTPError as error:
        raise StackNotReady(
            f"MLflow tracking rejected readiness (HTTP {error.code}); no benchmark was submitted."
        ) from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise StackNotReady("MLflow tracking is unreachable; no benchmark was submitted.") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise StackNotReady("MLflow returned invalid API data; no benchmark was submitted.") from None
    return {"ready": True, "mlflow": "authenticated"}
