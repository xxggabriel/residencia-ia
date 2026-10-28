"""Unit checks of fail-fast behavior; these do not claim external service validation."""

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

from continual_bench.application.config import Config
from continual_bench.application.readiness import StackNotReady, ensure_tracking_ready


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return b'{"experiments": []}'


def environment():
    return {
        "MLFLOW_TRACKING_PASSWORD": "unit-fixture-password",
        "MLFLOW_TRACKING_USERNAME": "fixture",
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    }


def test_short_password_fails_without_network_or_secret_disclosure():
    def unexpected(*args, **kwargs):
        pytest.fail("must reject before any network request")

    with pytest.raises(StackNotReady, match="at least 12") as error:
        ensure_tracking_ready({**environment(), "MLFLOW_TRACKING_PASSWORD": "short"}, unexpected)
    assert "short" not in str(error.value)


def test_authenticated_readiness_is_a_read_only_api_request():
    seen = []

    def opener(request, timeout):
        seen.append(request)
        assert timeout == 4
        return Response()

    assert ensure_tracking_ready(environment(), opener)["ready"]
    assert seen[0].get_method() == "GET"
    assert seen[0].full_url.endswith("/experiments/search?max_results=1")
    assert seen[0].has_header("Authorization")
    assert seen[0].data is None


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("http://fixture", 401, "unauthorized", {}, io.BytesIO(b"do-not-echo-body")),
        urllib.error.URLError("do-not-echo-connection-details"),
    ],
)
def test_unavailable_or_unauthenticated_tracking_is_explicit_and_redacted(error):
    def opener(*args, **kwargs):
        raise error

    with pytest.raises(StackNotReady) as result:
        ensure_tracking_ready(environment(), opener)
    assert "no benchmark was submitted" in str(result.value).lower()
    assert "do-not-echo" not in str(result.value)


def test_services_cli_fails_before_materialization(tmp_path, monkeypatch):
    from continual_bench.application import runner

    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "short")

    def unexpected(*args, **kwargs):
        pytest.fail("must not materialize or train")

    monkeypatch.setattr(runner, "prepare", unexpected)
    with pytest.raises(StackNotReady, match="at least 12"):
        runner.run(Config(), tmp_path / "no-artifacts", services=True)
    assert not (tmp_path / "no-artifacts").exists()


def test_zenml_fails_before_stack_mutation_or_pipeline_creation(tmp_path, monkeypatch):
    from continual_bench.adapters import zenml_pipeline

    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "short")

    def unexpected(*args, **kwargs):
        pytest.fail("must not create a pipeline or change stack")

    monkeypatch.setattr(zenml_pipeline, "configure_local_stack", unexpected)
    with pytest.raises(StackNotReady, match="at least 12"):
        zenml_pipeline.execute_pipeline(Config(), tmp_path, services=True)


def dag_readiness():
    path = Path(__file__).parents[1] / "dags/bench_readiness.py"
    spec = importlib.util.spec_from_file_location("bench_readiness_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("unavailable", ["mlflow", "worker"])
def test_airflow_admission_names_missing_dependency_without_submitting_jobs(unavailable):
    module = dag_readiness()

    def opener(url, timeout):
        assert timeout == 4
        assert "/runs" not in url
        if f"//{unavailable}:" in url:
            if unavailable == "worker":
                payload = json.dumps(
                    {"error": "MLflow requires an approved password of at least 12 characters"}
                )
                raise urllib.error.HTTPError(url, 503, "unready", {}, io.BytesIO(payload.encode()))
            raise urllib.error.URLError("unavailable")
        return Response()

    with pytest.raises(RuntimeError, match=unavailable) as error:
        module.require_dependencies(opener)
    assert "No benchmark was submitted" in str(error.value)


def test_airflow_admission_accepts_healthy_read_only_checks():
    module = dag_readiness()
    assert module.require_dependencies(lambda *args, **kwargs: Response())["ready"]
