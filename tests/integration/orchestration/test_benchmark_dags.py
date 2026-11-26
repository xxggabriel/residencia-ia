"""Tests for active Airflow DAGs (benchmark_campaign and bench_readiness) without legacy ZenML/bench_jobs."""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_dag_module(name: str):
    dags_dir = Path(__file__).parents[3] / "dags"
    if str(dags_dir) not in sys.path:
        sys.path.insert(0, str(dags_dir))
    spec = importlib.util.spec_from_file_location(name, dags_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_bench_readiness_excludes_zenml() -> None:
    readiness = _load_dag_module("bench_readiness")
    assert "zenml" not in readiness.HTTP_DEPENDENCIES
    assert set(readiness.HTTP_DEPENDENCIES) == {"mlflow", "minio", "worker"}


def test_benchmark_campaign_payload_and_parallel_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("airflow")
    campaign_dag = _load_dag_module("benchmark_campaign")

    payload = campaign_dag.build_campaign_payload(
        {
            "env": "docker",
            "config_path": "configs/campaigns/cara-electricity.yaml",
            "protocol": "cara",
            "dataset": "electricity",
            "campaign": "",
            "model": "random_forest",
            "policies": ["cara", "frozen"],
            "seeds": [0, 1],
            "save_checkpoints": False,
        }
    )
    assert payload["env"] == "docker"
    assert payload["config_path"] == "configs/campaigns/cara-electricity.yaml"
    assert payload["policies"] == ["cara", "frozen"]
    assert payload["seeds"] == [0, 1]
    assert payload["save_checkpoints"] is False

    calls: list[tuple[str, dict | None]] = []

    def fake_request(path: str, body: dict | None = None) -> dict:
        calls.append((path, body))
        if path == "/campaign-seeds":
            return {"seed_key": "0" * 24}
        if path == f"/campaign-seeds/{'0' * 24}":
            return {"status": "succeeded", "seed": 0, "seed_path": "/state/seed_0.pkl"}
        if path == "/campaign-consolidate":
            return {"consolidate_key": "1" * 24}
        if path == f"/campaign-consolidate/{'1' * 24}":
            return {"status": "succeeded", "protocol": "cara", "dataset": "electricity"}
        raise AssertionError(f"Unexpected path {path}")

    monkeypatch.setattr(campaign_dag, "_worker_request", fake_request)
    seed_job = {
        "campaign_key": "abc",
        "seed": 0,
        "config": {"name": "c"},
        "env": "docker",
        "save_checkpoints": False,
    }
    seed_res = campaign_dag.execute_seed_job(seed_job)
    assert seed_res["status"] == "succeeded"

    cons_res = campaign_dag.execute_consolidate_job([seed_res], [seed_job])
    assert cons_res["status"] == "succeeded"
    assert cons_res["protocol"] == "cara"
