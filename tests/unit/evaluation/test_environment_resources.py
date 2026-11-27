import json

from continual_bench.evaluation.environment import runtime_resources


def test_runtime_resources_exposes_only_reproduction_metadata(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "not-for-publication")
    monkeypatch.setenv("SSH_CONNECTION", "private-connection")
    resources = runtime_resources()
    assert resources["logical_cpus"] >= 1
    assert resources["thread_environment"]["OMP_NUM_THREADS"] == "2"
    assert set(resources["cgroup_v2_limits"]) <= {"cpu.max", "memory.max", "cpuset.cpus.effective"}
    encoded = json.dumps(resources)
    assert "not-for-publication" not in encoded and "private-connection" not in encoded
    assert "hostname" not in resources
