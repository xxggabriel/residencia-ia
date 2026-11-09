"""Admission, duplicate protection and eight independent executors; no external services."""

import importlib.util
import io
import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pytest

from continual_bench.infrastructure.storage import atomic_write
from continual_bench.orchestration.worker_pool import WorkerPool


class Response(io.BytesIO):
    status = 200


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_CONCURRENCY", "1")
    path = Path(__file__).parents[1] / "src/continual_bench/orchestration/server.py"
    backends = []
    for index in range(8):
        spec = importlib.util.spec_from_file_location(f"independent_worker_{index}", path)
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
        backends.append(backend)
    release = threading.Event()
    started = [threading.Event() for _ in backends]
    counts = [0] * 8
    peak = [0] * 8
    count_guard = threading.Lock()

    def perform(index, root, payload):
        path = root / payload["identity"] / "job.json"
        with count_guard:
            counts[index] += 1
            peak[index] = max(peak[index], counts[index])
        started[index].set()
        try:
            assert release.wait(10), "test workload was not released"
            atomic_write(path, json.dumps({"status": "succeeded", "run_key": payload["identity"]}).encode())
        finally:
            with count_guard:
                counts[index] -= 1

    def opener(request, timeout):
        url = urlparse(request.full_url)
        index = int(url.hostname.removeprefix("slot-"))
        backend = backends[index]
        if url.path == "/health":
            code, result = 200, backend.health_snapshot()
        elif url.path in {"/ready", "/datasets-ready"}:
            code, result = 200, {"ready": True, "datasets": 17, "code_sha256": "fixture-code"}
        else:
            root = tmp_path / ("epochs" if url.path.startswith("/epoch-runs") else "binary")
            if request.data is not None:
                payload = json.loads(request.data)
                code, result = backend.submit_job(
                    root, payload["identity"], payload,
                    lambda value: perform(index, root, value),
                    key_field="request_key" if url.path == "/epoch-runs" else "run_key")
            else:
                path = root / url.path.rsplit("/", 1)[-1] / "job.json"
                code, result = (200, json.loads(path.read_text())) if path.exists() else (404, {})
        if code >= 400:
            raise urllib.error.HTTPError(request.full_url, code, "fixture", {},
                                         io.BytesIO(json.dumps(result).encode()))
        response = Response(json.dumps(result).encode())
        response.status = code
        return response

    pool = WorkerPool([f"http://slot-{i}:8088" for i in range(8)], opener)
    yield pool, backends, started, release, peak, tmp_path
    release.set()
    for backend in backends:
        backend.EXECUTOR.shutdown(wait=True)


def test_eight_parallel_jobs_use_eight_executors_and_reject_a_ninth(cluster):
    pool, backends, started, release, peak, root = cluster
    with ThreadPoolExecutor(max_workers=8) as client:
        requests = [client.submit(pool.submit, "/epoch-runs" if i % 2 else "/runs",
                                  json.dumps({"identity": f"{i:024x}"}).encode()) for i in range(8)]
        assert all(result.result()[0] == 202 for result in requests)
    assert all(event.wait(2) for event in started)
    code, health = pool.get("/health")
    assert code == 200 and health["active"] == health["concurrency"] == 8
    assert peak == [1] * 8
    assert pool.submit("/runs", json.dumps({"identity": "f" * 24}).encode())[0] == 429
    release.set()
    for backend in backends:
        backend.EXECUTOR.shutdown(wait=True)
    for i in range(8):
        path = f"/epoch-runs/{i:024x}" if i % 2 else f"/runs/{i:024x}"
        assert pool.get(path)[1]["status"] == "succeeded"
    assert pool.get("/health")[1]["active"] == 0


def test_duplicate_job_across_workers_is_executed_once(cluster):
    pool, backends, started, release, peak, root = cluster
    body = json.dumps({"identity": "a" * 24}).encode()
    assert pool.submit("/runs", body)[0] == 202
    assert started[0].wait(2)
    # Placement selects an idle container, but the shared file claim blocks a duplicate.
    assert pool.submit("/runs", body)[0] == 202
    assert sum(backend.health_snapshot()["queue"] for backend in backends) == 1
    assert not any(event.is_set() for event in started[1:])
    release.set()
    for backend in backends:
        backend.EXECUTOR.shutdown(wait=True)
    assert pool.submit("/runs", body)[0] == 200
    assert peak == [1] + [0] * 7


def test_failed_job_releases_its_slot_and_reports_failure(cluster):
    pool, backends, started, release, peak, root = cluster
    backend = backends[0]

    def failure(_config):
        raise ValueError("fixture")

    identity = "b" * 24
    assert backend.submit_job(root, identity, {}, failure)[0] == 202
    backend.EXECUTOR.shutdown(wait=True)
    assert json.loads((root / identity / "job.json").read_text())["status"] == "failed"
    assert backend.health_snapshot()["queue"] == backend.health_snapshot()["active"] == 0
    assert backend.SLOTS.acquire(blocking=False)
    backend.SLOTS.release()
    # The job claim is no longer held after failure.
    from continual_bench.infrastructure.storage import run_lock
    with run_lock(root / ".job-claims", identity):
        pass


def test_timeout_after_submission_is_not_resent_to_another_worker():
    posts = []

    def opener(request, timeout):
        if request.data is not None:
            posts.append(request.full_url)
            raise urllib.error.URLError("timed out after possible acceptance")
        return Response(b'{"queue":0,"active":0,"concurrency":1}')

    pool = WorkerPool([f"http://slot-{i}" for i in range(8)], opener)
    assert pool.submit("/runs", b'{}')[0] == 503
    assert len(posts) == 1


def test_readiness_requires_all_eight_workers_and_status_can_fail_over(cluster):
    pool, backends, started, release, peak, root = cluster
    assert pool.get("/ready")[0] == pool.get("/datasets-ready")[0] == 200
    assert pool.get("/datasets-ready")[1]["code_sha256"] == "fixture-code"
    previous = pool.opener

    def unavailable(request, timeout):
        if urlparse(request.full_url).hostname == "slot-0":
            raise urllib.error.URLError("unavailable")
        return previous(request, timeout)

    body = json.dumps({"identity": "c" * 24}).encode()
    assert pool.submit("/runs", body)[0] == 202
    pool.opener = unavailable
    assert pool.get("/ready")[0] == 503
    assert pool.get("/health")[1]["concurrency"] == 7
    assert pool.get("/runs/" + "c" * 24)[0] == 200
    assert pool.get("/unexpected")[0] == pool.submit("/unexpected", b'{}')[0] == 404


@pytest.mark.parametrize("count", [1, 7, 9])
def test_standard_pool_requires_exactly_eight_distinct_workers(count):
    with pytest.raises(ValueError):
        WorkerPool([f"http://slot-{i}" for i in range(count)])
    with pytest.raises(ValueError):
        WorkerPool(["http://same-slot"] * 8)


def test_shared_tracking_probes_mlflow_once_for_a_healthy_pool():
    tracking_probes = []

    def opener(request, timeout):
        if request.full_url.endswith("/ready"):
            tracking_probes.append(request.full_url)
            return Response(b'{"ready":true}')
        return Response(b'{"queue":0,"active":0,"concurrency":1}')

    urls = [f"http://slot-{i}" for i in range(8)]
    pool = WorkerPool(urls, opener, shared_tracking=True)
    assert pool.get("/health")[1]["concurrency"] == 8
    assert pool.get("/ready")[1]["ready"] is True
    assert tracking_probes == ["http://slot-0/ready"]


def test_dataset_admission_requires_matching_code_and_preserves_dag_metadata():
    def opener(request, timeout):
        code = "other-code" if urlparse(request.full_url).hostname == "slot-7" else "same-code"
        return Response(json.dumps({"ready": True, "datasets": 17, "code_sha256": code}).encode())

    pool = WorkerPool([f"http://slot-{i}" for i in range(8)], opener)
    assert pool.get("/datasets-ready")[0] == 503
    assert not pool.get("/datasets-ready")[1]["ready"]
