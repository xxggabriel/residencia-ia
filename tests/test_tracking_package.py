import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from continual_bench.infrastructure.tracking import (
    AsyncMetricWriter,
    MlflowClientBackend,
    RestBackend,
    batch_payloads,
    missing_metric_points,
)
from continual_bench.infrastructure.tracking.artifacts import artifact_plan, upload_concurrently
from continual_bench.infrastructure.tracking.http import PooledHttpOpener
from continual_bench.infrastructure.tracking.rounds import publish_training_rounds


def test_batches_share_requests_and_respect_mlflow_limits():
    single = list(batch_payloads(range(300), [("p", "1")] * 20, [("t", "1")] * 10))
    assert len(single) == 1 and [len(part) for part in single[0]] == [300, 20, 10]

    chunks = list(batch_payloads(range(2500), [("p", str(i)) for i in range(150)], [("t", "1")] * 5))
    assert sum(len(m) for m, _, _ in chunks) == 2500
    assert sum(len(p) for _, p, _ in chunks) == 150
    for metrics, params, tags in chunks:
        assert len(metrics) <= 1000 and len(params) <= 100 and len(tags) <= 100
        assert len(metrics) + len(params) + len(tags) <= 1000
    assert list(batch_payloads()) == []


def test_reconciliation_reads_existing_histories_concurrently_and_rejects_conflicts():
    active, peak, lock = 0, 0, threading.Lock()

    class Client:
        concurrency = 4

        def get_metric_history(self, run_id, key):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return [(0, 1.0)] if key != "conflict" else [(0, 2.0)]

    points = [(f"k{i}", 0, 1.0) for i in range(8)] + [("new", 0, 3.0), ("k0", 1, 4.0)]
    started = time.monotonic()
    missing = missing_metric_points(Client(), "run", points, known_keys={f"k{i}" for i in range(8)})
    assert time.monotonic() - started < 0.3 and peak > 1
    assert sorted(missing) == [("k0", 1, 4.0), ("new", 0, 3.0)]
    with pytest.raises(ValueError, match="remote metric conflicts"):
        missing_metric_points(Client(), "run", [("conflict", 0, 1.0)], known_keys={"conflict"})


def test_async_writer_coalesces_off_thread_and_flush_drains_everything():
    sent, gate = [], threading.Event()

    def send(batch):
        gate.wait(1)
        sent.append(sorted((p["key"], p["step"]) for p in batch))

    writer = AsyncMetricWriter(send, interval=60)
    started = time.monotonic()
    writer.submit([{"key": "rows", "step": 1}])
    writer.submit([{"key": "rows", "step": 2}, {"key": "acc", "step": 2}])
    assert time.monotonic() - started < 0.1  # the caller never waits on the sender
    gate.set()
    writer.flush()
    writer.close()
    flat = [point for batch in sent for point in batch]
    assert ("rows", 2) in flat and ("acc", 2) in flat

    every = []
    keeping = AsyncMetricWriter(every.extend, interval=60, coalesce=False)
    keeping.submit([{"key": "loss", "step": 1}])
    keeping.submit([{"key": "loss", "step": 2}])
    keeping.close()
    assert sorted(p["step"] for p in every) == [1, 2]


def test_artifact_plan_publishes_each_file_once(tmp_path):
    for relative in (
        "artifacts/plots/a.png",
        "artifacts/data/index.json",
        "report.json",
        "tracking-observations/fits/fit-1/epochs.png",
        "tracking-observations/overview.png",
        ".pending-x/skip",
        "run.lock",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    plan = artifact_plan(tmp_path, exclude=("tracking-observations/fits/fit-1",))
    assert [target for _, target in plan] == [
        "data/index.json",
        "plots/a.png",
        "report.json",
        "tracking-observations/overview.png",
    ]
    uploaded = []
    upload_concurrently(lambda path, target: uploaded.append(target), plan, workers=4)
    assert sorted(uploaded) == sorted(target for _, target in plan)


def _report():
    return {
        "training_history": [
            {
                "fit_id": f"fit-{i}",
                "action": "initial" if i == 1 else "retrain",
                "start_seq": 0,
                "end_seq": 10 * i,
                "steps": [{"step": 1, "global_step": i, "kind": "epoch", "metrics": {"loss": 1 / i}}],
                "summary": {},
            }
            for i in (1, 2, 3)
        ]
    }


def test_one_child_run_algorithm_serves_rest_and_client_backends(tmp_path, monkeypatch):
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    experiment = client.create_experiment("rounds")
    parent = client.create_run(experiment).info.run_id
    backend = MlflowClientBackend(client)
    sdk = publish_training_rounds(backend, experiment, parent, _report(), now_ms=7)
    assert publish_training_rounds(backend, experiment, parent, _report(), now_ms=7) == sdk
    assert [r["fit_id"] for r in sdk] == ["fit-1", "fit-2", "fit-3"]

    calls = []

    def request(method, path, *, payload=None, params=None):
        api = path.removeprefix("/api/2.0/mlflow/")
        calls.append(api)
        body = payload or params or {}
        if api == "runs/search":
            runs = client.search_runs([experiment], body["filter"])
            return {
                "runs": [
                    {
                        "info": {"run_id": r.info.run_id, "status": r.info.status},
                        "data": {
                            "tags": [{"key": k, "value": v} for k, v in r.data.tags.items()],
                            "metrics": [{"key": k} for k in r.data.metrics],
                        },
                    }
                    for r in runs
                ]
            }
        if api == "runs/create":
            run = client.create_run(experiment, tags={t["key"]: t["value"] for t in body["tags"]})
            return {
                "run": {
                    "info": {"run_id": run.info.run_id, "status": "RUNNING"},
                    "data": {"tags": body["tags"]},
                }
            }
        if api == "runs/log-batch":
            client.log_batch(
                body["run_id"],
                metrics=[Metric(**m) for m in body.get("metrics", [])],
                params=[Param(**p) for p in body.get("params", [])],
                tags=[RunTag(**t) for t in body.get("tags", [])],
                synchronous=True,
            )
            return {}
        if api == "runs/update":
            client.set_terminated(body["run_id"], body["status"])
            return {}
        raise AssertionError(api)

    other = client.create_run(experiment).info.run_id
    rest = publish_training_rounds(RestBackend(request, concurrency=1), experiment, other, _report())
    # Create + a single combined log-batch + terminate per fit.
    assert calls.count("runs/log-batch") == 3 and calls.count("runs/update") == 3
    child = client.get_run(rest[1]["run_id"])
    assert child.data.tags["mlflow.parentRunId"] == other and child.data.params["end_seq"] == "20"
    calls.clear()
    assert publish_training_rounds(RestBackend(request), experiment, other, _report()) == rest
    assert calls == ["runs/search"]


class _Handler(BaseHTTPRequestHandler):
    connections = set()

    def do_GET(self):  # noqa: N802
        type(self).connections.add(self.client_address)
        status = 404 if self.path == "/missing" else 200
        body = b'{"error_code": "RESOURCE_DOES_NOT_EXIST"}' if status == 404 else b'{"ok": true}'
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_pooled_opener_reuses_connections_and_maps_errors():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.protocol_version = "HTTP/1.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        opener = PooledHttpOpener()
        url = f"http://127.0.0.1:{server.server_port}"
        for _ in range(5):
            with opener(urllib.request.Request(f"{url}/ok"), timeout=5) as response:
                assert response.read() == b'{"ok": true}'
        assert len(_Handler.connections) == 1
        with pytest.raises(urllib.error.HTTPError) as error:
            opener(urllib.request.Request(f"{url}/missing"), timeout=5)
        assert error.value.code == 404 and b"RESOURCE_DOES_NOT_EXIST" in error.value.read()
        with pytest.raises(urllib.error.URLError):
            opener(urllib.request.Request("http://127.0.0.1:9/unreachable"), timeout=1)
    finally:
        server.shutdown()


def test_legacy_homelab_facade_uses_shared_algorithm(tmp_path, monkeypatch):
    from mlflow import MlflowClient

    from continual_bench.infrastructure.training_tracking import publish_sdk_training_rounds

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=Path(tmp_path / "mlruns").as_uri())
    experiment = client.create_experiment("legacy")
    parent = client.create_run(experiment).info.run_id
    receipts = publish_sdk_training_rounds(client, experiment, parent, _report(), now_ms=1)
    assert len(receipts) == 3
    assert all(client.get_run(r["run_id"]).info.status == "FINISHED" for r in receipts)
