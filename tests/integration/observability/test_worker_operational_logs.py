import io
import json
import logging
import threading
from contextlib import closing
from datetime import datetime

import pytest

from continual_bench.observability.events import JsonFormatter, open_spool, task_context
from continual_bench.observability.forward import forward_batch, to_gelf
from continual_bench.orchestration import server

TASK = {
    "dag_id": "benchmark_campaign",
    "dag_run_id": "manual__observability",
    "task_id": "run_seed",
    "map_index": 0,
    "try_number": 1,
}


def test_seed_claim_keeps_origin_when_another_airflow_attempt_attaches(tmp_path, monkeypatch):
    release = threading.Event()
    finished = threading.Event()
    events = []
    payload = {"campaign_key": "campaign", "seed": 42}
    key = server.campaign_identity(payload)
    root = tmp_path / "campaign_seeds"
    path = root / key / "job.json"

    def emit(name, message, **fields):
        events.append((name, fields))
        if name == "seed.finished":
            finished.set()

    def perform(config, *, progress, execution):
        assert config == payload
        assert release.wait(5)
        path.write_text(json.dumps({"seed_key": key, "status": "succeeded", "worker_execution": execution}))

    monkeypatch.setattr(server, "emit_event", emit)
    origin = task_context(TASK)
    observer = task_context({**TASK, "try_number": 2})
    try:
        first_code, first = server.submit_job(
            root, key, payload, perform, key_field="seed_key", observer=origin
        )
        second_code, second = server.submit_job(
            root,
            key,
            payload,
            perform,
            key_field="seed_key",
            observer=observer,
        )
        assert first_code == second_code == 202
        assert second["status"] == "queued_or_running"
        assert first["seed_key"] == second["seed_key"] == key
        assert first["task_association"]["association_role"] == "origin"
        assert second["task_association"]["association_role"] == "observer"
        assert first["worker_execution"] == second["worker_execution"]
    finally:
        release.set()
        assert finished.wait(5)

    _, third = server.submit_job(root, key, payload, perform, key_field="seed_key", observer=observer)
    assert third["worker_execution"]["try_number"] == 1
    assert third["task_association"] == second["task_association"]
    assert [e["try_number"] for name, e in events if name in {"seed.started", "seed.finished"}] == [1, 1]


def test_event_spool_preserves_payload_when_forwarding_fails(tmp_path, monkeypatch):
    path = tmp_path / "events.sqlite3"
    record = logging.LogRecord("worker", logging.INFO, __file__, 1, "seed started", (), None)
    record.event_id = "a" * 32
    record.fields = {"event_name": "seed.started", "run_id": "benchmark-id", "worker_host": "worker-1"}
    line = JsonFormatter(task_context(TASK), operational=True).format(record)
    parsed = json.loads(line)
    assert parsed["run_id"] != parsed["dag_run_id"]
    assert datetime.fromisoformat(parsed["timestamp"]).utcoffset().total_seconds() == 0
    gelf = to_gelf(parsed)
    assert gelf["_event_id"] == record.event_id
    assert gelf["_run_id"] == "benchmark-id"
    assert gelf["_dag_run_id"] == TASK["dag_run_id"]

    with closing(open_spool(path)) as db, db:
        db.execute("INSERT INTO events(event_id, line) VALUES (?, ?)", (record.event_id, line))

    def unavailable(request, *, timeout):
        raise TimeoutError

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(TimeoutError):
        forward_batch(path, "http://graylog/gelf")

    class Accepted(io.BytesIO):
        status = 202

    def recovered(request, *, timeout):
        assert json.loads(request.data) == gelf
        return Accepted()

    monkeypatch.setattr("urllib.request.urlopen", recovered)
    assert forward_batch(path, "http://graylog/gelf")
    assert not forward_batch(path, "http://graylog/gelf")


def test_event_spool_retries_a_contended_write(tmp_path, monkeypatch):
    from continual_bench.observability import events

    path = tmp_path / "contended.sqlite"
    connect = events.open_spool
    attempts = 0

    def open_after_contention(path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            error = events.sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = events.sqlite3.SQLITE_BUSY
            raise error
        return connect(path)

    monkeypatch.setattr(events, "open_spool", open_after_contention)
    events.append_event(path, "id", "line")
    assert attempts == 3
    with closing(connect(path)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("SELECT event_id, line FROM events").fetchall() == [("id", "line")]
