"""JSON operational event contract shared by workers and the Airflow reader."""

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from contextlib import closing
from datetime import UTC, datetime
from uuid import uuid4

TASK_FIELDS = ("dag_id", "dag_run_id", "task_id", "map_index", "try_number")
SCHEMA_VERSION = 1


def task_context(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) != set(TASK_FIELDS):
        raise ValueError("invalid Airflow task context fields")
    for name in TASK_FIELDS[:3]:
        if not isinstance(value[name], str) or not 1 <= len(value[name]) <= 512:
            raise ValueError(f"invalid {name}")
    if type(value["map_index"]) is not int or value["map_index"] < -1:
        raise ValueError("invalid map_index")
    if type(value["try_number"]) is not int or value["try_number"] < 1:
        raise ValueError("invalid try_number")
    return {**value, "task_attempt_key": task_key(value)}


def task_key(context):
    encoded = json.dumps([context[name] for name in TASK_FIELDS]).encode()
    return hashlib.sha256(encoded).hexdigest()


class JsonFormatter(logging.Formatter):
    """Preserve the fields used by the cloud worker and add the event envelope."""

    def __init__(self, context, *, operational=False):
        super().__init__()
        self.context = context
        self.operational = operational

    def format(self, record):
        fields = getattr(record, "fields", {})
        entry = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": self.formatTime(record),
            **self.context,
            **fields,
        }
        if self.operational:
            entry.update(
                schema_version=SCHEMA_VERSION,
                event_id=record.event_id,
                timestamp=datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
                level=record.levelname,
                event_name=fields["event_name"],
                message=record.getMessage(),
                service="benchmark-worker",
                environment=os.environ.get("BENCH_LOG_ENVIRONMENT", "docker"),
            )
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def open_spool(path):
    db = sqlite3.connect(path, timeout=0.5)
    try:
        if db.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, line TEXT NOT NULL)"
        )
    except sqlite3.Error:
        db.close()
        raise
    return db


def append_event(path, event_id, line):
    for attempt in range(3):
        try:
            with closing(open_spool(path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("INSERT OR IGNORE INTO events(event_id, line) VALUES (?, ?)", (event_id, line))
                # Bound the backlog during an extended outage. Use seq, not wall-clock time.
                db.execute("DELETE FROM events WHERE seq <= last_insert_rowid() - 20000")
            return
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))


class OperationalHandler(logging.StreamHandler):
    def emit(self, record):
        line = self.format(record)
        try:
            self.stream.write(line + "\n")
            self.flush()
        except OSError:
            self.handleError(record)
        if path := os.environ.get("BENCH_EVENT_SPOOL"):
            try:
                append_event(path, record.event_id, line)
            except (OSError, sqlite3.Error):
                # Stdout remains the local record. Failures in telemetry cannot fail a job.
                self.handleError(record)


logger = logging.getLogger("continual_bench.worker.events")
handler = OperationalHandler(sys.stdout)
handler.setFormatter(JsonFormatter({}, operational=True))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def emit_event(event_name, message, *, event_id=None, level=logging.INFO, **fields):
    logger.log(
        level,
        message,
        extra={
            "event_id": event_id or uuid4().hex,
            "fields": {"event_name": event_name, **fields},
        },
    )
