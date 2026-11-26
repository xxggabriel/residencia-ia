"""Bounded Graylog queries and job-association rules for the task log reader."""

import base64
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from continual_bench.observability.events import SCHEMA_VERSION, task_key

PAGE_SIZE = 32
MAX_ROWS = 128
POLL_INTERVAL = 2
HEX_ID = re.compile(r"[a-f0-9]{32}")


class EventReader(Protocol):
    def read(self, context: dict, start: datetime) -> tuple[list[dict], bool]: ...


class GraylogReader:
    def __init__(self, url, stream, token_path, environment, opener=None):
        self.url = url.rstrip("/")
        self.stream = stream
        self.token_path = Path(token_path)
        self.environment = environment
        self.opener = opener or urllib.request.urlopen

    def read(self, context, start):
        if not re.fullmatch(r"[a-f0-9]{24}", self.stream):
            raise ValueError("invalid Graylog stream")
        token = self.token_path.read_text().strip()
        if not token:
            raise ValueError("empty Graylog token")
        authorization = base64.b64encode(f"{token}:token".encode()).decode()
        fields = (
            "timestamp,message,event_id,event_name,log_level,schema_version,service,environment,"
            "dag_id,dag_run_id,task_id,map_index,try_number,task_attempt_key,job_key,"
            "job_attempt_id,association_role,event_sequence"
        )
        key = task_key(context)
        params = {
            "query": f"task_attempt_key:{key} AND schema_version:{SCHEMA_VERSION}",
            "filter": f"streams:{self.stream}",
            "from": start.isoformat(timespec="milliseconds"),
            "to": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(timespec="milliseconds"),
            "limit": PAGE_SIZE,
            "fields": fields,
            "sort": "timestamp:asc",
            "decorate": "false",
        }
        events = {}
        deadline = time.monotonic() + 3
        offset = 0
        while offset < MAX_ROWS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Graylog query budget exhausted")
            params["offset"] = offset
            request = urllib.request.Request(
                f"{self.url}/api/search/universal/absolute?{urllib.parse.urlencode(params)}",
                headers={"Authorization": f"Basic {authorization}", "Accept": "application/json"},
            )
            with self.opener(request, timeout=min(remaining, 1.5)) as response:
                data = response.read(512 * 1024 + 1)
            if len(data) > 512 * 1024:
                raise ValueError("oversized Graylog response")
            result = json.loads(data)
            # The built-in timestamp is mapped even before the first project event.
            # Only require the event ID sort when crossing a page boundary.
            if params["sort"] == "timestamp:asc" and result["total_results"] > PAGE_SIZE:
                params["sort"] = "event_id:asc"
                continue
            rows = result["messages"]
            for row in rows:
                event = row["message"]
                event_id = event.get("event_id", "")
                if (
                    HEX_ID.fullmatch(event_id)
                    and event.get("task_attempt_key") == key
                    and event.get("schema_version") == SCHEMA_VERSION
                    and event.get("environment") == self.environment
                    and event.get("service") == "benchmark-worker"
                    and all(event.get(name) == value for name, value in context.items())
                ):
                    events[event_id] = event
            if len(rows) < PAGE_SIZE or offset + len(rows) >= result["total_results"]:
                return list(events.values()), False
            offset += len(rows)
        return list(events.values()), True


class ThrottledReader:
    """Share short-lived results across browser sessions without storing their cursors."""

    def __init__(self, source: EventReader):
        self.source = source
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.pending = set()
        self.slots = threading.BoundedSemaphore(4)

    def read(self, context, start):
        key = task_key(context)
        with self.lock:
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < POLL_INTERVAL:
                if cached[1] is None:
                    raise OSError("Graylog temporarily unavailable")
                return cached[1]
            if key in self.pending or not self.slots.acquire(blocking=False):
                return cached[1] if cached and cached[1] is not None else ([], False)
            self.pending.add(key)
        result = None
        try:
            result = self.source.read(context, start)
            return result
        finally:
            with self.lock:
                self.pending.remove(key)
                self.slots.release()
                self.cache[key] = (time.monotonic(), result)
                self.cache.move_to_end(key)
                if len(self.cache) > 128:
                    self.cache.popitem(last=False)


def configured_reader():
    backend = os.environ.get("BENCH_TASK_LOG_BACKEND", "local")
    if backend == "local":
        return None
    if backend != "graylog":
        raise ValueError(f"Invalid BENCH_TASK_LOG_BACKEND: {backend}")
    return ThrottledReader(
        GraylogReader(
            os.environ.get("BENCH_GRAYLOG_URL", "http://graylog:9000"),
            os.environ.get("BENCH_GRAYLOG_STREAM", ""),
            os.environ.get("BENCH_GRAYLOG_TOKEN_FILE", "/run/secrets/graylog-reader"),
            os.environ.get("BENCH_LOG_ENVIRONMENT", "homelab"),
        )
    )


def associated_events(events):
    associations = [e for e in events if e["event_name"] == "seed.task_associated"]
    owners = {(e["job_key"], e["job_attempt_id"]) for e in associations if e["association_role"] == "origin"}
    sequences = {key: set() for key in owners}
    finals = {}
    visible = associations.copy()
    for event in events:
        key = event.get("job_key"), event.get("job_attempt_id")
        if key in owners and type(event.get("event_sequence")) is int:
            index = event["event_sequence"]
            if not 1 <= index <= MAX_ROWS:
                continue
            sequences[key].add(index)
            visible.append(event)
            if event["event_name"] == "seed.finished":
                finals[key] = index
    complete = bool(associations) and all(
        key in finals and len(sequences[key].intersection(range(1, finals[key] + 1))) == finals[key]
        for key in owners
    )
    return sorted(
        visible, key=lambda e: (e["timestamp"], e.get("event_sequence", 0), e["event_id"])
    ), complete
