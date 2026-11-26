"""Run in the project's Airflow image to exercise its real logging classes."""

import io
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

try:
    import airflow
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Airflow image required") from error

from continual_bench.observability.airflow_logs import (
    FileTaskHandler,
    GraylogTaskLogHandler,
    StructuredLogMessage,
)
from continual_bench.observability.events import task_key
from continual_bench.observability.graylog import GraylogReader

CONTEXT = {
    "dag_id": "benchmark_campaign",
    "dag_run_id": "manual__graylog-tests",
    "task_id": "run_seed",
    "map_index": 1,
    "try_number": 2,
}
STAMP = "2026-11-26T16:00:00.000Z"


def event(event_id, name, **fields):
    return {
        **CONTEXT,
        "schema_version": 1,
        "service": "benchmark-worker",
        "environment": "homelab",
        "task_attempt_key": task_key(CONTEXT),
        "event_id": f"{event_id:032x}",
        "event_name": name,
        "message": name,
        "timestamp": STAMP,
        "log_level": "INFO",
        "job_key": "job",
        "job_attempt_id": "attempt",
        **fields,
    }


class WorkerLogTests(unittest.TestCase):
    def test_paginated_reader_handles_identical_timestamps_and_validates_scope(self):
        events = [{"message": event(i, "seed.started", event_sequence=i)} for i in range(38)]
        events[35]["message"]["map_index"] = 0
        events[36] = events[2]
        calls = []

        def request(req, timeout):
            params = parse_qs(urlsplit(req.full_url).query)
            self.assertEqual(params["filter"], ["streams:" + "a" * 24])
            self.assertLessEqual(timeout, 1.5)
            offset = int(params["offset"][0])
            calls.append((params["sort"][0], offset))
            return io.BytesIO(
                json.dumps(
                    {
                        "total_results": len(events),
                        "messages": events[offset : offset + int(params["limit"][0])],
                    }
                ).encode()
            )

        with tempfile.TemporaryDirectory() as folder:
            token = Path(folder) / "token"
            token.write_text("dummy-test-token")
            reader = GraylogReader("http://graylog", "a" * 24, token, "homelab", opener=request)
            rows, truncated = reader.read(CONTEXT, datetime.now(UTC))
        self.assertEqual(calls, [("timestamp:asc", 0), ("event_id:asc", 0), ("event_id:asc", 32)])
        self.assertEqual(len(rows), 36)
        self.assertFalse(truncated)

    def test_native_handler_keeps_logs_and_cursor_across_outage_and_late_indexing(self):
        association = event(1, "seed.task_associated", association_role="origin")
        start = event(2, "seed.started", event_sequence=1)
        missing = event(3, "seed.matrix_started", event_sequence=2)
        final = event(4, "seed.finished", event_sequence=3)
        outsider = event(5, "seed.finished", event_sequence=2, job_attempt_id="different-attempt")
        observed = event(6, "seed.task_associated", association_role="observer", job_attempt_id="other")
        reader = Mock()
        reader.read.side_effect = [
            ([association, start], False),
            OSError("unavailable"),
            ([association, start, final, outsider, observed], False),
            ([association, start, missing, final, outsider, observed], False),
        ]
        ti = SimpleNamespace(
            dag_id=CONTEXT["dag_id"],
            run_id=CONTEXT["dag_run_id"],
            task_id=CONTEXT["task_id"],
            map_index=CONTEXT["map_index"],
            try_number=CONTEXT["try_number"],
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        with patch.dict(os.environ, {"BENCH_TASK_LOG_BACKEND": "local"}):
            handler = GraylogTaskLogHandler(tempfile.gettempdir())
        handler.reader = reader
        native = StructuredLogMessage(event="native log")
        cursor = {}
        seen = []

        def native_read(_ti, _try, metadata):
            lines = [native] if not metadata else []
            return iter(lines), {"end_of_log": True, "log_pos": 1}

        with patch.object(FileTaskHandler, "read", side_effect=native_read):
            for attempt in range(4):
                logs, cursor = handler.read(ti, 2, json.loads(json.dumps(cursor)))
                seen.extend(log.event for log in logs)
                self.assertEqual(cursor["log_pos"], 1)
                self.assertEqual(cursor["end_of_log"], attempt == 3)
        self.assertEqual(seen.count("native log"), 1)
        self.assertEqual(seen.count("[worker] seed.started (job=job)"), 1)
        self.assertEqual(seen.count("[worker] seed.finished (job=job)"), 1)
        self.assertEqual(seen.count("[worker] seed.matrix_started (job=job)"), 1)
        self.assertTrue(any("temporarily unavailable" in message for message in seen))
        self.assertTrue(any("reading resumed" in message for message in seen))

    def test_airflow_version(self):
        self.assertEqual(airflow.__version__, "3.3.2")


if __name__ == "__main__":
    unittest.main()
