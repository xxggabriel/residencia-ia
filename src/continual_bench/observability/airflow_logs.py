"""Airflow 3.3 task log handler. Keep Airflow's file handler for native logs."""

import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from itertools import chain

from airflow.config_templates.airflow_local_settings import DEFAULT_LOGGING_CONFIG
from airflow.utils.log.file_task_handler import FileTaskHandler, StructuredLogMessage

from continual_bench.observability.events import task_key
from continual_bench.observability.graylog import MAX_ROWS, associated_events, configured_reader

TAIL_SECONDS = 300


class GraylogTaskLogHandler(FileTaskHandler):
    def __init__(self, base_log_folder):
        super().__init__(base_log_folder)
        self.reader = configured_reader()

    def read(self, task_instance, try_number=None, metadata=None):
        native, cursor = super().read(task_instance, try_number, metadata)
        ti = task_instance
        if (
            self.reader is None
            or (ti.dag_id, ti.task_id) != ("benchmark_campaign", "run_seed")
            or (try_number if try_number is not None else ti.try_number) < 1
        ):
            return native, cursor

        context = {
            "dag_id": ti.dag_id,
            "dag_run_id": ti.run_id,
            "task_id": ti.task_id,
            "map_index": ti.map_index,
            "try_number": try_number if try_number is not None else ti.try_number,
        }
        prior = (metadata or {}).get("worker_events", {})
        if prior.get("key") != task_key(context):
            prior = {}
        seen = set(prior.get("seen", []))
        local_done = cursor.get("end_of_log", False)
        now = datetime.now(UTC)
        since = (ti.start_date or now - timedelta(hours=1)) - timedelta(minutes=5)
        until = (
            prior.get("until")
            or max(now.timestamp(), ti.end_date.timestamp() if ti.end_date else 0) + TAIL_SECONDS
        )
        messages = []
        problem = None
        complete = False
        try:
            events, truncated = self.reader.read(context, since)
            visible, complete = associated_events(events)
            for event in visible:
                if event["event_id"] in seen:
                    continue
                if len(seen) >= MAX_ROWS:
                    truncated = True
                    break
                seen.add(event["event_id"])
                messages.append(
                    StructuredLogMessage(
                        timestamp=datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")),
                        level=event["log_level"].lower(),
                        event=f"[worker] {event['message']} (job={event['job_key']})",
                    )
                )
            if truncated:
                problem = "Graylog worker logs reached the per-task read limit."
        except (OSError, ValueError, KeyError, TypeError):
            problem = "Graylog worker logs temporarily unavailable. Local Airflow logs remain available."
        if problem and problem != prior.get("problem"):
            messages.append(StructuredLogMessage(level="warning", event=problem))
        elif not problem and prior.get("problem"):
            messages.append(StructuredLogMessage(level="info", event="Graylog worker log reading resumed."))

        if local_done and not complete and now.timestamp() >= until:
            messages.append(
                StructuredLogMessage(
                    level="warning", event="Graylog worker log tail timed out. Reload to query again."
                )
            )
        cursor.update(
            end_of_log=bool((metadata or {}).get("download_logs"))
            or (local_done and (complete and not problem or now.timestamp() >= until)),
            worker_events={
                "key": task_key(context),
                "seen": sorted(seen),
                "problem": problem,
                "until": until if local_done else None,
            },
        )
        # Airflow's terminal NDJSON loop reuses the supplied metadata. Advance it
        # in place and let the loop's idle timeout work when both sources are empty.
        if metadata is not None:
            metadata.update(cursor)
        native = iter(native)
        first = next(native, None)
        if first is None:
            return iter(messages) if messages else (), cursor
        return chain((first,), native, messages), cursor


LOGGING_CONFIG = deepcopy(DEFAULT_LOGGING_CONFIG)
if os.environ.get("BENCH_TASK_LOG_BACKEND", "local") != "local":
    LOGGING_CONFIG["handlers"]["task"]["class"] = (
        "continual_bench.observability.airflow_logs.GraylogTaskLogHandler"
    )
