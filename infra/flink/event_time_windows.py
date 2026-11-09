"""Managed Flink (PyFlink, Table API): event-time tumbling windows over the Kinesis event stream.

Semantics mirror `continual_bench.application.cloud.streaming.WindowState`, the executable
specification tested locally:
- event time = payload `timestamp` (seconds) × 1000; watermark = max event time − out-of-orderness;
- windows close when the watermark passes window end + allowed lateness;
- late records (behind a closed window) are written to a separate side table, not merged;
- checkpointing (configured on the application) restores window state after restarts.
"""

import json
import os

from pyflink.table import EnvironmentSettings, TableEnvironment

PROPERTIES = "/etc/flink/application_properties.json"


def properties():
    if os.path.isfile(PROPERTIES):
        groups = json.load(open(PROPERTIES))
        return next(g["PropertyMap"] for g in groups if g["PropertyGroupId"] == "windows")
    return {"stream": "continual-bench-events", "region": "us-east-1", "window_ms": "3600000",
            "out_of_orderness_ms": "60000", "allowed_lateness_ms": "300000", "output_path": "/tmp/windows/"}


def main():
    p = properties()
    env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
    env.get_config().set("table.exec.emit.allow-lateness", f"{int(p['allowed_lateness_ms'])} ms")
    env.get_config().set("table.exec.emit.late-fire.enabled", "true")
    env.get_config().set("table.exec.emit.late-fire.delay", "0 ms")
    out_of_orderness = int(p["out_of_orderness_ms"]) // 1000
    env.execute_sql(f"""
        CREATE TABLE events (
            event_id STRING, seq BIGINT, `timestamp` BIGINT, y BOOLEAN,
            event_time AS TO_TIMESTAMP_LTZ(`timestamp` * 1000, 3),
            WATERMARK FOR event_time AS event_time - INTERVAL '{out_of_orderness}' SECOND
        ) WITH (
            'connector' = 'kinesis', 'stream' = '{p['stream']}', 'aws.region' = '{p['region']}',
            'scan.stream.initpos' = 'TRIM_HORIZON', 'format' = 'json', 'json.ignore-parse-errors' = 'false'
        )""")
    window_seconds = int(p["window_ms"]) // 1000
    env.execute_sql(f"""
        CREATE TABLE windows (
            window_start TIMESTAMP(3), window_end TIMESTAMP(3), events BIGINT, positives BIGINT,
            first_seq BIGINT, last_seq BIGINT, distinct_events BIGINT
        ) WITH ('connector' = 'filesystem', 'path' = '{p['output_path']}windows/', 'format' = 'json')""")
    env.execute_sql(f"""
        CREATE TABLE late_events (event_id STRING, seq BIGINT, event_time TIMESTAMP(3))
        WITH ('connector' = 'filesystem', 'path' = '{p['output_path']}late/', 'format' = 'json')""")
    statements = env.create_statement_set()
    statements.add_insert_sql(f"""
        INSERT INTO windows
        SELECT window_start, window_end, COUNT(*), SUM(CASE WHEN y THEN 1 ELSE 0 END), MIN(seq), MAX(seq),
               COUNT(DISTINCT event_id)
        FROM TABLE(TUMBLE(TABLE events, DESCRIPTOR(event_time), INTERVAL '{window_seconds}' SECOND))
        GROUP BY window_start, window_end""")
    # Records whose event time is already behind the watermark plus lateness go to the side table.
    statements.add_insert_sql(f"""
        INSERT INTO late_events
        SELECT event_id, seq, CAST(event_time AS TIMESTAMP(3)) FROM events
        WHERE event_time < CURRENT_WATERMARK(event_time) - INTERVAL '{int(p['allowed_lateness_ms']) // 1000}' SECOND""")
    statements.execute()


if __name__ == "__main__":
    main()
