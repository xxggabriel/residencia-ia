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
    return {
        "stream_arn": os.environ["BENCH_KINESIS_STREAM_ARN"],
        "region": "us-east-1",
        "window_ms": "3600000",
        "out_of_orderness_ms": "60000",
        "allowed_lateness_ms": "300000",
        "output_path": "/tmp/windows/",
    }


def main():
    p = properties()
    env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
    env.get_config().set("table.local-time-zone", "UTC")
    # The append-only sink receives one aggregate when the allowed-lateness window closes.
    watermark_delay = (int(p["out_of_orderness_ms"]) + int(p["allowed_lateness_ms"])) / 1000
    env.execute_sql(f"""
        CREATE TABLE events (
            event_id STRING, seq BIGINT, `timestamp` BIGINT, y BOOLEAN,
            event_time AS TO_TIMESTAMP_LTZ(`timestamp` * 1000, 3),
            WATERMARK FOR event_time AS event_time - INTERVAL '{watermark_delay:.3f}' SECOND
        ) WITH (
            'connector' = 'kinesis', 'stream.arn' = '{p['stream_arn']}', 'aws.region' = '{p['region']}',
            'source.init.position' = 'TRIM_HORIZON', 'format' = 'json', 'json.ignore-parse-errors' = 'false'
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
    env.execute_sql("""
        CREATE TEMPORARY VIEW observed_events AS
        SELECT *, CAST(CURRENT_WATERMARK(event_time) AS TIMESTAMP(3)) AS seen_watermark FROM events
    """)
    env.execute_sql(f"""
        CREATE TEMPORARY VIEW assigned_events AS
        SELECT * FROM TABLE(TUMBLE(TABLE observed_events, DESCRIPTOR(event_time),
                                  INTERVAL '{window_seconds}' SECOND))
    """)
    statements = env.create_statement_set()
    statements.add_insert_sql("""
        INSERT INTO windows
        SELECT window_start, window_end, COUNT(DISTINCT event_id),
               COUNT(DISTINCT CASE WHEN y THEN event_id END), MIN(seq), MAX(seq), COUNT(DISTINCT event_id)
        FROM assigned_events
        WHERE seen_watermark IS NULL OR window_end > seen_watermark
        GROUP BY window_start, window_end""")
    statements.add_insert_sql("""
        INSERT INTO late_events
        SELECT event_id, seq, CAST(event_time AS TIMESTAMP(3)) FROM assigned_events
        WHERE window_end <= seen_watermark""")
    statements.execute()


if __name__ == "__main__":
    main()
