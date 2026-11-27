"""Drain persisted JSON lines to the internal Graylog GELF HTTP input."""

import json
import logging
import os
import socket
import sqlite3
import time
import urllib.request
from contextlib import closing
from datetime import datetime

from continual_bench.observability.events import open_spool

LEVELS = {"DEBUG": 7, "INFO": 6, "WARNING": 4, "ERROR": 3, "CRITICAL": 2}


def to_gelf(event):
    return {
        "version": "1.1",
        "host": event["worker_host"],
        "short_message": event["message"],
        "timestamp": datetime.fromisoformat(event["timestamp"]).timestamp(),
        "level": LEVELS.get(event["level"], 6),
        "_log_level": event["level"],
        **{
            f"_{name}": value
            for name, value in event.items()
            if name not in {"message", "level", "timestamp"} and value is not None
        },
    }


def forward_batch(path, url):
    with closing(open_spool(path)) as db:
        batch = db.execute("SELECT seq, line FROM events ORDER BY seq LIMIT 100").fetchall()
        if not batch:
            return False
        body = b"\n".join(json.dumps(to_gelf(json.loads(line))).encode() for _, line in batch)
        request = urllib.request.Request(url, body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.status != 202:
                raise OSError(f"unexpected GELF status {response.status}")
        with db:
            db.execute("DELETE FROM events WHERE seq <= ?", (batch[-1][0],))
    return True


def main():
    path = os.environ["BENCH_EVENT_SPOOL"]
    url = os.environ["BENCH_GELF_URL"]
    logging.basicConfig(level=logging.INFO)
    wait = 1
    while True:
        try:
            if wait > 1:
                logging.info("Retrying GELF delivery from %s", socket.gethostname())
            wait = 1 if forward_batch(path, url) else 0.5
        except (OSError, sqlite3.Error) as error:
            wait = min(max(wait * 2, 1), 15)
            logging.warning("GELF delivery pending (%s). Retrying in %ss", type(error).__name__, wait)
        time.sleep(wait)


if __name__ == "__main__":
    main()
