"""Idempotent metric reconciliation with concurrent history reads."""

import math
from concurrent.futures import ThreadPoolExecutor

from .backends import client_concurrency
from .ports import RunRecord


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _coordinates(row):
    return (row[0], row[1]) if isinstance(row, tuple) else (row.step, row.value)


def _known_keys(client, run_id):
    if not callable(getattr(client, "get_run", None)):
        return None
    run = client.get_run(run_id)
    if isinstance(run, RunRecord):
        return set(run.metric_keys)
    metrics = getattr(getattr(run, "data", None), "metrics", None)
    return set(metrics) if isinstance(metrics, dict) else None


def _workers(client, requested):
    if requested:
        return requested
    concurrency = getattr(client, "concurrency", None)
    return concurrency if isinstance(concurrency, int) and concurrency > 0 else client_concurrency(client)


def missing_metric_points(client, run_id, points, known_keys=None, *, max_workers=None):
    """Return the points absent from the run, rejecting conflicting values (timestamps ignored).

    Only keys already present on the run need their history; those reads run
    concurrently on remote stores instead of one round-trip after another.
    """
    desired = {}
    for key, step, value in points:
        if value is not None and not _finite(value):
            raise ValueError(f"nonfinite metric {key}")
        by_step = desired.setdefault(key, {})
        if step in by_step and by_step[step] != value:
            raise ValueError(f"conflicting metric {key} at step {step}")
        by_step[step] = value
    if known_keys is None:
        known_keys = _known_keys(client, run_id)
    remote = [key for key in desired if known_keys is None or key in known_keys]

    def history(key):
        return key, [_coordinates(row) for row in client.get_metric_history(run_id, key)]

    workers = min(_workers(client, max_workers), len(remote)) or 1
    if workers > 1:
        with ThreadPoolExecutor(workers, thread_name_prefix="mlflow-history") as pool:
            histories = dict(pool.map(history, remote))
    else:
        histories = dict(history(key) for key in remote)
    missing = []
    for key, by_step in desired.items():
        existing = set()
        for step, value in histories.get(key, ()):
            if step in by_step:
                if not _finite(value) or value != by_step[step]:
                    raise ValueError(f"remote metric conflicts for {key} at step {step}")
                existing.add(step)
        missing.extend(
            (key, step, value)
            for step, value in by_step.items()
            if step not in existing and value is not None
        )
    return missing
