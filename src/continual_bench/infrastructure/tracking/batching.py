"""The single implementation of MLflow ``runs/log-batch`` request limits."""

from collections.abc import Iterator, Sequence
from typing import TypeVar

MAX_METRICS = 1000
MAX_PARAMS = 100
MAX_TAGS = 100
MAX_ENTITIES = 1000

M = TypeVar("M")
P = TypeVar("P")
T = TypeVar("T")


def batch_payloads(
    metrics: Sequence[M] = (),
    params: Sequence[P] = (),
    tags: Sequence[T] = (),
) -> Iterator[tuple[list[M], list[P], list[T]]]:
    """Yield the fewest requests that respect every MLflow limit.

    Params and tags ride with the first metric chunks instead of separate
    requests: a report with 300 metrics, 20 params and 10 tags is one call.
    """
    metrics, params, tags = list(metrics), list(params), list(tags)
    m = p = t = 0
    while m < len(metrics) or p < len(params) or t < len(tags):
        chunk_params = params[p : p + MAX_PARAMS]
        chunk_tags = tags[t : t + MAX_TAGS]
        room = min(MAX_METRICS, MAX_ENTITIES - len(chunk_params) - len(chunk_tags))
        chunk_metrics = metrics[m : m + room]
        p += len(chunk_params)
        t += len(chunk_tags)
        m += len(chunk_metrics)
        yield chunk_metrics, chunk_params, chunk_tags
