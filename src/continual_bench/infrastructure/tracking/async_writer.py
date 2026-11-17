"""Send live metrics from a daemon thread so tracking never blocks training."""

import threading
from collections.abc import Callable


class AsyncMetricWriter:
    """Buffer points on the caller thread and send them every ``interval`` seconds.

    ``coalesce=True`` keeps only the newest point per key between sends (live
    progress snapshots); ``coalesce=False`` keeps every point (per-step series).
    Sends are serialized, so ``flush()`` returns only after every buffered point
    has been handed to ``send`` — callers flush before final writes or a finish.
    """

    def __init__(
        self,
        send: Callable[[list[dict]], object],
        interval: float,
        *,
        coalesce: bool = True,
        max_pending: int = 1000,
    ):
        self._send = send
        self._interval = interval
        self._coalesce = coalesce
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: dict | list = {} if coalesce else []
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(self, points: list[dict]) -> None:
        if not points:
            return
        with self._lock:
            if self._coalesce:
                self._pending.update((point["key"], point) for point in points)
            else:
                self._pending.extend(points)
            size = len(self._pending)
            if self._thread is None and not self._closed:
                self._thread = threading.Thread(target=self._run, name="mlflow-live-metrics", daemon=True)
                self._thread.start()
        if size >= self._max_pending:
            self._wake.set()

    def _take(self) -> list[dict]:
        with self._lock:
            batch = list(self._pending.values()) if self._coalesce else list(self._pending)
            self._pending.clear()
        return batch

    def _drain(self) -> None:
        with self._send_lock:
            if batch := self._take():
                self._send(batch)

    def _run(self) -> None:
        # The first snapshot goes out at once so the run shows activity immediately.
        while True:
            try:
                self._drain()
            except Exception:  # The sender records its own errors; never kill the thread.
                pass
            if self._closed:
                return
            self._wake.wait(self._interval)
            self._wake.clear()

    def flush(self) -> None:
        self._drain()

    def close(self, timeout: float | None = 5.0) -> None:
        self._closed = True
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        self._drain()
