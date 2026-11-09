"""Reference semantics of the Flink job: event-time tumbling windows, watermarks and lateness.

The managed Flink application (`infra/flink/event_time_windows.py`) implements the same
rules; this pure version is the executable specification its outputs are compared against.
A record is *late* when its window already closed (watermark ≥ window end + allowed lateness);
late records go to a side output instead of silently changing a published window.
"""

from dataclasses import dataclass, field


@dataclass
class WindowState:
    size_ms: int
    out_of_orderness_ms: int
    allowed_lateness_ms: int
    watermark: int = -(2**63)
    open: dict = field(default_factory=dict)  # window start → {"count", "errors", "label_counts"}
    closed: list = field(default_factory=list)
    late: list = field(default_factory=list)

    def snapshot(self) -> dict:
        """Checkpointable state (what Flink keeps in its state backend)."""
        return {
            "watermark": self.watermark,
            "open": {str(k): v for k, v in self.open.items()},
            "closed": list(self.closed),
            "late": list(self.late),
        }

    @classmethod
    def restore(cls, size_ms, out_of_orderness_ms, allowed_lateness_ms, state: dict) -> "WindowState":
        window = cls(size_ms, out_of_orderness_ms, allowed_lateness_ms, state["watermark"])
        window.open = {int(k): v for k, v in state["open"].items()}
        window.closed, window.late = list(state["closed"]), list(state["late"])
        return window

    def add(self, event_time_ms: int, label: int, error: int = 0) -> None:
        start = event_time_ms - event_time_ms % self.size_ms
        if self.watermark >= start + self.size_ms + self.allowed_lateness_ms:
            self.late.append({"event_time_ms": event_time_ms, "window_start": start})
        else:
            bucket = self.open.setdefault(start, {"count": 0, "errors": 0, "label_counts": {}})
            bucket["count"] += 1
            bucket["errors"] += error
            bucket["label_counts"][str(label)] = bucket["label_counts"].get(str(label), 0) + 1
        self.watermark = max(self.watermark, event_time_ms - self.out_of_orderness_ms)
        self._fire()

    def _fire(self):
        for start in sorted(self.open):
            if self.watermark >= start + self.size_ms + self.allowed_lateness_ms:
                self.closed.append(
                    {"window_start": start, "window_end": start + self.size_ms, **self.open.pop(start)}
                )

    def finish(self):
        self.watermark = 2**63 - 1
        self._fire()
        return self.closed
