"""Transport-independent ordering contract for replayed event streams.

Events carry a dense `seq` and a stable `event_id`. Redelivered records (seq below the cursor
with the expected ID) are counted and skipped; a gap, a reordering or a changed payload stops
the stream. A cursor is the next expected seq plus the transport position that produced it.
"""

from dataclasses import dataclass, field


class StreamOrderError(ValueError):
    pass


@dataclass
class OrderedStream:
    next_seq: int = 0
    duplicates: int = 0
    seen_ids: dict = field(default_factory=dict)  # seq → event_id for the bounded redelivery window
    window: int = 10_000

    def accept(
        self, seq: int, event_id: str, payload: str | None = None, expected: str | None = None
    ) -> bool:
        """True when the record is the next event; False for a verified duplicate."""
        if seq < self.next_seq:
            known = self.seen_ids.get(seq)
            if known is not None and known != event_id:
                raise StreamOrderError(f"duplicate seq {seq} carries a different event_id")
            self.duplicates += 1
            return False
        if seq > self.next_seq:
            raise StreamOrderError(f"gap: expected seq {self.next_seq}, received {seq}")
        if expected is not None and payload != expected:
            raise StreamOrderError(f"payload at seq {seq} differs from the immutable source")
        self.seen_ids[seq] = event_id
        if len(self.seen_ids) > self.window:
            self.seen_ids.pop(next(iter(self.seen_ids)))
        self.next_seq += 1
        return True
