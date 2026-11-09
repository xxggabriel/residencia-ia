"""Kinesis Data Streams transport for the streaming (event-prequential) mode.

Implements the same `contract()`/`events(manifest, source, deadline)` port as `KafkaTransport`.
The producer writes the immutable JSONL manifest to one shard with `event_id` as explicit hash
key material and `seq` inside the payload. The consumer restores order through a bounded reorder
buffer (PutRecords retries may reorder a failed record), drops verified redeliveries, refuses
gaps and payload changes, and checkpoints a cursor (next seq + shard sequence number) through an
injected object store so a restarted attempt resumes where the previous one stopped.
"""

import heapq
import json
import time
from pathlib import Path

from continual_bench.core.contracts import Event, event_json
from continual_bench.infrastructure.stream_order import OrderedStream, StreamOrderError

from .session import client as make_client

BATCH = 500
REORDER_WINDOW = 1000


class KinesisTransport:
    def __init__(
        self,
        stream: str,
        *,
        cursor_store=None,
        cursor_key: str | None = None,
        client=None,
        poll_seconds=0.2,
        stall_seconds=60,
    ):
        self.stream = stream
        self.client = client or make_client("kinesis")
        self.cursor_store, self.cursor_key = cursor_store, cursor_key
        self.poll_seconds, self.stall_seconds = poll_seconds, stall_seconds
        self.statistics = {"produced": 0, "duplicates": 0, "reordered": 0, "resumed_from": None}

    def contract(self):
        return {
            "name": "kinesis",
            "version": 1,
            "stream": self.stream,
            "ordering": "seq-in-payload",
            "reorder_window": REORDER_WINDOW,
        }

    # Cursor -----------------------------------------------------------------------------------
    def load_cursor(self):
        if not self.cursor_store or not self.cursor_key or self.cursor_store.head(self.cursor_key) is None:
            return None
        return json.loads(self.cursor_store.get_bytes(self.cursor_key))

    def save_cursor(self, cursor):
        if self.cursor_store and self.cursor_key:
            self.cursor_store.put_bytes(self.cursor_key, json.dumps(cursor, sort_keys=True).encode())

    # Producer ---------------------------------------------------------------------------------
    def produce(self, events, run_key, *, start_seq=0, deadline=None):
        began, pending = time.monotonic(), []

        def flush():
            records = pending[:]
            for _ in range(8):
                response = self.client.put_records(StreamName=self.stream, Records=[r for _, r in records])
                failed = [
                    item
                    for item, result in zip(records, response["Records"], strict=True)
                    if result.get("ErrorCode")
                ]
                self.statistics["produced"] += len(records) - len(failed)
                if not failed:
                    pending.clear()
                    return
                records = failed
                time.sleep(0.2)
            raise RuntimeError(f"Kinesis rejected {len(records)} records after retries")

        for event in events:
            if event.seq < start_seq:
                continue
            if deadline is not None and time.monotonic() - began > deadline:
                raise TimeoutError("Kinesis publishing deadline exceeded")
            # One partition key per run keeps the whole replay on one shard.
            pending.append((event.seq, {"Data": event_json(event).encode(), "PartitionKey": run_key}))
            if len(pending) == BATCH:
                flush()
        if pending:
            flush()

    # Consumer ---------------------------------------------------------------------------------
    def _shard(self):
        shards = self.client.list_shards(StreamName=self.stream)["Shards"]
        open_shards = [s for s in shards if "EndingSequenceNumber" not in s["SequenceNumberRange"]]
        if len(open_shards) != 1:
            raise ValueError("ordered replay requires exactly one open shard")
        return open_shards[0]["ShardId"]

    def records(self, shard, cursor):
        if cursor and cursor.get("sequence_number"):
            iterator = self.client.get_shard_iterator(
                StreamName=self.stream,
                ShardId=shard,
                ShardIteratorType="AFTER_SEQUENCE_NUMBER",
                StartingSequenceNumber=cursor["sequence_number"],
            )["ShardIterator"]
        else:
            iterator = self.client.get_shard_iterator(
                StreamName=self.stream, ShardId=shard, ShardIteratorType="TRIM_HORIZON"
            )["ShardIterator"]
        last = time.monotonic()
        while iterator:
            response = self.client.get_records(ShardIterator=iterator, Limit=1000)
            iterator = response.get("NextShardIterator")
            if response["Records"]:
                last = time.monotonic()
                yield from response["Records"]
            elif time.monotonic() - last > self.stall_seconds:
                raise TimeoutError("Kinesis consumption stalled")
            else:
                time.sleep(self.poll_seconds)

    def events(self, manifest, source, deadline):
        run_key, total = manifest["run_key"], manifest["samples"]
        cursor = self.load_cursor()
        if cursor and cursor.get("run_key") != run_key:
            raise ValueError("Kinesis cursor belongs to another run")
        start = cursor["next_seq"] if cursor else 0
        self.statistics["resumed_from"] = start
        self.produce(
            source, run_key, start_seq=cursor.get("produced", start) if cursor else 0, deadline=deadline
        )
        order = OrderedStream(next_seq=start)
        expected = Path(manifest["path"]).open()
        for _ in range(start):
            next(expected)
        heap, began, saved = [], time.monotonic(), start
        shard = self._shard()
        try:
            for record in self.records(shard, cursor):
                if time.monotonic() - began > deadline:
                    raise TimeoutError("Kinesis replay deadline exceeded")
                event = Event(**json.loads(record["Data"]))
                if event.seq < order.next_seq:
                    order.accept(event.seq, event.event_id)
                    continue
                heapq.heappush(heap, (event.seq, record["SequenceNumber"], event))
                if heap[0][0] != order.next_seq:
                    self.statistics["reordered"] += 1
                    if len(heap) > REORDER_WINDOW:
                        raise StreamOrderError(f"gap: expected seq {order.next_seq}, buffered {heap[0][0]}")
                    continue
                while heap and heap[0][0] <= order.next_seq:
                    seq, position, ready = heapq.heappop(heap)
                    payload = event_json(ready)
                    if seq < order.next_seq:
                        order.accept(seq, ready.event_id)
                        continue
                    order.accept(seq, ready.event_id, payload, next(expected).strip())
                    self.statistics["duplicates"] = order.duplicates
                    yield ready
                    # Only an empty reorder buffer makes the shard position a safe resume point.
                    if not heap and (order.next_seq - saved >= 1000 or order.next_seq == total):
                        self.save_cursor(
                            {
                                "run_key": run_key,
                                "next_seq": order.next_seq,
                                "sequence_number": position,
                                "shard": shard,
                                "produced": total,
                            }
                        )
                        saved = order.next_seq
                if order.next_seq >= total:
                    return
        finally:
            expected.close()
