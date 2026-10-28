"""Explicit real-source admission example; no dataset is bundled or downloaded."""

import csv
import hashlib
from pathlib import Path

from continual_bench.domain.contracts import Event


class TemporalCSV:
    def __init__(
        self,
        *,
        samples,
        seed,
        label_delay,
        epoch,
        path,
        sha256,
        license_id,
        source_url,
        timestamp_column,
        label_column,
        feature_columns,
    ):
        self.path = Path(path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.digest != sha256:
            raise ValueError("CSV checksum mismatch")
        if not license_id or not source_url:
            raise ValueError("explicit provenance/license required")
        if timestamp_column in feature_columns or label_column in feature_columns:
            raise ValueError("time and target cannot be included in feature_columns")
        self.samples, self.delay = samples, label_delay
        self.timestamp_column, self.label_column = timestamp_column, label_column
        self.features = feature_columns
        self.license_id, self.source_url = license_id, source_url

    def events(self):
        previous = -1
        count = 0
        with self.path.open(newline="") as f:
            for seq, row in enumerate(csv.DictReader(f)):
                if seq == self.samples:
                    break
                timestamp = int(row[self.timestamp_column])
                if timestamp <= previous:
                    raise ValueError("CSV must already be strictly ordered; no silent sorting")
                previous = timestamp
                label = row[self.label_column]
                if label not in ("0", "1"):
                    raise ValueError("explicit binary 0/1 label required")
                x = {}
                for key in self.features:
                    try:
                        x[key] = float(row[key])
                    except ValueError:
                        x[key] = row[key]
                count += 1
                identity = hashlib.sha256(f"{self.digest}:{seq}".encode()).hexdigest()
                yield Event(identity, seq, timestamp, timestamp + self.delay, x, label == "1", None)
        if count != self.samples:
            raise ValueError("CSV has fewer rows than declared samples")

    def manifest(self):
        return {
            "name": "temporal_csv",
            "version": self.digest,
            "samples": self.samples,
            "license": self.license_id,
            "source": self.source_url,
            "changepoints": None,
            "timestamp_kind": "declared_CSV_integer_UTC_seconds",
            "label_delay_seconds": self.delay,
        }


def register(registry):
    registry.register("source", "temporal_csv", TemporalCSV)
