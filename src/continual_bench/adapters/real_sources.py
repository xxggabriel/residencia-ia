"""Pinned OpenML streams admitted before benchmark execution; no implicit downloads."""

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

from continual_bench.domain.contracts import Event

REAL_SOURCES = {
    "electricity": {
        "openml_id": 151,
        "version": 1,
        "url": "https://openml.org/data/v1/download/2419/electricity.arff",
        "sha256": "2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2",
        "md5": "8ca97867d960ae029ae3a9ac2c923d34",
        "rows": 45312,
        "bytes": 3092013,
        "target": "class",
        "positive": "UP",
        "labels": ["UP", "DOWN"],
        "columns": ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand",
                    "transfer", "class"],
        "nominal": ["day", "class"],
        "license": "Public",
        "normalization": ("Published header declares Normalize and ReplaceMissingValues; "
                          "fit scope is unspecified"),
    },
    "airlines": {
        "openml_id": 1169,
        "version": 1,
        "url": "https://openml.org/data/v1/download/66526/airlines.arff",
        "sha256": "c7683a90b40c2cabb1ddc950413da8efa231c058327b8c95511c40af13516c14",
        "md5": "2c1189a18602a2cfdfe952b7d262562b",
        "rows": 539383,
        "bytes": 19168157,
        "target": "Delay",
        "positive": "1",
        "labels": ["0", "1"],
        "columns": ["Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek", "Time",
                    "Length", "Delay"],
        "nominal": ["Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Delay"],
        "license": "Public",
        "normalization": "No additional normalization in this adapter",
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(raw):
    return [value.strip().strip("'\"") for value in next(csv.reader([raw], skipinitialspace=True))]


def read_schema(stream):
    """Consume the dense ARFF header and leave the stream at the first data row."""
    schema = []
    relation_seen = False
    for line in stream:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        lower = line.lower()
        if lower.startswith("@relation "):
            relation_seen = True
        elif lower.startswith("@attribute "):
            match = re.fullmatch(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)", line, re.I)
            if not match:
                raise ValueError("Invalid ARFF attribute declaration")
            name = next(value for value in match.groups()[:3] if value is not None)
            dtype = match.group(4).strip()
            if any(field["name"] == name for field in schema):
                raise ValueError("Duplicate ARFF attribute")
            if dtype.lower() in {"numeric", "real", "integer"}:
                schema.append({"name": name, "type": "numeric"})
            elif dtype.startswith("{") and dtype.endswith("}"):
                values = _values(dtype[1:-1])
                if not values or "?" in values or len(set(values)) != len(values):
                    raise ValueError("Invalid ARFF nominal declaration")
                schema.append({"name": name, "type": "nominal", "values": values})
            else:
                raise ValueError("Unsupported ARFF attribute type")
        elif lower == "@data":
            if not relation_seen or not schema:
                raise ValueError("ARFF relation and attributes are required")
            return schema
        else:
            raise ValueError("Unexpected ARFF header directive")
    raise ValueError("ARFF data section missing")


def _iter_rows(stream, schema):
    for line in stream:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("{"):
            raise ValueError("Sparse ARFF is not admitted")
        values = _values(line)
        if len(values) != len(schema):
            raise ValueError("ARFF row does not match schema")
        row = {}
        for field, value in zip(schema, values):
            if value in {"", "?"}:
                raise ValueError("Missing ARFF values are not admitted")
            if field["type"] == "numeric":
                try:
                    parsed = float(value)
                except ValueError:
                    raise ValueError("Invalid numeric ARFF value") from None
                if not math.isfinite(parsed):
                    raise ValueError("Non-finite numeric ARFF value")
                row[field["name"]] = parsed
            else:
                if value not in field["values"]:
                    raise ValueError("ARFF nominal value outside declared categories")
                row[field["name"]] = value
        yield row


def iter_arff(path):
    """Read dense ARFF in published order, preserving nominal strings."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        schema = read_schema(stream)
        yield from _iter_rows(stream, schema)


def _validate_schema(schema, specification):
    if [field["name"] for field in schema] != specification["columns"]:
        raise ValueError("Pinned source schema columns changed")
    if [field["name"] for field in schema if field["type"] == "nominal"] != specification["nominal"]:
        raise ValueError("Pinned source schema types changed")
    target = next(field for field in schema if field["name"] == specification["target"])
    if set(target.get("values", [])) != set(specification["labels"]):
        raise ValueError("Pinned source target categories changed")


def validate_source(path, name):
    """Full admission check, intended for the explicit preparation stage."""
    spec = REAL_SOURCES[name]
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"Pinned {name} SHA-256 mismatch")
    with Path(path).open(encoding="utf-8", newline="") as stream:
        schema = read_schema(stream)
        _validate_schema(schema, spec)
        rows = sum(1 for _ in _iter_rows(stream, schema))
    if rows != spec["rows"]:
        raise ValueError("Pinned source row count changed")
    return {"name": name, "raw_sha256": spec["sha256"], "raw_rows": rows, "schema": schema}


class RealSource:
    """A deterministic prefix, with an explicitly simulated ordinal replay clock."""

    def __init__(self, name, samples, seed, label_delay, epoch, *, directory=None):
        self.name, self.samples = name, samples
        self.seed, self.delay, self.epoch = seed, label_delay, epoch
        self.spec = REAL_SOURCES[name]
        if not 0 < samples <= self.spec["rows"]:
            raise ValueError("Requested prefix exceeds pinned source rows")
        self.path = Path(directory or os.environ.get("BENCH_DATASETS", "/state/datasets")) / f"{name}.arff"
        if not self.path.is_file():
            raise ValueError(f"Pinned {name} source is absent; run scripts/prepare_real_sources.py first")
        if file_sha256(self.path) != self.spec["sha256"]:
            raise ValueError(f"Pinned {name} SHA-256 mismatch")
        with self.path.open(encoding="utf-8", newline="") as stream:
            self.schema = read_schema(stream)
        _validate_schema(self.schema, self.spec)
        descriptor = {"adapter": "openml-dense-arff-v1", "raw_sha256": self.spec["sha256"],
                      "selection": [0, samples], "schema": self.schema,
                      "positive_label": self.spec["positive"], "epoch": epoch, "delay": label_delay}
        self.version = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()

    def events(self):
        count = 0
        for seq, row in enumerate(iter_arff(self.path)):
            if seq >= self.samples:
                break
            target = row.pop(self.spec["target"])
            timestamp = self.epoch + seq
            identity = hashlib.sha256(f"{self.spec['sha256']}:{seq}".encode()).hexdigest()
            count += 1
            yield Event(identity, seq, timestamp, timestamp + self.delay, row,
                        target == self.spec["positive"], None)
        if count != self.samples:
            raise ValueError("Source has fewer rows than the declared prefix")

    def manifest(self):
        return {
            "name": self.name, "version": self.version, "samples": self.samples,
            "source": self.spec["url"], "source_metadata":
                f"https://www.openml.org/api/v1/json/data/{self.spec['openml_id']}",
            "openml_id": self.spec["openml_id"], "openml_version": self.spec["version"],
            "license": self.spec["license"], "license_note": "Exact source catalog label; not an SPDX grant",
            "raw_sha256": self.spec["sha256"], "raw_rows": self.spec["rows"],
            "raw_path": str(self.path), "selection": {"kind": "published_prefix", "range": [0, self.samples]},
            "schema": self.schema, "target": self.spec["target"], "positive_label": self.spec["positive"],
            "seed_ignored": True, "requested_seed": self.seed, "changepoints": None,
            "timestamp_kind": "simulated_epoch_plus_published_row_index_seconds",
            "real_event_timestamps_known": False, "real_label_availability_known": False,
            "label_delay_seconds": self.delay, "normalization": self.spec["normalization"],
            "order": "Published file order, unchanged; not an assertion of verified calendar chronology",
            "limitations": ["No known ground-truth drift points", "Replay clock is not real event time",
                            "Label delay is simulated", "Source preprocessing cannot be undone"],
        }
