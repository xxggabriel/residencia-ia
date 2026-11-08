"""Dense ARFF reader preserving published row order and nominal strings."""

import csv
import hashlib
import math
import re
from pathlib import Path


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
