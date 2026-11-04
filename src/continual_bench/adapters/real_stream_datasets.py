"""Explicit, bounded acquisition of the original INSECTS and UCI Gas streams."""

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from .real_sources import file_sha256
from .week09_sources import TaskSample

SPECS = {
    "insects": {
        "variant": "abrupt_balanced", "source_version": "River-0.26.1-public-file-mapping",
        "source": "https://sites.google.com/view/uspdsrepository",
        "url": ("https://drive.google.com/uc?export=download&"
                "id=1WQoIuuVgiuXfzv4kvao6XuLQG37V923O&confirm=t"),
        "filename": "abrupt_balanced.csv", "expected_bytes": 14_151_769,
        "max_bytes": 20 * 1024**2, "expected_rows": 52_848, "features": 33,
        "license": "CC-BY-4.0 (USP repository; attribution retained)",
        "license_reference": "https://sites.google.com/view/uspdsrepository",
        "license_note": "Repository warns that external-source datasets can have additional restrictions.",
    },
    "gas": {
        "variant": "uci-224-ten-batches-corrected-batch10",
        "source_version": "UCI224-batch10-corrected-2013-10-14",
        "source": "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
        "url": ("https://archive.ics.uci.edu/static/public/224/"
                "gas%2Bsensor%2Barray%2Bdrift%2Bdataset.zip"),
        "filename": "gas-sensor-array-drift.zip", "expected_bytes": None,
        "max_bytes": 16 * 1024**2, "expected_rows": 13_910, "features": 128,
        "license": "CC-BY-4.0 (current UCI catalog)",
        "license_reference": "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
        "license_note": ("Current UCI license says CC-BY-4.0; legacy descriptive text says research-only. "
                         "This Residencia IA acquisition is for research; no additional terms accepted."),
    },
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _insects_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for index, fields in enumerate(csv.reader(stream)):
            if len(fields) != 34:
                raise ValueError("INSECTS abrupt_balanced requires 33 features and one target per row")
            try:
                x = {f"f{i + 1}": float(value) for i, value in enumerate(fields[:-1])}
            except ValueError:
                raise ValueError("INSECTS numeric feature schema mismatch") from None
            label = fields[-1].strip()
            if any(not math.isfinite(v) for v in x.values()) or not label:
                raise ValueError("INSECTS contains a nonfinite feature or missing target")
            yield index, x, label, None, "abrupt_balanced.csv"


def _gas_rows(path):
    with zipfile.ZipFile(path) as archive:
        batches = {}
        for info in archive.infolist():
            match = re.fullmatch(r"(?:Dataset/)?batch(10|[1-9])\.dat", info.filename)
            if match:
                batch = int(match.group(1))
                if batch in batches or info.is_dir():
                    raise ValueError("Gas archive contains duplicate batch members")
                if info.file_size > 20 * 1024**2:
                    raise ValueError("Gas batch exceeds decompression budget")
                batches[batch] = info
        if set(batches) != set(range(1, 11)):
            raise ValueError("Gas archive must contain batches 1 through 10 exactly once")
        if sum(info.file_size for info in batches.values()) > 64 * 1024**2:
            raise ValueError("Gas archive exceeds total decompression budget")
        index = 0
        for batch in range(1, 11):
            with archive.open(batches[batch]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as stream:
                    for line in stream:
                        if len(line) > 50_000:
                            raise ValueError("Gas row exceeds text budget")
                        fields = line.split()
                        if not fields or fields[0] not in {"1", "2", "3", "4", "5", "6"}:
                            raise ValueError("Gas target is outside its six original classes")
                        x = {}
                        for field in fields[1:]:
                            key, value = field.split(":", 1)
                            if key in x:
                                raise ValueError("Gas duplicate feature index")
                            x[key] = float(value)
                        if set(x) != {str(i) for i in range(1, 129)}:
                            raise ValueError("Gas requires all 128 declared features")
                        if any(not math.isfinite(value) for value in x.values()):
                            raise ValueError("Gas contains nonfinite features")
                        yield index, x, fields[0], batch, batches[batch].filename
                        index += 1


def _rows(name, path):
    return _insects_rows(path) if name == "insects" else _gas_rows(path)


def _validate(name, path):
    spec = SPECS[name]
    labels, files = Counter(), {}
    rows = 0
    for index, _x, label, batch, member in _rows(name, path):
        if index != rows:
            raise ValueError("stream source offsets must be contiguous")
        rows += 1
        labels[label] += 1
        info = files.setdefault(member, {"member": member, "batch": batch, "rows": 0})
        info["rows"] += 1
    if rows != spec["expected_rows"]:
        raise ValueError("source cardinality differs from the pinned variant")
    if len(labels) != 6:
        raise ValueError("original stream must retain exactly six target classes")
    return {"rows": rows, "labels": dict(sorted(labels.items())), "files": list(files.values())}


def _download(url, path, *, max_bytes, timeout=60, opener=None):
    """No auth bypass, HTML confirmation handling, or retries after HTTP rejection."""
    request = urllib.request.Request(url, headers={"User-Agent": "residencia-ia-dataset-admission/1"})
    started, total = time.monotonic(), 0
    try:
        with (opener or urllib.request.urlopen)(request, timeout=min(timeout, 30)) as response:
            if response.status != 200:
                raise RuntimeError(f"dataset origin returned HTTP {response.status}")
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise RuntimeError("dataset origin returned HTML; no confirmation or authentication bypass")
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError("dataset download exceeds its byte budget")
            with Path(path).open("wb") as output:
                while block := response.read(64 * 1024):
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("dataset download deadline exceeded")
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("dataset download exceeds its byte budget")
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            return {"bytes": total, "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified")}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"dataset origin rejected acquisition with HTTP {error.code}") from None


def _acquire(name, cache, *, download):
    spec = SPECS[name]
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    directory = cache / name
    raw = directory / spec["filename"]
    receipt_path = directory / "acquisition.json"
    if directory.exists():
        if not raw.is_file() or not receipt_path.is_file() or raw.is_symlink() or receipt_path.is_symlink():
            raise ValueError("incomplete or symlinked stream cache; choose a new isolated cache directory")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("url") != spec["url"] or receipt.get("source_version") != spec["source_version"]
                or receipt.get("raw_sha256") != file_sha256(raw)):
            raise ValueError("stream cache checksum or source contract mismatch")
        return raw, receipt
    if not download:
        raise FileNotFoundError(f"{name} original source is not cached; explicit acquisition is required")
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=cache) as temporary:
        stage = Path(temporary)
        path = stage / spec["filename"]
        transport = _download(spec["url"], path, max_bytes=spec["max_bytes"])
        if spec["expected_bytes"] is not None and transport["bytes"] != spec["expected_bytes"]:
            raise ValueError("source byte length differs from the pinned variant")
        admission = _validate(name, path)
        receipt = {"schema_version": 1, "url": spec["url"], "source_version": spec["source_version"],
                   "raw_sha256": file_sha256(path), **transport, **admission,
                   "checksum_policy": "observed full-file SHA256 pinned in immutable acquisition receipt"}
        (stage / "acquisition.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
        try:
            os.rename(stage, directory)
        except OSError:
            if not directory.exists():
                raise
            # A concurrent successful acquisition owns the immutable directory.
            return _acquire(name, cache, download=False)
    return raw, receipt


class RealStreamSource:
    def __init__(self, name, raw, receipt, *, limit, seed):
        self.name, self.path, self.receipt = name, Path(raw), receipt
        self.samples, self.seed = limit, seed
        spec = SPECS[name]
        self.version = _digest({"adapter": "original-real-stream-v1", "name": name,
                                "raw_sha256": receipt["raw_sha256"], "limit": limit})
        self.spec = spec

    def samples_iter(self):
        if file_sha256(self.path) != self.receipt["raw_sha256"]:
            raise ValueError("raw stream changed after acquisition")
        for index, x, label, batch, _member in _rows(self.name, self.path):
            if index >= self.samples:
                break
            identity = _digest([self.receipt["raw_sha256"], index])
            yield TaskSample(identity, index, index, index, x, label,
                             task_id=batch - 1 if batch is not None else None,
                             group=str(batch) if batch is not None else None)

    def manifest(self):
        return {"name": self.name, "task": "multiclass", "adapter": "original-real-stream-v1",
                "version": self.version, "source_version": self.spec["source_version"],
                "source": self.spec["source"], "download_source": self.spec["url"],
                "variant": self.spec["variant"], "license": self.spec["license"],
                "license_reference": self.spec["license_reference"],
                "license_note": self.spec["license_note"],
                "raw_sha256": self.receipt["raw_sha256"], "raw_bytes": self.receipt["bytes"],
                "raw_rows": self.receipt["rows"], "samples": self.samples,
                "schema": {"features": self.spec["features"], "target": "six-class original string label"},
                "raw_class_counts": self.receipt["labels"], "files": self.receipt["files"],
                "order": ("published CSV row order" if self.name == "insects"
                          else "batch1..batch10, row order"),
                "selection": {"kind": "published_prefix", "seq_range": [0, self.samples]},
                "timestamp_kind": "simulated_ordinal; not original per-example collection time",
                "changepoints": None, "seed_ignored": True, "requested_seed": self.seed,
                "cache": str(self.path.parent), "label_delay_seconds": 0,
                "limitations": ["Prefix smoke may contain fewer classes and batches than the complete source",
                                "No true per-example drift annotations were supplied or invented"]}


def prepare_real_stream(name, cache, limit=100, seed=42, *, download=False):
    """Return an original-source multiclass stream, preserving at most `limit` rows."""
    if name not in SPECS:
        raise ValueError("real stream must be insects or gas")
    if type(limit) is not int or not 1 <= limit <= SPECS[name]["expected_rows"]:
        raise ValueError("invalid real-stream example limit")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid real-stream seed")
    raw, receipt = _acquire(name, cache, download=download)
    return RealStreamSource(name, raw, receipt, limit=limit, seed=seed)
