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

from continual_bench.core.contracts import Event, TaskSample

from .arff import file_sha256
from .base import validate_bounds

SPECS = {
    "insects": {
        "variant": "abrupt_balanced",
        "source_version": "River-0.26.1-public-file-mapping",
        "source": "https://sites.google.com/view/uspdsrepository",
        "url": ("https://drive.google.com/uc?export=download&id=1WQoIuuVgiuXfzv4kvao6XuLQG37V923O&confirm=t"),
        "filename": "abrupt_balanced.csv",
        "expected_bytes": 14_151_769,
        "max_bytes": 20 * 1024**2,
        "expected_rows": 52_848,
        "features": 33,
        "license": "CC-BY-4.0 (USP repository; attribution retained)",
        "license_reference": "https://sites.google.com/view/uspdsrepository",
        "license_note": "Repository warns that external-source datasets can have additional restrictions.",
    },
    "insects_abrupt_imbalanced": {
        "variant": "abrupt_imbalanced",
        "source_version": "USP-DS-Repository-Souza-2020",
        "source": "https://sites.google.com/view/uspdsrepository",
        "url": "https://drive.google.com/uc?export=download&id=1Z9W2_mwawobXeTihZDTDnFOCtfdgmgDa&confirm=t",
        "filename": "abrupt_imbalanced.csv",
        "expected_bytes": 94_893_622,
        "max_bytes": 105 * 1024**2,
        "expected_sha256": "985a5b46c2cb0e42e48a78d5001b57a79ed64fa69f9e5b0362a34cfb552dbc22",
        "expected_rows": 355_275,
        "features": 33,
        "positive_label": "2",
        "target_transformation": {
            "kind": "binary_one_vs_rest",
            "positive_source_label": "2",
            "mapping": "label == '2' -> 1, else 0",
            "binary_labels": ["0", "1"],
            "positive_label": "1",
        },
        "changepoints": (83859, 128651, 182320, 242883, 268380),
        "original_class_distribution": {
            "11": 54320,
            "12": 99557,
            "2": 101256,
            "3": 67237,
            "4": 21204,
            "5": 11701,
        },
        "binary_distribution_per_regime": [
            {"regime": 1, "start": 0, "end": 83859, "rows": 83859, "positive": 33314, "negative": 50545},
            {"regime": 2, "start": 83859, "end": 128651, "rows": 44792, "positive": 5535, "negative": 39257},
            {
                "regime": 3,
                "start": 128651,
                "end": 182320,
                "rows": 53669,
                "positive": 14904,
                "negative": 38765,
            },
            {
                "regime": 4,
                "start": 182320,
                "end": 242883,
                "rows": 60563,
                "positive": 15924,
                "negative": 44639,
            },
            {"regime": 5, "start": 242883, "end": 268380, "rows": 25497, "positive": 2985, "negative": 22512},
            {
                "regime": 6,
                "start": 268380,
                "end": 355275,
                "rows": 86895,
                "positive": 28594,
                "negative": 58301,
            },
        ],
        "drift_origin": (
            "Souza et al. (2020), Challenges in Benchmarking Stream Learning Algorithms "
            "with Real-world Data, Table 2 (p. 37), arXiv:2005.00113"
        ),
        "drift_semantics": (
            "abrupt concept drift; each changepoint is the 0-based inclusive start index of the new regime"
        ),
        "drift_convention": "0-based half-open [start, end) intervals",
        "license": "CC BY 4.0",
        "license_reference": "https://creativecommons.org/licenses/by/4.0/",
        "license_note": (
            "Creative Commons Attribution 4.0 International (USP Data Stream Repository; Souza et al. 2020)."
        ),
    },
    "insects_incremental_abrupt_reoccurring_imbalanced": {
        "variant": "incremental_abrupt_reoccurring_imbalanced",
        "source_version": "USP-DS-Repository-Souza-2020",
        "source": "https://sites.google.com/view/uspdsrepository",
        "url": (
            "https://drive.usercontent.google.com/download"
            "?id=1M6QfsernUlM0qvqXdbo9bPYAsHCcfuhb&export=download&confirm=t"
        ),
        "filename": "incremental_abrupt_imbalanced.csv",
        "expected_bytes": 120_798_571,
        "max_bytes": 135 * 1024**2,
        "expected_sha256": "d07e22c0521755f2452d96bb1c7068b26a281be2a627e32f21a4c0c6cfbfd9bc",
        "expected_rows": 452_044,
        "features": 33,
        "positive_label": "2",
        "target_transformation": {
            "kind": "binary_one_vs_rest",
            "positive_source_label": "2",
            "mapping": "label == '2' -> 1, else 0",
            "binary_labels": ["0", "1"],
            "positive_label": "1",
        },
        "changepoints": (150683, 301365),
        "original_class_distribution": {
            "11": 64895,
            "12": 134717,
            "2": 125354,
            "3": 83794,
            "4": 29953,
            "5": 13331,
        },
        "binary_distribution_per_regime": [
            {"regime": 1, "start": 0, "end": 150683, "rows": 150683, "positive": 41805, "negative": 108878},
            {
                "regime": 2,
                "start": 150683,
                "end": 301365,
                "rows": 150682,
                "positive": 41758,
                "negative": 108924,
            },
            {
                "regime": 3,
                "start": 301365,
                "end": 452044,
                "rows": 150679,
                "positive": 41791,
                "negative": 108888,
            },
        ],
        "drift_origin": (
            "Souza et al. (2020), Challenges in Benchmarking Stream Learning Algorithms "
            "with Real-world Data, Table 2 (p. 37), arXiv:2005.00113"
        ),
        "drift_semantics": (
            "incremental-abrupt-reoccurring concept drift; changepoints mark the 0-based inclusive "
            "start indices of abrupt transitions between reoccurring regimes"
        ),
        "drift_convention": "0-based half-open [start, end) intervals",
        "incremental_evolution_note": (
            "In addition to the two abrupt changepoints at 150683 and 301365, the stream exhibits "
            "continuous incremental evolution within regimes as described in Souza et al. (2020)."
        ),
        "license": "CC BY 4.0",
        "license_reference": "https://creativecommons.org/licenses/by/4.0/",
        "license_note": (
            "Creative Commons Attribution 4.0 International (USP Data Stream Repository; Souza et al. 2020)."
        ),
    },
    "gas": {
        "variant": "uci-224-ten-batches-corrected-batch10",
        "source_version": "UCI224-batch10-corrected-2013-10-14",
        "source": "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
        "url": ("https://archive.ics.uci.edu/static/public/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset.zip"),
        "filename": "gas-sensor-array-drift.zip",
        "expected_bytes": None,
        "max_bytes": 16 * 1024**2,
        "expected_rows": 13_910,
        "features": 128,
        "license": "CC-BY-4.0 (current UCI catalog)",
        "license_reference": "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
        "license_note": (
            "Current UCI license says CC-BY-4.0; legacy descriptive text says research-only. "
            "This Residencia IA acquisition is for research; no additional terms accepted."
        ),
    },
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _insects_rows(path, *, filename="abrupt_balanced.csv"):
    """Stream headerless 34-column USP INSECTS CSV rows in O(1) memory."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for index, fields in enumerate(csv.reader(stream)):
            if len(fields) != 34:
                raise ValueError("INSECTS requires 33 features and one target per row")
            try:
                x = {f"f{i + 1}": float(value) for i, value in enumerate(fields[:-1])}
            except ValueError:
                raise ValueError("INSECTS numeric feature schema mismatch") from None
            label = fields[-1].strip()
            if any(not math.isfinite(v) for v in x.values()) or not label:
                raise ValueError("INSECTS contains a nonfinite feature or missing target")
            yield index, x, label, None, filename


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
    if name == "gas":
        return _gas_rows(path)
    return _insects_rows(path, filename=SPECS[name]["filename"])


def _regime_bounds(changepoints, total_rows):
    points = [0, *(p for p in changepoints if 0 < p < total_rows), total_rows]
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _validate(name, path):
    spec = SPECS[name]
    labels, files = Counter(), {}
    changepoints = spec.get("changepoints") or ()
    positive_label = spec.get("positive_label")
    bounds = _regime_bounds(changepoints, spec["expected_rows"]) if positive_label else ()
    regime_stats = (
        [
            {"regime": i + 1, "start": s, "end": e, "rows": 0, "positive": 0, "negative": 0}
            for i, (s, e) in enumerate(bounds)
        ]
        if bounds
        else None
    )
    regime_idx = 0
    rows = 0
    for index, _x, label, batch, member in _rows(name, path):
        if index != rows:
            raise ValueError("stream source offsets must be contiguous")
        if regime_stats is not None:
            while regime_idx + 1 < len(regime_stats) and index >= regime_stats[regime_idx]["end"]:
                regime_idx += 1
            if index < regime_stats[regime_idx]["end"]:
                regime_stats[regime_idx]["rows"] += 1
                if label == positive_label:
                    regime_stats[regime_idx]["positive"] += 1
                else:
                    regime_stats[regime_idx]["negative"] += 1
        rows += 1
        labels[label] += 1
        info = files.setdefault(member, {"member": member, "batch": batch, "rows": 0})
        info["rows"] += 1
    if rows != spec["expected_rows"]:
        raise ValueError("source cardinality differs from the pinned variant")
    if len(labels) != 6:
        raise ValueError("original stream must retain exactly six target classes")
    result = {"rows": rows, "labels": dict(sorted(labels.items())), "files": list(files.values())}
    if regime_stats is not None:
        result["binary_distribution_per_regime"] = regime_stats
    return result


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
            return {
                "bytes": total,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
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
        expected_sha = spec.get("expected_sha256")
        if (
            receipt.get("url") != spec["url"]
            or receipt.get("source_version") != spec["source_version"]
            or receipt.get("raw_sha256") != file_sha256(raw)
            or (expected_sha is not None and receipt.get("raw_sha256") != expected_sha)
        ):
            raise ValueError("stream cache checksum or source contract mismatch")
        return raw, receipt
    if not download:
        raise FileNotFoundError(f"{name} original source is not cached; explicit acquisition is required")
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=cache) as temporary:
        stage = Path(temporary)
        path = stage / spec["filename"]
        transport = _download(
            spec["url"],
            path,
            max_bytes=spec["max_bytes"],
            timeout=300 if spec["max_bytes"] > 50 * 1024**2 else 60,
        )
        if spec["expected_bytes"] is not None and transport["bytes"] != spec["expected_bytes"]:
            raise ValueError("source byte length differs from the pinned variant")
        raw_sha256 = file_sha256(path)
        if spec.get("expected_sha256") is not None and raw_sha256 != spec["expected_sha256"]:
            raise ValueError("source SHA-256 differs from the pinned variant")
        admission = _validate(name, path)
        receipt = {
            "schema_version": 1,
            "url": spec["url"],
            "source_version": spec["source_version"],
            "raw_sha256": raw_sha256,
            **transport,
            **admission,
            "checksum_policy": "observed full-file SHA256 pinned in immutable acquisition receipt",
        }
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
        self.version = _digest(
            {
                "adapter": "original-real-stream-v1",
                "name": name,
                "raw_sha256": receipt["raw_sha256"],
                "limit": limit,
            }
        )
        self.spec = spec

    def samples_iter(self):
        if file_sha256(self.path) != self.receipt["raw_sha256"]:
            raise ValueError("raw stream changed after acquisition")
        for index, x, label, batch, _member in _rows(self.name, self.path):
            if index >= self.samples:
                break
            identity = _digest([self.receipt["raw_sha256"], index])
            yield TaskSample(
                identity,
                index,
                index,
                index,
                x,
                label,
                task_id=batch - 1 if batch is not None else None,
                group=str(batch) if batch is not None else None,
            )

    def manifest(self):
        return {
            "name": self.name,
            "task": "multiclass",
            "adapter": "original-real-stream-v1",
            "version": self.version,
            "source_version": self.spec["source_version"],
            "source": self.spec["source"],
            "download_source": self.spec["url"],
            "variant": self.spec["variant"],
            "license": self.spec["license"],
            "license_reference": self.spec["license_reference"],
            "license_note": self.spec["license_note"],
            "raw_sha256": self.receipt["raw_sha256"],
            "raw_bytes": self.receipt["bytes"],
            "raw_rows": self.receipt["rows"],
            "samples": self.samples,
            "schema": {"features": self.spec["features"], "target": "six-class original string label"},
            "raw_class_counts": self.receipt["labels"],
            "files": self.receipt["files"],
            "order": ("published CSV row order" if self.name == "insects" else "batch1..batch10, row order"),
            "selection": {"kind": "published_prefix", "seq_range": [0, self.samples]},
            "timestamp_kind": "simulated_ordinal; not original per-example collection time",
            "changepoints": None,
            "seed_ignored": True,
            "requested_seed": self.seed,
            "cache": str(self.path.parent),
            "label_delay_seconds": 0,
            "limitations": [
                "Prefix smoke may contain fewer classes and batches than the complete source",
                "No true per-example drift annotations were supplied or invented",
            ],
        }


INSECTS_BINARY_VARIANTS = (
    "insects_abrupt_imbalanced",
    "insects_incremental_abrupt_reoccurring_imbalanced",
)


class InsectsBinarySource:
    """Bounded-memory binary stream adapter for USP INSECTS imbalanced variants.

    Projects `label == '2'` -> 1 (True) and all other classes -> 0 (False) in O(1) memory.
    Drift annotations are retained in `manifest()` strictly for evaluation/auditing and are
    never leaked into features (`x`) or prospective fields (`Event.concept`, `TaskSample.task_id`).
    """

    def __init__(
        self,
        name,
        samples,
        seed,
        label_delay,
        epoch,
        *,
        directory=None,
        cache=None,
        download=False,
    ):
        if name not in INSECTS_BINARY_VARIANTS:
            raise ValueError(f"unsupported binary INSECTS variant: {name}")
        validate_bounds(samples, seed, label_delay, epoch)
        spec = SPECS[name]
        if samples > spec["expected_rows"]:
            raise ValueError("Requested prefix exceeds pinned source rows")
        self.name, self.samples, self.seed = name, samples, seed
        self.label_delay, self.epoch = label_delay, epoch
        self.spec = spec
        root = Path(directory or cache or os.environ.get("BENCH_DATASETS", "/state/datasets")).resolve()
        self.path, self.receipt = self._resolve(root, download=download)
        self.version = _digest(
            {
                "adapter": "insects-binary-stream-v1",
                "name": name,
                "variant": spec["variant"],
                "raw_sha256": self.receipt["raw_sha256"],
                "samples": samples,
                "label_delay": label_delay,
                "epoch": epoch,
                "positive_label": spec["positive_label"],
            }
        )

    def _resolve(self, root, *, download):
        spec = self.spec
        cached_dir = root / self.name
        if cached_dir.exists():
            return _acquire(self.name, root, download=False)
        flat_path = root / spec["filename"]
        if flat_path.is_file():
            if flat_path.is_symlink():
                raise ValueError("symlinked dataset file is not allowed")
            observed_sha = file_sha256(flat_path)
            if spec.get("expected_sha256") is not None and observed_sha != spec["expected_sha256"]:
                raise ValueError(f"Pinned {self.name} SHA-256 mismatch")
            admission = _validate(self.name, flat_path)
            receipt = {
                "schema_version": 1,
                "url": spec["url"],
                "source_version": spec["source_version"],
                "raw_sha256": observed_sha,
                "bytes": flat_path.stat().st_size,
                **admission,
            }
            return flat_path, receipt
        if download:
            return _acquire(self.name, root, download=True)
        raise FileNotFoundError(
            f"Pinned {self.name} source is absent; run scripts/prepare_real_sources.py first"
        )

    def _iter_projected(self):
        if file_sha256(self.path) != self.receipt["raw_sha256"]:
            raise ValueError("raw stream changed after acquisition")
        positive_source = self.spec["positive_label"]
        for index, x, label, _batch, _member in _insects_rows(self.path, filename=self.spec["filename"]):
            if index >= self.samples:
                break
            identity = _digest([self.receipt["raw_sha256"], index])
            timestamp = self.epoch + index
            yield identity, index, timestamp, x, label == positive_source

    def events(self):
        for identity, seq, timestamp, x, is_positive in self._iter_projected():
            yield Event(identity, seq, timestamp, timestamp + self.label_delay, x, is_positive, None)

    def samples_iter(self):
        for identity, seq, timestamp, x, is_positive in self._iter_projected():
            yield TaskSample(
                identity,
                seq,
                timestamp,
                timestamp + self.label_delay,
                x,
                "1" if is_positive else "0",
                split="train",
                task_id=None,
                group=None,
            )

    def manifest(self):
        all_changepoints = list(self.spec["changepoints"])
        active_changepoints = [p for p in all_changepoints if p < self.samples]
        regimes = self.receipt.get(
            "binary_distribution_per_regime",
            self.spec["binary_distribution_per_regime"],
        )
        drift_annotations = {
            "origin": self.spec["drift_origin"],
            "semantics": self.spec["drift_semantics"],
            "convention": self.spec["drift_convention"],
            "usage_restriction": (
                "evaluation_and_auditing_only; never exposed in features or prospective decisions"
            ),
            "changepoints": active_changepoints,
            "source_changepoints": all_changepoints,
            "regimes": regimes,
        }
        if "incremental_evolution_note" in self.spec:
            drift_annotations["incremental_evolution_note"] = self.spec["incremental_evolution_note"]
        return {
            "name": self.name,
            "task": "binary",
            "adapter": "insects-binary-stream-v1",
            "version": self.version,
            "source_version": self.spec["source_version"],
            "source": self.spec["source"],
            "url": self.spec["url"],
            "download_source": self.spec["url"],
            "variant": self.spec["variant"],
            "license": self.spec["license"],
            "license_reference": self.spec["license_reference"],
            "license_note": self.spec["license_note"],
            "raw_sha256": self.receipt["raw_sha256"],
            "raw_bytes": self.receipt["bytes"],
            "raw_rows": self.receipt["rows"],
            "samples": self.samples,
            "schema": {
                "features": self.spec["features"],
                "feature_names": [f"f{i + 1}" for i in range(self.spec["features"])],
                "target": "binary (label == '2' -> 1, else 0)",
                "source_target": "six-class original string label",
            },
            "labels": ["0", "1"],
            "positive_label": "1",
            "source_positive_label": self.spec["positive_label"],
            "target_transformation": self.spec["target_transformation"],
            "original_class_distribution": self.receipt.get(
                "labels", self.spec["original_class_distribution"]
            ),
            "raw_class_counts": self.receipt.get("labels", self.spec["original_class_distribution"]),
            "binary_distribution_per_regime": regimes,
            "files": self.receipt["files"],
            "order": "published CSV row order",
            "selection": {"kind": "published_prefix", "seq_range": [0, self.samples]},
            "timestamp_kind": "simulated_ordinal; not original per-example collection time",
            "source_changepoints": all_changepoints,
            "changepoints": active_changepoints,
            "drift_annotations": drift_annotations,
            "seed_ignored": True,
            "requested_seed": self.seed,
            "cache": str(self.path.parent),
            "label_delay_seconds": self.label_delay,
            "label_delay_is_simulated": True,
        }


def prepare_real_stream(name, cache, limit=100, seed=42, *, download=False):
    """Return an original-source stream, preserving at most `limit` rows."""
    if name not in SPECS:
        raise ValueError("real stream must be insects or gas (or a pinned INSECTS variant)")
    if type(limit) is not int or not 1 <= limit <= SPECS[name]["expected_rows"]:
        raise ValueError("invalid real-stream example limit")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid real-stream seed")
    raw, receipt = _acquire(name, cache, download=download)
    if name in INSECTS_BINARY_VARIANTS:
        return InsectsBinarySource(name, limit, seed, 0, 0, cache=cache, download=False)
    return RealStreamSource(name, raw, receipt, limit=limit, seed=seed)
