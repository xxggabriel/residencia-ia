"""Explicit pinned Hugging Face CIFAR mirror with bounded Parquet range reads.

This is a declared alternative source, never an automatic fallback after denied
access. Published LFS hashes are retained as references, not claimed verified
when only selected ranges and embedded image bytes have been acquired.
"""

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

from .image_datasets import PreparedImageSource
from .local import atomic_write, run_lock
from .temporal_vision import RangeReader
from .week09_sources import TaskSample

SOURCES = {
    "split_cifar10": {
        "repo": "uoft-cs/cifar10", "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
        "dataset": "cifar10", "label": "label", "classes": 10, "classes_per_task": 2,
        "splits": {
            "train": {"path": "plain_text/train-00000-of-00001.parquet", "bytes": 119705255,
                      "rows": 50000,
                      "lfs_sha256": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"},
            "test": {"path": "plain_text/test-00000-of-00001.parquet", "bytes": 23940850,
                     "rows": 10000,
                     "lfs_sha256": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"},
        },
    },
    "split_cifar100": {
        "repo": "uoft-cs/cifar100", "revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
        "dataset": "cifar100", "label": "fine_label", "classes": 100, "classes_per_task": 10,
        "splits": {
            "train": {"path": "cifar100/train-00000-of-00001.parquet", "bytes": 118518617,
                      "rows": 50000,
                      "lfs_sha256": "694865d6b990e234804f01268586c41e88bcbbb75e20858432c05ad4081aca23"},
            "test": {"path": "cifar100/test-00000-of-00001.parquet", "bytes": 23772751,
                     "rows": 10000,
                     "lfs_sha256": "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68"},
        },
    },
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _pixels(encoded):
    from PIL import Image

    if not isinstance(encoded, bytes) or not 0 < len(encoded) <= 4 * 1024**2:
        raise ValueError("CIFAR Parquet requires bounded embedded image bytes")
    with Image.open(io.BytesIO(encoded)) as image:
        if image.size != (32, 32) or image.mode != "RGB":
            raise ValueError("CIFAR Parquet image schema must be RGB 32x32")
        raw = image.tobytes()
    # Preserve binary CIFAR's channel-planar feature convention, not RGB interleaving.
    return {f"pixel_{i:04d}": raw[(i % 1024) * 3 + i // 1024] / 255 for i in range(3072)}


def _parquet_prefix(reader, specification, count, label_name, classes):
    import pyarrow.parquet as pq

    if reader.size != specification["bytes"]:
        raise ValueError("pinned CIFAR Parquet size changed")
    parquet = pq.ParquetFile(reader)
    try:
        if parquet.metadata.num_rows != specification["rows"]:
            raise ValueError("pinned CIFAR Parquet row count changed")
        if not {"img", label_name} <= set(parquet.schema_arrow.names):
            raise ValueError("CIFAR Parquet image/target columns missing")
        emitted = 0
        for group in range(parquet.num_row_groups):
            # Arrow decodes column chunks; reject unexpectedly huge row groups before allocation.
            if parquet.metadata.row_group(group).total_byte_size > 64 * 1024**2:
                raise ValueError("CIFAR Parquet row group exceeds decoded memory budget")
            batches = parquet.iter_batches(batch_size=min(count, 128), row_groups=[group],
                                            columns=["img", label_name], use_threads=False)
            for batch in batches:
                for row in batch.to_pylist():
                    label, image = row[label_name], row["img"]
                    if type(label) is not int or not 0 <= label < classes:
                        raise ValueError("CIFAR Parquet class label outside declared range")
                    if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
                        raise ValueError(
                            "CIFAR Parquet requires embedded images; external paths are not fetched"
                        )
                    yield emitted, _pixels(image["bytes"]), str(label), _sha(image["bytes"])
                    emitted += 1
                    if emitted == count:
                        return
        raise ValueError("CIFAR Parquet prefix ended before requested budget")
    finally:
        parquet.close()


def _read_prepared(directory, contract):
    path = directory / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    payload = (directory / "samples.jsonl").read_bytes()
    if manifest.get("acquisition_contract") != contract or _sha(payload) != manifest.get("samples_sha256"):
        raise ValueError("prepared CIFAR Parquet cache contract/checksum changed")
    samples = [TaskSample(**json.loads(line)) for line in payload.splitlines()]
    if len(samples) != contract["limit"] or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("prepared CIFAR Parquet cache example IDs/count changed")
    return PreparedImageSource(samples, manifest)


def prepare_cifar_parquet(name, cache, limit=100, seed=42, *, download=False):
    """Select exactly `limit` mirror originals, preserving official 80/20 membership.

    A warm prepared cache needs no network. A cold preparation requires explicit
    `download=True`; it has a shared 64 MiB range budget across train and test.
    """
    if name not in SOURCES:
        raise ValueError("unknown CIFAR Parquet dataset")
    if type(limit) is not int or not 10 <= limit <= 10_000:
        raise ValueError("CIFAR Parquet example budget must be in [10,10000]")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("CIFAR Parquet seed must be a valid integer")
    source = SOURCES[name]
    contract = json.loads(_json({"adapter": "cifar-parquet-ranges-v1", "name": name,
                                 "limit": limit, "seed": seed, "source": source}))
    identity = _sha(_json(contract))
    root = Path(cache) / "cifar-parquet"
    directory = root / "prepared" / identity
    with run_lock(root / "locks", identity):
        cached = _read_prepared(directory, contract)
        if cached is not None:
            return cached
        if not download:
            raise FileNotFoundError("prepared CIFAR Parquet cache absent; explicit download=True required")
        counts = {"train": limit * 4 // 5, "test": limit - limit * 4 // 5}
        rows, provenance = [], []
        remaining = 64 * 1024**2
        for split, count in counts.items():
            spec = source["splits"][split]
            url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{spec['path']}"
            reader = RangeReader(url, root / "ranges", max_bytes=remaining)
            try:
                for index, pixels, label, image_sha in _parquet_prefix(
                    reader, spec, count, source["label"], source["classes"]
                ):
                    task = int(label) // source["classes_per_task"]
                    original = [source["repo"], source["revision"], split, index]
                    rows.append((task, split, index, _sha(_json(original)), pixels, label, image_sha))
                details = reader.provenance()
                details.update({"split": split, "published_lfs_sha256_reference": spec["lfs_sha256"],
                                "whole_archive_checksum_verified": False,
                                "integrity": "SHA-256 ranges/images; full Parquet LFS hash unverified"})
                provenance.append(details)
                remaining -= reader.transferred
            finally:
                reader.close()
        rows.sort(key=lambda row: (row[0], row[1] != "train", row[2]))
        samples = [TaskSample(original_id, seq, seq, seq, pixels, label, task_id=task,
                              split=split, group=f"{source['dataset']}:{split}:{index}")
                   for seq, (task, split, index, original_id, pixels, label, _) in enumerate(rows)]
        if len(samples) != limit or len({s.sample_id for s in samples}) != limit:
            raise ValueError("CIFAR Parquet selected IDs/count violate the original-example budget")
        payload = b"".join(_json(asdict(sample)) + b"\n" for sample in samples)
        manifest = {
            "name": name, "task": "multiclass", "version": identity,
            "adapter": contract["adapter"], "acquisition_contract": contract,
            "samples": limit, "samples_sha256": _sha(payload), "seed": seed,
            "seed_usage": "recorded only; numeric class order and within-split prefixes fixed",
            "source": "https://huggingface.co/datasets/" + source["repo"],
            "source_version": source["revision"],
            "original_source": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "license": "unknown (Hugging Face dataset card); no new SPDX grant inferred",
            "license_reference": "https://huggingface.co/datasets/" + source["repo"],
            "source_variant": "explicit HF-hosted Parquet mirror; not original Toronto archive bytes",
            "split_counts": counts, "selection": "unshuffled prefix within each official mirror split",
            "class_order": list(range(source["classes"])),
            "protocol": {"kind": "class_incremental", "classes_per_task": source["classes_per_task"]},
            "natural_chronology": False, "changepoints": None,
            "timestamp_kind": "artificial ordinal task replay, not collection time",
            "preprocessing": "decoded RGB32x32, channel-planar uint8 pixels /255; no resizing or fitting",
            "schema": {"features": 3072, "pixel_range": [0, 1], "target": "fine class string"},
            "parquet_sources": provenance,
            "selected_images": [{"split": row[1], "row_index": row[2], "sha256": row[6]} for row in rows],
            "total_range_bytes": 64 * 1024**2 - remaining,
            "whole_parquet_lfs_checksums_verified": False,
            "coverage_note": "budgeted subset may omit classes/tasks; no full visual benchmark reproduction",
        }
        atomic_write(directory / "samples.jsonl", payload)
        atomic_write(directory / "manifest.json", _json(manifest))
        return PreparedImageSource(samples, manifest)
