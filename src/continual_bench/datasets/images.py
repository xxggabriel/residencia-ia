"""Pinned IDX/CIFAR binary acquisition and budgeted continual-learning image tasks.

Only explicitly requested acquisitions perform network I/O. Archive contents are
read in place; no pickle or archive extraction is used. Epochs belong to the
evaluation protocol and never inflate the set of original selected images here.
"""

import gzip
import hashlib
import json
import os
import random
import struct
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

from continual_bench.core.contracts import TaskSample
from continual_bench.infrastructure.storage import atomic_write, run_lock

from .arff import file_sha256

NAMES = {"split_mnist", "permuted_mnist", "split_cifar10", "split_cifar100"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def image_dataset_locks():
    candidates = (
        Path(__file__).resolve().parents[1] / "image-datasets.lock.json",
        Path(__file__).resolve().parents[3] / "configs/image-datasets.lock.json",
        Path("configs/image-datasets.lock.json"),
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError("image-datasets.lock.json must accompany the deployment")


def _verified(path, asset):
    if path.is_symlink() or path.stat().st_size != asset["bytes"] or file_sha256(path) != asset["sha256"]:
        raise ValueError(f"image archive checksum/size mismatch: {asset['filename']}")


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        target = urllib.parse.urlsplit(new_url)
        if target.scheme != "https" or target.username or target.password:
            raise ValueError("image download redirected to an unsafe URL")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def acquire_asset(cache, asset, *, download=False, opener=None, timeout=60, max_seconds=300):
    """Acquire one exact asset. HTTP denials/checksum failures are never retried.

    Explicit response/deadline bounds prevent unlimited bodies. Existing corrupt
    cache entries fail closed rather than being silently replaced or redownloaded.
    """
    url = urllib.parse.urlsplit(asset["url"])
    filename = asset["filename"]
    if (url.scheme != "https" or url.username or url.password
            or Path(filename).name != filename or filename in {"", ".", ".."}):
        raise ValueError("image asset requires HTTPS without credentials and a plain filename")
    if not 0 < asset["bytes"] <= 180_000_000 or len(asset["sha256"]) != 64:
        raise ValueError("image asset exceeds the pinned admission budget")
    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    with run_lock(root / ".download-locks", asset["sha256"]):
        if path.exists():
            _verified(path, asset)
            return path
        if not download:
            raise FileNotFoundError(f"image archive absent: {filename}; explicit download=True required")
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "continual-bench/0.1.0"})
        descriptor, pending = tempfile.mkstemp(dir=root, prefix=".download-")
        started = time.monotonic()
        try:
            digest, count = hashlib.sha256(), 0
            open_url = opener or urllib.request.build_opener(_HTTPSRedirect()).open
            with os.fdopen(descriptor, "wb") as output, open_url(
                request, timeout=timeout
            ) as response:
                final_url = urllib.parse.urlsplit(response.geturl())
                if final_url.scheme != "https" or final_url.username or final_url.password:
                    raise ValueError("image download redirected to an unsafe URL")
                if response.status != 200:
                    raise ValueError("image download did not return HTTP 200")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != asset["bytes"]:
                    raise ValueError("image download Content-Length differs from pinned size")
                while True:
                    if time.monotonic() - started > max_seconds:
                        raise TimeoutError("image archive download deadline exceeded")
                    block = response.read(min(1024 * 1024, asset["bytes"] - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > asset["bytes"]:
                        raise ValueError("image download exceeds pinned byte budget")
                    digest.update(block)
                    output.write(block)
                if count != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
                    raise ValueError("image download checksum/size mismatch")
                output.flush()
                os.fsync(output.fileno())
            os.replace(pending, path)
        finally:
            if os.path.exists(pending):
                os.unlink(pending)
    return path


def read_mnist(images, labels, count):
    """Read a bounded original-order prefix of raw 28x28 IDX records."""
    with gzip.open(images, "rb") as image_stream, gzip.open(labels, "rb") as label_stream:
        image_header, label_header = image_stream.read(16), label_stream.read(8)
        if len(image_header) != 16 or len(label_header) != 8:
            raise ValueError("truncated MNIST IDX header")
        magic, image_count, rows, columns = struct.unpack(">IIII", image_header)
        label_magic, label_count = struct.unpack(">II", label_header)
        if (magic != 2051 or label_magic != 2049 or image_count != label_count
                or (rows, columns) != (28, 28) or not 0 < count <= image_count):
            raise ValueError("MNIST IDX schema/count mismatch")
        for index in range(count):
            pixels, label = image_stream.read(784), label_stream.read(1)
            if len(pixels) != 784 or len(label) != 1 or label[0] > 9:
                raise ValueError("MNIST IDX record truncated or label invalid")
            yield index, pixels, label[0]


def read_cifar(archive_path, dataset, split, count):
    """Read binary members without extraction or deserializing Python objects."""
    if dataset == "cifar10":
        members = ([f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)]
                   if split == "train" else ["cifar-10-batches-bin/test_batch.bin"])
        record_size, label_offset, classes = 3073, 0, 10
    elif dataset == "cifar100":
        members = [f"cifar-100-binary/{split}.bin"]
        record_size, label_offset, classes = 3074, 1, 100
    else:
        raise ValueError("unknown CIFAR binary dataset")
    emitted = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for name in members:
            member = archive.getmember(name)
            if not member.isfile() or member.size % record_size:
                raise ValueError("CIFAR binary member type or size invalid")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("CIFAR binary member missing")
            with stream:
                for _ in range(member.size // record_size):
                    raw = stream.read(record_size)
                    if len(raw) != record_size or raw[label_offset] >= classes:
                        raise ValueError("CIFAR binary record truncated or label invalid")
                    if dataset == "cifar100" and raw[0] >= 20:
                        raise ValueError("CIFAR100 coarse label invalid")
                    yield emitted, raw[-3072:], raw[label_offset]
                    emitted += 1
                    if emitted == count:
                        return
    raise ValueError("CIFAR archive has fewer records than requested")


class PreparedImageSource:
    def __init__(self, samples, manifest):
        self.samples, self._manifest = tuple(samples), manifest

    def samples_iter(self):
        for sample in self.samples:
            yield replace(sample, x=dict(sample.x))

    def manifest(self):
        return json.loads(_json(self._manifest))


def prepare_image_dataset(name, cache, limit=100, seed=42, *, download=False):
    """Prepare exactly `limit` distinct originals, with 80% train and 20% test.

    Split tasks use declared numeric class order. Budgeted Permuted MNIST uses
    disjoint original images for its two tasks, avoiding hidden duplication under
    the 100-example total budget. Neither schedule asserts natural chronology.
    """
    if name not in NAMES:
        raise ValueError("unknown prepared image dataset")
    if type(limit) is not int or not 10 <= limit <= 10_000:
        raise ValueError("image example budget must be between 10 and 10000")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("image protocol seed is invalid")
    dataset = "mnist" if name.endswith("mnist") else name.removeprefix("split_")
    specification = image_dataset_locks()["datasets"][dataset]
    raw_root = Path(cache) / "raw" / dataset
    assets = specification["assets"]
    paths = {a["filename"]: acquire_asset(raw_root, a, download=download) for a in assets}
    train_count = limit * 4 // 5
    split_counts = {"train": train_count, "test": limit - train_count}
    dimension = 784 if dataset == "mnist" else 3072
    permutation = list(range(dimension))
    random.Random(seed).shuffle(permutation)
    rows = []
    for split, count in split_counts.items():
        if dataset == "mnist":
            prefix = "train" if split == "train" else "t10k"
            iterator = read_mnist(paths[f"{prefix}-images-idx3-ubyte.gz"],
                                  paths[f"{prefix}-labels-idx1-ubyte.gz"], count)
        else:
            iterator = read_cifar(next(iter(paths.values())), dataset, split, count)
        for index, pixels, label in iterator:
            if name == "permuted_mnist":
                task = index % 2
                order = permutation if task else range(dimension)
            else:
                task = label // (10 if dataset == "cifar100" else 2)
                order = range(dimension)
            raw_identity = [dataset, [a["sha256"] for a in assets], split, index]
            identity = hashlib.sha256(_json(raw_identity)).hexdigest()
            rows.append((task, split, index, identity,
                         {f"pixel_{i:04d}": pixels[j] / 255.0 for i, j in enumerate(order)}, str(label)))
    rows.sort(key=lambda r: (r[0], r[1] != "train", r[2]))
    samples = [TaskSample(identity, seq, seq, seq, x, label, task_id=task,
                          split=split, group=f"{dataset}:{split}:{index}")
               for seq, (task, split, index, identity, x, label) in enumerate(rows)]
    if len(samples) != limit or len({s.sample_id for s in samples}) != limit:
        raise ValueError("prepared image budget or original-image uniqueness violated")
    descriptor = {
        "adapter": "raw-image-protocol-v1", "name": name, "dataset": dataset,
        "task": "multiclass", "samples": limit, "seed": seed,
        "source": specification["source"], "license": specification["license"],
        "license_reference": specification["license_reference"], "raw_assets": assets,
        "source_version": "original fixed IDX/binary release, pinned SHA-256 archives",
        "class_order": list(range(100 if dataset == "cifar100" else 10)),
        "split_counts": split_counts, "selection": "unshuffled prefix within each official split",
        "order": "artificial task, official split, original row index",
        "timestamp_kind": "artificial ordinal task replay; not collection time",
        "natural_chronology": False, "changepoints": None,
        "schema": {"features": dimension, "pixel_range": [0, 1], "target": "class string"},
        "preprocessing": "raw uint8 pixels /255; no fit, resize, augmentation or pretrained features",
        "protocol": ({"kind": "disjoint_budgeted_permuted_mnist", "tasks": 2,
                      "task0": "identity", "task1_permutation": permutation,
                      "deviation": "100 distinct originals partitioned across tasks; no full-base repetition"}
                     if name == "permuted_mnist" else
                     {"kind": "class_incremental", "classes_per_task": 10 if dataset == "cifar100" else 2}),
        "coverage_note": "small prefixes need not cover every class/task; no full CL reproduction claimed",
    }
    descriptor["version"] = hashlib.sha256(_json(descriptor)).hexdigest()
    payload = b"".join(_json(asdict(sample)) + b"\n" for sample in samples)
    descriptor["samples_sha256"] = hashlib.sha256(payload).hexdigest()
    prepared = Path(cache) / "prepared" / descriptor["version"]
    atomic_write(prepared / "samples.jsonl", payload)
    atomic_write(prepared / "manifest.json", _json(descriptor))
    return PreparedImageSource(samples, descriptor)
