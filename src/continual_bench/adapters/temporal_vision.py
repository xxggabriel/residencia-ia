"""Bounded real-image admission for CORe50, Stream-51 and CLEAR 2022.

Only selected ZIP members are fetched. HTTP servers must honor byte ranges;
there is deliberately no fallback that downloads a multi-gigabyte archive.
"""

import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from continual_bench.adapters.local import atomic_write
from continual_bench.adapters.week09_sources import TaskSample

CLEAR_REVISION = "b518a845a98f1c913497ab98a19727cb65b74e65"
CLEAR_ROOT = ("https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/"
              + CLEAR_REVISION)
SOURCES = {
    "core50": {
        "source": "https://vlomonaco.github.io/core50/", "release": "128x128-original",
        "train_url": "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip",
        "license": "CC-BY-4.0", "license_reference": "https://vlomonaco.github.io/core50/",
        "archive_md5_reference": "745f3373fed08d69343f1058ee559e13",
    },
    "stream51": {
        "source": "https://github.com/tyler-hayes/Stream-51", "release": "CVPRW-2020-original",
        "train_url": "http://klab.cis.rit.edu/files/Stream-51.zip",
        "license": "MIT (publisher repository; dataset-specific rights remain with publisher)",
        "license_reference": "https://github.com/tyler-hayes/Stream-51/blob/master/LICENSE",
        "archive_md5_reference": "5f34d542b71af7e5ecb2226d1bc7297c",
    },
    **{name: {
        "source": "https://clear-benchmark.github.io/", "release": "July-2022-" + CLEAR_REVISION,
        "train_url": f"{CLEAR_ROOT}/{name}-train-image-only.zip",
        "test_url": f"{CLEAR_ROOT}/{name}-test.zip",
        "license": "CC-BY (benchmark); individual YFCC image rights retained",
        "license_reference": "https://clear-benchmark.github.io/",
    } for name in ("clear10", "clear100")},
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class DatasetAccessError(RuntimeError):
    """A source could not be accessed under its normal transport/access policy."""


class RangeReader(io.RawIOBase):
    """Seekable HTTP file with immutable range cache and a cumulative byte budget."""

    def __init__(self, url, cache, *, max_bytes=64 * 1024**2, opener=None):
        super().__init__()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("dataset archive must be a public HTTP(S) URL without credentials")
        self.url, self.cache = url, Path(cache) / _sha(url.encode())
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.transferred, self.position = max_bytes, 0, 0
        self.opener = opener or urllib.request.urlopen
        self.validator = None
        self.ledger = []
        self.size = None
        self.blocks = {}
        self._range(0, 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid archive seek origin")
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("negative archive seek")
        self.position = position
        return position

    def _range(self, start, count):
        if len(self.ledger) >= 2048:
            raise ValueError("ZIP sampling request budget exceeded")
        if count > self.max_bytes or self.transferred + count > self.max_bytes:
            raise ValueError("ZIP sampling transfer budget exceeded")
        key = self.cache / f"{start}-{count}.bin"
        metadata = key.with_suffix(".json")
        if key.exists() and metadata.exists():
            receipt = json.loads(metadata.read_text())
            payload = key.read_bytes()
            if _sha(payload) != receipt["sha256"] or len(payload) != count:
                raise ValueError("cached archive range checksum mismatch")
            size, validator = receipt["archive_bytes"], receipt["validator"]
        else:
            request = urllib.request.Request(self.url, headers={
                "Range": f"bytes={start}-{start + count - 1}", "User-Agent": "continual-bench/0.1",
                "Accept-Encoding": "identity",
            })
            for attempt in range(3):
                try:
                    with self.opener(request, timeout=30) as response:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if (response.status != 206 or match is None
                                or tuple(map(int, match.groups()[:2])) != (start, start + count - 1)):
                            raise DatasetAccessError(
                                "archive server did not honor exact byte range; no full download"
                            )
                        size = int(match.group(3))
                        validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                        payload = response.read(count + 1)
                        if len(payload) != count:
                            raise ValueError("archive range payload length mismatch")
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                        raise DatasetAccessError(f"archive request rejected with HTTP {error.code}") from None
                except (urllib.error.URLError, TimeoutError, OSError):
                    if attempt == 2:
                        host = urllib.parse.urlsplit(self.url).hostname
                        raise DatasetAccessError(f"archive transport unavailable for {host}") from None
                time.sleep(0.2 * (attempt + 1))
            receipt = {"start": start, "bytes": count, "archive_bytes": size,
                       "validator": validator, "sha256": _sha(payload)}
            atomic_write(key, payload)
            atomic_write(metadata, _json(receipt))
        if self.size is not None and (self.size != size or self.validator != validator):
            raise ValueError("archive changed between requested ranges")
        self.size, self.validator = size, validator
        self.transferred += len(payload)
        self.ledger.append(receipt)
        return payload

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        count = remaining if size < 0 else min(size, remaining)
        if count == 0:
            return b""
        if count <= 65536:
            start = self.position // 65536 * 65536
            block_size = min(65536, self.size - start)
            if self.position + count <= start + block_size:
                if start not in self.blocks:
                    self.blocks[start] = self._range(start, block_size)
                offset = self.position - start
                payload = self.blocks[start][offset:offset + count]
            else:
                payload = self._range(self.position, count)
        else:
            payload = self._range(self.position, count)
        self.position += len(payload)
        return payload

    def provenance(self):
        return {"url": self.url, "archive_bytes": self.size, "validator": self.validator,
                "ranges_bytes": self.transferred, "ranges": self.ledger,
                "whole_archive_checksum_verified": False,
                "integrity": "ZIP member CRC plus SHA-256 of fetched ranges and selected members"}


class _Archive:
    def __init__(self, url, cache, *, local_path=None):
        self.remote = None if local_path else RangeReader(url, cache)
        self.zip = zipfile.ZipFile(local_path or self.remote)
        names = self.zip.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names")
        self.names = names
        self.name_set = set(names)
        self.members = []
        self.url = url

    def close(self):
        self.zip.close()
        if self.remote is not None:
            self.remote.close()

    def member(self, name, *, max_bytes=64 * 1024**2):
        info = self.zip.getinfo(name)
        if info.file_size > max_bytes or info.compress_size > max_bytes:
            raise ValueError("selected ZIP member exceeds sampling budget")
        payload = self.zip.read(name)  # zipfile validates CRC; no archive extraction or pickle execution.
        self.members.append({"path": name, "sha256": _sha(payload), "bytes": len(payload), "crc32": info.CRC})
        return payload

    def find(self, suffix):
        matches = [name for name in self.names if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"archive requires one {suffix} metadata member")
        return matches[0]

    def metadata(self, suffix):
        return json.loads(self.member(self.find(suffix)))

    def provenance(self):
        return {**(self.remote.provenance() if self.remote is not None else {
            "url": self.url, "whole_archive_checksum_verified": False,
            "integrity": "ZIP member CRC and selected-member SHA-256; locally supplied archive",
        }), "members": self.members}


def _pixels(payload, bbox=None):
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as original:
        if original.width * original.height > 20_000_000:
            raise ValueError("selected image exceeds decoded pixel budget")
        image = original.convert("RGB")
        if bbox is not None:
            right, left, bottom, top = map(float, bbox)
            if not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height:
                raise ValueError("invalid Stream-51 object bounding box")
            image = image.crop((left, top, right, bottom))
        image = image.resize((16, 16), Image.Resampling.BILINEAR)
        return {f"pixel_{index}": value / 255.0 for index, value in enumerate(image.tobytes())}


def _indices(size, count):
    if count > size:
        raise ValueError("source split contains fewer samples than requested")
    return [i * size // count for i in range(count)] if count else []


def _core_rows(archive):
    pattern = re.compile(r"(?:.*/)?s(\d+)/o(\d+)/C_(\d+)_(\d+)_(\d+)\.png$", re.I)
    rows = []
    for path in archive.names:
        match = pattern.fullmatch(path)
        if not match:
            continue
        session, obj, _, _, frame = map(int, match.groups())
        rows.append({"path": path, "label": str(obj - 1), "task": session - 1,
                     "group": f"session-{session}:object-{obj}",
                     "split": "test" if session in {3, 7, 10} else "train",
                     "order": (session, obj, frame)})
    return sorted(rows, key=lambda row: row["order"])


def _stream_rows(archive, split):
    metadata = archive.metadata(f"Stream-51_meta_{split}.json")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Stream-51 metadata must contain a nonempty list")
    first_suffix = str(PurePosixPath(metadata[0][-1]))
    first_path = archive.find(first_suffix)
    prefix = first_path[:-len(first_suffix)]
    rows = []
    for offset, record in enumerate(metadata):
        if not isinstance(record, list) or len(record) != (7 if split == "train" else 4):
            raise ValueError("Stream-51 metadata record schema changed")
        suffix = str(PurePosixPath(record[-1]))
        path = prefix + suffix
        if path not in archive.name_set:
            raise ValueError("Stream-51 frame missing or ambiguous")
        rows.append({"path": path, "label": str(record[0]), "task": int(record[0]),
                     "group": f"class-{record[0]}:clip-{record[1]}:video-{record[2]}" if split == "train"
                     else "official-test", "split": split, "order": offset, "bbox": record[-2]})
    return rows


def _clear_rows(archive, split, count):
    labels = archive.member(archive.find("class_names.txt")).decode().splitlines()
    labels = [label for label in labels if label]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("CLEAR class names must be unique")
    metadata = archive.metadata("labeled_metadata.json")
    prefix = archive.find("labeled_metadata.json").rsplit("/", 1)[0]
    groups = [(bucket, class_index, classes[label])
              for bucket, classes in sorted(metadata.items(), key=lambda item: int(item[0]))
              for class_index, label in enumerate(labels)]
    # Bound metadata I/O too: touch at most count bucket/class metadata files.
    selected_groups = [groups[i] for i in _indices(len(groups), min(count, len(groups)))]
    rows = []
    for group_index, (bucket, class_index, relative) in enumerate(selected_groups):
        entries = list(json.loads(archive.member(f"{prefix}/{relative}")).values())
        quota = count // len(selected_groups) + (group_index < count % len(selected_groups))
        for offset in _indices(len(entries), quota):
            item = entries[offset]
            path = f"{prefix}/{item['IMG_PATH']}"
            if path not in archive.name_set:
                raise ValueError("CLEAR selected metadata references a missing frame")
            rows.append({"path": path, "label": str(class_index), "task": int(bucket),
                         "group": f"bucket-{bucket}", "split": split,
                         "order": (int(bucket), class_index, offset)})
    return sorted(rows, key=lambda row: row["order"])


class TemporalVisionSource:
    def __init__(self, samples, manifest):
        self.samples, self._manifest = tuple(samples), manifest

    def samples_iter(self):
        yield from self.samples

    def manifest(self):
        return self._manifest


def prepare_temporal_dataset(name, cache, limit=100, seed=42, *, download=False, local_archives=None):
    """Acquire exactly `limit` unique official examples, 80% train and 20% test.

    The prefix within each selected split is an order-preserving deterministic
    stride over the source, allowing smoke coverage without loading every image.
    The sampled replay clock is ordinal; original positions/sessions are in the
    manifest. A 100-example smoke is not an official full continual-learning run.
    """
    if name not in SOURCES or type(limit) is not int or not 10 <= limit <= 1000:
        raise ValueError("unsupported temporal dataset or limit outside 10..1000")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    spec = SOURCES[name]
    cache = Path(cache) / name
    cache.mkdir(parents=True, exist_ok=True)
    receipt_path = cache / f"prepared-{limit}.json"
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        raw = saved["samples"]
        if _sha(_json(raw)) != saved["manifest"]["samples_sha256"]:
            raise ValueError("prepared temporal sample checksum mismatch")
        manifest = saved["manifest"]
        descriptor = {k: v for k, v in manifest.items() if k not in {"requested_seed", "version"}}
        if (manifest.get("adapter") != "temporal-vision-v1" or manifest.get("samples") != limit
                or _sha(_json(descriptor)) != manifest.get("version")):
            raise ValueError("prepared temporal manifest checksum/version mismatch")
        manifest["requested_seed"] = seed
        return TemporalVisionSource([TaskSample(**sample) for sample in raw], manifest)
    if not download and not local_archives:
        raise DatasetAccessError("temporal dataset cache is absent; explicit acquisition is required")
    local_archives = local_archives or {}
    train = _Archive(spec["train_url"], cache / "ranges", local_path=local_archives.get("train"))
    test = None
    try:
        train_count = limit * 4 // 5
        if name == "core50":
            all_rows = _core_rows(train)
            streams = {split: (train, [r for r in all_rows if r["split"] == split])
                       for split in ("train", "test")}
        elif name == "stream51":
            streams = {split: (train, _stream_rows(train, split)) for split in ("train", "test")}
        else:
            test = _Archive(spec["test_url"], cache / "ranges", local_path=local_archives.get("test"))
            streams = {"train": (train, _clear_rows(train, "train", train_count)),
                       "test": (test, _clear_rows(test, "test", limit - train_count))}
        selected, samples = [], []
        for split, count in (("train", train_count), ("test", limit - train_count)):
            archive, rows = streams[split]
            for index in _indices(len(rows), count):
                row = rows[index]
                payload = archive.member(row["path"], max_bytes=8 * 1024**2)
                identity = _sha(_json([name, spec["release"], row["path"], _sha(payload)]))
                seq = len(samples)
                samples.append(TaskSample(identity, seq, seq, seq, _pixels(payload, row.get("bbox")),
                                          row["label"], row["task"], split, row["group"]))
                selected.append({**row, "source_split_index": index, "image_sha256": _sha(payload)})
        if len({sample.sample_id for sample in samples}) != limit:
            raise ValueError("temporal dataset selected duplicate frames")
        manifest = {
            "adapter": "temporal-vision-v1", "name": name, "task": "multiclass", **spec,
            "samples": limit, "train_samples": train_count, "test_samples": limit - train_count,
            "selection": "order-preserving deterministic stride in each official split",
            "seed_ignored": True, "requested_seed": seed,
            "clock": "artificial ordinal replay; source session/bucket/position retained in selection",
            "representation": "RGB pixels bilinear-resized to16x16, divided by255; no pretrained features",
            "natural_order": "session/object/frame" if name == "core50" else "official metadata order",
            "protocol": (
                "CORe50 session-ordered NI diagnostic subset" if name == "core50" else
                "Stream51 published source ordering; no novelty evaluation" if name == "stream51" else
                "CLEAR July2022 official train/test buckets; not paper streaming test-on-next-bucket"
            ),
            "raw_archives": [a.provenance() for a in (train, test) if a is not None],
            "selected": selected, "schema": {"features": 768, "target": "original class ID as string"},
            "license_note": "No new terms accepted; publisher attribution and image-specific rights retained",
        }
        raw = [asdict(sample) for sample in samples]
        manifest["samples_sha256"] = _sha(_json(raw))
        manifest["version"] = _sha(_json({k: v for k, v in manifest.items() if k != "requested_seed"}))
        manifest = json.loads(_json(manifest))
        atomic_write(receipt_path, _json({"samples": raw, "manifest": manifest}))
        return TemporalVisionSource(samples, manifest)
    finally:
        train.close()
        if test is not None:
            test.close()
