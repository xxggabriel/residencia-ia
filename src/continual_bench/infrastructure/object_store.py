"""Object storage port shared by the local fixture store and the S3 adapter.

Both implementations honour the same contract:

- keys are `/`-separated, relative to the store root;
- `if_absent=True` creates an object only when the key does not exist and raises
  `ObjectExists` otherwise, so evidence and completion markers are never overwritten;
- `head` returns size, version, user metadata and the SHA-256 recorded at upload time.
"""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


class ObjectExists(Exception):
    """A conditional create found an existing object at the key."""


@dataclass(frozen=True)
class StoredObject:
    key: str
    uri: str
    size: int
    version_id: str | None
    sha256: str | None
    metadata: dict = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_key(key: str) -> str:
    parts = key.split("/")
    if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid object key: {key!r}")
    return key


def split_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key` or `file:///root/key` → (bucket-or-root, key)."""
    for scheme in ("s3://", "file://"):
        if uri.startswith(scheme):
            rest = uri[len(scheme) :]
            if scheme == "file://":
                root, _, key = rest.rpartition("/objects/")
                return root + "/objects", clean_key(key)
            bucket, _, key = rest.partition("/")
            return bucket, clean_key(key)
    raise ValueError(f"unsupported object URI: {uri!r}")


class LocalObjectStore:
    """Filesystem store used by tests, CI smoke and offline replays of AWS runs."""

    def __init__(self, root):
        self.root = Path(root).resolve() / "objects"
        self.meta = Path(root).resolve() / "metadata"
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta.mkdir(parents=True, exist_ok=True)

    def uri(self, key: str) -> str:
        return f"file://{self.root}/{clean_key(key)}"

    def _path(self, key):
        return self.root / clean_key(key)

    def _write(self, key, source: Path, sha256: str, metadata: dict | None, if_absent: bool) -> StoredObject:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if if_absent:
            try:
                os.link(source, target)  # Atomic create-if-absent on POSIX filesystems.
            except FileExistsError as error:
                raise ObjectExists(key) from error
        else:
            os.replace(source, target)
        meta_path = self.meta / (clean_key(key) + ".json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"sha256": sha256, "metadata": dict(metadata or {}), "version_id": sha256[:16]}
        meta_path.write_text(json.dumps(record, sort_keys=True))
        return self.head(key)

    def put_bytes(
        self, key, data: bytes, *, if_absent=False, metadata=None, content_type=None
    ) -> StoredObject:
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            return self._write(key, Path(temporary), sha256_bytes(data), metadata, if_absent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def put_file(
        self, key, path, *, if_absent=False, metadata=None, sha256=None, content_type=None
    ) -> StoredObject:
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".pending-")
        os.close(fd)
        try:
            shutil.copyfile(path, temporary)
            return self._write(key, Path(temporary), sha256 or sha256_path(temporary), metadata, if_absent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def head(self, key) -> StoredObject | None:
        path = self._path(key)
        if not path.is_file():
            return None
        meta_path = self.meta / (clean_key(key) + ".json")
        record = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return StoredObject(
            key,
            self.uri(key),
            path.stat().st_size,
            record.get("version_id"),
            record.get("sha256"),
            record.get("metadata", {}),
        )

    def get_bytes(self, key) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def download(self, key, destination) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), destination)
        return destination

    def list(self, prefix: str = ""):
        base = self.root
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.startswith(".pending-"):
                key = path.relative_to(base).as_posix()
                if key.startswith(prefix):
                    yield key


def get_json(store, key):
    return json.loads(store.get_bytes(key))


def put_json(store, key, value, *, if_absent=False) -> StoredObject:
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    return store.put_bytes(key, data, if_absent=if_absent, content_type="application/json")


def put_json_once(store, key, value) -> tuple[StoredObject, bool]:
    """Create a JSON document once; an identical existing document is accepted (idempotent retry)."""
    try:
        return put_json(store, key, value, if_absent=True), True
    except ObjectExists:
        existing = get_json(store, key)
        if existing != json.loads(json.dumps(value, allow_nan=False)):
            raise
        return store.head(key), False
