"""Append-only model states with checkpoint-aligned recovery and immutable identities."""

import hashlib
import io
import json
import os
import pickle
from pathlib import Path

from .local import atomic_write


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _set_order(value):
    if isinstance(value, tuple):
        return ("tuple", tuple(_set_order(item) for item in value))
    if isinstance(value, frozenset):
        return ("frozenset", tuple(sorted(_set_order(item) for item in value)))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value).__name__, repr(value))
    raise ValueError("cannot verify canonical state containing unsupported set members")


class _StatePickler(pickle._Pickler):
    """Compare complete state while ignoring immutable literal aliases and set order.

    This is only a verification representation. Stored model blobs remain ordinary pickle.
    Dict ordering and aliases of mutable objects are preserved.
    """

    dispatch = pickle._Pickler.dispatch.copy()

    def __init__(self, stream):
        super().__init__(stream, protocol=5)
        self.literals = {}

    def save(self, obj, save_persistent_id=True):
        if type(obj) in (str, bytes):
            obj = self.literals.setdefault((type(obj), obj), obj)
        super().save(obj, save_persistent_id=save_persistent_id)

    def save_set(self, obj):
        self.save_reduce(set, (tuple(sorted(obj, key=_set_order)),), obj=obj)

    def save_frozenset(self, obj):
        self.save_reduce(frozenset, (tuple(sorted(obj, key=_set_order)),), obj=obj)

    dispatch[set] = save_set
    dispatch[frozenset] = save_frozenset


def _canonical_state(model):
    stream = io.BytesIO()
    _StatePickler(stream).dump(model)
    return stream.getvalue()


class ModelVersions:
    """Pickleable recorder: no open handles are retained in engine checkpoints."""

    def __init__(self, directory, dataset_version, code_sha256):
        self.directory = Path(directory)
        self.dataset_version = dataset_version
        self.code_sha256 = code_sha256
        self.entries = []
        self.offset = 0

    @property
    def journal(self):
        return self.directory / "models.bin"

    def record(self, model, training):
        blob = pickle.dumps(model, protocol=5)
        state_sha = hashlib.sha256(blob).hexdigest()
        view = {"dataset_version": self.dataset_version, "seq_range": training["seq_range"]}
        entry = {
            "ordinal": len(self.entries),
            "state_sha256": state_sha,
            "parent_version": self.entries[-1]["version_id"] if self.entries else None,
            "dataset_version": self.dataset_version,
            "training_view_version": digest_json(view),
            "code_sha256": self.code_sha256,
            "training": training,
            "offset": self.offset,
            "size": len(blob),
        }
        entry["version_id"] = digest_json(entry)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.journal.open("r+b" if self.journal.exists() else "w+b") as stream:
            # Discard an uncommitted tail after recovery from an older checkpoint.
            if stream.seek(0, 2) < self.offset:
                raise ValueError("model journal is shorter than the committed checkpoint")
            stream.seek(self.offset)
            stream.truncate()
            stream.write(blob)
        self.offset += len(blob)
        self.entries.append(entry)
        return entry["version_id"]

    def flush(self):
        if self.journal.exists():
            with self.journal.open("rb") as stream:
                os.fsync(stream.fileno())

    def finish(self, model, probe):
        if not self.entries:
            raise ValueError("trained model has no version")
        last = self.entries[-1]
        with self.journal.open("r+b") as stream:
            stream.truncate(self.offset)
            stream.seek(last["offset"])
            blob = stream.read(last["size"])
        if hashlib.sha256(blob).hexdigest() != last["state_sha256"]:
            raise ValueError("model state checksum mismatch")
        restored = pickle.loads(blob)  # Trusted, self-produced states only.
        if hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest() != last["state_sha256"]:
            # Checkpoint roundtrips can reorder sets or intern immutable strings without
            # changing learned state. Compare the full graph, not merely a prediction probe.
            if _canonical_state(model) != _canonical_state(restored):
                raise ValueError("final model state was not versioned")
        if restored.predict(probe) != model.predict(probe):
            raise ValueError("model serialization changed inference")
        atomic_write(self.directory / "model.pkl", blob)
        index = {
            "format_version": 1,
            "dataset_version": self.dataset_version,
            "code_sha256": self.code_sha256,
            "versions": self.entries,
        }
        encoded = json.dumps(index, indent=2).encode()
        atomic_write(self.directory / "model-versions.json", encoded)
        metadata = {
            "format_version": 1,
            "dataset_version": self.dataset_version,
            "code_sha256": self.code_sha256,
            "count": len(self.entries),
            "final_version_id": last["version_id"],
            "state_sha256": last["state_sha256"],
            "index_sha256": hashlib.sha256(encoded).hexdigest(),
            "journal_sha256": self.journal_sha256(),
            "inference_probe": {"features": probe, "probability": restored.predict(probe)},
        }
        atomic_write(self.directory / "model-metadata.json", json.dumps(metadata, indent=2).encode())
        self.flush()
        return metadata

    def journal_sha256(self):
        with self.journal.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()


def load_version(directory, version_id):
    directory = Path(directory)
    encoded = (directory / "model-versions.json").read_bytes()
    metadata_path = directory / "model-metadata.json"
    if metadata_path.exists():
        expected = json.loads(metadata_path.read_text())["index_sha256"]
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError("model index checksum mismatch")
    index = json.loads(encoded)
    entry = next(e for e in index["versions"] if e["version_id"] == version_id)
    if digest_json({k: v for k, v in entry.items() if k != "version_id"}) != version_id:
        raise ValueError("model version lineage checksum mismatch")
    with (directory / "models.bin").open("rb") as stream:
        stream.seek(entry["offset"])
        blob = stream.read(entry["size"])
    if hashlib.sha256(blob).hexdigest() != entry["state_sha256"]:
        raise ValueError("model version checksum mismatch")
    return pickle.loads(blob)
