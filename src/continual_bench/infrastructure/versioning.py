"""Append-only model states with checkpoint-aligned recovery and immutable identities."""

import hashlib
import io
import json
import os
import pickle
from pathlib import Path

from .storage import atomic_write


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

    def __init__(
        self, directory, dataset_version, code_sha256, *, artifact_mode="full", state_sample_every=1
    ):
        if artifact_mode not in {"full", "metrics_only"}:
            raise ValueError("artifact_mode must be full or metrics_only")
        self.directory = Path(directory)
        self.dataset_version = dataset_version
        self.code_sha256 = code_sha256
        self.entries = []
        self.offset = 0
        self.artifact_mode = artifact_mode
        if type(state_sample_every) is not int or state_sample_every < 1:
            raise ValueError("state_sample_every must be a positive integer")
        self.state_sample_every = state_sample_every
        self._last_blob = None

    @property
    def journal(self):
        return self.directory / "models.bin"

    def record(self, model, training):
        if self.artifact_mode == "metrics_only":
            return self._record_metric_version(model, training)
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

    def _record_metric_version(self, model, training):
        """Every update has lineage; state hashes/sizes are measurements at explicit cuts."""
        lineage = {
            "identity": "training-lineage-v1-not-a-model-state-hash",
            "ordinal": len(self.entries),
            "parent_version": self.entries[-1]["version_id"] if self.entries else None,
            "dataset_version": self.dataset_version,
            "training_view_version": digest_json(
                {"dataset_version": self.dataset_version, "seq_range": training["seq_range"]}
            ),
            "code_sha256": self.code_sha256,
            "training": training,
        }
        sample = training["action"] != "learn_one" or len(self.entries) % self.state_sample_every == 0
        blob = pickle.dumps(model, protocol=5) if sample else None
        entry = {
            **lineage,
            "version_id": digest_json(lineage),
            "retained": False,
            "offset": None,
            "state_sha256": hashlib.sha256(blob).hexdigest() if blob is not None else None,
            "size": len(blob) if blob is not None else None,
            "state_measurement": "sampled" if sample else "not_measured_between_state_sampling_cuts",
        }
        self._last_blob = blob
        self.entries.append(entry)
        return entry["version_id"]

    def flush(self):
        if self.artifact_mode == "full" and self.journal.exists():
            with self.journal.open("rb") as stream:
                os.fsync(stream.fileno())

    def record_candidate(self, model, seq):
        """Retain rejected states independently of the promoted-model journal."""
        blob = pickle.dumps(model, protocol=5)
        path = f"candidates/rejected-{seq:08d}.pkl"
        if self.artifact_mode == "full":
            atomic_write(self.directory / path, blob)
        metadata = {
            "file": path if self.artifact_mode == "full" else None,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "format": "pickle",
        }
        if self.artifact_mode == "metrics_only":
            metadata.update(retained=False, storage_bytes=0)
        return metadata

    def finish(self, model, probe):
        if not self.entries:
            raise ValueError("trained model has no version")
        last = self.entries[-1]
        if self.artifact_mode == "full":
            with self.journal.open("r+b") as stream:
                stream.truncate(self.offset)
                stream.seek(last["offset"])
                blob = stream.read(last["size"])
        else:
            blob = self._last_blob
            if blob is None:
                blob = pickle.dumps(model, protocol=5)
                last.update(
                    state_sha256=hashlib.sha256(blob).hexdigest(),
                    size=len(blob),
                    state_measurement="sampled_at_finalization",
                )
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
        if self.artifact_mode == "full":
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
            "storage_bytes": self.offset + len(blob) if self.artifact_mode == "full" else 0,
            "final_version_id": last["version_id"],
            "state_sha256": last["state_sha256"],
            "index_sha256": hashlib.sha256(encoded).hexdigest(),
            "journal_sha256": self.journal_sha256() if self.artifact_mode == "full" else None,
            "inference_probe": {"features": probe, "probability": restored.predict(probe)},
        }
        if self.artifact_mode == "metrics_only":
            metadata.update(
                artifact_mode="metrics_only",
                retained=False,
                serialized_state_bytes=len(blob),
                version_identity="training-lineage-v1-not-a-model-state-hash",
                state_measurement={
                    "every_updates": self.state_sample_every,
                    "initial_and_reset": True,
                    "final": True,
                    "unmeasured_values": None,
                    "scope": "serialized model size and SHA-256; transient bytes only",
                    "coordinate": "model_version_ordinal",
                },
                state_measurements=[
                    {
                        "ordinal": entry["ordinal"],
                        "seq_range": entry["training"]["seq_range"],
                        "serialized_bytes": entry["size"],
                        "sha256": entry["state_sha256"],
                        "phase": entry["state_measurement"],
                    }
                    for entry in self.entries
                    if entry["size"] is not None
                ],
            )
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
    if entry.get("retained") is False:
        raise ValueError("model weights were not retained in metrics_only mode")
    if digest_json({k: v for k, v in entry.items() if k != "version_id"}) != version_id:
        raise ValueError("model version lineage checksum mismatch")
    with (directory / "models.bin").open("rb") as stream:
        stream.seek(entry["offset"])
        blob = stream.read(entry["size"])
    if hashlib.sha256(blob).hexdigest() != entry["state_sha256"]:
        raise ValueError("model version checksum mismatch")
    return pickle.loads(blob)
