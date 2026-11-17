"""Confirmed Spot snapshots and publication receipts in the campaign object store."""

import json
import logging
import math
import pickle
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from continual_bench.infrastructure.object_store import (
    ObjectExists,
    clean_key,
    get_json,
    put_json_once,
    sha256_bytes,
    sha256_path,
)

from . import layout

CHECKPOINT_VERSION = "spot-checkpoint-v1"
SKIP_SUFFIXES = (".lock",)
TEMP_PREFIXES = (".pending-", ".predictions-", ".model-", ".prepare-")
PRIVATE_PREFIXES = (".zia-resume",)
_MANIFEST_SUFFIX = re.compile(r"(?P<attempt>[0-9]{6})/[0-9]{8}\.json")
_SHA256 = re.compile(r"[a-f0-9]{64}")


class CheckpointError(ValueError):
    """A checkpoint cannot be safely interpreted under this execution contract."""


class SpotGrantError(ValueError):
    """The instance is unauthorized to work on this attempt."""


class CampaignCancelled(InterruptedError):
    """The operator stopped the campaign."""


def output_files(directory, *, private=False):
    prefixes = TEMP_PREFIXES if private else TEMP_PREFIXES + PRIVATE_PREFIXES
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part.startswith(prefixes) for part in relative.parts) or path.name.endswith(SKIP_SUFFIXES):
            continue
        if path.is_symlink():
            raise CheckpointError(f"Unexpected symlink in benchmark evidence: {relative}")
        if path.is_file():
            yield relative.as_posix(), path


class SpotAttempt:
    def __init__(self, store, unit, instance_id, attempt, notice_file, *, stop_requested):
        self.store = store
        self.unit = unit
        self.instance_id = instance_id
        self.attempt = attempt
        self.notice_file = Path(notice_file) if notice_file else None
        self.signal_received = stop_requested
        self.grant = None
        self._cancelled = False
        self._last_poll = -math.inf

    @property
    def grant_key(self):
        return layout.spot_instance_key(self.unit.campaign_id, self.unit.run_id, self.attempt)

    def _poll_cancellation(self, *, force=False):
        if not self._cancelled and (force or time.monotonic() - self._last_poll >= 5):
            self._cancelled = self.store.head(layout.cancelled_key(self.unit.campaign_id)) is not None
            self._last_poll = time.monotonic()
        return self._cancelled

    def reason(self):
        if self._poll_cancellation():
            return "cancelled"
        if self.notice_file and self.notice_file.is_file():
            return "spot_interruption_notice"
        if self.signal_received():
            return "sigterm"
        return None

    def check_cancellation(self):
        if self._poll_cancellation(force=True):
            raise CampaignCancelled("Campaign cancellation marker exists")

    def check_publication(self, *, strict=False):
        if strict:
            self.check_cancellation()
            next_grant = layout.spot_instance_key(self.unit.campaign_id, self.unit.run_id, self.attempt + 1)
            if self.store.head(next_grant) is not None:
                raise SpotGrantError("A replacement instance already owns this run")
        reason = self.reason()
        if reason == "cancelled":
            raise CampaignCancelled("Campaign cancellation marker exists")
        if reason:
            raise InterruptedError("Instance interrupted before publishing benchmark evidence")

    def acquire(self, *, timeout=60):
        deadline = time.monotonic() + timeout
        while True:
            self.check_publication(strict=True)
            try:
                grant = get_json(self.store, self.grant_key)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise SpotGrantError("The control plane did not confirm the instance grant") from None
                time.sleep(1)
                continue
            if grant.get("instance_id") != self.instance_id or grant.get("attempt") != self.attempt:
                raise SpotGrantError("The instance grant belongs to another worker")
            # A new attempt only receives a grant once the predecessor is confirmed terminated.
            # This conditional create prevents two processes inside one grant from running at once.
            claim = self.grant_key.removesuffix("instance.json") + "worker.claim"
            try:
                self.store.put_bytes(claim, b"1", if_absent=True)
            except ObjectExists:
                raise SpotGrantError("The instance grant was already used by a worker") from None
            self.grant = grant
            return grant


class SpotCheckpoint:
    def __init__(self, store, unit, output, attempt, *, interval=60, requested, logger=None):
        if not math.isfinite(interval) or interval <= 0:
            raise CheckpointError("Checkpoint interval must be positive and finite")
        self.store = store
        self.output = Path(output)
        self.attempt = attempt
        self.interval = interval
        self.requested = requested
        self.log = logger or logging.getLogger(__name__)
        self.root = f"{layout.run_prefix(unit.campaign_id, unit.run_id)}/spot/checkpoints"
        self.manifests = self.root + "/manifests/"
        self.identity = {
            "run_id": unit.run_id,
            "campaign_id": unit.campaign_id,
            "contract_sha256": unit.contract_sha256,
            "release": unit.contract["release"],
            "snapshot": unit.contract.get("snapshot"),
            "source_sha256": unit.params.get("source_sha256"),
        }
        self.last_save = -math.inf
        self.sequence = 0
        self.receipt = None
        self.resumed = None
        self._state = None
        self._loaded = False
        self._file_cache = {}

    def due(self):
        return time.monotonic() - self.last_save >= self.interval or self.requested()

    def _blob(self, digest):
        return f"{self.root}/blobs/{digest}"

    def _store_file(self, path):
        stat = path.stat()
        fingerprint = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._file_cache.get(path)
        if cached and cached[0] == fingerprint:
            return cached[1]
        digest = sha256_path(path)
        key = self._blob(digest)
        stored = self.store.head(key)
        if stored is None:
            try:
                stored = self.store.put_file(key, path, if_absent=True, sha256=digest)
            except ObjectExists:
                stored = self.store.head(key)
        if stored is None or stored.sha256 != digest or stored.size != stat.st_size:
            raise CheckpointError(f"Corrupt checkpoint object {key}")
        reference = {"sha256": digest, "bytes": stat.st_size}
        # Keep entries for evidence paths. Scratch state files have unique names.
        if path.is_relative_to(self.output):
            self._file_cache[path] = fingerprint, reference
        return reference

    def _restore_file(self, reference, path):
        digest = reference["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CheckpointError("Invalid checkpoint object hash")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.store.download(self._blob(digest), path)
        if path.stat().st_size != reference["bytes"] or sha256_path(path) != digest:
            raise CheckpointError(f"Checkpoint object {digest} failed verification")

    def archive_files(self, *, private, check_stop=None):
        references = {}
        for name, path in output_files(self.output, private=private):
            if check_stop:
                check_stop()
            references[name] = self._store_file(path)
        return references

    def restore_files(self, references, directory, *, check_stop=None):
        for name, reference in references.items():
            if check_stop:
                check_stop()
            # An archive can only create children of the fresh output directory.
            self._restore_file(reference, directory / clean_key(name))

    def save(self, state, *, force=False):
        if not force and not self.due():
            return None
        if not isinstance(state, dict):
            raise CheckpointError("Evaluators must provide checkpoint state as a dict")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.output.parent) as temporary:
            path = Path(temporary) / "state.pickle"
            with path.open("wb") as stream:
                pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
            manifest = {
                "version": CHECKPOINT_VERSION,
                "identity": self.identity,
                "attempt": self.attempt,
                "state": self._store_file(path),
                "files": self.archive_files(private=True),
            }
        self.sequence += 1
        key = f"{self.manifests}{self.attempt:06d}/{self.sequence:08d}.json"
        confirmed, _ = put_json_once(self.store, key, manifest)
        self.receipt = confirmed.uri
        self.last_save = time.monotonic()
        self.log.info("checkpoint confirmed", extra={"fields": {"checkpoint": self.receipt}})
        return self.receipt

    def _restore_checkpoint(self, key, attempt):
        data = self.store.get_bytes(key)
        stored = self.store.head(key)
        if stored is None or stored.sha256 != sha256_bytes(data):
            raise CheckpointError("Checkpoint manifest checksum could not be verified")
        manifest = json.loads(data)
        if (
            manifest["version"] != CHECKPOINT_VERSION
            or manifest["identity"] != self.identity
            or manifest["attempt"] != attempt
        ):
            raise CheckpointError("Checkpoint manifest differs from the benchmark contract")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.output.parent) as temporary:
            staged = Path(temporary) / "output"
            staged.mkdir()
            self.restore_files(manifest["files"], staged)
            state_path = Path(temporary) / "state.pickle"
            self._restore_file(manifest["state"], state_path)
            # Only load same-release, checksum-verified state from the private benchmark bucket.
            with state_path.open("rb") as stream:
                state = pickle.load(stream)
            if not isinstance(state, dict):
                raise CheckpointError("Unexpected checkpoint state type")
            if self.output.is_dir() and not any(self.output.iterdir()):
                self.output.rmdir()
            if self.output.exists():
                raise CheckpointError("Checkpoint restore requires an empty benchmark output directory")
            staged.rename(self.output)
        return state

    def prepare_restore(self):
        if self._loaded:
            return
        candidates = []
        for key in self.store.list(self.manifests):
            match = _MANIFEST_SUFFIX.fullmatch(key.removeprefix(self.manifests))
            if match and (attempt := int(match["attempt"])) < self.attempt:
                candidates.append((key, attempt))
        for key, attempt in sorted(candidates, reverse=True):
            try:
                self._state = self._restore_checkpoint(key, attempt)
            except (ValueError, KeyError, TypeError, FileNotFoundError, EOFError, pickle.UnpicklingError):
                self.log.warning("could not verify checkpoint %s", self.store.uri(key), exc_info=True)
                continue
            self.receipt = self.resumed = self.store.uri(key)
            self.last_save = time.monotonic()
            self.log.info("checkpoint restored", extra={"fields": {"checkpoint": self.resumed}})
            break
        else:
            if candidates:
                raise CheckpointError("No previously confirmed checkpoint passed verification")
        self._loaded = True

    def load(self):
        self.prepare_restore()
        return self._state
