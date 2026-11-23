"""Immutable publication recovery for metrics, without a training checkpoint."""

import re
from pathlib import Path

from continual_bench.evaluation.artifact_policy import verify_metrics_only_files
from continual_bench.infrastructure.object_store import ObjectExists, clean_key, sha256_path

from . import layout
from .checkpoints import CheckpointError, output_files


class MetricEvidenceArchive:
    """Archive completed metric evidence only; no evaluator state or resume API exists."""

    def __init__(self, store, unit, output):
        self.store = store
        self.output = Path(output)
        self.root = f"{layout.run_prefix(unit.campaign_id, unit.run_id)}/metrics-evidence/blobs"

    def archive_files(self, *, private=False, check_stop=None):
        if private:
            raise ValueError("metrics-only publication cannot archive private training state")
        verify_metrics_only_files(self.output)
        references = {}
        for name, path in output_files(self.output):
            if check_stop:
                check_stop()
            digest = sha256_path(path)
            key = f"{self.root}/{digest}"
            size = path.stat().st_size
            stored = self.store.head(key)
            if stored is None:
                try:
                    stored = self.store.put_file(key, path, if_absent=True, sha256=digest)
                except ObjectExists:
                    stored = self.store.head(key)
            if stored is None or stored.sha256 != digest or stored.size != size:
                raise CheckpointError("Immutable metric evidence failed checksum verification")
            references[name] = {"sha256": digest, "bytes": size}
        return references

    def restore_files(self, references, directory, *, check_stop=None):
        directory = Path(directory)
        for name, reference in references.items():
            if check_stop:
                check_stop()
            digest = reference.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise CheckpointError("Invalid metric evidence hash")
            path = directory / clean_key(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.store.download(f"{self.root}/{digest}", path)
            if path.stat().st_size != reference.get("bytes") or sha256_path(path) != digest:
                raise CheckpointError("Restored metric evidence failed checksum verification")
        verify_metrics_only_files(directory)
