"""One deterministic metric/provenance archive instead of many remote artifact requests."""

import hashlib
import json
import os
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

from .artifact_policy import verify_metrics_only_files

ARCHIVE_NAME = "metric-evidence.zip"
MANIFEST_NAME = "metric-evidence-manifest.json"
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".html"}
RAW_INPUTS = {"events.jsonl", "samples.jsonl", "continual-probes.json"}
PUBLICATION_RECEIPTS = {"benchmark-receipt.json", "receipt.json", MANIFEST_NAME}


def _info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info._compresslevel = 1
    return info


def build_metric_evidence_bundle(directory) -> dict:
    """Keep complete text evidence and hashes; exclude inputs, images and serialized state.

    Files are streamed, and source timestamps do not enter ZIP metadata. A source
    changing during packaging is rejected. The internal manifest lists each retained
    member's bytes and SHA-256; its own digest and the archive digest are in the receipt.
    """
    root = Path(directory)
    verify_metrics_only_files(root)
    selected, excluded = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError("metrics-only evidence cannot contain symlinks")
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        name = relative.as_posix()
        if path.name in RAW_INPUTS:
            excluded.append({"path": name, "reason": "raw_input_data"})
        elif path.name == ARCHIVE_NAME or (len(relative.parts) == 1 and path.name in PUBLICATION_RECEIPTS):
            continue
        elif path.suffix.lower() not in TEXT_SUFFIXES:
            excluded.append({"path": name, "reason": "text_evidence_only_static_rendering_deferred"})
        else:
            selected.append((relative.as_posix(), path))
    if not selected:
        raise ValueError("metrics-only evidence bundle has no metric or provenance files")
    with NamedTemporaryFile(dir=root, prefix=".pending-metric-evidence-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        members = []
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in selected:
                before = path.stat()
                digest, count = hashlib.sha256(), 0
                with path.open("rb") as source, archive.open(_info(name), "w", force_zip64=True) as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"metric evidence changed while packaging: {name}")
                members.append({"path": name, "bytes": count, "sha256": digest.hexdigest()})
            manifest = {
                "version": "metric-evidence-bundle-v1",
                "members": members,
                "excluded": excluded,
                "dashboard": "extract archive and open dashboard.html; no native artifact URL is implied",
                "retention": (
                    "metrics, provenance and verification records; no weights or training checkpoints"
                ),
            }
            payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            archive.writestr(_info(MANIFEST_NAME), payload, compresslevel=1)
        destination = root / ARCHIVE_NAME
        os.replace(temporary, destination)
        with destination.open("rb") as stream:
            archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "version": manifest["version"],
            "artifact_path": ARCHIVE_NAME,
            "sha256": archive_sha,
            "bytes": destination.stat().st_size,
            "member_count": len(members) + 1,
            "evidence_file_count": len(members),
            "evidence_bytes": sum(member["bytes"] for member in members),
            "manifest": {
                "path": MANIFEST_NAME,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "dashboard": "extract metric-evidence.zip and open dashboard.html",
        }
    finally:
        temporary.unlink(missing_ok=True)
