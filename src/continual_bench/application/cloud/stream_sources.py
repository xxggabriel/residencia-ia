"""Write-once prepared real-stream inputs for independent batch-prequential jobs.

The manifest is published last. The prepared Parquet hash identifies the source
and is fixed in the campaign contract together with the manifest hash. Workers
download and revalidate both; this module does not acquire remote raw datasets.
"""

import json
import re
from pathlib import Path

from continual_bench.datasets.pinned_streams import (
    PINNED_NAMES,
    PREPARATION,
    PinnedRealStreamProvider,
    expected_rows,
    source_contract,
)
from continual_bench.datasets.streams import input_paths, load_stream_manifest
from continual_bench.datasets.zia import FORMAT, SCOPES
from continual_bench.infrastructure.object_store import ObjectExists, sha256_bytes, sha256_path

MAX_MANIFEST_BYTES = 256 * 1024
MAX_PREPARED_BYTES = 512 * 1024**2


def _keys(dataset, scope, sha256):
    if dataset not in PINNED_NAMES or scope not in SCOPES or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("unsupported prepared stream source identity")
    prefix = f"datasets/stream-sources/{dataset}/{scope}/{sha256}"
    return f"{prefix}/prepared.parquet", f"{prefix}/manifest.json"


def _admit_manifest(manifest, dataset, scope, sha256):
    if (
        manifest.get("format") != FORMAT
        or manifest.get("provider") != PinnedRealStreamProvider.name
        or manifest.get("preparation_version") != PREPARATION
        or manifest.get("dataset") != dataset
        or manifest.get("scope") != scope
        or manifest.get("sha256") != sha256
        or manifest.get("rows") != expected_rows(dataset, scope)
        or manifest.get("source") != {**source_contract(dataset), "classes": ["0", "1"], "model": "xgboost"}
        or type(manifest.get("bytes")) is not int
        or not 0 < manifest["bytes"] <= MAX_PREPARED_BYTES
    ):
        raise ValueError("published stream manifest differs from the pinned preparation contract")


def _descriptor(store, dataset, scope, sha256, manifest_sha256, manifest):
    parquet_key, manifest_key = _keys(dataset, scope, sha256)
    return {
        "dataset": dataset,
        "scope": scope,
        "sha256": sha256,
        "prepared_sha256": sha256,
        "manifest_sha256": manifest_sha256,
        "rows": manifest["rows"],
        "bytes": manifest["bytes"],
        "uri": store.uri(parquet_key),
        "manifest_uri": store.uri(manifest_key),
    }


def _put_file_once(store, key, path, digest, metadata):
    try:
        store.put_file(key, path, if_absent=True, sha256=digest, metadata=metadata)
    except ObjectExists:
        head = store.head(key)
        if (
            head is None
            or head.sha256 != digest
            or head.size != path.stat().st_size
            or any(head.metadata.get(k) != v for k, v in metadata.items())
        ):
            raise


def publish_stream_source(store, root, dataset, scope):
    """Publish a locally admitted full/prefix pair without replacing earlier bytes."""
    path, manifest = load_stream_manifest(root, dataset, scope)
    manifest_path = input_paths(root, dataset, scope)[1]
    if any(p.is_symlink() for p in (path, manifest_path, path.parent)):
        raise ValueError("symlinked prepared stream input is not allowed")
    sha256 = sha256_path(path)
    parquet_key, manifest_key = _keys(dataset, scope, sha256)
    _admit_manifest(manifest, dataset, scope, sha256)
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("prepared stream manifest exceeds its byte limit")
    manifest_sha256 = sha256_path(manifest_path)
    metadata = {"format": PREPARATION, "dataset": dataset, "scope": scope, "rows": str(manifest["rows"])}
    _put_file_once(store, parquet_key, path, sha256, metadata)
    # A manifest is the final completion marker, so partial uploads cannot be admitted.
    _put_file_once(store, manifest_key, manifest_path, manifest_sha256, metadata)
    return verify_stream_source(store, dataset, scope, sha256)


def verify_stream_source(store, dataset, scope, sha256):
    """Verify immutable metadata and both objects before creating a campaign unit."""
    parquet_key, manifest_key = _keys(dataset, scope, sha256)
    head = store.head(manifest_key)
    prepared = store.head(parquet_key)
    if head is None or prepared is None or not 0 < head.size <= MAX_MANIFEST_BYTES:
        raise ValueError("prepared stream pair is missing or incomplete")
    raw_manifest = store.get_bytes(manifest_key)
    manifest_sha256 = sha256_bytes(raw_manifest)
    if len(raw_manifest) != head.size or manifest_sha256 != head.sha256:
        raise ValueError("prepared stream manifest SHA-256 mismatch")
    manifest = json.loads(raw_manifest)
    _admit_manifest(manifest, dataset, scope, sha256)
    metadata = {"format": PREPARATION, "dataset": dataset, "scope": scope, "rows": str(manifest["rows"])}
    if (
        prepared.sha256 != sha256
        or prepared.size != manifest["bytes"]
        or any(prepared.metadata.get(k) != v for k, v in metadata.items())
        or any(head.metadata.get(k) != v for k, v in metadata.items())
    ):
        raise ValueError("prepared stream object metadata differs from the pinned manifest")
    return _descriptor(store, dataset, scope, sha256, manifest_sha256, manifest)


def materialize_stream_source(store, dataset, scope, sha256, workdir):
    """Download to the provider layout and verify bytes, schema and full row count."""
    descriptor = verify_stream_source(store, dataset, scope, sha256)
    parquet_key, manifest_key = _keys(dataset, scope, sha256)
    root = Path(workdir)
    path, manifest_path = input_paths(root, dataset, scope)
    if any(p.is_symlink() for p in (root, path.parent, path, manifest_path)):
        raise ValueError("symlinked materialization destination is not allowed")
    root.mkdir(parents=True, exist_ok=True)
    store.download(parquet_key, path)
    store.download(manifest_key, manifest_path)
    if sha256_path(path) != sha256 or sha256_path(manifest_path) != descriptor["manifest_sha256"]:
        raise ValueError("downloaded prepared stream SHA-256 mismatch")
    loaded, manifest = load_stream_manifest(root, dataset, scope)
    if loaded != path or manifest["rows"] != descriptor["rows"]:
        raise ValueError("downloaded prepared stream row contract mismatch")
    return root
