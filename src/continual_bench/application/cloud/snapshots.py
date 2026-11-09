"""Dataset snapshots in object storage: acquire/import raw, prepare, publish, verify, materialize.

Evidence is write-once. A retry reuses every object whose recorded SHA-256 matches the catalog
and refuses to replace a mismatching one. The completion marker is written last, after the
manifest and every referenced object were verified.
"""

import hashlib
import json
import shutil
from pathlib import Path

from continual_bench.datasets.zia import DATASETS, FORMAT, SMOKE_ROWS, load_manifest, prepare
from continual_bench.infrastructure.object_store import ObjectExists, get_json, put_json_once, sha256_path

from . import layout
from .contracts import SnapshotRef, canonical
from .sources import download, source_files

SNAPSHOT_VERSION = "snapshot-v1"
LICENSE = "CC-BY-NC-4.0"


class SnapshotNotReady(RuntimeError):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"snapshot {reason}: {detail}")
        self.reason, self.detail = reason, detail


def transform_code_sha256() -> str:
    import continual_bench.datasets.zia as module

    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _put_file_once(store, key, path, sha256, metadata):
    """Upload unless an identical object exists; never overwrite a different one."""
    existing = store.head(key)
    if existing is not None:
        if existing.sha256 != sha256:
            raise ObjectExists(f"{key} holds sha256={existing.sha256}, refusing to overwrite with {sha256}")
        return existing, False
    try:
        return store.put_file(key, path, if_absent=True, sha256=sha256, metadata=metadata), True
    except ObjectExists:
        existing = store.head(key)
        if existing is None or existing.sha256 != sha256:
            raise
        return existing, False


def _record(stored, **extra):
    return {
        "key": stored.key,
        "uri": stored.uri,
        "version_id": stored.version_id,
        "sha256": stored.sha256,
        "bytes": stored.size,
        **extra,
    }


# Acquisition ---------------------------------------------------------------------------------------
def acquire(
    store, snapshot: SnapshotRef, workdir, *, mode="download", import_uri=None, fetch=download, importer=None
) -> dict:
    """Place the pinned raw files in storage (download over HTTPS or import an S3 copy)."""
    workdir = Path(workdir)
    files = []
    for source in source_files(snapshot.dataset, snapshot.revision):
        key = layout.raw_key(snapshot.dataset, snapshot.revision, source.name)
        existing = store.head(key)
        if existing is not None and existing.sha256 == source.sha256:
            files.append(_record(existing, name=source.name, source_url=source.url, retrieval="reused"))
            continue
        if existing is not None:
            raise ObjectExists(f"{key} exists with a different sha256; refusing to overwrite raw evidence")
        local = workdir / "raw" / source.name
        if mode == "download":
            evidence = fetch(source, local)
        elif mode == "import":
            if importer is None or not import_uri:
                raise ValueError("import requires an importer and import_uri")
            importer(import_uri.rstrip("/") + "/" + source.name, local)
            evidence = {"sha256": sha256_path(local), "bytes": local.stat().st_size}
            if evidence["sha256"] != source.sha256:
                raise ValueError(f"imported {source.name} differs from the catalog checksum")
        else:
            raise ValueError(f"raw files are absent and mode={mode} cannot acquire them")
        stored, _ = _put_file_once(
            store,
            key,
            local,
            source.sha256,
            {"dataset": snapshot.dataset, "revision": snapshot.revision, "source": source.name},
        )
        files.append(
            _record(stored, name=source.name, source_url=source.url, retrieval=mode, download=evidence)
        )
        local.unlink(missing_ok=True)
    manifest = {
        "version": "raw-v1",
        "dataset": snapshot.dataset,
        "revision": snapshot.revision,
        "source": DATASETS[snapshot.dataset],
        "license": LICENSE,
        "files": [{k: f[k] for k in ("name", "key", "sha256", "bytes", "source_url")} for f in files],
    }
    key = layout.raw_manifest_key(snapshot.dataset, snapshot.revision)
    _, created = put_json_once(store, key, manifest)
    return {"raw_manifest_uri": store.uri(key), "created": created, "files": files}


def verify_raw(store, snapshot: SnapshotRef) -> dict:
    key = layout.raw_manifest_key(snapshot.dataset, snapshot.revision)
    if store.head(key) is None:
        raise SnapshotNotReady("missing", f"raw manifest {key}")
    manifest = get_json(store, key)
    expected = {s.name: s.sha256 for s in source_files(snapshot.dataset, snapshot.revision)}
    recorded = {f["name"]: f for f in manifest["files"]}
    if set(recorded) != set(expected):
        raise SnapshotNotReady("incompatible", "raw manifest lists other files than the catalog")
    for name, sha in expected.items():
        head = store.head(recorded[name]["key"])
        if head is None:
            raise SnapshotNotReady("incomplete", f"raw object missing: {name}")
        if head.sha256 != sha or recorded[name]["sha256"] != sha or head.size != recorded[name]["bytes"]:
            raise SnapshotNotReady("incompatible", f"raw object differs from the catalog: {name}")
    return manifest


# Preparation ---------------------------------------------------------------------------------------
def prepared_status(store, snapshot: SnapshotRef):
    manifest_key = layout.prepared_key(snapshot, "json")
    parquet_key = layout.prepared_key(snapshot, "parquet")
    if store.head(manifest_key) is None:
        return None
    manifest = get_json(store, manifest_key)
    head = store.head(parquet_key)
    if head is None or head.sha256 != manifest["sha256"] or head.size != manifest["bytes"]:
        raise SnapshotNotReady("incompatible", "prepared manifest does not describe the stored Parquet")
    return manifest


def prepare_snapshot(
    store, snapshot: SnapshotRef, workdir, *, image_digest: str | None = None, prepare_fn=prepare
) -> dict:
    """Read raw from storage, run the project's `prepare()` and publish the prepared pair once."""
    workdir = Path(workdir)
    existing = prepared_status(store, snapshot)
    if existing is not None:
        return {"prepared": existing, "reused": True}
    raw = verify_raw(store, snapshot)
    for item in raw["files"]:
        target = workdir / "raw" / item["name"]
        if not target.is_file() or sha256_path(target) != item["sha256"]:
            store.download(item["key"], target)
    manifest = prepare_fn(workdir, snapshot.dataset, snapshot.scope)
    parquet, validated = load_manifest(workdir, snapshot.dataset, snapshot.scope)
    if validated != manifest or sha256_path(parquet) != manifest["sha256"]:
        raise ValueError("prepared output failed the project's input contract")
    expected_rows = SMOKE_ROWS if snapshot.scope == "smoke" else DATASETS[snapshot.dataset]["rows"]
    if manifest["rows"] != expected_rows or manifest["format"] != FORMAT:
        raise ValueError("prepared row count or format differs from the catalog")
    stored, _ = _put_file_once(
        store,
        layout.prepared_key(snapshot, "parquet"),
        parquet,
        manifest["sha256"],
        {"dataset": snapshot.dataset, "scope": snapshot.scope, "format": FORMAT},
    )
    put_json_once(
        store,
        layout.prepared_key(snapshot, "provenance.json"),
        {
            "image_digest": image_digest,
            "transform_code_sha256": transform_code_sha256(),
            "prepared_sha256": manifest["sha256"],
            "schema": parquet_schema(parquet),
        },
    )
    # The project's manifest bytes are published verbatim (load_manifest compares them as JSON).
    manifest_path = parquet.with_suffix(".json")
    _put_file_once(
        store,
        layout.prepared_key(snapshot, "json"),
        manifest_path,
        sha256_path(manifest_path),
        {"dataset": snapshot.dataset, "scope": snapshot.scope},
    )
    return {"prepared": manifest, "reused": False, "parquet": _record(stored)}


def parquet_schema(path) -> list[dict]:
    import pyarrow.parquet as pq

    return [{"name": field.name, "type": str(field.type)} for field in pq.read_schema(path)]


# Publication and verification ----------------------------------------------------------------------
def publish_snapshot(store, snapshot: SnapshotRef) -> dict:
    raw = verify_raw(store, snapshot)
    prepared = prepared_status(store, snapshot)
    if prepared is None:
        raise SnapshotNotReady("incomplete", "prepared objects are not published")
    parquet = store.head(layout.prepared_key(snapshot, "parquet"))
    zia_manifest = store.head(layout.prepared_key(snapshot, "json"))
    provenance_key = layout.prepared_key(snapshot, "provenance.json")
    provenance = get_json(store, provenance_key) if store.head(provenance_key) else {}
    if prepared["source"] != DATASETS[snapshot.dataset] or prepared["scope"] != snapshot.scope:
        raise SnapshotNotReady("incompatible", "prepared manifest belongs to another source or scope")
    manifest = {
        "version": SNAPSHOT_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot": snapshot.model_dump(),
        "license": prepared.get("license", LICENSE),
        "format": prepared["format"],
        "rows": prepared["rows"],
        "features": prepared["features"],
        "schema": provenance.get("schema"),
        "ordering": ["event_time", "event_id"],
        "raw": {
            "manifest_uri": store.uri(layout.raw_manifest_key(snapshot.dataset, snapshot.revision)),
            "files": raw["files"],
        },
        "prepared": {
            "parquet": _record(parquet),
            "manifest": _record(zia_manifest),
            "sha256": prepared["sha256"],
            "bytes": prepared["bytes"],
        },
        "transform": {
            "version": snapshot.preparation_version,
            "code_sha256": provenance.get("transform_code_sha256"),
            "image_digest": provenance.get("image_digest"),
        },
    }
    manifest_key = layout.snapshot_manifest_key(snapshot)
    stored, created = put_json_once(store, manifest_key, manifest)
    marker = {
        "snapshot_id": snapshot.snapshot_id,
        "manifest_key": manifest_key,
        "manifest_sha256": hashlib.sha256(store.get_bytes(manifest_key)).hexdigest(),
        "manifest_version_id": stored.version_id,
        "prepared_sha256": prepared["sha256"],
    }
    put_json_once(store, layout.snapshot_marker_key(snapshot), marker)
    return {
        "manifest_uri": store.uri(manifest_key),
        "marker_uri": store.uri(layout.snapshot_marker_key(snapshot)),
        "created": created,
        "snapshot_id": snapshot.snapshot_id,
        "manifest": manifest,
    }


def verify_snapshot(store, snapshot: SnapshotRef) -> dict:
    """Benchmark preflight: absent, incomplete or incompatible snapshots stop the campaign."""
    marker_key, manifest_key = layout.snapshot_marker_key(snapshot), layout.snapshot_manifest_key(snapshot)
    if store.head(manifest_key) is None and store.head(marker_key) is None:
        raise SnapshotNotReady("missing", snapshot.snapshot_id)
    if store.head(marker_key) is None:
        raise SnapshotNotReady("incomplete", f"completion marker absent for {snapshot.snapshot_id}")
    marker = get_json(store, marker_key)
    data = store.get_bytes(manifest_key)
    if hashlib.sha256(data).hexdigest() != marker["manifest_sha256"]:
        raise SnapshotNotReady("incompatible", "snapshot manifest changed after its marker")
    manifest = json.loads(data)
    if (
        manifest.get("version") != SNAPSHOT_VERSION
        or manifest["snapshot"] != snapshot.model_dump()
        or manifest["format"] != FORMAT
        or manifest["prepared"]["sha256"] != marker["prepared_sha256"]
    ):
        raise SnapshotNotReady("incompatible", "snapshot manifest does not match the request")
    expected_rows = SMOKE_ROWS if snapshot.scope == "smoke" else DATASETS[snapshot.dataset]["rows"]
    if manifest["rows"] != expected_rows:
        raise SnapshotNotReady("incompatible", "row count differs from the catalog scope")
    head = store.head(manifest["prepared"]["parquet"]["key"])
    if head is None:
        raise SnapshotNotReady("incomplete", "prepared Parquet missing")
    if head.sha256 != manifest["prepared"]["sha256"] or head.size != manifest["prepared"]["bytes"]:
        raise SnapshotNotReady("incompatible", "prepared Parquet differs from the manifest")
    return {**manifest, "manifest_sha256": marker["manifest_sha256"], "manifest_uri": store.uri(manifest_key)}


def materialize(store, manifest: dict, root) -> Path:
    """Download the verified pair into the layout `load_manifest()` reads, checking SHA-256 locally."""
    root = Path(root)
    snapshot = SnapshotRef(**manifest["snapshot"])
    folder = root / snapshot.dataset
    parquet, zia = folder / f"{snapshot.scope}.parquet", folder / f"{snapshot.scope}.json"
    if not (parquet.is_file() and sha256_path(parquet) == manifest["prepared"]["sha256"]):
        store.download(manifest["prepared"]["parquet"]["key"], parquet)
    store.download(manifest["prepared"]["manifest"]["key"], zia)
    if sha256_path(parquet) != manifest["prepared"]["sha256"]:
        shutil.rmtree(folder)
        raise ValueError("downloaded snapshot differs from its manifest")
    load_manifest(root, snapshot.dataset, snapshot.scope)
    return root


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(canonical(manifest)).hexdigest()
