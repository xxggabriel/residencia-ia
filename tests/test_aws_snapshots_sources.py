"""Catalog-driven acquisition, bounded downloads and write-once snapshot publication."""

import hashlib
import io
import json
import urllib.error

import pytest

from continual_bench.application.cloud import layout, snapshots
from continual_bench.application.cloud.contracts import SnapshotRef
from continual_bench.application.cloud.fixtures import fixture_parquet
from continual_bench.application.cloud.sources import SourceFile, download, host_allowed, source_files
from continual_bench.datasets.zia import DATASETS
from continual_bench.infrastructure.object_store import LocalObjectStore, ObjectExists

SOCIAL = DATASETS["social"]["revision"]


def test_sources_come_from_the_catalog_only():
    files = source_files("social", SOCIAL)
    assert [f.name for f in files] == [DATASETS["social"]["file"], DATASETS["social"]["nodes"]]
    assert all(
        f.url.startswith(
            f"https://huggingface.co/datasets/ziadatalabs/FreeSocialNetwork100M/resolve/{SOCIAL}/"
        )
        for f in files
    )
    with pytest.raises(ValueError):
        source_files("social", "0" * 40)
    assert host_allowed("https://cdn-lfs-us-1.hf.co/x") and host_allowed("https://huggingface.co/x")
    assert not host_allowed("http://huggingface.co/x") and not host_allowed(
        "https://huggingface.co.evil.io/x"
    )


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_resumes_with_range_and_verifies_sha(tmp_path):
    payload = b"x" * 1000
    source = SourceFile("file.parquet", "https://huggingface.co/f", hashlib.sha256(payload).hexdigest())
    calls = []

    def opener(request, timeout):
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        start = (
            int(request.get_header("Range").split("=")[1].rstrip("-")) if request.get_header("Range") else 0
        )
        return FakeResponse(payload[start:], 206 if start else 200)

    destination = tmp_path / "raw" / "file.parquet"
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "file.parquet.partial").write_bytes(payload[:400])
    result = download(source, destination, opener=opener, sleep=lambda s: None)
    assert destination.read_bytes() == payload
    assert result["resumed_bytes"] == 400 and calls[-1] == "bytes=400-"
    assert download(source, destination, opener=opener)["reused"] is True


def test_download_refuses_checksum_mismatch_size_and_foreign_urls(tmp_path):
    source = SourceFile("f", "https://huggingface.co/f", "0" * 64, max_bytes=10)
    with pytest.raises(ValueError, match="limit"):
        download(source, tmp_path / "f", opener=lambda r, timeout: FakeResponse(b"y" * 11))
    wrong = SourceFile("f", "https://huggingface.co/f", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        download(wrong, tmp_path / "g", opener=lambda r, timeout: FakeResponse(b"y"))
    assert not (tmp_path / "g.partial").exists()
    with pytest.raises(ValueError, match="allowlist"):
        download(SourceFile("f", "https://example.com/f", "0" * 64), tmp_path / "h")


def test_redirect_handler_refuses_hosts_outside_the_source():
    from continual_bench.application.cloud.sources import _AllowlistRedirects

    handler = _AllowlistRedirects(("huggingface.co", "hf.co"))
    import urllib.request

    request = urllib.request.Request("https://huggingface.co/a")
    with pytest.raises(urllib.error.HTTPError, match="allowlist"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://evil.example/b")
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://cdn-lfs.hf.co/b") is not None


@pytest.fixture
def fake_catalog(monkeypatch):
    raw = {"edges.parquet": b"edges-bytes", "nodes.parquet": b"nodes-bytes"}
    files = [
        SourceFile(name, f"https://huggingface.co/{name}", hashlib.sha256(data).hexdigest())
        for name, data in raw.items()
    ]
    monkeypatch.setattr(snapshots, "source_files", lambda dataset, revision: files)
    return raw


def fetcher(raw):
    def fetch(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw[source.name])
        return {"sha256": source.sha256, "bytes": len(raw[source.name])}

    return fetch


def prepare_fixture(root, dataset, scope):
    return fixture_parquet(root / dataset / f"{scope}.parquet", dataset)


def test_snapshot_pipeline_publishes_marker_last_and_reuses_evidence(tmp_path, fake_catalog):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = SnapshotRef(dataset="social", revision=SOCIAL, scope="smoke")
    with pytest.raises(snapshots.SnapshotNotReady, match="missing"):
        snapshots.verify_snapshot(store, snapshot)
    first = snapshots.acquire(store, snapshot, tmp_path / "work", fetch=fetcher(fake_catalog))
    assert first["created"] and {f["retrieval"] for f in first["files"]} == {"download"}
    again = snapshots.acquire(store, snapshot, tmp_path / "work", fetch=lambda *a: pytest.fail("refetched"))
    assert not again["created"] and {f["retrieval"] for f in again["files"]} == {"reused"}
    with pytest.raises(snapshots.SnapshotNotReady, match="incomplete"):
        snapshots.publish_snapshot(store, snapshot)
    prepared = snapshots.prepare_snapshot(
        store, snapshot, tmp_path / "prep", image_digest="sha256:" + "1" * 64, prepare_fn=prepare_fixture
    )
    assert not prepared["reused"]
    assert snapshots.prepare_snapshot(store, snapshot, tmp_path / "prep2", prepare_fn=None)["reused"]
    assert store.head(layout.snapshot_marker_key(snapshot)) is None
    published = snapshots.publish_snapshot(store, snapshot)
    assert published["created"] and published["manifest"]["transform"]["image_digest"] == "sha256:" + "1" * 64
    assert [c["name"] for c in published["manifest"]["schema"]] == ["event_time", "label", "feature"]
    assert not snapshots.publish_snapshot(store, snapshot)["created"]
    manifest = snapshots.verify_snapshot(store, snapshot)
    root = snapshots.materialize(store, manifest, tmp_path / "data")
    assert (root / "social" / "smoke.parquet").is_file()


def test_snapshot_preflight_detects_incomplete_and_incompatible(tmp_path, fake_catalog):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = SnapshotRef(dataset="social", revision=SOCIAL, scope="smoke")
    snapshots.acquire(store, snapshot, tmp_path / "w", fetch=fetcher(fake_catalog))
    snapshots.prepare_snapshot(store, snapshot, tmp_path / "p", prepare_fn=prepare_fixture)
    snapshots.publish_snapshot(store, snapshot)
    marker = store.root / layout.snapshot_marker_key(snapshot)
    saved = marker.read_bytes()
    marker.unlink()
    with pytest.raises(snapshots.SnapshotNotReady, match="incomplete"):
        snapshots.verify_snapshot(store, snapshot)
    marker.write_bytes(saved)
    manifest_path = store.root / layout.snapshot_manifest_key(snapshot)
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "rows": 5}))
    with pytest.raises(snapshots.SnapshotNotReady, match="incompatible"):
        snapshots.verify_snapshot(store, snapshot)


def test_raw_evidence_is_never_overwritten(tmp_path, fake_catalog):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = SnapshotRef(dataset="social", revision=SOCIAL, scope="smoke")
    store.put_bytes(layout.raw_key("social", SOCIAL, "edges.parquet"), b"tampered", if_absent=True)
    with pytest.raises(ObjectExists):
        snapshots.acquire(store, snapshot, tmp_path / "w", fetch=fetcher(fake_catalog))


def test_import_mode_verifies_catalog_checksums(tmp_path, fake_catalog):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = SnapshotRef(dataset="social", revision=SOCIAL, scope="smoke")

    def importer(uri, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(fake_catalog[uri.rsplit("/", 1)[1]])

    result = snapshots.acquire(
        store, snapshot, tmp_path / "w", mode="import", import_uri="s3://b/imports/social", importer=importer
    )
    assert {f["retrieval"] for f in result["files"]} == {"import"}
    other = SnapshotRef(dataset="social", revision=SOCIAL, scope="full")
    store2 = LocalObjectStore(tmp_path / "store2")
    with pytest.raises(ValueError, match="catalog checksum"):
        snapshots.acquire(
            store2,
            other,
            tmp_path / "w2",
            mode="import",
            import_uri="s3://b/x",
            importer=lambda uri, d: (d.parent.mkdir(parents=True, exist_ok=True), d.write_bytes(b"?")),
        )
    with pytest.raises(ValueError, match="cannot acquire"):
        snapshots.acquire(LocalObjectStore(tmp_path / "s3"), other, tmp_path / "w3", mode="prepare")
