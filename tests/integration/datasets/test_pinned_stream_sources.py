"""Full-source streaming preparation and immutable cloud input admission."""

import hashlib
import json
from dataclasses import replace

import pyarrow.parquet as pq
import pytest

from continual_bench.application.cloud.stream_sources import (
    materialize_stream_source,
    publish_stream_source,
    verify_stream_source,
)
from continual_bench.datasets import openml, real_streams, streams
from continual_bench.datasets.pinned_streams import (
    PINNED_NAMES,
    SMOKE_ROWS,
    PinnedRealStreamProvider,
    expected_rows,
)
from continual_bench.infrastructure.object_store import LocalObjectStore


@pytest.fixture
def pinned_raw(tmp_path, monkeypatch):
    """Pin a small but >10k-row source, preserving each family's original schema."""
    root = tmp_path / "raw"
    root.mkdir()
    count = 10_037
    for name, source in openml.REAL_SOURCES.items():
        spec = dict(source)
        header = ["@relation source"]
        for column in spec["columns"]:
            values = spec["labels"] if column == spec["target"] else ["A", "B"]
            kind = "{" + ",".join(values) + "}" if column in spec["nominal"] else "numeric"
            header.append(f"@attribute {column} {kind}")
        rows = []
        for seq in range(count):
            values = []
            for column in spec["columns"]:
                if column == spec["target"]:
                    values.append(spec["labels"][seq % 2])
                elif column in spec["nominal"]:
                    values.append("A" if seq % 2 == 0 else "B")
                else:
                    values.append(str(seq / 1000))
            rows.append(",".join(values))
        path = root / f"{name}.arff"
        path.write_text("\n".join([*header, "@data", *rows]) + "\n")
        spec.update(rows=count, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        monkeypatch.setitem(openml.REAL_SOURCES, name, spec)
    for name in real_streams.INSECTS_BINARY_VARIANTS:
        spec = dict(real_streams.SPECS[name])
        labels = ["11", "12", "2", "3", "4", "5"]
        rows = [",".join([*([str(seq / 1000)] * 33), labels[seq % 6]]) for seq in range(count)]
        path = root / spec["filename"]
        path.write_text("\n".join(rows) + "\n")
        spec.update(expected_rows=count, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        monkeypatch.setitem(real_streams.SPECS, name, spec)
    return root, count


@pytest.mark.parametrize("dataset", PINNED_NAMES)
def test_full_stream_uses_all_original_rows_without_fitting_a_transform(tmp_path, pinned_raw, dataset):
    raw, count = pinned_raw
    provider = PinnedRealStreamProvider(raw)
    first = provider.prepare(tmp_path / "first", dataset, "full")
    second = provider.prepare(tmp_path / "second", dataset, "full")
    assert first == second and first["rows"] == count > 10_000
    assert first["selection"] == {"kind": "complete_source", "seq_range": [0, count]}
    path, loaded = provider.load_manifest(tmp_path / "first", dataset, "full")
    assert loaded == first
    table = pq.read_table(path)
    assert table["event_time"].to_pylist() == list(range(count))
    assert table["label"].type.bit_width == 8
    assert first["source"]["classes"] == ["0", "1"]
    assert first["real_event_timestamps_known"] is False
    assert first["real_label_availability_known"] is False
    assert "end of completed" in first["label_availability"]
    if dataset in openml.REAL_SOURCES:
        spec = openml.REAL_SOURCES[dataset]
        expected = [int(spec["labels"][seq % 2] == spec["positive"]) for seq in range(count)]
        assert first["feature_encoding"]["nominal"]
    else:
        expected = [int(seq % 6 == 2) for seq in range(count)]
    assert table["label"].to_pylist() == expected
    assert provider.prepare(tmp_path / "first", dataset, "full") == first


@pytest.mark.parametrize("dataset", PINNED_NAMES)
def test_smoke_keeps_published_prefix_and_fixed_source_metadata(tmp_path, pinned_raw, dataset):
    raw, count = pinned_raw
    provider = PinnedRealStreamProvider(raw)
    manifest = provider.prepare(tmp_path / "prepared", dataset, "smoke")
    assert manifest["rows"] == SMOKE_ROWS < count
    assert manifest["source"]["raw_rows"] == count
    assert manifest["selection"] == {"kind": "published_prefix", "seq_range": [0, SMOKE_ROWS]}
    assert expected_rows(dataset, "full") == count
    assert provider.spec(dataset).block_rows["full"] >= 1_000
    assert dataset in streams.stream_names()
    assert streams.stream_spec(dataset).provider == provider.name


def test_pinned_preparation_refuses_modified_raw_and_symlinks(tmp_path, pinned_raw):
    raw, _ = pinned_raw
    source = raw / "electricity.arff"
    source.write_text(source.read_text().replace("0.001", "0.002", 1))
    with pytest.raises(ValueError, match="SHA-256"):
        PinnedRealStreamProvider(raw).prepare(tmp_path / "out", "electricity", "full")
    source.unlink()
    source.symlink_to(raw / "airlines.arff")
    with pytest.raises(ValueError, match="symlinked"):
        PinnedRealStreamProvider(raw).prepare(tmp_path / "out", "electricity", "full")


def test_pinned_prepared_pair_rejects_tampered_count_and_payload(tmp_path, pinned_raw):
    raw, _ = pinned_raw
    provider = PinnedRealStreamProvider(raw)
    root = tmp_path / "prepared"
    manifest = provider.prepare(root, "electricity", "full")
    path, manifest_path = streams.input_paths(root, "electricity", "full")
    manifest_path.write_text(json.dumps({**manifest, "rows": manifest["rows"] - 1}))
    with pytest.raises(ValueError, match="contract mismatch"):
        provider.load_manifest(root, "electricity", "full")
    manifest_path.write_text(json.dumps(manifest))
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)
    with pytest.raises((ValueError, OSError)):
        provider.load_manifest(root, "electricity", "full")


@pytest.mark.parametrize("scope", ["smoke", "full"])
def test_cloud_stream_pair_is_write_once_and_materialized_with_both_hashes(tmp_path, pinned_raw, scope):
    raw, count = pinned_raw
    provider = PinnedRealStreamProvider(raw)
    root = tmp_path / "prepared"
    manifest = provider.prepare(root, "electricity", scope)
    store = LocalObjectStore(tmp_path / "store")
    descriptor = publish_stream_source(store, root, "electricity", scope)
    assert descriptor["prepared_sha256"] == manifest["sha256"] == descriptor["sha256"]
    assert descriptor["rows"] == (count if scope == "full" else SMOKE_ROWS)
    assert "/datasets/stream-sources/electricity/" in descriptor["uri"]
    assert publish_stream_source(store, root, "electricity", scope) == descriptor
    assert verify_stream_source(store, "electricity", scope, descriptor["sha256"]) == descriptor
    copied = materialize_stream_source(store, "electricity", scope, descriptor["sha256"], tmp_path / "copied")
    assert copied == tmp_path / "copied"
    assert streams.load_stream_manifest(copied, "electricity", scope)[1] == manifest


def test_cloud_stream_requires_complete_pair_and_rejects_replaced_bytes(tmp_path, pinned_raw):
    raw, _ = pinned_raw
    root = tmp_path / "prepared"
    manifest = PinnedRealStreamProvider(raw).prepare(root, "electricity", "full")
    store = LocalObjectStore(tmp_path / "store")
    with pytest.raises(ValueError, match="missing or incomplete"):
        verify_stream_source(store, "electricity", "full", manifest["sha256"])
    published = publish_stream_source(store, root, "electricity", "full")
    key = next(k for k in store.list() if k.endswith("prepared.parquet"))
    target = store.root / key
    target.write_bytes(b"x" * target.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_stream_source(store, "electricity", "full", manifest["sha256"], tmp_path / "copy")
    assert published["manifest_sha256"]


def test_cloud_source_admits_s3_adapters_checksum_metadata(tmp_path, pinned_raw):
    class S3MetadataStore(LocalObjectStore):
        def head(self, key):
            entry = super().head(key)
            return replace(entry, metadata={**entry.metadata, "sha256": entry.sha256}) if entry else None

    raw, _ = pinned_raw
    root = tmp_path / "prepared"
    PinnedRealStreamProvider(raw).prepare(root, "electricity", "full")
    store = S3MetadataStore(tmp_path / "store")
    first = publish_stream_source(store, root, "electricity", "full")
    assert publish_stream_source(store, root, "electricity", "full") == first
    manifest_key = next(key for key in store.list() if key.endswith("manifest.json"))
    metadata_path = store.meta / (manifest_key + ".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["rows"] = "1"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata differs"):
        verify_stream_source(store, "electricity", "full", first["sha256"])


@pytest.mark.parametrize(
    "dataset,scope,digest",
    [("../bad", "full", "a" * 64), ("electricity", "../bad", "a" * 64), ("electricity", "full", "../bad")],
)
def test_cloud_stream_rejects_unregistered_identity(tmp_path, dataset, scope, digest):
    with pytest.raises(ValueError, match="identity"):
        verify_stream_source(LocalObjectStore(tmp_path), dataset, scope, digest)


def test_cloud_stream_rejects_symlink_materialization_destination(tmp_path, pinned_raw):
    raw, _ = pinned_raw
    root = tmp_path / "prepared"
    manifest = PinnedRealStreamProvider(raw).prepare(root, "electricity", "full")
    store = LocalObjectStore(tmp_path / "store")
    publish_stream_source(store, root, "electricity", "full")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        materialize_stream_source(store, "electricity", "full", manifest["sha256"], link)
