import io
import json
import urllib.error
import zipfile

import pytest

from continual_bench.datasets.temporal_vision import (
    DatasetAccessError,
    RangeReader,
    prepare_temporal_dataset,
)


def png():
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (12, 100, 210)).save(stream, format="PNG")
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self, data, start, stop, total, status=206, validator='"stable"'):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{stop}/{total}", "ETag": validator}


class RangeServer:
    def __init__(self, payload, *, status=206):
        self.payload, self.status, self.calls = payload, status, []

    def __call__(self, request, timeout):
        start, stop = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        self.calls.append((start, stop))
        return Response(self.payload[start:stop + 1], start, stop, len(self.payload), self.status)


def test_range_reader_is_seekable_budgeted_cached_and_corruption_checked(tmp_path):
    server = RangeServer(b"0123456789" * 100)
    reader = RangeReader("https://fixture.example/archive.zip", tmp_path, opener=server)
    reader.seek(-8, 2)
    assert reader.read(4) == b"2345"
    calls = len(server.calls)
    reader.seek(-8, 2)
    assert reader.read(4) == b"2345" and len(server.calls) == calls
    assert reader.provenance()["whole_archive_checksum_verified"] is False
    (reader.cache / "0-1.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="checksum"):
        RangeReader("https://fixture.example/archive.zip", tmp_path, opener=server)
    with pytest.raises(ValueError, match="budget"):
        RangeReader("https://fixture.example/large.zip", tmp_path, max_bytes=0, opener=server)


def test_range_reader_never_accepts_full_download_or_retries_denied_access(tmp_path):
    server = RangeServer(b"dataset", status=200)
    with pytest.raises(DatasetAccessError, match="exact byte range"):
        RangeReader("https://fixture.example/a.zip", tmp_path, opener=server)
    assert len(server.calls) == 1
    calls = []

    def denied(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, None)

    with pytest.raises(DatasetAccessError, match="403"):
        RangeReader("https://fixture.example/b.zip", tmp_path, opener=denied)
    assert len(calls) == 1


def test_core50_local_zip_preserves_official_session_splits_and_original_classes(tmp_path):
    archive = tmp_path / "core50.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for session in (1, 3):
            for frame in range(100):
                obj = frame % 2 + 1
                z.writestr(f"core50_128x128/s{session}/o{obj}/C_{session:02}_{obj:02}_{frame:03}.png", png())
    source = prepare_temporal_dataset("core50", tmp_path / "cache", local_archives={"train": archive})
    samples = list(source.samples_iter())
    assert len(samples) == 100 and len({s.sample_id for s in samples}) == 100
    assert [s.split for s in samples] == ["train"] * 80 + ["test"] * 20
    assert {s.y for s in samples} == {"0", "1"}
    assert {s.task_id for s in samples} == {0, 2}
    assert all(len(s.x) == 768 and all(0 <= value <= 1 for value in s.x.values()) for s in samples)
    assert source.manifest()["raw_archives"][0]["whole_archive_checksum_verified"] is False
    repeated = prepare_temporal_dataset("core50", tmp_path / "cache")
    assert samples == list(repeated.samples_iter())
    assert source.manifest() == repeated.manifest()


@pytest.mark.parametrize("name", ["clear10", "clear100"])
def test_clear_reads_official_bucket_class_metadata_without_training_test_images(tmp_path, name):
    archives = {}
    for split, total in (("train", 50), ("test", 20)):
        archive = tmp_path / f"{name}-{split}.zip"
        archives[split] = archive
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{split}/class_names.txt", "class_a\nclass_b\n")
            metadata = {}
            for bucket in range(2):
                metadata[str(bucket)] = {}
                for label in ("class_a", "class_b"):
                    relative = f"metadata/{bucket}_{label}.json"
                    metadata[str(bucket)][label] = relative
                    entries = {}
                    for i in range(total):
                        path = f"labeled_images/{bucket}/{label}/{i}.png"
                        entries[str(i)] = {"IMG_PATH": path}
                        z.writestr(f"{split}/{path}", png())
                    z.writestr(f"{split}/{relative}", json.dumps(entries))
            z.writestr(f"{split}/labeled_metadata.json", json.dumps(metadata))
    source = prepare_temporal_dataset(name, tmp_path / "cache", local_archives=archives)
    samples = list(source.samples_iter())
    assert len(samples) == 100
    assert sum(s.split == "train" for s in samples) == 80
    assert sum(s.split == "test" for s in samples) == 20
    assert {s.task_id for s in samples} == {0, 1}
    assert source.manifest()["release"].startswith("July-2022-")


def test_temporal_cache_only_mode_and_replay_integrity_fail_closed(tmp_path):
    with pytest.raises(DatasetAccessError, match="cache is absent"):
        prepare_temporal_dataset("clear10", tmp_path)
    folder = tmp_path / "core50"
    folder.mkdir()
    bad = {"samples": [], "manifest": {"samples_sha256": "bad"}}
    (folder / "prepared-100.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="checksum"):
        prepare_temporal_dataset("core50", tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        prepare_temporal_dataset("unrelated", tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        prepare_temporal_dataset("stream51", tmp_path)  # Removed: no reachable official source.
