import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from continual_bench.datasets import cifar_parquet as module


def png(mode="RGB", size=(32, 32), color=(10, 20, 30)):
    stream = io.BytesIO()
    Image.new(mode, size, color=color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def mirror(monkeypatch):
    calls, files = [], {}
    for name in module.SOURCES:
        source = json.loads(json.dumps(module.SOURCES[name]))
        for split, count in (("train", 100), ("test", 30)):
            rows = [{"img": {"bytes": png(), "path": None},
                     source["label"]: i % source["classes"]} for i in range(count)]
            sink = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(rows), sink, row_group_size=10)
            payload = sink.getvalue()
            source["splits"][split].update(bytes=len(payload), rows=count,
                                           lfs_sha256=hashlib.sha256(payload).hexdigest())
            files[(name, split)] = payload
        monkeypatch.setitem(module.SOURCES, name, source)

    class Reader(io.BytesIO):
        def __init__(self, url, cache, max_bytes):
            name = "split_cifar100" if "/cifar100/" in url else "split_cifar10"
            split = "train" if "/train-" in url else "test"
            payload = files[(name, split)]
            if len(payload) > max_bytes:
                raise ValueError("fixture exceeds shared range budget")
            super().__init__(payload)
            self.url, self.size, self.transferred = url, len(payload), len(payload)
            calls.append(url)

        def provenance(self):
            return {"url": self.url, "archive_bytes": self.size, "validator": "fixture-etag",
                    "ranges_bytes": self.transferred, "ranges": [],
                    "whole_archive_checksum_verified": False, "integrity": "fixture"}

    monkeypatch.setattr(module, "RangeReader", Reader)
    return calls, files, Reader


def test_rgb_features_follow_cifar_channel_planar_order():
    values = module._pixels(png())
    assert len(values) == 3072
    assert values["pixel_0000"] == values["pixel_1023"] == 10 / 255
    assert values["pixel_1024"] == values["pixel_2047"] == 20 / 255
    assert values["pixel_2048"] == values["pixel_3071"] == 30 / 255
    with pytest.raises(ValueError, match="RGB 32x32"):
        module._pixels(png(mode="L", color=10))
    with pytest.raises(ValueError, match="RGB 32x32"):
        module._pixels(png(size=(16, 16)))
    with pytest.raises(ValueError, match="embedded"):
        module._pixels(None)


@pytest.mark.parametrize("name", ["split_cifar10", "split_cifar100"])
def test_prepare_keeps_100_unique_mirror_originals_splits_classes_and_tasks(tmp_path, mirror, name):
    calls, _, _ = mirror
    source = module.prepare_cifar_parquet(name, tmp_path, download=True)
    rows = list(source.samples_iter())
    assert len(rows) == len({r.sample_id for r in rows}) == 100
    assert sum(r.split == "train" for r in rows) == 80
    assert sum(r.split == "test" for r in rows) == 20
    divisor = module.SOURCES[name]["classes_per_task"]
    assert all(r.task_id == int(r.y) // divisor for r in rows)
    assert all(len(r.x) == 3072 and r.x["pixel_1024"] == 20 / 255 for r in rows)
    assert [r.seq for r in rows] == list(range(100))
    manifest = source.manifest()
    assert manifest["source_version"] == module.SOURCES[name]["revision"]
    assert manifest["whole_parquet_lfs_checksums_verified"] is False
    assert manifest["natural_chronology"] is False and manifest["changepoints"] is None
    assert len(manifest["selected_images"]) == 100
    assert len(calls) == 2 and all("/resolve/" in url for url in calls)
    assert all("?" not in item["url"] for item in manifest["parquet_sources"])
    assert all(item["whole_archive_checksum_verified"] is False for item in manifest["parquet_sources"])
    assert "LFS hash unverified" in manifest["parquet_sources"][0]["integrity"]
    cached = module.prepare_cifar_parquet(name, tmp_path, download=False)
    assert rows == list(cached.samples_iter()) and cached.manifest() == manifest
    assert len(calls) == 2


def test_explicit_acquisition_cache_corruption_and_no_network_repair(tmp_path, mirror):
    calls, _, _ = mirror
    with pytest.raises(FileNotFoundError, match="explicit download=True"):
        module.prepare_cifar_parquet("split_cifar10", tmp_path)
    module.prepare_cifar_parquet("split_cifar10", tmp_path, download=True)
    samples_path = next((tmp_path / "cifar-parquet/prepared").glob("*/samples.jsonl"))
    samples_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module.prepare_cifar_parquet("split_cifar10", tmp_path, download=True)
    assert len(calls) == 2


def test_parquet_rejects_unexpected_pinned_size_count_and_class(tmp_path, mirror):
    _, _, reader_factory = mirror
    source = module.SOURCES["split_cifar10"]
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/train-x.parquet"
    reader = reader_factory(url, tmp_path, max_bytes=64 * 1024**2)
    with pytest.raises(ValueError, match="size changed"):
        list(module._parquet_prefix(reader, {**source["splits"]["train"], "bytes": 1}, 5, "label", 10))
    reader = reader_factory(url, tmp_path, max_bytes=64 * 1024**2)
    with pytest.raises(ValueError, match="row count changed"):
        list(module._parquet_prefix(reader, {**source["splits"]["train"], "rows": 1}, 5, "label", 10))
    reader = reader_factory(url, tmp_path, max_bytes=64 * 1024**2)
    with pytest.raises(ValueError, match="class label"):
        list(module._parquet_prefix(reader, source["splits"]["train"], 5, "label", 1))


def test_missing_embedded_images_never_fetch_external_paths():
    sink = io.BytesIO()
    table = pa.Table.from_pylist([{"img": {"bytes": None, "path": "https://untrusted.invalid/a.png"},
                                  "label": 0}])
    pq.write_table(table, sink)
    reader = io.BytesIO(sink.getvalue())
    reader.size = len(sink.getvalue())
    with pytest.raises(ValueError, match="external paths are not fetched"):
        list(module._parquet_prefix(reader, {"bytes": reader.size, "rows": 1}, 1, "label", 10))


def test_invalid_budgets_fail_before_acquisition(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        module.prepare_cifar_parquet("mnist", tmp_path)
    with pytest.raises(ValueError, match="budget"):
        module.prepare_cifar_parquet("split_cifar10", tmp_path, limit=1)
    with pytest.raises(ValueError, match="seed"):
        module.prepare_cifar_parquet("split_cifar10", tmp_path, seed=-1)
    assert not (tmp_path / "cifar-parquet").exists()
