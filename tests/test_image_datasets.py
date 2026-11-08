import gzip
import hashlib
import io
import json
import struct
import tarfile
import urllib.error

import pytest

from continual_bench.datasets import images as images


def asset(path):
    return {"filename": path.name, "url": "https://example.org/" + path.name,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def mnist_cache(tmp_path, monkeypatch):
    directory = tmp_path / "raw/mnist"
    directory.mkdir(parents=True)
    assets = []
    for prefix, count in (("train", 100), ("t10k", 30)):
        image_file = directory / f"{prefix}-images-idx3-ubyte.gz"
        label_file = directory / f"{prefix}-labels-idx1-ubyte.gz"
        with gzip.open(image_file, "wb") as stream:
            stream.write(struct.pack(">IIII", 2051, count, 28, 28))
            for i in range(count):
                stream.write(bytes((i + k) % 256 for k in range(784)))
        with gzip.open(label_file, "wb") as stream:
            stream.write(struct.pack(">II", 2049, count))
            stream.write(bytes(i % 10 for i in range(count)))
        assets.extend((asset(image_file), asset(label_file)))
    specification = {"source": "https://example.org/fixture", "license": "synthetic-fixture-only",
                     "license_reference": "https://example.org/license", "assets": assets}
    monkeypatch.setattr(images, "image_dataset_locks", lambda: {"datasets": {"mnist": specification}})
    return directory


@pytest.mark.parametrize("name", ["split_mnist", "permuted_mnist"])
def test_mnist_preserves_100_originals_official_splits_and_reproducible_tasks(tmp_path, monkeypatch, name):
    mnist_cache(tmp_path, monkeypatch)
    source = images.prepare_image_dataset(name, tmp_path, limit=100)
    samples = list(source.samples_iter())
    assert len(samples) == len({s.sample_id for s in samples}) == 100
    assert sum(s.split == "train" for s in samples) == 80
    assert sum(s.split == "test" for s in samples) == 20
    assert all(len(s.x) == 784 and all(0 <= value <= 1 for value in s.x.values()) for s in samples)
    assert all(s.y in {str(i) for i in range(10)} for s in samples)
    assert [s.seq for s in samples] == list(range(100))
    assert samples == list(images.prepare_image_dataset(name, tmp_path).samples_iter())
    manifest = source.manifest()
    assert manifest["natural_chronology"] is False and manifest["changepoints"] is None
    assert manifest["split_counts"] == {"train": 80, "test": 20}
    assert manifest["samples_sha256"]
    persisted = next((tmp_path / "prepared").glob("*/samples.jsonl")).read_bytes()
    assert hashlib.sha256(persisted).hexdigest() == manifest["samples_sha256"]
    if name == "split_mnist":
        assert all(s.task_id == int(s.y) // 2 for s in samples)
    else:
        assert {s.task_id for s in samples} == {0, 1}
        assert "no full-base repetition" in manifest["protocol"]["deviation"]


def test_permutation_changes_only_features_not_original_ids_splits_or_labels(tmp_path, monkeypatch):
    mnist_cache(tmp_path, monkeypatch)
    first = {s.sample_id: s for s in images.prepare_image_dataset(
        "permuted_mnist", tmp_path, seed=42).samples_iter()}
    second = {s.sample_id: s for s in images.prepare_image_dataset(
        "permuted_mnist", tmp_path, seed=43).samples_iter()}
    assert first.keys() == second.keys()
    assert all((s.y, s.split, s.group) == (second[k].y, second[k].split, second[k].group)
               for k, s in first.items())
    assert all(s.x == second[k].x for k, s in first.items() if s.task_id == 0)
    assert any(s.x != second[k].x for k, s in first.items() if s.task_id == 1)
    sample = next(iter(first.values()))
    sample.x["pixel_0000"] = -1
    assert all(v >= 0 for s in images.prepare_image_dataset("permuted_mnist", tmp_path).samples_iter()
               for v in s.x.values())


def cifar_archive(path, dataset, *, invalid_label=False, symlink=False):
    with tarfile.open(path, "w:gz") as archive:
        for split, count in (("train", 32), ("test", 8)):
            name = (f"cifar-100-binary/{split}.bin" if dataset == "cifar100" else
                    "cifar-10-batches-bin/" + ("data_batch_1.bin" if split == "train" else "test_batch.bin"))
            labels = (bytes([0, i % 100]) if dataset == "cifar100" else bytes([i % 10])
                      for i in range(count))
            payload = b"".join(label + bytes([i % 256]) * 3072 for i, label in enumerate(labels))
            if invalid_label:
                payload = bytes([255]) + payload[1:]
            info = tarfile.TarInfo(name)
            if symlink:
                info.type, info.linkname = tarfile.SYMTYPE, "/outside"
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_cifar_binary_loader_keeps_fine_classes_and_bounded_official_splits(tmp_path, monkeypatch, dataset):
    root = tmp_path / "raw" / dataset
    root.mkdir(parents=True)
    path = root / f"{dataset}.tar.gz"
    cifar_archive(path, dataset)
    spec = {"assets": [asset(path)], "source": "https://example.org/fixture",
            "license": "synthetic-fixture-only", "license_reference": "https://example.org/license"}
    monkeypatch.setattr(images, "image_dataset_locks", lambda: {"datasets": {dataset: spec}})
    source = images.prepare_image_dataset("split_" + dataset, tmp_path, limit=20)
    samples = list(source.samples_iter())
    assert len(samples) == 20 and len({s.sample_id for s in samples}) == 20
    assert all(len(s.x) == 3072 for s in samples)
    assert sum(s.split == "train" for s in samples) == 16
    assert {s.y for s in samples} == {str(i) for i in range(16 if dataset == "cifar100" else 10)}
    assert all(s.task_id == int(s.y) // (10 if dataset == "cifar100" else 2) for s in samples)


@pytest.mark.parametrize("symlink", [True, False])
def test_cifar_refuses_invalid_label_or_archive_link(tmp_path, symlink):
    path = tmp_path / "bad.tar.gz"
    cifar_archive(path, "cifar10", invalid_label=not symlink, symlink=symlink)
    with pytest.raises(ValueError, match="label invalid|member type"):
        list(images.read_cifar(path, "cifar10", "train", 2))


def test_idx_rejects_header_mismatch_and_truncation(tmp_path):
    image, label = tmp_path / "images.gz", tmp_path / "labels.gz"
    with gzip.open(image, "wb") as stream:
        stream.write(struct.pack(">IIII", 2051, 2, 28, 28) + bytes(784))
    with gzip.open(label, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 2) + bytes([0, 1]))
    with pytest.raises(ValueError, match="truncated"):
        list(images.read_mnist(image, label, 2))
    with gzip.open(label, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 3) + bytes([0, 1, 2]))
    with pytest.raises(ValueError, match="schema/count"):
        list(images.read_mnist(image, label, 2))


def test_download_is_explicit_atomic_checked_and_cache_reuse_does_not_call_network(tmp_path):
    payload = b"small pinned fixture"
    spec = {"filename": "fixture.gz", "url": "https://example.org/fixture.gz",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def geturl(self):
            return spec["url"]

    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(payload)

    with pytest.raises(FileNotFoundError, match="explicit"):
        images.acquire_asset(tmp_path, spec, opener=opener)
    path = images.acquire_asset(tmp_path, spec, download=True, opener=opener)
    assert path.read_bytes() == payload and len(calls) == 1
    images.acquire_asset(tmp_path, spec, download=True, opener=opener)
    assert len(calls) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        images.acquire_asset(tmp_path, spec, download=True, opener=opener)
    assert len(calls) == 1


def test_http_denials_do_not_retry_and_failed_download_leaves_no_asset(tmp_path):
    spec = {"filename": "fixture.gz", "url": "https://example.org/fixture.gz",
            "bytes": 5, "sha256": "0" * 64}
    calls = []

    def denied(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        images.acquire_asset(tmp_path, spec, download=True, opener=denied)
    assert len(calls) == 1 and not (tmp_path / spec["filename"]).exists()
    assert not [path for path in tmp_path.glob(".download-*") if path.is_file()]


def test_download_rejects_oversized_body_and_no_final_asset_is_committed(tmp_path):
    spec = {"filename": "fixture.gz", "url": "https://example.org/fixture.gz",
            "bytes": 5, "sha256": "0" * 64}

    class Response(io.BytesIO):
        status, headers = 200, {}

        def geturl(self):
            return spec["url"]

    with pytest.raises(ValueError, match="byte budget"):
        images.acquire_asset(tmp_path, spec, download=True,
                             opener=lambda *args, **kwargs: Response(b"too many bytes"))
    assert not (tmp_path / spec["filename"]).exists()
    assert not [path for path in tmp_path.glob(".download-*") if path.is_file()]


def test_lock_catalog_pins_real_assets_and_loaders_reject_unknown_budget():
    locks = images.image_dataset_locks()
    assert set(locks["datasets"]) == {"mnist", "cifar10", "cifar100"}
    assert all(a["url"].startswith("https://") and len(a["sha256"]) == 64
               for spec in locks["datasets"].values() for a in spec["assets"])
    with pytest.raises(ValueError, match="unknown"):
        images.prepare_image_dataset("not_mnist", "/unused")
    with pytest.raises(ValueError, match="budget"):
        images.prepare_image_dataset("split_mnist", "/unused", limit=1)
    assert json.dumps(locks)
