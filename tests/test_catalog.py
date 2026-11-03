import hashlib

import pytest

from continual_bench.adapters.catalog import catalog_source, put_immutable


def test_local_cache_location_never_changes_shared_dataset_catalog():
    dataset = {"sha256": "fixture", "source": "https://example.org/dataset", "cache": "/host-a/cache"}
    other = {**dataset, "cache": "/host-b/cache"}
    assert catalog_source(dataset) == catalog_source(other)
    assert "cache" not in catalog_source(dataset)


def test_catalog_refuses_overwriting_a_version_with_different_bytes(tmp_path):
    pytest.importorskip("botocore")
    from botocore.exceptions import ClientError

    class Store:
        bucket = "fixture"

        def __init__(self):
            self.objects = {}
            self.client = self
            self.uploads = 0

        def head_object(self, Bucket, Key):
            if Key not in self.objects:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {"Metadata": self.objects[Key]}

        def upload_file(self, path, bucket, key, ExtraArgs):
            self.objects[key] = ExtraArgs["Metadata"]
            self.uploads += 1

    path = tmp_path / "bytes"
    path.write_bytes(b"first")
    store = Store()
    assert put_immutable(store, "dataset/version", path) == "s3://fixture/dataset/version"
    assert store.objects["dataset/version"]["sha256"] == hashlib.sha256(b"first").hexdigest()
    put_immutable(store, "dataset/version", path)
    assert store.uploads == 1
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable"):
        put_immutable(store, "dataset/version", path)
