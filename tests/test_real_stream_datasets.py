import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from continual_bench.datasets.openml import file_sha256
from continual_bench.datasets.real_streams import (
    SPECS,
    _download,
    _gas_rows,
    _insects_rows,
    prepare_real_stream,
)


def insects_fixture(path, *, rows=6):
    path.write_text("".join(
        ",".join([str(i + j / 100) for j in range(33)] + [str(i % 6 + 1)]) + "\n"
        for i in range(rows)
    ))
    return path


def gas_fixture(path, *, batches=10, invalid=False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for batch in range(1, batches + 1):
            features = " ".join(f"{i}:{i + batch / 100}" for i in range(1, 129))
            label = "7" if invalid else str((batch - 1) % 6 + 1)
            archive.writestr(f"Dataset/batch{batch}.dat", f"{label} {features}\n")
    return path


def test_insects_parser_preserves_original_six_class_targets(tmp_path):
    rows = list(_insects_rows(insects_fixture(tmp_path / "fixture.csv")))
    assert [r[0] for r in rows] == list(range(6))
    assert {r[2] for r in rows} == {str(i) for i in range(1, 7)}
    assert all(len(r[1]) == 33 and r[3] is None for r in rows)
    (tmp_path / "bad.csv").write_text("1,2,3,label\n")
    with pytest.raises(ValueError, match="33 features"):
        list(_insects_rows(tmp_path / "bad.csv"))


def test_gas_parser_preserves_numeric_batch_order_and_128_features(tmp_path):
    path = gas_fixture(tmp_path / "fixture.zip")
    rows = list(_gas_rows(path))
    assert [r[3] for r in rows] == list(range(1, 11))
    assert [r[0] for r in rows] == list(range(10))
    assert all(len(r[1]) == 128 and isinstance(r[2], str) for r in rows)
    assert {r[2] for r in rows} == {str(i) for i in range(1, 7)}
    with pytest.raises(ValueError, match="1 through 10"):
        list(_gas_rows(gas_fixture(tmp_path / "missing.zip", batches=9)))
    with pytest.raises(ValueError, match="six original"):
        list(_gas_rows(gas_fixture(tmp_path / "invalid.zip", invalid=True)))


@pytest.mark.parametrize("name", ["insects", "gas"])
def test_acquisition_admission_prefix_cache_replay_and_corruption(tmp_path, monkeypatch, name):
    import continual_bench.datasets.real_streams as module

    fixture = (insects_fixture(tmp_path / "original.csv") if name == "insects" else
               gas_fixture(tmp_path / "original.zip"))
    raw_rows = 6 if name == "insects" else 10
    monkeypatch.setitem(SPECS, name, {**SPECS[name], "expected_rows": raw_rows,
                                    "expected_bytes": fixture.stat().st_size})
    calls = []

    def download(url, path, **kwargs):
        calls.append(url)
        shutil.copyfile(fixture, path)
        return {"bytes": fixture.stat().st_size, "etag": None, "last_modified": None}

    monkeypatch.setattr(module, "_download", download)
    source = prepare_real_stream(name, tmp_path / "cache", limit=3, download=True)
    events = list(source.samples_iter())
    assert len(events) == 3 and len({e.sample_id for e in events}) == 3
    assert all(e.split == "train" and type(e.y) is str and e.seq == i for i, e in enumerate(events))
    replay = prepare_real_stream(name, tmp_path / "cache", limit=3, seed=99, download=False)
    assert events == list(replay.samples_iter()) and len(calls) == 1
    manifest = source.manifest()
    assert manifest["raw_sha256"] == file_sha256(fixture)
    assert manifest["task"] == "multiclass" and manifest["changepoints"] is None
    assert manifest["selection"]["seq_range"] == [0, 3]
    receipt = json.loads((source.path.parent / "acquisition.json").read_text())
    assert receipt["rows"] == raw_rows and len(receipt["labels"]) == 6
    source.path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed after"):
        list(source.samples_iter())
    with pytest.raises(ValueError, match="checksum"):
        prepare_real_stream(name, tmp_path / "cache", limit=3, download=False)


def test_missing_cache_never_acquires_without_explicit_permission(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicit acquisition"):
        prepare_real_stream("gas", tmp_path, download=False)
    with pytest.raises(ValueError, match="example limit"):
        prepare_real_stream("gas", tmp_path, limit=0)
    with pytest.raises(ValueError, match="insects or gas"):
        prepare_real_stream("unknown", tmp_path)


def test_failed_admission_removes_partial_cache_and_explicit_retry_is_clean(tmp_path, monkeypatch):
    import continual_bench.datasets.real_streams as module

    original = insects_fixture(tmp_path / "fixture.csv")
    monkeypatch.setitem(SPECS, "insects", {**SPECS["insects"], "expected_rows": 6,
                                         "expected_bytes": original.stat().st_size})
    attempts = []

    def download(url, path, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            path.write_text("partial")
            raise TimeoutError("injected interruption")
        shutil.copyfile(original, path)
        return {"bytes": original.stat().st_size}

    monkeypatch.setattr(module, "_download", download)
    with pytest.raises(TimeoutError):
        prepare_real_stream("insects", tmp_path / "cache", limit=3, download=True)
    assert list((tmp_path / "cache").iterdir()) == []
    source = prepare_real_stream("insects", tmp_path / "cache", limit=3, download=True)
    assert len(list(source.samples_iter())) == 3 and len(attempts) == 2


def test_download_rejects_html_and_enforces_bytes_without_auth_bypass(tmp_path):
    class Response(io.BytesIO):
        status = 200

        def __init__(self, payload, headers):
            super().__init__(payload)
            self.headers = headers

    with pytest.raises(RuntimeError, match="HTML"):
        _download("https://example.org", tmp_path / "bad", max_bytes=5,
                  opener=lambda *a, **k: Response(b"html", {"Content-Type": "text/html"}))
    with pytest.raises(ValueError, match="byte budget"):
        _download("https://example.org", tmp_path / "large", max_bytes=3,
                  opener=lambda *a, **k: Response(b"large", {}))
    receipt = _download("https://example.org", tmp_path / "valid", max_bytes=5,
                        opener=lambda *a, **k: Response(b"valid", {"ETag": "fixture"}))
    assert receipt["bytes"] == 5 and receipt["etag"] == "fixture"
    assert Path(tmp_path / "valid").read_bytes() == b"valid"
