import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from continual_bench.config import Config
from continual_bench.datasets.arrays import NATIVE_ROWS
from continual_bench.datasets.catalog import CATALOG, dataset_spec
from continual_bench.datasets.local_manifest import LocalDatasetManifest, LocalManifestSource
from continual_bench.datasets.real_streams import (
    SPECS,
    InsectsBinarySource,
    prepare_real_stream,
)


def _write_usp_csv(path, labels):
    lines = []
    for idx, label in enumerate(labels):
        features = [f"{idx + j / 100:.4f}" for j in range(33)]
        lines.append(",".join([*features, label]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_real_streams_lock_and_catalog_pin_both_imbalanced_insects_variants():
    root = Path(__file__).parents[1]
    lock = json.loads((root / "configs/contracts/real-streams.lock.json").read_text())
    catalog = json.loads((root / "configs/contracts/dataset-catalog.json").read_text())

    assert "insects" in lock and lock["insects"]["expected_rows"] == 52_848
    assert "insects" in CATALOG and CATALOG.get("insects").task == "multiclass"

    abrupt = lock["insects_abrupt_imbalanced"]
    assert abrupt["expected_rows"] == 355_275
    assert abrupt["expected_sha256"] == "985a5b46c2cb0e42e48a78d5001b57a79ed64fa69f9e5b0362a34cfb552dbc22"
    assert abrupt["license"] == "CC BY 4.0"
    assert abrupt["drift_ground_truth"]["changepoints"] == [83859, 128651, 182320, 242883, 268380]
    assert abrupt["drift_ground_truth"]["convention"] == "0-based half-open [start, end) intervals"
    assert sum(abrupt["original_class_distribution"].values()) == 355_275
    assert sum(r["rows"] for r in abrupt["binary_distribution_per_regime"]) == 355_275
    assert sum(r["positive"] for r in abrupt["binary_distribution_per_regime"]) == 101_256

    reocc = lock["insects_incremental_abrupt_reoccurring_imbalanced"]
    assert reocc["expected_rows"] == 452_044
    assert reocc["expected_sha256"] == "d07e22c0521755f2452d96bb1c7068b26a281be2a627e32f21a4c0c6cfbfd9bc"
    assert "drive.usercontent.google.com/download" in reocc["download"]
    assert "confirm=t" in reocc["download"]
    assert reocc["drift_ground_truth"]["changepoints"] == [150683, 301365]
    assert "incremental_evolution_note" in reocc["drift_ground_truth"]
    assert sum(reocc["original_class_distribution"].values()) == 452_044
    assert sum(r["rows"] for r in reocc["binary_distribution_per_regime"]) == 452_044
    assert sum(r["positive"] for r in reocc["binary_distribution_per_regime"]) == 125_354

    by_name = {entry["name"]: entry for entry in catalog["local_candidates"]}
    for name in ("insects_abrupt_imbalanced", "insects_incremental_abrupt_reoccurring_imbalanced"):
        assert by_name[name]["status"] == "admitted_pinned_binary_with_drift_annotations"
        assert dataset_spec(name).task == "binary"
        cfg = Config.model_validate({"dataset": {"name": name, "samples": 1000}})
        cfg.validate_components()
    assert NATIVE_ROWS["insects_abrupt_imbalanced"] == 355_275
    assert NATIVE_ROWS["insects_incremental_abrupt_reoccurring_imbalanced"] == 452_044


@pytest.mark.parametrize(
    "name,changepoints",
    [
        ("insects_abrupt_imbalanced", (2, 5)),
        ("insects_incremental_abrupt_reoccurring_imbalanced", (3, 6)),
    ],
)
def test_insects_binary_source_streams_headerless_csv_projects_label_2_and_prevents_leakage(
    tmp_path, monkeypatch, name, changepoints
):
    labels = ["11", "2", "12", "3", "2", "4", "5", "2"]
    csv_path = _write_usp_csv(tmp_path / SPECS[name]["filename"], labels)
    raw_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    monkeypatch.setitem(
        SPECS,
        name,
        {
            **SPECS[name],
            "expected_rows": len(labels),
            "expected_bytes": csv_path.stat().st_size,
            "expected_sha256": raw_sha,
            "changepoints": changepoints,
        },
    )

    source = InsectsBinarySource(name, len(labels), 42, 3, 1000, directory=tmp_path)
    events = list(source.events())
    assert len(events) == len(labels)
    assert [e.y for e in events] == [label == "2" for label in labels]
    assert [e.seq for e in events] == list(range(len(labels)))
    assert [e.timestamp for e in events] == [1000 + i for i in range(len(labels))]
    assert all(e.available_at == e.timestamp + 3 for e in events)
    # Non-leakage: no drift annotations or targets in features or prospective fields.
    expected_keys = {f"f{i + 1}" for i in range(33)}
    assert all(set(e.x) == expected_keys for e in events)
    assert all(e.concept is None for e in events)

    samples = list(source.samples_iter())
    assert [s.y for s in samples] == ["1" if label == "2" else "0" for label in labels]
    assert all(s.task_id is None and s.group is None for s in samples)

    manifest = source.manifest()
    assert manifest["task"] == "binary"
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["raw_sha256"] == raw_sha
    assert manifest["target_transformation"]["positive_source_label"] == "2"
    assert manifest["changepoints"] == list(changepoints)
    assert manifest["drift_annotations"]["convention"] == "0-based half-open [start, end) intervals"
    assert manifest["drift_annotations"]["usage_restriction"].startswith("evaluation_and_auditing_only")
    assert sum(r["rows"] for r in manifest["binary_distribution_per_regime"]) == len(labels)
    assert sum(r["positive"] for r in manifest["binary_distribution_per_regime"]) == 3

    # Prefix slicing filters active changepoints without mutating source_changepoints.
    prefix_source = InsectsBinarySource(name, changepoints[0], 42, 0, 0, directory=tmp_path)
    prefix_manifest = prefix_source.manifest()
    assert prefix_manifest["changepoints"] == []
    assert prefix_manifest["source_changepoints"] == list(changepoints)


def test_prepare_real_stream_returns_binary_source_for_imbalanced_insects(tmp_path, monkeypatch):
    import continual_bench.datasets.real_streams as module

    labels = ["11", "2", "12", "3", "4", "5", "2", "11"]
    fixture = _write_usp_csv(tmp_path / "fixture.csv", labels)
    raw_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    name = "insects_abrupt_imbalanced"
    monkeypatch.setitem(
        SPECS,
        name,
        {
            **SPECS[name],
            "expected_rows": len(labels),
            "expected_bytes": fixture.stat().st_size,
            "expected_sha256": raw_sha,
            "changepoints": (3, 6),
        },
    )
    monkeypatch.setattr(
        module,
        "_download",
        lambda url, path, **kwargs: (
            path.write_bytes(fixture.read_bytes()),
            {"bytes": fixture.stat().st_size, "etag": None, "last_modified": None},
        )[1],
    )

    source = prepare_real_stream(name, tmp_path / "cache", limit=5, download=True)
    assert isinstance(source, InsectsBinarySource)
    samples = list(source.samples_iter())
    assert len(samples) == 5
    assert [s.y for s in samples] == ["0", "1", "0", "0", "0"]


def test_local_manifest_supports_csv_headerless_and_one_vs_rest_projection(tmp_path):
    labels = ["11", "2", "12", "3", "4", "5"]
    raw = _write_usp_csv(tmp_path / "abrupt_imbalanced.csv", labels)
    spec = {
        "schema_version": 1,
        "name": "insects_abrupt_imbalanced",
        "variant": "abrupt_imbalanced",
        "version": "usp-v1",
        "source": "https://sites.google.com/view/uspdsrepository",
        "license": "CC BY 4.0",
        "license_reference": "https://creativecommons.org/licenses/by/4.0/",
        "access_approved": True,
        "format": "csv_headerless",
        "files": [{"path": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "rows": 6}],
        "features": {f"f{i + 1}": "numeric" for i in range(33)},
        "target": "class",
        "labels": ["2", "3", "4", "5", "11", "12"],
        "task": "binary",
        "positive_label": "2",
        "target_projection": "one_vs_rest",
        "order": "published",
        "order_description": "Original USP published CSV order",
        "changepoints": [2, 4],
        "preparation": "Headerless 34-column CSV with one-vs-rest projection label=='2'->1 else 0",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(spec))
    LocalDatasetManifest.model_validate(spec)

    source = LocalManifestSource("insects_abrupt_imbalanced", 6, 42, 1, 100, manifest_path=manifest_path)
    events = list(source.events())
    assert [e.y for e in events] == [False, True, False, False, False, False]
    assert all(e.concept is None and len(e.x) == 33 for e in events)


def test_prepare_real_sources_script_downloads_validates_and_publishes_insects(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[1] / "scripts/prepare_real_sources.py"
    spec = importlib.util.spec_from_file_location("prepare_real_sources", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    labels = ["11", "2", "12", "3", "4", "5", "2", "11"]
    name = "insects_incremental_abrupt_reoccurring_imbalanced"
    fixture = _write_usp_csv(tmp_path / "upstream.csv", labels)
    raw_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()

    monkeypatch.setitem(
        SPECS,
        name,
        {
            **SPECS[name],
            "expected_rows": len(labels),
            "expected_bytes": fixture.stat().st_size,
            "expected_sha256": raw_sha,
            "changepoints": (3, 6),
        },
    )

    def fake_download(url, path, **kwargs):
        assert "drive.usercontent.google.com/download" in url
        assert "confirm=t" in url
        path.write_bytes(fixture.read_bytes())
        return {"bytes": fixture.stat().st_size, "etag": None, "last_modified": None}

    monkeypatch.setattr(module, "_download", fake_download)
    dest = tmp_path / "admitted"
    receipt = module.prepare(name, dest)
    assert receipt["raw_rows"] == len(labels)
    assert receipt["raw_sha256"] == raw_sha
    assert receipt["changepoints"] == [3, 6]
    assert (dest / SPECS[name]["filename"]).is_file()
    assert (dest / f"{name}.admission.json").is_file()

    uploaded = []

    class FakeS3:
        def upload_file(self, filename, bucket, key):
            uploaded.append((Path(filename).name, bucket, key))

    published = module.publish_s3(receipt, dest, "continual-bench-data-test", client=FakeS3())
    assert published["data_uri"].endswith(f"/{name}/{SPECS[name]['filename']}")
    assert len(uploaded) == 2

    # Checksum mismatch is rejected without leaving pending temp files.
    (dest / SPECS[name]["filename"]).write_text("corrupted\n")
    with pytest.raises(ValueError, match="byte length|SHA-256"):
        module.prepare(name, dest)
    assert not list(dest.glob(".pending-*"))
