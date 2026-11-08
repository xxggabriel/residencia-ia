import hashlib
import json
from dataclasses import replace

import pytest

from continual_bench.adapters.task_sources import (
    FriedmanDriftSource,
    LocalDatasetManifest,
    LocalManifestSource,
    MovingSyntheticSource,
    TaskSample,
    permuted_tasks,
    split_class_tasks,
    temporal_tasks,
)


@pytest.mark.parametrize("name", ["hyperplane", "rbf_drift"])
def test_moving_streams_replay_with_binary_targets_and_no_fake_changepoints(name):
    first = MovingSyntheticSource(name, 50, 42, 3, 1000)
    second = MovingSyntheticSource(name, 50, 42, 3, 1000)
    events = list(first.events())
    assert events == list(second.events())
    assert len(events) == 50 and len({e.event_id for e in events}) == 50
    assert all(type(e.y) is bool and e.available_at == e.timestamp + 3 for e in events)
    assert all(e.concept is None and all(isinstance(k, str) for k in e.x) for e in events)
    assert first.manifest()["changepoints"] is None
    assert first.manifest()["version"] == first.version == second.manifest()["version"]
    assert events != list(MovingSyntheticSource(name, 50, 43, 3, 1000).events())


@pytest.mark.parametrize("name,params", [
    ("hyperplane", {"mag_change": 0}), ("hyperplane", {"n_drift_features": 0}),
    ("rbf_drift", {"change_speed": 0}), ("rbf_drift", {"n_drift_centroids": 0}),
    ("rbf_drift", {"n_centroids": 5, "n_drift_centroids": 10}),
])
def test_moving_sources_reject_static_or_invalid_drift(name, params):
    with pytest.raises(ValueError):
        MovingSyntheticSource(name, 50, 42, 0, 0, **params)


@pytest.mark.parametrize("drift_type", ["gra", "gsg", "lea"])
def test_friedman_preserves_regression_and_drift_positions(drift_type):
    source = FriedmanDriftSource(80, 42, drift_type=drift_type)
    samples = list(source.samples_iter())
    assert samples == list(source.samples_iter())
    assert len(samples) == 80 and all(type(s.y) is float for s in samples)
    assert len({s.y for s in samples}) > 2
    assert source.manifest()["version"] == source.version
    assert all(0 < p < 80 for p in source.manifest()["changepoints"])
    with pytest.raises(ValueError, match="regression"):
        source.events()


def test_friedman_rejects_absent_or_overlapping_drift():
    with pytest.raises(ValueError, match="positions"):
        FriedmanDriftSource(50, 42, position=(50, 100))
    with pytest.raises(ValueError, match="nonoverlapping"):
        FriedmanDriftSource(50, 42, drift_type="gsg", position=(10, 20), transition_window=15)


@pytest.fixture
def local_manifest(tmp_path):
    raw = tmp_path / "insects.csv"
    raw.write_text("wing,class,time\n1.1,a,10\n2.2,b,11\n3.3,c,12\n")
    spec = dict(schema_version=1, name="insects", variant="fixture-three-class", version="fixture-v1",
                source="https://example.org/source", license="fixture-only", access_approved=True,
                license_reference="https://example.org/license", format="csv", task="multiclass",
                files=[dict(path=raw.name, sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), rows=3)],
                features={"wing": "numeric"}, target="class", labels=["a", "b", "c"],
                order="timestamp", order_description="fixture timestamps", timestamp_column="time",
                preparation="No transformation; synthetic fixture, not INSECTS data")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path, raw, spec


def _source(path, **kwargs):
    return LocalManifestSource("insects", 3, 42, 2, 0, manifest_path=path, **kwargs)


def test_local_admission_preserves_multiclass_order_and_traceability(local_manifest):
    path, _, spec = local_manifest
    source = _source(path)
    samples = list(source.samples_iter())
    assert [s.y for s in samples] == ["a", "b", "c"]
    assert [s.timestamp for s in samples] == [10, 11, 12]
    assert [s.available_at for s in samples] == [12, 13, 14]
    assert samples == list(_source(path).samples_iter())
    assert source.validate_all()["rows"] == 3
    assert source.manifest()["license"] == spec["license"]
    assert source.manifest()["changepoints"] is None
    with pytest.raises(ValueError, match="multiclass"):
        list(source.events())


def test_local_admission_rejects_corruption_missing_access_and_target_leakage(local_manifest):
    path, raw, spec = local_manifest
    with pytest.raises(ValueError, match="approval"):
        LocalDatasetManifest.model_validate({**spec, "access_approved": False})
    with pytest.raises(ValueError, match="leak"):
        LocalDatasetManifest.model_validate({**spec, "features": {"class": "categorical"}})
    raw.write_text(raw.read_text() + "4.4,a,13\n")
    with pytest.raises(ValueError, match="SHA-256"):
        _source(path)


def test_local_admission_rejects_schema_timestamp_and_count_failure(local_manifest):
    path, raw, spec = local_manifest
    for text, match in [("wing,class,time\n1,a,10\n2,b,9\n3,c,12\n", "decreased"),
                        ("wing,class,time\nnan,a,10\n2,b,11\n3,c,12\n", "non-finite"),
                        ("wing,class,time\n1,a,10.5\n2,b,11\n3,c,12\n", "integer"),
                        ("other,class,time\n1,a,10\n2,b,11\n3,c,12\n", "schema"),
                        ("wing,class,time\n1,a,10\n2,b,11\n", "row count")]:
        raw.write_text(text)
        spec["files"][0]["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        path.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match=match):
            _source(path).validate_all()


def test_local_admission_binary_mapping_is_explicit(local_manifest):
    path, raw, spec = local_manifest
    raw.write_text("wing,class,time\n1,a,10\n2,b,11\n3,a,12\n")
    spec.update(task="binary", labels=["a", "b"], positive_label="b")
    spec["files"][0]["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    path.write_text(json.dumps(spec))
    assert [e.y for e in _source(path).events()] == [False, True, False]
    spec["labels"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="exactly two"):
        LocalDatasetManifest.model_validate(spec)


def test_local_cache_cannot_escape_manifest_directory(local_manifest):
    path, _, spec = local_manifest
    spec["files"][0]["path"] = "../outside.csv"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="inside"):
        _source(path)


def test_local_source_rechecks_hash_before_emitting_and_rejects_duplicate_files(local_manifest):
    path, raw, spec = local_manifest
    source = _source(path)
    raw.write_text(raw.read_text().replace("1.1", "9.9"))
    with pytest.raises(ValueError, match="changed after admission"):
        list(source.samples_iter())
    spec["files"].append({**spec["files"][0], "path": "copy.csv"})
    with pytest.raises(ValueError, match="repeat sample identities"):
        LocalDatasetManifest.model_validate(spec)


def test_local_manifest_censors_annotation_points_outside_selected_prefix(local_manifest):
    path, _, spec = local_manifest
    spec["changepoints"] = [1, 2]
    path.write_text(json.dumps(spec))
    source = LocalManifestSource("insects", 2, 42, 0, 0, manifest_path=path)
    assert source.manifest()["changepoints"] == [1]
    assert list(source.manifest()["source_changepoints"]) == [1, 2]


def test_gas_libsvm_keeps_batch_order_and_all_six_class_labels(tmp_path):
    files = []
    for batch in range(1, 3):
        path = tmp_path / f"batch{batch}.dat"
        path.write_text("".join(f"{label} 1:{label}.0 2:{batch}.0\n" for label in range(1, 7)))
        files.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          rows=6, batch=str(batch)))
    spec = dict(name="gas", variant="fixture-2-batches", version="fixture-v1",
                source="https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
                license="fixture-only", license_reference="https://example.org/license", access_approved=True,
                format="libsvm", task="multiclass", files=files, features={"1": "numeric", "2": "numeric"},
                target="label", labels=[str(i) for i in range(1, 7)], order="batch",
                order_description="provided batch order", preparation="Synthetic fixture only")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    source = LocalManifestSource("gas", 12, 42, 0, 0, manifest_path=manifest)
    records = list(source.samples_iter())
    assert [s.group for s in records] == ["1"] * 6 + ["2"] * 6
    assert [s.y for s in records[:6]] == [str(i) for i in range(1, 7)]
    assert source.validate_all()["rows"] == 12


def task_samples():
    return [TaskSample(str(i), i, i, i, {"p0": i, "p1": i + 1, "p2": i + 2}, str(i % 4),
                       split="test" if i >= 4 else "train", group=str(i // 4)) for i in range(8)]


def test_split_protocol_preserves_labels_split_and_task_membership():
    tasks = split_class_tasks(task_samples(), [0, 1, 2, 3])
    assert [{s.y for s in t} for t in tasks] == [{"0", "1"}, {"2", "3"}]
    assert all(s.task_id == i for i, t in enumerate(tasks) for s in t)
    assert {s.sample_id for t in tasks for s in t if s.split == "train"} == {"0", "1", "2", "3"}
    with pytest.raises(ValueError, match="overlaps"):
        split_class_tasks(task_samples() + task_samples(), [0, 1, 2, 3])


def test_permuted_protocol_is_reproducible_and_does_not_mutate_inputs():
    samples = task_samples()
    tasks, permutations = permuted_tasks(samples, 3, 42, ["p0", "p1", "p2"])
    assert (tasks, permutations) == permuted_tasks(samples, 3, 42, ["p0", "p1", "p2"])
    assert permutations[0] == ["p0", "p1", "p2"]
    assert tasks[0][0].x == samples[0].x
    assert all(t[0].split == "train" and t[-1].split == "test" for t in tasks)
    assert samples == task_samples()


def test_temporal_protocol_preserves_official_sessions_and_within_session_order():
    samples = task_samples()
    tasks = temporal_tasks(samples, ["0", "1"])
    assert [[s.timestamp for s in task] for task in tasks] == [list(range(4)), list(range(4, 8))]
    with pytest.raises(ValueError, match="decreased"):
        temporal_tasks([replace(samples[0], timestamp=10), samples[1], *samples[2:]], ["0", "1"])
    with pytest.raises(ValueError, match="official"):
        temporal_tasks(samples, ["0"])
