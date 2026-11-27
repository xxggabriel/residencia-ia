"""Publication counts include the evidence uploaded to linked child runs."""

import json
import shutil

import pytest

from continual_bench.application.cloud.worker import TrackingPublicationError, complete_zia_tracking
from continual_bench.infrastructure.aws import mlflow_tracking
from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker
from continual_bench.infrastructure.object_store import LocalObjectStore
from continual_bench.infrastructure.observation_tracking import ARTIFACT_ROOT


@pytest.fixture
def publication(tmp_path, monkeypatch):
    """Control only rendering and MLflow transport; use the real uploader and worker guard."""
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.json").write_text("{}")
    for index in range(80):
        (output / f"metric-{index:03d}.json").write_text(json.dumps({"value": index}))
    fit_root = output / ARTIFACT_ROOT / "fits" / "fit-000001"
    fit_root.mkdir(parents=True)
    for name in ("data.json", "availability.json", "plots-manifest.json"):
        (fit_root / name).write_text("{}")
    assert sum(path.is_file() for path in output.rglob("*")) == 84

    tracker = SageMakerMlflowTracker(
        tracking_uri="http://mlflow.example",
        dataset="ecommerce",
        policy="frozen",
        model_name="xgboost",
        run_id="publication-count-fixture",
        attempt_key="original-1",
    )
    tracker.bind_spot_attempt("original-1", instance_id="original-task", attempt=1)
    tracker.mlflow_run_id = "parent"
    tracker.experiment_id = "experiment"
    tracker.artifact_uri = "s3://artifact-store/native/parent"
    child = {
        "fit_id": "fit-000001",
        "run_id": "child",
        "artifact_uri": "s3://artifact-store/native/child",
    }
    events = []

    def log_batch(**payload):
        events.append(payload)
        return True

    def finish(status):
        events.append({"status": status})
        return True

    def bind_observations(directory, _report, parent_id, children):
        # Synthetic rendering fixture: the three original fit files move to
        # the child and each native run gains its own real lineage file.
        assert parent_id == "parent" and children == [child]
        root = directory / ARTIFACT_ROOT
        (root / "mlflow-lineage.json").write_text(json.dumps({"parent": parent_id}))
        (fit_root / "mlflow-lineage.json").write_text(json.dumps({"child": children[0]["run_id"]}))
        return root

    monkeypatch.setattr(mlflow_tracking, "write_observation_evidence", bind_observations)
    monkeypatch.setattr(tracker, "log_batch", log_batch)
    monkeypatch.setattr(tracker, "finish", finish)
    monkeypatch.setattr(tracker, "log_completed_metrics", lambda _points: True)
    monkeypatch.setattr(tracker, "publish_training_rounds", lambda _report: [child])
    return tracker, output, fit_root, child, events


def final_receipt():
    return {
        "attempt": {"attempt_key": "original-1", "worker_id": "original-task"},
        "outputs_uri": "s3://artifact-store/original-attempt",
        "published": {"files": 84},
        "links": {},
    }


def report(seed):
    return {
        "artifact_mode": "metrics_only",
        "training_history": [{"fit_id": "fit-000001", "seed": seed}],
        "contract": {"iterative_training": {"seed": seed}},
    }


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_worker_guard_accepts_verified_parent_and_child_uploads(publication, tmp_path, seed):
    tracker, output, fit_root, child, events = publication
    store = LocalObjectStore(tmp_path / "store")
    final = final_receipt()
    complete_zia_tracking(tracker, store, final, output, report(seed), None)
    uploaded = final["tracking"]["artifacts"]
    assert final["tracking"]["complete"] is True
    assert uploaded["files"] == 86
    assert uploaded["parent"]["files"] == 82 < final["published"]["files"]
    assert uploaded["children"] == [
        {
            **child,
            "artifact_uri": f"{child['artifact_uri']}/{ARTIFACT_ROOT}",
            "files": 4,
            "bytes": sum(path.stat().st_size for path in fit_root.rglob("*") if path.is_file()),
        }
    ]
    assert uploaded["bytes"] == uploaded["parent"]["bytes"] + uploaded["children"][0]["bytes"]
    objects = [store.head(key) for key in store.list("native/")]
    assert len(objects) == uploaded["files"]
    assert sum(item.size for item in objects) == uploaded["bytes"]
    assert sum(item.key.startswith("native/parent/") for item in objects) == 82
    assert sum(item.key.startswith("native/child/") for item in objects) == 4
    assert {"key": "benchmark_complete", "value": "true"} in events[-1]["tags"]

    # Existing immutable artifacts still produce real verified receipts on retry.
    retried = final_receipt()
    complete_zia_tracking(tracker, store, retried, output, report(seed), None)
    assert retried["tracking"]["artifacts"] == uploaded


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("failure", ["missing", "upload", "receipt_files", "receipt_bytes"])
def test_child_failure_cannot_mark_publication_complete(publication, tmp_path, monkeypatch, seed, failure):
    tracker, output, fit_root, _child, events = publication
    store = LocalObjectStore(tmp_path / "store")
    if failure == "missing":
        bind = mlflow_tracking.write_observation_evidence

        def missing(*args):
            root = bind(*args)
            shutil.rmtree(fit_root)
            return root

        monkeypatch.setattr(mlflow_tracking, "write_observation_evidence", missing)
    elif failure == "upload":
        put_file = store.put_file

        def failing(key, path, **kwargs):
            if key.startswith("native/child/") and key.endswith("data.json"):
                raise OSError("child upload unavailable")
            return put_file(key, path, **kwargs)

        monkeypatch.setattr(store, "put_file", failing)
    else:
        upload = mlflow_tracking.upload_directory_to_mlflow_artifacts

        def incomplete(store, directory, uri, **kwargs):
            receipt = upload(store, directory, uri, **kwargs)
            if uri.startswith("s3://artifact-store/native/child/"):
                receipt[failure.removeprefix("receipt_")] -= 1
            return receipt

        monkeypatch.setattr(mlflow_tracking, "upload_directory_to_mlflow_artifacts", incomplete)
    final = final_receipt()
    with pytest.raises(TrackingPublicationError, match="Incomplete MLflow publication"):
        complete_zia_tracking(tracker, store, final, output, report(seed), None)
    assert "tracking" not in final
    assert not any(event.get("status") == "FINISHED" for event in events)
    assert not any(
        tag == {"key": "benchmark_complete", "value": "true"}
        for event in events
        for tag in event.get("tags", [])
    )
    assert tracker.api_errors
    if failure == "upload":
        assert "child upload unavailable" in tracker.api_errors[-1]
