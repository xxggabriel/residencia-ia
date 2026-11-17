import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "infra/sagemaker/catalog/catalog.py"
SPEC = importlib.util.spec_from_file_location("studio_catalog", SOURCE)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)

SNAPSHOT = ("social", "source-rev", "zia-temporal-v1", "smoke")
SNAPSHOT_ID = "/".join(SNAPSHOT)
MANIFEST_KEY = "datasets/social/source-rev/manifests/zia-temporal-v1/smoke/snapshot.json"
MARKER_KEY = MANIFEST_KEY.replace("snapshot.json", "_SUCCESS.json")
PARQUET_KEY = "datasets/social/source-rev/prepared/zia-temporal-v1/smoke/data.parquet"
RUN_ID = "a" * 32
PREPARED = "b" * 64
FINAL_KEY = "campaigns/example/runs/unit/final.json"


@pytest.fixture
def publisher():
    return catalog.Publisher(
        Mock(),
        bucket="test-bucket",
        domain="domain",
        project="project",
        portal="https://catalog.example",
        mlflow_arn="arn:aws:sagemaker:us-east-1:0:mlflow-app/app-x",
        mlflow_ui="https://mlflow.example",
    )


def event(key, bucket="test-bucket"):
    return {
        "source": "aws.s3",
        "detail-type": "Object Created",
        "detail": {"bucket": {"name": bucket}, "object": {"key": key, "version-id": "event-version"}},
    }


def test_snapshot_reads_pinned_objects_and_rejects_hash_mismatch(publisher):
    manifest = {
        "snapshot_id": SNAPSHOT_ID,
        "version": "snapshot-v1",
        "rows": 42,
        "license": "test",
        "prepared": {
            "sha256": PREPARED,
            "bytes": 123,
            "parquet": {"key": PARQUET_KEY, "sha256": PREPARED, "version_id": "data-v1"},
        },
    }
    raw = json.dumps(manifest).encode()
    marker = {
        "snapshot_id": SNAPSHOT_ID,
        "manifest_key": MANIFEST_KEY,
        "manifest_version_id": "manifest-v1",
        "manifest_sha256": catalog.digest(raw),
        "prepared_sha256": PREPARED,
    }
    objects = {MARKER_KEY: json.dumps(marker).encode(), MANIFEST_KEY: raw}
    publisher.s3.get_object.side_effect = lambda **kw: {"Body": io.BytesIO(objects[kw["Key"]])}
    publisher.s3.head_object.return_value = {"ContentLength": 123, "Metadata": {"sha256": PREPARED}}
    fields = publisher.snapshot(SNAPSHOT, "marker-v1")
    assert fields["SnapshotManifestSha256"] == marker["manifest_sha256"]
    assert [call.kwargs["VersionId"] for call in publisher.s3.get_object.call_args_list] == [
        "marker-v1",
        "manifest-v1",
    ]
    publisher.s3.head_object.assert_called_once_with(
        Bucket="test-bucket",
        Key=PARQUET_KEY,
        VersionId="data-v1",
    )
    objects[MANIFEST_KEY] += b" "
    with pytest.raises(ValueError, match="manifest hash"):
        publisher.snapshot(SNAPSHOT, "marker-v1")


def test_event_ignores_wrong_bucket_and_non_final_keys(publisher):
    assert publisher.process(event(MARKER_KEY, "wrong")) == {"status": "ignored"}
    attempt_key = "campaigns/example/runs/unit/attempts/1/final.json"
    assert publisher.process(event(attempt_key)) == {"status": "ignored"}
    publisher.s3.get_object.assert_not_called()


def test_replayed_evaluation_updates_catalog_once_and_preserves_metrics(publisher):
    fields = {"SnapshotManifestSha256": "c" * 64, "PreparedSha256": PREPARED, "MlflowEvaluationCount": 0}
    final = {
        "kind": "zia",
        "campaign_id": "example",
        "run_id": "unit",
        "snapshot": {"id": SNAPSHOT_ID, "manifest_sha256": "c" * 64, "prepared_sha256": PREPARED},
        "links": {"mlflow_run_id": RUN_ID, "mlflow_experiment_id": "5", "mlflow_artifact_uri": "s3://x"},
    }
    run = {
        "info": {
            "run_id": RUN_ID,
            "experiment_id": "5",
            "artifact_uri": "s3://x",
            "status": "FINISHED",
            "lifecycle_stage": "deleted",
        },
        "data": {"metrics": [{"key": "holdout/log_loss", "value": 0.3}], "tags": []},
    }
    original = {
        "id": "asset",
        "revision": "1",
        "name": "snapshot",
        "description": "snapshot",
        "formsOutput": [
            {"formName": "SnapshotMetadata", "content": json.dumps(fields)},
            {"formName": "AssetCommonDetailsForm", "content": '{"readMe":"Original"}'},
        ],
    }
    publisher.read = Mock(return_value=json.dumps(final).encode())
    publisher.snapshot = Mock(return_value=fields)
    publisher.ensure_asset = Mock(side_effect=lambda *_: original)
    publisher.publish = Mock(return_value="listing")
    publisher.forms = lambda forms: [{"formName": k, "content": json.dumps(v)} for k, v in forms.items()]
    publisher.form_types = ("1", {})

    def revise(**kw):
        original.update(revision="2", formsOutput=kw["formsInput"])
        return original

    def mlflow(path, **kw):
        if path == "log-batch":
            run["data"]["tags"].extend(kw["tags"])
        return {"run": run}

    publisher.datazone.create_asset_revision.side_effect = revise
    publisher.mlflow = Mock(side_effect=mlflow)
    assert publisher.process(event(FINAL_KEY))["status"] == "inactive_mlflow_run"
    publisher.ensure_asset.assert_not_called()
    run["info"]["lifecycle_stage"] = "active"
    assert publisher.process(event(FINAL_KEY))["tagged"]
    assert not publisher.process(event(FINAL_KEY))["tagged"]
    publisher.datazone.create_asset_revision.assert_called_once()
    assert catalog.metadata_forms(original)["SnapshotMetadata"]["MlflowEvaluationCount"] == 1
    assert run["data"]["metrics"] == [{"key": "holdout/log_loss", "value": 0.3}]


def test_recent_readme_is_bounded_and_deduplicated():
    readme = "Original"
    for index in range(35):
        link = f"- [unit {index}](https://example.com/#/experiments/1/runs/{index:032x}?workspace=default)"
        readme = catalog.readme_with_run(readme, link)
    assert readme.startswith("Original\n\n")
    assert readme.count("/runs/") == 20
    assert catalog.readme_with_run(readme, link) == readme
