"""Object storage (local and S3) conditional writes and the AWS profile contracts."""

import pytest
from botocore.stub import ANY, Stubber
from pydantic import ValidationError

from continual_bench.application.cloud.contracts import (
    CampaignRequest,
    DatasetRequest,
    ReleaseManifest,
    SnapshotRef,
    run_identity,
)
from continual_bench.datasets.zia import DATASETS
from continual_bench.infrastructure.object_store import (
    LocalObjectStore,
    ObjectExists,
    get_json,
    put_json_once,
    split_uri,
)

SOCIAL = DATASETS["social"]["revision"]


def test_local_store_creates_once_and_reports_digest(tmp_path):
    store = LocalObjectStore(tmp_path)
    first = store.put_bytes("a/b.json", b"one", if_absent=True)
    assert first.sha256 and first.size == 3
    with pytest.raises(ObjectExists):
        store.put_bytes("a/b.json", b"two", if_absent=True)
    assert store.get_bytes("a/b.json") == b"one"
    assert list(store.list("a/")) == ["a/b.json"]
    assert split_uri(store.uri("a/b.json"))[1] == "a/b.json"
    with pytest.raises(ValueError):
        store.put_bytes("../escape", b"x")


def test_put_json_once_accepts_identical_retry_and_refuses_changes(tmp_path):
    store = LocalObjectStore(tmp_path)
    _, created = put_json_once(store, "plan.json", {"a": 1})
    _, again = put_json_once(store, "plan.json", {"a": 1})
    assert (created, again) == (True, False)
    with pytest.raises(ObjectExists):
        put_json_once(store, "plan.json", {"a": 2})
    assert get_json(store, "plan.json") == {"a": 1}


def _s3(client_factory=None):
    import boto3

    from continual_bench.infrastructure.aws.s3 import S3ObjectStore

    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y")
    return S3ObjectStore("bucket", client=client, kms_key_id="arn:aws:kms:us-east-1:1:key/k"), client


def test_s3_put_bytes_is_conditional_and_maps_precondition_failures():
    store, client = _s3()
    with Stubber(client) as stub:
        stub.add_response(
            "put_object",
            {"VersionId": "v1"},
            {
                "Bucket": "bucket",
                "Key": "k.json",
                "Body": b"{}",
                "ChecksumSHA256": ANY,
                "Metadata": ANY,
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": ANY,
                "BucketKeyEnabled": True,
                "IfNoneMatch": "*",
            },
        )
        stub.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412)
        stored = store.put_bytes("k.json", b"{}", if_absent=True)
        assert stored.version_id == "v1" and stored.metadata["sha256"] == stored.sha256
        with pytest.raises(ObjectExists):
            store.put_bytes("k.json", b"{}", if_absent=True)


def test_s3_multipart_completion_is_conditional(tmp_path, monkeypatch):
    from continual_bench.infrastructure.aws import s3

    monkeypatch.setattr(s3, "SINGLE_PUT_LIMIT", 4)
    monkeypatch.setattr(s3, "PART_BYTES", 5)
    path = tmp_path / "big.bin"
    path.write_bytes(b"0123456789AB")
    store, client = _s3()
    with Stubber(client) as stub:
        stub.add_client_error("head_object", service_error_code="404", http_status_code=404)
        stub.add_response("create_multipart_upload", {"UploadId": "u"})
        for number in (1, 2, 3):
            stub.add_response(
                "upload_part",
                {"ETag": f'"{number}"', "ChecksumSHA256": "c"},
                {
                    "Bucket": "bucket",
                    "Key": "big.bin",
                    "UploadId": "u",
                    "PartNumber": number,
                    "Body": ANY,
                    "ChecksumAlgorithm": "SHA256",
                    "ChecksumSHA256": ANY,
                },
            )
        stub.add_response(
            "complete_multipart_upload",
            {"VersionId": "v9"},
            {
                "Bucket": "bucket",
                "Key": "big.bin",
                "UploadId": "u",
                "MultipartUpload": ANY,
                "IfNoneMatch": "*",
            },
        )
        stored = store.put_file("big.bin", path, if_absent=True)
    assert stored.version_id == "v9" and stored.size == 12


def test_s3_multipart_aborts_on_conflict(tmp_path, monkeypatch):
    from continual_bench.infrastructure.aws import s3

    monkeypatch.setattr(s3, "SINGLE_PUT_LIMIT", 4)
    path = tmp_path / "big.bin"
    path.write_bytes(b"0123456789")
    store, client = _s3()
    with Stubber(client) as stub:
        stub.add_client_error("head_object", service_error_code="404", http_status_code=404)
        stub.add_response("create_multipart_upload", {"UploadId": "u"})
        stub.add_response("upload_part", {"ETag": '"1"'})
        stub.add_client_error(
            "complete_multipart_upload", service_error_code="PreconditionFailed", http_status_code=412
        )
        stub.add_response(
            "abort_multipart_upload", {}, {"Bucket": "bucket", "Key": "big.bin", "UploadId": "u"}
        )
        with pytest.raises(ObjectExists):
            store.put_file("big.bin", path, if_absent=True)


def test_s3_get_missing_object_is_file_not_found():
    store, client = _s3()
    with Stubber(client) as stub:
        stub.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
        with pytest.raises(FileNotFoundError):
            store.get_bytes("missing")


def test_snapshot_and_dataset_requests_follow_the_catalog():
    snapshot = SnapshotRef(dataset="social", revision=SOCIAL, scope="smoke")
    assert snapshot.snapshot_id == f"social/{SOCIAL}/zia-temporal-v1/smoke"
    with pytest.raises(ValidationError):
        SnapshotRef(dataset="social", revision="0" * 40, scope="smoke")
    with pytest.raises(ValidationError):
        SnapshotRef(dataset="social", revision=SOCIAL, scope="smoke", preparation_version="other-v9")
    assert DatasetRequest(dataset="social", scope="full").snapshot().revision == SOCIAL
    with pytest.raises(ValidationError):
        DatasetRequest(dataset="social", scope="full", mode="import")
    with pytest.raises(ValidationError):
        DatasetRequest(dataset="social", scope="full", import_uri="s3://bucket/x")
    with pytest.raises(ValidationError, match="url|extra"):
        DatasetRequest(dataset="social", scope="full", url="https://evil.example/x")


def test_campaign_request_consistency():
    snapshot = {"dataset": "social", "revision": SOCIAL, "scope": "full"}
    request = CampaignRequest(campaign_id="social-full-v1", snapshot=snapshot)
    assert (request.dataset, request.scope, request.matrix.policies) == (
        "social",
        "full",
        ("frozen", "drift", "incremental", "periodic"),
    )
    with pytest.raises(ValidationError):
        CampaignRequest(campaign_id="Bad_ID", snapshot=snapshot)
    with pytest.raises(ValidationError):
        CampaignRequest(campaign_id="c", snapshot=snapshot, scope="smoke")
    with pytest.raises(ValidationError):
        CampaignRequest(campaign_id="c")
    with pytest.raises(ValidationError):
        CampaignRequest(
            campaign_id="c", snapshot=snapshot, matrix={"kind": "zia", "policies": ["drift", "drift"]}
        )
    with pytest.raises(ValidationError):
        CampaignRequest(campaign_id="c", snapshot=snapshot, max_parallel=0)


def test_release_manifest_requires_pinned_revisions():
    image = {
        "repository": "1.dkr.ecr.us-east-1.amazonaws.com/r",
        "digest": "sha256:" + "a" * 64,
        "tag": "git-" + "b" * 40,
    }
    base = {
        "git_sha": "b" * 40,
        "stack": "s",
        "images": {"runtime": image, "ml": image},
        "job_definitions": {
            "benchmark": "arn:aws:batch:us-east-1:1:job-definition/x:3",
            "prepare": "arn:aws:batch:us-east-1:1:job-definition/y:3",
        },
        "control_function": "arn:aws:lambda:us-east-1:1:function:c:7",
    }
    assert ReleaseManifest(**base).images["runtime"].uri.endswith("@sha256:" + "a" * 64)
    with pytest.raises(ValidationError):
        ReleaseManifest(
            **{
                **base,
                "job_definitions": {
                    **base["job_definitions"],
                    "benchmark": "arn:aws:batch:us-east-1:1:job-definition/x",
                },
            }
        )
    with pytest.raises(ValidationError):
        ReleaseManifest(**{**base, "control_function": "arn:aws:lambda:us-east-1:1:function:c"})
    with pytest.raises(ValidationError):
        ReleaseManifest(**{**base, "images": {"runtime": {**image, "digest": "latest"}, "ml": image}})


def test_run_identity_is_deterministic_and_contract_sensitive():
    contract = {"kind": "zia", "params": {"policy": "drift"}, "release": {"git_sha": "a" * 40}}
    assert run_identity("c1", contract) == run_identity("c1", dict(reversed(contract.items())))
    assert run_identity("c1", contract)[0] != run_identity("c2", contract)[0]
    changed = {**contract, "release": {"git_sha": "b" * 40}}
    assert run_identity("c1", contract)[0] != run_identity("c1", changed)[0]
