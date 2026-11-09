"""Kinesis transport ordering/resume, event-time windows, Feature Store checks, SageMaker profiles,
Model Registry idempotency, EMF dimensions and USD accounting."""

import io
import json
import tarfile
from pathlib import Path

import boto3
import numpy as np
import pytest
from botocore.stub import ANY, Stubber

from continual_bench.application.cloud.costs import attempt_estimate, load_prices, reconcile
from continual_bench.application.cloud.features import (
    TemporalLeakage,
    assert_no_leakage,
    parity,
    point_in_time,
)
from continual_bench.application.cloud.fixtures import fixture_parquet
from continual_bench.application.cloud.records import jsonl
from continual_bench.application.cloud.streaming import WindowState
from continual_bench.core.contracts import Event, event_json
from continual_bench.infrastructure.aws.emf import emf_record
from continual_bench.infrastructure.aws.kinesis import KinesisTransport
from continual_bench.infrastructure.object_store import LocalObjectStore
from continual_bench.infrastructure.stream_order import OrderedStream, StreamOrderError


def test_ordered_stream_drops_verified_duplicates_and_refuses_gaps():
    stream = OrderedStream()
    assert stream.accept(0, "e0") and stream.accept(1, "e1")
    assert stream.accept(0, "e0") is False and stream.duplicates == 1
    with pytest.raises(StreamOrderError, match="different event_id"):
        stream.accept(1, "other")
    with pytest.raises(StreamOrderError, match="gap"):
        stream.accept(5, "e5")
    with pytest.raises(StreamOrderError, match="immutable"):
        stream.accept(2, "e2", payload="a", expected="b")


class FakeKinesis:
    """One shard; PutRecords can fail a record once to force a retry that reorders the stream."""

    def __init__(self, fail_once=(), duplicate=()):
        self.records, self.fail_once, self.duplicate = [], set(fail_once), set(duplicate)
        self.iterators = {}

    def put_records(self, StreamName, Records):
        results = []
        for record in Records:
            seq = json.loads(record["Data"])["seq"]
            if seq in self.fail_once:
                self.fail_once.discard(seq)
                results.append({"ErrorCode": "ProvisionedThroughputExceededException"})
                continue
            self.records.append(record["Data"])
            if seq in self.duplicate:
                self.records.append(record["Data"])
            results.append({"SequenceNumber": str(len(self.records))})
        return {"FailedRecordCount": sum("ErrorCode" in r for r in results), "Records": results}

    def list_shards(self, StreamName):
        return {"Shards": [{"ShardId": "shard-0", "SequenceNumberRange": {"StartingSequenceNumber": "0"}}]}

    def get_shard_iterator(self, StreamName, ShardId, ShardIteratorType, StartingSequenceNumber=None):
        start = int(StartingSequenceNumber) if ShardIteratorType == "AFTER_SEQUENCE_NUMBER" else 0
        return {"ShardIterator": f"it-{start}"}

    def get_records(self, ShardIterator, Limit):
        start = int(ShardIterator.split("-")[1])
        chunk = self.records[start : start + 3]
        return {
            "Records": [{"Data": d, "SequenceNumber": str(start + i + 1)} for i, d in enumerate(chunk)],
            "NextShardIterator": f"it-{start + len(chunk)}"
            if chunk or start < len(self.records)
            else f"it-{start}",
        }


def _events(n):
    return [
        Event(
            event_id=f"e{i}",
            seq=i,
            timestamp=1_700_000_000 + i,
            x={"a": float(i)},
            y=i % 2 == 0,
            available_at=i,
        )
        for i in range(n)
    ]


def _manifest(tmp_path, events):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(event_json(e) + "\n" for e in events))
    return {"run_key": "run-1", "samples": len(events), "path": str(path)}


def test_kinesis_transport_restores_order_and_drops_duplicates(tmp_path):
    events = _events(1205)
    fake = FakeKinesis(fail_once={7, 600}, duplicate={10, 999})
    store = LocalObjectStore(tmp_path / "s")
    transport = KinesisTransport(
        "stream", client=fake, cursor_store=store, cursor_key="cursor.json", poll_seconds=0, stall_seconds=1
    )
    delivered = list(transport.events(_manifest(tmp_path, events), iter(events), deadline=60))
    assert [e.seq for e in delivered] == list(range(1205))
    assert transport.statistics["duplicates"] >= 2 and transport.statistics["reordered"] > 0
    cursor = json.loads(store.get_bytes("cursor.json"))
    assert cursor["next_seq"] == 1205 and cursor["run_key"] == "run-1"
    assert transport.contract()["name"] == "kinesis"


def test_kinesis_transport_resumes_from_a_compatible_cursor(tmp_path):
    events = _events(2500)
    fake = FakeKinesis()
    store = LocalObjectStore(tmp_path / "s")
    manifest = _manifest(tmp_path, events)
    first = KinesisTransport(
        "stream", client=fake, cursor_store=store, cursor_key="c.json", poll_seconds=0, stall_seconds=1
    )
    stream = first.events(manifest, iter(events), deadline=60)
    consumed = [next(stream).seq for _ in range(1500)]
    stream.close()
    cursor = json.loads(store.get_bytes("c.json"))
    assert 1000 <= cursor["next_seq"] <= 1500
    resumed = KinesisTransport(
        "stream", client=fake, cursor_store=store, cursor_key="c.json", poll_seconds=0, stall_seconds=1
    )
    rest = [e.seq for e in resumed.events(manifest, iter(events), deadline=60)]
    assert rest[0] == cursor["next_seq"] and rest[-1] == 2499 and len(fake.records) == 2500
    assert consumed[: cursor["next_seq"]] + rest == list(range(2500))
    store.put_bytes("c.json", json.dumps({**cursor, "run_key": "other"}).encode())
    with pytest.raises(ValueError, match="another run"):
        next(resumed.events(manifest, iter(events), deadline=60))


def test_kinesis_transport_refuses_gaps(tmp_path):
    events = _events(50)
    fake = FakeKinesis()
    transport = KinesisTransport("stream", client=fake, poll_seconds=0, stall_seconds=0.01)
    transport.produce(iter(events), "run-1")
    fake.records = [r for r in fake.records if json.loads(r)["seq"] != 20]
    transport.produce = lambda *a, **k: None
    with pytest.raises((StreamOrderError, TimeoutError)):
        list(transport.events(_manifest(tmp_path, events), iter(events), deadline=60))


def test_event_time_windows_late_records_and_checkpoint_restore():
    window = WindowState(size_ms=1000, out_of_orderness_ms=200, allowed_lateness_ms=300)
    for t in (100, 900, 1100, 950, 2600, 1050, 400):
        window.add(t, label=1)
    snapshot = json.loads(json.dumps(window.snapshot()))
    restored = WindowState.restore(1000, 200, 300, snapshot)
    for state in (window, restored):
        state.add(5000, label=0)
        state.finish()
    assert window.closed == restored.closed and window.late == restored.late
    first = next(w for w in window.closed if w["window_start"] == 0)
    assert first["count"] == 3  # 100, 900 and the out-of-order 950
    # 2600 moves the watermark to 2400 ≥ 2000 + lateness: 1050 and 400 arrive after their windows closed.
    assert [late["event_time_ms"] for late in window.late] == [1050, 400]
    assert next(w for w in window.closed if w["window_start"] == 1000)["count"] == 1


def test_point_in_time_join_parity_and_leakage():
    records = [{"entity_id": "a", "event_time": 10, "v": 1}, {"entity_id": "a", "event_time": 20, "v": 2}]
    joined = point_in_time(
        [{"entity_id": "a", "event_time": 15}, {"entity_id": "a", "event_time": 5}], records
    )
    assert joined[0]["features"]["v"] == 1 and joined[1]["features"] is None
    with pytest.raises(TemporalLeakage):
        assert_no_leakage([{"event_time": 10, "feature_time": 11}])
    assert assert_no_leakage([{"event_time": 10, "feature_time": 10}]) == 1
    report = parity([{"event_id": 1, "f": 0.5}], [{"event_id": 1, "f": 0.5000001}], ["f"], tolerance=1e-6)
    assert report["ok"]
    assert not parity([{"event_id": 1, "f": 0.5}], [{"event_id": 2, "f": 0.5}], ["f"])["ok"]


def test_attempt_estimate_and_reconciliation_keep_estimates_observed_and_allocations_apart():
    prices = load_prices()
    estimate = attempt_estimate(
        started_ms=0, stopped_ms=30_000, vcpus=1, memory_gib=2, ephemeral_gib=50, prices=prices
    )
    assert estimate["billed_seconds"] == 60
    assert estimate["components"]["vcpu_usd"] == pytest.approx(0.04048 / 60)
    assert estimate["components"]["ephemeral_storage_usd"] == pytest.approx(30 * 0.000111 / 60)
    attempts = [{"run_id": f"r{i}", "estimate": {"total_usd": 0.01}, "vcpu_seconds": 3600} for i in range(4)]
    cur = [
        {"campaign": "c", "run_id": "r0", "split_cost": 0.012},
        {"product_code": "AmazonVPC", "cost": 1.0},
        {"product_code": "CodeBuild", "cost": 0.5},
        {"product_code": "AmazonRoute53", "cost": 0.1},
    ]
    report = reconcile("c", attempts, cur, period_vcpu_seconds=4 * 3600 * 2)
    assert report["estimated_usd"] == pytest.approx(0.04) and report["observed_task_usd"] == pytest.approx(
        0.012
    )
    assert report["allocation_share"] == 0.5
    assert report["allocated_shared_usd"] == {"network": 0.5, "ci_cd": 0.25}
    assert report["unattributed_period_usd"] == pytest.approx(0.1)


def test_emf_keeps_identifiers_out_of_dimensions():
    record = emf_record(
        {"RunFailed": 1, "Missing": None},
        {"Dataset": "social", "Policy": "drift"},
        {"run_id": "abc", "campaign_id": "c"},
    )
    assert record["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [["Dataset", "Policy"]]
    assert record["run_id"] == "abc" and "Missing" not in record
    with pytest.raises(ValueError, match="cardinality"):
        emf_record({"RunFailed": 1}, {"RunId": "abc"})


def test_records_are_single_line_json():
    assert jsonl([{"a": 1}, {"b": [1, 2]}]).decode().splitlines() == ['{"a": 1}', '{"b": [1, 2]}']


def test_sagemaker_train_serve_and_process_contracts(tmp_path):
    from continual_bench.orchestration import sagemaker

    channel = tmp_path / "train"
    fixture_parquet(channel / "smoke.parquet")
    result = sagemaker.train(channel, tmp_path / "model", {"validation_fraction": 0.1, "max_iterations": 10})
    splits = result["metadata"]["splits"]
    assert (
        splits["train"][1] == splits["validation"][0]
        and splits["validation"][1] == splits["holdout_start_excluded"]
    )
    assert result["validation"]["accuracy"] > 0.9
    predictor = sagemaker.load_predictor(tmp_path / "model")
    rows = sagemaker.score(predictor, json.dumps({"instances": [[0.0], [3.0]]}).encode(), "application/json")
    assert np.allclose(np.sum(rows, axis=1), 1) and np.argmax(rows[1]) == 3
    assert sagemaker.score(predictor, b"0.0\n3.0\n", "text/csv") == rows
    profile = sagemaker.process(channel, tmp_path / "out")
    assert profile["smoke.parquet"]["rows"] == 10_000 and (tmp_path / "out" / "profile.json").is_file()


def test_sagemaker_profiles_build_valid_requests():
    import importlib.util

    spec = importlib.util.spec_from_file_location("jobs", Path("scripts/aws/sagemaker_jobs.py"))
    jobs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jobs)
    kms = "arn:aws:kms:us-east-1:123456789012:key/example"
    common = {"bucket": "b", "role": "arn:aws:iam::1:role/r", "git_sha": "a" * 40, "kms_key_arn": kms}
    training = jobs.training_request(jobs.profile("training"), image="ml@sha256:1", **common)
    tuning = jobs.tuning_request(
        jobs.profile("tuning"), jobs.profile("training"), image="ml@sha256:1", **common
    )
    processing = jobs.processing_request(jobs.profile("processing"), image="rt@sha256:1", **common)
    assert processing["ProcessingOutputConfig"]["KmsKeyId"] == kms
    assert training["OutputDataConfig"]["KmsKeyId"] == kms
    assert tuning["TrainingJobDefinition"]["OutputDataConfig"]["KmsKeyId"] == kms
    stubber_client = boto3.client(
        "sagemaker", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
    )
    with Stubber(stubber_client) as stub:
        stub.add_response("create_training_job", {"TrainingJobArn": "arn:t"}, training)
        stub.add_response(
            "create_hyper_parameter_tuning_job", {"HyperParameterTuningJobArn": "arn:h"}, tuning
        )
        stub.add_response("create_processing_job", {"ProcessingJobArn": "arn:p"}, processing)
        stubber_client.create_training_job(**training)
        stubber_client.create_hyper_parameter_tuning_job(**tuning)
        stubber_client.create_processing_job(**processing)
    config = tuning["HyperParameterTuningJobConfig"]
    assert config["HyperParameterTuningJobObjective"]["MetricName"] == "validation:log_loss"
    assert config["ResourceLimits"]["MaxNumberOfTrainingJobs"] == 8


def test_model_registry_registers_pending_candidates_once():
    from continual_bench.infrastructure.aws.model_registry import SageMakerModelRegistry

    client = boto3.client(
        "sagemaker", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
    )
    registry = SageMakerModelRegistry(client=client)
    with Stubber(client) as stub:
        stub.add_response("list_model_packages", {"ModelPackageSummaryList": []})
        stub.add_response(
            "create_model_package",
            {"ModelPackageArn": "arn:mp/1"},
            {
                "ModelPackageGroupName": "g",
                "ModelPackageDescription": "d",
                "ModelApprovalStatus": "PendingManualApproval",
                "CustomerMetadataProperties": {"campaign_id": "c", "run_id": "r1"},
                "InferenceSpecification": ANY,
            },
        )
        stub.add_response(
            "list_model_packages",
            {
                "ModelPackageSummaryList": [
                    {
                        "ModelPackageGroupName": "g",
                        "ModelPackageArn": "arn:mp/1",
                        "CreationTime": 0,
                        "ModelPackageStatus": "Completed",
                    }
                ]
            },
        )
        stub.add_response(
            "describe_model_package",
            {
                "ModelPackageName": "g",
                "ModelPackageArn": "arn:mp/1",
                "CreationTime": 0,
                "ModelPackageStatus": "Completed",
                "ModelPackageStatusDetails": {"ValidationStatuses": []},
                "CustomerMetadataProperties": {"run_id": "r1"},
            },
        )
        first = registry.register(
            group="g",
            run_id="r1",
            image="ml@sha256:1",
            model_data_uri="s3://b/m.tar.gz",
            metadata={"campaign_id": "c"},
            description="d",
        )
        second = registry.register(
            group="g",
            run_id="r1",
            image="ml@sha256:1",
            model_data_uri="s3://b/m.tar.gz",
            metadata={"campaign_id": "c"},
            description="d",
        )
    assert first == {"model_package_arn": "arn:mp/1", "created": True}
    assert second == {"model_package_arn": "arn:mp/1", "created": False}


def test_model_archive_is_reproducible_and_loadable(tmp_path):
    from continual_bench.application.cloud.worker import model_tarball
    from continual_bench.models.zia import ZiaPredictor
    from continual_bench.orchestration import sagemaker

    channel = tmp_path / "train"
    fixture_parquet(channel / "smoke.parquet")
    sagemaker.train(channel, tmp_path / "model", {"max_iterations": 5})
    metadata = json.loads((tmp_path / "model" / "model-metadata.json").read_text())
    archive = model_tarball(tmp_path / "model", metadata)
    assert archive == model_tarball(tmp_path / "model", metadata)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp_path / "extracted", filter="data")
    assert ZiaPredictor(tmp_path / "extracted").features == ["feature"]
