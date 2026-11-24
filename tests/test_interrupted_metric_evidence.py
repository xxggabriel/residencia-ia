"""Interrupted metric observations survive without creating a resumable model."""

import json

import pytest

from continual_bench.application.cloud import layout, planning, worker
from continual_bench.application.cloud.checkpoints import CampaignCancelled, CheckpointError
from continual_bench.application.cloud.contracts import CampaignRequest, RunUnit
from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.infrastructure.object_store import (
    LocalObjectStore,
    get_json,
    put_json,
    sha256_bytes,
    split_uri,
)


@pytest.fixture(params=[42, 43, 44])
def metric_case(tmp_path, request):
    seed = request.param
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path)
    release = publish_fixture_release(store)
    campaign = CampaignRequest(
        campaign_id="interrupted-metrics",
        snapshot=snapshot.model_dump(),
        matrix={
            "kind": "zia",
            "policies": ["frozen"],
            "seeds": [seed],
            "artifact_mode": "metrics_only",
        },
    )
    plan = planning.build_plan(campaign, release, snapshot_manifest=verify_snapshot(store, snapshot))
    unit = RunUnit(**plan["units"][0])
    put_json(store, unit.unit_key, unit.model_dump(mode="json"))
    env = {
        "AWS_BATCH_JOB_ID": f"metrics-{seed}",
        "AWS_BATCH_JOB_ATTEMPT": "1",
        "BENCH_FARGATE_SPOT": "true",
        "BENCH_GIT_REVISION": release.git_sha,
        "BENCH_IMAGE_DIGEST": release.images["runtime"].digest,
        "BENCH_IMAGE_URI": release.images["runtime"].uri,
    }
    return store, unit, env


class FakeTermination:
    instances = []

    def __init__(self, *, on_terminate):
        self.received = False
        self.on_terminate = on_terminate
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def terminate(self):
        self.received = True
        self.on_terminate(15)


def write_observations(output, seed):
    fit = output / "training-fits-partial" / "initial.json"
    fit.parent.mkdir(parents=True)
    fit.write_text(
        json.dumps(
            {
                "seed": seed,
                "steps": [
                    {"step": 1, "kind": "epoch", "metrics": {"loss": 0.7, "accuracy": 0.5}},
                    {"step": 2, "kind": "epoch", "metrics": {"loss": 0.6, "accuracy": 0.7}},
                ],
            }
        )
    )
    (output / "partial-system-metrics.json").write_text(
        json.dumps({"complete": False, "process_cpu_seconds": 0.4, "rss_bytes": 10000})
    )
    (output / "prediction-schema.json").write_text(
        json.dumps({"columns": ["seq", "label", "prediction", "p_0", "p_1"]})
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    predictions = output / "predictions" / "part-0000.parquet"
    predictions.parent.mkdir()
    pq.write_table(
        pa.table({"seq": [0], "label": [1], "prediction": [1], "p_0": [0.1], "p_1": [0.9]}),
        predictions,
    )


class DrainingTracker:
    """The final queued observation becomes visible only after native flush."""

    def __init__(self, output):
        self.output = output
        self.drained = False

    def bind_spot_attempt(self, *_args, **_kwargs):
        pass

    def start_live(self):
        pass

    def progress(self, *_args):
        pass

    def record_spot_attempt(self, *_args, **_kwargs):
        return True

    def publish_run_report(self, *_args, **_kwargs):
        pytest.fail("an incomplete evaluation must not publish a completed report")

    def flush(self):
        if not self.drained:
            with (self.output / "live-metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": 3, "values": {"loss": 0.55}}) + "\n")
            self.drained = True


@pytest.mark.parametrize("fault", ["signal", "cancelled", "evaluation"])
def test_failure_archives_incomplete_metrics_after_drain(metric_case, tmp_path, monkeypatch, fault):
    store, unit, env = metric_case
    root = tmp_path / "first"
    tracker = DrainingTracker(root / "run")
    monkeypatch.setattr(worker, "GracefulTerminationHandler", FakeTermination)
    writes = []
    original_file, original_bytes = store.put_file, store.put_bytes

    def put_file(key, path, **kwargs):
        if "/partial/" in key:
            assert kwargs["if_absent"] is True
            writes.append(key)
        return original_file(key, path, **kwargs)

    def put_bytes(key, payload, **kwargs):
        if "/partial/" in key:
            assert kwargs["if_absent"] is True
            writes.append(key)
        return original_bytes(key, payload, **kwargs)

    def evaluate(_unit, _store, workdir, _log, **kwargs):
        assert kwargs.get("checkpoint") is None
        output = workdir / "run"
        write_observations(output, unit.params["seed"])
        # Prepared inputs belong outside the metrics directory and must stay there.
        (workdir / "raw-inputs.csv").write_text("private feature values")
        kwargs["progress"](2, {"loss": 0.6, "accuracy": 0.7})
        if fault == "signal":
            FakeTermination.instances[-1].terminate()
            assert kwargs["stop_requested"]()
            raise InterruptedError("Spot signal during epoch")
        if fault == "cancelled":
            raise CampaignCancelled("cancelled during epoch")
        raise RuntimeError("evaluation failed after valid observations")

    def forbid(*_args, **_kwargs):
        pytest.fail("partial metrics must not save or restore training state")

    monkeypatch.setattr(store, "put_file", put_file)
    monkeypatch.setattr(store, "put_bytes", put_bytes)
    monkeypatch.setattr(worker, "execute_zia", evaluate)
    monkeypatch.setattr(worker, "SpotCheckpoint", forbid)
    monkeypatch.setattr(worker, "verify_native_loader", forbid)
    monkeypatch.setattr(worker, "model_tarball", forbid)
    code = worker.run_unit(store, unit.unit_key, root, env=env, tracker=tracker, clock=lambda: 1234.0)
    expected = {
        "signal": (worker.EXIT_INTERRUPTED, "INTERRUPTED", "InterruptedError"),
        "cancelled": (worker.EXIT_CANCELLED, "CANCELLED", "CampaignCancelled"),
        "evaluation": (worker.EXIT_FAILED, "FAILED", "RuntimeError"),
    }[fault]
    attempt = layout.attempt_key_for(env["AWS_BATCH_JOB_ID"], "1")
    status = get_json(store, layout.attempt_status_key(unit.campaign_id, unit.run_id, attempt))
    assert (code, status["state"], status["diagnostic"]["error_type"]) == expected
    assert status["diagnostic"]["exit_code"] == code and status["finished_at"] == 1234.0
    receipt = status["partial_metrics"]
    manifest_key = split_uri(receipt["manifest_uri"])[1]
    assert writes[-1] == manifest_key
    manifest = get_json(store, manifest_key)
    assert manifest["complete"] is False and manifest["scientific_status"] == "incomplete"
    assert manifest["labels"]["seed"] == str(unit.params["seed"])
    assert manifest["evaluation_recovery"] == "restart_unit_without_training_checkpoint"
    prefix = manifest_key.rsplit("/", 1)[0]
    for relative, reference in manifest["files"].items():
        data = store.get_bytes(f"{prefix}/{relative}")
        assert reference == {"sha256": sha256_bytes(data), "bytes": len(data)}
        assert data == (root / "run" / relative).read_bytes()
    assert receipt["manifest_sha256"] == sha256_bytes(store.get_bytes(manifest_key))
    assert {
        "training-fits-partial/initial.json",
        "partial-system-metrics.json",
        "prediction-schema.json",
        "predictions/part-0000.parquet",
        "live-metrics.jsonl",
    } == set(manifest["files"])
    ledger = [json.loads(row) for row in store.get_bytes(f"{prefix}/live-metrics.jsonl").splitlines()]
    assert [point["step"] for point in ledger] == [2, 3]
    assert not any("raw-inputs" in key or "/checkpoints/" in key for key in store.list())
    assert store.head(layout.final_key(unit.campaign_id, unit.run_id)) is None
    assert store.head(f"{layout.run_prefix(unit.campaign_id, unit.run_id)}/pending-publication.json") is None


def test_retry_restarts_without_restoring_incomplete_observations(metric_case, tmp_path, monkeypatch):
    store, unit, env = metric_case
    calls = []
    monkeypatch.setattr(worker, "GracefulTerminationHandler", FakeTermination)

    def evaluate(_unit, _store, workdir, _log, **kwargs):
        assert kwargs.get("checkpoint") is None
        assert not (workdir / "run").exists()
        calls.append(workdir)
        write_observations(workdir / "run", unit.params["seed"])
        FakeTermination.instances[-1].terminate()
        raise InterruptedError("new attempt must restart")

    monkeypatch.setattr(worker, "execute_zia", evaluate)
    for attempt in (1, 2):
        assert (
            worker.run_unit(
                store,
                unit.unit_key,
                tmp_path / str(attempt),
                env={**env, "AWS_BATCH_JOB_ATTEMPT": str(attempt)},
            )
            == worker.EXIT_INTERRUPTED
        )
    assert len(calls) == 2
    receipts = [key for key in store.list() if key.endswith("partial-metrics-manifest.json")]
    assert len(receipts) == 2
    assert store.head(layout.final_key(unit.campaign_id, unit.run_id)) is None


def test_partial_manifest_is_immutable_and_idempotent(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    output = tmp_path / "run"
    write_observations(output, 42)
    status = {"campaign_id": "test", "run_id": "run", "attempt_key": "one", "labels": {"seed": "42"}}
    prefix = "campaigns/test/runs/run/attempts/one"
    first = worker.archive_partial_metrics(store, output, prefix, status)
    assert worker.archive_partial_metrics(store, output, prefix, status) == first
    key = split_uri(first["manifest_uri"])[1]
    before = store.get_bytes(key)
    changed = output / "partial-system-metrics.json"
    changed.write_text('{"rss_bytes": 999}')
    with pytest.raises(CheckpointError, match="Immutable partial metric evidence differs"):
        worker.archive_partial_metrics(store, output, prefix, status)
    assert store.get_bytes(key) == before


def test_partial_archive_checks_stored_size_in_addition_to_hash_metadata(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    output = tmp_path / "run"
    write_observations(output, 42)
    status = {"campaign_id": "test", "run_id": "run", "attempt_key": "one", "labels": {"seed": "42"}}
    prefix = "campaigns/test/runs/run/attempts/one"
    worker.archive_partial_metrics(store, output, prefix, status)
    # Simulate an incomplete stored transfer retaining its original hash metadata.
    (store.root / prefix / "partial/partial-system-metrics.json").write_bytes(b"truncated")
    with pytest.raises(CheckpointError, match="Immutable partial metric evidence differs"):
        worker.archive_partial_metrics(store, output, prefix, status)


@pytest.mark.parametrize(
    "forbidden",
    [
        "models/nativeweights.json",
        "candidates/nativeweights.json",
        "nested/model.ubj",
        "weights.pkl",
        ".zia-resume/state.json",
    ],
)
def test_model_state_blocks_all_partial_uploads(tmp_path, monkeypatch, forbidden):
    store = LocalObjectStore(tmp_path / "store")
    output = tmp_path / "run"
    write_observations(output, 42)
    path = output / forbidden
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"learned_parameters": [1, 2]}')

    def forbidden_put(*_args, **_kwargs):
        pytest.fail("guard must run before even legitimate metrics are uploaded")

    monkeypatch.setattr(store, "put_file", forbidden_put)
    monkeypatch.setattr(store, "put_bytes", forbidden_put)
    with pytest.raises(ValueError, match="model or checkpoint state"):
        worker.archive_partial_metrics(store, output, "attempt", {})


def test_guard_rechecks_files_written_by_asynchronous_flush(tmp_path, monkeypatch):
    store = LocalObjectStore(tmp_path / "store")
    output = tmp_path / "run"
    write_observations(output, 42)

    class UnsafeTracker:
        def flush(self):
            (output / "weights.pkl").write_bytes(b"unexpected learned state")

    def forbidden_put(*_args, **_kwargs):
        pytest.fail("files emitted by flush must also pass the guard before any upload")

    monkeypatch.setattr(store, "put_file", forbidden_put)
    monkeypatch.setattr(store, "put_bytes", forbidden_put)
    with pytest.raises(ValueError, match="model or checkpoint state"):
        worker.archive_partial_metrics(store, output, "attempt", {}, tracker=UnsafeTracker())


@pytest.mark.parametrize("upload_fault", ["file", "manifest"])
def test_partial_upload_failure_retains_original_failure(metric_case, tmp_path, monkeypatch, upload_fault):
    store, unit, env = metric_case
    monkeypatch.setattr(worker, "GracefulTerminationHandler", FakeTermination)
    original_file, original_bytes = store.put_file, store.put_bytes

    def evaluate(_unit, _store, workdir, _log, **_kwargs):
        write_observations(workdir / "run", unit.params["seed"])
        FakeTermination.instances[-1].terminate()
        raise InterruptedError("original Spot interruption")

    def put_file(key, path, **kwargs):
        if upload_fault == "file" and "/partial/" in key:
            raise OSError("partial evidence upload unavailable")
        return original_file(key, path, **kwargs)

    def put_bytes(key, payload, **kwargs):
        if upload_fault == "manifest" and key.endswith("partial-metrics-manifest.json"):
            raise OSError("partial manifest upload unavailable")
        return original_bytes(key, payload, **kwargs)

    monkeypatch.setattr(worker, "execute_zia", evaluate)
    monkeypatch.setattr(store, "put_file", put_file)
    monkeypatch.setattr(store, "put_bytes", put_bytes)
    assert worker.run_unit(store, unit.unit_key, tmp_path / "first", env=env) == worker.EXIT_INTERRUPTED
    attempt = layout.attempt_key_for(env["AWS_BATCH_JOB_ID"], "1")
    status = get_json(store, layout.attempt_status_key(unit.campaign_id, unit.run_id, attempt))
    assert status["state"] == "INTERRUPTED"
    assert status["diagnostic"]["message"] == "original Spot interruption"
    assert status["diagnostic"]["partial_metrics_archive"]["error_type"] == "OSError"
    assert "partial_metrics" not in status
    assert not any(key.endswith("partial-metrics-manifest.json") for key in store.list())
    assert store.head(layout.final_key(unit.campaign_id, unit.run_id)) is None


def test_status_upload_failure_does_not_skip_partial_drain_or_replace_exit_code(
    metric_case, tmp_path, monkeypatch
):
    store, unit, env = metric_case
    monkeypatch.setattr(worker, "GracefulTerminationHandler", FakeTermination)
    original_bytes = store.put_bytes
    rejected = []

    def evaluate(_unit, _store, workdir, _log, **_kwargs):
        write_observations(workdir / "run", unit.params["seed"])
        FakeTermination.instances[-1].terminate()
        raise InterruptedError("original Spot interruption")

    def put_bytes(key, payload, **kwargs):
        if key.endswith("/status.json") and json.loads(payload).get("state") == "INTERRUPTED":
            rejected.append(key)
            raise OSError("status storage unavailable")
        return original_bytes(key, payload, **kwargs)

    monkeypatch.setattr(worker, "execute_zia", evaluate)
    monkeypatch.setattr(store, "put_bytes", put_bytes)
    assert worker.run_unit(store, unit.unit_key, tmp_path / "first", env=env) == worker.EXIT_INTERRUPTED
    assert len(rejected) == 2
    assert len([key for key in store.list() if key.endswith("partial-metrics-manifest.json")]) == 1
    assert store.head(layout.final_key(unit.campaign_id, unit.run_id)) is None


def test_failed_publication_probe_does_not_block_partial_metric_drain(metric_case, tmp_path, monkeypatch):
    store, unit, env = metric_case
    original_head = store.head
    pending_key = f"{layout.run_prefix(unit.campaign_id, unit.run_id)}/pending-publication.json"
    evaluation_failed = False
    failed_probes = []

    def evaluate(_unit, _store, workdir, _log, **_kwargs):
        nonlocal evaluation_failed
        write_observations(workdir / "run", unit.params["seed"])
        evaluation_failed = True
        raise RuntimeError("original evaluation failure")

    def head(key):
        # Earlier publication checks must succeed so this reproduces the
        # secondary HEAD failure only after the evaluator has already failed.
        if evaluation_failed and key == pending_key:
            failed_probes.append(key)
            raise OSError("pending publication HEAD unavailable")
        return original_head(key)

    monkeypatch.setattr(worker, "execute_zia", evaluate)
    monkeypatch.setattr(store, "head", head)
    assert worker.run_unit(store, unit.unit_key, tmp_path / "first", env=env) == worker.EXIT_FAILED
    assert failed_probes == [pending_key]
    attempt = layout.attempt_key_for(env["AWS_BATCH_JOB_ID"], "1")
    status = get_json(store, layout.attempt_status_key(unit.campaign_id, unit.run_id, attempt))
    assert status["state"] == "FAILED"
    assert status["diagnostic"]["error_type"] == "RuntimeError"
    assert status["diagnostic"]["message"] == "original evaluation failure"
    assert status["diagnostic"]["exit_code"] == worker.EXIT_FAILED
    assert status["diagnostic"]["publication_probe"] == {
        "error_type": "OSError",
        "message": "pending publication HEAD unavailable",
    }
    manifest = get_json(store, split_uri(status["partial_metrics"]["manifest_uri"])[1])
    assert manifest["complete"] is False
    assert "training-fits-partial/initial.json" in manifest["files"]
    assert original_head(layout.final_key(unit.campaign_id, unit.run_id)) is None
    assert original_head(pending_key) is None
