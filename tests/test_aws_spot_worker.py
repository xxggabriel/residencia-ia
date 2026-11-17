"""Worker recovery against an object store with the same conditional-create contract as S3."""

import json

import pytest

from continual_bench.application.cloud import layout, planning, worker
from continual_bench.application.cloud.checkpoints import SpotCheckpoint
from continual_bench.application.cloud.contracts import CampaignRequest, RunUnit
from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json, put_json, split_uri


@pytest.fixture
def spot_case(tmp_path):
    store = LocalObjectStore(tmp_path / "durable")
    release = publish_fixture_release(store)
    request = CampaignRequest(campaign_id="spot-worker-test", matrix={
        "kind": "experiment_matrix", "path": "configs/matrices/binary-comparison.yaml",
    })
    plan = planning.build_plan(request, release)
    unit = RunUnit(**plan["units"][0])
    put_json(store, unit.unit_key, unit.model_dump(mode="json"))
    return store, unit, release


def grant(store, unit, release, attempt, notice, **extra):
    instance_id = f"i-{attempt:017d}"
    put_json(
        store, layout.spot_instance_key(unit.campaign_id, unit.run_id, attempt),
        {"instance_id": instance_id, "attempt": attempt, "log_group": "worker-logs",
         "log_stream": f"campaign/{instance_id}", **extra},
        if_absent=True,
    )
    return {
        "BENCH_EC2_INSTANCE_ID": instance_id,
        "BENCH_EC2_ATTEMPT": str(attempt),
        "BENCH_SPOT_NOTICE_FILE": str(notice),
        "BENCH_GIT_REVISION": release.git_sha,
        "BENCH_IMAGE_DIGEST": release.images["runtime"].digest,
        "BENCH_IMAGE_URI": release.images["runtime"].uri,
    }


def test_worker_restores_the_last_complete_and_valid_checkpoint(spot_case, tmp_path, monkeypatch):
    store, unit, _ = spot_case
    output = tmp_path / "first" / "run"
    output.mkdir(parents=True)
    predictions = output / "predictions"
    predictions.mkdir()
    first_part = predictions / "part-1"
    first_part.write_bytes(b"predictions 1")
    first = SpotCheckpoint(store, unit, output, 1, requested=lambda: False)
    confirmed = first.save({"cursor": 2})
    assert confirmed and not first.due()
    second_part = predictions / "part-2"
    second_part.write_bytes(b"unconfirmed predictions 2")
    original_put = store.put_bytes

    def fail_manifest(key, data, **kwargs):
        if key.startswith(first.manifests):
            raise OSError("instance disappeared after object uploads")
        return original_put(key, data, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store, "put_bytes", fail_manifest)
        with pytest.raises(OSError):
            first.save({"cursor": 3}, force=True)
    second = SpotCheckpoint(store, unit, tmp_path / "second" / "run", 2, requested=lambda: False)
    assert second.load() == {"cursor": 2}
    assert second.resumed == confirmed
    assert sorted(p.name for p in (second.output / "predictions").iterdir()) == ["part-1"]

    (second.output / "predictions" / second_part.name).write_bytes(b"corrupt newer checkpoint")
    bad_checkpoint = second.save({"cursor": 3}, force=True)
    manifest = get_json(store, split_uri(bad_checkpoint)[1])
    digest = manifest["files"]["predictions/part-2"]["sha256"]
    (store.root / second.root / "blobs" / digest).write_bytes(b"corrupted transfer")
    third = SpotCheckpoint(store, unit, tmp_path / "third" / "run", 3, requested=lambda: False)
    assert third.load() == {"cursor": 2}
    assert third.resumed == confirmed
    assert (third.output / "predictions" / "part-1").read_bytes() == b"predictions 1"
    assert not (third.output / "predictions" / "part-2").exists()


class HostLost(BaseException):
    """Simulate a host loss that cannot run exception/finalization handlers."""


def evaluate_result(root, values):
    directory = root / "run" / "binary" / "spot-test"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    report = {"run_key": "spot-test", "predictions": values}
    report_path.write_text(json.dumps(report))
    (directory / "model.bin").write_bytes(bytes(values))
    return {
        "report": report, "report_path": report_path,
        "output": root / "run", "headline": {"accuracy": 0.5},
    }


@pytest.mark.parametrize("fault", ["notice", "no_notice", "publication"])
def test_spot_recovery_keeps_results_and_original_publication_path(
    spot_case, tmp_path, monkeypatch, fault,
):
    store, unit, release = spot_case
    notice_file = tmp_path / "notice.json"
    first = grant(store, unit, release, 1, notice_file)
    first_attempt = layout.attempt_key_for(first["BENCH_EC2_INSTANCE_ID"], 1)
    first_status = layout.attempt_status_key(unit.campaign_id, unit.run_id, first_attempt)
    visits = []
    calls = 0

    def evaluate(_store, _unit, workdir, *, checkpoint):
        nonlocal calls
        calls += 1
        restored = checkpoint.load()
        state = restored or {"predictions": []}
        for n in range(len(state["predictions"]), 4):
            visits.append(n)
            state["predictions"].append(n)
            if n == 1:
                checkpoint.save(state, force=True)
            if calls == 1 and n == 2:
                if fault == "notice":
                    notice_file.touch()
                    assert checkpoint.requested()
                    checkpoint.save(state, force=True)
                    raise InterruptedError("interruption notice")
                if fault == "no_notice":
                    raise HostLost()
        return evaluate_result(workdir, state["predictions"])

    original_put = store.put_file
    canonical_writes = []

    def flaky_put(key, path, **kwargs):
        if "/attempts/" in key:
            canonical_writes.append(key)
            if fault == "publication" and len(canonical_writes) == 2:
                raise OSError("partial S3 publication")
        return original_put(key, path, **kwargs)

    monkeypatch.setattr(worker, "execute_matrix_case", evaluate)
    monkeypatch.setattr(store, "put_file", flaky_put)
    if fault == "no_notice":
        with pytest.raises(HostLost):
            worker.run_unit(store, unit.unit_key, tmp_path / "first", env=first)
        assert get_json(store, first_status)["state"] == "RUNNING"
    else:
        exit_code = worker.run_unit(store, unit.unit_key, tmp_path / "first", env=first)
        status = get_json(store, first_status)
        if fault == "notice":
            assert exit_code == worker.EXIT_INTERRUPTED
            assert status["interruption_reason"] == "spot_interruption_notice"
            assert status["checkpoint"].endswith("/000001/00000002.json")
        else:
            assert exit_code == worker.EXIT_FAILED
            assert status["state"] == "PUBLICATION_FAILED"
    final_key = layout.final_key(unit.campaign_id, unit.run_id)
    assert store.head(final_key) is None

    second = grant(
        store, unit, release, 2, tmp_path / "next-notice",
        previous_attempt_key=first_attempt, recovery_reason=fault,
    )
    assert worker.run_unit(store, unit.unit_key, tmp_path / "second", env=second) == worker.EXIT_OK
    final = get_json(store, final_key)
    assert final["publication_attempt"]["instance_id"] == second["BENCH_EC2_INSTANCE_ID"]
    assert store.head(layout.analytics_marker_key(unit.campaign_id, unit.run_id))
    report = get_json(store, split_uri(final["report_uri"])[1])
    assert report["predictions"] == [0, 1, 2, 3]
    if fault == "publication":
        original_prefix = layout.attempt_prefix(unit.campaign_id, unit.run_id, first_attempt)
        assert final["published"]["prefix"] == original_prefix
        assert calls == 1
        assert len(set(canonical_writes)) == final["published"]["files"]
        assert len(list(store.list(original_prefix + "/"))) == final["published"]["files"] + 1
    else:
        assert calls == 2
        assert visits == ([0, 1, 2, 3] if fault == "notice" else [0, 1, 2, 2, 3])
        assert final["attempt"]["resumed_checkpoint"] is not None
        assert final["attempt"]["resumed_checkpoint"] != store.uri(final_key)
    reused = worker.run_unit(store, unit.unit_key, tmp_path / "duplicate", env=second)
    assert reused == worker.EXIT_CONTRACT


@pytest.mark.parametrize("blocker", ["cancelled", "replaced"])
def test_ineligible_attempt_cannot_start_evaluation(spot_case, tmp_path, monkeypatch, blocker):
    store, unit, release = spot_case
    env = grant(store, unit, release, 1, tmp_path / "notice")
    if blocker == "cancelled":
        put_json(store, layout.cancelled_key(unit.campaign_id), {"state": "CANCELLED"}, if_absent=True)
        expected = worker.EXIT_CANCELLED
    else:
        grant(store, unit, release, 2, tmp_path / "next-notice")
        expected = worker.EXIT_CONTRACT

    def forbidden(*args, **kwargs):
        pytest.fail("ineligible attempt entered evaluator")

    monkeypatch.setattr(worker, "execute_matrix_case", forbidden)
    assert worker.run_unit(store, unit.unit_key, tmp_path / "blocked", env=env) == expected
    assert store.head(layout.final_key(unit.campaign_id, unit.run_id)) is None


def test_zia_spot_worker_recovers_real_predictions_and_fit_lineage(tmp_path, monkeypatch):
    store = LocalObjectStore(tmp_path / "durable")
    snapshot = publish_fixture_snapshot(store, tmp_path)
    release = publish_fixture_release(store)
    snapshot_manifest = verify_snapshot(store, snapshot)

    def unit_for(campaign_id):
        request = CampaignRequest(
            campaign_id=campaign_id, snapshot=snapshot.model_dump(), matrix={
                "kind": "zia", "policies": ["incremental"],
            },
        )
        plan = planning.build_plan(request, release, snapshot_manifest=snapshot_manifest)
        unit = RunUnit(**plan["units"][0])
        put_json(store, unit.unit_key, unit.model_dump(mode="json"))
        return unit

    reference = unit_for("spot-reference")
    ref_env = grant(store, reference, release, 1, tmp_path / "ref-notice")
    assert worker.run_unit(store, reference.unit_key, tmp_path / "reference", env=ref_env) == worker.EXIT_OK

    candidate = unit_for("spot-resumed")
    first = grant(store, candidate, release, 1, tmp_path / "first-notice")
    original = SpotCheckpoint.save

    def crash_at_confirmed_checkpoint(self, state, **kwargs):
        confirmed = original(self, state, **kwargs)
        if confirmed:
            raise HostLost()
        return confirmed

    with monkeypatch.context() as patch:
        patch.setattr(SpotCheckpoint, "save", crash_at_confirmed_checkpoint)
        with pytest.raises(HostLost):
            worker.run_unit(store, candidate.unit_key, tmp_path / "first", env=first)
    second = grant(store, candidate, release, 2, tmp_path / "second-notice", recovery_reason="instance_lost")
    assert worker.run_unit(store, candidate.unit_key, tmp_path / "resumed", env=second) == worker.EXIT_OK

    def documents(unit):
        final = get_json(store, layout.final_key(unit.campaign_id, unit.run_id))
        report = get_json(store, split_uri(final["report_uri"])[1])
        predictions = get_json(store, split_uri(final["predictions_manifest_uri"])[1])
        return final, report, predictions

    _, expected_report, expected_predictions = documents(reference)
    final, actual_report, actual_predictions = documents(candidate)
    assert final["attempt"]["resumed_checkpoint"].endswith("/000001/00000001.json")
    assert [part["sha256"] for part in actual_predictions["parts"]] == [
        part["sha256"] for part in expected_predictions["parts"]
    ]
    assert [(fit["fit_id"], fit["checkpoint"]["sha256"]) for fit in actual_report["training_history"]] == [
        (fit["fit_id"], fit["checkpoint"]["sha256"]) for fit in expected_report["training_history"]
    ]
    assert actual_report["holdout_confusion"] == expected_report["holdout_confusion"]
    assert not any("/.zia-resume/" in key for key in store.list(final["published"]["prefix"] + "/"))
