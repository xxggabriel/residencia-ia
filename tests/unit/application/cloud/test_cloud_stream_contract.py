"""Cloud contracts for complete seeded streams without persisted training state."""

import json
import os
import signal

import pytest

from continual_bench.application.cloud import layout, planning, worker
from continual_bench.application.cloud.contracts import (
    STREAM_METRICS_PROTOCOL,
    CampaignRequest,
    RunUnit,
)
from continual_bench.application.cloud.fixtures import publish_fixture_release, publish_fixture_snapshot
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json, put_json, split_uri


def seeded_case(tmp_path, *, dataset="social", policies=("frozen",)):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path, dataset)
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id="complete-seeded-stream",
        snapshot=snapshot.model_dump(),
        matrix={
            "kind": "zia",
            "policies": policies,
            "seeds": [42, 43, 44],
            "artifact_mode": "metrics_only",
            "iterative_training": {"max_epochs": 3, "patience": 3},
        },
    )
    plan = planning.build_plan(request, release, snapshot_manifest=verify_snapshot(store, snapshot))
    for row in plan["units"]:
        put_json(store, row["unit_key"], row)
    return store, release, plan


def fargate_env(release, attempt):
    return {
        "AWS_BATCH_JOB_ID": "seeded-stream-job",
        "AWS_BATCH_JOB_ATTEMPT": str(attempt),
        "BENCH_FARGATE_SPOT": "true",
        "BENCH_GIT_REVISION": release.git_sha,
        "BENCH_IMAGE_DIGEST": release.images["runtime"].digest,
        "BENCH_IMAGE_URI": release.images["runtime"].uri,
    }


def test_three_seeds_are_independent_execution_contracts(tmp_path):
    _, _, plan = seeded_case(tmp_path, policies=("frozen", "cara", "upf"))
    assert len(plan["units"]) == len({u["run_id"] for u in plan["units"]}) == 9
    for unit in plan["units"]:
        assert unit["contract"]["protocol"] == STREAM_METRICS_PROTOCOL
        assert unit["params"]["artifact_mode"] == "metrics_only"
        assert unit["params"]["seed"] == int(unit["labels"]["seed"])
        assert unit["params"]["iterative_training"]["seed"] == unit["params"]["seed"]
        assert unit["labels"]["model"] == "multinomial_logistic"


@pytest.mark.parametrize("seeds", [[42, 42], [-1], [2**31]])
def test_invalid_seed_sets_are_rejected(seeds):
    with pytest.raises(ValueError, match="seeds"):
        CampaignRequest(
            campaign_id="bad-seeds",
            dataset="electricity",
            scope="full",
            source_digests={"electricity": "a" * 64},
            matrix={"kind": "stream", "seeds": seeds},
        )


def test_prepared_stream_plan_binds_the_whole_verified_descriptor(tmp_path):
    request = CampaignRequest(
        campaign_id="electricity-full",
        dataset="electricity",
        scope="full",
        source_digests={"electricity": "a" * 64},
        matrix={
            "kind": "stream",
            "policies": ["frozen"],
            "seeds": [42, 43, 44],
            "artifact_mode": "metrics_only",
        },
    )
    descriptor = {
        "dataset": "electricity",
        "scope": "full",
        "prepared_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "rows": 45312,
    }
    release = publish_fixture_release(LocalObjectStore(tmp_path))
    plan = planning.build_plan(request, release, stream_manifest=descriptor)
    assert len(plan["units"]) == 3 and plan["snapshot"] is None
    assert all(u["contract"]["stream_source"] == descriptor for u in plan["units"])
    assert all("snapshot" not in u["params"] for u in plan["units"])
    with pytest.raises(ValueError, match="differs"):
        planning.build_plan(request, release, stream_manifest={**descriptor, "prepared_sha256": "c" * 64})


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("fault", ["interruption", "publication"])
def test_stream_spot_restarts_without_weights_and_recovers_metric_publication(
    tmp_path,
    monkeypatch,
    seed,
    fault,
):
    store, release, plan = seeded_case(tmp_path)
    unit = RunUnit(**next(u for u in plan["units"] if u["params"]["seed"] == seed))
    evaluate = worker.execute_zia
    visits = []
    publication_failed = False
    original_put = store.put_file

    def forbid(*args, **kwargs):
        pytest.fail("metrics-only stream attempted to persist/load/register model state")

    def execute(*args, **kwargs):
        visits.append(True)
        assert kwargs.get("checkpoint") is None
        if fault == "interruption" and len(visits) == 1:
            os.kill(os.getpid(), signal.SIGTERM)
            raise InterruptedError("new attempt must restart the evaluator")
        return evaluate(*args, **kwargs)

    def fail_publication(key, path, **kwargs):
        nonlocal publication_failed
        if fault == "publication" and "/attempts/" in key and not publication_failed:
            publication_failed = True
            raise OSError("canonical evidence transfer interrupted")
        return original_put(key, path, **kwargs)

    monkeypatch.setattr(worker, "SpotCheckpoint", forbid)
    monkeypatch.setattr(worker, "verify_native_loader", forbid)
    monkeypatch.setattr(worker, "model_tarball", forbid)
    monkeypatch.setattr(worker, "execute_zia", execute)
    monkeypatch.setattr(store, "put_file", fail_publication)
    first = worker.run_unit(store, unit.unit_key, tmp_path / "first", env=fargate_env(release, 1))
    assert first == (worker.EXIT_INTERRUPTED if fault == "interruption" else worker.EXIT_FAILED)
    assert (
        worker.run_unit(
            store,
            unit.unit_key,
            tmp_path / "second",
            env=fargate_env(release, 2),
            registry=forbid,
        )
        == worker.EXIT_OK
    )
    assert len(visits) == (2 if fault == "interruption" else 1)
    final = get_json(store, layout.final_key(unit.campaign_id, unit.run_id))
    assert final["artifact_mode"] == "metrics_only" and final["model"] is None
    assert final["attempt"]["checkpoint"] is None
    assert final["labels"]["seed"] == str(seed)
    report = get_json(store, split_uri(final["report_uri"])[1])
    assert report["contract"]["iterative_training"]["seed"] == seed
    assert all(not (fit.get("checkpoint") or {}).get("file") for fit in report["training_history"])
    assert all(not version.get("file") for version in report["model_versions"])
    assert not any("/checkpoints/" in key for key in store.list("campaigns/"))
    assert not any("model.tar.gz" in key or "model-metadata.json" in key for key in store.list("campaigns/"))
    for root in (
        tmp_path / "first" / "run",
        tmp_path / "second" / "run",
        tmp_path / "second" / "publication",
    ):
        verify_metrics_only_files(root)
    ledger = json.loads(store.get_bytes(final["published"]["prefix"] + "/live-metrics.jsonl").splitlines()[0])
    assert ledger["prefix"].startswith("attempts/") and ledger["step"] == 0


def test_matrix_cache_requires_completed_same_store_evidence(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    key = "campaigns/source/runs/" + "a" * 20 + "/attempts/task-1/matrix-metrics.json.gz"
    path = tmp_path / "matrix-metrics.json.gz"
    path.write_bytes(b"metrics only")
    stored = store.put_file(key, path)
    reference = {"uri": store.uri(key), "sha256": stored.sha256}
    with pytest.raises(FileNotFoundError):
        worker.verify_matrix_metric_cache(store, reference)
    put_json(
        store,
        layout.final_key("source", "a" * 20),
        {
            "artifact_mode": "metrics_only",
            "outputs_uri": store.uri(key.rpartition("/")[0]),
        },
    )
    assert worker.verify_matrix_metric_cache(store, reference) == key
    with pytest.raises(worker.ContractError, match="SHA-256"):
        worker.verify_matrix_metric_cache(store, {**reference, "sha256": "b" * 64})
    with pytest.raises(worker.ContractError, match="outside"):
        worker.verify_matrix_metric_cache(
            store,
            {
                **reference,
                "uri": "s3://another/" + key,
            },
        )


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_cached_policy_uses_the_completed_candidate_metrics_without_loading_weights(tmp_path, seed):
    store, release, plan = seeded_case(tmp_path, policies=("cara",))
    original = RunUnit(**next(u for u in plan["units"] if u["params"]["seed"] == seed))
    assert (
        worker.run_unit(store, original.unit_key, tmp_path / "source", env=fargate_env(release, 1))
        == worker.EXIT_OK
    )
    source_final = get_json(store, layout.final_key(original.campaign_id, original.run_id))
    cache_uri = source_final["outputs_uri"] + "/matrix-metrics.json.gz"
    cache_sha = store.head(split_uri(cache_uri)[1]).sha256
    request = CampaignRequest(
        campaign_id="cached-upf-policy",
        snapshot=original.params["snapshot"],
        matrix={
            "kind": "zia",
            "policies": ["upf"],
            "seeds": [seed],
            "artifact_mode": "metrics_only",
            "iterative_training": {"max_epochs": 3, "patience": 3},
            "metric_cache_sources": {str(seed): {"uri": cache_uri, "sha256": cache_sha}},
        },
    )
    cached_plan = planning.build_plan(
        request,
        release,
        snapshot_manifest=worker.zia_snapshot(store, original),
    )
    unit = RunUnit(**cached_plan["units"][0])
    put_json(store, unit.unit_key, unit.model_dump(mode="json"))
    env = {**fargate_env(release, 1), "AWS_BATCH_JOB_ID": "cached-policy-job"}
    assert worker.run_unit(store, unit.unit_key, tmp_path / "cached", env=env) == worker.EXIT_OK
    final = get_json(store, layout.final_key(unit.campaign_id, unit.run_id))
    report = get_json(store, split_uri(final["report_uri"])[1])
    assert report["contract"]["matrix_metrics_cache"]["reused"] is True
    assert report["contract"]["matrix_metrics_cache"]["sha256"] == cache_sha
    reused = [
        fit for fit in report["training_history"] if fit.get("execution_origin") == "reused_metric_evidence"
    ]
    assert reused and all(fit["source_matrix"]["seed"] == seed for fit in reused)
    assert all(not (fit.get("checkpoint") or {}).get("file") for fit in report["training_history"])
    assert final["model"] is None
