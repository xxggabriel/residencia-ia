"""Campaign plans, deterministic run IDs and the one-run worker (protocol unchanged)."""

import json
from pathlib import Path

import pytest
import yaml

from continual_bench.application.cloud import layout, planning
from continual_bench.application.cloud.contracts import CampaignRequest
from continual_bench.application.cloud.fixtures import (
    FIXTURE_SHA,
    publish_fixture_release,
    publish_fixture_snapshot,
)
from continual_bench.application.cloud.snapshots import verify_snapshot
from continual_bench.application.cloud.worker import EXIT_CONTRACT, EXIT_OK, EXIT_SNAPSHOT, run_unit
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json

QUEUE = {"queue_state": "ENABLED", "queue_status": "VALID", "max_vcpus": 8}


@pytest.fixture
def campaign(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    snapshot = publish_fixture_snapshot(store, tmp_path)
    release = publish_fixture_release(store)
    request = CampaignRequest(campaign_id="social-smoke", snapshot=snapshot.model_dump())
    plan = planning.build_plan(request, release, snapshot_manifest=verify_snapshot(store, snapshot))
    return store, snapshot, release, request, plan


def test_zia_plan_has_four_independent_units(campaign):
    store, _, release, request, plan = campaign
    assert [u["labels"]["policy"] for u in plan["units"]] == ["frozen", "drift", "incremental", "periodic"]
    assert (
        len({u["run_id"] for u in plan["units"]}) == 4
        and plan["concurrency"] == 4
        and plan["mode"] == "inline"
    )
    assert all(u["job_name"] == f"cb-social-smoke-{u['run_id']}" for u in plan["units"])
    assert all(
        u["contract"]["release"]["job_definition"] == release.job_definitions["benchmark"]
        for u in plan["units"]
    )
    again = planning.build_plan(request, release, snapshot_manifest=verify_snapshot(store, campaign[1]))
    assert again == plan
    other = publish_fixture_release(store, git_sha="2" * 40)
    moved = planning.build_plan(request, other, snapshot_manifest=verify_snapshot(store, campaign[1]))
    assert {u["run_id"] for u in moved["units"]}.isdisjoint({u["run_id"] for u in plan["units"]})


def test_plan_is_published_once_before_submission(campaign):
    store, snapshot, release, request, plan = campaign
    published = planning.publish_plan(store, plan)
    assert published["units"] == 4 and len(published["items"]) == 4
    item = published["items"][0]
    assert set(item) == {
        "campaign_id",
        "run_id",
        "job_name",
        "unit_uri",
        "job_definition",
        "max_attempts",
        "task_timeout_seconds",
        "labels",
    }
    assert planning.publish_plan(store, plan)["plan_uri"] == published["plan_uri"]
    changed = planning.build_plan(
        request.model_copy(update={"max_attempts": 3}),
        release,
        snapshot_manifest=verify_snapshot(store, snapshot),
    )
    with pytest.raises(planning.CampaignConflict):
        planning.publish_plan(store, changed)


def test_latest_release_resolution_is_pinned_in_the_plan(campaign):
    store, *_ = campaign
    assert planning.resolve_release(store, "latest").git_sha == FIXTURE_SHA
    publish_fixture_release(store, git_sha="3" * 40)
    assert planning.resolve_release(store, "latest").git_sha == "3" * 40
    assert planning.resolve_release(store, FIXTURE_SHA).git_sha == FIXTURE_SHA


def test_capacity_admission():
    evidence = planning.admit_capacity(4, 1, quota_vcpus=6, queue=QUEUE)
    assert evidence["max_parallel_admitted"] == 6
    with pytest.raises(planning.CapacityError, match="compute environment"):
        planning.admit_capacity(9, 1, quota_vcpus=100, queue=QUEUE)
    with pytest.raises(planning.CapacityError, match="quota"):
        planning.admit_capacity(4, 1, quota_vcpus=2, queue=QUEUE)
    with pytest.raises(planning.CapacityError, match="queue"):
        planning.admit_capacity(1, 1, quota_vcpus=8, queue={**QUEUE, "queue_state": "DISABLED"})


def test_existing_experiment_matrix_becomes_independent_units(tmp_path):
    store = LocalObjectStore(tmp_path)
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id="matrix",
        matrix={"kind": "experiment_matrix", "path": "configs/matrices/binary-comparison.yaml"},
    )
    plan = planning.build_plan(request, release)
    from continual_bench.application.matrix import ExperimentMatrix
    from continual_bench.config import load_model

    expected = load_model(ExperimentMatrix, "configs/matrices/binary-comparison.yaml").plan()
    assert len(plan["units"]) == len(expected["cases"]) and plan["mode"] == "distributed"
    assert {u["kind"] for u in plan["units"]} == {"matrix_case"}
    assert len({u["run_id"] for u in plan["units"]}) == len(plan["units"])


def test_matrix_without_a_published_source_binding_is_refused(tmp_path):
    matrix = tmp_path / "configs" / "matrices" / "online.yaml"
    matrix.parent.mkdir(parents=True)
    matrix.write_text(
        yaml.safe_dump({"name": "online", "datasets": [{"name": "electricity", "samples": 300}]})
    )
    (tmp_path / "configs" / "base").mkdir()
    (tmp_path / "configs" / "base" / "tracking.yaml").write_text(
        Path("configs/base/tracking.yaml").read_text()
    )
    store = LocalObjectStore(tmp_path / "s")
    request = CampaignRequest(
        campaign_id="m", matrix={"kind": "experiment_matrix", "path": "configs/matrices/online.yaml"}
    )
    with pytest.raises(ValueError, match="source_digests for: electricity"):
        planning.build_plan(request, publish_fixture_release(store), project_root=tmp_path)


def test_project_suite_preserves_all_cases_and_binds_external_inputs(tmp_path):
    from continual_bench.datasets.catalog import CATALOG

    sources = {
        spec.name: "a" * 64 for spec in CATALOG
        if spec.family in {"openml", "real_stream", "image", "temporal_vision"}
    }
    request = CampaignRequest(
        campaign_id="suite", matrix={"kind": "project_suite"}, source_digests=sources,
    )
    plan = planning.build_plan(request, publish_fixture_release(LocalObjectStore(tmp_path)))
    assert plan["mode"] == "distributed" and len(plan["units"]) == 548
    bound = [unit for unit in plan["units"] if unit["params"].get("source_sha256")]
    assert len(bound) == 113 and {u["labels"]["dataset"] for u in bound} == sources.keys()
    assert len({u["run_id"] for u in plan["units"]}) == 548


def test_worker_executes_a_checksum_bound_external_epoch_source(tmp_path, monkeypatch):
    from continual_bench.application.cloud import matrix_sources
    from continual_bench.core.contracts import TaskSample
    from continual_bench.datasets.images import PreparedImageSource

    source = PreparedImageSource(
        [
            TaskSample(str(i), i, i, i, {"f": float(i % 2)}, str(i % 2),
                       split="train" if i < 80 else "test")
            for i in range(100)
        ],
        {"name": "insects", "task": "multiclass", "labels": ["0", "1"], "version": "fixture"},
    )
    monkeypatch.setattr(matrix_sources, "load_dataset", lambda *a, **kw: source)
    store = LocalObjectStore(tmp_path / "store")
    binding = matrix_sources.publish_source(store, "insects", cache=tmp_path, workdir=tmp_path,
                                            examples=100, seed=42)
    path = tmp_path / "configs" / "matrices"
    path.mkdir(parents=True)
    (path / "epochs.yaml").write_text(
        "name: epochs\nevaluator: epochs\nmodels: [{name: hoeffding_tree}]\n"
        "datasets: [{name: insects, samples: 100}]\n"
    )
    (tmp_path / "configs" / "base").mkdir()
    (tmp_path / "configs" / "base" / "tracking.yaml").write_text("{}")
    request = CampaignRequest(
        campaign_id="epochs", matrix={"kind": "experiment_matrix", "path": "configs/matrices/epochs.yaml"},
        source_digests={"insects": binding["sha256"]},
    )
    plan = planning.build_plan(request, publish_fixture_release(store), project_root=tmp_path)
    matrix_sources.verify_plan_sources(store, plan["units"])
    plan["units"][0]["params"]["config"]["examples"] = 101
    with pytest.raises(ValueError, match="count and seed"):
        matrix_sources.verify_plan_sources(store, plan["units"])
    plan["units"][0]["params"]["config"]["examples"] = 100
    planning.publish_plan(store, plan)
    assert run_unit(store, plan["units"][0]["unit_key"], tmp_path / "worker", env=_env()) == EXIT_OK
    rows = store.get_bytes(layout.analytics_key("runs", "epochs", plan["units"][0]["run_id"]))
    assert json.loads(rows)["model"] == "hoeffding_tree" and json.loads(rows)["seed"] == 42


def test_openml_is_materialized_from_pinned_arff_without_changing_the_run_config(tmp_path, monkeypatch):
    from continual_bench.application.cloud import matrix_sources
    from continual_bench.config import Config
    from continual_bench.datasets.openml import REAL_SOURCES
    from continual_bench.infrastructure.object_store import sha256_path

    raw = tmp_path / "electricity.arff"
    raw.write_text(
        "@relation fixture\n@attribute x numeric\n@attribute class {UP,DOWN}\n@data\n"
        + "0,UP\n1,DOWN\n" * 150
    )
    monkeypatch.setitem(REAL_SOURCES, "electricity", {
        **REAL_SOURCES["electricity"], "sha256": sha256_path(raw), "rows": 300,
        "columns": ["x", "class"], "nominal": ["class"],
    })
    store = LocalObjectStore(tmp_path / "store")
    result = matrix_sources.publish_source(
        store, "electricity", cache=tmp_path, workdir=tmp_path, real_directory=tmp_path,
        examples=100, seed=42,
    )
    config = Config(dataset={"name": "electricity", "samples": 300})
    source = matrix_sources.materialize_source(
        store, "electricity", result["sha256"], tmp_path / "private", config, "binary"
    )
    assert config.dataset.params == {}
    assert len(list(source.events())) == 300
    assert source.path == tmp_path / "private" / "electricity.arff"


def _env(index=0, digest=None):
    return {
        "AWS_BATCH_JOB_ID": f"job-{index}",
        "AWS_BATCH_JOB_ATTEMPT": "1",
        **({"BENCH_IMAGE_DIGEST": digest} if digest else {}),
    }


def test_worker_runs_publishes_and_reuses(campaign, tmp_path):
    store, snapshot, release, _, plan = campaign
    planning.publish_plan(store, plan)
    unit = plan["units"][2]  # incremental
    emitted = []
    code = run_unit(
        store,
        unit["unit_key"],
        tmp_path / "w",
        env=_env(0, release.images["runtime"].digest),
        emit=lambda *a, **k: emitted.append(a),
    )
    assert code == EXIT_OK
    final = get_json(store, layout.final_key("social-smoke", unit["run_id"]))
    assert final["attempt"]["job_id"] == "job-0" and final["labels"]["policy"] == "incremental"
    assert final["model"]["loader_check"]["rows"] == 256
    assert final["model"]["loader_check"]["max_abs_difference"] <= 1e-9
    prefix = layout.attempt_prefix("social-smoke", unit["run_id"], "job-0-1")
    for name in (
        "report.json",
        "predictions-manifest.json",
        "model/model.tar.gz",
        "artifacts/data/index.json",
        "model-metadata.json",
        "catalog-links.json",
    ):
        assert store.head(f"{prefix}/{name}") is not None, name
    metrics = (
        store.get_bytes(layout.analytics_key("metrics", "social-smoke", unit["run_id"])).decode().splitlines()
    )
    rows = [json.loads(line) for line in metrics]
    assert {r["partition"] for r in rows} >= {"prequential", "holdout"}
    windows = [
        json.loads(x)
        for x in store.get_bytes(layout.analytics_key("windows", "social-smoke", unit["run_id"]))
        .decode()
        .splitlines()
    ]
    assert all(w["step"] == w["end_seq"] for w in windows) and {w["partition"] for w in windows} == {
        "prequential",
        "holdout",
    }
    assert emitted and "run_id" not in emitted[0][1]
    assert run_unit(store, store.uri(unit["unit_key"]), tmp_path / "w2", env=_env(1)) == EXIT_OK
    status = get_json(store, layout.attempt_status_key("social-smoke", unit["run_id"], "job-1-1"))
    assert status["state"] == "REUSED"
    assert get_json(store, layout.final_key("social-smoke", unit["run_id"]))["attempt"]["job_id"] == "job-0"


def test_worker_matches_the_local_protocol(campaign, tmp_path):
    from continual_bench.config import TrackingConfig
    from continual_bench.evaluation.zia import run

    store, snapshot, release, _, plan = campaign
    planning.publish_plan(store, plan)
    unit = plan["units"][1]  # drift
    assert run_unit(store, unit["unit_key"], tmp_path / "aws", env=_env()) == EXIT_OK
    prefix = layout.attempt_prefix("social-smoke", unit["run_id"], "job-0-1")
    aws_report = json.loads(store.get_bytes(f"{prefix}/report.json"))
    aws_predictions = json.loads(store.get_bytes(f"{prefix}/predictions-manifest.json"))
    settings = TrackingConfig.model_validate(
        yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    ).model_dump()
    local = run(tmp_path / "aws" / "data", "social", "smoke", "drift", tmp_path / "local", settings)
    local_predictions = json.loads((tmp_path / "local" / "predictions-manifest.json").read_text())
    volatile = {"prediction_seconds", "prediction_throughput_per_second"}
    strip = [{k: v for k, v in w.items() if k not in volatile} for w in local["windows"]]
    assert strip == [{k: v for k, v in w.items() if k not in volatile} for w in aws_report["windows"]]
    assert [p["sha256"] for p in local_predictions["parts"]] == [
        p["sha256"] for p in aws_predictions["parts"]
    ]
    assert [v["sha256"] for v in local["model_versions"]] == [
        v["sha256"] for v in aws_report["model_versions"]
    ]
    assert local["contract"]["holdout_start"] == aws_report["contract"]["holdout_start"]
    assert local["holdout_confusion"] == aws_report["holdout_confusion"]


def test_worker_failures_return_specific_exit_codes(campaign, tmp_path):
    store, snapshot, release, _, plan = campaign
    planning.publish_plan(store, plan)
    unit_b = plan["units"][3]
    marker = store.root / layout.snapshot_marker_key(snapshot)
    saved = marker.read_bytes()
    marker.unlink()
    assert run_unit(store, unit_b["unit_key"], tmp_path / "b", env=_env(5)) == EXIT_SNAPSHOT
    status = get_json(store, layout.attempt_status_key("social-smoke", unit_b["run_id"], "job-5-1"))
    assert status["state"] == "FAILED" and status["diagnostic"]["error_type"] == "SnapshotNotReady"
    marker.write_bytes(saved)
    assert store.head(layout.final_key("social-smoke", unit_b["run_id"])) is None
    assert run_unit(store, unit_b["unit_key"], tmp_path / "c", env=_env(6)) == EXIT_OK
    assert (
        get_json(store, layout.attempt_status_key("social-smoke", unit_b["run_id"], "job-5-1"))["state"]
        == "FAILED"
    )


def test_worker_refuses_another_release_image(campaign, tmp_path):
    store, _, _, _, plan = campaign
    planning.publish_plan(store, plan)
    unit = plan["units"][0]
    assert (
        run_unit(store, unit["unit_key"], tmp_path / "x", env=_env(7, "sha256:" + "9" * 64)) == EXIT_CONTRACT
    )
    status = get_json(store, layout.attempt_status_key("social-smoke", unit["run_id"], "job-7-1"))
    assert status["diagnostic"]["error_type"] == "ContractError"


def test_loader_sample_reads_only_the_needed_row_groups(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from continual_bench.application.cloud.worker import read_row_range

    path = tmp_path / "groups.parquet"
    pq.write_table(
        pa.table({"seq": list(range(95)), "x": [float(i) for i in range(95)]}), path, row_group_size=10
    )
    calls = []
    original = pq.ParquetFile.read_row_group
    monkeypatch.setattr(
        pq.ParquetFile, "read_row_group", lambda self, i, **kw: calls.append(i) or original(self, i, **kw)
    )
    assert read_row_range(path, ["seq"], 17, 30).column("seq").to_pylist() == list(range(17, 47))
    assert calls == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="outside"):
        read_row_range(path, ["seq"], 200, 5)


def test_analytics_publication_is_repaired_after_finalization(campaign, tmp_path, monkeypatch):
    from continual_bench.application.cloud import lifecycle, records

    store, snapshot, release, _, plan = campaign
    planning.publish_plan(store, plan)
    unit = plan["units"][0]
    original, calls = records.publish_rows, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("transient storage error after the final manifest")
        return original(*args, **kwargs)

    monkeypatch.setattr(records, "publish_rows", flaky)
    assert run_unit(store, unit["unit_key"], tmp_path / "a", env=_env(0)) != EXIT_OK
    assert store.head(layout.final_key("social-smoke", unit["run_id"])) is not None
    assert store.head(layout.analytics_marker_key("social-smoke", unit["run_id"])) is None
    assert store.head(layout.analytics_key("metrics", "social-smoke", unit["run_id"])) is None
    assert run_unit(store, unit["unit_key"], tmp_path / "b", env=_env(1)) == EXIT_OK
    reused = get_json(store, layout.attempt_status_key("social-smoke", unit["run_id"], "job-1-1"))
    assert reused["state"] == "REUSED" and reused["analytics"] == "repaired"
    marker = get_json(store, layout.analytics_marker_key("social-smoke", unit["run_id"]))
    assert marker["attempt_key"] == "job-0-1"
    rows = store.get_bytes(layout.analytics_key("metrics", "social-smoke", unit["run_id"])).decode()
    assert '"attempt_key": "job-0-1"' in rows
    summary = lifecycle.consolidate(store, "social-smoke")
    assert summary["succeeded"] == 1
    assert get_json(store, layout.summary_key("social-smoke"))["units"][0]["analytics"] == "present"
