"""Verify matrix publication/recovery and native train evidence without cloud services."""

import json
import math
import runpy
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from continual_bench.application.cloud import layout, planning, worker
from continual_bench.application.cloud.contracts import CampaignRequest
from continual_bench.application.cloud.fixtures import publish_fixture_release
from continual_bench.evaluation.artifacts.svg import nice_ticks
from continual_bench.infrastructure.aws import matrix_tracking, native_tracking
from continual_bench.infrastructure.aws.benchmark_tracking import aws_metric_points
from continual_bench.infrastructure.mlflow_figures import matplotlib_available
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json, sha256_path


def test_nearly_constant_axis_has_finitely_many_distinct_ticks():
    high = math.nextafter(1.0, math.inf)
    ticks = nice_ticks(1.0, high)
    assert len(ticks) <= 7
    assert 1.0 in ticks and high in ticks
    assert ticks == sorted(set(ticks))


def test_matrix_and_native_parent_points_preserve_scope_and_coordinates():
    report = {
        "metrics": {"test": {"n": 12, "f1": 0.88}, "train": {"f1": 0.9}},
        "validation": {"log_loss": 0.42},
        "windows": [
            {"end_seq": 31, "scope": "prequential", "accuracy": 0.7},
            {"end_seq": 40, "scope": "holdout", "accuracy": 0.9},
        ],
        "epochs": [
            {"epoch": 1, "metrics": {"f1": 0.6}, "training_seconds": 1.0},
            {"epoch": 2, "metrics": {"f1": 0.8}, "training_seconds": 1.1},
        ],
        "system_metrics": {"samples": [{"timestamp_ms": 800, "values": {"cpu": 36.2}}]},
    }
    points = {
        (v["key"], v["step"]): (v["value"], v["timestamp"])
        for v in aws_metric_points(report, {"f1": 0.88, "split": "test"})
    }
    assert points["summary/test/f1", 0] == (0.88, 0)
    assert points["summary/train/f1", 0] == (0.9, 0)
    assert points["validation/log_loss", 0] == (0.42, 0)
    assert points["prequential/windows/accuracy", 31] == (0.7, 0)
    assert points["holdout/windows/accuracy", 40] == (0.9, 0)
    assert points["holdout/epochs/f1", 1] == (0.6, 0)
    assert points["holdout/epochs/f1", 2] == (0.8, 0)
    assert points["compute/epochs/training_seconds", 2] == (1.1, 0)
    assert points["system/cpu", 0] == (36.2, 800)
    assert points["final/f1", 0] == (0.88, 0) and ("final/split", 0) not in points


def test_native_uploads_fit_checkpoint_and_reuses_matching_parent(tmp_path, monkeypatch):
    calls = []
    parent = {
        "run_id": "parent",
        "status": "RUNNING",
        "artifact_uri": "s3://artifacts/mlflow/parent",
    }

    def request(_method, endpoint, *, payload=None, params=None):
        calls.append((endpoint, payload))
        if endpoint.endswith("/get-by-name"):
            return {"experiment": {"experiment_id": "1"}}
        if endpoint.endswith("/search") and "cb_training_run_key" in payload["filter"]:
            return {"runs": [{"info": parent}]} if calls.count((endpoint, payload)) == 2 else {}
        if endpoint.endswith("/create"):
            if payload["run_name"] == "job":
                return {"run": {"info": parent}}
            return {"run": {"info": {"run_id": "child", "status": "RUNNING"}}}
        return {}

    def tracker():
        value = native_tracking.NativeJobRun("http://fake", "exec", "tuning", "job", dataset="ecommerce")
        monkeypatch.setattr(value, "_request", request)
        return value

    work = tmp_path / "work"
    (work / "artifacts" / "data").mkdir(parents=True)
    (work / "artifacts" / "data" / "index.json").write_text('{"figures":[]}')
    (work / "model.ubj").write_bytes(b"model state")
    fit = {
        "fit_id": "fit-000001",
        "action": "initial",
        "checkpoint": {
            "file": "model.ubj",
            "sha256": sha256_path(work / "model.ubj"),
        },
        "steps": [{"step": 1, "global_step": 1, "kind": "epoch", "metrics": {"validation:log_loss": 0.5}}],
        "summary": {},
    }
    report = {"training_history": [fit]}
    (work / "report.json").write_text(json.dumps(report))
    (work / "training-history.json").write_text(json.dumps({"version": 1, "fits": [fit]}))
    store = LocalObjectStore(tmp_path / "objects")
    current = tracker()
    receipt = current.publish_training_evidence(report, work, store=store)
    current.finish_publication()
    assert receipt["child_runs"] == [{"fit_id": "fit-000001", "run_id": "child", "metric_points": 2}]
    assert store.get_bytes("mlflow/parent/model.ubj") == b"model state"
    assert store.head("mlflow/parent/data/index.json")
    assert store.head("mlflow/parent/training-history.json")
    assert tracker().ensure_run() == "parent"
    parent_creates = [
        data for url, data in calls if url.endswith("/runs/create") and data["run_name"] == "job"
    ]
    assert len(parent_creates) == 1
    assert {"key": "cb-dataset", "value": "ecommerce"} in parent_creates[0]["tags"]
    assert {"key": "cb_training_run_key", "value": "job"} in parent_creates[0]["tags"]
    assert any(
        p["step"] == 1 and p["key"] == "validation/log_loss"
        for _, data in calls
        if data and data.get("run_id") == "child"
        for p in data.get("metrics", [])
    )
    (work / "model.ubj").write_bytes(b"wrong state")
    with pytest.raises(ValueError, match="checksum"):
        current.publish_training_evidence(report, work, store=store)


def test_matrix_retries_only_publication_and_keeps_the_original_training_attempt(tmp_path, monkeypatch):
    store = LocalObjectStore(tmp_path)
    request = CampaignRequest(
        campaign_id="matrix-retry",
        matrix={
            "kind": "experiment_matrix",
            "path": "configs/matrices/binary-comparison.yaml",
        },
    )
    plan = planning.build_plan(request, publish_fixture_release(store))
    unit = plan["units"][0]
    planning.publish_plan(store, {**plan, "units": [unit]})
    report = {"run_key": "retained", "training_history": []}
    attempts = []
    eval_attempts = []

    class Tracker:
        def __init__(self, _unit, attempt, *_args):
            attempts.append(attempt)

        def start(self):
            pass

        def metrics(self, *_args):
            pass

        def publish_training_evidence(self, result, path, **_kwargs):
            assert result == report
            assert (path / "training-history.json").is_file()
            assert (path / "artifacts" / "data" / "index.json").is_file()
            if len(attempts) == 1:
                raise RuntimeError("Temporary tracking error")
            return {
                "mlflow_run_id": "original-parent",
                "mlflow_experiment_id": "1",
                "artifact_uri": "s3://artifacts/mlflow/parent",
                "child_runs": [],
                "artifacts": {"files": 4},
            }

        def finish_publication(self):
            pass

    def evaluate(_store, _unit, workdir, **_kwargs):
        eval_attempts.append(workdir)
        root = workdir / "run"
        nested = root / "binary" / "retained"
        (nested / "artifacts" / "data").mkdir(parents=True)
        (nested / "artifacts" / "data" / "index.json").write_text('{"figures":[]}')
        (nested / "training-history.json").write_text('{"version":1,"fits":[]}')
        report_file = nested / "report.json"
        report_file.write_text(json.dumps(report))
        return {"output": root, "report": report, "headline": {"accuracy": 0.9}, "report_path": report_file}

    monkeypatch.setattr(matrix_tracking, "MatrixMlflowRun", Tracker)
    monkeypatch.setattr(worker, "execute_matrix_case", evaluate)
    env = {"MLFLOW_TRACKING_URI": "http://fake", "AWS_BATCH_JOB_ID": "job", "AWS_BATCH_JOB_ATTEMPT": "1"}
    key = layout.unit_key(request.campaign_id, unit["run_id"])
    assert worker.run_unit(store, key, tmp_path / "first", env=env) == worker.EXIT_FAILED
    final_key = layout.final_key(request.campaign_id, unit["run_id"])
    assert store.head(final_key) is None
    env["AWS_BATCH_JOB_ATTEMPT"] = "2"
    assert worker.run_unit(store, key, tmp_path / "retry", env=env) == worker.EXIT_OK
    final = get_json(store, final_key)
    assert len(eval_attempts) == 1 and attempts[0] == attempts[1] == final["attempt"]["attempt_key"]
    assert final["tracking"]["complete"] and final["links"]["mlflow_run_id"] == "original-parent"
    assert final["report_uri"].endswith("/binary/retained/report.json")
    assert final["training_history_uri"].endswith("/binary/retained/training-history.json")


@pytest.mark.parametrize(
    ("evaluator", "options", "step_kind"),
    [
        (
            "binary",
            {
                "modes": ["streaming"],
                "strategies": [{"policy": {"name": "online", "action": "learn_one"}}],
                "base": {"warmup": 20, "metric_window": 10},
            },
            "update",
        ),
        ("typed", {"typed_policies": ["online"]}, "update"),
        pytest.param("epochs", {"base": {"epochs": 2}}, "epoch", marks=pytest.mark.observation_pngs),
    ],
)
def test_batch_publishes_a_real_matrix_case(evaluator, options, step_kind, tmp_path, monkeypatch):
    snapshot = tmp_path / "configs"
    (snapshot / "matrices").mkdir(parents=True)
    (snapshot / "base").mkdir()
    (snapshot / "base" / "tracking.yaml").write_text("{}")
    (snapshot / "matrices" / "smoke.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "smoke",
                "seeds": [42],
                "evaluator": evaluator,
                "models": [{"name": "logistic"}],
                "on_incompatible": "error",
                "datasets": [{"name": "sea", "samples": 120}],
                **options,
            }
        )
    )
    store = LocalObjectStore(tmp_path / "store")
    request = CampaignRequest(
        campaign_id="matrix-smoke",
        matrix={
            "kind": "experiment_matrix",
            "path": "configs/matrices/smoke.yaml",
        },
    )
    plan = planning.build_plan(request, publish_fixture_release(store), project_root=tmp_path)
    planning.publish_plan(store, plan)
    [unit] = plan["units"]
    created = []

    def rest(_client, _method, endpoint, *, payload=None, params=None):
        if endpoint.endswith("/get-by-name"):
            return {"experiment": {"experiment_id": "1"}}
        if endpoint.endswith("/runs/create"):
            tags = {tag["key"]: tag["value"] for tag in payload["tags"]}
            created.append(tags)
            run = f"run-{len(created)}"
            return {
                "run": {
                    "info": {
                        "run_id": run,
                        "status": "RUNNING",
                        "artifact_uri": f"s3://artifacts/mlflow/{run}",
                    }
                }
            }
        return {}

    monkeypatch.setattr(matrix_tracking.MatrixMlflowRun, "_request", rest)
    env = {"MLFLOW_TRACKING_URI": "http://fake", "AWS_BATCH_JOB_ID": "job", "AWS_BATCH_JOB_ATTEMPT": "1"}
    assert worker.run_unit(store, unit["unit_key"], tmp_path / "work", env=env) == worker.EXIT_OK
    final = get_json(store, layout.final_key(request.campaign_id, unit["run_id"]))
    assert final["tracking"]["complete"]
    assert final["tracking"]["child_runs"]
    assert any(tags.get("cb-step-kind") == step_kind for tags in created)
    assert store.head("mlflow/run-1/training-history.json")
    assert store.head("mlflow/run-1/data/index.json")
    if evaluator == "epochs" and matplotlib_available():
        child = final["tracking"]["child_runs"][0]["run_id"]
        overview = store.get_bytes("mlflow/run-1/tracking-observations/overview/overview.png")
        epochs = store.get_bytes(f"mlflow/{child}/tracking-observations/epochs.png")
        assert overview.startswith(b"\x89PNG\r\n\x1a\n")
        assert epochs.startswith(b"\x89PNG\r\n\x1a\n")


def test_studio_verifies_per_run_matrix_indices_without_a_cross_dataset_plot(monkeypatch):
    source = Path(__file__).parents[1] / "infra" / "sagemaker" / "callback" / "index.py"
    callback = runpy.run_path(str(source))
    monkeypatch.setenv("DATA_BUCKET", "data")
    sfn, sm, s3 = Mock(), Mock(), Mock()
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Executing"}
    summary = {
        "complete": True,
        "snapshot": None,
        "tracking": {"required": True, "complete": True},
        "units": [{"tracking": {"mlflow_run_id": "parent"}}],
    }
    s3.get_object.side_effect = lambda **_: {"Body": BytesIO(json.dumps(summary).encode())}
    message = {
        "token": "abc",
        "pipelineExecutionArn": "arn:studio-execution",
        "arguments": {"stage": "verify", "summary_uri": "s3://data/campaigns/matrix/summary.json"},
    }
    assert callback["tick"](message, sfn, sm, s3) == {"done": True}
    sm.send_pipeline_execution_step_failure.assert_called_once()
    summary["units"][0]["artifacts_index_uri"] = (
        "s3://data/campaigns/matrix/runs/run/attempt/artifacts/index.json"
    )
    assert callback["tick"](message, sfn, sm, s3) == {"done": True}
    sm.send_pipeline_execution_step_success.assert_called_once()


@pytest.mark.parametrize(
    ("kind", "name"), [("training", "TrainingJobName"), ("tuning", "HyperParameterTuningJobName")]
)
def test_direct_native_requests_use_configured_tracking_and_bucket_key(kind, name):
    script = Path(__file__).parents[1] / "scripts" / "aws" / "sagemaker_jobs.py"
    request = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(script),
                kind,
                "--release",
                "example",
                "--dry-run",
                "--mlflow-arn",
                "app",
                "--kms-key-arn",
                "data-key",
            ]
        )
    )
    definition = request["TrainingJobDefinition"] if kind == "tuning" else request
    env = definition["Environment"]
    assert env["MLFLOW_TRACKING_URI"] == "app"
    assert env["BENCH_KMS_KEY_ID"] == "data-key"
    assert env["BENCH_NATIVE_EXECUTION_ID"] == request[name]
    assert env["BENCH_NATIVE_STAGE"] == kind


def test_matrix_mlflow_run_uses_short_sha_experiment_and_strategy_run_name():
    from continual_bench.application.cloud.contracts import RunUnit

    git_sha = "075a88cd957f3ab64ee930b88b3753798d9889a6"
    unit = RunUnit(
        run_id="1424bfc8ed2cbd966ae1",
        campaign_id="energy-airlines-1117-0912",
        kind="matrix_case",
        params={"evaluator": "binary"},
        contract={"release": {"git_sha": git_sha, "compute": "batch-fargate-spot"}},
        contract_sha256="a" * 64,
        job_name="cb-energy-airlines-1117-0912-1424bfc8ed2cbd966ae1",
        unit_key="campaigns/energy-airlines-1117-0912/units/1424bfc8ed2cbd966ae1.json",
        labels={
            "dataset": "electricity",
            "model": "logistic",
            "policy": "drift",
            "detector": "adwin",
        },
    )
    run = matrix_tracking.MatrixMlflowRun(unit, "job-1", "job", "http://fake", log=Mock())
    assert run.experiment_name == "075a88c-continual-bench-matrix"
    assert run.run_name == "energy-airlines-1117-0912-1424bfc8ed2c-drift-adwin-job-1"

    unit_online = unit.model_copy(
        update={
            "labels": {"dataset": "electricity", "model": "logistic", "policy": "online", "detector": "none"}
        }
    )
    run_online = matrix_tracking.MatrixMlflowRun(unit_online, "job-1", "job", "http://fake", log=Mock())
    assert run_online.run_name == "energy-airlines-1117-0912-1424bfc8ed2c-online-job-1"
    assert run.timeout_seconds == 15


def test_pooled_http_opener_wraps_timeout_and_tracker_retries_transient_url_error(monkeypatch):
    import urllib.error
    import urllib.request

    from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker
    from continual_bench.infrastructure.tracking.http import PooledHttpOpener

    opener = PooledHttpOpener()
    monkeypatch.setattr(opener._pool, "request", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("read")))
    req = urllib.request.Request("https://example.com/api")
    with pytest.raises(urllib.error.URLError):
        opener(req, timeout=1)

    attempts = {"count": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"experiment": {"experiment_id": "1"}}'

    def flaky_opener(http_req, timeout=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError("RemoteDisconnected")
        return _Resp()

    monkeypatch.setattr("continual_bench.infrastructure.aws.mlflow_tracking.time.sleep", lambda *_: None)
    tracker = SageMakerMlflowTracker(
        tracking_uri="http://fake",
        dataset="electricity",
        policy="drift",
        model_name="logistic",
        run_id="run-1",
        http_opener=flaky_opener,
    )
    tracker._rest_endpoint = "http://fake"
    res = tracker._request("GET", "/api/2.0/mlflow/experiments/get-by-name")
    assert attempts["count"] == 2
    assert res["experiment"]["experiment_id"] == "1"
