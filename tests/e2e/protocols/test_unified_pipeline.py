"""Unified tests for CARA/UPF protocols, MLflow hierarchy, artifact rules, environments, and DAGs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from continual_bench.application.cloud.dependencies import resolve_runtime
from continual_bench.backends import CHECKPOINT_EXTENSIONS, RuntimeSettings, resolve_backend
from continual_bench.datasets.streams_unified import _STREAM_SOURCES, StreamDatasetSpec
from continual_bench.models.adapters import ModelConfig
from continual_bench.protocols.base import CampaignConfig
from continual_bench.protocols.campaign import load_campaign_config, run_campaign
from continual_bench.tracking.mlflow_publisher import build_experiment_name, publish_campaign_to_mlflow
from continual_bench.tracking.tracker import BenchmarkTracker


def _synth_binary_stream(spec: StreamDatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    n_rows = 1200
    X = rng.normal(size=(n_rows, 4))
    shift = np.linspace(-1.0, 1.0, n_rows)
    y = ((X[:, 0] + 0.7 * shift * X[:, 1]) > 0.0).astype(np.int64)
    return X, y


class _FakeMlflowClient:
    """In-memory MlflowClient double validating experiment isolation, nested runs, and artifacts."""

    def __init__(self) -> None:
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.artifacts: dict[str, list[str]] = {}
        self._exp_seq = 0
        self._run_seq = 0

    def get_experiment_by_name(self, name: str) -> Any | None:
        exp_id = self.experiments.get(name)
        if exp_id is None:
            return None
        return SimpleNamespace(experiment_id=exp_id, name=name)

    def create_experiment(self, name: str) -> str:
        self._exp_seq += 1
        exp_id = f"exp-{self._exp_seq}"
        self.experiments[name] = exp_id
        return exp_id

    def create_run(self, experiment_id: str, tags: dict[str, str] | None = None) -> Any:
        self._run_seq += 1
        run_id = f"run-{self._run_seq}"
        self.runs[run_id] = {
            "experiment_id": experiment_id,
            "tags": dict(tags or {}),
            "params": {},
            "metrics": [],
            "status": "RUNNING",
        }
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, status="RUNNING", artifact_uri=f"s3://mlflow/{run_id}"),
            data=SimpleNamespace(tags=dict(tags or {}), metrics={}),
        )

    def log_batch(
        self,
        run_id: str,
        metrics: list[Any] = (),
        params: list[Any] = (),
        tags: list[Any] = (),
        synchronous: bool = True,
    ) -> None:
        del synchronous
        rec = self.runs[run_id]
        for m in metrics:
            rec["metrics"].append((m.key, m.value, m.step))
        for p in params:
            rec["params"][p.key] = p.value
        for t in tags:
            rec["tags"][t.key] = t.value

    def log_artifact(self, run_id: str, local_path: str) -> None:
        self.artifacts.setdefault(run_id, []).append(Path(local_path).name)

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self.runs[run_id]["status"] = status


def test_environment_configs_switch_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "BENCH_ENV",
        "BENCH_ENV_CONFIG",
        "BENCH_PLATFORM",
        "BENCH_OUTPUT_DIR",
        "BENCH_TRACKING_URI",
        "MLFLOW_TRACKING_URI",
        "S3_ENDPOINT_URL",
        "BENCH_BUCKET",
        "BENCH_SAVE_MODEL_CHECKPOINTS",
    ):
        monkeypatch.delenv(var, raising=False)

    local_s = RuntimeSettings.from_env(env_name="local")
    assert local_s.env_name == "local"
    assert local_s.platform == "local"
    assert local_s.save_model_checkpoints is False
    assert local_s.generate_plots is False
    assert local_s.flush_artifacts_at_end_only is True

    docker_s = RuntimeSettings.from_env(env_name="docker")
    assert docker_s.env_name == "docker"
    assert docker_s.platform == "homelab"
    assert docker_s.s3_endpoint_url == "http://minio:9000"
    assert docker_s.tracking_uri == "http://mlflow:5000"
    assert str(docker_s.output_dir) == "s3://benchmarks"

    monkeypatch.setenv("BENCH_BUCKET", "my-aws-bucket")
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/cb",
    )
    aws_s = RuntimeSettings.from_env(env_name="aws")
    assert aws_s.env_name == "aws"
    assert aws_s.platform == "aws"
    assert aws_s.bucket == "my-aws-bucket"
    assert aws_s.tracking_uri.startswith("arn:aws:sagemaker:")

    local_rt = resolve_runtime("local", settings=local_s)
    assert local_rt.mode == "cli"


def test_mlflow_hierarchy_experiments_policies_and_retrains(tmp_path: Path) -> None:
    fake_client = _FakeMlflowClient()
    settings = RuntimeSettings(
        env_name="local",
        platform="local",
        output_dir=tmp_path,
        tracking_uri="http://mlflow:5000",
        save_model_checkpoints=False,
        generate_plots=False,
    )

    assert build_experiment_name("cara-electricity", "electricity") == "cara-electricity/electricity"

    artifact_file = tmp_path / "benchmark_summary.json"
    artifact_file.write_text("[]", encoding="utf-8")
    pol_artifact = tmp_path / "policy_summary.json"
    pol_artifact.write_text("[]", encoding="utf-8")
    seed_artifact = tmp_path / "seed_timeline.csv"
    seed_artifact.write_text("batch_index,accuracy\n25,0.75\n", encoding="utf-8")
    retrain_artifact = tmp_path / "cara-s0-v1-b28.json"
    retrain_artifact.write_text('{"version_name": "cara-s0-v1-b28"}', encoding="utf-8")
    stray_plot = tmp_path / "plot.svg"
    stray_plot.write_text("<svg/>", encoding="utf-8")
    stray_ckpt = tmp_path / "model.pkl"
    stray_ckpt.write_bytes(b"ckpt")

    receipt = publish_campaign_to_mlflow(
        settings=settings,
        run_id="cara-electricity_20261126T000000Z",
        metadata={
            "campaign_name": "cara-electricity",
            "protocol": "cara",
            "dataset": "electricity",
            "model": "random_forest",
            "model_params": {"n_estimators": 100},
            "seeds": [0, 1],
            "policies": ["cara", "frozen"],
            "staleness_mode": "relative",
            "kernel_gamma": 20.0,
        },
        aggregate_rows=[
            {
                "protocol": "cara",
                "dataset": "electricity",
                "policy": "cara",
                "regime": "price_grid",
                "metric": "average_accuracy",
                "mean": 0.72,
                "std": 0.01,
                "n_seeds": 2,
            },
            {
                "protocol": "cara",
                "dataset": "electricity",
                "policy": "frozen",
                "regime": "price_grid",
                "metric": "average_accuracy",
                "mean": 0.64,
                "std": 0.02,
                "n_seeds": 2,
            },
        ],
        seed_summaries=[
            {"seed": 0, "policy": "cara", "regime": "price_grid", "average_accuracy": 0.72},
            {"seed": 0, "policy": "frozen", "regime": "price_grid", "average_accuracy": 0.64},
        ],
        regime_rows=[
            {"seed": 0, "policy": "cara", "regime": "price_grid", "strategy_cost": 1.25},
        ],
        batch_rows=[
            {
                "seed": 0,
                "policy": "cara",
                "regime": "price_grid",
                "batch_index": 25,
                "accuracy": 0.75,
                "error": 0.25,
                "staleness": 0.02,
                "model_age": 0,
                "deployed_model": 25,
                "cumulative_cost": 0.25,
                "retrain_decision": False,
                "fit_time_s": 0.0,
            }
        ],
        retrain_events=[
            {
                "seed": 0,
                "policy": "cara",
                "version_index": 1,
                "version_name": "cara-s0-v1-b28",
                "batch_index": 28,
                "step": 28,
                "fit_seconds": 0.12,
                "train_seconds": 0.12,
                "train_samples": 100,
                "window_size": 100,
                "retrain_cost": 1.5,
                "post_retrain_error": 0.25,
                "post_retrain_accuracy": 0.75,
            }
        ],
        paper_comparison_rows=[
            {
                "policy": "cara",
                "metric": "average_accuracy",
                "regime": "price_grid",
                "observed_mean": 0.72,
                "observed_std": 0.01,
                "published_value": 0.68,
                "published_spread": None,
                "abs_diff": 0.04,
                "rel_diff_pct": 5.88,
            }
        ],
        system_metrics=[{"timestamp_ms": 1000, "values": {"cpu_utilization_percentage": 25.0}}],
        artifact_paths={
            "benchmark_summary.json": str(artifact_file),
            "policies/cara/policy_summary.json": str(pol_artifact),
            "policies/cara/seeds/seed-0/seed_timeline.csv": str(seed_artifact),
            "policies/cara/seeds/seed-0/retrains/cara-s0-v1-b28.json": str(retrain_artifact),
            "plot.svg": str(stray_plot),
            "model.pkl": str(stray_ckpt),
        },
        client=fake_client,
    )

    assert receipt is not None
    assert receipt["experiment_name"] == "cara-electricity/electricity"
    campaign_run_id = receipt["campaign_run_id"]
    cara_run_id = receipt["policy_runs"]["cara"]
    frozen_run_id = receipt["policy_runs"]["frozen"]
    assert fake_client.runs[cara_run_id]["tags"]["mlflow.parentRunId"] == campaign_run_id
    assert fake_client.runs[frozen_run_id]["tags"]["mlflow.parentRunId"] == campaign_run_id

    # Seed run nested under Policy run with timeline & cost sensitivity metrics + seed artifacts
    cara_seed0_id = receipt["seed_runs"]["cara"][0]
    seed_run = fake_client.runs[cara_seed0_id]
    assert seed_run["tags"]["mlflow.parentRunId"] == cara_run_id
    assert seed_run["tags"]["mlflow.runName"] == "seed-0"
    assert any(k == "prequential/batches/accuracy" and s == 25 for k, _, s in seed_run["metrics"])
    assert any(k == "cost_sensitivity/total_cost" for k, _, _ in seed_run["metrics"])
    assert fake_client.artifacts[cara_seed0_id] == ["seed_timeline.csv"]
    assert fake_client.artifacts[cara_run_id] == ["policy_summary.json"]

    # Retrain version run nested under Seed run with version name + artifact
    retrain_ids = receipt["retrain_runs"]["cara"]
    assert len(retrain_ids) == 1
    retrain_run = fake_client.runs[retrain_ids[0]]
    assert retrain_run["tags"]["mlflow.parentRunId"] == cara_seed0_id
    assert retrain_run["tags"]["mlflow.runName"] == "cara-s0-v1-b28"
    assert retrain_run["tags"]["cb-model-version"] == "v1"
    assert retrain_run["params"]["batch_index"] == "28"
    assert retrain_run["params"]["version_name"] == "cara-s0-v1-b28"
    assert any(k == "training/fit_seconds" for k, _, _ in retrain_run["metrics"])
    assert any(k == "training/post_retrain_error" for k, _, _ in retrain_run["metrics"])
    assert fake_client.artifacts[retrain_ids[0]] == ["cara-s0-v1-b28.json"]

    # Plot and checkpoint excluded from MLflow artifacts
    assert fake_client.artifacts[campaign_run_id] == ["benchmark_summary.json"]
    assert any(
        k == "system/cpu_utilization_percentage"
        for k, _, _ in fake_client.runs[campaign_run_id]["metrics"]
    )


def test_end_only_artifact_flush_and_no_plots_or_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(_STREAM_SOURCES, "electricity", _synth_binary_stream)
    out_dir = tmp_path / "run_out"

    # Verify BenchmarkTracker buffers until flush()
    settings = RuntimeSettings(platform="local", output_dir=out_dir, save_model_checkpoints=False)
    backend = resolve_backend("local", settings)
    tracker = BenchmarkTracker(backend=backend, run_id="r1", settings=settings)
    tracker.log_metadata({"dataset": "electricity"})
    tracker.log_seed_summary(0, "cara", {"average_accuracy": 0.7})
    assert not (out_dir / "r1").exists()
    tracker.flush()
    assert (out_dir / "r1" / "metadata.json").exists()

    # Run full CARA campaign and verify paper comparison + zero plots + zero checkpoints
    cfg = CampaignConfig(
        name="cara-electricity-test",
        protocol="cara",
        dataset=StreamDatasetSpec(
            name="electricity",
            protocol="cara",
            num_batches=8,
            offline_batches=3,
            online_batches=5,
            split_ratio=0.9,
            query_ratio=0.1,
            seed=1,
        ),
        model=ModelConfig(name="random_forest", params={"n_estimators": 5, "n_jobs": 1}, seed=0),
        seeds=[0],
        policies=["cara", "frozen", "adwin"],
        output_dir=str(out_dir / "campaign"),
    )
    res = run_campaign(cfg, platform="local", save_model_checkpoints=False)
    assert "paper_comparison.json" in res.artifacts
    assert "paper_comparison.csv" in res.artifacts
    assert not any(k.endswith((".svg", ".png")) for k in res.artifacts)
    assert not list((out_dir / "campaign").rglob("*.svg"))
    assert not list((out_dir / "campaign").rglob("*.png"))
    for p in (out_dir / "campaign").rglob("*"):
        if p.is_file():
            assert p.suffix.lower() not in CHECKPOINT_EXTENSIONS


@pytest.mark.parametrize(
    "yaml_path",
    [
        "configs/campaigns/cara-electricity.yaml",
        "configs/campaigns/cara-airlines.yaml",
        "configs/campaigns/cara-insects-abrupt.yaml",
        "configs/campaigns/cara-insects-reoccurring.yaml",
        "configs/campaigns/upf-electricity.yaml",
        "configs/campaigns/upf-airlines.yaml",
        "configs/campaigns/upf-insects-abrupt.yaml",
        "configs/campaigns/upf-insects-reoccurring.yaml",
    ],
)
def test_all_paper_campaign_yamls_validate(yaml_path: str) -> None:
    cfg = load_campaign_config(yaml_path)
    assert cfg.protocol in {"cara", "upf"}
    assert len(cfg.policies) >= 1
    assert len(cfg.seeds) >= 1


def test_airflow_readiness_has_no_zenml_and_campaign_payload() -> None:
    readiness_spec = importlib.util.spec_from_file_location(
        "bench_readiness", Path(__file__).parents[3] / "dags/bench_readiness.py"
    )
    readiness = importlib.util.module_from_spec(readiness_spec)
    assert readiness_spec and readiness_spec.loader
    readiness_spec.loader.exec_module(readiness)
    assert "zenml" not in readiness.HTTP_DEPENDENCIES
    assert set(readiness.HTTP_DEPENDENCIES) == {"mlflow", "minio", "worker"}

    # Check server.py has no --zenml flag
    server_text = Path("src/continual_bench/orchestration/server.py").read_text(encoding="utf-8")
    assert "--zenml" not in server_text
    assert "/campaigns" in server_text


@pytest.mark.parametrize("distributed", [False, True], ids=["local", "airflow-worker"])
@pytest.mark.parametrize("protocol_name", ["cara", "upf"])
def test_campaign_tracking_is_incremental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol_name: str, distributed: bool
) -> None:
    import json

    from mlflow.tracking import MlflowClient

    from continual_bench.orchestration import server
    from continual_bench.protocols.cara import CaraProtocol
    from continual_bench.protocols.upf import UpfProtocol

    monkeypatch.setitem(_STREAM_SOURCES, "electricity", _synth_binary_stream)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        "incremental/electricity", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    cfg = CampaignConfig(
        name="incremental",
        protocol=protocol_name,
        dataset=StreamDatasetSpec(
            name="electricity",
            protocol=protocol_name,
            num_batches=8,
            offline_batches=3,
            online_batches=5,
            split_ratio=0.8,
            query_ratio=0.1,
        ),
        model=ModelConfig(name="sklearn"),
        policies=["cara"] if protocol_name == "cara" else ["upf"],
        seeds=[1, 2],
        kappa_grid=[1.0, 2.0],
        alpha_grid=[0.0, 0.5],
        output_dir=str(tmp_path / "output"),
        tracking_uri=uri,
    )

    def runs(level: str):
        return client.search_runs([experiment_id], f"tags.`cb-hierarchy-level` = '{level}'")

    proto = CaraProtocol if protocol_name == "cara" else UpfProtocol
    build = proto.build_seed_matrix

    def assert_matrix_visible(self, config, seed, **kwargs):
        roots = runs("campaign")
        assert len(roots) == 1 and roots[0].info.status == "RUNNING"
        matrices = {int(r.data.tags["cb-seed"]): r for r in runs("matrix")}
        current = matrices[seed]
        assert current.info.status == "RUNNING"
        assert current.data.tags["mlflow.parentRunId"] == roots[0].info.run_id
        if seed == 2:
            previous = matrices[1]
            assert previous.info.status == "FINISHED"
            assert previous.data.metrics["matrix/duration_seconds"] > 0
            assert "matrix_summary.json" in {a.path for a in client.list_artifacts(previous.info.run_id)}
        return build(self, config, seed, **kwargs)

    monkeypatch.setattr(proto, "build_seed_matrix", assert_matrix_visible)
    finish = BenchmarkTracker.flush
    interrupted = False

    def assert_seed_results_visible(self):
        nonlocal interrupted
        seeds = runs("seed")
        assert len(seeds) == 2
        assert {r.info.status for r in seeds} == {"FINISHED"}
        for run in seeds:
            assert "summary/average_accuracy" in run.data.metrics
            assert {"seed_summary.json", "seed_timeline.csv", "seed_regimes.csv"} <= {
                a.path for a in client.list_artifacts(run.info.run_id)
            }
        if distributed and not interrupted:
            assert runs("campaign")[0].info.status == "RUNNING"
            interrupted = True
            raise RuntimeError("interrupted consolidation")
        return finish(self)

    monkeypatch.setattr(BenchmarkTracker, "flush", assert_seed_results_visible)

    if distributed:
        root = tmp_path / "worker"
        monkeypatch.setattr(server, "ROOT", root)
        payload = {"campaign_key": "test", "config": cfg.model_dump(mode="json"), "env": "local"}
        seed_paths = []
        for seed in cfg.seeds:
            job = {**payload, "seed": seed}
            server.perform_campaign_seed(job)
            path = root / "campaign_seeds" / server.campaign_identity(job) / "job.json"
            status = json.loads(path.read_text())
            assert status["status"] == "succeeded", status
            seed_paths.append(status["seed_path"])

        job = {**payload, "seed_paths": seed_paths}
        server.perform_campaign_consolidate(job)
        path = root / "campaign_consolidate" / server.campaign_identity(job) / "job.json"
        assert json.loads(path.read_text())["error"] == "interrupted consolidation"
        server.perform_campaign_consolidate(job)
        status = json.loads(path.read_text())
        assert status["status"] == "succeeded", status
        receipt = status["mlflow_receipt"]
    else:
        receipt = run_campaign(cfg, env_name="local").mlflow_receipt

    assert receipt is not None
    assert len(runs("campaign")) == 1
    assert len(runs("matrix")) == 2
    assert len(runs("seed")) == 2
    assert client.get_run(receipt["campaign_run_id"]).info.status == "FINISHED"


def test_matrix_failure_closes_active_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from continual_bench.protocols.cara import CaraProtocol
    from continual_bench.tracking import mlflow_publisher

    client = _FakeMlflowClient()
    monkeypatch.setattr(
        mlflow_publisher, "_resolve_tracking_adapter",
        lambda uri: (client, mlflow_publisher.MlflowClientBackend(client)),
    )
    monkeypatch.setitem(_STREAM_SOURCES, "electricity", _synth_binary_stream)
    cfg = CampaignConfig(
        name="failure",
        protocol="cara",
        dataset={"name": "electricity", "num_batches": 8, "offline_batches": 3, "online_batches": 5},
        model={"name": "logistic"},
        policies=["cara"],
        seeds=[1],
        output_dir=str(tmp_path),
        tracking_uri="http://mlflow",
    )

    def fail(*args, **kwargs):
        raise RuntimeError("matrix error")

    monkeypatch.setattr(CaraProtocol, "build_seed_matrix", fail)
    with pytest.raises(RuntimeError, match="matrix error"):
        run_campaign(cfg, env_name="local")
    assert {
        r["tags"]["cb-hierarchy-level"]: r["status"] for r in client.runs.values()
    } == {"campaign": "FAILED", "matrix": "FAILED"}
