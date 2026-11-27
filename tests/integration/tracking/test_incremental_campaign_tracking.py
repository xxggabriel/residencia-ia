"""Exercise campaign run lifecycles against a local MLflow tracking store."""

from pathlib import Path

import numpy as np
import pytest

from continual_bench.datasets.streams_unified import _STREAM_SOURCES, StreamDatasetSpec
from continual_bench.models.adapters import ModelConfig
from continual_bench.protocols.base import CampaignConfig
from continual_bench.protocols.campaign import run_campaign
from continual_bench.tracking.tracker import BenchmarkTracker


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient

    def stream(spec):
        rng = np.random.default_rng(spec.seed)
        x = rng.normal(size=(1200, 4))
        return x, (x[:, 0] > 0).astype(np.int64)

    monkeypatch.setitem(_STREAM_SOURCES, "electricity", stream)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlflow").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "incremental/electricity", artifact_location=(tmp_path / "artifacts").as_uri()
    )

    def runs(level):
        return client.search_runs([experiment], f"tags.`cb-hierarchy-level` = '{level}'")

    return uri, client, runs


def _config(tmp_path: Path, uri: str, protocol: str) -> CampaignConfig:
    return CampaignConfig(
        name="incremental",
        protocol=protocol,
        dataset=StreamDatasetSpec(
            name="electricity",
            protocol=protocol,
            num_batches=8,
            offline_batches=3,
            online_batches=5,
            split_ratio=0.8,
            query_ratio=0.1,
        ),
        model=ModelConfig(name="sklearn"),
        policies=[protocol],
        seeds=[1, 2],
        kappa_grid=[1.0, 2.0],
        alpha_grid=[0.0, 0.5],
        output_dir=str(tmp_path / "output"),
        tracking_uri=uri,
    )


@pytest.mark.parametrize("protocol", ["cara", "upf"])
def test_seed_runs_publish_before_campaign_finishes(tmp_path, tracking, monkeypatch, protocol):
    uri, client, runs = tracking
    run_matrix = BenchmarkTracker.run_matrix

    def observe_matrix(self, seed, build, fit_times):
        def checked_build():
            roots = runs("campaign")
            assert len(roots) == 1 and roots[0].info.status == "RUNNING"
            matrices = {int(r.data.tags["cb-seed"]): r for r in runs("matrix")}
            assert matrices[seed].info.status == "RUNNING"
            assert matrices[seed].data.tags["mlflow.parentRunId"] == roots[0].info.run_id
            if seed == 2:
                previous = matrices[1]
                assert previous.info.status == "FINISHED"
                assert previous.data.metrics["matrix/duration_seconds"] > 0
                assert len(client.get_metric_history(previous.info.run_id, "training/fit_seconds")) == 8
                assert "matrix_summary.json" in {
                    artifact.path for artifact in client.list_artifacts(previous.info.run_id)
                }
            return build()

        return run_matrix(self, seed, checked_build, fit_times)

    monkeypatch.setattr(BenchmarkTracker, "run_matrix", observe_matrix)
    flush = BenchmarkTracker.flush

    def observe_seed_results(self):
        seeds = runs("seed")
        assert len(seeds) == 2
        assert {r.info.status for r in seeds} == {"FINISHED"}
        for seed in seeds:
            assert "summary/average_accuracy" in seed.data.metrics
            assert {"seed_summary.json", "seed_regimes.csv"} <= {
                artifact.path for artifact in client.list_artifacts(seed.info.run_id)
            }
        assert runs("campaign")[0].info.status == "RUNNING"
        return flush(self)

    monkeypatch.setattr(BenchmarkTracker, "flush", observe_seed_results)
    result = run_campaign(_config(tmp_path, uri, protocol), platform="local")
    assert len(runs("seed")) == 2
    assert len(runs("matrix")) == 2
    assert result.mlflow_receipt is not None
    assert client.get_run(result.mlflow_receipt["campaign_run_id"]).info.status == "FINISHED"


def test_failed_matrix_and_campaign_are_marked_failed(tmp_path, tracking, monkeypatch):
    uri, _, runs = tracking
    run_matrix = BenchmarkTracker.run_matrix

    def fail_during_build(self, seed, build, fit_times):
        def fail():
            assert runs("matrix")[0].info.status == "RUNNING"
            raise RuntimeError("matrix interrupted")

        return run_matrix(self, seed, fail, fit_times)

    monkeypatch.setattr(BenchmarkTracker, "run_matrix", fail_during_build)
    with pytest.raises(RuntimeError, match="matrix interrupted"):
        run_campaign(_config(tmp_path, uri, "cara"), platform="local")
    assert [r.info.status for r in runs("campaign")] == ["FAILED"]
    assert [r.info.status for r in runs("matrix")] == ["FAILED"]
