"""Metrics-only is a retention choice, not a change to scientific predictions."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from continual_bench.config import TrackingConfig
from continual_bench.datasets import zia
from continual_bench.datasets.streams import ZiaStreamProvider
from continual_bench.evaluation import stream_benchmark as stream
from continual_bench.evaluation.zia_artifacts import tracking_report, write_zia_artifacts
from continual_bench.evaluation.zia_predictions import prediction_manifest
from continual_bench.evaluation.zia_publication import replay_predictions, run_documents
from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files


@pytest.fixture
def metric_stream(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    monkeypatch.setattr(zia, "SMOKE_ROWS", 800)
    monkeypatch.setattr(ZiaStreamProvider, "BLOCK_ROWS", {"smoke": 100, "full": 100})
    ids = np.arange(800)
    for dataset in ("social", "fraud"):
        classes = len(zia.DATASETS[dataset]["classes"])
        base = ids % classes
        y = ((base + ids // 250) % classes).astype(np.uint8)
        path = tmp_path / dataset / "smoke.parquet"
        path.parent.mkdir()
        pq.write_table(
            pa.table({"event_time": ids // 3, "label": y, "feature": base.astype(np.float32)}),
            path,
            row_group_size=71,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "scope": "smoke",
                    "source": zia.DATASETS[dataset],
                    "format": zia.FORMAT,
                    "features": ["feature"],
                    "rows": len(y),
                    "bytes": path.stat().st_size,
                    "sha256": zia.sha256_file(path),
                }
            )
        )
    tracking = TrackingConfig.model_validate(
        yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    ).model_dump()
    tracking["retraining"] = {"n_alphas": 2, "mc_samples": 3}
    return tmp_path, tracking


def evaluate(fixture, dataset, policy, seed, mode, name, **kwargs):
    root, tracking = fixture
    settings = copy.deepcopy(tracking)
    settings["retraining"]["seed"] = seed
    return stream.run(
        root,
        dataset,
        "smoke",
        policy,
        root / name,
        settings,
        iterative_config={"seed": seed, "max_epochs": 2},
        artifact_mode=mode,
        **kwargs,
    )


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("dataset", ["social", "fraud"])
@pytest.mark.parametrize("policy", stream.POLICIES)
def test_metrics_only_preserves_full_predictions_and_retention(metric_stream, dataset, policy, seed):
    root, _ = metric_stream
    full = evaluate(metric_stream, dataset, policy, seed, "full", "full")
    actual = evaluate(metric_stream, dataset, policy, seed, "metrics_only", "only")
    keys = (
        "n",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "macro_roc_auc",
        "macro_pr_auc",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier",
        "update_count",
        "retrain_count",
    )
    for scope in ("metrics", "holdout"):
        assert {k: actual[scope].get(k) for k in keys} == {k: full[scope].get(k) for k in keys}
    assert actual["prequential_confusion"] == full["prequential_confusion"]
    assert actual["holdout_confusion"] == full["holdout_confusion"]
    assert (
        actual["tracking_observations"]["retention"]["matrix"]
        == full["tracking_observations"]["retention"]["matrix"]
    )
    assert [v["sha256"] for v in actual["model_versions"]] == [v["sha256"] for v in full["model_versions"]]
    assert actual["metrics"]["model_storage_bytes"] == 0
    assert all(v["file"] is None for v in actual["model_versions"])
    assert all(f["seed"] == seed for f in actual["training_history"])
    assert not (root / "only" / "models").exists()
    import pyarrow.parquet as pq
    from pyarrow import concat_tables

    def predictions(name):
        return concat_tables(
            [pq.read_table(path) for path in sorted((root / name / "predictions").glob("*.parquet"))]
        )

    assert predictions("full").equals(predictions("only"))
    verify_metrics_only_files(root / "only")
    if "retraining" in actual:
        assert actual["retraining"]["deployed"] == full["retraining"]["deployed"]
        assert actual["metrics"]["materialization_fit_count"] == len(actual["model_versions"])
        assert actual["metrics"]["actual_total_fit_count"] == len(actual["training_history"])


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("dataset", ["social", "fraud"])
def test_metric_matrix_cache_reuses_measurements_without_reusing_weights(metric_stream, dataset, seed):
    root, _ = metric_stream
    cache = root / "matrix-metrics.json.gz"
    original = evaluate(
        metric_stream, dataset, "cara", seed, "metrics_only", "builder", matrix_metrics_cache=cache
    )
    control = evaluate(metric_stream, dataset, "upf", seed, "metrics_only", "control")
    reused = evaluate(
        metric_stream, dataset, "upf", seed, "metrics_only", "reused", matrix_metrics_cache=cache
    )
    assert not original["contract"]["matrix_metrics_cache"]["reused"]
    assert reused["contract"]["matrix_metrics_cache"]["reused"]
    assert control["retraining"]["deployed"] == reused["retraining"]["deployed"]
    assert control["prequential_confusion"] == reused["prequential_confusion"]
    assert [v["sha256"] for v in control["model_versions"]] == [v["sha256"] for v in reused["model_versions"]]
    assert reused["metrics"]["actual_total_fit_count"] == len(reused["model_versions"])
    assert reused["metrics"]["reused_matrix_candidate_fit_count"] > 0
    assert reused["tracking_observations"]["effort"]["fit_count"] == len(reused["model_versions"])
    with gzip.open(cache, "rt") as file:
        document = json.load(file)
    assert document["schema"] == "zia-matrix-metrics-v1"
    assert not ({"points", "queries", "payload", "sub_loss"} & document["matrix"].keys())
    assert all(s["file"] is None and "payload" not in s for s in document["states"])
    with pytest.raises(ValueError, match="execution contract"):
        evaluate(
            metric_stream, dataset, "cara", seed + 1, "metrics_only", "wrong", matrix_metrics_cache=cache
        )


def test_metrics_only_documents_and_plot_data_have_no_native_model(metric_stream):
    root, _ = metric_stream
    report = evaluate(metric_stream, "social", "frozen", 42, "metrics_only", "only")
    evidence = prediction_manifest(root / "only", report)
    adapted = tracking_report(report, evidence)
    metadata, links, documents = run_documents(report, adapted, "s3://evidence/run")
    assert metadata is None and links["model_states_uri"] is None
    assert "model-metadata.json" not in documents
    assert documents["model-versions.json"]["storage_bytes"] == 0
    result = write_zia_artifacts(root / "only", report, adapted)
    assert result["figures"] == 0 and result["data_files"] > 0
    assert (root / "only/artifacts/data/rendering/status.json").is_file()
    verify_metrics_only_files(root / "only")
    with pytest.raises(ValueError, match="retained"):
        replay_predictions(root / "only", report, root)


def test_metrics_only_refuses_training_checkpoint_before_loading_input(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        stream.run(
            tmp_path,
            "social",
            "smoke",
            "frozen",
            tmp_path / "out",
            {},
            artifact_mode="metrics_only",
            checkpoint=object(),
        )


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_metrics_only_rejects_mismatched_policy_seed(metric_stream, seed):
    root, tracking = metric_stream
    settings = copy.deepcopy(tracking)
    settings["retraining"]["seed"] = seed + 1
    with pytest.raises(ValueError, match="geometry/forecast seed"):
        stream.run(
            root,
            "social",
            "smoke",
            "cara",
            root / "bad-seed",
            settings,
            iterative_config={"seed": seed, "max_epochs": 2},
            artifact_mode="metrics_only",
        )


def test_metrics_only_rejects_preexisting_weights_before_evaluation(metric_stream):
    root, _ = metric_stream
    output = root / "only"
    (output / "models").mkdir(parents=True)
    (output / "models" / "old.json").write_text('{"trees": []}')
    with pytest.raises(ValueError, match="model or checkpoint"):
        evaluate(metric_stream, "social", "frozen", 42, "metrics_only", "only")
