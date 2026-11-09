from functools import partial
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import yaml

from continual_bench.core.metrics import Metrics
from continual_bench.evaluation.zia_metrics import ClassificationMetrics
from continual_bench.models.zia import FraudOrReturnBooster, MultinomialLogistic


def test_binned_metrics_preserve_binary_confusion_and_calibration():
    labels = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
    scores = np.array([0.25, 0.5, 0.5, 0.75, 1.0])
    exact = Metrics()
    for y, p in zip(labels, scores, strict=True):
        exact.add(bool(y), float(p))
    actual = ClassificationMetrics(2)
    for start, stop in ((0, 2), (2, 5)):
        partial = ClassificationMetrics(2)
        p = scores[start:stop]
        partial.add(labels[start:stop], np.column_stack((1 - p, p)))
        actual.merge(partial)
    result, _ = actual.metrics()
    assert result == pytest.approx({key: exact.values().get(key, result[key]) for key in result})


@pytest.mark.parametrize("learner", ["boost", "logistic"])
def test_models_roundtrip_and_continue(learner):
    if learner == "boost":
        pytest.importorskip("xgboost")
        make = FraudOrReturnBooster
    else:
        make = partial(MultinomialLogistic, 2)
    x = np.arange(100, dtype=np.float32).reshape(-1, 1) / 100
    y = (x[:, 0] >= 0.5).astype(np.uint8)
    first = make()
    first.fit(x, y, False)
    prediction = first.predict(x)
    assert np.mean(prediction.argmax(axis=1) == y) > 0.9
    restored = make()
    restored.load(first.payload()[0])
    np.testing.assert_allclose(restored.predict(x), prediction)
    restored.fit(x, y, True)
    np.testing.assert_allclose(restored.predict(x).sum(axis=1), 1)


def test_smoke_coverage_temporal_ties_and_policies(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from continual_bench.config import TrackingConfig
    from continual_bench.datasets.zia import DATASETS, FORMAT, sha256_file
    from continual_bench.evaluation.environment import encoded
    from continual_bench.evaluation.zia import POLICIES, run
    from continual_bench.evaluation.zia_artifacts import tracking_report, write_zia_artifacts
    from continual_bench.evaluation.zia_predictions import prediction_manifest
    from continual_bench.evaluation.zia_publication import replay_predictions

    folder = tmp_path / "social"
    folder.mkdir()
    path = folder / "smoke.parquet"
    timestamps = np.arange(10_000) // 13
    y = (np.arange(10_000) % 4).astype(np.uint8)
    pq.write_table(pa.table({"event_time": timestamps, "label": y, "feature": y.astype(np.float32)}), path)
    (folder / "smoke.json").write_bytes(encoded({
        "dataset": "social", "scope": "smoke", "source": DATASETS["social"], "format": FORMAT,
        "features": ["feature"], "rows": 10_000, "bytes": path.stat().st_size, "sha256": sha256_file(path),
    }))
    tracking = TrackingConfig.model_validate(
        yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    ).model_dump()
    reports = {}
    for policy in POLICIES:
        output = tmp_path / policy
        report = run(tmp_path, "social", "smoke", policy, output, tracking)
        reports[policy] = report
        summary, holdout = report["metrics"], report["holdout"]
        assert summary["initial_train_rows"] + summary["n"] + holdout["n"] == 10_000
        windows = report["windows"]
        assert all(a["end_time_ms"] < b["start_time_ms"] for a, b in zip(windows, windows[1:]))
        assert report["contract"]["holdout_start"] == 8996
        assert sum((output / version["file"]).stat().st_size
                   for version in report["model_versions"]) == summary["model_storage_bytes"]
        assert summary["aws_cost_total_usd"] > 0
    assert reports["frozen"]["metrics"]["update_count"] == 0
    assert reports["periodic"]["metrics"]["retrain_count"] == 2
    assert reports["incremental"]["metrics"]["training_unique_examples"] == 8996
    report = reports["incremental"]
    native = tmp_path / "incremental"
    replay = tmp_path / "replay"
    copytree(native / "models", replay / "models")
    replay_predictions(replay, report, tmp_path)
    evidence = prediction_manifest(replay, report)
    assert evidence["origin"] == "replayed_from_checkpoints"
    assert evidence["parts"] == prediction_manifest(native, report)["parts"]
    tracking = tracking_report(report, evidence)
    assert tracking["tracking"]["metrics"]["fading_accuracy"] == pytest.approx(1)
    assert tracking["tracking"]["metrics"]["drift_tp"] is None
    assert {w["scope"] for w in tracking["windows"]} == {"prequential"}
    write_zia_artifacts(replay, report, tracking)
    for artifact in ("data/index.json", "plots/performance/holdout_confusion.svg",
                     "plots/performance/fading_accuracy.svg", "plots/costs/aws_usd.svg"):
        assert (replay / "artifacts" / artifact).is_file()
