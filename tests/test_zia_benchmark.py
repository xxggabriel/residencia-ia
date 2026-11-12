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
@pytest.mark.parametrize("online_update", [False, True])
def test_models_roundtrip_and_continue(learner, online_update):
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
    if online_update:
        restored.fit_online(x, 1 - y)
        assert not np.array_equal(restored.predict(x), prediction)
        if learner == "boost":
            assert restored.model.num_boosted_rounds() == first.model.num_boosted_rounds()
    else:
        restored.fit(x, y, True)
    np.testing.assert_allclose(restored.predict(x).sum(axis=1), 1)
    reloaded = make()
    reloaded.load(restored.payload()[0])
    np.testing.assert_array_equal(reloaded.predict(x), restored.predict(x))


@pytest.mark.parametrize("policy", ["drift_adwin", "drift_ddm"])
def test_zia_detectors_receive_the_same_binary_errors(policy):
    from continual_bench.evaluation.zia import drift_alarms

    class Recorder:
        def __init__(self):
            self.errors = []

        def update(self, error):
            self.errors.append(error)
            self.drift_detected = len(self.errors) == 2

    detector = Recorder()
    assert drift_alarms(detector, policy, np.array([[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]]),
                        np.array([1, 1, 0]), {}) == 1
    assert detector.errors == [0, 1, 1]


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
    assert reports["online"]["metrics"]["training_unique_examples"] == 8996
    assert reports["online"]["metrics"]["retrain_count"] == 0
    assert reports["online"]["model"] == reports["frozen"]["model"]
    online_warmup = reports["online"]["model_versions"][0]
    assert online_warmup["sha256"] == reports["frozen"]["model_versions"][0]["sha256"]
    assert {v["action"] for v in reports["online"]["model_versions"]} == {"initial", "online"}
    assert reports["cost_heuristic"]["metrics"]["retrain_count"] == 0
    for policy in ("incremental", "online"):
        report = reports[policy]
        native = tmp_path / policy
        replay = tmp_path / f"replay-{policy}"
        copytree(native / "models", replay / "models")
        replay_predictions(replay, report, tmp_path)
        assert prediction_manifest(replay, report)["parts"] == prediction_manifest(native, report)["parts"]
    report = reports["incremental"]
    replay = tmp_path / "replay-incremental"
    evidence = prediction_manifest(replay, report)
    assert evidence["origin"] == "replayed_from_checkpoints"
    tracking = tracking_report(report, evidence)
    assert tracking["tracking"]["metrics"]["fading_accuracy"] == pytest.approx(1)
    assert tracking["tracking"]["metrics"]["drift_tp"] is None
    assert {w["scope"] for w in tracking["windows"]} == {"prequential"}
    write_zia_artifacts(replay, report, tracking)
    for artifact in ("data/index.json", "plots/performance/holdout_confusion.svg",
                     "plots/performance/fading_accuracy.svg", "plots/costs/aws_usd.svg"):
        assert (replay / "artifacts" / artifact).is_file()


def test_cost_threshold_is_the_bayes_rule_for_calibrated_scores():
    from continual_bench.evaluation.zia_metrics import decision_threshold

    assert decision_threshold({"false_positive": 1, "false_negative": 1}) == 0.5
    assert decision_threshold({"false_positive": 1, "false_negative": 39}) == pytest.approx(0.025)
    assert decision_threshold({"false_positive": 0, "false_negative": 0}) == 0.5


def test_threshold_changes_decisions_but_not_ranking_metrics():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    p = np.array([0.05, 0.15, 0.3, 0.45])
    scores = np.column_stack((1 - p, p))
    default, costed = ClassificationMetrics(2), ClassificationMetrics(2, threshold=0.2)
    default.add(labels, scores)
    costed.add(labels, scores)
    at_half, _ = default.metrics()
    at_cost, _ = costed.metrics()
    assert at_half["balanced_accuracy"] == 0.5 and at_half["tp"] == 0
    assert at_cost["balanced_accuracy"] == 1.0 and at_cost["tp"] == 2
    for name in ("roc_auc", "pr_auc", "macro_roc_auc", "log_loss", "brier"):
        assert at_cost[name] == at_half[name]
    assert at_cost["roc_auc"] == 1.0
    with pytest.raises(ValueError, match="different decision thresholds"):
        default.merge(costed)


def test_detectors_use_the_cost_threshold():
    from continual_bench.evaluation.zia import drift_alarms

    class Recorder:
        drift_detected = False

        def __init__(self):
            self.errors = []

        def update(self, error):
            self.errors.append(error)

    detector = Recorder()
    drift_alarms(detector, "drift_ddm", np.array([[0.8, 0.2], [0.95, 0.05]]), np.array([1, 0]), {}, 0.1)
    assert detector.errors == [0, 0]


def test_binary_run_records_and_replays_the_cost_threshold(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("xgboost")
    import pyarrow.parquet as pq

    from continual_bench.config import TrackingConfig
    from continual_bench.datasets.zia import DATASETS, FORMAT, sha256_file
    from continual_bench.evaluation.environment import encoded
    from continual_bench.evaluation.zia import run
    from continual_bench.evaluation.zia_predictions import prediction_manifest
    from continual_bench.evaluation.zia_publication import replay_predictions

    folder = tmp_path / "ecommerce"
    folder.mkdir()
    path = folder / "smoke.parquet"
    rng = np.random.default_rng(7)
    feature = rng.random(10_000).astype(np.float32)
    y = (rng.random(10_000) < 0.3 * feature).astype(np.uint8)
    pq.write_table(pa.table({"event_time": np.arange(10_000) // 7, "label": y, "feature": feature}), path)
    (folder / "smoke.json").write_bytes(encoded({
        "dataset": "ecommerce", "scope": "smoke", "source": DATASETS["ecommerce"], "format": FORMAT,
        "features": ["feature"], "rows": 10_000, "bytes": path.stat().st_size, "sha256": sha256_file(path),
    }))
    raw = yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    raw["costs"].update(false_positive=1.0, false_negative=9.0)
    tracking = TrackingConfig.model_validate(raw).model_dump()
    output = tmp_path / "incremental"
    report = run(tmp_path, "ecommerce", "smoke", "incremental", output, tracking)
    assert report["contract"]["decision_threshold"] == pytest.approx(0.1)
    assert report["contract"]["decision_rule"].startswith("binary: p>=0.1 ")
    assert report["metrics"]["tp"] > 0 and report["metrics"]["roc_auc"] > 0.6
    native = prediction_manifest(output, report)
    assert native["decision_threshold"] == pytest.approx(0.1)
    replay = tmp_path / "replay"
    copytree(output / "models", replay / "models")
    replay_predictions(replay, report, tmp_path)
    assert prediction_manifest(replay, report)["parts"] == native["parts"]
