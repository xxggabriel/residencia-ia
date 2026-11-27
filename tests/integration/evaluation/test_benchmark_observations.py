import pytest
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_curve

from continual_bench.evaluation.tracking_observations import (
    batch_quality,
    observations,
    process_peak_rss_bytes,
)
from continual_bench.infrastructure.observation_tracking import parent_observation_points


def test_binary_quality_preserves_ties_and_operating_threshold():
    labels = [1, 0, 1, 0, 0, 1, 0]
    scores = [0.91, 0.91, 0.8, 0.08, 0.01, 0.035, 0.035]
    threshold = 0.03
    predictions = [int(score >= threshold) for score in scores]
    quality = batch_quality(
        labels,
        predictions,
        classes=[0, 1],
        positive_class=1,
        decision_threshold=threshold,
        scores=scores,
        low_fpr=True,
    )
    assert quality["accuracy"] == accuracy_score(labels, predictions)
    assert quality["macro_f1"] == pytest.approx(f1_score(labels, predictions, labels=[0, 1], average="macro"))
    assert quality["average_precision"] == pytest.approx(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    assert quality["recall_at_fpr_0_01"] == max(tpr[fpr <= 0.01]) == 0
    assert quality["confusion_matrix"] == [[1, 3], [0, 3]]
    assert quality["positive_precision"] == 0.5
    assert quality["decision_threshold"] == threshold


def test_multiclass_fixed_universe_counts_untrained_predictions_as_errors():
    quality = batch_quality(["a", "a", "b"], ["b", "a", "__untrained__"], classes=["a", "b", "c"])
    assert quality["classes"] == ["a", "b", "c", "__untrained__"]
    assert quality["class_support"] == [2, 1, 0, 0]
    assert quality["class_recall"] == [0.5, 0.0, None, None]
    assert quality["accuracy"] == 1 / 3
    assert quality["macro_f1"] == pytest.approx(2 / 9)


def test_regression_is_aggregated_over_predictions_and_missing_values_have_reasons():
    quality = batch_quality([0, 0, 6], [0, 0, 0], target_unit="kWh")
    assert quality["mae"] == 2
    assert quality["rmse"] == pytest.approx(12**0.5)
    assert quality["residuals"]["error"] == [0, 0, 6]
    report = {"tracking_observations": observations("regression", {"test": quality}, drift={"alarms": []})}
    data = report["tracking_observations"]
    assert data["scopes"]["test"]["average_precision"] is None
    assert data["scopes"]["test"]["availability"]["average_precision"]["reason"] == "not_applicable_to_task"
    assert data["drift"]["alarm_count"] == 0
    assert data["drift"]["recall"] is None
    assert data["resources"]["availability"]["cost_total"]["reason"]
    points = list(parent_observation_points(report))
    assert ("observations_v1/quality/test/rmse", 0, quality["rmse"]) in points
    assert process_peak_rss_bytes() is None or process_peak_rss_bytes() > 0


def test_release_delay_uses_change_timestamp_on_the_source_clock():
    from continual_bench.evaluation.tracking import detector_metrics

    report = {
        "dataset": {"changepoints": [2]},
        "config": {"detector": {"name": "adwin"}, "dataset": {"epoch": 0}},
        "observed": 10,
        "metrics": {"n": 10},
        "alarms": [{"seq": 4, "released_at": 125}],
    }
    settings = {"detection_window": 4}
    legacy, old_details, _ = detector_metrics(report, settings)
    values, details, _ = detector_metrics(report, settings, change_timestamps={2: 110})
    assert values == legacy
    assert legacy["drift_delay_release_mean"] == 123
    assert values["drift_delay_mean_samples"] == 2
    assert "release_delay_source_clock" not in old_details["matches"][0]
    assert details["matches"][0]["release_delay_source_clock"] == 15


def test_iterative_publisher_restarts_from_retained_report(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    from continual_bench.evaluation.environment import encoded
    from continual_bench.infrastructure.iterative_tracking import IterativeMlflowTracker

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlflow").as_uri()
    client = MlflowClient(tracking_uri=uri)
    args = {"dataset": "sea", "task": "binary", "model": "logistic", "campaign": "publication-retry"}
    first = IterativeMlflowTracker(uri, client=client, **args)
    run_id = first.start({"seed": 42}, 18)
    first.log_epoch({"train:loss": 0.3})
    directory = tmp_path / "evidence"
    directory.mkdir()
    report = {"summary": {"accuracy": 0.9}, "complete": True}
    (directory / "result.json").write_bytes(encoded(report))
    first.fail("interrupted tracking")
    retry = IterativeMlflowTracker(uri, client=client, **args)
    published = retry.resume_publication(directory)
    assert published["mlflow_run_id"] == run_id
    assert retry.resume_publication(directory) == published
    assert client.get_run(run_id).info.status == "FINISHED"
    assert [(m.step, m.value) for m in client.get_metric_history(run_id, "train:loss")] == [(1, 0.3)]
    assert len(client.get_metric_history(run_id, "summary/accuracy")) == 1
    assert len(client.search_runs([first.experiment_id])) == 1


def test_rest_uses_paginated_history_when_resuming_an_accepted_batch(monkeypatch):
    from continual_bench.infrastructure.aws.mlflow_tracking import SageMakerMlflowTracker

    tracker = SageMakerMlflowTracker(
        tracking_uri="http://mlflow.example",
        dataset="sea",
        policy="periodic",
        model_name="logistic",
        run_id="rest-retry",
    )
    tracker.mlflow_run_id = "original"
    ledger = [{"key": "observations_v1/step/loss", "step": i, "value": 1 / i, "timestamp": 0} for i in (1, 2)]
    desired = ledger.copy() + [{"key": ledger[0]["key"], "step": 3, "value": 0.25, "timestamp": 0}]
    calls = []

    def request(_method, endpoint, *, payload=None, params=None):
        if endpoint.endswith("/runs/get"):
            return {"run": {"data": {"metrics": ledger[-1:]}}}
        if endpoint.endswith("/get-history"):
            page = int(params.get("page_token", "0"))
            response = {"metrics": ledger[page : page + 1]}
            if page + 1 < len(ledger):
                response["next_page_token"] = str(page + 1)
            return response
        assert endpoint.endswith("/log-batch")
        calls.extend(payload["metrics"])
        ledger.extend(payload["metrics"])
        return {}

    monkeypatch.setattr(tracker, "_request", request)
    assert tracker.log_completed_metrics(desired)
    assert tracker.log_completed_metrics(desired)
    assert calls == desired[2:]
    assert ledger == desired
