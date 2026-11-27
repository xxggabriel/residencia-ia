"""Complete metric-only native publication and attempt-specific live evidence."""

import json
from collections import Counter

import pytest

from continual_bench.infrastructure.aws import mlflow_tracking


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_metrics_only_publishes_uncapped_system_and_retained_progress_without_models(
    tmp_path,
    monkeypatch,
    seed,
):
    report = {
        "artifact_mode": "metrics_only",
        "dataset": "social",
        "policy": "frozen",
        "contract": {"artifact_mode": "metrics_only", "iterative_training": {"seed": seed}},
        "metrics": {"accuracy": 0.8},
        "system_metrics": {
            "samples": [{"timestamp_ms": 1000 + i, "values": {"rss_bytes": i + 1}} for i in range(600)],
        },
    }
    prefix = "attempts/restarted-2/progress"
    (tmp_path / "live-metrics.jsonl").write_text(
        "".join(
            json.dumps(
                {"step": i, "timestamp": 2000 + i, "prefix": prefix, "values": {"rows_processed": i * 10}}
            )
            + "\n"
            for i in range(600)
        )
    )
    tracker = mlflow_tracking.SageMakerMlflowTracker(
        tracking_uri="http://example.invalid",
        dataset="social",
        policy="frozen",
        scope="smoke",
        model_name="multinomial_logistic",
        run_id="run",
    )
    tracker.seed = seed
    tracker.mlflow_run_id = "parent"
    tracker.artifact_uri = "s3://metrics/parent"
    collected = []

    def forbidden(*args, **kwargs):
        pytest.fail("metric publication tried to load or register model weights")

    monkeypatch.setattr(tracker, "_publish_with_library", forbidden)
    monkeypatch.setattr(tracker, "register_run_models", forbidden)
    monkeypatch.setattr(tracker, "ensure_run", lambda: "parent")
    monkeypatch.setattr(tracker, "log_batch", lambda **kwargs: True)
    monkeypatch.setattr(tracker, "log_completed_metrics", lambda points: collected.extend(points) or True)
    monkeypatch.setattr(tracker, "publish_training_rounds", lambda report: [])
    monkeypatch.setattr(tracker, "stage_observation_artifacts", lambda *args: [])
    monkeypatch.setattr(tracker, "finish", lambda status: True)
    monkeypatch.setattr(
        mlflow_tracking,
        "upload_directory_to_mlflow_artifacts",
        lambda *args, **kwargs: {
            "files": 1,
            "bytes": 1,
        },
    )
    result = tracker.publish_run_report(report, output_dir=tmp_path)
    assert result["published"] is True
    counts = Counter(p["key"] for p in collected)
    assert counts["system/rss_bytes"] == 600
    assert counts[prefix + "/rows_processed"] == 600
    assert {p["step"] for p in collected if p["key"] == "system/rss_bytes"} == set(range(600))
    assert {p["value"] for p in collected if p["key"] == prefix + "/rows_processed"} == {
        float(i * 10) for i in range(600)
    }
    assert mlflow_tracking.report_params(tracker, report)["seed"] == str(seed)
    assert {"key": "cb-seed", "value": str(seed)} in tracker.mlflow_tags
    assert not any(key.startswith("registered_") for key in result)


def test_metrics_only_live_points_do_not_coalesce_or_mix_attempts(monkeypatch):
    tracker = mlflow_tracking.SageMakerMlflowTracker(
        tracking_uri="http://example.invalid",
        dataset="fraud",
        policy="frozen",
        model_name="xgboost",
        run_id="run",
        attempt_key="task-2",
    )
    tracker.artifact_mode = "metrics_only"
    tracker._native_checked = True
    captured = []

    class Writer:
        def __init__(self, sender, interval, *, coalesce, max_pending):
            assert coalesce is False

        def submit(self, points):
            captured.extend(points)

    monkeypatch.setattr(mlflow_tracking, "AsyncMetricWriter", Writer)
    for step in range(10):
        tracker.progress(step, {"wall_seconds": float(step)})
    assert len(captured) == 10
    assert {p["key"] for p in captured} == {"attempts/task-2/progress/wall_seconds"}
    assert [p["step"] for p in captured] == list(range(10))


@pytest.mark.parametrize("bad", [float("nan"), "1", True])
def test_retained_progress_rejects_invalid_metric_values(tmp_path, bad):
    (tmp_path / "live-metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "timestamp": 3,
                "prefix": "attempts/task-1/progress",
                "values": {"n": bad},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="value"):
        mlflow_tracking.retained_progress_items(tmp_path)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_cached_candidate_child_runs_identify_the_original_execution(seed):
    from continual_bench.infrastructure.training_tracking import training_round_mlflow

    source = {
        "git_revision": "a" * 40,
        "code_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "seed": seed,
        "policy": "cara",
    }
    report = {
        "contract": {"matrix_metrics_cache": {"sha256": "d" * 64}},
        "training_history": [
            {
                "fit_id": "fit-000001",
                "action": "candidate",
                "summary": {},
                "execution_origin": "reused_metric_evidence",
                "source_matrix": source,
                "steps": [{"step": 1, "global_step": 1, "kind": "epoch", "metrics": {"loss": 0.1}}],
            }
        ],
    }
    tags, params, points = next(training_round_mlflow("parent", report))
    assert tags["cb-fit-execution-origin"] == params["execution_origin"] == "reused_metric_evidence"
    assert params["source_matrix_seed"] == str(seed)
    assert params["source_matrix_git_revision"] == "a" * 40
    assert params["source_matrix_cache_sha256"] == tags["cb-source-matrix-cache-sha256"] == "d" * 64
    assert len(params["source_matrix_sha256"]) == 64 and points
