import json
from pathlib import Path

from continual_bench.application.campaign import Campaign, execute_campaign
from continual_bench.application.runner import evaluate, prepare, run
from continual_bench.config import Config


def configuration():
    return Config.model_validate({
        "name": "task_evaluation-integration", "dataset": {"samples": 120, "label_delay": 7},
        "warmup": 20, "metric_window": 20, "replay_window": 40,
        "policy": {"name": "performance", "min_samples": 20, "min_window": 8,
                   "cooldown": 10, "params": {"monitor_window": 10, "error_threshold": 0}},
        "promotion": {"validation_samples": 8, "min_improvement": 1},
        "tracking": {"probe_samples": 4, "continual_references": False,
                     "costs": {"train_sample": 1, "update": 2},
                     "task_evaluation": {"horizon": 8, "gap": 2, "max_windows": 3}},
    })


def test_runner_resume_preserves_frozen_horizons_and_prices_rejected_candidates(tmp_path):
    cfg = configuration()
    expected = run(cfg, tmp_path / "fresh")
    manifest = prepare(cfg, tmp_path / "resume")
    evaluate(cfg, manifest, tmp_path / "resume", stop_after=43)
    actual = evaluate(cfg, manifest, tmp_path / "resume")
    assert expected["prediction_digest"] == actual["prediction_digest"]
    assert expected["task_evaluation"]["near_future"] == actual["task_evaluation"]["near_future"]
    assert actual["task_evaluation"]["near_future"]["completed_windows"] == 3
    assert actual["candidates"] and not actual["updates"]
    trained = sum(c["train_samples"] for c in actual["candidates"])
    metrics = actual["tracking"]["metrics"]
    assert metrics["training_presentations"] == cfg.warmup + trained
    assert metrics["cost_update_events"] == 2 * len(actual["candidates"])
    assert metrics["cost_training_presentations"] == cfg.warmup + trained
    assert metrics["upf_total_cost"] is None
    rejected = actual["task_evaluation"]["resources"]["candidates"]["rejected_actions"]
    assert rejected == len(actual["candidates"])
    supplement = tmp_path / "resume" / cfg.identity() / "task_evaluation-metrics.json"
    assert json.loads(supplement.read_text()) == actual["task_evaluation"]


def test_declarative_smoke_campaign_runs_all_cases_and_reuses_reports(tmp_path):
    path = Path(__file__).parents[1] / "configs/campaign-task-smoke.json"
    specification = Campaign.model_validate_json(path.read_text())
    first = execute_campaign(specification, tmp_path)
    second = execute_campaign(specification, tmp_path)
    assert first == second and second["complete"]
    assert len(second["cases"]) == 6
    for case in second["cases"].values():
        report = json.loads(Path(case["report"]).read_text())
        assert report["task_evaluation"]["near_future"]["completed_windows"] == 3
        if report["config"]["dataset"]["name"] in {"hyperplane", "rbf_drift"}:
            assert report["task_evaluation"]["detection"]["metrics"]["drift_f1"] is None
