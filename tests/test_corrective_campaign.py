import argparse

import pytest

from scripts.run_corrective_benchmarks import all_jobs, assert_tracking_receipts, make_plan


def arguments(tmp_path, **changes):
    options = {
        "matrix": "configs/matrices/insects-corrected.yaml",
        "families": ["matrix", "upf", "paired"],
        "replication_datasets": ["electricity", "airlines", "gauss", "circles"],
        "output": tmp_path,
        "parallel_cases": 2,
        "seeds": [42, 43, 44],
        "samples": 100,
        "tracking": True,
    }
    return argparse.Namespace(**(options | changes))


def test_full_plan_preserves_every_configuration_without_model_retention(tmp_path):
    plan = make_plan(arguments(tmp_path))
    cases = plan["matrix"]["cases"]
    assert len(cases) == 60 and not plan["matrix"]["skipped"]
    assert len(all_jobs(plan)) == 68
    assert len({j["job_id"] for j in all_jobs(plan)}) == 68
    assert all(c["config"]["artifact_mode"] == "metrics_only" for c in cases)
    assert all(c["config"]["tracking"]["task_evaluation"]["max_windows"] == 4 for c in cases)
    assert not any(c["coordinates"]["policy"] == "online" for c in cases)
    assert plan["exclusions"][0]["configurations"] == 4
    assert plan["exclusions"][0]["runs"] == 12
    assert plan["replications"]["seeds"] == [42, 43, 44]
    assert plan["replications"]["trials"] == 3
    assert plan["replications"]["full_source"]
    assert all(
        c["config"]["policy"]["params"]["seed"] == c["coordinates"]["seed"]
        for c in cases
        if c["coordinates"]["policy"] == "upf"
    )
    assert plan["resource_contract"]["parallel_cases"] == 2
    assert plan["tracking_required"]


def test_smoke_plan_exercises_all_policies_and_both_models(tmp_path):
    plan = make_plan(
        arguments(tmp_path, matrix="configs/matrices/corrective-smoke.yaml", families=["matrix"])
    )
    assert len(plan["matrix"]["cases"]) == 30
    assert {c["coordinates"]["policy"] for c in plan["matrix"]["cases"]} == {
        "none",
        "drift",
        "cara",
        "upf",
        "oracle",
    }


def test_campaign_rejects_unrestricted_model_retention(tmp_path):
    with pytest.raises(ValueError, match="artifact_mode"):
        make_plan(arguments(tmp_path, matrix="configs/matrices/insects-72.yaml"))


@pytest.mark.parametrize("receipt", [{"complete": False}, {"error_type": "TimeoutError"}, "invalid"])
def test_incomplete_nested_tracking_receipt_blocks_completion(receipt):
    with pytest.raises(RuntimeError, match="publication"):
        assert_tracking_receipts([{"trials_detail": [{"policy_tracking": {"mlflow": receipt}}]}])


def test_complete_tracking_receipt_is_accepted():
    assert_tracking_receipts({"candidate_tracking": {"mlflow": {"complete": True}}})


def test_no_tracking_receipt_can_be_checked_without_mutating_source():
    values = {"analysis": {"cost": 0.2}, "mlflow": None}
    assert_tracking_receipts(values)
    assert values["mlflow"] is None
