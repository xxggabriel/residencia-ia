"""Descriptive comparisons must preserve paired data and metric direction without assuming reproduction."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


def comparison_module():
    spec = importlib.util.spec_from_file_location(
        "compare_nonlinear_campaign", Path(__file__).parents[1] / "scripts/compare_nonlinear_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(model, dataset, mode, seed, *, previous=False):
    offline = mode == "offline"
    cfg = {
        "name": "versioned-v4" if previous else "nonlinear-v5",
        "model": {"name": model, "params": {} if previous else {"fixture_parameter": model}},
        "dataset": {"name": dataset, "seed": seed, "samples": 100, "label_delay": 3},
        "mode": mode,
        "warmup": 10,
        "detector": {"name": "none"},
        "policy": {"name": "none" if offline else "periodic"},
        "training_budget": None if offline else 90,
    }
    updates = [] if offline else [{"train_samples": 10}, {"train_samples": 20}]
    report = {
        "run_key": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:24],
        "config": cfg,
        "complete": True,
        "dataset": {"sha256": hashlib.sha256(f"{dataset}:{seed}".encode()).hexdigest()},
        "environment": {"code_sha256": "old-source" if previous else "new-source"},
        "metrics": {
            "n": 40 if offline else 90,
            "accuracy": 0.6 if seed == 42 else 0.7,
            "f1": 0.6,
            "brier": 0.2,
            "log_loss": 0.5,
        },
        "alarms": [],
        "updates": updates,
        "training": None if offline else {"used": 30},
        "compute_seconds": 2.5,
        "wall_seconds_this_attempt": 3.0,
        "peak_rss_bytes_process": 1000,
        "windows": [{"end_seq": 99, "accuracy": 0.6}],
    }
    if offline:
        report["split"] = {"train": [0, 57], "test": [60, 100]}
    else:
        report["prediction_digest"] = f"digest-{dataset}-{seed}"
    if model == "hoeffding_tree":
        report["metrics"].update(
            {"accuracy": 0.8 if seed == 42 else 0.6, "f1": 0.5, "brier": 0.1, "log_loss": 0.7}
        )
    elif model == "knn":
        report["metrics"]["accuracy"] = 0.6 if seed == 42 else 0.8
    return {"report": report, "publication_verified": True, "receipt": {"model_uri": "fixture://model"}}


@pytest.fixture
def evidence():
    cases = [("sea", "streaming", 42), ("sea", "streaming", 43), ("airlines", "offline", 42)]
    previous = [record("logistic", *case, previous=True) for case in cases]
    current = [record(model, *case) for case in cases for model in ("logistic", "hoeffding_tree", "knn")]
    return (
        {"complete_campaign": True, "runs_verified": len(current), "records": current},
        {"complete_campaign": True, "runs_verified": len(previous), "records": previous},
    )


def analyze(evidence, **kwargs):
    return comparison_module().analyze(*evidence, expected_cases=3, synthetic_replicates=2, **kwargs)


def test_pairing_preserves_cases_and_uses_sample_sd_and_metric_direction(evidence):
    result = analyze(evidence)
    assert result["runs"] == 9 and result["cases_per_model"] == 3
    assert result["dataset_sha_pairs_verified"] == 9
    assert len(result["groups"]) == 6 and len(result["paired_model_comparisons"]) == 4
    tree = next(
        r
        for r in result["paired_model_comparisons"]
        if r["model"] == "hoeffding_tree" and r["dataset"] == "sea"
    )
    assert tree["strategy"] == "periodic@90"
    acc = tree["metrics"]["accuracy"]
    assert acc["mean"] == pytest.approx(0.05)
    assert acc["sd"] == pytest.approx(0.21213203435596426)
    assert (acc["wins"], acc["ties"], acc["losses"]) == (1, 0, 1)
    assert tree["metrics"]["brier"]["wins"] == 2
    assert tree["metrics"]["brier"]["higher_is_better"] is False
    assert tree["metrics"]["log_loss"]["losses"] == 2
    knn = next(r for r in result["paired_model_comparisons"] if r["model"] == "knn" and r["dataset"] == "sea")
    assert (knn["metrics"]["accuracy"]["wins"], knn["metrics"]["accuracy"]["ties"]) == (1, 1)
    group = next(r for r in result["groups"] if r["model"] == "logistic" and r["dataset"] == "sea")
    assert group["metrics"]["accuracy"]["mean"] == pytest.approx(0.65)
    assert group["metrics"]["accuracy"]["sd"] == pytest.approx(0.07071067811865474)
    assert group["training_used"]["mean"] == 30 and group["updates"]["mean"] == 2
    assert group["compute_seconds"]["mean"] == 2.5 and group["wall_seconds"]["mean"] == 3
    assert group["peak_rss_bytes"]["mean"] == 1000
    real = next(r for r in result["groups"] if r["model"] == "logistic" and r["dataset"] == "airlines")
    assert real["metrics"]["accuracy"]["n"] == 1 and real["metrics"]["accuracy"]["sd"] is None
    assert real["training_used"]["mean"] == 0


def test_logistic_bridge_reports_complete_and_synthetic_real_counts(evidence):
    bridge = analyze(evidence)["logistic_v4_comparison"]
    assert bridge["all"]["cases"] == bridge["all"]["metrics_exact_matches"] == 3
    assert bridge["all"]["prediction_digest_exact_matches"] == 2
    assert bridge["all"]["offline_split_and_window_exact_matches"] == 1
    assert bridge["synthetic"]["cases"] == 2 and bridge["real"]["cases"] == 1
    assert not bridge["differences"]


def test_logistic_metric_or_digest_differences_are_evidence_not_an_abort(evidence):
    current, previous = deepcopy(evidence)
    report = current["records"][0]["report"]
    report["metrics"]["accuracy"] = 0.65
    report["prediction_digest"] = "different"
    bridge = analyze((current, previous))["logistic_v4_comparison"]
    assert bridge["all"]["metrics_exact_matches"] == 2
    assert bridge["all"]["prediction_digest_exact_matches"] == 1
    assert len(bridge["differences"]) == 1
    difference = bridge["differences"][0]
    assert not difference["metrics_equal"] and not difference["prediction_digest_equal"]
    assert difference["metric_deltas"]["accuracy"] == pytest.approx(0.05)
    assert difference["previous_prediction_digest"] == "digest-sea-42"


def test_logistic_offline_differences_are_recorded_separately(evidence):
    current, previous = deepcopy(evidence)
    report = next(
        r["report"]
        for r in current["records"]
        if r["report"]["config"]["model"]["name"] == "logistic" and r["report"]["config"]["mode"] == "offline"
    )
    report["windows"][0]["accuracy"] = 0.9
    bridge = analyze((current, previous))["logistic_v4_comparison"]
    assert bridge["all"]["metrics_exact_matches"] == 3
    assert bridge["all"]["offline_split_exact_matches"] == 1
    assert bridge["all"]["offline_window_exact_matches"] == 0
    assert not bridge["differences"][0]["windows_equal"]


def test_missing_logistic_digest_is_not_counted_as_a_match(evidence):
    current, previous = deepcopy(evidence)
    del current["records"][0]["report"]["prediction_digest"]
    bridge = analyze((current, previous))["logistic_v4_comparison"]
    assert bridge["all"]["missing_prediction_digests"] == 1
    assert bridge["all"]["prediction_digest_exact_matches"] == 1
    assert bridge["differences"][0]["current_prediction_digest"] is None


@pytest.mark.parametrize(
    "change,message",
    [
        ("dataset", "dataset SHA mismatch"),
        ("case", "not present in v4"),
        ("duplicate", "duplicate run_key"),
        ("model", "unexpected model family"),
        ("evaluation", "evaluation count mismatch"),
        ("source", "code changed during campaign"),
    ],
)
def test_incompatible_or_duplicate_evidence_is_rejected(evidence, change, message):
    current, previous = deepcopy(evidence)
    report = current["records"][1]["report"]
    if change == "dataset":
        report["dataset"]["sha256"] = "wrong-dataset"
    elif change == "case":
        report["config"]["dataset"]["label_delay"] = 9
    elif change == "duplicate":
        current["records"][1] = deepcopy(current["records"][0])
    elif change == "model":
        report["config"]["model"]["name"] = "unplanned"
    elif change == "evaluation":
        report["metrics"]["n"] = 1
    else:
        report["environment"]["code_sha256"] = "other-source"
    with pytest.raises(AssertionError, match=message):
        analyze((current, previous))


def test_wrong_replication_is_rejected(evidence):
    with pytest.raises(AssertionError, match="unexpected group replicate count"):
        comparison_module().analyze(*evidence, expected_cases=3, synthetic_replicates=3)


def test_unverified_or_nonfinite_inputs_are_rejected(evidence):
    current, previous = deepcopy(evidence)
    current["records"][0]["publication_verified"] = False
    with pytest.raises(AssertionError, match="unverified/incomplete"):
        analyze((current, previous))
    current["records"][0]["publication_verified"] = True
    current["records"][0]["report"]["metrics"]["brier"] = float("nan")
    with pytest.raises(AssertionError, match="invalid brier"):
        analyze((current, previous))


def frozen_manifest(current):
    return {
        "source_sha256": "new-source",
        "expected_runs": len(current["records"]),
        "configurations": [
            {"run_key": row["report"]["run_key"], "config": deepcopy(row["report"]["config"])}
            for row in current["records"]
        ],
    }


def test_explicit_per_run_source_override_authorizes_only_that_run(evidence):
    current, previous = deepcopy(evidence)
    manifest = frozen_manifest(current)
    current["records"][1]["report"]["environment"]["code_sha256"] = "recovered-source"
    manifest["configurations"][1]["source_sha256"] = "recovered-source"
    result = analyze((current, previous), manifest=manifest)
    assert result["source_sha256"] == "new-source"
    assert result["source_revisions"] == ["new-source", "recovered-source"]
    assert result["source_counts"] == {"new-source": 8, "recovered-source": 1}
    assert result["logistic_v4_comparison"]["all"]["metrics_exact_matches"] == 3


@pytest.mark.parametrize("override", ["global_only", "wrong_run", "wrong_hash"])
def test_unlisted_per_run_source_is_not_authorized_by_revision_list(evidence, override):
    current, previous = deepcopy(evidence)
    manifest = frozen_manifest(current)
    current["records"][1]["report"]["environment"]["code_sha256"] = "recovered-source"
    manifest["source_revisions"] = ["new-source", "recovered-source"]
    if override == "wrong_run":
        manifest["configurations"][2]["source_sha256"] = "recovered-source"
    elif override == "wrong_hash":
        manifest["configurations"][1]["source_sha256"] = "another-source"
    with pytest.raises(AssertionError, match="unauthorized per-run code source"):
        analyze((current, previous), manifest=manifest)


def test_source_authorization_also_requires_unchanged_manifest_configuration(evidence):
    current, previous = deepcopy(evidence)
    manifest = frozen_manifest(current)
    current["records"][1]["report"]["config"]["model"]["params"]["fixture_parameter"] = "retuned"
    with pytest.raises(AssertionError, match="configuration differs from manifest"):
        analyze((current, previous), manifest=manifest)


def test_source_manifest_must_cover_exact_run_ids(evidence):
    current, previous = deepcopy(evidence)
    manifest = frozen_manifest(current)
    manifest["configurations"][1]["run_key"] = "unknown-run"
    with pytest.raises(AssertionError, match="manifest run coverage differs"):
        analyze((current, previous), manifest=manifest)
