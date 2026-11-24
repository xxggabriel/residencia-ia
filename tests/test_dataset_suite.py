import argparse
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from continual_bench.config import IterativeTrainingConfig
from continual_bench.evaluation.zia_metrics import ClassificationMetrics
from scripts import run_dataset_suite as suite
from scripts import verify_dataset_suite as qa


def arguments(tmp_path, **changes):
    return argparse.Namespace(
        **{
            "root": tmp_path / "inputs",
            "output": tmp_path / "suite",
            "scope": "smoke",
            "datasets": list(suite.DATASETS),
            "policies": list(suite.POLICIES),
            "seeds": [42, 43, 44],
            "tracking_config": suite.PROJECT / "configs/base/tracking-dataset-suite.yaml",
            "max_epochs": 3,
            "online_only": False,
            "workers": 1,
            "tracking": True,
            **changes,
        }
    )


def mock_sources(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"immutable prepared input")

    def load(root, dataset, scope):
        return path, {"sha256": suite.sha256_file(path), "rows": 10000, "dataset": dataset, "scope": scope}

    monkeypatch.setattr(suite, "load_stream_manifest", load)
    monkeypatch.setattr(suite, "code_identity", lambda: ({"git_revision": "a" * 40}, {"packages": {}}))


def test_suite_requires_all_three_seeds_and_real_names(tmp_path, monkeypatch):
    mock_sources(tmp_path, monkeypatch)
    plan = suite.make_plan(arguments(tmp_path))
    assert len(plan["cases"]) == 231
    assert len({c["case_id"] for c in plan["cases"]}) == 231
    assert plan["excluded_policies"] == ["online"]
    assert all(
        c["config"]["tracking_settings"]["retraining"]["seed"] == c["config"]["seed"] for c in plan["cases"]
    )
    assert all(c["config"]["iterative_training"]["seed"] == c["config"]["seed"] for c in plan["cases"])
    assert all(c["config"]["artifact_mode"] == "metrics_only" for c in plan["cases"])
    with pytest.raises(ValueError, match="requires seeds"):
        suite.make_plan(arguments(tmp_path, seeds=[42, 43, 45]))
    with pytest.raises(ValueError, match="Online"):
        suite.make_plan(arguments(tmp_path, policies=["online"]))


def test_generated_base_requests_do_not_launch_online_or_lose_seeds(tmp_path, monkeypatch):
    mock_sources(tmp_path, monkeypatch)
    args = arguments(
        tmp_path,
        write_requests=tmp_path / "requests",
        release="a" * 40,
        request_stage="base",
        metric_cache_sources=None,
    )
    plan = suite.make_plan(args)
    suite.write_requests(args, plan)
    index = qa.read(args.write_requests / "index.json")
    assert len(index["requests"]) == 7 and index["stage"] == "base"
    assert len({r["campaign_id"] for r in index["requests"]}) == 7
    for row in index["requests"]:
        request = qa.read(args.write_requests / row["file"])["request"]
        assert request["matrix"]["policies"] == list(suite.BASE_POLICIES)
        assert request["matrix"]["seeds"] == [42, 43, 44]
        assert request["max_parallel"] == 32 and request["queue_timeout_seconds"] == 86400
        assert request["matrix"]["artifact_mode"] == "metrics_only"


def test_cached_requests_require_all_three_verified_cache_references(tmp_path, monkeypatch):
    mock_sources(tmp_path, monkeypatch)
    args = arguments(
        tmp_path,
        write_requests=tmp_path / "requests",
        release="a" * 40,
        request_stage="cached",
        metric_cache_sources=None,
    )
    with pytest.raises(ValueError, match="all three CARA"):
        suite.write_requests(args, suite.make_plan(args))


def make_case(tmp_path, seed=42, classes=2):
    directory = tmp_path / f"case-{seed}-{classes}"
    directory.mkdir()
    (directory / "predictions").mkdir()
    training = IterativeTrainingConfig(max_epochs=3, patience=3, seed=seed).model_dump()
    key = "b" * 24
    labels = np.arange(12, dtype=np.uint8) % classes
    probabilities = np.full((12, classes), 0.1 / (classes - 1))
    probabilities[np.arange(12), labels] = 0.9
    config = {
        "dataset": "fixture",
        "scope": "smoke",
        "policy": "frozen",
        "seed": seed,
        "artifact_mode": "metrics_only",
        "input_sha256": "c" * 64,
        "iterative_training": training,
    }
    case = {"case_id": key, "config": config}
    table = pa.table(
        {
            "seq": np.arange(4, 16, dtype=np.uint32),
            "event_time_ms": np.arange(4, 16, dtype=np.int64),
            "label": labels,
            "prediction": labels,
            "holdout": np.arange(4, 16) >= 12,
            "model_end_seq": np.full(12, 4, np.uint32),
            **{f"p_{i}": probabilities[:, i] for i in range(classes)},
        }
    )
    path = directory / "predictions/part-0000.parquet"
    pq.write_table(table, path)
    counts = {}
    for name, start, stop in (("prequential", 0, 8), ("holdout", 8, 12)):
        metrics = ClassificationMetrics(classes, 0.5)
        metrics.add(labels[start:stop], probabilities[start:stop])
        counts[name] = metrics.metrics()[0]
    window1, window2 = deepcopy(counts["prequential"]), deepcopy(counts["holdout"])
    window1.update(end_seq=12, scope="prequential", start_time_ms=4, end_time_ms=11)
    window2.update(end_seq=16, scope="holdout", start_time_ms=12, end_time_ms=15)
    summary = {
        **counts["prequential"],
        "initial_train_rows": 4,
        "dataset_rows": 16,
        "training_used_post_warmup": 0,
    }
    names = [str(i) for i in range(classes)]
    report = {
        "complete": True,
        "artifact_mode": "metrics_only",
        "dataset": "fixture",
        "scope": "smoke",
        "policy": "frozen",
        "model_params": {"seed": seed},
        "metrics": summary,
        "holdout": counts["holdout"],
        "windows": [window1, window2],
        "cost_sensitivity": [[0, 0], [1, 0]],
        "protocol": "fixture",
        "contract": {
            "input": {"sha256": config["input_sha256"], "rows": 16, "source": {"classes": names}},
            "iterative_training": training,
            "score_bins": 4096,
            "decision_threshold": 0.5,
            "holdout_start": 12,
            "environment": {},
            "block_rows": 4,
            "decision_rule": "fixture",
        },
        "training_history": [{"seed": seed, "checkpoint": {"file": None}}],
        "model_versions": [{"file": None}],
        "alarms": [],
        "costs": {"settings": {}},
    }
    index = {
        "dataset_sha256": config["input_sha256"],
        "classes": names,
        "rows": 12,
        "fading_accuracy": [],
        "fading_factor": 0.99,
        "origin": "native",
        "parts": [{"file": path.name, "bytes": path.stat().st_size, "rows": 12, "sha256": qa.sha256(path)}],
    }
    qa.save(directory / "report.json", report)
    qa.save(directory / "predictions-manifest.json", index)
    return directory, case


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("classes", [2, 4])
def test_prediction_reconstruction_preserves_binary_and_multiclass_semantics(tmp_path, seed, classes):
    directory, case = make_case(tmp_path, seed, classes)
    result = qa.verify_case(directory, case, common_range=(8, 12))
    metrics = result["common_comparison"]["metrics"]
    assert result["prediction_rows"] == 12 and result["fit_seed_checks"] == 1
    assert metrics["accuracy"] == 1 and metrics["n"] == 4
    assert metrics["log_loss"] == pytest.approx(-np.log(0.9))
    expected_half_brier = (0.1**2 + (classes - 1) * (0.1 / (classes - 1)) ** 2) / 2
    assert metrics["brier_half_sum"] == pytest.approx(expected_half_brier)


@pytest.mark.parametrize("field", ["accuracy", "pr_auc"])
def test_reconstruction_rejects_forged_report_metrics(tmp_path, field):
    directory, case = make_case(tmp_path)
    report = qa.read(directory / "report.json")
    report["metrics"][field] = 0
    qa.save(directory / "report.json", report)
    with pytest.raises(ValueError, match="differs"):
        qa.verify_case(directory, case)


def test_reconstruction_rejects_seed_loss_model_files_and_prediction_corruption(tmp_path):
    directory, case = make_case(tmp_path)
    report = qa.read(directory / "report.json")
    report["training_history"][0].pop("seed")
    qa.save(directory / "report.json", report)
    with pytest.raises(ValueError, match="per-fit seed"):
        qa.verify_case(directory, case)
    report["training_history"][0]["seed"] = 42
    qa.save(directory / "report.json", report)
    model = directory / "weights.npy"
    model.write_bytes(b"forbidden state")
    with pytest.raises(ValueError, match="model or checkpoint"):
        qa.verify_case(directory, case)
    model.unlink()
    path = directory / "predictions/part-0000.parquet"
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        qa.verify_case(directory, case)


def test_common_cohort_starts_after_last_offline_block():
    report = {
        "retraining": {"setup": {"offline_batches": 3}},
        "metrics": {"initial_train_rows": 4},
        "windows": [{"scope": "prequential", "end_seq": end} for end in (8, 12, 16, 20)],
    }
    assert qa.calibration_end(report) == 12


def test_comparison_does_not_invent_sd_for_one_defined_seed(tmp_path):
    results = [
        {
            "dataset": "x",
            "policy": "upf",
            "seed": seed,
            "common_comparison": {"metrics": {"n": 10, "pr_auc": value}},
        }
        for seed, value in zip((42, 43, 44), (None, 0.7, None), strict=True)
    ]
    qa.comparison_tables(results, tmp_path)
    import csv

    rows = list(csv.DictReader((tmp_path / "comparison-mean-sd.csv").open()))
    ap = next(r for r in rows if r["metric"] == "pr_auc")
    assert ap["n_defined"] == "1" and ap["n_undefined"] == "2" and ap["sample_sd"] == ""


def test_suite_readback_aggregates_matching_cohorts_and_keeps_receipt_hashes(tmp_path):
    import shutil

    root = tmp_path / "suite"
    (root / "cases").mkdir(parents=True)
    (root / "results").mkdir()
    code, cases = {"git_revision": "a" * 40}, []
    for seed in (42, 43, 44):
        directory, case = make_case(tmp_path, seed)
        case["case_id"] = str(seed)
        target = root / "cases" / case["case_id"]
        shutil.move(directory, target)
        files = {p.relative_to(target).as_posix(): qa.sha256(p) for p in target.rglob("*") if p.is_file()}
        qa.save(
            root / "results" / (case["case_id"] + ".json"),
            {
                "status": "passed",
                "case": case,
                "code": code,
                "files": files,
                "tracking": None,
            },
        )
        cases.append(case)
    qa.save(
        root / "suite-plan.json",
        {
            "schema": "ordered-dataset-suite-v1",
            "code": code,
            "cases": cases,
            "scope": "smoke",
            "tracking_required": False,
        },
    )
    result = qa.verify_suite(root)
    assert result["complete"] and result["coverage"]["seeds"] == [42, 43, 44]
    assert all(c["common_comparison"]["seq_range"] == [4, 12] for c in result["cases"])
    assert result["checks"]["native_tracking_catalog_verified"] is False
    assert (root / "comparison-by-seed.csv").is_file()
    report = qa.read(root / "cases/42/report.json")
    report["metrics"]["accuracy"] = 0
    qa.save(root / "cases/42/report.json", report)
    with pytest.raises(ValueError, match="evidence changed"):
        qa.verify_suite(root)


def test_native_readback_checks_entire_history_not_only_latest(tmp_path):
    directory, _ = make_case(tmp_path)
    report = qa.read(directory / "report.json")
    adapted = qa.tracking_report(report, qa.read(directory / "predictions-manifest.json"))
    points = qa.metric_points(adapted["tracking"])

    class Client:
        def get_run(self, run_id):
            return SimpleNamespace(info=SimpleNamespace(status="FINISHED"))

        def get_metric_history(self, run_id, key):
            return [
                SimpleNamespace(step=step, value=value)
                for name, step, value in points
                if name == key and value is not None
            ]

    receipt = {"complete": True, "mlflow_run_id": "run"}
    verified = qa.verify_native(directory, report, receipt, Client())
    assert verified["complete"] and verified["points"] > 10
    client = Client()
    original = client.get_metric_history
    client.get_metric_history = lambda run, key: original(run, key)[-1:]
    with pytest.raises(ValueError, match="axis differs"):
        qa.verify_native(directory, report, receipt, client)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_actual_evaluator_evidence_passes_reconstruction_with_the_effective_seed(tmp_path, monkeypatch, seed):
    from continual_bench.config import TrackingConfig
    from continual_bench.config.loader import load_model
    from continual_bench.datasets.streams import StreamDataset
    from continual_bench.evaluation import stream_benchmark

    path = tmp_path / "fixture.parquet"
    labels = (np.arange(128) % 2).astype(np.uint8)
    pq.write_table(
        pa.table(
            {
                "event_time": np.arange(128, dtype=np.int64),
                "label": labels,
                "feature": np.arange(128, dtype=np.float32) / 128,
            }
        ),
        path,
    )
    manifest = {
        "dataset": "fixture",
        "scope": "smoke",
        "rows": 128,
        "sha256": qa.sha256(path),
        "features": ["feature"],
        "source": {"classes": ["0", "1"], "revision": "fixture-v1"},
        "license": "test fixture",
    }
    monkeypatch.setattr(stream_benchmark, "load_stream_manifest", lambda *_: (path, manifest))
    monkeypatch.setattr(
        stream_benchmark,
        "stream_spec",
        lambda *_: StreamDataset(
            "fixture", "catalog", ("0", "1"), "multinomial_logistic", {"smoke": 16, "full": 16}
        ),
    )
    training = IterativeTrainingConfig(max_epochs=3, patience=3, seed=seed).model_dump()
    settings = load_model(
        TrackingConfig, suite.PROJECT / "configs/base/tracking-dataset-suite.yaml"
    ).model_dump()
    settings["retraining"] = {"seed": seed}
    directory = tmp_path / "result"
    report = stream_benchmark.run(
        tmp_path,
        "fixture",
        "smoke",
        "frozen",
        directory,
        settings,
        iterative_config=training,
        artifact_mode="metrics_only",
    )
    case = {
        "case_id": "fixture",
        "config": {
            "dataset": "fixture",
            "scope": "smoke",
            "policy": "frozen",
            "seed": seed,
            "input_sha256": manifest["sha256"],
            "iterative_training": training,
        },
    }
    result = qa.verify_case(directory, case)
    assert report["complete"] and result["complete"]
    assert result["source_rows"] == 128 and result["fit_seed_checks"] == 1
