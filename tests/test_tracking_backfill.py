"""Offline backfill may infer frozen scores but must preserve original evidence and missing timers."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from continual_bench.application.config import TrackingConfig
from continual_bench.domain.metrics import Metrics


def script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backfill():
    return script("backfill_tracking_metrics")


@pytest.fixture
def offline():
    probabilities = [0.8, 0.2, 0.7, 0.1]
    events = [{"seq": seq, "event_id": f"event-{seq}", "x": {"value": float(seq), "category": "a"},
               "y": label, "timestamp": 1000 + seq, "available_at": 1003 + seq}
              for seq, label in enumerate([True, False, True, False, False, True])]

    def summary(rows):
        metrics = Metrics()
        for event, probability in rows:
            metrics.add(event["y"], probability)
        return {key: metrics.values()[key] for key in ("n", "accuracy", "f1", "brier", "log_loss")}

    test = list(zip(events[2:], probabilities))
    report = {
        "run_key": "fixture-run", "complete": True, "observed": 6,
        "config": {"mode": "offline", "model": {"name": "logistic"}, "warmup": 1,
                   "training_budget": None, "metric_window": 2, "batch_size": 1,
                   "policy": {"name": "none", "action": "reset_retrain"},
                   "detector": {"name": "none"}, "dataset": {"epoch": 1000}},
        "protocol": "temporal_holdout",
        "metrics": summary(test),
        "windows": [{"end_seq": 3, **summary(test[:2])}, {"end_seq": 5, **summary(test[2:])}],
        "split": {"train": [0, 2], "test": [2, 6]},
        "dataset": {"sha256": "a" * 64, "changepoints": None},
        "environment": {"code_sha256": "b" * 64},
        "model_versions": {"state_sha256": "c" * 64},
        "updates": [], "alarms": [], "retention_probes": [],
        "compute_seconds": 12, "wall_seconds_this_attempt": 15, "peak_rss_bytes_process": 4000,
    }
    receipt = {"model_uri": "models:/frozen-offline/1", "mlflow_run_id": "tracked"}
    return report, events, probabilities, receipt


def recovered_response(request, probabilities):
    return {
        "uri": request["uri"], "model": request["model"], "fresh_process": True,
        "seqs": [row["seq"] for row in request["rows"]],
        "event_ids": [row["event_id"] for row in request["rows"]],
        "probabilities": probabilities,
        "bundled_loader": "/artifact/code/continual_bench/adapters/model_loader.py",
        "bundled_model_code": "/artifact/code/continual_bench/adapters/river.py",
    }


def install_fake_inference(monkeypatch, backfill, probabilities, changes=None):
    requests = []

    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert command[1].endswith("recover_offline_scores.py")
        assert kwargs["check"] and kwargs["capture_output"] and kwargs["text"]
        request = json.loads(kwargs["input"])
        requests.append(request)
        result = recovered_response(request, probabilities)
        result.update(changes or {})
        return SimpleNamespace(stdout=json.dumps(result))

    monkeypatch.setattr(backfill.subprocess, "run", run)
    return requests


def test_frozen_score_enrichment_is_exact_and_leaves_original_and_timers_untouched(backfill, offline):
    report, events, probabilities, _ = offline
    original = copy.deepcopy(report)
    tracking = backfill.enriched_report(
        report, events, TrackingConfig().model_dump(mode="json"), probabilities=probabilities
    )["tracking"]
    assert report == original
    for key in backfill.LEGACY_METRICS:
        assert tracking["metrics"][key] == report["metrics"][key]
    assert tracking["metrics"]["roc_auc"] == 0.5
    assert tracking["metrics"]["average_precision"] == 0.75
    assert tracking["metrics"]["ece"] == pytest.approx(0.5)
    assert tracking["metrics"]["mae_probability"] == pytest.approx(0.5)
    assert tracking["series"]["roc_auc"] == [{"step": 3, "value": 1}, {"step": 5, "value": 0}]
    for key in ("prediction_seconds", "prediction_latency_mean_seconds", "initial_train_seconds",
                "train_seconds", "cl_average_accuracy", "cl_forward_transfer"):
        assert tracking["metrics"][key] is None
    assert tracking["metrics"]["evaluation_compute_seconds"] == 12
    assert tracking["metrics"]["evaluation_attempt_wall_seconds"] == 15
    assert len(tracking["derivation"]["tool_script_sha256"]) == 64


def test_legacy_default_keeps_all_score_metrics_missing(backfill, offline):
    report, events, _, _ = offline
    tracking = backfill.enriched_report(report, events, TrackingConfig().model_dump(mode="json"))["tracking"]
    assert all(tracking["metrics"][key] is None
               for key in ("roc_auc", "average_precision", "ece", "mce", "mae_probability"))


@pytest.mark.parametrize("change", ["n", "accuracy", "f1", "brier", "log_loss", "window"])
def test_any_legacy_metric_disagreement_prevents_recovery(backfill, offline, change):
    report, events, probabilities, _ = offline
    if change == "window":
        report["windows"][0]["brier"] += 0.001
    else:
        report["metrics"][change] += 1 if change == "n" else 0.001
    with pytest.raises(ValueError, match="differ from original"):
        backfill.scored_metrics(report, events, probabilities)


def test_score_recovery_tolerance_is_absolute_and_does_not_replace_original_legacy_values(backfill, offline):
    report, events, probabilities, _ = offline
    report["metrics"]["brier"] += 5e-13
    metrics, _, _ = backfill.scored_metrics(report, events, probabilities)
    assert metrics["brier"] == report["metrics"]["brier"]


def test_recovery_subprocess_receives_only_exact_suffix_and_cache_is_idempotent(
    monkeypatch, backfill, offline, tmp_path
):
    report, events, probabilities, receipt = offline
    raw = json.dumps(report).encode()
    original = tmp_path / "report.json"
    original.write_bytes(raw)
    original_stat = original.stat()
    source_sha = hashlib.sha256(raw).hexdigest()
    requests = install_fake_inference(monkeypatch, backfill, probabilities)
    first, provenance = backfill.recover_offline_scores(report, events, receipt, tmp_path, source_sha)
    assert len(requests) == 1
    assert requests[0]["rows"] == [
        {"seq": event["seq"], "event_id": event["event_id"], "features": event["x"]} for event in events[2:]
    ]
    cache = tmp_path / "offline-score-recovery.json"
    ledger = tmp_path / "predictions.jsonl"
    before = {path: path.stat().st_mtime_ns for path in (cache, ledger)}
    second, replay = backfill.recover_offline_scores(report, events, receipt, tmp_path, source_sha)
    assert first == second == probabilities
    assert provenance == replay
    assert len(requests) == 1
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in before.items())
    assert original.read_bytes() == raw and original.stat().st_mtime_ns == original_stat.st_mtime_ns
    rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    assert [row["seq"] for row in rows] == [2, 3, 4, 5]
    assert all("prediction_seconds" not in row and "released_at" not in row for row in rows)
    assert provenance["origin"] == "recovered_frozen_offline_registry_inference"
    enriched = backfill.enriched_report(
        report, events, TrackingConfig().model_dump(mode="json"), probabilities=first, recovery=provenance
    )
    tracking = enriched["tracking"]
    assert tracking["prediction_ledger"]["sha256"] == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert tracking["derivation"]["offline_score_recovery"]["cache_sha256"] == (
        hashlib.sha256(cache.read_bytes()).hexdigest()
    )
    assert len(tracking["derivation"]["recover_helper_sha256"]) == 64


def test_cache_committed_before_ledger_can_resume_without_another_model_inference(
    monkeypatch, backfill, offline, tmp_path
):
    report, events, probabilities, receipt = offline
    requests = install_fake_inference(monkeypatch, backfill, probabilities)
    backfill.recover_offline_scores(report, events, receipt, tmp_path, "d" * 64)
    ledger = tmp_path / "predictions.jsonl"
    first = ledger.read_bytes()
    ledger.unlink()
    backfill.recover_offline_scores(report, events, receipt, tmp_path, "d" * 64)
    assert ledger.read_bytes() == first
    assert len(requests) == 1


@pytest.mark.parametrize("mismatch", ["source_report", "model_uri", "sample_ids", "ledger", "cache_result"])
def test_cached_supplement_rejects_changed_provenance_or_content(
    monkeypatch, backfill, offline, tmp_path, mismatch
):
    report, events, probabilities, receipt = offline
    install_fake_inference(monkeypatch, backfill, probabilities)
    backfill.recover_offline_scores(report, events, receipt, tmp_path, "d" * 64)
    source_sha = "d" * 64
    if mismatch == "source_report":
        source_sha = "e" * 64
    elif mismatch == "model_uri":
        receipt["model_uri"] += "-changed"
    elif mismatch == "sample_ids":
        events[2]["event_id"] = "other"
    elif mismatch == "ledger":
        (tmp_path / "predictions.jsonl").write_text("changed")
    else:
        path = tmp_path / "offline-score-recovery.json"
        cached = json.loads(path.read_bytes())
        cached["result"]["probabilities"][0] = 0.9
        path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="differs"):
        backfill.recover_offline_scores(report, events, receipt, tmp_path, source_sha)


def test_unknown_prediction_ledger_is_never_replaced(monkeypatch, backfill, offline, tmp_path):
    report, events, probabilities, receipt = offline
    requests = install_fake_inference(monkeypatch, backfill, probabilities)
    ledger = tmp_path / "predictions.jsonl"
    ledger.write_bytes(b"unknown evidence")
    with pytest.raises(ValueError, match="no authenticated"):
        backfill.recover_offline_scores(report, events, receipt, tmp_path, "d" * 64)
    assert ledger.read_bytes() == b"unknown evidence" and not requests


@pytest.mark.parametrize("change", [{"fresh_process": False}, {"seqs": [3, 2, 4, 5]},
                                    {"event_ids": []}, {"uri": "models:/other/1"},
                                    {"bundled_loader": "/app/src/continual_bench/adapters/model_loader.py"}])
def test_invalid_subprocess_identity_is_rejected_before_writing_supplements(
    monkeypatch, backfill, offline, tmp_path, change
):
    report, events, probabilities, receipt = offline
    install_fake_inference(monkeypatch, backfill, probabilities, change)
    with pytest.raises(ValueError):
        backfill.recover_offline_scores(report, events, receipt, tmp_path, "d" * 64)
    assert not list(tmp_path.iterdir())


def test_wrong_frozen_predictions_never_commit_cache_or_ledger(monkeypatch, backfill, offline, tmp_path):
    report, events, _, receipt = offline
    install_fake_inference(monkeypatch, backfill, [0.9] * 4)
    with pytest.raises(ValueError, match="differ from original"):
        backfill.recover_offline_scores(report, events, receipt, tmp_path, "d" * 64)
    assert not list(tmp_path.iterdir())


def test_prequential_cannot_recover_from_final_model(monkeypatch, backfill, offline, tmp_path):
    report, events, probabilities, receipt = offline
    report["config"]["mode"] = "streaming"
    requests = install_fake_inference(monkeypatch, backfill, probabilities)
    with pytest.raises(ValueError, match="offline"):
        backfill.recover_offline_scores(report, events, receipt, tmp_path, "d" * 64)
    with pytest.raises(ValueError, match="offline"):
        backfill.scored_metrics(report, events, probabilities)
    assert not requests


@pytest.mark.parametrize("change", ["count", "nan", "negative", "window_partition"])
def test_score_and_window_contract_is_fail_closed(backfill, offline, change):
    report, events, probabilities, _ = offline
    if change == "count":
        probabilities.pop()
    elif change == "nan":
        probabilities[0] = float("nan")
    elif change == "negative":
        probabilities[0] = -1
    else:
        report["windows"] = report["windows"][:1]
    with pytest.raises(ValueError):
        backfill.scored_metrics(report, events, probabilities)


@pytest.mark.parametrize("schema_type,dtype", [("double", "float64"), ("long", "int64")])
def test_helper_casts_numeric_schema_and_uses_bundled_inference_only(schema_type, dtype):
    pytest.importorskip("pandas")
    helper = script("recover_offline_scores")
    implementation = SimpleNamespace(__file__="/artifact/code/continual_bench/adapters/river.py")
    loader = SimpleNamespace(__file__="/artifact/code/continual_bench/adapters/model_loader.py")
    schema = [SimpleNamespace(name="value", type=SimpleNamespace(name=schema_type))]
    seen = []

    def predict(frame):
        seen.append(str(frame["value"].dtype))
        return [0.2, 0.8]

    model = SimpleNamespace(_model_impl=SimpleNamespace(model=type("RiverModel", (), {})()),
                            metadata=SimpleNamespace(get_input_schema=lambda: SimpleNamespace(inputs=schema)),
                            predict=predict)
    request = {"uri": "models:/fixture/1", "model": "logistic", "rows": [
        {"seq": 2, "event_id": "a", "features": {"value": 1}},
        {"seq": 3, "event_id": "b", "features": {"value": 2}},
    ]}
    response = helper.recover_loaded(request, model, loader, implementation)
    assert response["probabilities"] == [0.2, 0.8]
    assert response["seqs"] == [2, 3] and response["event_ids"] == ["a", "b"]
    assert seen == [dtype]
    assert response["fresh_process"] is True
    loader.__file__ = "/app/src/continual_bench/adapters/model_loader.py"
    with pytest.raises(ValueError, match="bundled"):
        helper.recover_loaded(request, model, loader, implementation)


def test_helper_main_refuses_current_package_preimport(monkeypatch):
    import io

    helper = script("recover_offline_scores")
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    with pytest.raises(RuntimeError, match="current training package"):
        helper.main()


def test_recovery_helper_import_does_not_load_training_package_in_fresh_process():
    helper_path = Path(__file__).parents[1] / "scripts/recover_offline_scores.py"
    code = (
        "import runpy,sys;runpy.run_path(sys.argv[1],run_name='import_only');"
        "assert not any(k.startswith('continual_bench') for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code, str(helper_path)],
                   check=True, capture_output=True, timeout=30)


def test_cli_recovers_only_offline_when_opted_in_and_replay_preserves_originals(
    monkeypatch, backfill, offline, tmp_path
):
    report, events, probabilities, receipt = offline
    event_raw = b"".join(backfill.json_bytes(event) + b"\n" for event in events)
    report["dataset"]["sha256"] = hashlib.sha256(event_raw).hexdigest()
    prequential = copy.deepcopy(report)
    prequential.pop("split")
    prequential["run_key"] = "prequential-fixture"
    prequential["config"]["mode"] = "streaming"
    prequential["config"]["warmup"] = 2
    prequential["protocol"] = "event_prequential"
    state = tmp_path / "runs"
    records, rows, originals = [], [], {}
    for item in (report, prequential):
        directory = state / item["run_key"]
        directory.mkdir(parents=True)
        raw = backfill.json_bytes(item)
        saved = {"report.json": raw, "events.jsonl": event_raw,
                 "receipt.json": backfill.json_bytes(receipt),
                 "job.json": backfill.json_bytes({"status": "succeeded", "started_at": 1, "finished_at": 5}),
                 "manifest.json": b"original manifest", "checkpoint.bin": b"original checkpoint",
                 "models.bin": b"original models", "training-source.zip": b"original source"}
        for name, blob in saved.items():
            path = directory / name
            path.write_bytes(blob)
            originals[path] = blob, path.stat().st_mtime_ns
        records.append({"report": item, "receipt": receipt, "sha256": hashlib.sha256(raw).hexdigest()})
        rows.append({"run_key": item["run_key"], "config": item["config"]})
    inputs = tmp_path / "inputs.json"
    inputs.write_bytes(backfill.json_bytes({
        "complete_campaign": True, "runs_verified": 2, "records": records,
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(backfill.json_bytes({"source_sha256": report["environment"]["code_sha256"],
                                             "configurations": rows}))
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    output = tmp_path / "backfill.json"
    argv = ["backfill_tracking_metrics.py", "--inputs", str(inputs), "--manifest", str(manifest),
            "--settings", str(profile), "--state", str(state), "--output", str(output), "--workers", "1"]
    requests = install_fake_inference(monkeypatch, backfill, probabilities)
    monkeypatch.setattr(sys, "argv", argv)
    backfill.main()
    assert not requests
    assert json.loads(output.read_bytes())["offline_scores_recovered"] == 0
    monkeypatch.setattr(sys, "argv", argv + ["--recover-offline-scores"])
    backfill.main()
    result = json.loads(output.read_bytes())
    assert result["offline_scores_recovered"] == 1
    assert [row["offline_scores_recovered"] for row in result["records"]] == [True, False]
    assert len(requests) == 1
    assert not (state / prequential["run_key"] / "predictions.jsonl").exists()
    backfill.main()
    assert len(requests) == 1
    assert all(path.read_bytes() == blob and path.stat().st_mtime_ns == stamp
               for path, (blob, stamp) in originals.items())


def test_fresh_helper_loads_a_local_pyfunc_bundled_implementation_without_training(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    package = tmp_path / "bundle" / "continual_bench"
    adapters = package / "adapters"
    adapters.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (adapters / "__init__.py").write_text("")
    (adapters / "river.py").write_text(
        "class RiverModel:\n"
        "    def predict(self, features):\n"
        "        return {2: .8, 3: .2, 4: .7, 5: .1}[int(features['value'])]\n"
        "    def learn(self, features, label):\n"
        "        raise AssertionError('recovery must never train')\n"
    )
    (adapters / "model_loader.py").write_text(
        "from .river import RiverModel\n"
        "import numpy as np\n"
        "class Predictor:\n"
        "    def __init__(self): self.model = RiverModel()\n"
        "    def predict(self, frame, params=None):\n"
        "        return np.array([self.model.predict(row) for row in frame.to_dict(orient='records')])\n"
        "def _load_pyfunc(path): return Predictor()\n"
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "marker").write_text("frozen untrained test fixture")
    destination = tmp_path / "model"
    mlflow.pyfunc.save_model(
        path=str(destination), loader_module="continual_bench.adapters.model_loader", data_path=str(data),
        code_paths=[str(package)], pip_requirements=["numpy", "pandas"],
        signature=ModelSignature(inputs=Schema([ColSpec("double", "value")]),
                                 outputs=Schema([ColSpec("double")])),
    )
    request = {"uri": str(destination), "model": "logistic", "rows": [
        {"seq": seq, "event_id": f"event-{seq}", "features": {"value": seq}} for seq in range(2, 6)
    ]}
    helper = Path(__file__).parents[1] / "scripts/recover_offline_scores.py"
    completed = subprocess.run([sys.executable, str(helper)], input=json.dumps(request), text=True,
                               capture_output=True, timeout=60, check=True)
    response = json.loads(completed.stdout)
    assert response["probabilities"] == [0.8, 0.2, 0.7, 0.1]
    assert Path(response["bundled_loader"]).is_relative_to(destination / "code")
    assert Path(response["bundled_model_code"]).is_relative_to(destination / "code")
