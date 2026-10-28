"""Enrich existing runs without training, replacing reports, or registering models."""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from continual_bench.adapters.local import atomic_write, run_lock
from continual_bench.application.config import TrackingConfig
from continual_bench.application.tracking import build_tracking, confusion_metrics, metric_catalog
from continual_bench.domain.metrics import Metrics

LEGACY_METRICS = ("n", "accuracy", "f1", "brier", "log_loss")
RECOVERY_ORIGIN = "recovered_frozen_offline_registry_inference"


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def selected_events(report, events):
    by_seq = {event["seq"]: event for event in events}
    if len(by_seq) != len(events):
        raise ValueError("recovery source has duplicate sequences")
    selection = report.get("split", {}).get("test", [report["config"]["warmup"], report["observed"]])
    return [by_seq[seq] for seq in range(*selection)]


def scored_metrics(report, events, probabilities):
    if report["config"]["mode"] != "offline":
        raise ValueError("Frozen final-model score recovery is valid only for offline holdouts")
    selected = selected_events(report, events)
    if len(probabilities) != len(selected):
        raise ValueError("recovered score count differs from the exact test suffix")
    global_metrics = Metrics()
    by_seq = {}
    for event, probability in zip(selected, probabilities):
        global_metrics.add(bool(event["y"]), probability)
        by_seq[event["seq"]] = (bool(event["y"]), probability)

    def checked(accumulator, original):
        values = accumulator.values()
        for key in LEGACY_METRICS:
            equal = values[key] == original[key] if key == "n" else math.isclose(
                values[key], original[key], rel_tol=0, abs_tol=1e-12
            )
            if not equal:
                raise ValueError(f"recovered offline scores differ from original {key}")
            values[key] = original[key]
        return values

    aggregate = checked(global_metrics, report["metrics"])
    windows, covered = [], []
    for window in report["windows"]:
        accumulator = Metrics()
        sequences = range(window["end_seq"] - window["n"] + 1, window["end_seq"] + 1)
        for seq in sequences:
            if seq not in by_seq:
                raise ValueError("historical metric window is outside the test suffix")
            label, probability = by_seq[seq]
            accumulator.add(label, probability)
            covered.append(seq)
        windows.append({"end_seq": window["end_seq"], **checked(accumulator, window)})
    if covered != [event["seq"] for event in selected]:
        raise ValueError("historical metric windows do not partition the exact test suffix")
    predictions = [{"event_id": event["event_id"], "seq": event["seq"], "label": bool(event["y"]),
                    "probability": probability, "timestamp": event["timestamp"],
                    "available_at": event["available_at"]}
                   for event, probability in zip(selected, probabilities)]
    return aggregate, windows, predictions


def recover_offline_scores(report, events, receipt, directory, source_report_sha256):
    """Create or verify supplemental inference evidence; never retrain or replace originals."""
    if report["config"]["mode"] != "offline":
        raise ValueError("offline score recovery cannot be applied to a prequential run")
    directory = Path(directory)
    helper = Path(__file__).with_name("recover_offline_scores.py")
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    selected = selected_events(report, events)
    request = {"uri": receipt["model_uri"], "model": report["config"]["model"]["name"],
               "rows": [{"seq": event["seq"], "event_id": event["event_id"], "features": event["x"]}
                        for event in selected]}
    provenance = {
        "version": "offline-frozen-score-recovery-v1",
        "source_report_sha256": source_report_sha256,
        "dataset_sha256": report["dataset"]["sha256"],
        "model_uri": receipt["model_uri"],
        "model_state_sha256": report["model_versions"]["state_sha256"],
        "helper_sha256": helper_sha,
        "request_sha256": hashlib.sha256(json_bytes(request)).hexdigest(),
    }
    cache_path = directory / "offline-score-recovery.json"
    ledger_path = directory / "predictions.jsonl"
    if cache_path.exists():
        cache_raw = cache_path.read_bytes()
        cached = json.loads(cache_raw)
        if any(cached.get(key) != value for key, value in provenance.items()):
            raise ValueError("offline recovery cache provenance differs")
        recovered = cached["result"]
        if cached.get("result_sha256") != hashlib.sha256(json_bytes(recovered)).hexdigest():
            raise ValueError("offline recovery cache result checksum differs")
    else:
        if ledger_path.exists():
            raise ValueError("existing predictions have no authenticated offline recovery cache")
        completed = subprocess.run(
            [sys.executable, str(helper)], input=json_bytes(request).decode(), text=True,
            capture_output=True, timeout=300, check=True,
        )
        recovered = json.loads(completed.stdout)
        cache_raw = json_bytes({**provenance, "result": recovered,
                                "result_sha256": hashlib.sha256(json_bytes(recovered)).hexdigest()})
    if (recovered.get("uri") != request["uri"] or recovered.get("model") != request["model"]
            or recovered.get("fresh_process") is not True
            or recovered.get("seqs") != [event["seq"] for event in selected]
            or recovered.get("event_ids") != [event["event_id"] for event in selected]):
        raise ValueError("offline recovery response identity or selection differs")
    for field in ("bundled_loader", "bundled_model_code"):
        if Path(recovered[field]).parents[2].name != "code":
            raise ValueError("offline recovery did not use bundled inference code")
    _, _, predictions = scored_metrics(report, events, recovered["probabilities"])
    ledger = b"".join(json_bytes(row) + b"\n" for row in predictions)
    if ledger_path.exists() and ledger_path.read_bytes() != ledger:
        raise ValueError("existing supplemental prediction ledger differs")
    # Cache first: an interrupted ledger write can resume without another Registry inference.
    if not cache_path.exists():
        atomic_write(cache_path, cache_raw)
    if not ledger_path.exists():
        atomic_write(ledger_path, ledger)
    return recovered["probabilities"], {
        **provenance,
        "cache_sha256": hashlib.sha256(cache_raw).hexdigest(),
        "prediction_ledger_sha256": hashlib.sha256(ledger).hexdigest(),
        "rows": len(predictions),
        "origin": RECOVERY_ORIGIN,
        "bundled_loader": recovered["bundled_loader"],
        "bundled_model_code": recovered["bundled_model_code"],
    }


def enriched_report(
    report, events, settings, artifact_sizes=None, job=None, probabilities=None, recovery=None
):
    enriched = copy.deepcopy(report)
    labels = {e["seq"]: bool(e["y"]) for e in events}
    selection = report.get("split", {}).get("test", [report["config"]["warmup"], report["observed"]])
    positives = sum(labels[i] for i in range(*selection))
    enriched["metrics"] = confusion_metrics(report["metrics"], positives)
    for window in enriched["windows"]:
        end = window["end_seq"]
        values = confusion_metrics(window, sum(labels[i] for i in range(end - window["n"] + 1, end + 1)))
        values["end_seq"] = end
        window.update(values)
    predictions = None
    if probabilities is not None:
        enriched["metrics"], enriched["windows"], predictions = scored_metrics(report, events, probabilities)
    tracking = build_tracking(enriched, settings, predictions=predictions)
    tracking["derivation"] = {
        "kind": "posthoc_from_frozen_report_and_labels",
        "confusion": "integer counts reconstructed from exact accuracy/F1 and evaluation label supports",
        "retention": "sampled concept probes, not boundary task matrix",
        "unavailable": "shadows, reference models and individual timers were not captured; "
                       "ranking scores unavailable unless explicitly recovered from frozen offline models",
        "original_report_preserved": True,
        "tool_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if recovery is not None:
        tracking["derivation"]["offline_score_recovery"] = recovery
        tracking["derivation"]["recover_helper_sha256"] = recovery["helper_sha256"]
        tracking["prediction_ledger"] = {
            "sha256": recovery["prediction_ledger_sha256"], "rows": recovery["rows"],
            "origin": RECOVERY_ORIGIN,
        }
    for key, value in (artifact_sizes or {}).items():
        tracking["metrics"][key] = value
        tracking["availability"][key] = {
            "status": "available",
            "reason": None,
            "scope": "audited_payload_bytes",
        }
    if job and job.get("started_at") is not None and job.get("finished_at") is not None:
        elapsed = job["finished_at"] - job["started_at"]
        if elapsed < 0:
            raise ValueError("negative worker elapsed time")
        tracking["metrics"]["pipeline_elapsed_seconds"] = elapsed
        tracking["availability"]["pipeline_elapsed_seconds"] = {
            "status": "available",
            "reason": None,
            "scope": "successful_worker_attempt_including_publication",
        }
    enriched["tracking"] = tracking
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="/state/nonlinear-v5-inputs.json")
    parser.add_argument("--manifest", default="/app/configs/campaign-nonlinear-v5.json")
    parser.add_argument("--state", default="/state/runs")
    parser.add_argument("--settings", default="/app/configs/tracking-metrics-profile.json")
    parser.add_argument("--output", default="/state/tracking-v1-backfill.json")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--recover-offline-scores", action="store_true",
                        help="Infer only the exact test suffix of frozen offline Registry models")
    args = parser.parse_args()
    source = Path(args.inputs).read_bytes()
    inputs = json.loads(source)
    manifest = json.loads(Path(args.manifest).read_text())
    assert inputs["complete_campaign"] and inputs["runs_verified"] == len(inputs["records"])
    assert len(manifest["configurations"]) == len(inputs["records"])
    expected = {r["run_key"]: r for r in manifest["configurations"]}
    settings = TrackingConfig.model_validate_json(Path(args.settings).read_text()).model_dump(mode="json")
    _, definitions = metric_catalog()
    records = inputs["records"][: args.limit] if args.limit else inputs["records"]
    client = objects = None
    if args.publish:
        from mlflow.tracking import MlflowClient

        from continual_bench.adapters.services import S3Objects
        from continual_bench.application.readiness import ensure_tracking_ready

        ensure_tracking_ready()
        client, objects = MlflowClient(), S3Objects()

    def process(record):
        key = record["report"]["run_key"]
        directory = Path(args.state) / key
        with run_lock(args.state, key):
            originals = [
                "report.json",
                "manifest.json",
                "receipt.json",
                "checkpoint.bin",
                "models.bin",
                "model-versions.json",
                "model-metadata.json",
                "training-source.zip",
                "events.jsonl",
                "job.json",
            ]
            before = {name: (directory / name).stat() for name in originals if (directory / name).exists()}
            raw = (directory / "report.json").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == record["sha256"]
            report = json.loads(raw)
            assert report == record["report"] and report["config"] == expected[key]["config"]
            assert report["environment"]["code_sha256"] == expected[key].get(
                "source_sha256", manifest["source_sha256"]
            )
            event_bytes = (directory / "events.jsonl").read_bytes()
            assert hashlib.sha256(event_bytes).hexdigest() == report["dataset"]["sha256"]
            events = [json.loads(line) for line in event_bytes.splitlines()]
            job = json.loads((directory / "job.json").read_text())
            assert job["status"] == "succeeded"
            probabilities = recovery = None
            if args.recover_offline_scores and report["config"]["mode"] == "offline":
                assert json.loads((directory / "receipt.json").read_text()) == record["receipt"]
                probabilities, recovery = recover_offline_scores(
                    report, events, record["receipt"], directory, record["sha256"]
                )
            enriched = enriched_report(report, events, settings, record.get("artifact_sizes"), job,
                                       probabilities, recovery)
            result = {
                "run_key": key,
                "mlflow_run_id": record["receipt"]["mlflow_run_id"],
                "training_code_sha256": report["environment"]["code_sha256"],
                "source_report_sha256": record["sha256"],
                "offline_scores_recovered": recovery is not None,
                "available": sum(
                    v["status"] == "available" for v in enriched["tracking"]["availability"].values()
                ),
                "unavailable": sum(
                    v["status"] != "available" for v in enriched["tracking"]["availability"].values()
                ),
            }
            if args.publish:
                from continual_bench.adapters.metric_tracking import publish_tracking

                tracked = client.get_run(result["mlflow_run_id"])
                assert tracked.data.tags["run_key"] == key and tracked.info.status == "FINISHED"
                assert tracked.data.params["code_sha256"] == report["environment"]["code_sha256"]
                path = directory / "metric-definitions.json"
                if path.exists() and path.read_bytes() != definitions:
                    raise ValueError("existing metric definitions differ; use a new tracking version")
                atomic_write(path, definitions)
                result["publication"] = publish_tracking(
                    client, result["mlflow_run_id"], enriched, directory, objects=objects, derived=True
                )
            for name, stat in before.items():
                after = (directory / name).stat()
                assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ino,
                ), f"original modified: {key}/{name}"
            assert (directory / "report.json").read_bytes() == raw
            result["originals_preserved"] = True
            return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(process, records), 1):
            results.append(result)
            if index % 25 == 0:
                print(json.dumps({"processed": index, "total": len(records)}), flush=True)
    result = {
        "version": "residency-metrics-v1",
        "published": args.publish,
        "recover_offline_scores": args.recover_offline_scores,
        "offline_scores_recovered": sum(r["offline_scores_recovered"] for r in results),
        "complete_campaign": len(results) == len(inputs["records"]),
        "runs": len(results),
        "inputs_sha256": hashlib.sha256(source).hexdigest(),
        "source_counts": dict(Counter(r["training_code_sha256"] for r in results)),
        "settings": settings,
        "records": results,
    }
    atomic_write(Path(args.output), json.dumps(result, indent=2).encode())
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
