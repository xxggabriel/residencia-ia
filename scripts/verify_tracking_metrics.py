"""Read-only audit of native and historically derived tracking sidecars and MLflow histories."""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from continual_bench.adapters.local import atomic_write
from continual_bench.adapters.metric_tracking import EXECUTION_SCOPE, VERSION, metric_points
from continual_bench.adapters.services import S3Objects
from continual_bench.adapters.versioning import digest_json
from continual_bench.application.runner import environment
from continual_bench.application.tracking import confusion_metrics, metric_catalog
from continual_bench.domain.metrics import Metrics

LEGACY_METRICS = ("n", "accuracy", "f1", "brier", "log_loss")
SCORE_METRICS = ("roc_auc", "average_precision", "ece", "mce", "mae_probability")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_object(objects, key, expected_sha=None, expected_size=None):
    head = objects.client.head_object(Bucket=objects.bucket, Key=key)
    size = head["ContentLength"]
    require(size <= 64 * 1024**2, f"audit refuses a large object: {key}")
    if expected_size is not None:
        require(size == expected_size, f"S3 object length differs: {key}")
    if expected_sha is not None:
        require(
            head.get("Metadata", {}).get("sha256") == expected_sha,
            f"S3 immutable metadata SHA differs: {key}",
        )
    body = objects.client.get_object(Bucket=objects.bucket, Key=key)["Body"]
    try:
        raw = body.read(size + 1)
    finally:
        body.close()
    require(len(raw) == size, f"S3 body length differs: {key}")
    if expected_sha is not None:
        require(sha(raw) == expected_sha, f"S3 body SHA differs: {key}")
    return raw


def check_histories(client, run, tracking):
    """One read per known metric key; reject duplicate steps, unexpected values and false nulls."""
    desired = defaultdict(dict)
    for key, step, value in metric_points(tracking):
        desired[key][step] = value
    known = set(run.data.metrics)
    finite_points = null_points = history_keys = 0
    for key, points in desired.items():
        history = client.get_metric_history(run.info.run_id, key) if key in known else ()
        history_keys += key in known
        actual = {}
        for point in history:
            require(point.step not in actual, f"duplicate remote metric step: {key}/{point.step}")
            require(
                type(point.value) in (int, float) and math.isfinite(point.value),
                f"nonfinite remote metric: {key}/{point.step}",
            )
            actual[point.step] = point.value
        expected = {step: value for step, value in points.items() if value is not None}
        require(actual == expected, f"remote history differs from sidecar: {key}")
        finite_points += len(expected)
        null_points += len(points) - len(expected)
    expected_keys = {key for key, values in desired.items() if any(v is not None for v in values.values())}
    require(
        {key for key in known if key.startswith("tracking_v1/")} == expected_keys,
        "remote tracking metric key set differs from the complete sidecar",
    )
    return {"finite_points": finite_points, "null_points": null_points, "history_keys": history_keys}


def check_predictions(report, tracking, raw, events, native):
    rows = [json.loads(line) for line in raw.splitlines()]
    metadata = tracking["prediction_ledger"]
    require(sha(raw) == metadata["sha256"] and len(rows) == metadata["rows"], "prediction ledger differs")
    selection = report.get("split", {}).get("test", [report["config"]["warmup"], report["observed"]])
    require([row["seq"] for row in rows] == list(range(*selection)), "ledger evaluation suffix differs")
    originals = {event["seq"]: event for event in events}
    total = Metrics()
    for row in rows:
        event = originals[row["seq"]]
        require(
            row["event_id"] == event["event_id"] and row["label"] == bool(event["y"]),
            "ledger sample identity or label differs",
        )
        require(
            row["timestamp"] == event["timestamp"] and row["available_at"] == event["available_at"],
            "ledger source timing differs",
        )
        if native:
            require(row["released_at"] >= row["available_at"], "ledger exposes unavailable label")
        else:
            require(
                "prediction_seconds" not in row and "released_at" not in row,
                "recovered historical ledger fabricates operational timing",
            )
        total.add(row["label"], row["probability"])

    def check(values, original, tracked):
        for key, value in values.items():
            if key in original:
                equal = (
                    value == original[key]
                    if native or key == "n"
                    else math.isclose(value, original[key], rel_tol=0, abs_tol=1e-12)
                )
                require(equal, f"ledger recomputation differs from original {key}")
            expected = original[key] if not native and key in LEGACY_METRICS else value
            require(tracked[key] == expected, f"published recomputation differs for {key}")

    check(total.values(), report["metrics"], tracking["metrics"])
    # Python 3.12 sum() and sequential += can differ by an ulp. The independent
    # recomputation uses accurate summation; remote history/sidecar checks stay exact.
    mae = math.fsum(abs(int(row["label"]) - row["probability"]) for row in rows) / len(rows)
    require(math.isclose(tracking["metrics"]["mae_probability"], mae, rel_tol=0, abs_tol=1e-15),
            "published probability MAE differs")
    by_seq = {row["seq"]: row for row in rows}
    covered = []
    for window in report["windows"]:
        accumulator = Metrics()
        for seq in range(window["end_seq"] - window["n"] + 1, window["end_seq"] + 1):
            row = by_seq[seq]
            accumulator.add(row["label"], row["probability"])
            covered.append(seq)
        tracked = {}
        for key in accumulator.values():
            points = {point["step"]: point["value"] for point in tracking["series"][key]}
            tracked[key] = points[window["end_seq"]]
        check(accumulator.values(), window, tracked)
    require(covered == [row["seq"] for row in rows], "windows do not partition prediction ledger")
    return len(rows), len(total.values())


def check_references(report, tracking, files):
    references = tracking.get("reference_models", [])
    if not references:
        return 0
    probes = json.loads(files["continual-probes.json"])
    require(sha(files["continual-probes.json"]) == tracking["probe_dataset"]["sha256"], "probe SHA differs")
    require(
        probes["dataset_sha256"] == report["dataset"]["sha256"]
        and probes["code_sha256"] == report["environment"]["code_sha256"],
        "probe lineage differs",
    )
    ends = tracking["continual"]["contract"]["task_ends"]
    require(probes["task_ends"] == ends, "probe episode boundaries differ")
    require(
        [episode["episode"] for episode in probes["episodes"]] == list(range(len(ends))),
        "probe episode identities differ",
    )
    identities = set()
    for reference in references:
        identity = reference["name"], reference["task"]
        require(identity not in identities, "duplicate reference model")
        identities.add(identity)
        task = reference["task"]
        start = ends[task - 1] if task and reference["name"] == "task_only" else 0
        selection = [start, ends[task]]
        require(
            reference["seq_range"] == selection and reference["train_samples"] == selection[1] - start,
            "reference training prefix or episode differs",
        )
        prefix = f"references/{reference['name']}-{task}/"
        metadata = json.loads(files[prefix + "model-metadata.json"])
        index_raw = files[prefix + "model-versions.json"]
        index = json.loads(index_raw)
        model, journal = files[prefix + "model.pkl"], files[prefix + "models.bin"]
        require(metadata == reference["metadata"], "reference metadata differs from sidecar")
        for record in (metadata, index):
            require(
                record["dataset_version"] == report["dataset"]["sha256"]
                and record["code_sha256"] == report["environment"]["code_sha256"],
                "reference code or data lineage differs",
            )
        require(metadata["count"] == len(index["versions"]) == 1, "reference has unexpected versions")
        require(
            sha(index_raw) == metadata["index_sha256"]
            and sha(journal) == metadata["journal_sha256"]
            and sha(model) == metadata["state_sha256"],
            "reference content SHA differs",
        )
        entry = index["versions"][0]
        require(
            entry["ordinal"] == 0 and entry["parent_version"] is None and entry["offset"] == 0,
            "reference version chain differs",
        )
        require(entry["size"] == len(journal) and journal == model, "reference state and journal differ")
        require(
            entry["state_sha256"] == metadata["state_sha256"]
            and entry["dataset_version"] == report["dataset"]["sha256"]
            and entry["code_sha256"] == report["environment"]["code_sha256"],
            "reference state lineage differs",
        )
        require(
            entry["training"]["seq_range"] == selection
            and entry["training"]["action"] == "reference_initial_fit",
            "reference training view differs",
        )
        require(
            entry["training_view_version"]
            == digest_json(
                {
                    "dataset_version": report["dataset"]["sha256"],
                    "seq_range": selection,
                }
            ),
            "reference training view hash differs",
        )
        require(
            entry["version_id"]
            == metadata["final_version_id"]
            == digest_json({key: value for key, value in entry.items() if key != "version_id"}),
            "reference version ID differs",
        )
    require(
        identities
        == {(name, task) for name in ("task_only", "joint_prefix_one_pass") for task in range(len(ends))},
        "reference model coverage differs",
    )
    return len(references)


def check_historical_classification(report, tracking, events):
    labels = {event["seq"]: bool(event["y"]) for event in events}
    selection = report.get("split", {}).get("test", [report["config"]["warmup"], report["observed"]])
    expected = confusion_metrics(report["metrics"], sum(labels[seq] for seq in range(*selection)))
    for name, value in expected.items():
        if name not in SCORE_METRICS:
            require(tracking["metrics"][name] == value, f"historical classification differs: {name}")
    for window in report["windows"]:
        sequences = range(window["end_seq"] - window["n"] + 1, window["end_seq"] + 1)
        expected = confusion_metrics(window, sum(labels[seq] for seq in sequences))
        for name, value in expected.items():
            if name != "end_seq" and name not in SCORE_METRICS:
                actual = {point["step"]: point["value"] for point in tracking["series"][name]}
                require(
                    actual[window["end_seq"]] == value, f"historical window classification differs: {name}"
                )
    retention = {}
    for point in report.get("retention_probes", []):
        retention[point["concept"]] = point
    for name, field in (
        ("retention_final_mean_accuracy", "accuracy"),
        ("retention_final_mean_drop", "drop_from_best_probe"),
    ):
        value = sum(point[field] for point in retention.values()) / len(retention) if retention else None
        require(tracking["metrics"][name] == value, f"historical sampled retention differs: {name}")


class Auditor:
    def __init__(self, client, objects, state, source, definitions, definitions_raw):
        self.client = client
        self.objects = objects
        self.state = Path(state)
        self.source = source
        self.definitions = definitions
        self.definitions_sha = sha(definitions_raw)

    def verify(self, request):
        try:
            return self._verify(request)
        except Exception as error:
            raise RuntimeError(f"tracking audit failed for {request['run_key']}: {error}") from error

    def _verify(self, request):
        key, native = request["run_key"], request["native"]
        directory = self.state / key
        raw = (directory / "report.json").read_bytes()
        report = json.loads(raw)
        receipt = json.loads((directory / "receipt.json").read_bytes())
        job = json.loads((directory / "job.json").read_bytes())
        require(
            job["status"] == "succeeded" and report["complete"] and report["run_key"] == key,
            "benchmark job or report is incomplete",
        )
        require(report["config"] == request["config"], "frozen benchmark configuration differs")
        require(report["environment"]["code_sha256"] == request["source"], "training source differs")
        if not native:
            require(
                sha(raw) == request["report_sha256"] and report == request["report"],
                "historical original report changed",
            )
            require(receipt == request["receipt"], "historical receipt changed")
        require(read_object(self.objects, f"{key}/report.json") == raw, "original S3 report differs")
        run = self.client.get_run(receipt["mlflow_run_id"])
        tags = run.data.tags
        require(
            run.info.status == "FINISHED" and tags["run_key"] == key, "MLflow run identity/status differs"
        )
        require(
            run.data.params["code_sha256"] == request["source"]
            and run.data.params["dataset_version"] == report["dataset"]["sha256"],
            "MLflow lineage differs",
        )
        require(
            tags["tracking_v1_status"] == "complete"
            and tags["tracking_v1_origin"] == ("native" if native else "derived"),
            "tracking is incomplete",
        )
        payload_sha = tags["tracking_v1_payload_sha256"]
        require(len(payload_sha) == 64, "invalid sidecar payload SHA")
        if not native:
            require(
                payload_sha == request["published_payload_sha256"], "backfill receipt payload SHA differs"
            )
        prefix = f"tracking/{VERSION}/{payload_sha}"
        payload = read_object(self.objects, prefix + "/payload.json", payload_sha)
        envelope = json.loads(payload)
        tracking = envelope["tracking"]
        require(envelope["version"] == tracking["version"] == VERSION, "tracking version differs")
        require(envelope["derived"] is not native, "sidecar derivation origin differs")
        require(
            envelope["report_sha256"] == sha(raw)
            and envelope["training_code_sha256"] == request["source"]
            and envelope["evaluation_code_sha256"] == self.source
            and envelope["dataset_sha256"] == report["dataset"]["sha256"],
            "sidecar lineage differs",
        )
        require(
            envelope["settings_sha256"]
            == tags["tracking_v1_settings_sha256"]
            == sha(encoded(tracking["settings"])),
            "tracking settings SHA differs",
        )
        require(
            tracking["definitions_sha256"] == tags["tracking_v1_definitions_sha256"] == self.definitions_sha,
            "metric definitions SHA differs",
        )
        require(
            int(tags["tracking_v1_unavailable_count"])
            == sum(value is None for value in tracking["metrics"].values()),
            "unavailability tag differs",
        )
        for name, value in tracking["metrics"].items():
            availability = tracking["availability"][name]
            available = availability.get("status") == "available" or availability.get("available") is True
            require(available == (value is not None), f"metric availability contradicts its value: {name}")
            if value is None:
                require(bool(availability.get("reason")), f"unavailable metric has no reason: {name}")
        require(
            set(self.definitions) <= (set(tracking["metrics"]) | set(tracking["series"])),
            "tracking omits catalog definitions",
        )
        pipeline = tracking["metrics"]["pipeline_elapsed_seconds"]
        require(
            type(pipeline) in (int, float) and math.isfinite(pipeline) and pipeline >= 0,
            "final pipeline elapsed time is not measured",
        )
        families = Counter()
        for name, definition in self.definitions.items():
            value = tracking["metrics"].get(name)
            state = (
                "series"
                if name in tracking["series"]
                else "available"
                if value is not None
                else "unavailable"
            )
            families[f"{definition['family']}:{state}"] += 1
        histories = check_histories(self.client, run, tracking)
        files, hashes = {}, []
        for item in envelope["files"]:
            path = PurePosixPath(item["path"])
            require(
                not path.is_absolute() and ".." not in path.parts and item["path"] not in files,
                "unsafe or duplicate sidecar file path",
            )
            require(
                item["path"]
                in {
                    "predictions.jsonl",
                    "metric-definitions.json",
                    "continual-probes.json",
                    "tracking/execution-v1.json",
                }
                or path.parts[0] == "references",
                "audit refuses a non-supplemental file or historical model journal",
            )
            blob = read_object(self.objects, prefix + "/" + item["path"], item["sha256"], item["bytes"])
            files[item["path"]] = blob
            hashes.append({"run_key": key, **item})
        require(sha(files["metric-definitions.json"]) == self.definitions_sha, "published definitions differ")
        event_raw = (directory / "events.jsonl").read_bytes()
        require(sha(event_raw) == report["dataset"]["sha256"], "source event checksum differs")
        events = [json.loads(line) for line in event_raw.splitlines()]
        if native:
            original_tracking = copy.deepcopy(report["tracking"])
            final_tracking = copy.deepcopy(tracking)
            for field in ("metrics", "availability"):
                original_tracking[field].pop("pipeline_elapsed_seconds", None)
                final_tracking[field].pop("pipeline_elapsed_seconds", None)
            final_tracking.pop("execution_scope", None)
            require(
                original_tracking == final_tracking, "native sidecar changed more than final execution timing"
            )
            execution = json.loads(files["tracking/execution-v1.json"])
            require(
                execution["source_report_sha256"] == sha(raw)
                and execution["run_id"] == run.info.run_id
                and execution["scope"] == EXECUTION_SCOPE
                and execution["elapsed_seconds"] == pipeline,
                "final execution evidence differs",
            )
        else:
            require(
                tracking["derivation"]["original_report_preserved"] is True, "original preservation missing"
            )
            require(
                tracking["derivation"]["tool_script_sha256"]
                == sha(Path(__file__).with_name("backfill_tracking_metrics.py").read_bytes()),
                "derivation tool SHA differs",
            )
            for name in (
                "cl_average_accuracy",
                "cl_average_incremental_accuracy",
                "cl_forgetting",
                "cl_backward_transfer",
                "cl_intransigence",
                "cl_forward_transfer",
                "prediction_seconds",
                "initial_train_seconds",
                "train_seconds",
            ):
                require(
                    tracking["metrics"][name] is None, f"historical unavailable metric was fabricated: {name}"
                )
        if not native:
            check_historical_classification(report, tracking, events)
        ledger_rows = metric_count = 0
        offline = report["config"]["mode"] == "offline"
        if native or offline:
            ledger_rows, metric_count = check_predictions(
                report, tracking, files["predictions.jsonl"], events, native
            )
        else:
            require(
                "predictions.jsonl" not in files
                and all(tracking["metrics"][name] is None for name in SCORE_METRICS),
                "prequential historical scores were fabricated",
            )
        if not native and offline:
            recovery = tracking["derivation"]["offline_score_recovery"]
            cache_raw = (directory / "offline-score-recovery.json").read_bytes()
            cache = json.loads(cache_raw)
            require(
                recovery["origin"]
                == tracking["prediction_ledger"]["origin"]
                == "recovered_frozen_offline_registry_inference",
                "offline recovery origin differs",
            )
            require(
                recovery["source_report_sha256"] == sha(raw)
                and recovery["model_uri"] == receipt["model_uri"]
                and recovery["cache_sha256"] == sha(cache_raw),
                "offline cache provenance differs",
            )
            require(
                cache["source_report_sha256"] == sha(raw)
                and cache["model_uri"] == receipt["model_uri"]
                and cache["result_sha256"] == sha(encoded(cache["result"])),
                "offline cache result differs",
            )
            require(
                recovery["helper_sha256"]
                == tracking["derivation"]["recover_helper_sha256"]
                == sha(Path(__file__).with_name("recover_offline_scores.py").read_bytes()),
                "recovery helper SHA differs",
            )
        references = check_references(report, tracking, files) if native else 0
        reloaded = None
        if native:
            version = self.client.get_model_version(
                receipt["registered_model"], receipt["registered_model_version"]
            )
            require(
                version.status == "READY"
                and version.run_id == run.info.run_id
                and version.tags["dataset_version"] == report["dataset"]["sha256"]
                and version.tags["model_sha256"] == report["model_versions"]["state_sha256"]
                and version.tags["code_sha256"] == request["source"]
                and version.tags["model"] == report["config"]["model"]["name"],
                "main model Registry lineage differs",
            )
            probe = {
                "uri": receipt["model_uri"],
                "probe": report["model_versions"]["inference_probe"],
                "model": report["config"]["model"]["name"],
                "dataset": report["dataset"]["name"],
            }
            process = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("registry_inference_probe.py"))],
                input=json.dumps(probe),
                text=True,
                capture_output=True,
                timeout=180,
                check=True,
            )
            reloaded = json.loads(process.stdout)
            require(
                reloaded["fresh_process"] and reloaded["prediction_identical"], "Registry inference differs"
            )
        return {
            "run_key": key,
            "origin": "native" if native else "derived",
            "source_sha256": request["source"],
            "report_sha256": sha(raw),
            "payload_sha256": payload_sha,
            **histories,
            "ledger_rows": ledger_rows,
            "recomputed_metric_fields": metric_count,
            "offline_scores_recovered": offline and not native,
            "sampled_retention": bool(report.get("retention_probes")),
            "references": references,
            "registry_reload": reloaded,
            "families": dict(families),
            "file_hashes": hashes,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/state/tracking-v1-validation-manifest.json")
    parser.add_argument(
        "--backfill", help="Completed, published backfill result; omit to audit only native runs"
    )
    parser.add_argument("--inputs", default="/state/nonlinear-v5-inputs.json")
    parser.add_argument("--output", default="/state/tracking-v1-verification.json")
    parser.add_argument("--state", default="/state/runs")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    manifest_raw = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_raw)
    definitions, definitions_raw = metric_catalog()
    require(len(definitions["definitions"]) == 177, "metric definition inventory is not the frozen 177")
    require(sha(definitions_raw) == manifest["definitions_sha256"], "manifest metric definitions differ")
    source = manifest["source_sha256"]
    require(environment()["code_sha256"] == source, "auditor source differs from deployed evaluation source")
    native = [
        {**row, "native": True, "source": row.get("source_sha256", source)}
        for row in manifest["configurations"]
    ]
    require(len(native) == manifest["expected_runs"] == 8, "native validation requires all eight runs")
    require({row["source"] for row in native} == {source}, "native runs use inconsistent deployed sources")
    requests = list(native)
    input_sha = backfill_sha = None
    if args.backfill:
        input_raw = Path(args.inputs).read_bytes()
        inputs = json.loads(input_raw)
        backfill_raw = Path(args.backfill).read_bytes()
        backfill = json.loads(backfill_raw)
        input_sha, backfill_sha = sha(input_raw), sha(backfill_raw)
        require(
            inputs["complete_campaign"] and inputs["runs_verified"] == len(inputs["records"]) == 867,
            "historical input campaign is incomplete",
        )
        require(
            backfill["published"]
            and backfill["complete_campaign"]
            and backfill["runs"] == 867
            and backfill["inputs_sha256"] == input_sha
            and backfill["offline_scores_recovered"] == 51,
            "historical publication or offline recovery is incomplete",
        )
        by_key = {row["run_key"]: row for row in backfill["records"]}
        require(len(by_key) == 867, "backfill run IDs are duplicated")
        sources = Counter()
        for record in inputs["records"]:
            report = record["report"]
            key = report["run_key"]
            require(
                by_key[key]["originals_preserved"]
                and by_key[key]["source_report_sha256"] == record["sha256"],
                "backfill original evidence checksum differs",
            )
            sources[report["environment"]["code_sha256"]] += 1
            requests.append(
                {
                    "run_key": key,
                    "native": False,
                    "config": report["config"],
                    "source": report["environment"]["code_sha256"],
                    "report": report,
                    "report_sha256": record["sha256"],
                    "receipt": record["receipt"],
                    "published_payload_sha256": by_key[key]["publication"]["payload_sha256"],
                }
            )
        require(
            dict(sources) == inputs["source_counts"] == backfill["source_counts"]
            and sorted(sources.values()) == [2, 865],
            "historical per-run source revisions differ",
        )
    require(len({row["run_key"] for row in requests}) == len(requests), "audit run IDs overlap or duplicate")
    from mlflow.tracking import MlflowClient

    auditor = Auditor(
        MlflowClient(), S3Objects(), args.state, source, definitions["definitions"], definitions_raw
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(auditor.verify, requests), 1):
            results.append(result)
            if index % 25 == 0:
                print(json.dumps({"verified": index, "total": len(requests)}), flush=True)
    counts = Counter(row["origin"] for row in results)
    require(sum(row["references"] for row in results) == 18, "native reference coverage differs from 18")
    if args.backfill:
        derived = [row for row in results if row["origin"] == "derived"]
        require(
            sum(row["offline_scores_recovered"] for row in derived) == 51
            and sum(row["sampled_retention"] for row in derived) == 720,
            "historical score or retention coverage differs",
        )
    families = Counter()
    file_hashes = []
    for row in results:
        families.update(row["families"])
        file_hashes.extend(row.pop("file_hashes"))
    result = {
        "version": VERSION,
        "complete": True,
        "runs_verified": len(results),
        "origins": dict(counts),
        "evaluation_code_sha256": source,
        "definitions_sha256": sha(definitions_raw),
        "definition_count": 177,
        "manifest_sha256": sha(manifest_raw),
        "inputs_sha256": input_sha,
        "backfill_sha256": backfill_sha,
        "auditor_script_sha256": sha(Path(__file__).read_bytes()),
        "source_counts": dict(Counter(row["source_sha256"] for row in results)),
        "finite_metric_points": sum(row["finite_points"] for row in results),
        "unavailable_metric_points": sum(row["null_points"] for row in results),
        "history_keys_read": sum(row["history_keys"] for row in results),
        "references_verified": sum(row["references"] for row in results),
        "registry_reloads": sum(row["registry_reload"] is not None for row in results),
        "prediction_rows_recomputed": sum(row["ledger_rows"] for row in results),
        "derived_offline_runs_recomputed": sum(row["offline_scores_recovered"] for row in results),
        "derived_prequential_score_runs_unavailable": counts["derived"]
        - sum(row["offline_scores_recovered"] for row in results),
        "definition_family_availability": dict(families),
        "supplemental_file_hashes_verified": len(file_hashes),
        "supplemental_payload_bytes_verified": sum(item["bytes"] for item in file_hashes),
        "file_hash_manifest_sha256": sha(
            encoded(sorted(file_hashes, key=lambda item: (item["run_key"], item["path"])))
        ),
        "historical_model_journals_read": 0,
        "records": results,
    }
    atomic_write(Path(args.output), json.dumps(result, indent=2).encode())
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
