"""Export completed AWS suite evidence with one prediction case in local scratch.

Only S3 HEAD, GET and LIST operations are available to this command. Source data,
predictions and published objects remain in S3. Full report bytes are retained in
lossless gzip shards; predictions are downloaded temporarily for reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "aws-ordered-dataset-evidence-v1"
PROTOCOL = "ordered-stream-batch-prequential-metrics-v2"
DATASET_ROWS = {
    "electricity": 45_312,
    "airlines": 539_383,
    "insects_abrupt_imbalanced": 355_275,
    "insects_incremental_abrupt_reoccurring_imbalanced": 452_044,
}
POLICIES = (
    "frozen",
    "drift",
    "drift_adwin",
    "drift_ddm",
    "incremental",
    "periodic",
    "cara",
    "cara_cumulative",
    "cara_periodic",
    "upf",
    "oracle",
)
SEEDS = (42, 43, 44)
CACHED_POLICIES = ("cara_cumulative", "cara_periodic", "upf", "oracle")
STAGE_POLICIES = {
    "base": tuple(policy for policy in POLICIES if policy not in CACHED_POLICIES),
    "cached": CACHED_POLICIES,
}
FORBIDDEN = re.compile(
    r"(?:^|/)(?:models|checkpoints|candidates|\.zia-resume)(?:/|$)"
    r"|\.(?:pkl|pickle|ubj|npy|npz|pt|pth|ckpt|bin)$|/(?:model|checkpoint)\.tar\.gz$"
)
RAW_NAMES = {"events.jsonl", "samples.jsonl", "continual-probes.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_verifier():
    path = PROJECT / "scripts/verify_dataset_suite.py"
    spec = importlib.util.spec_from_file_location("ordered_suite_evidence_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def s3_parts(uri):
    parsed = urlsplit(uri)
    key = parsed.path.lstrip("/")
    require(
        parsed.scheme == "s3" and parsed.netloc and key and not parsed.query and not parsed.fragment,
        "expected an S3 object URI without a query or fragment",
    )
    require(all(p not in ("", ".", "..") for p in key.rstrip("/").split("/")), "invalid S3 object key")
    return parsed.netloc, key


class ReadOnlyS3:
    """Small CLI adapter with an explicit read-only operation allowlist."""

    def __init__(self, region, *, runner=subprocess.run, timeout=1800):
        self.region, self.runner, self.timeout = region, runner, timeout

    def _call(self, operation, params, destination=None):
        require(operation in {"head-object", "get-object", "list-objects-v2"}, "S3 mutation is not allowed")
        command = [
            "aws",
            "s3api",
            operation,
            "--region",
            self.region,
            "--output",
            "json",
            "--no-paginate",
            "--cli-input-json",
            canonical(params).decode(),
        ]
        if destination is not None:
            command.append(str(destination))
        result = self.runner(command, capture_output=True, text=True, timeout=self.timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"read-only S3 {operation} failed with exit code {result.returncode}")
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def head(self, uri):
        bucket, key = s3_parts(uri)
        return self._call("head-object", {"Bucket": bucket, "Key": key})

    def download(self, uri, destination, head):
        bucket, key = s3_parts(uri)
        params = {"Bucket": bucket, "Key": key}
        if head.get("VersionId") not in (None, "null"):
            params["VersionId"] = head["VersionId"]
        else:
            require(bool(head.get("ETag")), "an unversioned download requires an ETag")
            params["IfMatch"] = head["ETag"]
        return self._call("get-object", params, destination)

    def list(self, prefix_uri):
        bucket, prefix = s3_parts(prefix_uri)
        params, result, tokens = {"Bucket": bucket, "Prefix": prefix}, [], set()
        while True:
            page = self._call("list-objects-v2", params)
            result.extend(
                {
                    "uri": f"s3://{bucket}/{r['Key']}",
                    "bytes": r["Size"],
                    "etag": r.get("ETag"),
                    "last_modified": r.get("LastModified"),
                    "verification": "listed_not_downloaded",
                }
                for r in page.get("Contents", [])
            )
            token = page.get("NextContinuationToken")
            if not token:
                require(not page.get("IsTruncated"), "S3 listing was truncated without a continuation token")
                return sorted(result, key=lambda r: r["uri"])
            require(token not in tokens, "S3 pagination repeated a continuation token")
            tokens.add(token)
            params["ContinuationToken"] = token


class ScratchBudget:
    def __init__(self, root, maximum_bytes):
        self.root, self.maximum_bytes = Path(root), maximum_bytes
        self.peak_bytes = 0

    def used(self):
        total = 0
        for path in self.root.rglob("*"):
            require(not path.is_symlink(), "scratch contains a symlink")
            if path.is_file():
                total += path.stat().st_size
        self.peak_bytes = max(self.peak_bytes, total)
        require(total <= self.maximum_bytes, "scratch budget exceeded")
        return total

    def admit(self, incoming):
        require(type(incoming) is int and incoming >= 0, "invalid source object size")
        used = self.used()
        require(used + incoming <= self.maximum_bytes, "one case exceeds the configured scratch budget")
        require(shutil.disk_usage(self.root).free >= incoming, "insufficient local scratch space")


def download_verified(reader, uri, destination, budget, expected_sha256=None):
    head = reader.head(uri)
    size = head["ContentLength"]
    metadata_sha = head.get("Metadata", {}).get("sha256")
    expected = expected_sha256 or metadata_sha
    require(
        isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected),
        "source object lacks a verifiable SHA-256",
    )
    if metadata_sha is not None:
        require(metadata_sha == expected, "object upload metadata differs from its expected checksum")
    budget.admit(size)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = reader.download(uri, destination, head)
    require(
        destination.stat().st_size == size and sha256(destination) == expected,
        "downloaded object failed its size or SHA-256 check",
    )
    if head.get("VersionId") not in (None, "null"):
        require(receipt.get("VersionId") == head["VersionId"], "downloaded S3 version differs from HEAD")
    require(receipt.get("ETag") == head.get("ETag"), "downloaded object ETag differs from HEAD")
    budget.used()
    return {
        "uri": uri,
        "bytes": size,
        "sha256": expected,
        "etag": head.get("ETag"),
        "version_id": head.get("VersionId"),
        "last_modified": head.get("LastModified"),
        "version_binding": "version_id" if head.get("VersionId") not in (None, "null") else "if_match_etag",
        "verification": "downloaded_size_and_sha256_verified",
    }


def gzip_blob(source, output, relative_stem, *, max_file_bytes):
    """Losslessly shard source bytes; no vector, null or JSON type is rewritten."""
    source, output = Path(source), Path(output)
    require(4096 <= max_file_bytes <= 64 * 1024**2, "exported file limit must be between 4 KiB and 64 MiB")
    chunk_bytes = max_file_bytes // 2
    total, offset, parts = source.stat().st_size, 0, []
    source_sha = sha256(source)
    with source.open("rb") as stream:
        while offset < total or not parts:
            name = relative_stem + (".gz" if total <= chunk_bytes else f".part-{len(parts) + 1:05d}.gz")
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".pending")
            count, raw_sha = 0, hashlib.sha256()
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as target:
                    while count < chunk_bytes:
                        value = stream.read(min(1024**2, chunk_bytes - count))
                        if not value:
                            break
                        target.write(value)
                        raw_sha.update(value)
                        count += len(value)
            require(
                temporary.stat().st_size <= max_file_bytes, "compressed shard exceeds the export file limit"
            )
            temporary.replace(path)
            parts.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "source_byte_range": [offset, offset + count],
                    "source_sha256": raw_sha.hexdigest(),
                }
            )
            offset += count
    require(offset == total, "gzip sharding did not retain every source byte")
    return {
        "encoding": "gzip-byte-ranges-v1",
        "source_bytes": total,
        "source_sha256": source_sha,
        "parts": parts,
        "reconstruction": "gunzip each part in listed order and concatenate raw bytes",
    }


def restore_blob(blob, output, destination, budget):
    budget.admit(blob["source_bytes"])
    position, full_sha = 0, hashlib.sha256()
    with Path(destination).open("wb") as target:
        for part in blob["parts"]:
            path = Path(output) / part["path"]
            require(
                path.is_file() and path.stat().st_size == part["bytes"] and sha256(path) == part["sha256"],
                "retained gzip shard changed",
            )
            require(part["source_byte_range"][0] == position, "gzip shard byte ranges are not contiguous")
            count, piece_sha = 0, hashlib.sha256()
            with gzip.open(path, "rb") as stream:
                while chunk := stream.read(1024**2):
                    count += len(chunk)
                    require(
                        position + count <= part["source_byte_range"][1],
                        "gzip shard exceeds its declared range",
                    )
                    target.write(chunk)
                    full_sha.update(chunk)
                    piece_sha.update(chunk)
            require(
                position + count == part["source_byte_range"][1]
                and piece_sha.hexdigest() == part["source_sha256"],
                "gzip shard content differs from its source byte range",
            )
            position += count
    require(
        position == blob["source_bytes"] and full_sha.hexdigest() == blob["source_sha256"],
        "reconstructed report differs from its original bytes",
    )
    budget.used()


def expected_matrix(rows=None, policies=None, seeds=None):
    return {
        (dataset, policy, seed)
        for dataset in (rows or DATASET_ROWS)
        for policy in (policies or POLICIES)
        for seed in (seeds or SEEDS)
    }


def campaign_uri(bucket, campaign, suffix):
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", campaign), "invalid campaign identity")
    return f"s3://{bucket}/campaigns/{campaign}/{suffix}"


def checked_outputs_uri(final, bucket, campaign, run_id):
    prefix = campaign_uri(bucket, campaign, f"runs/{run_id}/attempts/")
    uri = final["outputs_uri"]
    require(
        uri.startswith(prefix) and len(uri.removeprefix(prefix).split("/")) == 1,
        "final outputs leave their own campaign attempt",
    )
    s3_parts(uri)
    return uri


def download_json(reader, uri, destination, budget, expected=None):
    source = download_verified(reader, uri, destination, budget, expected)
    return read(destination), source


def collect_units(state, reader, work, budget, *, coordinates):
    """Bind current S3 plans/finals to the coordinator's previously verified identities."""
    bucket, units, seen = state["bucket"], [], set()
    for dataset, stages in sorted(state["datasets"].items()):
        for stage, record in sorted(stages.items()):
            campaign = record["request"]["campaign_id"]
            require(
                record["request"]["release"] == state["release"], "campaign release differs from suite state"
            )
            uri = campaign_uri(bucket, campaign, "plan.json")
            plan, plan_source = download_json(reader, uri, work / "plan.json", budget)
            (work / "plan.json").unlink()
            if set(state["datasets"]) == set(DATASET_ROWS):
                expected = {(policy, seed) for policy in STAGE_POLICIES[stage] for seed in SEEDS}
                actual = {(u["params"]["policy"], u["params"]["seed"]) for u in plan["units"]}
                require(actual == expected and len(plan["units"]) == len(expected), "campaign matrix differs")
            for unit in plan["units"]:
                run_id, params = unit["run_id"], unit["params"]
                coordinate = (dataset, params["policy"], params["seed"])
                require(
                    params.get("dataset") == dataset and coordinate in coordinates and coordinate not in seen,
                    "unexpected or duplicate suite coordinate",
                )
                require(re.fullmatch(r"[a-f0-9]{20}", run_id), "invalid run identity")
                require(
                    unit["campaign_id"] == campaign and unit["contract"]["protocol"] == PROTOCOL,
                    "unit campaign or protocol differs",
                )
                require(unit["contract_sha256"] == digest(unit["contract"]), "unit contract checksum differs")
                verified = state["verified_finals"].get(run_id)
                require(verified is not None, "unit has no coordinator verification receipt")
                final_uri = campaign_uri(bucket, campaign, f"runs/{run_id}/final.json")
                final, final_source = download_json(reader, final_uri, work / "final.json", budget)
                (work / "final.json").unlink()
                require(
                    digest(final) == verified["final_sha256"], "previously verified final manifest changed"
                )
                require(final["contract_sha256"] == unit["contract_sha256"], "final unit contract differs")
                require(
                    final.get("artifact_mode") == "metrics_only" and final.get("model") is None,
                    "final manifest advertises persisted model state",
                )
                require(
                    final.get("tracking", {}).get("complete") is True
                    and final["tracking"].get("required") is True,
                    "final publication is incomplete",
                )
                outputs = checked_outputs_uri(final, bucket, campaign, run_id)
                require(final["report_uri"] == outputs + "/report.json", "unexpected report path")
                require(
                    final["report_sha256"] == verified["report_sha256"],
                    "report differs from verified receipt",
                )
                require(
                    (verified["dataset"], verified["policy"], verified["seed"]) == coordinate,
                    "coordinator coordinate differs",
                )
                objects = reader.list(outputs + "/")
                require(
                    objects and all(row["uri"].startswith(outputs + "/") for row in objects),
                    "output listing is empty or leaves its attempt",
                )
                for row in objects:
                    relative = row["uri"].removeprefix(outputs + "/")
                    require(
                        not FORBIDDEN.search("/" + relative) and Path(relative).name not in RAW_NAMES,
                        "published output contains model state or raw input data",
                    )
                require(
                    len({row["uri"] for row in objects}) == len(objects), "duplicate object in S3 inventory"
                )
                units.append(
                    {
                        "unit": unit,
                        "final": final,
                        "dataset": dataset,
                        "stage": stage,
                        "sources": {"plan": plan_source, "final": final_source},
                        "inventory": objects,
                    }
                )
                seen.add(coordinate)
    require(
        seen == coordinates and len(units) == len(coordinates), "suite does not cover the complete matrix"
    )
    require(
        set(state["verified_finals"]) == {u["unit"]["run_id"] for u in units},
        "coordinator contains unexpected verified runs",
    )
    return units


def report_contract(item, report, rows):
    unit, final = item["unit"], item["final"]
    config, contract = unit["params"], report["contract"]
    dataset = item["dataset"]
    source = unit["contract"].get("stream_source") or unit["contract"].get("snapshot") or {}
    require(
        report.get("complete") is True and report["protocol"] == PROTOCOL,
        "report is incomplete or uses another protocol",
    )
    require(
        (report["dataset"], report["policy"], report["scope"]) == (dataset, config["policy"], "full"),
        "report coordinates differ from its unit",
    )
    require(
        contract["input"]["rows"] == report["metrics"]["dataset_rows"] == rows[dataset],
        "source coverage differs",
    )
    require(contract["input"]["sha256"] == source["prepared_sha256"], "prepared source identity differs")
    require(
        contract["iterative_training"] == config["iterative_training"],
        "training or randomness contract differs",
    )
    require(
        config["iterative_training"]["seed"] == config["seed"],
        "unit training seed differs from its coordinate",
    )
    require(
        config["iterative_training"]["max_epochs"] == 100,
        "full evaluation requires the declared 100-epoch budget",
    )
    require(final["release"] == unit["contract"]["release"], "release binding differs")
    return {
        "case_id": unit["run_id"],
        "config": {
            "dataset": dataset,
            "policy": config["policy"],
            "scope": "full",
            "seed": config["seed"],
            "input_sha256": contract["input"]["sha256"],
            "iterative_training": config["iterative_training"],
        },
    }


def retained_json(value, output, stem, work, maximum, budget):
    path = work / "retained.json"
    used, count = budget.used(), 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("wb") as stream:
        for chunk in encoder.iterencode(value):
            payload = chunk.encode()
            count += len(payload)
            require(used + count <= budget.maximum_bytes, "derived metric evidence exceeds scratch budget")
            stream.write(payload)
    budget.used()
    blob = gzip_blob(path, output, stem, max_file_bytes=maximum)
    path.unlink()
    return blob


def validate_live_ledger(path):
    count = 0
    with Path(path).open() as stream:
        for line in stream:
            require(bool(line.strip()), "live metric ledger contains an empty record")
            row = json.loads(line)
            require(
                set(row) == {"step", "timestamp", "prefix", "values"}, "unexpected live metric ledger schema"
            )
            require(
                type(row["step"]) is int and row["step"] >= 0 and type(row["timestamp"]) is int,
                "invalid native progress coordinate",
            )
            require(
                re.fullmatch(r"attempts/[^/]+/progress", row["prefix"]), "unexpected progress attempt axis"
            )
            require(
                isinstance(row["values"], dict)
                and all(
                    isinstance(k, str) and type(v) in (int, float) and math.isfinite(v)
                    for k, v in row["values"].items()
                ),
                "live ledger is not scalar metric evidence",
            )
            count += 1
    return count


def reused_case(item, case, common, output):
    """A sealed local audit can be resumed without downloading predictions again."""
    root = output / "cases" / item["unit"]["run_id"]
    receipt_path = root / "receipt.json"
    if not receipt_path.exists():
        return None
    receipt = read(receipt_path)
    require(
        receipt["case"] == case and receipt["cohort_seq_range"] == list(common),
        "retained case audit belongs to another scientific contract",
    )
    require(
        sha256(root / "qa.json") == receipt["qa_sha256"]
        and sha256(root / "provenance.json") == receipt["provenance_sha256"],
        "retained case audit files changed",
    )
    provenance, qa = read(root / "provenance.json"), read(root / "qa.json")
    require(
        provenance["unit"] == item["unit"]
        and provenance["final"] == item["final"]
        and provenance["listed_output_objects"] == item["inventory"],
        "retained source publication changed",
    )
    for blob in provenance["retained"].values():
        for part in blob["parts"]:
            path = output / part["path"]
            require(
                path.is_file() and path.stat().st_size == part["bytes"] and sha256(path) == part["sha256"],
                "retained case metric evidence changed",
            )
    require(
        qa["complete"] is True and qa["common_comparison"]["seq_range"] == list(common),
        "retained prediction reconstruction is incomplete",
    )
    return qa


def case_evidence(item, report_blob, case, common, reader, output, work, budget, maximum, verifier):
    run_id, final = item["unit"]["run_id"], item["final"]
    previous = reused_case(item, case, common, output)
    if previous is not None:
        return previous
    restore_blob(report_blob, output, work / "report.json", budget)
    report = read(work / "report.json")
    prefix, sources, retained = final["outputs_uri"], {}, {}
    inventory = {r["uri"]: r for r in item["inventory"]}
    essentials = {
        "predictions-manifest.json": final["predictions_manifest_uri"],
        "training-history.json": final["training_history_uri"],
        "artifacts/data/index.json": final["artifacts_index_uri"],
        "metric-definitions.json": prefix + "/metric-definitions.json",
    }
    for relative, uri in essentials.items():
        require(
            uri == prefix + "/" + relative and uri in inventory,
            "essential evidence is outside its output inventory",
        )
        sources[relative] = download_verified(reader, uri, work / relative, budget)
        retained[relative] = gzip_blob(
            work / relative, output, f"metadata/{run_id}/{relative}", max_file_bytes=maximum
        )
    index = read(work / "predictions-manifest.json")
    require(index["dataset_sha256"] == case["config"]["input_sha256"], "prediction input identity differs")
    require(
        read(work / "training-history.json") == {"version": 1, "fits": report["training_history"]},
        "published per-fit history differs from the full report",
    )
    part_sizes = sum(part["bytes"] for part in index["parts"])
    budget.admit(part_sizes)
    seen = set()
    for part in index["parts"]:
        name = part["file"]
        require(
            Path(name).name == name and name.endswith(".parquet") and name not in seen,
            "unsafe or duplicate prediction part",
        )
        seen.add(name)
        relative, uri = "predictions/" + name, prefix + "/predictions/" + name
        require(uri in inventory and inventory[uri]["bytes"] == part["bytes"], "prediction inventory differs")
        sources[relative] = download_verified(reader, uri, work / relative, budget, part["sha256"])
    qa = verifier.verify_case(work, case, common_range=common)
    adapted = verifier.tracking_report(report, index)
    require(
        sha256(work / "metric-definitions.json") == adapted["tracking"]["definitions_sha256"],
        "retained metric definitions differ from the executed verifier contract",
    )
    retained["tracking-v1.json"] = retained_json(
        adapted["tracking"], output, f"tracking/{run_id}/tracking-v1.json", work, maximum, budget
    )
    live = prefix + "/live-metrics.jsonl"
    if live in inventory:
        path = work / "live-metrics.jsonl"
        sources["live-metrics.jsonl"] = download_verified(reader, live, path, budget)
        qa["live_metric_records"] = validate_live_ledger(path)
        retained["live-metrics.jsonl"] = gzip_blob(
            path, output, f"tracking/{run_id}/live-metrics.jsonl", max_file_bytes=maximum
        )
    qa["scope"] = "prediction_and_retained_metric_evidence; native MLflow API audit is independent"
    qa["scratch_peak_bytes"] = budget.peak_bytes
    provenance = {
        "unit": item["unit"],
        "final": final,
        "cohort_seq_range": list(common),
        "source_objects": {**item["sources"], **sources},
        "listed_output_objects": item["inventory"],
        "retained": {"full_report": report_blob, **retained},
        "predictions": "all parts verified and reconstructed; temporary local copy removed after this case",
        "metric_scope_contract": report["contract"].get("metrics_only"),
        "native_mlflow_api_audit": {"verified": False, "reason": "separate_readback_required"},
    }
    save(output / "cases" / run_id / "qa.json", qa)
    save(output / "cases" / run_id / "provenance.json", provenance)
    save(
        output / "cases" / run_id / "receipt.json",
        {
            "case": case,
            "cohort_seq_range": list(common),
            "qa_sha256": sha256(output / "cases" / run_id / "qa.json"),
            "provenance_sha256": sha256(output / "cases" / run_id / "provenance.json"),
        },
    )
    return qa


def scalar_rows(report, case):
    coordinates = {k: case["config"][k] for k in ("dataset", "policy", "seed")}
    for scope, values in (
        ("whole_prequential_reported", report["metrics"]),
        ("final_holdout", report["holdout"]),
    ):
        for metric, value in values.items():
            if value is None or type(value) in (int, float):
                yield {
                    **coordinates,
                    "scope": scope,
                    "metric": metric,
                    "value": value,
                    "availability": "undefined" if value is None else "defined",
                    "cohort_n": values["n"],
                }


def write_scalars(rows, output):
    if not rows:
        return
    with (output / "reported-scalars-by-seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["policy"], row["scope"], row["metric"]].append(row)
    aggregate = []
    for (dataset, policy, scope, metric), group in sorted(groups.items()):
        values = [r["value"] for r in group if r["value"] is not None]
        aggregate.append(
            {
                "dataset": dataset,
                "policy": policy,
                "scope": scope,
                "metric": metric,
                "n_seeds_present": len(group),
                "n_defined": len(values),
                "n_undefined": len(group) - len(values),
                "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
                "sd_ddof": 1,
            }
        )
    with (output / "reported-scalars-mean-sd.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)


def validate_selection(state, status, rows, coordinates):
    """The CLI exports only the selected suite; synthetic row overrides serve local tests."""
    require(
        set(state["datasets"]) == set(rows) and not set(rows) & {"fraud", "ecommerce", "social"},
        "coordinator selection differs from the selected datasets; preserve historical state separately",
    )
    if set(rows) != set(DATASET_ROWS):
        return
    require(coordinates == expected_matrix(rows), "the selected suite requires every policy and seed")
    requests = state.get("contract", {}).get("requests", {})
    require(
        set(requests) == set(rows)
        and all(
            state["datasets"][dataset].get("base", {}).get("request") == request
            for dataset, request in requests.items()
        ),
        "coordinator differs from its selected request contract",
    )
    campaigns = {}
    for dataset, stages in state["datasets"].items():
        require(set(stages) == set(STAGE_POLICIES), "all selected base and cached campaigns are required")
        for stage, record in stages.items():
            request = record["request"]
            matrix = request.get("matrix", {})
            policies = matrix.get("policies", [])
            require(
                request.get("dataset") == dataset
                and request.get("release") == state["release"]
                and request.get("scope") == "full"
                and not request.get("snapshot")
                and matrix.get("kind") == "stream"
                and matrix.get("artifact_mode") == "metrics_only"
                and matrix.get("seeds") == list(SEEDS)
                and matrix.get("iterative_training", {}).get("max_epochs") == 100
                and set(policies) == set(STAGE_POLICIES[stage])
                and len(policies) == len(STAGE_POLICIES[stage]),
                "campaign request differs from the selected full metrics-only matrix",
            )
            caches = matrix.get("metric_cache_sources") or {}
            require(
                not caches if stage == "base" else set(caches) == {str(seed) for seed in SEEDS},
                "cached campaign requires three independently seeded CARA cache receipts",
            )
            campaign = request["campaign_id"]
            require(campaign not in campaigns, "duplicate campaign identity")
            campaigns[campaign] = len(STAGE_POLICIES[stage]) * len(SEEDS)
    require(
        status.get("expected_units") == len(coordinates)
        and status.get("expected_campaigns") == len(rows) * len(STAGE_POLICIES)
        and set(status.get("campaigns", {})) == set(campaigns)
        and all(
            status["campaigns"][campaign].get("complete") is True
            and status["campaigns"][campaign].get("expected_units")
            == status["campaigns"][campaign].get("verified_units")
            == count
            for campaign, count in campaigns.items()
        ),
        "all selected campaign completion receipts are required",
    )


def export_evidence(
    state_dir,
    output,
    scratch_root,
    reader,
    *,
    scratch_bytes=20 * 1024**3,
    max_file_bytes=64 * 1024**2,
    expected_rows=None,
    coordinates=None,
):
    state_dir, output, scratch_root = map(lambda p: Path(p).resolve(), (state_dir, output, scratch_root))
    require(
        output != state_dir and not output.is_relative_to(state_dir),
        "output must not alter coordinator state",
    )
    require(not scratch_root.is_relative_to(state_dir), "scratch must not write inside coordinator state")
    require(
        scratch_root != output and not output.is_relative_to(scratch_root),
        "scratch cleanup must not contain the export",
    )
    state = read(state_dir / "suite-state.json")
    status = read(state_dir / "latest-status.json")
    rows = expected_rows or DATASET_ROWS
    coordinates = expected_matrix(rows) if coordinates is None else set(coordinates)
    require(state["schema"] == "aws-ordered-dataset-suite-v1", "unsupported coordinator state")
    validate_selection(state, status, rows, coordinates)
    require(
        status.get("complete") is True
        and status.get("verified_units") == len(coordinates)
        and status.get("batch", {}).get("active") == 0,
        "the full suite must finish before final export",
    )
    require(len(state["verified_finals"]) == len(coordinates), "not every unit has a verified final receipt")
    require(scratch_bytes > 0, "positive scratch budget is required")
    output.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "state_contract": state["contract"],
        "verified_finals": state["verified_finals"],
        "datasets": state["datasets"],
        "release": state["release"],
        "bucket": state["bucket"],
        "region": state["region"],
        "expected_rows": rows,
        "expected_coordinates": [list(c) for c in sorted(coordinates)],
        "exporter_sha256": sha256(__file__),
        "verifier_sha256": sha256(PROJECT / "scripts/verify_dataset_suite.py"),
        "scratch_bytes": scratch_bytes,
        "max_file_bytes": max_file_bytes,
    }
    contract_path = output / "export-contract.json"
    if contract_path.exists():
        require(
            read(contract_path) == identity,
            "changed source or exporter contract requires a new output directory",
        )
    else:
        save(contract_path, identity)
    verifier = load_verifier()
    save(output / "source-state.json", state)
    save(output / "source-completion.json", status)
    with tempfile.TemporaryDirectory(prefix="dataset-evidence-catalog-", dir=scratch_root) as temporary:
        work = Path(temporary)
        budget = ScratchBudget(work, scratch_bytes)
        items = collect_units(state, reader, work, budget, coordinates=coordinates)
    summaries, source_records, scalars = {}, {}, []
    for item in items:
        run_id, final = item["unit"]["run_id"], item["final"]
        descriptor_path = output / "reports" / run_id / "source.json"
        with tempfile.TemporaryDirectory(prefix="dataset-evidence-report-", dir=scratch_root) as temporary:
            work = Path(temporary)
            budget = ScratchBudget(work, scratch_bytes)
            report_path = work / "report.json"
            if descriptor_path.exists():
                source_record = read(descriptor_path)
                require(
                    source_record["source"]["sha256"] == final["report_sha256"],
                    "retained report identity changed",
                )
                restore_blob(source_record["blob"], output, report_path, budget)
            else:
                source = download_verified(
                    reader, final["report_uri"], report_path, budget, final["report_sha256"]
                )
                blob = gzip_blob(
                    report_path, output, f"reports/{run_id}/report.full.json", max_file_bytes=max_file_bytes
                )
                source_record = {"source": source, "blob": blob}
                save(descriptor_path, source_record)
            report = read(report_path)
            case = report_contract(item, report, rows)
            summary = {
                "case": case,
                "input_sha256": case["config"]["input_sha256"],
                "calibration_end": verifier.calibration_end(report),
                "holdout_start": report["contract"]["holdout_start"],
            }
            summaries[run_id], source_records[run_id] = summary, source_record
            scalars.extend(scalar_rows(report, case))
        item["sources"]["report"] = source_record["source"]
    groups = defaultdict(list)
    for run_id, summary in summaries.items():
        groups[summary["case"]["config"]["dataset"], summary["input_sha256"]].append(run_id)
    require(
        len(groups) == len(rows), "each dataset must share one source identity across all policies and seeds"
    )
    ranges = {}
    for key, runs in groups.items():
        start = max(summaries[r]["calibration_end"] for r in runs)
        end = min(summaries[r]["holdout_start"] for r in runs)
        require(start < end, "no common post-calibration prequential cohort")
        ranges[key] = (start, end)
    results, peak_scratch = [], 0
    for item in items:
        run_id = item["unit"]["run_id"]
        summary = summaries[run_id]
        common = ranges[summary["case"]["config"]["dataset"], summary["input_sha256"]]
        with tempfile.TemporaryDirectory(prefix="dataset-evidence-case-", dir=scratch_root) as temporary:
            work = Path(temporary)
            budget = ScratchBudget(work, scratch_bytes)
            result = case_evidence(
                item,
                source_records[run_id]["blob"],
                summary["case"],
                common,
                reader,
                output,
                work,
                budget,
                max_file_bytes,
                verifier,
            )
            budget.used()
            peak_scratch = max(peak_scratch, budget.peak_bytes, result["scratch_peak_bytes"])
            results.append(result)
        save(
            output / "export-progress.json",
            {
                "complete": False,
                "audited_cases": len(results),
                "expected_cases": len(coordinates),
                "last_run_id": run_id,
            },
        )
        print(
            json.dumps({"run_id": run_id, "audited_cases": len(results), "expected_cases": len(coordinates)}),
            flush=True,
        )
    verifier.comparison_tables(results, output)
    write_scalars(scalars, output)
    save(
        output / "cohorts.json",
        {
            "cohorts": [
                {"dataset": key[0], "input_sha256": key[1], "seq_range": list(value)}
                for key, value in ranges.items()
            ]
        },
    )
    current = read(state_dir / "suite-state.json")
    require(
        all(current[k] == state[k] for k in ("contract", "verified_finals", "datasets", "release")),
        "coordinator source identities changed during export",
    )
    save(
        output / "export-progress.json",
        {"complete": True, "audited_cases": len(results), "expected_cases": len(coordinates)},
    )
    files = [
        {"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "SHA256SUMS"}
    ]
    require(
        all(f["bytes"] <= max_file_bytes for f in files), "an exported file exceeds its configured size limit"
    )
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": state["release"],
        "export_contract_sha256": sha256(contract_path),
        "coverage": {
            "cases": len(results),
            "campaigns": sum(len(stages) for stages in state["datasets"].values()),
            "cohorts": len(ranges),
            "datasets": sorted(rows),
            "seeds": sorted({c[2] for c in coordinates}),
            "policies": sorted({c[1] for c in coordinates}),
        },
        "checks": {
            "finals_bound_to_coordinator": True,
            "full_source_coverage": True,
            "all_prediction_parts_verified": True,
            "prediction_metrics_reconstructed": True,
            "no_model_state": True,
            "full_report_bytes_preserved": True,
        },
        "scratch": {
            "maximum_bytes": scratch_bytes,
            "observed_peak_case_bytes": peak_scratch,
            "policy": "one case at a time; temporary local predictions removed after verification",
        },
        "native_mlflow_api_readback": {"verified": False, "reason": "separate_independent_audit_required"},
        "interpretation": {
            "primary": "paired policy comparisons within a stream and common sequence range",
            "cross_dataset": "descriptive; tasks, label distributions and block budgets differ",
            "ranking": "4096-bin histogram estimates, binary-positive and multiclass-macro scopes retained",
            "costs": (
                "logical deployment, physical execution, matrix reuse and reconstruction "
                "scopes remain in full contracts"
            ),
            "undefined": "nulls preserved; n_defined and sample SD with ddof=1 are explicit",
        },
        "files": files,
    }
    save(output / "manifest.json", manifest)
    with (output / "SHA256SUMS").open("w") as stream:
        for entry in [*files, {"path": "manifest.json", "sha256": sha256(output / "manifest.json")}]:
            stream.write(f"{entry['sha256']}  {entry['path']}\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--scratch-gib", type=int, default=20)
    parser.add_argument("--max-file-mib", type=int, default=64)
    args = parser.parse_args(argv)
    state = read(args.state_dir / "suite-state.json")
    reader = ReadOnlyS3(state["region"])
    manifest = export_evidence(
        args.state_dir,
        args.output,
        args.scratch_root,
        reader,
        scratch_bytes=args.scratch_gib * 1024**3,
        max_file_bytes=args.max_file_mib * 1024**2,
    )
    print(
        json.dumps(
            {
                "complete": manifest["complete"],
                "coverage": manifest["coverage"],
                "files": len(manifest["files"]),
                "manifest": str(args.output / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
