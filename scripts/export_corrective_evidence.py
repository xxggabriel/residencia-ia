"""Export metric-only evidence for a versioned corrective benchmark campaign.

The source directory stays untouched. Raw inputs, models and training snapshots
are never copied. Native reports and prediction ledgers retain the information
needed to audit every reported metric; paired scores are recalculated on common
event IDs with the benchmark's own exact binary Metrics implementation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import io
import ipaddress
import json
import math
import re
import textwrap
import zlib
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev, stdev
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from continual_bench.core.metrics import Metrics

SCHEMA = "corrective-evidence-v2"
DENSE_ARRAY_FIELDS = frozenset({"roc_curve", "pr_curve", "residuals"})
PROTECTED_ARRAY_FIELDS = frozenset(
    {
        "series",
        "windows",
        "training_history",
        "system_metrics",
        "confusion_matrix",
        "performance_matrix",
        "staleness_matrix",
        "matrix",
        "matrices",
    }
)
INDEXED_ARRAY = object()
LARGE_PAYLOAD_BYTES = 1024 * 1024
TABLE_SHARD_TRIGGER_BYTES = 95 * 1024 * 1024
TABLE_SHARD_MAX_BYTES = 64 * 1024 * 1024
COORDINATES = ("dataset", "model", "policy", "detector", "seed", "mode")
SECRET_KEY = re.compile(r"password|secret|token|authorization|cookie|credential|access.key", re.I)
MODEL_SUFFIXES = {".pkl", ".pickle", ".bin", ".ubj", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"}
TABLE_NAMES = (
    "metrics-long",
    "coverage",
    "paired-by-seed",
    "aggregate-metrics",
    "replication-by-seed",
    "replication-aggregate",
    "intentional-exclusions",
    "historical-comparison",
    "vector-index",
)
METRIC_SECTIONS = (
    "metrics",
    "summary",
    "windows",
    "compute",
    "compute_seconds",
    "peak_rss_bytes_process",
    "training",
    "training_history",
    "validation",
    "holdout",
    "class_metrics",
    "holdout_class_metrics",
    "tracking",
    "tracking_observations",
    "observations_v1",
    "system_metrics",
    "costs",
    "retraining",
    "upf",
    "retention_probes",
    "alarms",
    "updates",
    "task_evaluation",
)
PREDICTION_FIELDS = (
    "event_id",
    "seq",
    "timestamp",
    "available_at",
    "released_at",
    "probability",
    "label",
    "previous_probability",
    "comparison_id",
    "episode",
    "concept",
    "prediction_seconds",
)


def json_read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def compact_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_write(path, raw, *, compresslevel=9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as output,
        gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=compresslevel) as packed,
    ):
        packed.write(raw)


def write_payload(path, value, *, source_bytes):
    """Preserve the JSON value and numeric types while adapting only its encoding."""
    compact = compact_json_bytes(value)
    canonical_sha256 = hashlib.sha256(compact).hexdigest()
    large = source_bytes >= LARGE_PAYLOAD_BYTES
    raw = compact if large else json_bytes(value)
    gzip_write(path, raw, compresslevel=6 if large else 9)
    return {
        "json_format": "compact" if large else "pretty",
        "gzip_level": 6 if large else 9,
        "uncompressed_bytes": len(raw),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_compact_sha256": canonical_sha256,
        "compressed_bytes": path.stat().st_size,
    }


def portable(value, state, *, key=""):
    """Strip connection secrets and machine paths without changing numeric evidence."""
    if isinstance(value, dict):
        return {
            name: portable(item, state, key=name)
            for name, item in value.items()
            if not SECRET_KEY.search(name)
            and name.lower() not in {"hostname", "nodename", "host", "username", "user_name"}
        }
    if isinstance(value, list):
        return [portable(item, state, key=key) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith("file://"):
        return "file://<artifact-path>/" + Path(urlsplit(value).path).name
    if value.startswith(("http://", "https://", "s3://")):
        url = urlsplit(value)
        hostname = url.hostname or ""
        private = hostname in {"localhost", "mlflow", "minio"}
        try:
            private |= not ipaddress.ip_address(hostname).is_global
        except ValueError:
            pass
        authority = "<tracking-endpoint>" if private else url.netloc.rsplit("@", 1)[-1]
        query = urlencode(
            [
                (name, value)
                for name, value in parse_qsl(url.query, keep_blank_values=True)
                if not SECRET_KEY.search(name) and not re.search(r"signature|x-amz", name, re.I)
            ]
        )
        return urlunsplit((url.scheme, authority, url.path, query, url.fragment))
    if value.startswith("/"):
        path = Path(value)
        try:
            return "<campaign>/" + path.relative_to(state).as_posix()
        except ValueError:
            return "<external-path>/" + path.name
    return value


def assert_portable(value):
    text = json.dumps(value, ensure_ascii=False)
    if re.search(r"homelab|/Users/|/home/|192\.168\.\d+\.\d+", text, re.I):
        raise ValueError("nonportable execution-specific text remains in evidence")


def contained(state, value):
    path = (state / value).resolve()
    if not path.is_relative_to(state):
        raise ValueError("evidence source escapes campaign directory")
    return path


def prediction_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if not {"event_id", "seq", "label", "probability"} <= row.keys():
                    raise ValueError("prediction ledger requires event_id, seq, label and probability")
                if not isinstance(row["event_id"], str) or not row["event_id"]:
                    raise ValueError("prediction event ID must be a nonempty string")
                if type(row["seq"]) is not int or row["seq"] < 0 or row["label"] not in (False, True):
                    raise ValueError("prediction seq/label violates the binary metric contract")
                yield row


@dataclass
class Case:
    key: str
    coordinates: dict
    status: str
    report: Path | None
    prediction: Path | None
    expected: bool = True


def cases_from_plan(state, plan, status):
    result = []
    for entry in plan.get("matrix", {}).get("cases", []):
        key = entry["case_key"]
        coordinates = dict(entry.get("coordinates") or {})
        config = entry.get("config") or {}
        for name in COORDINATES:
            if name in coordinates:
                continue
            value = config.get(name)
            if name == "seed":
                value = (config.get("dataset") or {}).get("seed")
            coordinates[name] = value.get("name") if isinstance(value, dict) else value
        item = (status.get("matrix") or {}).get(key) or {}
        report = contained(state, item["report"]) if item.get("report") else None
        if report is None or not report.is_file():
            candidates = [
                p
                for p in (state / "matrix" / "cases" / key).rglob("report.json")
                if "artifacts" not in p.parts
            ]
            if len(candidates) > 1:
                raise ValueError(f"ambiguous report for case {key}")
            report = contained(state, str(candidates[0].relative_to(state))) if candidates else None
        passed = item.get("status") == "passed"
        complete = report is not None and json_read(report).get("complete") is True
        observed_status = "passed" if passed and complete else item.get("status", "pending")
        if passed and not complete:
            observed_status = "missing_or_incomplete_report"
        prediction = report.parent / "predictions.jsonl" if report else None
        if prediction and not prediction.is_file():
            prediction = None
        result.append(Case(key, coordinates, observed_status, report, prediction))
    if not result and not plan.get("replications", {}).get("jobs"):
        raise ValueError("campaign-plan.json has no expected matrix cases")
    if len({c.key for c in result}) != len(result):
        raise ValueError("duplicate case keys in campaign plan")
    return result


def numeric_leaves(value, prefix="", *, step=0, context=None):
    """Yield all numeric or explicitly unavailable leaves with their native scope."""
    context = context or {}
    if value is None or type(value) in (int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite metric at {prefix}")
        yield prefix, step, value, context
    elif isinstance(value, dict):
        local = dict(context)
        for field in ("step", "end_seq", "seq", "round", "epoch", "batch", "index"):
            if type(value.get(field)) is int:
                step = value[field]
                local[field] = step
                break
        for field in ("scope", "split", "kind", "fit_id", "unit", "timestamp_ms", "start_seq"):
            if field in value and not isinstance(value[field], (dict, list)):
                local[field] = value[field]
        if set(value) >= {"step", "value"}:
            yield from numeric_leaves(value["value"], prefix, step=step, context=local)
        else:
            for key, item in value.items():
                if key in {
                    "availability",
                    "features",
                    "parameters",
                    "config",
                    "environment",
                    "retrain_points",
                    "served_by",
                    "oracle_retrain_points",
                }:
                    continue
                yield from numeric_leaves(item, f"{prefix}/{key}".strip("/"), step=step, context=local)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            local = {
                **context,
                "list_index": index,
                "list_indices": [*context.get("list_indices", []), index],
            }
            yield from numeric_leaves(item, prefix, step=index, context=local)


def writer(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("w", encoding="utf-8", newline="")
    output = csv.DictWriter(stream, fields, extrasaction="ignore")
    output.writeheader()
    return stream, output


def pointer_token(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def vector_sha256(value):
    return hashlib.sha256(compact_json_bytes(value)).hexdigest()


class VectorIndex:
    """Stream references to exact dense arrays retained in complete native payloads."""

    def __init__(self, path, resources):
        stream, self.table = writer(path, ["artifact", "json_pointer", "container", "length", "sha256"])
        resources.enter_context(stream)
        self.count = 0

    def record(self, artifact, pointer, container, value):
        self.table.writerow(
            {
                "artifact": artifact,
                "json_pointer": pointer,
                "container": container,
                "length": len(value),
                "sha256": vector_sha256(value),
            }
        )
        self.count += 1


def indexed_csv_view(value, artifact, index, *, pointer="", container=None):
    """Create only a CSV view; leave the complete archived scientific payload intact.

    Only arrays below the three explicit dense-curve fields are represented by
    catalog references. Scalars inside those fields and every other list,
    including temporal series and matrices, remain available to numeric_leaves.
    """
    if isinstance(value, list):
        temporal = bool(value) and all(
            isinstance(item, dict) and {"step", "value"} <= item.keys() for item in value
        )
        if container is not None and not temporal:
            index.record(artifact, pointer, container, value)
            return INDEXED_ARRAY
        return [
            indexed_csv_view(item, artifact, index, pointer=f"{pointer}/{position}")
            for position, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: indexed_csv_view(
                item,
                artifact,
                index,
                pointer=f"{pointer}/{pointer_token(key)}",
                container=(
                    None
                    if key in PROTECTED_ARRAY_FIELDS
                    else container or (key if key in DENSE_ARRAY_FIELDS else None)
                ),
            )
            for key, item in value.items()
        }
    return value


def _csv_record_bytes(row):
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerow(row)
    return buffer.getvalue().encode("utf-8")


def _logical_csv_digest(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    digest, size = hashlib.sha256(), 0
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def shard_csv_gzip(source, output, *, max_bytes=TABLE_SHARD_MAX_BYTES):
    """Split a canonical CSV by records, with bounded gzip files and repeated headers.

    Probe a copied compressor once per bounded batch. Only batches that fit are
    committed; a boundary batch is recompressed in the next part. The final gzip
    trailer is included in every size check. No data row is split or discarded.
    """
    source, output = Path(source), Path(output)
    if max_bytes < 64:
        raise ValueError("CSV shard budget must be at least 64 bytes")
    output.mkdir(parents=True, exist_ok=True)
    logical_name = source.name[:-3] if source.name.endswith(".gz") else source.name
    if not logical_name.endswith(".csv"):
        raise ValueError("CSV shard input must be .csv or .csv.gz")
    stem = logical_name[:-4]
    original_bytes, original_sha256 = _logical_csv_digest(source)
    logical_hash = hashlib.sha256()
    logical_bytes = processed_rows = 0
    parts = []
    state = {"stream": None}
    header = None

    def start():
        path = output / f"{stem}.part-{len(parts) + 1:05d}.csv.gz"
        stream = path.open("wb")
        compressor = zlib.compressobj(level=9, wbits=31)
        stream.write(compressor.compress(header) + compressor.flush(zlib.Z_SYNC_FLUSH))
        state.update(
            stream=stream,
            path=path,
            compressor=compressor,
            rows=0,
            first_row=processed_rows,
            raw_bytes=len(header),
            raw_hash=hashlib.sha256(header),
        )
        if stream.tell() + len(compressor.copy().flush(zlib.Z_FINISH)) >= max_bytes:
            raise ValueError("CSV header exceeds shard size budget")

    def finish():
        stream = state["stream"]
        stream.write(state["compressor"].flush(zlib.Z_FINISH))
        stream.close()
        state["stream"] = None
        path = state["path"]
        if path.stat().st_size >= max_bytes:
            raise ValueError("CSV shard exceeds compressed size budget")
        parts.append(
            {
                "path": path.name,
                "rows": state["rows"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "uncompressed_bytes": state["raw_bytes"],
                "uncompressed_sha256": state["raw_hash"].hexdigest(),
                "source_row_range": [state["first_row"], processed_rows],
            }
        )

    def append(records):
        nonlocal processed_rows
        raw = b"".join(records)
        trial = state["compressor"].copy()
        encoded = trial.compress(raw) + trial.flush(zlib.Z_SYNC_FLUSH)
        final_bytes = state["stream"].tell() + len(encoded) + len(trial.copy().flush(zlib.Z_FINISH))
        if final_bytes >= max_bytes:
            if state["rows"]:
                finish()
                start()
                return append(records)
            if len(records) == 1:
                raise ValueError("one CSV record exceeds shard size budget")
            middle = len(records) // 2
            append(records[:middle])
            return append(records[middle:])
        state["stream"].write(encoded)
        state["compressor"] = trial
        state["raw_hash"].update(raw)
        state["raw_bytes"] += len(raw)
        state["rows"] += len(records)
        processed_rows += len(records)

    opener = gzip.open if str(source).endswith(".gz") else open
    try:
        with opener(source, "rt", encoding="utf-8", newline="") as stream:
            records = csv.reader(stream)
            fields = next(records, None)
            if fields is None:
                raise ValueError("CSV shard input has no header")
            header = _csv_record_bytes(fields)
            logical_hash.update(header)
            logical_bytes += len(header)
            start()
            batch, batch_bytes = [], 0
            batch_limit = min(1024 * 1024, max(64, max_bytes // 8))
            for row in records:
                raw = _csv_record_bytes(row)
                logical_hash.update(raw)
                logical_bytes += len(raw)
                batch.append(raw)
                batch_bytes += len(raw)
                if batch_bytes >= batch_limit:
                    append(batch)
                    batch, batch_bytes = [], 0
            if batch:
                append(batch)
            finish()
    finally:
        if state["stream"] is not None:
            state["stream"].close()
    if logical_bytes != original_bytes or logical_hash.hexdigest() != original_sha256:
        raise ValueError("CSV canonical record encoding does not match the source bytes")
    return {
        "format": "csv+gzip-shards",
        "header": fields,
        "header_repeated": True,
        "data_rows": processed_rows,
        "logical_csv_bytes": original_bytes,
        "logical_csv_sha256": original_sha256,
        "source_artifact": source.name,
        "source_sha256": sha256(source),
        "gzip_level": 9,
        "max_shard_bytes": max_bytes,
        "shards": parts,
    }


def table_parts(entry):
    """Resolve a manifest table entry in its required concatenation order."""
    return [entry] if isinstance(entry, str) else [part["path"] for part in entry["shards"]]


def _clean_table_parts(output, name, keep=()):
    pattern = re.compile(re.escape(name) + r"\.part-\d{5}\.csv\.gz$")
    for path in output.iterdir():
        if path.is_file() and pattern.fullmatch(path.name) and path.name not in keep:
            path.unlink()


def pack_large_tables(
    output,
    threshold=20 * 1024 * 1024,
    *,
    shard_trigger=TABLE_SHARD_TRIGGER_BYTES,
    shard_max=TABLE_SHARD_MAX_BYTES,
):
    """Keep small table paths compatible; shard only gzips exceeding the measured limit."""
    table_files = {}
    for name in TABLE_NAMES:
        path = output / (name + ".csv")
        compressed = path.with_suffix(".csv.gz")
        if not path.exists():
            continue
        if path.stat().st_size > threshold:
            with path.open("rb") as source, compressed.open("wb") as stream:
                with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as packed:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        packed.write(block)
                        if stream.tell() > shard_trigger:
                            break  # The finalized file can only grow beyond this size.
            if compressed.stat().st_size > shard_trigger:
                entry = shard_csv_gzip(path, output, max_bytes=shard_max)
                table_files[path.name] = entry
                compressed.unlink()
                _clean_table_parts(output, name, table_parts(entry))
            else:
                table_files[path.name] = compressed.name
                _clean_table_parts(output, name)
            path.unlink()
        else:
            compressed.unlink(missing_ok=True)
            table_files[path.name] = path.name
            _clean_table_parts(output, name)
    return table_files


def distribution(values, ddof):
    valid = [v for v in values if v is not None]
    n = len(valid)
    deviation = (stdev(valid) if ddof else pstdev(valid)) if n > ddof else None
    return {
        "n": n,
        "missing": len(values) - n,
        "mean": mean(valid) if n else None,
        "std": deviation,
        "ddof": ddof,
        "divisor": n - ddof if n > ddof else None,
    }


def aggregate(rows, group_keys, *, ddof):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row.get("value"))
    return [
        {**dict(zip(group_keys, key, strict=True)), **distribution(values, ddof)}
        for key, values in sorted(groups.items(), key=lambda item: str(item[0]))
    ]


def _availability(report, path, value):
    name = path.split("/")[-1]
    found = (report.get("tracking") or {}).get("availability", {}).get(name)
    if isinstance(found, dict) and value is None:
        return found.get("reason") or "undefined_in_source"
    return "" if value is not None else "undefined_in_source"


def copy_ledger(source, destination):
    count, seen = 0, set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("wb") as output,
        gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as packed,
    ):
        for row in prediction_rows(source):
            identity = row["event_id"]
            if identity in seen:
                raise ValueError("duplicate prediction event ID")
            seen.add(identity)
            clean = {key: row[key] for key in PREDICTION_FIELDS if key in row}
            packed.write((json.dumps(clean, sort_keys=True, allow_nan=False) + "\n").encode())
            count += 1
    return count


def paired_metrics(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case.coordinates["dataset"], case.coordinates["model"]].append(case)
    results, cohorts = [], []
    for (dataset, model), expected in sorted(groups.items()):
        complete = [c for c in expected if c.status == "passed" and c.prediction]
        cohort = {
            "dataset": dataset,
            "model": model,
            "expected_cases": len(expected),
            "complete_cases_with_ledger": len(complete),
            "missing_cases": [c.key for c in expected if c not in complete],
        }
        common = None
        for case in complete:
            identities = {r["event_id"] for r in prediction_rows(case.prediction)}
            common = identities if common is None else common & identities
        common = common or set()
        cohort.update(
            {
                "evaluated_common_events": len(common),
                "all_expected_cases_available": len(complete) == len(expected),
                "event_id_set_sha256": hashlib.sha256("\n".join(sorted(common)).encode()).hexdigest(),
            }
        )
        cohorts.append(cohort)
        if not common:
            continue
        labels = {}
        for case in complete:
            metrics = Metrics()
            first_seq, last_seq = None, None
            for row in prediction_rows(case.prediction):
                if row["event_id"] not in common:
                    continue
                if row["event_id"] in labels and labels[row["event_id"]] != row["label"]:
                    raise ValueError("paired event ID has inconsistent labels")
                labels[row["event_id"]] = row["label"]
                metrics.add(bool(row["label"]), row["probability"])
                first_seq = row["seq"] if first_seq is None else min(first_seq, row["seq"])
                last_seq = row["seq"] if last_seq is None else max(last_seq, row["seq"])
            values = metrics.values()
            baseline = max(values["positive_rate"], 1 - values["positive_rate"])
            values["majority_accuracy"] = baseline
            for key, value in values.items():
                results.append(
                    {
                        **case.coordinates,
                        "case_key": case.key,
                        "scope": "paired_event_intersection",
                        "metric": key,
                        "value": value,
                        "available": value is not None,
                        "evaluated": metrics.n,
                        "first_seq": first_seq,
                        "last_seq": last_seq,
                        "all_expected_cases_available": cohort["all_expected_cases_available"],
                    }
                )
    return results, cohorts


def intentional_exclusions(plan, cases):
    exclusions = plan.get("exclusions") or []
    if isinstance(exclusions, dict):
        exclusions = [exclusions]
    rows = []
    for exclusion in exclusions:
        if not isinstance(exclusion, dict) or not exclusion.get("reason"):
            raise ValueError("intentional exclusion requires a documented reason")
        seeds = exclusion.get("seeds") or []
        if exclusion.get("family") == "matrix":
            overlap = [
                case.key
                for case in cases
                if all(
                    case.coordinates.get(name) == exclusion[name]
                    for name in ("dataset", "model", "policy")
                    if name in exclusion
                )
                and (not seeds or case.coordinates.get("seed") in seeds)
            ]
            if overlap:
                raise ValueError("a deliberately excluded configuration is still in the execution plan")
        rows.append({**exclusion, "status": "deliberately_excluded", "seeds": list(seeds)})
    return rows


def full_source_coverage_valid(coverage):
    total = coverage.get("source_rows")
    sizes = coverage.get("batch_rows") or []
    ranges = (
        coverage.get("source_row_ranges")
        or coverage.get("included_row_ranges")
        or coverage.get("source_ranges")
        or []
    )
    if (
        coverage.get("full_source") is not True
        or type(total) is not int
        or total <= 0
        or coverage.get("selected_rows") != total
        or len(sizes) != 15
        or len(ranges) != 15
        or any(type(size) is not int or size < 1 for size in sizes)
        or sum(sizes) != total
    ):
        return False
    cursor = 0
    for position, (size, interval) in enumerate(zip(sizes, ranges, strict=True)):
        if isinstance(interval, dict):
            start, stop = interval.get("start"), interval.get("stop")
            if interval.get("batch", position) != position:
                return False
            if "train_rows" in interval and "holdout_rows" in interval:
                if interval["train_rows"] + interval["holdout_rows"] != size:
                    return False
        elif isinstance(interval, list) and len(interval) == 2:
            start, stop = interval
        else:
            return False
        if start != cursor or stop != cursor + size:
            return False
        cursor = stop
    return (
        cursor == total
        and coverage.get("excluded_rows", 0) == 0
        and coverage.get("all_source_rows_included", True) is True
        and coverage.get("offline_rows", sum(sizes[:7])) == sum(sizes[:7])
        and coverage.get("online_rows", sum(sizes[7:])) == sum(sizes[7:])
    )


def replication_coverage(payload, planned):
    rows = []
    for dataset in datasets_in_replication(payload):
        details = dataset.get("trials_detail", dataset.get("repetitions", []))
        observed = [trial.get("seed", trial.get("trial")) for trial in details]
        expected = planned.get("seeds") or dataset.get("seeds") or observed
        expected = list(expected)
        complete = (
            observed != []
            and all(type(seed) is int for seed in observed)
            and len(set(observed)) == len(observed)
            and set(observed) == set(expected)
            and len(observed) == dataset.get("trials", len(observed))
        )
        full_source_required = planned.get("full_source") is True and dataset["dataset"] in {
            "electricity",
            "airlines",
        }
        trial_coverage = []
        for trial in details:
            coverage = trial.get("coverage") or {}
            trial_coverage.append({"seed": trial.get("seed", trial.get("trial")), "coverage": coverage})
            if full_source_required:
                complete &= full_source_coverage_valid(coverage)
        rows.append(
            {
                "dataset": dataset["dataset"],
                "expected_seeds": expected,
                "observed_seeds": observed,
                "declared_trials": dataset.get("trials"),
                "full_source_required": full_source_required,
                "trial_coverage": trial_coverage,
                "complete": bool(complete),
            }
        )
    return rows


def historical_comparison(directory, aggregates, cohorts, exclusions):
    """Describe shared configurations without treating changed protocols as a causal experiment."""
    directory = Path(directory).resolve()
    if (directory / "dados").is_dir():
        directory = directory / "dados"
    source = directory / "metricas-pareadas-medias.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        historical = list(csv.DictReader(stream))
    datasets = {
        "Abrupto": "insects_abrupt_imbalanced",
        "Recorrente": "insects_incremental_abrupt_reoccurring_imbalanced",
    }
    models = {"Logística": "logistic", "Hoeffding": "hoeffding_tree"}
    policies = {
        "ADWIN": "drift",
        "CARA": "cara",
        "Congelado": "none",
        "Online": "online",
        "Oracle": "oracle",
        "UPF": "upf",
    }
    metrics = {
        "f1": "f1_macro",
        "ap": "average_precision",
        "acuracia": "accuracy",
        "recall": "recall",
        "precisao": "precision",
    }
    current = {
        (r["dataset"], r["model"], r["policy"], r["metric"]): r
        for r in aggregates
        if r["scope"] == "paired_event_intersection"
    }
    population = {(r["dataset"], r["model"]): r for r in cohorts}
    results, unmatched = [], []
    for row in historical:
        key = datasets[row["dataset"]], models[row["modelo"]], policies[row["politica"]]
        if not any((*key, metric) in current for metric in metrics.values()):
            deliberate = any(e.get("family") == "matrix" and e.get("policy") == key[2] for e in exclusions)
            unmatched.append(
                {
                    "dataset": key[0],
                    "model": key[1],
                    "policy": key[2],
                    "status": "deliberately_excluded" if deliberate else "not_shared_in_current_evidence",
                }
            )
            continue
        for old_metric, metric in metrics.items():
            now = current.get((*key, metric))
            if now is None:
                continue
            cohort = population[key[:2]]
            results.append(
                {
                    "dataset": key[0],
                    "model": key[1],
                    "policy": key[2],
                    "metric": metric,
                    "historical_mean": float(row[old_metric]),
                    "historical_std": float(row[old_metric + "_dp"]),
                    "historical_repetitions": int(float(row["seeds"])),
                    "historical_ddof": 1,
                    "historical_evaluated_per_run": int(float(row["avaliados"])),
                    "historical_first_seq": int(float(row["passo_inicial"])),
                    "historical_last_seq": int(float(row["passo_final"])),
                    "current_mean": now["mean"],
                    "current_std": now["std"],
                    "current_repetitions": now["n"],
                    "current_ddof": now["ddof"],
                    "current_evaluated_per_run": cohort["evaluated_common_events"],
                    "comparison_scope": "descriptive_only_protocols_and_event_populations_differ",
                }
            )
    coverage = directory / "cobertura.csv"
    counts = {}
    if coverage.is_file():
        with coverage.open(encoding="utf-8-sig", newline="") as stream:
            coverage_rows = list(csv.DictReader(stream))
        counts = {
            "historical_planned_units": sum(int(row["previstas"]) for row in coverage_rows),
            "historical_completed_units": sum(int(row["completas"]) for row in coverage_rows),
            "coverage_source_sha256": sha256(coverage),
        }
    metadata = {
        "source_file": source.name,
        "source_sha256": sha256(source),
        **counts,
        "shared_configurations": len({(r["dataset"], r["model"], r["policy"]) for r in results}),
        "unmatched_historical_configurations": unmatched,
        "interpretation": "Descriptive side-by-side values only. Protocols, training availability, "
        "replay, execution conditions and evaluated populations differ; no causal effect, paired test, "
        "significance or superiority is inferred. Planned coverage is not directly "
        "comparable after deliberate policy exclusion.",
    }
    return results, metadata


def datasets_in_replication(value):
    if isinstance(value, dict) and "dataset" in value:
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from datasets_in_replication(child)
    elif isinstance(value, list):
        for child in value:
            yield from datasets_in_replication(child)


def candidate_reports(value):
    if isinstance(value, dict) and "training_history" in value and "contract" in value:
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from candidate_reports(child)
    elif isinstance(value, list):
        for child in value:
            yield from candidate_reports(child)


def replication_rows(payload, job):
    for dataset in datasets_in_replication(payload):
        protocol = dataset.get("protocol") or job
        for trial in dataset.get("trials_detail", dataset.get("repetitions", [])):
            seed = trial.get("seed", trial.get("trial"))
            for policy, points in trial.get("curves", trial.get("curves_by_policy", {})).items():
                for point in points:
                    param = "kappa" if "kappa" in point else "alpha"
                    context = {
                        "job": job,
                        "protocol": protocol,
                        "dataset": dataset["dataset"],
                        "seed": seed,
                        "policy": policy,
                        "parameter": param,
                        "parameter_value": point.get(param),
                        "point_type": "curve",
                        "parameters": json.dumps(point.get("parameters", {}), sort_keys=True),
                        "schedule": json.dumps(point.get("retrain_points", [])),
                        "served_by": json.dumps(point.get("served_by", [])),
                    }
                    for path, step, value, nested in numeric_leaves(point):
                        if path in {"alpha", "kappa"}:
                            continue
                        yield {
                            **context,
                            "metric": path,
                            "step": step,
                            "value": value,
                            "available": value is not None,
                            "coordinate": json.dumps(nested, sort_keys=True),
                        }
            for policy, summary in trial.get("summary", {}).items():
                for path, step, value, nested in numeric_leaves(summary):
                    yield {
                        "job": job,
                        "protocol": protocol,
                        "dataset": dataset["dataset"],
                        "seed": seed,
                        "policy": policy,
                        "parameter": "",
                        "parameter_value": "",
                        "point_type": "summary",
                        "metric": path,
                        "step": step,
                        "value": value,
                        "available": value is not None,
                        "coordinate": json.dumps(nested, sort_keys=True),
                    }
            for field in (
                "performance_matrix",
                "staleness_matrix",
                "forecast",
                "wrong_alpha",
                "cumulative_strategy_cost",
                "matrices",
            ):
                for path, step, value, nested in numeric_leaves(trial.get(field), field):
                    if field not in trial:
                        continue
                    yield {
                        "job": job,
                        "protocol": protocol,
                        "dataset": dataset["dataset"],
                        "seed": seed,
                        "policy": "diagnostic",
                        "parameter": "",
                        "parameter_value": "",
                        "point_type": "analysis",
                        "metric": path,
                        "step": step,
                        "value": value,
                        "available": value is not None,
                        "coordinate": json.dumps(nested, sort_keys=True),
                    }
        for report in candidate_reports(dataset.get("candidate_tracking", [])):
            for section in METRIC_SECTIONS:
                if section not in report:
                    continue
                for path, step, value, nested in numeric_leaves(report[section], section):
                    yield {
                        "job": job,
                        "protocol": protocol,
                        "dataset": dataset["dataset"],
                        "seed": report["contract"].get("seed"),
                        "policy": "candidate_matrix",
                        "parameter": "",
                        "parameter_value": "",
                        "point_type": "candidate",
                        "metric": path,
                        "step": step,
                        "value": value,
                        "available": value is not None,
                        "coordinate": json.dumps(nested, sort_keys=True),
                    }


def figures(output, paired_aggregate, replication_aggregate, cohorts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generated = []
    coverage = {(row["dataset"], row["model"]): row for row in cohorts}
    datasets = sorted({(r["dataset"], r["model"]) for r in paired_aggregate})
    for dataset, model in datasets:
        selected = [
            r
            for r in paired_aggregate
            if (r["dataset"], r["model"], r["metric"]) == (dataset, model, "f1_macro")
            and r["mean"] is not None
        ]
        if not selected:
            continue
        selected.sort(key=lambda r: r["policy"])
        fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
        bars = axis.bar(
            [r["policy"] + ("*" if r["policy"].startswith("oracle") else "") for r in selected],
            [r["mean"] for r in selected],
            yerr=[r["std"] or 0 for r in selected],
            capsize=4,
        )
        axis.bar_label(bars, labels=[f"n={r['n']}" for r in selected], padding=4, fontsize=8)
        title = f"{dataset} / {model}"
        if not coverage[dataset, model]["all_expected_cases_available"]:
            title += " (cobertura parcial)"
        axis.set(title=textwrap.fill(title, 64), ylabel="F1 macro nos mesmos eventos", ylim=(0, 1.08))
        axis.tick_params(axis="x", rotation=25)
        footnote = "Média e DP amostral entre execuções; fluxo fixo, sem inferência de independência."
        if any(r["std"] is None for r in selected):
            footnote += " DP indisponível quando n=1."
        if any(r["policy"].startswith("oracle") for r in selected):
            footnote += " * Oracle: referência retrospectiva."
        fig.supxlabel(textwrap.fill(footnote, 100), x=0.02, ha="left", fontsize=8)
        target = output / "figures" / f"{dataset}-{model}-paired-f1.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(target.relative_to(output).as_posix())
    for job, dataset in sorted({(r["job"], r["dataset"]) for r in replication_aggregate}):
        selected = [
            r
            for r in replication_aggregate
            if (r["job"], r["dataset"], r["point_type"], r["metric"]) == (job, dataset, "curve", "cost")
            and r["mean"] is not None
        ]
        if not selected:
            continue
        fig, axis = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        for policy in sorted({r["policy"] for r in selected}):
            rows = sorted([r for r in selected if r["policy"] == policy], key=lambda r: r["parameter_value"])
            axis.plot(
                [r["parameter_value"] for r in rows], [r["mean"] for r in rows], marker=".", label=policy
            )
        axis.set(
            title=textwrap.fill(f"{job} / {dataset}", 64),
            xlabel="alpha (unidades do objetivo)",
            ylabel="Custo do objetivo",
        )
        axis.legend(fontsize=8)
        repetitions = sorted({r["n"] for r in selected})
        count = str(repetitions[0]) if len(repetitions) == 1 else "/".join(map(str, repetitions))
        fig.supxlabel(f"Média de n={count} execuções; DP populacional no CSV.", x=0.02, ha="left", fontsize=8)
        target = output / "figures" / f"{job}-{dataset}-cost.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(target.relative_to(output).as_posix())
    return generated


def export(state, output, *, make_plots=True, require_complete=False, historical_insects=None):
    with ExitStack() as resources:
        return _export(
            state,
            output,
            make_plots=make_plots,
            require_complete=require_complete,
            historical_insects=historical_insects,
            resources=resources,
        )


def _export(state, output, *, make_plots, require_complete, historical_insects, resources):
    state, output = Path(state).resolve(), Path(output).resolve()
    if output == state or output.is_relative_to(state / "matrix"):
        raise ValueError("output must not replace campaign input files")
    plan, status = (json_read(state / name) for name in ("campaign-plan.json", "campaign-status.json"))
    cases = cases_from_plan(state, plan, status)
    exclusions = intentional_exclusions(plan, cases)
    complete_cases = [c for c in cases if c.status == "passed"]
    jobs = status.get("replications") or {}
    expected_jobs = plan.get("replications", {}).get("jobs", [])
    planned_names = {
        str(job.get("job_id") or job.get("name") or job.get("id") or job.get("job"))
        for job in expected_jobs
        if isinstance(job, dict)
    }
    if "None" in planned_names:
        raise ValueError("replication plan contains a job without an identity")
    expected_names = planned_names or set(jobs)
    replication_checks = {}
    for name in expected_names:
        item = jobs.get(name, {})
        if item.get("status") == "passed" and item.get("output"):
            path = contained(state, item["output"])
            if path.is_file():
                replication_checks[name] = replication_coverage(
                    json_read(path), plan.get("replications") or {}
                )
    jobs_complete = all(
        jobs.get(name, {}).get("status") == "passed"
        and jobs[name].get("output")
        and contained(state, jobs[name]["output"]).is_file()
        and replication_checks.get(name)
        and all(row["complete"] for row in replication_checks[name])
        for name in expected_names
    )
    complete = (
        len(complete_cases) == len(cases)
        and all(case.prediction is not None for case in complete_cases)
        and jobs_complete
        and status.get("complete") is True
    )
    if require_complete and not complete:
        raise ValueError("campaign is incomplete; refusing a completed-evidence claim")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not (output / "manifest.json").exists():
        raise ValueError("nonempty output does not contain a generated evidence manifest")
    if (output / "manifest.json").exists():
        previous = json_read(output / "manifest.json")
        if previous.get("schema") != SCHEMA or previous.get("campaign_id") != plan.get("campaign_id"):
            raise ValueError("existing evidence belongs to a different campaign or schema")
        if json_read(output / "campaign-plan.json") != portable(plan, state):
            raise ValueError("the evidence execution plan changed; use a new output directory")
    records, original_scores = [], []
    vectors = VectorIndex(output / "vector-index.csv", resources)
    fieldnames = [
        "case_key",
        *COORDINATES,
        "scope",
        "metric",
        "step",
        "value",
        "available",
        "reason",
        "coordinate",
    ]
    stream, metrics_csv = writer(output / "metrics-long.csv", fieldnames)
    try:
        for case in cases:
            record = {"case_key": case.key, **case.coordinates, "status": case.status}
            if case.report is None or case.status != "passed":
                records.append(record)
                continue
            report = json_read(case.report)
            clean = portable(report, state)
            assert_portable(clean)
            target = output / "reports" / f"{case.key}.json.gz"
            payload_encoding = write_payload(target, clean, source_bytes=case.report.stat().st_size)
            csv_report = indexed_csv_view(clean, target.relative_to(output).as_posix(), vectors)
            record.update(
                {
                    "report": target.relative_to(output).as_posix(),
                    "source_report_sha256": sha256(case.report),
                    "retained_report_sha256": sha256(target),
                    "payload_encoding": payload_encoding,
                    "run_key": report.get("run_key"),
                    "data_sha256": (report.get("dataset") or {}).get("sha256"),
                    "code_sha256": (report.get("environment") or {}).get("code_sha256"),
                    "git_revision": (report.get("environment") or {}).get("git_revision"),
                }
            )
            for section in METRIC_SECTIONS:
                if section not in csv_report:
                    continue
                for metric, step, value, context in numeric_leaves(csv_report[section], section):
                    metrics_csv.writerow(
                        {
                            **case.coordinates,
                            "case_key": case.key,
                            "scope": section,
                            "metric": metric,
                            "step": step,
                            "value": value,
                            "available": value is not None,
                            "reason": _availability(report, metric, value),
                            "coordinate": json.dumps(context, sort_keys=True),
                        }
                    )
            for metric, value in (report.get("metrics") or {}).items():
                if value is None or type(value) in (int, float):
                    original_scores.append(
                        {
                            **case.coordinates,
                            "case_key": case.key,
                            "scope": "original_run_scope",
                            "metric": metric,
                            "value": value,
                        }
                    )
            if case.prediction:
                target = output / "predictions" / f"{case.key}.jsonl.gz"
                record["prediction_rows"] = copy_ledger(case.prediction, target)
                record["prediction_ledger"] = target.relative_to(output).as_posix()
                record["source_prediction_sha256"] = sha256(case.prediction)
            records.append(record)
            del csv_report, clean, report
    finally:
        stream.close()

    coverage_fields = ["case_key", *COORDINATES, "status", "run_key", "prediction_rows", "report"]
    stream, table = writer(output / "coverage.csv", coverage_fields)
    with stream:
        table.writerows(records)
    stream, table = writer(
        output / "intentional-exclusions.csv",
        ["family", "dataset", "model", "policy", "seeds", "configurations", "runs", "reason", "status"],
    )
    with stream:
        table.writerows({**row, "seeds": json.dumps(row["seeds"])} for row in exclusions)
    paired, cohorts = paired_metrics(cases)
    stream, table = writer(
        output / "paired-by-seed.csv",
        [
            "case_key",
            *COORDINATES,
            "scope",
            "metric",
            "value",
            "available",
            "evaluated",
            "first_seq",
            "last_seq",
            "all_expected_cases_available",
        ],
    )
    with stream:
        table.writerows(paired)
    group_keys = ["dataset", "model", "policy", "mode", "scope", "metric"]
    aggregates = aggregate(paired + original_scores, group_keys, ddof=1)
    stream, table = writer(
        output / "aggregate-metrics.csv", [*group_keys, "n", "missing", "mean", "std", "ddof", "divisor"]
    )
    with stream:
        table.writerows(aggregates)
    historical_metadata = None
    if historical_insects is not None:
        comparison, historical_metadata = historical_comparison(
            historical_insects, aggregates, cohorts, exclusions
        )
        comparison_fields = [
            "dataset",
            "model",
            "policy",
            "metric",
            "historical_mean",
            "historical_std",
            "historical_repetitions",
            "historical_ddof",
            "historical_evaluated_per_run",
            "historical_first_seq",
            "historical_last_seq",
            "current_mean",
            "current_std",
            "current_repetitions",
            "current_ddof",
            "current_evaluated_per_run",
            "comparison_scope",
        ]
        stream, table = writer(output / "historical-comparison.csv", comparison_fields)
        with stream:
            table.writerows(comparison)
        excluded_count = sum(row.get("runs", row.get("configurations", 0)) for row in exclusions)
        text = (
            "# Comparação descritiva com o registro histórico\n\n"
            "As colunas apresentam as configurações compartilhadas entre os dois registros. "
            "Os protocolos, a disponibilidade dos rótulos, o replay e as populações avaliadas diferem. "
            "A comparação não identifica um efeito causal, significância estatística ou superioridade.\n\n"
            f"Cobertura histórica: {historical_metadata.get('historical_completed_units', 'não informada')} "
            f"de {historical_metadata.get('historical_planned_units', 'não informada')} unidades. "
            f"Cobertura atual: {len(complete_cases)} de {len(cases)} unidades previstas no escopo atual. "
            f"Foram excluídas deliberadamente {excluded_count} unidades, conforme o plano; "
            "essa mudança impede equiparar os denominadores das campanhas.\n\n"
            "Os desvios padrão de INSECTS são amostrais (ddof=1). As seeds repetem execuções de um fluxo "
            "fixo e não equivalem necessariamente a novas realizações independentes. "
            "Valores abaixo arredondados a seis casas; o CSV mantém a precisão disponível.\n\n"
            "| Fluxo | Modelo | Política | Métrica | Histórico: média ± DP (n) | Atual: média ± DP (n) "
            "| Eventos por run: histórico / atual |\n| --- | --- | --- | --- | ---: | ---: | ---: |\n"
        )

        def display(value):
            return "n/d" if value is None else f"{value:.6f}"

        for row in comparison:
            text += (
                f"| {row['dataset']} | {row['model']} | {row['policy']} | {row['metric']} | "
                f"{display(row['historical_mean'])} ± {display(row['historical_std'])} "
                f"({row['historical_repetitions']}) | {display(row['current_mean'])} ± "
                f"{display(row['current_std'])} ({row['current_repetitions']}) | "
                f"{row['historical_evaluated_per_run']} / {row['current_evaluated_per_run']} |\n"
            )
        (output / "historical-comparison.md").write_text(text, encoding="utf-8")

    rep_fields = [
        "job",
        "protocol",
        "dataset",
        "seed",
        "policy",
        "parameter",
        "parameter_value",
        "point_type",
        "metric",
        "step",
        "value",
        "available",
        "parameters",
        "schedule",
        "served_by",
        "coordinate",
    ]
    rep_aggregate_rows, rep_records = [], []
    stream, table = writer(output / "replication-by-seed.csv", rep_fields)
    with stream:
        for job in sorted(set(jobs) | expected_names):
            item = jobs.get(job, {})
            record = {
                "job": job,
                "status": item.get("status", "pending"),
                "datasets": replication_checks.get(job, []),
            }
            if item.get("status") == "passed" and item.get("output"):
                path = contained(state, item["output"])
                payload = json_read(path)
                clean = portable(payload, state)
                assert_portable(clean)
                target = output / "replications" / f"{job}.json.gz"
                payload_encoding = write_payload(target, clean, source_bytes=path.stat().st_size)
                csv_payload = indexed_csv_view(clean, target.relative_to(output).as_posix(), vectors)
                record.update(
                    {
                        "artifact": target.relative_to(output).as_posix(),
                        "source_sha256": sha256(path),
                        "payload_encoding": payload_encoding,
                    }
                )
                for row in replication_rows(csv_payload, job):
                    table.writerow(row)
                    if row["point_type"] in {"curve", "summary"}:
                        rep_aggregate_rows.append(row)
                    del row
                # Candidate/analysis points have already reached the CSV. Do not
                # retain their rows or a previous job's complete JSON payload.
                del payload, clean, csv_payload
            rep_records.append(record)
    rep_keys = [
        "job",
        "protocol",
        "dataset",
        "policy",
        "parameter",
        "parameter_value",
        "point_type",
        "metric",
        "step",
    ]
    rep_aggregates = aggregate(rep_aggregate_rows, rep_keys, ddof=0)
    stream, table = writer(
        output / "replication-aggregate.csv", [*rep_keys, "n", "missing", "mean", "std", "ddof", "divisor"]
    )
    with stream:
        table.writerows(rep_aggregates)
    generated_figures = (
        figures(
            output,
            [r for r in aggregates if r["scope"] == "paired_event_intersection"],
            rep_aggregates,
            cohorts,
        )
        if make_plots
        else []
    )
    for name, payload in [("campaign-plan.json", plan), ("campaign-status.json", status)]:
        clean = portable(payload, state)
        assert_portable(clean)
        (output / name).write_bytes(json_bytes(clean))
    catalogs = {
        sha256(case.report.parent / "metric-definitions.json"): case.report.parent / "metric-definitions.json"
        for case in complete_cases
        if (case.report.parent / "metric-definitions.json").is_file()
    }
    if len(catalogs) > 1:
        raise ValueError("metric definition catalog changed within one campaign")
    if catalogs:
        (output / "metric-definitions.json").write_bytes(next(iter(catalogs.values())).read_bytes())
    # Flush the streaming catalog before sizing, compressing and hashing it.
    resources.close()
    table_files = pack_large_tables(output)
    manifest = {
        "schema": SCHEMA,
        "campaign_id": plan.get("campaign_id"),
        "complete": complete,
        "expected_cases": len(cases),
        "completed_cases": len(complete_cases),
        "cases": records,
        "replications": rep_records,
        "intentional_exclusions": exclusions,
        "seed_contract": {
            "matrix_expected": sorted({case.coordinates["seed"] for case in cases}),
            "matrix_observed_complete": sorted({case.coordinates["seed"] for case in complete_cases}),
            "replications_expected": plan.get("replications", {}).get("seeds"),
        },
        "historical_comparison": historical_metadata,
        "paired_cohorts": cohorts,
        "figures": generated_figures,
        "table_files": table_files,
        "representation": {
            "version": 2,
            "scientific_payload_schema": "unchanged; complete reports and replications retained",
            "payload_encoding": {
                "large_source_threshold_bytes": LARGE_PAYLOAD_BYTES,
                "large_reports_replications": "sorted compact UTF-8 JSON + deterministic gzip level 6",
                "small_reports_replications": "sorted pretty UTF-8 JSON + deterministic gzip level 9",
                "plans_manifest_metadata": "pretty UTF-8 JSON",
                "per_artifact_details": "cases[].payload_encoding and replications[].payload_encoding",
            },
            "tabular_scalars_series_matrices": "retained in the metric CSVs",
            "table_files_contract": {
                "version": 2,
                "single_file": "relative filename string",
                "sharded_table": "descriptor with format=csv+gzip-shards and an ordered shards list",
                "reconstruction": "read parts in listed order; keep the first CSV header and skip subsequent "
                "repeated headers; use UTF-8 and the canonical Excel CSV dialect",
                "source_row_range": "zero-based, half-open data-row positions excluding the repeated header",
                "shard_trigger_compressed_bytes": TABLE_SHARD_TRIGGER_BYTES,
                "max_shard_compressed_bytes": TABLE_SHARD_MAX_BYTES,
                "shard_bound": "each compressed part is strictly smaller than max_shard_compressed_bytes",
            },
            "indexed_dense_array_fields": sorted(DENSE_ARRAY_FIELDS),
            "vector_index": table_files["vector-index.csv"],
            "indexed_arrays": vectors.count,
            "json_pointer": "RFC 6901, relative to the complete artifact JSON root",
            "vector_hash": "SHA-256 of UTF-8 JSON array with sort_keys=True, ensure_ascii=False, "
            "separators=(',',':'), allow_nan=False",
            "array_values": "stored in full at each cataloged artifact/pointer location",
            "numeric_leaves_default": "unchanged; dense-vector references are explicit export views only",
        },
        "retention": "metrics_reports_prediction_ledgers_and_provenance_only_no_model_or_checkpoint",
        "exporter": {
            "script_sha256": sha256(Path(__file__)),
            "metric_implementation_sha256": sha256(inspect.getfile(Metrics)),
        },
        "normalization": "Machine paths and connection secrets omitted; numeric evidence unchanged. "
        "source_report_sha256 identifies original bytes; retained_report_sha256 identifies gzip bytes.",
        "metric_contract": {
            "paired": "core.metrics.Metrics on common event_id intersection per dataset/model",
            "classification_threshold": 0.5,
            "f1_macro": "arithmetic mean of both binary class F1",
            "average_precision": "exact noninterpolated ranking with tied scores grouped",
            "binary_seed_std": "sample standard deviation, ddof=1; null when n<=1",
            "replication_std": "population standard deviation, ddof=0, as the replication protocol",
            "fixed_stream": "Seeds repeat executions; they need not produce independent stream realizations.",
            "cost": "Scenario estimates/objective units remain distinct from invoiced expenditure.",
        },
    }
    forbidden = [
        p
        for p in output.rglob("*")
        if p.is_file() and (p.suffix.lower() in MODEL_SUFFIXES or "checkpoint" in p.name.lower())
    ]
    if forbidden:
        raise ValueError("model or checkpoint file found in evidence output")
    manifest["files"] = [
        {"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "SHA256SUMS"}
    ]
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    sums = [
        f"{sha256(p)}  {p.relative_to(output).as_posix()}"
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--historical-insects",
        type=Path,
        help="Read-only Week 10 INSECTS archive or its dados directory for a descriptive table",
    )
    args = parser.parse_args()
    result = export(
        args.state,
        args.output,
        make_plots=not args.no_plots,
        require_complete=args.require_complete,
        historical_insects=args.historical_insects,
    )
    print(
        json.dumps({key: result[key] for key in ("schema", "complete", "expected_cases", "completed_cases")})
    )


if __name__ == "__main__":
    main()
