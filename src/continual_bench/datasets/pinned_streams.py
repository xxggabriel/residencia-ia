"""Full, ordered preparation of pinned binary real streams without fitting transforms.

The ordinal replay clock is deliberately simulated. Labels are released at the
end of each evaluation block by the batch-prequential evaluator; these files do
not claim that the original examples carried known collection/availability times.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from continual_bench.infrastructure.storage import atomic_write

from . import openml, real_streams, zia
from .arff import _iter_rows, file_sha256, read_schema

PREPARATION = "pinned-real-batch-stream-v1"
SMOKE_ROWS = 2_000
WRITE_ROWS = 8_192
PINNED_NAMES = (*openml.REAL_SOURCES, *real_streams.INSECTS_BINARY_VARIANTS)


def source_contract(dataset):
    """Only immutable public source metadata; never local paths or observed labels."""
    if dataset in openml.REAL_SOURCES:
        spec = openml.REAL_SOURCES[dataset]
        return {
            "url": spec["url"],
            "raw_sha256": spec["sha256"],
            "raw_rows": spec["rows"],
            "positive_source_label": spec["positive"],
            "target": spec["target"],
            "normalization": spec["normalization"],
            "license": spec["license"],
            "license_note": "Exact OpenML catalog label; not an SPDX grant",
        }
    if dataset in real_streams.INSECTS_BINARY_VARIANTS:
        spec = real_streams.SPECS[dataset]
        return {
            "url": spec["url"],
            "raw_sha256": spec["expected_sha256"],
            "raw_rows": spec["expected_rows"],
            "target_transformation": spec["target_transformation"],
            "positive_source_label": spec["positive_label"],
            "changepoints": list(spec["changepoints"]),
            "drift_annotations_usage": "evaluation only; never a feature or policy input",
            "source_version": spec["source_version"],
            "license": spec["license"],
            "license_reference": spec["license_reference"],
            "license_note": spec["license_note"],
        }
    raise ValueError(f"unregistered pinned real stream: {dataset}")


def expected_rows(dataset, scope):
    if scope not in zia.SCOPES:
        raise ValueError(f"unsupported scope: {scope}")
    total = source_contract(dataset)["raw_rows"]
    return total if scope == "full" else min(total, SMOKE_ROWS)


class PinnedRealStreamProvider:
    """Bounded-memory conversion with unchanged published row order and fixed classes."""

    name = "pinned_real"

    def __init__(self, real_directory=None):
        self.real_directory = real_directory

    def names(self):
        return PINNED_NAMES

    def spec(self, dataset):
        from .streams import StreamDataset

        rows = source_contract(dataset)["raw_rows"]
        # Small sources still need an offline phase and multiple evaluation blocks.
        block_rows = {"smoke": 200, "full": max(1_000, math.ceil(rows * 0.9 / 24))}
        return StreamDataset(dataset, self.name, ("0", "1"), "xgboost", block_rows)

    def _raw_path(self, dataset):
        root = Path(self.real_directory or os.environ.get("BENCH_DATASETS", "/state/datasets"))
        filename = (
            f"{dataset}.arff" if dataset in openml.REAL_SOURCES else real_streams.SPECS[dataset]["filename"]
        )
        candidates = (root / filename, root / dataset / filename)
        for path in candidates:
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("symlinked pinned source is not allowed")
            if path.is_file():
                return path
        raise FileNotFoundError(f"pinned {dataset} source absent; prepare the original sources first")

    def _rows(self, dataset, path):
        if dataset in openml.REAL_SOURCES:
            spec = openml.REAL_SOURCES[dataset]
            admission = openml.validate_source(path, dataset)
            fields = [field for field in admission["schema"] if field["name"] != spec["target"]]
            features = [field["name"] for field in fields]
            nominal = {
                f["name"]: {v: i for i, v in enumerate(f["values"])} for f in fields if f["type"] == "nominal"
            }

            def rows():
                with path.open(encoding="utf-8", newline="") as stream:
                    schema = read_schema(stream)
                    for seq, row in enumerate(_iter_rows(stream, schema)):
                        x = [nominal[name][row[name]] if name in nominal else row[name] for name in features]
                        yield seq, int(row[spec["target"]] == spec["positive"]), x

            return features, {"nominal": nominal, "numeric": "identity cast to float32"}, rows()
        spec = real_streams.SPECS[dataset]
        if file_sha256(path) != spec["expected_sha256"]:
            raise ValueError(f"Pinned {dataset} SHA-256 mismatch")
        real_streams._validate(dataset, path)
        features = [f"f{i + 1}" for i in range(spec["features"])]
        return (
            features,
            {"numeric": "identity cast to float32"},
            (
                (seq, int(label == spec["positive_label"]), [x[name] for name in features])
                for seq, x, label, _batch, _member in real_streams._insects_rows(
                    path, filename=spec["filename"]
                )
            ),
        )

    def prepare(self, root, dataset, scope):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path, manifest_path = zia.input_paths(root, dataset, scope)
        if manifest_path.exists():
            return self.load_manifest(root, dataset, scope)[1]
        if path.exists() or path.is_symlink() or manifest_path.is_symlink():
            raise ValueError("incomplete or symlinked prepared stream pair")
        raw = self._raw_path(dataset)
        features, encodings, rows = self._rows(dataset, raw)
        total = expected_rows(dataset, scope)
        schema = pa.schema(
            [("event_time", pa.int64()), ("label", pa.uint8()), *((f, pa.float32()) for f in features)]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent, prefix=".prepare-") as temp:
            output = Path(temp) / "prepared.parquet"
            chunk, written = [], 0

            def write(writer):
                nonlocal written
                columns = list(zip(*chunk, strict=True))
                xs = list(zip(*columns[2], strict=True))
                arrays = [pa.array(columns[0], type=pa.int64()), pa.array(columns[1], type=pa.uint8())]
                arrays.extend(pa.array(values, type=pa.float32()) for values in xs)
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
                written += len(chunk)
                chunk.clear()

            with pq.ParquetWriter(output, schema, compression="zstd") as writer:
                for seq, label, x in rows:
                    if seq == total:
                        break
                    if seq != written + len(chunk):
                        raise ValueError("published source offsets are not contiguous")
                    chunk.append((seq, label, x))
                    if len(chunk) == WRITE_ROWS:
                        write(writer)
                if chunk:
                    write(writer)
            if written != total:
                raise ValueError("prepared row count differs from the requested complete source/prefix")
            if file_sha256(raw) != source_contract(dataset)["raw_sha256"]:
                raise ValueError("pinned raw source changed during preparation")
            manifest = {
                "format": zia.FORMAT,
                "provider": self.name,
                "preparation_version": PREPARATION,
                "dataset": dataset,
                "scope": scope,
                "source": {**source_contract(dataset), "classes": ["0", "1"], "model": "xgboost"},
                "selection": {
                    "kind": "complete_source" if scope == "full" else "published_prefix",
                    "seq_range": [0, total],
                },
                "timestamp_kind": "simulated_ordinal_replay; not measured event time",
                "order": "published file order unchanged",
                "real_event_timestamps_known": False,
                "real_label_availability_known": False,
                "label_availability": "all labels released at end of completed batch-prequential block",
                "feature_encoding": encodings,
                "rows": written,
                "features": features,
                "sha256": file_sha256(output),
                "bytes": output.stat().st_size,
            }
            output.replace(path)
            atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2).encode())
        return manifest

    def load_manifest(self, root, dataset, scope):
        import pyarrow.parquet as pq

        path, manifest_path = zia.input_paths(root, dataset, scope)
        if path.is_symlink() or manifest_path.is_symlink() or path.parent.is_symlink():
            raise ValueError("symlinked prepared input is not allowed")
        manifest = json.loads(manifest_path.read_text())
        wanted_source = {**source_contract(dataset), "classes": ["0", "1"], "model": "xgboost"}
        with pq.ParquetFile(path) as source:
            if (
                manifest.get("format") != zia.FORMAT
                or manifest.get("provider") != self.name
                or manifest.get("preparation_version") != PREPARATION
                or manifest.get("dataset") != dataset
                or manifest.get("scope") != scope
                or manifest.get("source") != wanted_source
                or manifest.get("rows") != expected_rows(dataset, scope)
                or source.metadata.num_rows != manifest["rows"]
                or manifest.get("bytes") != path.stat().st_size
                or source.schema_arrow.names != ["event_time", "label", *manifest["features"]]
                or file_sha256(path) != manifest.get("sha256")
            ):
                raise ValueError("prepared pinned real stream contract mismatch")
        return path, manifest
