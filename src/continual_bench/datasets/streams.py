"""Ordered stream datasets for the batch-prequential benchmark (Strategy + Registry).

The benchmark consumes a prepared, chronologically sorted Parquet file
(``event_time, label, features…``) plus a JSON manifest. Each provider knows
how to prepare and validate that pair for its datasets:

* ``ZiaStreamProvider`` – the pinned Zia Data Labs streams (fraud, social, ecommerce);
* ``CatalogStreamProvider`` – every classification dataset of the project catalog
  (``datasets.catalog``), materialized deterministically into the same format.

The evaluator only talks to this module, so adding a dataset means registering
a provider, never editing the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from continual_bench.infrastructure.storage import atomic_write

from . import zia

FORMAT = zia.FORMAT
SCOPES = zia.SCOPES


@dataclass(frozen=True)
class StreamDataset:
    """What the benchmark needs to know about a dataset, independent of its origin."""

    name: str
    provider: str
    classes: tuple[str, ...]
    model: str
    block_rows: dict[str, int]
    model_params: dict = field(default_factory=dict)

    @property
    def class_count(self) -> int:
        return len(self.classes)


class StreamProvider(Protocol):
    name: str

    def names(self) -> tuple[str, ...]: ...

    def spec(self, dataset: str) -> StreamDataset: ...

    def prepare(self, root, dataset: str, scope: str) -> dict: ...

    def load_manifest(self, root, dataset: str, scope: str) -> tuple[Path, dict]: ...


def sha256_file(path) -> str:
    return zia.sha256_file(path)


def input_paths(root, dataset, scope):
    folder = Path(root) / dataset
    return folder / f"{scope}.parquet", folder / f"{scope}.json"


def default_model(classes: int) -> str:
    return "xgboost" if classes == 2 else "multinomial_logistic"


class ZiaStreamProvider:
    """The pinned Zia streams; preparation and validation stay in ``datasets.zia``."""

    name = "zia"
    BLOCK_ROWS = {"smoke": 1000, "full": 250_000}
    # Class imbalance of the pinned binary streams (formerly hard-coded in the booster).
    MODEL_PARAMS = {"fraud": {"scale_pos_weight": 39.0}, "ecommerce": {"scale_pos_weight": 10.0}}

    def names(self):
        return tuple(zia.DATASETS)

    def spec(self, dataset):
        source = zia.DATASETS[dataset]
        return StreamDataset(dataset, self.name, tuple(source["classes"]), source["model"],
                             dict(self.BLOCK_ROWS), dict(self.MODEL_PARAMS.get(dataset, {})))

    def prepare(self, root, dataset, scope):
        return zia.prepare(root, dataset, scope)

    def load_manifest(self, root, dataset, scope):
        return zia.load_manifest(root, dataset, scope)


class CatalogStreamProvider:
    """Any classification dataset of ``datasets.catalog`` as an ordered Parquet stream.

    Samples keep their catalog order (timestamp, then sequence). Numeric features
    are kept, booleans become 0/1, nominal values a stable CRC32 code and vector
    features one column per component, so every catalog family maps onto the
    numeric matrix the stream models expect.
    """

    name = "catalog"
    EXAMPLES = {"smoke": 2_000, "full": 10_000}
    BLOCK_ROWS = {"smoke": 200, "full": 1_000}
    SEED = 42

    def __init__(self, cache="state/datasets", real_directory=None, *, download=False):
        self.cache = cache
        self.real_directory = real_directory
        self.download = download

    def names(self):
        from .catalog import CATALOG

        return tuple(spec.name for spec in CATALOG if spec.task in ("binary", "multiclass")
                     and spec.prepare_kind != "none")

    def _source(self, dataset, scope):
        from .catalog import dataset_spec, load_dataset

        spec = dataset_spec(dataset)
        examples = self.EXAMPLES[scope]
        source = load_dataset(dataset, self.cache, examples=examples, seed=self.SEED,
                              download=self.download, real_directory=self.real_directory)
        return spec, source

    def spec(self, dataset, classes=None):
        if dataset not in self.names():
            raise ValueError(f"catalog dataset {dataset} is not a classification stream")
        if classes is None:
            classes = self._declared_classes(dataset)
        classes = tuple(str(label) for label in classes)
        return StreamDataset(dataset, self.name, classes, default_model(len(classes)), dict(self.BLOCK_ROWS))

    def _declared_classes(self, dataset):
        from .catalog import dataset_spec

        if dataset_spec(dataset).task == "binary":
            return ("0", "1")
        raise LookupError("multiclass classes are declared by the prepared manifest")

    def prepare(self, root, dataset, scope):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        path, manifest_path = input_paths(root, dataset, scope)
        if manifest_path.exists():
            return self.load_manifest(root, dataset, scope)[1]
        spec, source = self._source(dataset, scope)
        samples = sorted(source.samples_iter(), key=lambda sample: (sample.timestamp, sample.seq))
        if not samples:
            raise ValueError(f"catalog dataset {dataset} produced no samples")
        declared = (source.manifest() or {}).get("labels")
        labels = [str(label) for label in declared] if declared else sorted({str(s.y) for s in samples})
        index = {label: position for position, label in enumerate(labels)}
        rows = [_encode(sample.x) for sample in samples]
        features = sorted({name for row in rows for name in row})
        matrix = np.zeros((len(rows), len(features)), dtype=np.float32)
        column = {name: position for position, name in enumerate(features)}
        for i, row in enumerate(rows):
            for name, value in row.items():
                matrix[i, column[name]] = value
        times = np.array([int(sample.timestamp) for sample in samples], dtype=np.int64)
        y = np.array([index[str(sample.y)] for sample in samples], dtype=np.uint8)
        names = [_column_name(name) for name in features]
        schema = pa.schema([("event_time", pa.int64()), ("label", pa.uint8()),
                            *((name, pa.float32()) for name in names)])
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent, prefix=".prepare-") as temp:
            output = Path(temp) / "prepared.parquet"
            table = pa.Table.from_arrays(
                [pa.array(times), pa.array(y), *(pa.array(matrix[:, i]) for i in range(len(names)))],
                schema=schema,
            )
            pq.write_table(table, output, compression="zstd", row_group_size=65_536)
            manifest = {
                "format": FORMAT, "provider": self.name, "dataset": dataset, "scope": scope,
                "source": self._source_entry(spec, labels, source),
                "license": "see dataset source", "rows": len(samples), "features": names,
                "sha256": sha256_file(output), "bytes": output.stat().st_size,
            }
            output.replace(path)
            atomic_write(manifest_path, json.dumps(manifest, indent=2).encode())
        return manifest

    def _source_entry(self, spec, labels, source):
        upstream = hashlib.sha256(
            json.dumps(source.manifest() or {}, sort_keys=True, default=str).encode()
        ).hexdigest()
        return {
            "catalog": spec.name, "family": spec.family, "task": spec.task, "classes": labels,
            "model": default_model(len(labels)), "examples": self.EXAMPLES, "seed": self.SEED,
            # Revision of the generated stream: the catalog source manifest digest.
            "revision": f"catalog-{upstream[:12]}", "upstream_sha256": upstream,
        }

    def load_manifest(self, root, dataset, scope):
        from pyarrow.parquet import ParquetFile

        path, manifest_path = input_paths(root, dataset, scope)
        manifest = json.loads(manifest_path.read_text())
        with ParquetFile(path) as source:
            names = source.schema_arrow.names
            if (manifest.get("format") != FORMAT or manifest.get("provider") != self.name
                    or manifest["dataset"] != dataset or manifest["scope"] != scope
                    or source.metadata.num_rows != manifest["rows"]
                    or manifest["bytes"] != path.stat().st_size
                    or names != ["event_time", "label", *manifest["features"]]):
                raise ValueError(f"prepared input contract mismatch: {path}")
        return path, manifest


def _column_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "_-." else "_" for char in str(name)) or "feature"


def _encode(x) -> dict[str, float]:
    """Flatten one feature mapping into finite numeric columns."""
    values: dict[str, float] = {}

    def add(name, value):
        if isinstance(value, bool):
            values[name] = float(value)
        elif isinstance(value, (int, float)):
            values[name] = float(value) if math.isfinite(value) else 0.0
        elif isinstance(value, str):
            values[name] = float(zlib.crc32(value.encode()) % 65_536)
        elif value is None:
            values[name] = 0.0
        elif hasattr(value, "tolist"):
            add(name, value.tolist())
        elif isinstance(value, (list, tuple)):
            for position, item in enumerate(value):
                add(f"{name}_{position}", item)
        else:
            raise TypeError(f"unsupported feature type for {name}: {type(value).__name__}")

    items = x.items() if isinstance(x, dict) else enumerate(x)
    for name, value in items:
        add(str(name), value)
    return values


class StreamRegistry:
    """Resolve a dataset name to its provider; Zia names win over catalog names."""

    def __init__(self, providers):
        self.providers = list(providers)

    def provider(self, dataset) -> StreamProvider:
        for provider in self.providers:
            if dataset in provider.names():
                return provider
        raise ValueError(f"unknown stream dataset: {dataset}")

    def names(self) -> tuple[str, ...]:
        seen = []
        for provider in self.providers:
            seen.extend(name for name in provider.names() if name not in seen)
        return tuple(seen)

    def spec(self, dataset, manifest=None) -> StreamDataset:
        provider = self.provider(dataset)
        if manifest is not None and isinstance(provider, CatalogStreamProvider):
            return provider.spec(dataset, manifest["source"]["classes"])
        return provider.spec(dataset)

    def prepare(self, root, dataset, scope) -> dict:
        return self.provider(dataset).prepare(root, dataset, scope)

    def load_manifest(self, root, dataset, scope) -> tuple[Path, dict]:
        return self.provider(dataset).load_manifest(root, dataset, scope)


STREAMS = StreamRegistry([ZiaStreamProvider(), CatalogStreamProvider()])


def stream_names() -> tuple[str, ...]:
    return STREAMS.names()


def stream_spec(dataset, manifest=None) -> StreamDataset:
    return STREAMS.spec(dataset, manifest)


def prepare_stream(root, dataset, scope) -> dict:
    return STREAMS.prepare(root, dataset, scope)


def load_stream_manifest(root, dataset, scope) -> tuple[Path, dict]:
    return STREAMS.load_manifest(root, dataset, scope)


def dataset_model_params(dataset) -> dict:
    """Model defaults a provider declares for ``dataset`` (empty when unknown)."""
    try:
        provider = STREAMS.provider(dataset)
    except ValueError:
        return {}
    return dict(provider.spec(dataset).model_params) if isinstance(provider, ZiaStreamProvider) else {}
