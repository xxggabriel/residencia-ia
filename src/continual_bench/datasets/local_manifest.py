"""Explicit local-data admission (INSECTS, Gas, prepared CL inputs); never downloads data.

Prepared real-data manifests describe an exact local variant, including its order and rights.
"""

import csv
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from continual_bench.core.contracts import Event, TaskSample

from .arff import file_sha256, iter_arff
from .base import stable_identity, validate_bounds


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataFile(_Strict):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(ge=1)
    batch: str | None = None


class LocalDatasetManifest(_Strict):
    schema_version: Literal[1] = 1
    name: str
    variant: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: str = Field(pattern=r"^https://")
    license: str = Field(min_length=1)
    license_reference: str = Field(pattern=r"^https://")
    access_approved: bool
    format: Literal["csv", "arff", "libsvm", "jsonl"]
    files: tuple[DataFile, ...] = Field(min_length=1)
    features: dict[str, Literal["numeric", "categorical"]] = Field(min_length=1)
    target: str
    labels: tuple[str, ...] = ()
    task: Literal["binary", "multiclass", "regression"]
    positive_label: str | None = None
    order: Literal["published", "batch", "timestamp", "artificial"]
    order_description: str = Field(min_length=1)
    timestamp_column: str | None = None
    split_column: str | None = None
    group_column: str | None = None
    changepoints: tuple[int, ...] | None = None
    preparation: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_contract(self):
        if not self.access_approved:
            raise ValueError("dataset license/access approval is required")
        if self.task == "binary" and (len(set(self.labels)) != 2 or self.positive_label not in self.labels):
            raise ValueError("binary source requires exactly two labels and an explicit positive label")
        if self.task == "multiclass" and len(set(self.labels)) < 2:
            raise ValueError("multiclass source requires declared labels")
        if self.task == "regression" and (self.labels or self.positive_label):
            raise ValueError("regression must preserve its numeric target")
        metadata = [self.target, self.timestamp_column, self.split_column, self.group_column]
        if set(self.features) & {k for k in metadata if k is not None}:
            raise ValueError("target and metadata must not leak into features")
        if len({f.path for f in self.files}) != len(self.files):
            raise ValueError("duplicate dataset files")
        if len({f.sha256 for f in self.files}) != len(self.files):
            raise ValueError("duplicate dataset file content would repeat sample identities")
        if self.order == "timestamp" and not self.timestamp_column:
            raise ValueError("timestamp order requires a timestamp column")
        if self.order == "batch" and any(f.batch is None for f in self.files):
            raise ValueError("batch order requires a batch ID for every file")
        if self.changepoints is not None and (tuple(sorted(set(self.changepoints))) != self.changepoints
                                             or any(p < 1 for p in self.changepoints)):
            raise ValueError("invalid annotated changepoints")
        return self


class LocalManifestSource:
    """INSECTS/Gas/prepared CL inputs with content hashes and unchanged row order.

    `manifest_path` points to a caller-reviewed exact variant. Raw files must be
    present beneath the manifest directory. No inferred license or drift truth.
    """

    def __init__(self, name, samples, seed, label_delay, epoch, *, manifest_path):
        validate_bounds(samples, seed, label_delay, epoch)
        self.manifest_path = Path(manifest_path)
        self.spec = LocalDatasetManifest.model_validate_json(self.manifest_path.read_text())
        if self.spec.name != name:
            raise ValueError("dataset name does not match admitted manifest")
        if samples > sum(f.rows for f in self.spec.files):
            raise ValueError("requested prefix exceeds admitted rows")
        self.paths = []
        root = self.manifest_path.parent.resolve()
        for entry in self.spec.files:
            if Path(entry.path).is_absolute():
                raise ValueError("dataset path must be relative to the local cache directory")
            path = (root / entry.path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("dataset path must stay inside the local cache directory")
            if not path.is_file() or file_sha256(path) != entry.sha256:
                raise ValueError("dataset file absent or SHA-256 mismatch")
            self.paths.append(path)
        self.name, self.samples, self.seed = name, samples, seed
        self.label_delay, self.epoch = label_delay, epoch
        self.version = stable_identity({"manifest": self.spec.model_dump(mode="json"), "samples": samples,
                                  "epoch": epoch, "label_delay": label_delay})

    def _rows(self, path):
        if self.spec.format == "arff":
            yield from iter_arff(path)
            return
        with path.open(encoding="utf-8", newline="") as stream:
            if self.spec.format == "csv":
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError("CSV requires unique column names")
                yield from reader
            elif self.spec.format == "jsonl":
                for line in stream:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL requires an object for each row")
                    yield row
            else:
                for line in stream:
                    fields = line.split()
                    if not fields:
                        raise ValueError("empty LIBSVM row")
                    row = {self.spec.target: fields[0]}
                    for field in fields[1:]:
                        key, value = field.split(":", 1)
                        if key in row:
                            raise ValueError("duplicate LIBSVM feature")
                        row[key] = value
                    yield row

    def samples_iter(self, *, full_validation=False):
        seq, last_timestamp = 0, -1
        metadata = {self.spec.target, self.spec.timestamp_column,
                    self.spec.split_column, self.spec.group_column}
        expected = set(self.spec.features) | (metadata - {None})
        for entry, path in zip(self.spec.files, self.paths):
            if file_sha256(path) != entry.sha256:
                raise ValueError("dataset SHA-256 changed after admission")
            count = 0
            for raw in self._rows(path):
                if count >= entry.rows:
                    raise ValueError("dataset file row count exceeds manifest")
                if set(raw) != expected:
                    raise ValueError("dataset row does not match declared schema")
                x = {}
                for key, dtype in self.spec.features.items():
                    value = raw[key]
                    if value is None or value == "" or value == "?":
                        raise ValueError("missing feature")
                    if dtype == "numeric":
                        value = float(value)
                        if not math.isfinite(value):
                            raise ValueError("non-finite feature")
                    else:
                        value = str(value)
                    x[key] = value
                y = (float(raw[self.spec.target]) if self.spec.task == "regression"
                     else str(raw[self.spec.target]))
                if self.spec.task == "regression":
                    if not math.isfinite(y):
                        raise ValueError("non-finite target")
                elif y not in self.spec.labels:
                    raise ValueError("undeclared target label")
                stamp = str(raw[self.spec.timestamp_column]) if self.spec.timestamp_column else None
                if stamp is not None and not stamp.isdecimal():
                    raise ValueError("source timestamp must be a nonnegative integer")
                timestamp = int(stamp) if stamp is not None else self.epoch + seq
                if timestamp < last_timestamp or timestamp < 0:
                    raise ValueError("dataset timestamp order decreased")
                last_timestamp = timestamp
                split = str(raw[self.spec.split_column]) if self.spec.split_column else "train"
                if split not in {"train", "validation", "test"}:
                    raise ValueError("invalid split")
                group = str(raw[self.spec.group_column]) if self.spec.group_column else entry.batch
                sample = TaskSample(stable_identity([entry.sha256, count]), seq, timestamp,
                                    timestamp + self.label_delay, x, y, split=split, group=group)
                seq, count = seq + 1, count + 1
                yield sample
                if seq >= self.samples and not full_validation:
                    return
            if count != entry.rows:
                raise ValueError("dataset file row count differs from manifest")
        if seq < self.samples:
            raise ValueError("dataset has fewer samples than requested")

    def validate_all(self):
        """Explicit full schema, row-count and temporal admission before campaigns."""
        rows = sum(1 for _ in self.samples_iter(full_validation=True))
        return {"version": self.version, "rows": rows, "raw_sha256": [f.sha256 for f in self.spec.files]}

    def events(self):
        if self.spec.task != "binary":
            raise ValueError("binary engine cannot consume multiclass/regression; use samples_iter()")
        previous = -1
        for sample in self.samples_iter():
            if sample.timestamp <= previous:
                raise ValueError("binary engine requires strictly increasing timestamps")
            if sample.split != "train":
                raise ValueError("binary prequential stream must not mix held-out splits")
            previous = sample.timestamp
            yield Event(sample.sample_id, sample.seq, sample.timestamp, sample.available_at,
                        sample.x, sample.y == self.spec.positive_label, None)

    def manifest(self):
        return {**self.spec.model_dump(mode="json"), "adapter": "local-manifest-v1", "version": self.version,
                "source_version": self.spec.version, "samples": self.samples, "requested_seed": self.seed,
                "source_changepoints": self.spec.changepoints,
                "changepoints": ([p for p in self.spec.changepoints if p < self.samples]
                                 if self.spec.changepoints is not None else None),
                "seed_ignored": True, "cache": str(self.manifest_path.parent.resolve()),
                "timestamp_kind": ("declared_source_integer" if self.spec.timestamp_column
                                   else "simulated_ordinal"),
                "label_delay_seconds": self.label_delay, "label_delay_is_simulated": True}
