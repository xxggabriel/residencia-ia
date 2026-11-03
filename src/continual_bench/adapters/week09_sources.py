"""Bounded drifting streams and explicit local-data admission; never downloads data.

Binary Event is deliberately not reused for multiclass or regression. Prepared
real-data manifests describe an exact local variant, including its order and rights.
"""

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from river.datasets import synth

from continual_bench.domain.contracts import Event

from .real_sources import file_sha256, iter_arff


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bounded(samples, seed, label_delay, epoch):
    if not 1 <= samples <= 100_000 or not 0 <= seed < 2**31 or label_delay < 0 or epoch < 0:
        raise ValueError("invalid bounded source parameters")


@dataclass(frozen=True)
class TaskSample:
    sample_id: str
    seq: int
    timestamp: int
    available_at: int
    x: dict[str, Any]
    y: str | float
    task_id: int | None = None
    split: str = "train"
    group: str | None = None


class MovingSyntheticSource:
    """River-native continuously moving binary concepts; no discrete truth points."""

    def __init__(self, name, samples, seed, label_delay, epoch, *, n_features=10,
                 n_drift_features=2, mag_change=0.001, noise_percentage=0.05,
                 sigma=0.1, n_centroids=20, n_drift_centroids=10, change_speed=0.001):
        _bounded(samples, seed, label_delay, epoch)
        if name not in {"hyperplane", "rbf_drift"}:
            raise ValueError("unsupported moving synthetic source")
        if not 2 <= n_features <= 1000:
            raise ValueError("n_features must be between 2 and 1000")
        if name == "hyperplane":
            if not 1 <= n_drift_features <= n_features or not 0 < mag_change <= 1:
                raise ValueError("hyperplane requires moving features and positive mag_change")
            params = dict(seed=seed, n_features=n_features, n_drift_features=n_drift_features,
                          mag_change=mag_change, noise_percentage=noise_percentage, sigma=sigma)
            constructor = synth.Hyperplane
        else:
            if not 1 <= n_drift_centroids <= n_centroids <= 10_000 or not 0 < change_speed <= 1:
                raise ValueError("rbf_drift requires moving centroids and positive change_speed")
            params = dict(seed_model=seed, seed_sample=seed, n_classes=2, n_features=n_features,
                          n_centroids=n_centroids, n_drift_centroids=n_drift_centroids,
                          change_speed=change_speed)
            constructor = synth.RandomRBFDrift
        # Instantiate now so native parameter errors fail before a run is admitted.
        constructor(**params)
        self.constructor, self.params = constructor, params
        self.name, self.samples, self.seed = name, samples, seed
        self.label_delay, self.epoch = label_delay, epoch
        self.version = ""
        self.version = _identity({k: v for k, v in self.manifest().items() if k != "version"})

    def events(self):
        for seq, (x, y) in enumerate(islice(self.constructor(**self.params), self.samples)):
            if y not in (0, 1):
                raise ValueError("binary generator emitted a non-binary target")
            timestamp = self.epoch + seq
            yield Event(_identity([self.version, seq]), seq, timestamp, timestamp + self.label_delay,
                        {str(k): v for k, v in x.items()}, bool(y), None)

    def manifest(self):
        return {"adapter": "river-moving-v2", "name": self.name, "version": self.version,
                "river_version": "0.26.1",
                "samples": self.samples, "seed": self.seed, "params": self.params,
                "task": "binary", "drift_kind": "continuously_moving_concept",
                "changepoints": None, "timestamp_kind": "simulated_ordinal_seconds",
                "epoch": self.epoch, "label_delay_seconds": self.label_delay,
                "license": "BSD-3-Clause (generator implementation)",
                "source": "https://github.com/online-ml/river/tree/0.26.1/river/datasets/synth"}


class FriedmanDriftSource:
    """Native regression samples; binary runner rejection prevents target collapse."""

    def __init__(self, samples, seed, label_delay=0, epoch=0, *, name="friedman_drift",
                 drift_type="gra", position=None, transition_window=0):
        _bounded(samples, seed, label_delay, epoch)
        count = 3 if drift_type == "lea" else 2
        position = tuple(position or (samples * i // (count + 1) for i in range(1, count + 1)))
        if (len(position) != count or tuple(sorted(set(position))) != position
                or position[0] < 1 or position[-1] >= samples):
            raise ValueError("drift positions must be strictly increasing inside the sample range")
        if transition_window < 0 or (drift_type == "gsg" and
                                   any(b - a <= transition_window for a, b in zip(position, position[1:]))):
            raise ValueError("transition windows must be nonnegative and nonoverlapping")
        self.params = dict(seed=seed, drift_type=drift_type, position=position,
                           transition_window=transition_window)
        synth.FriedmanDrift(**self.params)
        self.name, self.samples, self.seed = name, samples, seed
        self.label_delay, self.epoch = label_delay, epoch
        self.version = ""
        self.version = _identity({k: v for k, v in self.manifest().items() if k != "version"})

    def samples_iter(self):
        for seq, (x, y) in enumerate(islice(synth.FriedmanDrift(**self.params), self.samples)):
            timestamp = self.epoch + seq
            yield TaskSample(_identity([self.version, seq]), seq, timestamp, timestamp + self.label_delay,
                             {str(k): v for k, v in x.items()}, float(y))

    def events(self):
        raise ValueError("FriedmanDrift is regression; use samples_iter() and a regression evaluator")

    def manifest(self):
        # River uses a zero-based counter: the new regime starts at seq=position.
        return {"adapter": "river-friedman-v1", "name": self.name, "version": self.version,
                "river_version": "0.26.1",
                "task": "regression", "samples": self.samples, "params": self.params,
                "changepoints": list(self.params["position"]),
                "drift_kind": "concept", "epoch": self.epoch,
                "timestamp_kind": "simulated_ordinal_seconds", "label_delay_seconds": self.label_delay,
                "license": "BSD-3-Clause (generator implementation)",
                "source": "https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/friedman.py"}


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
        _bounded(samples, seed, label_delay, epoch)
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
        self.version = _identity({"manifest": self.spec.model_dump(mode="json"), "samples": samples,
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
                sample = TaskSample(_identity([entry.sha256, count]), seq, timestamp,
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


def split_class_tasks(samples, class_order, classes_per_task=2):
    """Split MNIST/CIFAR protocol on prepared samples; preserves labels and splits."""
    class_order = tuple(str(c) for c in class_order)
    if (not class_order or len(set(class_order)) != len(class_order) or classes_per_task < 1
            or len(class_order) % classes_per_task):
        raise ValueError("class order must be unique and evenly divisible into tasks")
    mapping = {label: i // classes_per_task for i, label in enumerate(class_order)}
    seen = set()
    tasks = [[] for _ in range(len(class_order) // classes_per_task)]
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError("sample identity overlaps across protocol inputs")
        seen.add(sample.sample_id)
        if str(sample.y) not in mapping:
            raise ValueError("sample label is absent from declared class order")
        task = mapping[str(sample.y)]
        tasks[task].append(replace(sample, task_id=task))
    if any(not task for task in tasks):
        raise ValueError("every declared task must contain samples")
    return tasks


def permuted_tasks(samples, tasks, seed, feature_order):
    """Fixed permutation per task shared across train/test; identity task zero."""
    feature_order = tuple(feature_order)
    if tasks < 1 or tasks > 100 or not feature_order or len(set(feature_order)) != len(feature_order):
        raise ValueError("invalid permutation protocol")
    samples = tuple(samples)
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("duplicate sample identities")
    rng, output, permutations = random.Random(seed), [], []
    for task in range(tasks):
        permutation = list(feature_order)
        if task:
            rng.shuffle(permutation)
        permutations.append(permutation)
        transformed = []
        for sample in samples:
            if set(sample.x) != set(feature_order):
                raise ValueError("permutation feature schema mismatch")
            transformed.append(replace(sample, sample_id=_identity([sample.sample_id, task, permutation]),
                                       task_id=task,
                                       x={k: sample.x[v] for k, v in zip(feature_order, permutation)}))
        output.append(transformed)
    return output, permutations


def temporal_tasks(samples, group_order):
    """Preserve official CORe50/Stream51/CLEAR groups and source splits.

    The caller supplies the official session/bucket order and normalized records;
    this function never invents a split or shuffles a video sequence.
    """
    group_order = tuple(str(g) for g in group_order)
    if not group_order or len(set(group_order)) != len(group_order):
        raise ValueError("group order must contain unique groups")
    mapping = {group: i for i, group in enumerate(group_order)}
    output = [[] for _ in group_order]
    seen, groups_by_split = set(), {}
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError("sample identity overlaps across protocol inputs")
        seen.add(sample.sample_id)
        if sample.group not in mapping:
            raise ValueError("sample group is absent from official group order")
        key = (sample.group, sample.split)
        if sample.timestamp < groups_by_split.get(key, -1):
            raise ValueError("within-group temporal order decreased")
        groups_by_split[key] = sample.timestamp
        output[mapping[sample.group]].append(replace(sample, task_id=mapping[sample.group]))
    if any(not group for group in output):
        raise ValueError("every declared temporal group must contain samples")
    return output
