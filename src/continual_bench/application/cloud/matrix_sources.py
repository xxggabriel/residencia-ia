"""Publish explicit external inputs and verify them before a private worker uses them."""

import json
from dataclasses import asdict
from pathlib import Path

from continual_bench.core.contracts import TaskSample
from continual_bench.datasets.base import EventSamples
from continual_bench.datasets.catalog import DATASETS, EPOCH, PREPARED_LABEL_DELAY, dataset_spec, load_dataset
from continual_bench.datasets.images import PreparedImageSource
from continual_bench.datasets.openml import REAL_SOURCES, validate_source
from continual_bench.datasets.real_streams import (
    INSECTS_BINARY_VARIANTS,
)
from continual_bench.datasets.real_streams import (
    SPECS as STREAM_SPECS,
)
from continual_bench.datasets.real_streams import (
    _acquire as acquire_real_stream,
)
from continual_bench.datasets.real_streams import (
    _validate as validate_real_stream,
)
from continual_bench.infrastructure.object_store import ObjectExists, sha256_path

from .contracts import OFFLINE_MATRIX_FAMILIES, canonical
from .layout import matrix_source_key

MAX_BYTES = 256 * 1024**2


def publish_source(store, dataset, *, cache, workdir, examples, seed, real_directory=None, download=False):
    spec = dataset_spec(dataset)
    if spec.family in OFFLINE_MATRIX_FAMILIES or (dataset not in DATASETS and spec.family != "usp_stream"):
        raise ValueError(f"{dataset} does not need a published external source")
    if spec.family == "openml":
        if not real_directory:
            raise ValueError("OpenML sources require --real-directory with the pinned ARFF files")
        path = Path(real_directory) / f"{dataset}.arff"
        validate_source(path, dataset)
        metadata = {"format": "openml-arff-v1"}
    elif spec.family == "usp_stream":
        if real_directory:
            root = Path(real_directory)
            if (root / dataset).exists():
                path, _ = acquire_real_stream(dataset, root, download=False)
            else:
                path = root / STREAM_SPECS[dataset]["filename"]
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"USP stream source is absent or symlinked: {path}")
                expected_sha = STREAM_SPECS[dataset].get("expected_sha256")
                if expected_sha is not None and sha256_path(path) != expected_sha:
                    raise ValueError(f"Pinned {dataset} SHA-256 mismatch")
                validate_real_stream(dataset, path)
        else:
            path, _ = acquire_real_stream(dataset, cache, download=download)
        metadata = {"format": "usp-csv-v1"}
    else:
        source = load_dataset(dataset, cache, examples=examples, seed=seed, download=download)
        path = Path(workdir) / "source.json"
        path.write_bytes(
            canonical(
                {"manifest": source.manifest(), "samples": [asdict(row) for row in source.samples_iter()]}
            )
        )
        metadata = {"format": "task-samples-v1", "examples": str(examples), "seed": str(seed)}
    sha256 = sha256_path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"matrix source exceeds the {MAX_BYTES // 1024**2} MiB source limit")
    key = matrix_source_key(dataset, sha256)
    try:
        store.put_file(key, path, if_absent=True, sha256=sha256, metadata=metadata)
    except ObjectExists:
        head = store.head(key)
        if not head or head.sha256 != sha256 or any(head.metadata.get(k) != v for k, v in metadata.items()):
            raise
    return {"dataset": dataset, "sha256": sha256, "uri": store.uri(key)}


def verify_source(store, dataset, sha256):
    key = matrix_source_key(dataset, sha256)
    head = store.head(key)
    if head is None or head.sha256 != sha256 or not 0 < head.size <= MAX_BYTES:
        raise ValueError(f"published matrix source is absent or invalid: {key}")
    if dataset in REAL_SOURCES:
        if sha256 != REAL_SOURCES[dataset]["sha256"] or head.metadata.get("format") != "openml-arff-v1":
            raise ValueError(f"{dataset} source differs from the pinned OpenML archive")
    elif dataset in INSECTS_BINARY_VARIANTS:
        expected_sha = STREAM_SPECS[dataset].get("expected_sha256")
        if (expected_sha is not None and sha256 != expected_sha) or head.metadata.get(
            "format"
        ) != "usp-csv-v1":
            raise ValueError(f"{dataset} source differs from the pinned USP stream CSV")
    elif head.metadata.get("format") != "task-samples-v1":
        raise ValueError(f"{dataset} requires a prepared sample bundle")
    return head


def verify_plan_sources(store, units):
    heads = {}
    for unit in units:
        sha256 = unit["params"].get("source_sha256")
        if not sha256:
            continue
        dataset = unit["labels"]["dataset"]
        key = dataset, sha256
        if key not in heads:
            heads[key] = verify_source(store, dataset, sha256)
        config = unit["params"]["config"]
        if dataset in REAL_SOURCES:
            epochs = unit["params"]["evaluator"] == "epochs"
            samples = config["examples"] if epochs else config["dataset"]["samples"]
            if samples > REAL_SOURCES[dataset]["rows"]:
                raise ValueError(f"{dataset} requested more rows than the pinned OpenML source")
        elif dataset in INSECTS_BINARY_VARIANTS:
            epochs = unit["params"]["evaluator"] == "epochs"
            samples = config["examples"] if epochs else config["dataset"]["samples"]
            if samples > STREAM_SPECS[dataset]["expected_rows"]:
                raise ValueError(f"{dataset} requested more rows than the pinned USP stream source")
        elif any(heads[key].metadata.get(k) != str(config[k]) for k in ("examples", "seed")):
            raise ValueError(f"{dataset} sample bundle does not match the requested count and seed")


def materialize_source(store, dataset, sha256, workdir, config, evaluator):
    head = verify_source(store, dataset, sha256)
    if dataset in REAL_SOURCES:
        target_name = f"{dataset}.arff"
    elif dataset in INSECTS_BINARY_VARIANTS:
        target_name = STREAM_SPECS[dataset]["filename"]
    else:
        target_name = "source.json"
    target = Path(workdir) / target_name
    store.download(head.key, target)
    if sha256_path(target) != sha256:
        raise ValueError(f"downloaded matrix source checksum differs for {dataset}")
    if dataset in REAL_SOURCES:
        spec = dataset_spec(dataset)
        if evaluator == "epochs":
            return spec.prepared(workdir, examples=config.examples, seed=config.seed, real_directory=workdir)
        fields = config.dataset.model_dump(exclude={"name", "params"})
        return spec.open_stream(**fields, directory=workdir)
    if dataset in INSECTS_BINARY_VARIANTS:
        spec = dataset_spec(dataset)
        if evaluator == "epochs":
            return EventSamples(
                spec.open_stream(
                    samples=config.examples,
                    seed=config.seed,
                    label_delay=PREPARED_LABEL_DELAY,
                    epoch=EPOCH,
                    directory=workdir,
                )
            )
        fields = config.dataset.model_dump(exclude={"name", "params"})
        source = spec.open_stream(**fields, directory=workdir)
        return EventSamples(source) if evaluator == "typed" else source
    if (head.metadata.get("examples"), head.metadata.get("seed")) != (str(config.examples), str(config.seed)):
        raise ValueError(f"{dataset} sample bundle does not match the requested count and seed")
    value = json.loads(target.read_bytes())
    return PreparedImageSource((TaskSample(**sample) for sample in value["samples"]), value["manifest"])
