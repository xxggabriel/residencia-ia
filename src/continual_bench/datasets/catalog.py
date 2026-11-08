"""Dataset catalog: the single declaration of every admitted dataset.

Each `DatasetSpec` states the target type, family and two factories:
`stream` (a `DatasetConfig`-driven Event/TaskSample source for the prequential
evaluators) and `prepare` (a bounded, cached TaskSample source for epochs).
Factories are import strings, so configuration validation never imports River.
"""

from dataclasses import asdict, dataclass, field
from typing import Callable

from continual_bench.core.registry import Registry, resolve
from continual_bench.core.tasks import BINARY, MULTICLASS, REGRESSION

from .base import as_sample_source

EPOCH = 1_735_689_600
PREPARED_LABEL_DELAY = 3


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str | None  # None: declared at admission by the dataset's own manifest.
    family: str
    stream: str | Callable | None = None
    stream_defaults: dict = field(default_factory=dict)
    prepare: str | Callable | None = None
    prepare_kind: str = "stream"  # "stream": wraps `stream`; "cache": acquires into a local cache.
    metadata: dict = field(default_factory=dict)

    def open_stream(self, **params):
        if self.stream is None:
            raise ValueError(f"dataset {self.name} has no streaming adapter")
        return resolve(self.stream)(**self.stream_defaults, **params)

    def stream_factory(self) -> Callable:
        return lambda **params: self.open_stream(**params)

    def prepared(self, cache, *, examples, seed, download=False, real_directory=None):
        if self.prepare_kind == "stream":
            params = dict(samples=examples, seed=seed, label_delay=PREPARED_LABEL_DELAY, epoch=EPOCH)
            if self.family == "openml":
                params["directory"] = real_directory
            return as_sample_source(self.open_stream(**params))
        if self.prepare is None:
            raise ValueError(f"dataset {self.name} has no bounded epoch loader")
        return resolve(self.prepare)(self.name, cache, limit=examples, seed=seed, download=download)


CATALOG: Registry[DatasetSpec] = Registry("source")
_SYNTHETIC = "continual_bench.datasets.synthetic"
_LOCAL = "continual_bench.datasets.local_manifest:LocalManifestSource"
for _name in ("sea", "agrawal", "stagger"):
    CATALOG.add(DatasetSpec(_name, BINARY, "river_abrupt", f"{_SYNTHETIC}:SyntheticSource", {"name": _name},
                            metadata={"changepoints": "synthetic", "probes": True}))
for _name in ("hyperplane", "rbf_drift"):
    CATALOG.add(DatasetSpec(_name, BINARY, "river_moving", f"{_SYNTHETIC}:MovingSyntheticSource",
                            {"name": _name}, metadata={"changepoints": None}))
CATALOG.add(DatasetSpec("friedman_drift", REGRESSION, "river_regression", f"{_SYNTHETIC}:FriedmanDriftSource",
                        metadata={"changepoints": "synthetic"}))
for _name in ("electricity", "airlines"):
    CATALOG.add(DatasetSpec(_name, BINARY, "openml", "continual_bench.datasets.openml:RealSource",
                            {"name": _name}, metadata={"changepoints": None}))
for _name in ("insects", "gas"):
    CATALOG.add(DatasetSpec(_name, MULTICLASS, "real_stream", _LOCAL, {"name": _name},
                            "continual_bench.datasets.real_streams:prepare_real_stream", "cache"))
for _name in ("split_mnist", "permuted_mnist"):
    CATALOG.add(DatasetSpec(_name, MULTICLASS, "image", _LOCAL, {"name": _name},
                            "continual_bench.datasets.images:prepare_image_dataset", "cache"))
for _name in ("split_cifar10", "split_cifar100"):
    CATALOG.add(DatasetSpec(_name, MULTICLASS, "image", _LOCAL, {"name": _name},
                            "continual_bench.datasets.cifar_parquet:prepare_cifar_parquet", "cache"))
for _name in ("core50", "clear10", "clear100"):
    CATALOG.add(DatasetSpec(_name, MULTICLASS, "temporal_vision", _LOCAL, {"name": _name},
                            "continual_bench.datasets.temporal_vision:prepare_temporal_dataset", "cache"))
# Local-manifest aliases without a bounded epoch loader; their manifest declares the task.
for _name in ("split_cifar", "clear"):
    CATALOG.add(DatasetSpec(_name, None, "local_manifest", _LOCAL, {"name": _name}, prepare_kind="none"))

# Ordered datasets with a bounded epoch loader: the full-suite admission contract.
DATASETS = tuple(spec.name for spec in CATALOG if spec.prepare_kind != "none")


def dataset_spec(name: str) -> DatasetSpec:
    return CATALOG.get(name)


def datasets_for(task: str) -> tuple[str, ...]:
    """Names whose native target type is `task` (local manifests may declare others)."""
    return tuple(spec.name for spec in CATALOG if spec.task == task)


def load_dataset(name, cache, *, examples=100, seed=42, download=False, real_directory=None):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name}")
    if type(examples) is not int or not 10 <= examples <= 10_000:
        raise ValueError("examples must be an integer in [10,10000]")
    return CATALOG.get(name).prepared(cache, examples=examples, seed=seed, download=download,
                                      real_directory=real_directory)


def sample_dict(sample):
    """Canonical evidence retains original ID/task/split, never just a feature matrix."""
    return asdict(sample)
