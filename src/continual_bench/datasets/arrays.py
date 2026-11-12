"""Memory-bounded datasets for long iterative training.

Tabular sources keep their (small) feature dicts; image sources keep one uint8 matrix and build the
River feature dict of a single row on demand, so 10k CIFAR images cost ~30 MB instead of ~3 GB.
"""

from pathlib import Path

from continual_bench.core.tasks import MULTICLASS, REGRESSION  # noqa: F401
from continual_bench.datasets.catalog import DATASETS, dataset_spec

TABULAR_LIMIT = 100_000
IMAGE_LIMIT = 10_000
IMAGE_FAMILIES = {"image", "temporal_vision"}
NATIVE_ROWS = {"electricity": 45_312, "airlines": 539_383, "insects": 52_848, "gas": 13_910}
NOMINAL = {"agrawal": ["elevel", "car", "zipcode"], "stagger": ["size", "color", "shape"]}


class ListDataset:
    kind = "tabular"

    def __init__(self, name, task, xs, ys, tasks, manifest):
        self.name, self.task, self.xs, self.ys, self.tasks, self.meta = name, task, xs, ys, tasks, manifest

    def __len__(self):
        return len(self.ys)

    def row(self, index):
        return self.xs[index], self.ys[index]


class ImageDataset:
    kind = "image"

    def __init__(self, name, x, y, tasks, manifest):
        self.name, self.task, self.x, self.tasks, self.meta = name, MULTICLASS, x, tasks, manifest
        self.ys = [str(v) for v in y]
        self.keys = [f"pixel_{i:04d}" for i in range(x.shape[1])]

    def __len__(self):
        return len(self.ys)

    def row(self, index):
        return dict(zip(self.keys, (self.x[index] / 255.0).tolist())), self.ys[index]


def _tabular(name, cache, real_directory, download):
    limit = min(TABULAR_LIMIT, NATIVE_ROWS.get(name, TABULAR_LIMIT))
    source = dataset_spec(name).prepared(cache, examples=limit, seed=42, download=download,
                                         real_directory=real_directory)
    manifest = source.manifest()
    xs, ys, tasks = [], [], []
    for sample in source.samples_iter():
        xs.append(sample.x)
        ys.append(sample.y)
        tasks.append(sample.task_id)
    if len(ys) != limit:
        raise ValueError(f"{name}: expected {limit} samples, got {len(ys)}")
    manifest = {k: manifest[k] for k in ("name", "task", "version", "samples") if k in manifest}
    return ListDataset(name, manifest.get("task", MULTICLASS), xs, ys, tasks, manifest)


def _image(name, cache, download):
    import numpy as np

    spec = dataset_spec(name)
    if name in {"split_mnist", "permuted_mnist"}:
        arrays = _mnist(name, cache, download)
    elif spec.family == "image":
        from .cifar_parquet import prepare_cifar_arrays
        arrays = prepare_cifar_arrays(name, cache, IMAGE_LIMIT, download=download)
    else:
        from .temporal_vision import prepare_temporal_arrays
        arrays = prepare_temporal_arrays(name, Path(cache), IMAGE_LIMIT, download=download)
    tasks = [int(t) for t in arrays["task"]]
    return ImageDataset(name, arrays["x"], arrays["y"], tasks,
                        {"name": name, "task": MULTICLASS, "samples": int(len(tasks)),
                         "features": int(arrays["x"].shape[1]), "dtype": str(np.asarray(arrays["x"]).dtype)})


def _mnist(name, cache, download):
    import random

    import numpy as np

    from .images import acquire_asset, image_dataset_locks, read_mnist

    specification = image_dataset_locks()["datasets"]["mnist"]
    raw_root = Path(cache) / "raw" / "mnist"
    paths = {a["filename"]: acquire_asset(raw_root, a, download=download) for a in specification["assets"]}
    permutation = list(range(784))
    random.Random(42).shuffle(permutation)
    counts = {"train": IMAGE_LIMIT * 4 // 5, "test": IMAGE_LIMIT - IMAGE_LIMIT * 4 // 5}
    rows = []
    for split, count in counts.items():
        prefix = "train" if split == "train" else "t10k"
        for index, pixels, label in read_mnist(paths[f"{prefix}-images-idx3-ubyte.gz"],
                                               paths[f"{prefix}-labels-idx1-ubyte.gz"], count):
            raw = np.frombuffer(pixels, dtype=np.uint8)
            task = index % 2 if name == "permuted_mnist" else label // 2
            permuted = name == "permuted_mnist" and task
            rows.append((task, split != "train", index, raw[permutation] if permuted else raw, label))
    rows.sort(key=lambda r: r[:3])
    return {"x": np.stack([r[3] for r in rows]), "y": np.array([str(r[4]) for r in rows]),
            "task": np.array([r[0] for r in rows], dtype=np.int64)}


def load_iterative(name, cache, *, real_directory, download=True):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name}")
    if dataset_spec(name).family in IMAGE_FAMILIES:
        return _image(name, cache, download)
    return _tabular(name, cache, real_directory, download)
