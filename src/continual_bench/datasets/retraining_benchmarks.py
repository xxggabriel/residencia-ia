"""Datasets of the UPF (Regol et al. 2025) and CARA (Mahadevan & Mathioudakis 2024) papers.

Each function returns a list of batches ``(x, y)`` (one per time step). Synthetic generators follow
the formulas printed in the papers; the sentence "``<= 0.5 eps``" of UPF's *circles* rule does not
define ``eps``, so it is a parameter (default 1, squared radius 0.5) and the choice is recorded in
:data:`ASSUMPTIONS`. Real datasets are split in temporal order; this module never downloads anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSUMPTIONS = {
    "upf_circles_epsilon": "UPF A.6 prints `<= 0.5 eps` without defining eps; default eps = 1",
    "upf_gauss_rule": "UPF A.6 prints y = 1[(4*r1 - 0.5)**2 > r2], read literally",
    "upf_train_test_split": "UPF does not state the within-dataset split; 80/20 temporal-free random split",
    "upf_time_index": "mu_1(t), mu_2(t) and the circle centre use the dataset index t within a trial",
}


def upf_gauss(n_datasets: int, size: int = 5000, seed: int = 0, start: int = 0):
    rng = np.random.default_rng(seed)
    batches = []
    for k in range(n_datasets):
        t = start + k
        centre = np.array([(t + 1) / 100, 0.5 - (t + 1) / 100])
        x = rng.normal(centre, 0.1, size=(size, 2))
        batches.append((x, ((4 * x[:, 0] - 0.5) ** 2 > x[:, 1]).astype(np.int64)))
    return batches


def upf_circles(n_datasets: int, size: int = 5000, seed: int = 0, start: int = 0, epsilon: float = 1.0):
    rng = np.random.default_rng(seed)
    batches = []
    for k in range(n_datasets):
        centre = 0.2 + 0.02 * (start + k)
        x = rng.uniform(0, 1, size=(size, 2))
        inside = (x[:, 0] - centre) ** 2 + (x[:, 1] - centre) ** 2 <= 0.5 * epsilon
        batches.append((x, inside.astype(np.int64)))
    return batches


def _cara_concept(name: str):
    """``(sample(t, rng, size) -> x, label(t, x) -> y)`` of a CARA synthetic dataset."""
    if name == "gauss":
        def sample(t, rng, size):
            centre = ((t + 1) % 15) / 30
            return rng.normal([centre, 0.5 - centre], 0.1, size=(size, 2))

        def label(t, x):
            return (x[:, 1] > 4 * (x[:, 0] - 0.5) ** 2).astype(np.int64)
    elif name == "circle":
        def sample(t, rng, size):
            return rng.uniform(0, 1, size=(size, 2))

        def label(t, x):
            centre, radius2 = 0.2 + 0.02 * (t % 25), (0.2 + 0.01 * (t % 25)) ** 2
            return (((x - centre) ** 2).sum(axis=1) <= radius2).astype(np.int64)
    elif name == "covcon":
        def sample(t, rng, size):
            mean = ((t + 1) % 7) / 10
            return rng.normal([mean, mean], 0.1, size=(size, 2))

        def label(t, x):
            above = x[:, 0] * np.sin(np.pi * x[:, 0]) > x[:, 1]
            return (~above if (t // 10) % 2 == 1 else above).astype(np.int64)
    else:
        raise ValueError(f"unknown CARA dataset {name}")
    return sample, label


def _cara_batches(name, n_batches, size, seed):
    sample, label = _cara_concept(name)
    rng = np.random.default_rng(seed)
    batches = []
    for t in range(n_batches):
        x = sample(t, rng, size)
        batches.append((x, label(t, x)))
    return batches


def cara_gauss(n_batches: int = 100, size: int = 1000, seed: int = 0):
    """CARA Gauss: gradual covariate shift, fixed parabolic boundary ``x2 > 4 (x1 - 0.5)^2``."""
    return _cara_batches("gauss", n_batches, size, seed)


def cara_circle(n_batches: int = 100, size: int = 1000, seed: int = 0):
    """CARA Circle: uniform points, circle whose centre and radius drift (gradual concept drift)."""
    return _cara_batches("circle", n_batches, size, seed)


def cara_covcon(n_batches: int = 100, size: int = 1000, seed: int = 0):
    """CARA CovCon: moving Gaussian (covariate drift); the boundary inequality flips every 10 batches."""
    return _cara_batches("covcon", n_batches, size, seed)


def cara_static_queries(name: str, n_batches: int = 100, n_queries: int = 100, seed: int = 0):
    """CARA "-S" query stream: static Gaussian at (0.5, 0.5), sigma 0.015, labelled by the batch concept."""
    _, label = _cara_concept(name)
    rng = np.random.default_rng(seed + 7919)
    batches = []
    for t in range(n_batches):
        x = rng.normal(0.5, 0.015, size=(n_queries, 2))
        batches.append((x, label(t, x)))
    return batches


def split_batch(x, y, rng, train_fraction: float = 0.8):
    order = rng.permutation(len(y))
    cut = int(len(y) * train_fraction)
    return x[order[:cut]], y[order[:cut]], x[order[cut:]], y[order[cut:]]


def electricity_batches(path: str | Path, size: int = 2000):
    """NSW electricity (Harries) in the file's chronological order, 6 features, UP = 1.

    Uses features period, nswprice, nswdemand, vicprice, vicdemand and transfer, as in UPF Table 4
    (6 features); ``date`` and ``day`` are dropped.
    """
    rows, labels, started = [], [], False
    for line in Path(path).read_text().splitlines():
        if line.lower().startswith("@data"):
            started = True
            continue
        if not started or not line.strip():
            continue
        fields = line.strip().split(",")
        rows.append([float(v) for v in fields[2:8]])
        labels.append(1 if fields[8].strip().upper() == "UP" else 0)
    x, y = np.asarray(rows), np.asarray(labels, dtype=np.int64)
    count = len(y) // size
    return [(x[i * size:(i + 1) * size], y[i * size:(i + 1) * size]) for i in range(count)]


def _cached(cache: Path, build):
    if cache.is_file():
        with np.load(cache) as stored:
            return stored["x"], stored["y"]
    x, y = build()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def _chunks(x, y, n_batches):
    size = len(y) // n_batches
    return [(x[i * size:(i + 1) * size], y[i * size:(i + 1) * size]) for i in range(n_batches)]


def covertype_arrays(path: str | Path):
    """LibSVM binary Covertype (581,012 rows, 54 features, classes 1/2 -> 0/1) in file order."""
    import bz2

    def build():
        rows, labels = [], []
        with bz2.open(path, "rt") as stream:
            for line in stream:
                head, *pairs = line.split()
                row = np.zeros(54, dtype=np.float32)
                for pair in pairs:
                    index, value = pair.split(":")
                    row[int(index) - 1] = float(value)
                rows.append(row)
                labels.append(int(float(head)) - 1)
        return np.vstack(rows), np.asarray(labels, dtype=np.int64)

    return _cached(Path(path).with_suffix(".npz"), build)


def covertype_batches(path: str | Path, n_batches: int = 100):
    x, y = covertype_arrays(path)
    return _chunks(x, y, n_batches)


def airlines_arrays(path: str | Path):
    """OpenML airlines (539,383 rows): 7 features, nominal ones ordinal-encoded, Delay = 1 positive.

    UPF (Table 4) lists 7 features for XGBoost. CARA one-hot encodes the nominals (7,198 columns); this
    adapter does not, which is a recorded deviation for the CARA experiment.
    """
    from .arff import iter_arff

    def build():
        columns = ["Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length"]
        codes: dict[str, dict] = {c: {} for c in ("Airline", "AirportFrom", "AirportTo", "DayOfWeek")}
        rows, labels = [], []
        for row in iter_arff(path):
            labels.append(1 if str(row.pop("Delay")) == "1" else 0)
            values = []
            for column in columns:
                value = row[column]
                values.append(float(codes[column].setdefault(value, len(codes[column])))
                              if column in codes else float(value))
            rows.append(values)
        return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    return _cached(Path(path).with_suffix(".npz"), build)


def airlines_batches(path: str | Path, n_batches: int = 100):
    x, y = airlines_arrays(path)
    return _chunks(x, y, n_batches)


def windowed_batches(x, y, size: int, count: int, start: int):
    """``count`` consecutive batches of ``size`` rows beginning at batch ``start`` (UPF trial windows)."""
    return [(x[(start + i) * size:(start + i + 1) * size], y[(start + i) * size:(start + i + 1) * size])
            for i in range(count)]
