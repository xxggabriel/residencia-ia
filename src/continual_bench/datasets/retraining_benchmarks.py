"""Datasets of the UPF (Regol et al. 2025) and CARA (Mahadevan & Mathioudakis 2024) papers.

Each function returns a list of batches ``(x, y)`` (one per time step). CARA generators follow the
authors' experiment configuration. For UPF, the sentence "``<= 0.5 eps``" of UPF's *circles* rule does not
define ``eps``, so it is a parameter (default 1, squared radius 0.5) and the choice is recorded in
:data:`ASSUMPTIONS`. Real datasets are split in temporal order; this module never downloads anything.
"""

from __future__ import annotations

from bisect import bisect_right
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


def _cara_labels(name, points, seed):
    if name == "gauss":
        return [(4 * (x[:, 0] - 0.5) ** 2 > x[:, 1]).astype(int) for x in points]
    if name == "covcon":
        labels = []
        for t, x in enumerate(points):
            boundary = 0.6 * np.sin(np.pi * x[:, 0])
            below = boundary < x[:, 1] if (t // 10) % 2 == 0 else boundary > x[:, 1]
            labels.append(below.astype(int))
        return labels
    if name != "circle":
        raise ValueError(f"unknown CARA dataset {name}")
    rng = np.random.default_rng(seed)
    drifts = (15, 45, 85)
    centres = (0.2, 0.4, 0.6, 0.8)
    radii = (0.15, 0.2, 0.25, 0.3)

    def inside(x, concept):
        return ((x[:, 0] - centres[concept]) ** 2 + (x[:, 1] - 0.5) ** 2 <= radii[concept] ** 2)

    labels = []
    for t, x in enumerate(points):
        concept = bisect_right(drifts, t)
        since_drift = t - drifts[concept - 1] + 1 if concept else 5
        if since_drift < 5:
            y = inside(x, concept - 1)
            switched = rng.choice(len(x), size=int((since_drift / 5) * len(x)), replace=False)
            y[switched] = inside(x[switched], concept)
        else:
            y = inside(x, concept)
        labels.append(y.astype(int))
    return labels


def cara_gauss(n_batches: int = 100, size: int = 1000, seed: int = 0):
    """Gauss stream with the reference repository's seed and standard deviation."""
    rng = np.random.default_rng(seed)
    centres = [((t + 1) % 15) / 30 for t in range(n_batches)]
    points = [rng.normal([c, 0.5 - c], 0.05, size=(size, 2)) for c in centres]
    return list(zip(points, _cara_labels("gauss", points, seed), strict=True))


def cara_circle(n_batches: int = 100, size: int = 1000, seed: int = 0):
    """Circles stream with mixtures of old and new concepts at the three drifts."""
    rng = np.random.default_rng(seed)
    points = [rng.random(size=(size, 2)) for _ in range(n_batches)]
    return list(zip(points, _cara_labels("circle", points, seed), strict=True))


def cara_covcon(n_batches: int = 100, size: int = 1000, seed: int = 0):
    """CovCon stream with covariate shift and a concept flip every ten batches."""
    rng = np.random.default_rng(seed)
    points = [rng.normal(((t + 1) % 7) / 10, 0.1, size=(size, 2)) for t in range(n_batches)]
    return list(zip(points, _cara_labels("covcon", points, seed), strict=True))


def cara_static_queries(name: str, n_batches: int = 100, n_queries: int = 100, seed: int = 0):
    """Static queries with standard deviation 0.01 in the reference configuration."""
    rng = np.random.default_rng(seed)
    points = [rng.normal((0.5, 0.5), 0.01, size=(n_queries, 2)) for _ in range(n_batches)]
    return list(zip(points, _cara_labels(name, points, seed), strict=True))


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
    """Load seven Airlines features and the binary Delay label from CSV or OpenML ARFF.

    UPF (Table 4) lists 7 features for XGBoost. CARA uses the separate one-hot CSV adapter.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        # The numeric Scikit-Multiflow source used in UPF A.6 already encodes the categories.
        rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
        if rows.shape[1] != 8 or not np.isin(rows[:, -1], [0, 1]).all():
            raise ValueError("expected seven Airlines features and a binary Delay column")
        return rows[:, :-1], rows[:, -1].astype(np.int64)

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

    return _cached(path.with_suffix(".npz"), build)


def airlines_batches(path: str | Path, n_batches: int = 100):
    x, y = airlines_arrays(path)
    return _chunks(x, y, n_batches)


def windowed_batches(x, y, size: int, count: int, start: int):
    """``count`` consecutive batches of ``size`` rows beginning at batch ``start`` (UPF trial windows)."""
    return [(x[(start + i) * size:(start + i + 1) * size], y[(start + i) * size:(start + i + 1) * size])
            for i in range(count)]


def cara_real_batches(name: str, path: str | Path, n_batches: int = 100):
    """Load and preprocess the source files named in the CARA repository."""
    import csv

    from scipy.sparse import hstack
    from sklearn.datasets import load_svmlight_file
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, OneHotEncoder

    path = Path(path)
    if name == "covertype":
        # load_svmlight_file opens .bz2 transparently when given a filename.
        x, y = load_svmlight_file(str(path))
        return list(zip(np.array_split(x.toarray(), n_batches), np.array_split(y - 1, n_batches),
                        strict=True))
    with path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        rows = list(reader)
    def column(index):
        values = [row[index] for row in rows]
        try:
            return np.asarray(values, dtype=np.int64)
        except ValueError:
            return np.asarray(values)

    y = column(-1)
    if name == "electricity":
        x = np.asarray([row[:-1] for row in rows], dtype=float)
        x = MinMaxScaler().fit_transform(x)
        return list(zip(np.array_split(x, n_batches), np.array_split(y, n_batches), strict=True))
    if name != "airlines":
        raise ValueError(f"unknown CARA dataset {name}")
    airline, flight, origin, destination, day, time, length = (
        header.index(column)
        for column in ("Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length")
    )
    categories = np.empty((len(rows), 5), dtype=object)
    for j, index in enumerate((airline, flight, origin, destination, day)):
        categories[:, j] = column(index)
    numerical = np.asarray([(row[time], row[length]) for row in rows], dtype=float)
    x = hstack([OneHotEncoder().fit_transform(categories), numerical]).tocsr()
    x = MaxAbsScaler().fit_transform(x)
    size = len(y) // n_batches + 1
    return [(x[i * size:(i + 1) * size], y[i * size:(i + 1) * size]) for i in range(n_batches)]
