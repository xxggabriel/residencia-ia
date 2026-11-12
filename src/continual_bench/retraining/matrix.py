"""Performance matrices for batch retraining decisions (CARA, Mahadevan & Mathioudakis 2024;
UPF, Regol et al. 2025).

``err[i, j]`` is the 0-1 error rate of the model trained at the end of batch ``i`` on the later
batch ``j``. Training at the end of batch ``i`` and deploying from ``i + 1`` keeps the protocol
test-then-train: no model is ever scored on the labels that trained it. Both papers evaluate a
model retrained at ``t`` on batch ``t`` itself; that accounting needs a held-out slice of every
batch and is deliberately not used here (all schedules, the Oracle included, share this timing).

The matrix is built once, with all future entries, so the Oracle and the offline sweeps can use
it. Online policies only ever receive a :class:`MatrixView`, which masks everything that was not
observable at decision time.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

N_POINTS = 512  # labeled points kept per batch for the CARA similarity-weighted staleness
N_QUERIES = 100  # query points per batch (CARA-D: sampled from the data stream itself)


class Learner(Protocol):
    """Trains one model per batch and scores any trained model on any batch."""

    def fit(self, index: int, x: np.ndarray, y: np.ndarray) -> object:
        """Train the model produced at the end of batch ``index``; the learner owns its window."""

    def losses(self, state: object, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-row 0-1 loss of the trained model."""


@dataclass
class PerformanceMatrix:
    err: np.ndarray
    offline: int
    points: list[np.ndarray]
    queries: list[np.ndarray]
    sub_loss: dict[tuple[int, int], np.ndarray]
    feature_means: np.ndarray
    fit_seconds: np.ndarray
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Batch whose data trained the model of each decision point (paper timing: point p >= offline
    # trains on batch p + 1). ``None`` means the model of point p was trained on batch p.
    source: np.ndarray | None = None

    def __post_init__(self):
        size = self.err.shape[0]
        if self.err.shape != (size, size) or size < 4:
            raise ValueError("performance matrix needs at least four batches")
        if not 2 <= self.offline <= size - 2:
            raise ValueError("offline batches must leave at least two online batches")
        upper = np.triu_indices(size, 1)
        values = self.err[upper]
        # The last batch trains no model, so its row stays empty; every other pair is observed.
        observed = np.isfinite(values)
        if not observed.any() or np.any(values[observed] < 0) or np.any(values[observed] > 1):
            raise ValueError("matrix errors must be rates in [0, 1]")

    @property
    def batches(self) -> int:
        return self.err.shape[0]

    @property
    def horizon(self) -> int:
        """Online batches ``T`` (the first ``offline`` batches are the offline phase)."""
        return self.batches - self.offline

    @property
    def initial(self) -> int:
        """Model deployed when the online phase starts: the one trained on the last offline batch."""
        return self.offline - 1

    def view(self, upto: int, models: Iterable[int]) -> MatrixView:
        return MatrixView(self, upto, frozenset(models))

    def offline_view(self) -> MatrixView:
        return self.view(self.offline - 1, range(self.offline))

    def shift(self, j: int) -> float:
        """Mean-feature L1 shift between batch ``j`` and ``j - 1`` (UPF's ``z_shift``)."""
        if j <= 0:
            return 0.0
        return float(np.abs(self.feature_means[j] - self.feature_means[j - 1]).sum())

    def scaler(self) -> tuple[np.ndarray, np.ndarray]:
        sample = np.concatenate(self.points[: self.offline])
        scale = sample.std(axis=0)
        return sample.mean(axis=0), np.where(scale > 1e-12, scale, 1.0)

    def staleness(self, t: int, model: int) -> float:
        """Relative staleness of ``model`` for the queries of batch ``t`` (CARA, eq. 1-3).

        ``psi(Q_t, D_t, M) - psi(Q_t, D_model, M)`` with an RBF similarity, averaged over queries so
        it is an error-rate-like quantity that shares the unit of the retraining cost ``alpha``.
        """
        if not model < t:
            raise ValueError("staleness is defined for a model trained before batch t")
        mean, scale = self._scale
        queries = (self.queries[t] - mean) / scale
        gamma = 1.0 / queries.shape[1]

        query_norm = (queries**2).sum(axis=1)[:, None]

        def psi(batch: int, loss: np.ndarray) -> float:
            data = (self.points[batch] - mean) / scale
            distance = query_norm + (data**2).sum(axis=1)[None, :] - 2 * queries @ data.T
            return float((np.exp(-gamma * np.maximum(distance, 0.0)) * loss[None, :]).mean())

        own = model if self.source is None else int(self.source[model])
        return psi(t, self.sub_loss[model, t]) - psi(own, self.sub_loss[model, model])

    @property
    def _scale(self):
        if not hasattr(self, "_cached_scale"):
            self._cached_scale = self.scaler()
        return self._cached_scale

    def staleness_matrix(self) -> np.ndarray:
        """``s[t, model]`` for every observed ``model < t`` (full matrix; reporting and the Oracle only)."""
        if getattr(self, "_staleness_all", None) is None:
            size = self.batches
            result = np.full((size, size), np.nan)
            for t in range(1, size):
                for model in range(t):
                    if (model, t) in self.sub_loss:
                        result[t, model] = self.staleness(t, model)
            self._staleness_all = result
        return self._staleness_all

    def to_dict(self) -> dict:
        def clean(array):
            return [[None if not math.isfinite(v) else float(v) for v in row] for row in array]

        return {"offline": self.offline, "batches": self.batches, "err": clean(self.err),
                "fit_seconds": [None if not math.isfinite(v) else float(v) for v in self.fit_seconds],
                "rows": self.rows.tolist()}


class MatrixView:
    """What is observable after batch ``upto``: errors of *trained* models on batches ``<= upto``."""

    def __init__(self, matrix: PerformanceMatrix, upto: int, models: frozenset[int]):
        if not 0 <= upto < matrix.batches or not models or max(models) > upto:
            raise ValueError("a view cannot contain models trained after its horizon")
        self._matrix, self.upto, self.models = matrix, upto, models

    def err(self, model: int, batch: int) -> float:
        if model not in self.models or not model < batch <= self.upto:
            raise ValueError("error not observable at this decision time")
        return float(self._matrix.err[model, batch])

    def shift(self, batch: int) -> float:
        if batch > self.upto:
            raise ValueError("shift not observable at this decision time")
        return self._matrix.shift(batch)

    def staleness(self, t: int, model: int) -> float:
        if model not in self.models or t > self.upto:
            raise ValueError("staleness not observable at this decision time")
        return self._matrix.staleness(t, model)

    def observations(self) -> list[tuple[int, int, float]]:
        """All ``(model, batch, error)`` triples visible now: the UPF regression dataset."""
        return [(i, j, float(self._matrix.err[i, j]))
                for i in sorted(self.models) for j in range(i + 1, self.upto + 1)]


def build_matrix(
    stream: Iterable[tuple[np.ndarray, np.ndarray]],
    learner: Learner,
    *,
    offline: int | None = None,
    seed: int = 0,
    n_points: int = N_POINTS,
    n_queries: int = N_QUERIES,
    query_stream: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    workers: int = 1,
) -> tuple[PerformanceMatrix, list[object]]:
    """Single pass over ordered batches; returns the matrix and every trained model state.

    The model of the final batch is never trained (it could not serve any later batch). With
    ``query_stream`` (CARA's "-S" streams) the queries are the given labeled points instead of a
    sample of the batch, and ``err`` is measured on them. ``workers > 1`` scores the trained models of
    one batch in threads (``learner.losses`` must be thread-safe).
    """
    rng = np.random.default_rng(seed)
    states: list[object] = []
    errs: list[list[float]] = []
    points, queries, means, seconds, rows = [], [], [], [], []
    sub_loss: dict[tuple[int, int], np.ndarray] = {}
    iterator = iter(stream)
    query_iterator = None if query_stream is None else iter(query_stream)
    pending = next(iterator, None)
    j = 0
    while pending is not None:
        x, y = pending
        pending = next(iterator, None)
        x = np.asarray(x)
        if not len(y) or len(x) != len(y):
            raise ValueError("every batch needs rows and matching labels")
        point_index = rng.choice(len(y), size=min(n_points, len(y)), replace=False)
        if query_iterator is None:
            query_index = rng.choice(len(y), size=min(n_queries, len(y)), replace=False)
            queries.append(x[query_index].astype(np.float64))
            query_x = query_y = None
        else:
            query_x, query_y = (np.asarray(v) for v in next(query_iterator))
            queries.append(query_x.astype(np.float64))
        points.append(x[point_index].astype(np.float64))
        means.append(x.mean(axis=0, dtype=np.float64))
        rows.append(len(y))
        def score(state, x=x, y=y, query_x=query_x, query_y=query_y):
            loss = np.asarray(learner.losses(state, x, y), dtype=np.float64)
            on_queries = loss if query_x is None else np.asarray(
                learner.losses(state, query_x, query_y), dtype=np.float64)
            return loss, float(on_queries.mean())

        if workers > 1 and len(states) > 1:
            with ThreadPoolExecutor(workers) as pool:
                scored = list(pool.map(score, states))
        else:
            scored = [score(state) for state in states]
        for i, (loss, error) in enumerate(scored):
            errs[i].append(error)
            sub_loss[i, j] = loss[point_index]
        if pending is not None:
            started = time.perf_counter()
            state = learner.fit(j, x, y)
            seconds.append(time.perf_counter() - started)
            states.append(state)
            errs.append([])
            sub_loss[j, j] = np.asarray(
                learner.losses(state, x[point_index], y[point_index]), dtype=np.float64)
        else:
            seconds.append(float("nan"))
        j += 1
    size = j
    err = np.full((size, size), np.nan)
    for i, row in enumerate(errs):
        err[i, i + 1:i + 1 + len(row)] = row
    if offline is None:
        offline = max(2, size // 3)
    matrix = PerformanceMatrix(
        err=err, offline=offline, points=points, queries=queries, sub_loss=sub_loss,
        feature_means=np.asarray(means), fit_seconds=np.asarray(seconds), rows=np.asarray(rows))
    return matrix, states


def build_paper_matrix(
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    learner: Learner,
    offline: int,
    *,
    seed: int = 0,
    n_points: int = N_POINTS,
    n_queries: int = N_QUERIES,
) -> tuple[PerformanceMatrix, list[object]]:
    """UPF protocol timing: every batch has a train and a held-out test split.

    A model of an offline point ``p < offline`` trains on batch ``p``. For an online decision point
    ``p`` the model trains on batch ``p + 1`` (the data that arrives with the retrain, never observed
    when deciding) and is scored on that batch's *test* split, as in the UPF paper where a retrained
    model serves its own time step. ``err[p, j]`` is always measured on the test split of ``j``.
    """
    size = len(batches)
    if not 2 <= offline <= size - 2:
        raise ValueError("offline batches must leave at least two online batches")
    rng = np.random.default_rng(seed)
    source = np.array([p if p < offline else p + 1 for p in range(size)])
    points, queries, means = [], [], []
    for x_train, y_train, x_test, _ in batches:
        pool = np.concatenate([x_train, x_test])
        points.append(pool[rng.choice(len(pool), size=min(n_points, len(pool)), replace=False)]
                      .astype(np.float64))
        queries.append(pool[rng.choice(len(pool), size=min(n_queries, len(pool)), replace=False)]
                       .astype(np.float64))
        means.append(np.concatenate([x_train, x_test]).mean(axis=0, dtype=np.float64))
    states, seconds = [], []
    for p in range(size - 1):
        started = time.perf_counter()
        states.append(learner.fit(int(source[p]), batches[source[p]][0], batches[source[p]][1]))
        seconds.append(time.perf_counter() - started)
    seconds.append(float("nan"))
    err = np.full((size, size), np.nan)
    sub_loss: dict[tuple[int, int], np.ndarray] = {}
    for p, state in enumerate(states):
        for j in range(p + 1, size):
            x_test, y_test = batches[j][2], batches[j][3]
            err[p, j] = float(np.mean(learner.losses(state, x_test, y_test)))
            sub_loss[p, j] = np.asarray(learner.losses(state, points[j], _labels_for(batches[j], points[j])),
                                        dtype=np.float64)
        sub_loss[p, p] = np.asarray(
            learner.losses(state, points[source[p]], _labels_for(batches[source[p]], points[source[p]])),
            dtype=np.float64)
    matrix = PerformanceMatrix(
        err=err, offline=offline, points=points, queries=queries, sub_loss=sub_loss,
        feature_means=np.asarray(means), fit_seconds=np.asarray(seconds),
        rows=np.asarray([len(b[1]) + len(b[3]) for b in batches]), source=source)
    return matrix, states


def _labels_for(batch, points):
    """Labels of sampled pool rows: rebuild them by matching rows of the (train + test) pool."""
    x = np.concatenate([batch[0], batch[2]])
    y = np.concatenate([batch[1], batch[3]])
    index = {row.tobytes(): k for k, row in enumerate(x.astype(np.float64))}
    return y[[index[row.tobytes()] for row in points]]
