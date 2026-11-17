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
from typing import Callable, Iterable, Protocol

import numpy as np

N_POINTS = 512  # labeled points kept per batch for the CARA similarity-weighted staleness
N_QUERIES = 100  # query points per batch (CARA-D: sampled from the data stream itself)
POLICY_MISSING_FEATURES = "warmup_column_mean_or_zero"


def _finite_feature_stats(features):
    observed = np.isfinite(features)
    sums = features.sum(axis=0, dtype=np.float64, where=observed)
    return sums, observed.sum(axis=0)


def _geometry_sample(features, fill):
    return np.where(np.isfinite(features), features, fill)


class Learner(Protocol):
    """Trains one model per batch and scores any trained model on any batch."""

    def fit(self, index: int, x: np.ndarray, y: np.ndarray, *context: np.ndarray) -> object:
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
    # Paper replay has held-out errors for models serving their own online training batches.
    paper_timing: bool = False

    def __post_init__(self):
        size = self.err.shape[0]
        if self.err.shape != (size, size) or size < 4:
            raise ValueError("performance matrix needs at least four batches")
        if not 2 <= self.offline <= size - 2:
            raise ValueError("offline batches must leave at least two online batches")
        upper = np.triu_indices(size, 1)
        values = self.err[upper]
        # The operational matrix has no model for the last batch.
        observed = np.isfinite(values)
        if not observed.any() or np.any(values[observed] < 0) or np.any(values[observed] > 1):
            raise ValueError("matrix errors must be rates in [0, 1]")
        if self.paper_timing:
            diagonal = self.err.diagonal()[self.offline:]
            if not np.all((diagonal >= 0) & (diagonal <= 1)):
                raise ValueError("paper timing requires held-out errors for every online candidate")

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

    def kernel_gamma(self) -> float:
        """Pipeline RBF scale from raw offline features, separate from the CARA replication settings."""
        if not hasattr(self, "_gamma"):
            sample = np.concatenate(self.points[: self.offline])
            variance = float(sample.var())
            self._gamma = 1.0 / (sample.shape[1] * (variance if variance > 1e-12 else 1.0))
        return self._gamma

    def staleness(self, t: int, model: int) -> float:
        """Relative staleness of ``model`` for the queries of batch ``t`` (CARA, eq. 1-3).

        ``psi(Q_t, D_t, M) - psi(Q_t, D_model, M)`` with an RBF similarity, averaged over queries so
        it is an error-rate-like quantity that shares the unit of the retraining cost ``alpha``.
        """
        if not model < t:
            raise ValueError("staleness is defined for a model trained before batch t")
        queries = self.queries[t]
        gamma = self.kernel_gamma()

        query_norm = (queries**2).sum(axis=1)[:, None]

        def psi(batch: int, loss: np.ndarray) -> float:
            data = self.points[batch]
            distance = query_norm + (data**2).sum(axis=1)[None, :] - 2 * queries @ data.T
            return float((np.exp(-gamma * np.maximum(distance, 0.0)) * loss[None, :]).mean())

        return psi(t, self.sub_loss[model, t]) - psi(model, self.sub_loss[model, model])

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

    def _first_evaluation(self, model: int) -> int:
        return model if self._matrix.paper_timing and model >= self._matrix.offline else model + 1

    def err(self, model: int, batch: int) -> float:
        if model not in self.models or not self._first_evaluation(model) <= batch <= self.upto:
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
                for i in sorted(self.models) for j in range(self._first_evaluation(i), self.upto + 1)]


def build_matrix(
    stream: Iterable[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]],
    learner: Learner,
    *,
    offline: int | None = None,
    seed: int = 0,
    n_points: int = N_POINTS,
    n_queries: int = N_QUERIES,
    query_stream: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    workers: int = 1,
    on_block_complete: Callable[[int, int, int], None] | None = None,
) -> tuple[PerformanceMatrix, list[object]]:
    """Single pass over ordered batches; returns the matrix and every trained model state.

    The model of the final batch is never trained (it could not serve any later batch). With
    ``query_stream`` (CARA's "-S" streams) the queries are the given labeled points instead of a
    sample of the batch, and ``err`` is measured on them. ``workers > 1`` scores the trained models of
    one batch in threads (``learner.losses`` must be thread-safe). An optional third array in a
    batch (such as timestamps) is forwarded to the learner's fit only.
    The optional callback receives the number of completed blocks, processed rows
    and candidate models after each block, including the last scoring-only block.

    The RBF samples and feature means used by the policies fill missing values with
    the observed column mean in the warm-up batch, or zero when that column has no
    observations. This reference is fixed before any later batch is processed.
    The learner consumes the unchanged features, including missing values.
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
    processed_rows = 0
    feature_fill = None
    while pending is not None:
        x, y, *fit_context = pending
        pending = next(iterator, None)
        x = np.asarray(x)
        if not len(y) or len(x) != len(y):
            raise ValueError("every batch needs rows and matching labels")
        sums, observed = _finite_feature_stats(x)
        if feature_fill is None:
            feature_fill = np.divide(sums, observed, out=np.zeros_like(sums), where=observed > 0)
        point_index = rng.choice(len(y), size=min(n_points, len(y)), replace=False)
        if query_iterator is None:
            query_index = rng.choice(len(y), size=min(n_queries, len(y)), replace=False)
            queries.append(_geometry_sample(x[query_index], feature_fill))
            query_x = query_y = None
        else:
            query_x, query_y = (np.asarray(v) for v in next(query_iterator))
            queries.append(_geometry_sample(query_x, feature_fill))
        points.append(_geometry_sample(x[point_index], feature_fill))
        means.append((sums + (len(x) - observed) * feature_fill) / len(x))
        rows.append(len(y))
        def score(state, x=x, y=y, query_x=query_x, query_y=query_y):
            loss = np.asarray(learner.losses(state, x, y), dtype=np.float64)
            on_queries = loss if query_x is None else np.asarray(
                learner.losses(state, query_x, query_y), dtype=np.float64)
            return loss[point_index], float(on_queries.mean())

        if workers > 1 and len(states) > 1:
            with ThreadPoolExecutor(workers) as pool:
                scored = list(pool.map(score, states))
        else:
            scored = [score(state) for state in states]
        for i, (sampled_loss, error) in enumerate(scored):
            errs[i].append(error)
            sub_loss[i, j] = sampled_loss
        if pending is not None:
            started = time.perf_counter()
            state = learner.fit(j, x, y, *fit_context)
            seconds.append(time.perf_counter() - started)
            states.append(state)
            errs.append([])
            sub_loss[j, j] = np.asarray(
                learner.losses(state, x[point_index], y[point_index]), dtype=np.float64)
        else:
            seconds.append(float("nan"))
        j += 1
        if on_block_complete is not None:
            processed_rows += len(y)
            on_block_complete(j, processed_rows, len(states))
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
    separate_query_split: bool = False,
) -> tuple[PerformanceMatrix, list[object]]:
    """Train each candidate on its own batch and score it on held-out rows.

    The seven offline batches contribute the strict upper triangle (21 pairs in Table 4).
    Online candidates also have an observed diagonal after they have been deployed.
    With separate_query_split, CARA estimates staleness on labeled training rows
    and uses test features as unlabeled queries.
    """
    size = len(batches)
    if not 2 <= offline <= size - 2:
        raise ValueError("offline batches must leave at least two online batches")
    rng = np.random.default_rng(seed)
    points, queries, means, labels = [], [], [], []
    states, seconds = [], []
    for i, (x_train, y_train, x_test, y_test) in enumerate(batches):
        pool = np.concatenate([x_train, x_test])
        reference = x_train if separate_query_split else pool
        reference_labels = y_train if separate_query_split else np.concatenate([y_train, y_test])
        query_pool = x_test if separate_query_split else pool
        point_indices = rng.choice(len(reference), size=min(n_points, len(reference)), replace=False)
        query_indices = rng.choice(len(query_pool), size=min(n_queries, len(query_pool)), replace=False)
        points.append(reference[point_indices].astype(np.float64))
        labels.append(reference_labels[point_indices])
        queries.append(query_pool[query_indices].astype(np.float64))
        means.append(pool.mean(axis=0, dtype=np.float64))
        started = time.perf_counter()
        states.append(learner.fit(i, x_train, y_train))
        seconds.append(time.perf_counter() - started)
    err = np.full((size, size), np.nan)
    sub_loss: dict[tuple[int, int], np.ndarray] = {}
    for i, state in enumerate(states):
        for j in range(i, size):
            sub_loss[i, j] = np.asarray(learner.losses(state, points[j], labels[j]), dtype=np.float64)
            if j > i or i >= offline:
                err[i, j] = float(np.mean(learner.losses(state, batches[j][2], batches[j][3])))
    matrix = PerformanceMatrix(
        err=err, offline=offline, points=points, queries=queries, sub_loss=sub_loss,
        feature_means=np.asarray(means), fit_seconds=np.asarray(seconds),
        rows=np.asarray([len(b[1]) + len(b[3]) for b in batches]), paper_timing=True)
    return matrix, states
