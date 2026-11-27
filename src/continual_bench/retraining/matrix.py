"""Performance matrices for batch retraining decisions (CARA, Mahadevan & Mathioudakis 2024;
UPF, Regol et al. 2025).

``err[i, j]`` is the 0-1 error rate of the model trained at the end of batch ``i`` on the later
batch ``j``. Training at the end of batch ``i`` and deploying from ``i + 1`` keeps the protocol
test-then-train: no model is ever scored on the labels that trained it. CARA's same-batch
staleness-cost accounting is distinct from its shifted prequential accuracy evaluation.
The separate paper replay supplies a held-out slice of every batch for its diagonal errors.

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
    # Optional event-time contract. err remains a retrospective scoring matrix;
    # a MatrixView uses only losses whose labels have arrived at its decision cutoff.
    decision_cutoffs: np.ndarray | None = None
    label_available_at: list[np.ndarray] | None = None
    point_available_at: list[np.ndarray] | None = None
    row_loss: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    training_records: list[dict] = field(default_factory=list)

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
            diagonal = self.err.diagonal()[self.offline :]
            if not np.all((diagonal >= 0) & (diagonal <= 1)):
                raise ValueError("paper timing requires held-out errors for every online candidate")
        if self.decision_cutoffs is not None:
            if (
                self.label_available_at is None
                or self.point_available_at is None
                or len(self.decision_cutoffs) != size
                or len(self.label_available_at) != size
                or len(self.point_available_at) != size
                or np.any(np.diff(self.decision_cutoffs) <= 0)
            ):
                raise ValueError("event-time metadata must cover every ordered matrix block")

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

    def staleness(self, t: int, model: int, *, cutoff: int | None = None) -> float:
        """Relative staleness of ``model`` for the queries of batch ``t`` (CARA, eq. 1-3).

        ``psi(Q_t, D_t, M) - psi(Q_t, D_model, M)`` with an RBF similarity, averaged over queries so
        it is an error-rate-like quantity that shares the unit of the retraining cost ``alpha``.
        """
        if not model < t:
            raise ValueError("staleness is defined for a model trained before batch t")
        cache = self.__dict__.setdefault("_staleness_cache", {})
        key = (t, model, cutoff)
        if key in cache:
            return cache[key]
        queries = self.queries[t]
        gamma = self.kernel_gamma()

        query_norm = (queries**2).sum(axis=1)[:, None]

        def psi(batch: int, loss: np.ndarray) -> float:
            data = self.points[batch]
            if cutoff is not None and self.point_available_at is not None:
                visible = self.point_available_at[batch] <= cutoff
                if not visible.any():
                    return float("nan")
                data, loss = data[visible], loss[visible]
            distance = query_norm + (data**2).sum(axis=1)[None, :] - 2 * queries @ data.T
            return float((np.exp(-gamma * np.maximum(distance, 0.0)) * loss[None, :]).mean())

        value = psi(t, self.sub_loss[model, t]) - psi(model, self.sub_loss[model, model])
        cache[key] = value
        return value

    def error_at(self, model: int, batch: int, cutoff: int | None) -> float:
        """Observed label error, separate from the full retrospective scoring error."""
        if cutoff is None or self.label_available_at is None:
            return float(self.err[model, batch])
        visible = self.label_available_at[batch] <= cutoff
        if not visible.any():
            return float("nan")
        if visible.all():
            return float(self.err[model, batch])
        return float(self.row_loss[model, batch][visible].mean())

    def candidate_available(self, batch: int) -> bool:
        """An untrained candidate is retained for diagnostics, never promoted as a retrain."""
        return not self.training_records or (
            batch < len(self.training_records) and self.training_records[batch]["train_samples"] > 0
        )

    @property
    def admissible_retrains(self) -> frozenset[int]:
        """Candidate indices with released training labels, including a paper diagonal if present."""
        return frozenset(t for t in range(self.offline, self.batches) if self.candidate_available(t))

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

        result = {
            "offline": self.offline,
            "batches": self.batches,
            "err": clean(self.err),
            "fit_seconds": [None if not math.isfinite(v) else float(v) for v in self.fit_seconds],
            "rows": self.rows.tolist(),
        }
        if self.decision_cutoffs is not None:
            result["temporal_contract"] = {
                "version": "released-labels-v1",
                "decision_cutoffs": self.decision_cutoffs.tolist(),
                "decision_information": "only observed rows with available_at <= cutoff",
                "scoring_information": "full retrospective labels, never passed as decision errors",
                "candidate_training": self.training_records,
                "train_seq_range_convention": (
                    "half-open source sequence interval; reserved rows may be absent"
                ),
                "observed_label_counts_axes": ["decision_batch", "observed_batch"],
                "observed_error_columns": ["model_batch", "evaluated_batch", "error_on_released_labels"],
                "observed_label_counts": [
                    [
                        int(np.count_nonzero(self.label_available_at[batch] <= cutoff))
                        if batch <= upto
                        else None
                        for batch in range(self.batches)
                    ]
                    for upto, cutoff in enumerate(self.decision_cutoffs)
                ],
                "staleness_query_counts": [len(query) for query in self.queries],
                "observed_staleness_point_counts": [
                    [
                        int(np.count_nonzero(self.point_available_at[batch] <= cutoff))
                        if batch <= upto
                        else None
                        for batch in range(self.batches)
                    ]
                    for upto, cutoff in enumerate(self.decision_cutoffs)
                ],
                "observed_errors": [
                    {
                        "decision_batch": upto,
                        "cutoff_timestamp": int(self.decision_cutoffs[upto]),
                        "values": [list(item) for item in self.view(upto, range(upto)).observations()],
                    }
                    for upto in range(1, self.batches)
                ],
            }
        return result


class MatrixView:
    """What is observable after batch ``upto``: errors of *trained* models on batches ``<= upto``."""

    def __init__(self, matrix: PerformanceMatrix, upto: int, models: frozenset[int]):
        if not 0 <= upto < matrix.batches or not models or max(models) > upto:
            raise ValueError("a view cannot contain models trained after its horizon")
        self._matrix, self.upto, self.models = matrix, upto, models
        self.cutoff = None if matrix.decision_cutoffs is None else int(matrix.decision_cutoffs[upto])

    def _first_evaluation(self, model: int) -> int:
        return model if self._matrix.paper_timing and model >= self._matrix.offline else model + 1

    def err(self, model: int, batch: int) -> float:
        if model not in self.models or not self._first_evaluation(model) <= batch <= self.upto:
            raise ValueError("error not observable at this decision time")
        value = self._matrix.error_at(model, batch, self.cutoff)
        if not math.isfinite(value):
            raise ValueError("no labels observable at this decision time")
        return value

    def shift(self, batch: int) -> float:
        if batch > self.upto:
            raise ValueError("shift not observable at this decision time")
        return self._matrix.shift(batch)

    def staleness(self, t: int, model: int) -> float:
        if model not in self.models or t > self.upto:
            raise ValueError("staleness not observable at this decision time")
        return self._matrix.staleness(t, model, cutoff=self.cutoff)

    def observations(self) -> list[tuple[int, int, float]]:
        """All ``(model, batch, error)`` triples visible now: the UPF regression dataset."""
        return [
            (i, j, value)
            for i in sorted(self.models)
            for j in range(self._first_evaluation(i), self.upto + 1)
            if math.isfinite(value := self._matrix.error_at(i, j, self.cutoff))
        ]


@dataclass
class MatrixBuildState:
    """The completed columns of a streaming build and its sampling position."""

    states: list[object] = field(default_factory=list)
    errs: list[list[float]] = field(default_factory=list)
    points: list[np.ndarray] = field(default_factory=list)
    queries: list[np.ndarray] = field(default_factory=list)
    means: list[np.ndarray] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    sub_loss: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    feature_fill: np.ndarray | None = None
    rng_state: dict | None = None
    processed_rows: int = 0


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
    state: MatrixBuildState | None = None,
    checkpoint: Callable[[MatrixBuildState], None] | None = None,
    loss_dtype: type = np.float64,
    label_available_at: list[np.ndarray] | None = None,
    decision_cutoffs: list[int] | np.ndarray | None = None,
) -> tuple[PerformanceMatrix, list[object]]:
    """Single pass over ordered batches; returns the matrix and every trained model state.

    The model of the final batch is never trained (it could not serve any later batch). With
    ``query_stream`` (CARA's "-S" streams) the queries are the given labeled points instead of a
    sample of the batch, and ``err`` is measured on them. ``workers > 1`` scores the trained models of
    one batch in threads (``learner.losses`` must be thread-safe). An optional third array in a
    batch (such as timestamps) is forwarded to the learner's fit only.
    The optional callback receives the number of completed blocks, processed rows
    and candidate models after each block, including the last scoring-only block.
    When restoring ``state``, both input streams and the learner must start at the
    first unprocessed block. A checkpoint callback consumes the mutable state
    synchronously after each completed column.

    The RBF samples and feature means used by the policies fill missing values with
    the observed column mean in the warm-up batch, or zero when that column has no
    observations. This reference is fixed before any later batch is processed.
    The learner consumes the unchanged features, including missing values.
    """
    if (label_available_at is None) != (decision_cutoffs is None):
        raise ValueError("label availability and decision cutoffs must be supplied together")
    if label_available_at is not None and (state is not None or checkpoint is not None):
        raise ValueError("event-time matrix builds do not support partial matrix checkpoints")
    point_available_at = [] if label_available_at is not None else None
    row_loss: dict[tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    state = MatrixBuildState() if state is None else state
    if state.rng_state is not None:
        rng.bit_generator.state = state.rng_state
    states, errs = state.states, state.errs
    points, queries, means = state.points, state.queries, state.means
    seconds, rows = state.seconds, state.rows
    sub_loss = state.sub_loss
    iterator = iter(stream)
    query_iterator = None if query_stream is None else iter(query_stream)
    pending = next(iterator, None)
    j = len(rows)
    feature_fill = state.feature_fill
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
        if label_available_at is not None:
            if j >= len(label_available_at) or len(label_available_at[j]) != len(y):
                raise ValueError("label availability must match every batch row")
            point_available_at.append(np.asarray(label_available_at[j])[point_index])
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
            loss = np.asarray(learner.losses(state, x, y), dtype=loss_dtype)
            on_queries = (
                loss
                if query_x is None
                else np.asarray(learner.losses(state, query_x, query_y), dtype=loss_dtype)
            )
            return loss[point_index], float(on_queries.mean()), loss

        if workers > 1 and len(states) > 1:
            with ThreadPoolExecutor(workers) as pool:
                scored = list(pool.map(score, states))
        else:
            scored = [score(state) for state in states]
        for i, (sampled_loss, error, full_loss) in enumerate(scored):
            errs[i].append(error)
            sub_loss[i, j] = sampled_loss
            if label_available_at is not None:
                row_loss[i, j] = full_loss
        if pending is not None:
            started = time.perf_counter()
            candidate = learner.fit(j, x, y, *fit_context)
            seconds.append(time.perf_counter() - started)
            states.append(candidate)
            errs.append([])
            sub_loss[j, j] = np.asarray(
                learner.losses(candidate, x[point_index], y[point_index]), dtype=loss_dtype
            )
        else:
            seconds.append(float("nan"))
        j += 1
        state.processed_rows += len(y)
        if checkpoint is not None:
            state.feature_fill = feature_fill
            state.rng_state = rng.bit_generator.state
            checkpoint(state)
        if on_block_complete is not None:
            on_block_complete(j, state.processed_rows, len(states))
    size = j
    err = np.full((size, size), np.nan)
    for i, row in enumerate(errs):
        err[i, i + 1 : i + 1 + len(row)] = row
    if offline is None:
        offline = max(2, size // 3)
    matrix = PerformanceMatrix(
        err=err,
        offline=offline,
        points=points,
        queries=queries,
        sub_loss=sub_loss,
        feature_means=np.asarray(means),
        fit_seconds=np.asarray(seconds),
        rows=np.asarray(rows),
        label_available_at=label_available_at,
        point_available_at=point_available_at,
        decision_cutoffs=None if decision_cutoffs is None else np.asarray(decision_cutoffs, dtype=np.int64),
        row_loss=row_loss,
        training_records=getattr(learner, "training_records", []),
    )
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
    completed_rows: set[int] | None = None,
    err_init: np.ndarray | None = None,
    sub_loss_init: dict[tuple[int, int], np.ndarray] | None = None,
    on_row_complete: Callable[[int, np.ndarray, dict[tuple[int, int], np.ndarray]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
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
        if stop_requested is not None and stop_requested():
            raise InterruptedError(f"Spot interruption during UPF fit at batch {i}")
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
    err = err_init.copy() if err_init is not None else np.full((size, size), np.nan)
    sub_loss: dict[tuple[int, int], np.ndarray] = dict(sub_loss_init) if sub_loss_init is not None else {}
    done_rows = set(completed_rows) if completed_rows is not None else set()
    for i, state in enumerate(states):
        if i in done_rows:
            continue
        for j in range(i, size):
            sub_loss[i, j] = np.asarray(learner.losses(state, points[j], labels[j]), dtype=np.float64)
            if j > i or i >= offline:
                err[i, j] = float(np.mean(learner.losses(state, batches[j][2], batches[j][3])))
        done_rows.add(i)
        if on_row_complete is not None:
            on_row_complete(i, err, sub_loss)
    matrix = PerformanceMatrix(
        err=err,
        offline=offline,
        points=points,
        queries=queries,
        sub_loss=sub_loss,
        feature_means=np.asarray(means),
        fit_seconds=np.asarray(seconds),
        rows=np.asarray([len(b[1]) + len(b[3]) for b in batches]),
        paper_timing=True,
    )
    return matrix, states
