"""Cost-aware model updates (CARA, UPF, Oracle) on the batch-prequential Zia streams.

The full-retention path stores candidates while reading the stream once. The metrics-only path
reopens the immutable stream and scores one candidate at a time, without persisting model state.
Both produce the same replay-window performance matrix for the policies and the Oracle. A promoted
candidate is reconstructed and hash-checked in metrics-only mode; that additional physical training
is recorded separately from the logical deployed candidate (see :func:`verify_realised`).
"""

from __future__ import annotations

import gzip
import json
import time
from collections import deque
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from continual_bench.retraining import PerformanceMatrix, SimResult, build_matrix, simulate
from continual_bench.retraining.analysis import analyze, deployed_metrics, json_parameters
from continual_bench.retraining.matrix import POLICY_MISSING_FEATURES
from continual_bench.retraining.rules import Oracle

from .zia_metrics import decide

MATRIX_CACHE_SCHEMA = "zia-matrix-metrics-v1"


def write_matrix_metrics_cache(path, contract, matrix, states, history, source_execution):
    """Persist derived measurements only, never points, labels, payloads or weights."""
    path = Path(path)
    if path.is_symlink() or path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("matrix metric cache must be an ordinary .json.gz file")
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_data = matrix.to_dict()
    stale = matrix.staleness_matrix()
    document = {
        "schema": MATRIX_CACHE_SCHEMA,
        "contract": contract,
        "source_execution": source_execution,
        "matrix": {
            **matrix_data,
            "feature_means": matrix.feature_means.tolist(),
            "staleness": [[float(v) if np.isfinite(v) else None for v in row] for row in stale],
        },
        "states": [
            {
                **{k: v for k, v in state.items() if k not in {"payload", "fit"}},
                "fit_id": state["fit"]["fit_id"],
            }
            for state in states
        ],
        "fits": history.fits,
    }
    with TemporaryDirectory(dir=path.parent, prefix=".matrix-metrics-") as scratch:
        candidate = Path(scratch) / "cache.json.gz"
        with candidate.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            import io

            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                json.dump(document, stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
        if path.exists():
            raise FileExistsError("matrix metric cache already exists")
        candidate.replace(path)


def read_matrix_metrics_cache(path, contract, history):
    path = Path(path)
    if path.is_symlink() or path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("matrix metric cache must be an ordinary .json.gz file")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema") != MATRIX_CACHE_SCHEMA or document.get("contract") != contract:
        raise ValueError("matrix metric cache differs from the execution contract")
    value = document["matrix"]
    err = np.asarray(value["err"], dtype=np.float64)
    rows = np.asarray(value["rows"], dtype=np.int64)
    size = len(rows)
    stale = np.asarray(value["staleness"], dtype=np.float64)
    upper = np.triu_indices(size, 1)
    if (
        err.shape != (size, size)
        or stale.shape != (size, size)
        or not np.all(np.isfinite(err[upper]))
        or not np.all(np.isfinite(stale.T[upper]))
        or np.any(rows <= 0)
        or len(document["states"]) != size - 1
        or len(document["fits"]) != size - 1
    ):
        raise ValueError("matrix metric cache is incomplete")
    matrix = PerformanceMatrix(
        err=err,
        offline=value["offline"],
        points=[],
        queries=[],
        sub_loss={},
        feature_means=np.asarray(value["feature_means"]),
        fit_seconds=np.asarray(value["fit_seconds"], dtype=np.float64),
        rows=rows,
    )
    matrix._staleness_all = stale
    matrix._staleness_cache = {(t, m, None): float(stale[t, m]) for t in range(1, size) for m in range(t)}
    fits = document["fits"]
    for index, fit in enumerate(fits, start=1):
        if fit["fit_id"] != f"fit-{index:06d}" or fit["checkpoint"]["file"] is not None:
            raise ValueError("matrix metric cache contains invalid candidate metadata")
        fit.update(execution_origin="reused_metric_evidence", source_matrix=document["source_execution"])
    history.fits = fits
    history.global_step = max((row["global_step"] for fit in fits for row in fit["steps"]), default=0)
    by_id = {fit["fit_id"]: fit for fit in fits}
    states = document["states"]
    for state in states:
        if state["file"] is not None or "payload" in state:
            raise ValueError("matrix metric cache contains a model payload")
        state["fit"] = by_id[state.pop("fit_id")]
    return matrix, states, document["source_execution"]


DEFAULTS = {
    "offline_blocks": None,
    "mc_samples": 100,
    "delta": 0.95,
    "n_alphas": 10,
    "seed": 0,
    "comparison_rules": ["cara", "cara_cumulative", "cara_periodic", "upf", "oracle", "frozen"],
}


def settings(tracking_settings: dict) -> dict:
    merged = {**DEFAULTS, **(tracking_settings.get("retraining") or {})}
    if merged["offline_blocks"] is not None and merged["offline_blocks"] < 2:
        raise ValueError("retraining.offline_blocks must be at least 2")
    return merged


class ZiaLearner:
    """Model trained at the end of block ``i`` on the last ``replay`` blocks (block 0 stands alone)."""

    def __init__(
        self, model, threshold, config, replay: int, history, scratch, tracker, *, artifact_mode="full"
    ):
        self.model, self.threshold, self.config = model, threshold, config
        self.history, self.scratch, self.tracker = history, scratch, tracker
        self.window: deque = deque(maxlen=replay)
        self.rows = 0
        self.artifact_mode = artifact_mode
        self.active_state = None

    def fit(self, index, x, y, timestamps):
        start = self.rows
        self.rows += len(y)
        if index:
            self.window.append((start, x, y, timestamps))
            fit_start = self.window[0][0]
            fit_x = np.concatenate([w[1] for w in self.window])
            fit_y = np.concatenate([w[2] for w in self.window])
            fit_timestamps = np.concatenate([w[3] for w in self.window])
        else:
            fit_start, fit_x, fit_y, fit_timestamps = start, x, y, timestamps
        self.history.check_interrupted()
        timer = time.monotonic()
        self.model.fit(
            fit_x,
            fit_y,
            False,
            config=self.config,
            tracker=self.tracker,
            timestamps=fit_timestamps,
            capture_sigterm=self.history.stop_requested is None,
            stop_requested=self.history.stop_requested,
        )
        fit = self.history.capture(
            self.model.training_summary,
            action="candidate",
            block_index=index,
            start=fit_start,
            timestamps=fit_timestamps,
            seconds=time.monotonic() - timer,
            current_block_start=start,
        )
        payload, extension = self.model.payload()
        path = self.scratch / f"{fit['fit_id']}.{extension}" if self.artifact_mode == "full" else None
        if path is not None:
            path.write_bytes(payload)
        fit["checkpoint"] = {"file": None, "sha256": sha256(payload).hexdigest(), "bytes": len(payload)}
        self.history.save_partial(fit)
        self.history.check_interrupted()
        state = {
            "file": path.name if path is not None else None,
            "fit": fit,
            "rows": len(fit_y),
            "extension": extension,
        }
        if path is None:
            state["payload"] = payload
            fit["checkpoint_unavailable_reason"] = "metrics_only_no_persisted_model_state"
        self.active_state = state
        return state

    def losses(self, state, x, y):
        self.history.check_interrupted()
        if self.active_state is not state:
            self.model.load(
                state["payload"] if state.get("file") is None else (self.scratch / state["file"]).read_bytes()
            )
            self.active_state = state
        return decide(self.model.predict(x), self.threshold) != y


def bounded_stream_matrix(
    blocks_factory, model, threshold, config, replay, options, *, history, tracker=None, progress=None
) -> tuple[PerformanceMatrix, list[dict]]:
    """Build identical metric coordinates while retaining only one native candidate.

    ``blocks_factory(start_row)`` reopens an immutable ordered stream at a completed
    block boundary. No learner state or weights are written, and no full dataset
    is materialized. Reading rows again is explicit computational work; only the
    compact geometry samples, errors and per-fit evidence survive each candidate.
    """
    from continual_bench.retraining.matrix import (
        N_POINTS,
        N_QUERIES,
        _finite_feature_stats,
        _geometry_sample,
    )

    rng = np.random.default_rng(options["seed"])
    points, queries, means, rows, indices = [], [], [], [], []
    fill = None
    for x, y, *_ in blocks_factory(0):
        history.check_interrupted()
        sums, observed = _finite_feature_stats(x)
        if fill is None:
            fill = np.divide(sums, observed, out=np.zeros_like(sums), where=observed > 0)
        chosen = rng.choice(len(y), size=min(N_POINTS, len(y)), replace=False)
        query = rng.choice(len(y), size=min(N_QUERIES, len(y)), replace=False)
        indices.append(chosen)
        points.append(_geometry_sample(x[chosen], fill))
        queries.append(_geometry_sample(x[query], fill))
        means.append((sums + (len(x) - observed) * fill) / len(x))
        rows.append(len(y))
    size = len(rows)
    if size < 4:
        raise ValueError("performance matrix needs at least four batches")
    err = np.full((size, size), np.nan)
    seconds = np.full(size, np.nan)
    sub_loss, states = {}, []
    offsets = np.concatenate(([0], np.cumsum(rows)))
    learner = ZiaLearner(
        model, threshold, config, replay, history, None, tracker, artifact_mode="metrics_only"
    )
    scoring_rows = 0
    for i, (x, y, *context) in enumerate(blocks_factory(0)):
        history.check_interrupted()
        if i == size - 1:
            break
        timer = time.perf_counter()
        candidate = learner.fit(i, x, y, *context)
        seconds[i] = time.perf_counter() - timer
        sub_loss[i, i] = np.asarray(learner.losses(candidate, x[indices[i]], y[indices[i]]), dtype=np.bool_)
        for j, (future_x, future_y, *_) in enumerate(blocks_factory(int(offsets[i + 1])), start=i + 1):
            history.check_interrupted()
            if j >= size or len(future_y) != rows[j]:
                raise ValueError("reopened matrix stream changed its block boundaries")
            loss = np.asarray(learner.losses(candidate, future_x, future_y), dtype=np.bool_)
            err[i, j] = float(loss.mean())
            sub_loss[i, j] = loss[indices[j]].copy()
            scoring_rows += len(future_y)
        if j != size - 1:
            raise ValueError("reopened matrix stream lost rows")
        candidate.pop("payload")
        states.append(candidate)
        learner.active_state = None
        if progress is not None:
            progress(
                int(offsets[i + 1]),
                {
                    "matrix/blocks_completed": i + 1,
                    "matrix/rows_processed": int(offsets[i + 1]),
                    "matrix/candidate_models": len(states),
                    "matrix/model_block_pairs": len(sub_loss) - len(states),
                    "matrix/rows_scored": scoring_rows,
                },
            )
    if progress is not None:
        progress(
            int(offsets[-1]),
            {
                "matrix/blocks_completed": size,
                "matrix/rows_processed": int(offsets[-1]),
                "matrix/candidate_models": len(states),
                "matrix/model_block_pairs": size * (size - 1) // 2,
                "matrix/rows_scored": scoring_rows,
            },
        )
    matrix = PerformanceMatrix(
        err=err,
        offline=options["offline_blocks"] or max(2, size // 3),
        points=points,
        queries=queries,
        sub_loss=sub_loss,
        feature_means=np.asarray(means),
        fit_seconds=seconds,
        rows=np.asarray(rows),
    )
    return matrix, states


def stream_matrix(
    blocks,
    model,
    threshold,
    config,
    replay,
    options,
    *,
    history,
    scratch,
    tracker=None,
    progress=None,
    resume=None,
    checkpoint=None,
) -> tuple[PerformanceMatrix, list[dict]]:
    learner = ZiaLearner(model, threshold, config, replay, history, scratch, tracker)
    state = None
    if resume is not None:
        state, learner.rows, learner.window = resume

    def report_block(blocks_completed, rows_processed, candidates):
        progress(
            rows_processed,
            {
                "matrix/blocks_completed": blocks_completed,
                "matrix/rows_processed": rows_processed,
                "matrix/candidate_models": candidates,
                "matrix/model_block_pairs": blocks_completed * (blocks_completed - 1) // 2,
            },
        )

    return build_matrix(
        blocks,
        learner,
        offline=options["offline_blocks"],
        seed=options["seed"],
        on_block_complete=report_block if progress is not None else None,
        state=state,
        checkpoint=(
            (lambda state: checkpoint((state, learner.rows, learner.window)))
            if checkpoint is not None
            else None
        ),
        loss_dtype=np.bool_,
    )


def verify_realised(matrix: PerformanceMatrix, windows: list[dict], policy) -> None:
    """The evaluated stream must reproduce the matrix errors of the deployed models."""
    deployed = matrix.initial
    retrains = set(policy.retrains)
    stream = [w for w in windows if w["scope"] == "prequential"]
    for j, window in enumerate(stream, start=1):
        if j >= matrix.offline:
            if abs(window["error_rate"] - matrix.err[deployed, j]) > 1e-9:
                raise ValueError("retraining matrix disagrees with the evaluated stream")
            if j in retrains:
                deployed = j


def simulation_of(policy, matrix: PerformanceMatrix) -> SimResult:
    """The schedule the run actually executed, expressed as a simulation result."""
    deployed, served, errors = matrix.initial, [], []
    retrains = set(policy.retrains)
    for j in range(matrix.offline, matrix.batches):
        served.append(deployed)
        errors.append(float(matrix.err[deployed, j]))
        if j in retrains:
            deployed = j
    return SimResult(
        policy.kind,
        policy.alpha,
        list(policy.retrains),
        served,
        errors,
        policy.rule.parameters(),
        list(getattr(policy.rule, "forecasts", [])),
    )


def finalize(policy, matrix: PerformanceMatrix, states: list[dict], options: dict) -> tuple[dict, dict]:
    """Paper-style analysis and the flat scalar metrics of the policy that ran."""
    analysis = analyze(
        matrix,
        rules=tuple(options["comparison_rules"]),
        n_alphas=options["n_alphas"],
        focus_alpha=policy.alpha,
        delta=options["delta"],
        samples=options["mc_samples"],
        seed=options["seed"],
    )
    analysis["setup"]["policy_missing_features"] = POLICY_MISSING_FEATURES
    result = simulation_of(policy, matrix)
    oracle = simulate(Oracle(), matrix, policy.alpha)
    metrics = deployed_metrics(
        analysis, policy.kind if policy.kind in analysis["rules"] else "oracle", result, oracle
    )
    metrics["retraining_matrix_fit_seconds"] = float(np.nansum(matrix.fit_seconds))
    metrics["retraining_offline_batches"] = matrix.offline
    metrics["retraining_online_batches"] = matrix.horizon
    analysis["deployed"] = {
        "policy": policy.kind,
        "alpha": policy.alpha,
        "retrain_points": list(policy.retrains),
        "served_by": result.deployed,
        "parameters": json_parameters(result.parameters),
        "oracle_retrain_points": oracle.retrains,
        "model_rows": [s["rows"] for s in states],
    }
    return analysis, metrics
