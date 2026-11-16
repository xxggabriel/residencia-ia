"""Cost-aware model updates (CARA, UPF, Oracle) on the batch-prequential Zia streams.

The stream is read once to train the model that a retrain at the end of every block would deploy
(the same replay window as ``periodic``) and to score every trained model on every later block. That
performance matrix feeds the offline phase of the policies and the Oracle; the online run then simply
swaps in the pre-trained model whenever the policy retrains, so the evaluated stream realises the
matrix exactly (checked in :func:`verify_realised`).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from continual_bench.retraining import PerformanceMatrix, SimResult, build_matrix, simulate
from continual_bench.retraining.analysis import analyze, deployed_metrics, json_parameters
from continual_bench.retraining.rules import Oracle

from .zia_metrics import decide

DEFAULTS = {"offline_blocks": None, "mc_samples": 100, "delta": 0.95, "n_alphas": 10, "seed": 0,
            "comparison_rules": ["cara", "cara_cumulative", "cara_periodic", "upf", "oracle", "frozen"]}


def settings(tracking_settings: dict) -> dict:
    merged = {**DEFAULTS, **(tracking_settings.get("retraining") or {})}
    if merged["offline_blocks"] is not None and merged["offline_blocks"] < 2:
        raise ValueError("retraining.offline_blocks must be at least 2")
    return merged


class ZiaLearner:
    """Model trained at the end of block ``i`` on the last ``replay`` blocks (block 0 stands alone)."""

    def __init__(self, model, threshold, config, replay: int):
        self.model, self.threshold, self.config = model, threshold, config
        self.window: deque = deque(maxlen=replay)
        self.rows = 0

    def fit(self, index, x, y):
        start = self.rows
        self.rows += len(y)
        if index:
            self.window.append((start, x, y))
            fit_start = self.window[0][0]
            fit_x = np.concatenate([w[1] for w in self.window])
            fit_y = np.concatenate([w[2] for w in self.window])
        else:
            fit_start, fit_x, fit_y = start, x, y
        timer = time.monotonic()
        iterations = self.model.fit(fit_x, fit_y, False, config=self.config, tracker=None)
        seconds = time.monotonic() - timer
        payload, extension = self.model.payload()
        return {"payload": payload, "extension": extension, "seconds": seconds, "iterations": iterations,
                "start": fit_start, "rows": len(fit_y), "index": index}

    def losses(self, state, x, y):
        self.model.load(state["payload"])
        return (decide(self.model.predict(x), self.threshold) != y).astype(np.float64)


def stream_matrix(blocks, model, threshold, config, replay, options) -> tuple[PerformanceMatrix, list[dict]]:
    matrix, states = build_matrix(
        ((x, y) for x, y in blocks), ZiaLearner(model, threshold, config, replay),
        offline=options["offline_blocks"], seed=options["seed"])
    return matrix, states


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
    return SimResult(policy.kind, policy.alpha, list(policy.retrains), served, errors,
                     policy.rule.parameters(), list(getattr(policy.rule, "forecasts", [])))


def finalize(policy, matrix: PerformanceMatrix, states: list[dict], options: dict) -> tuple[dict, dict]:
    """Paper-style analysis and the flat scalar metrics of the policy that ran."""
    analysis = analyze(matrix, rules=tuple(options["comparison_rules"]), n_alphas=options["n_alphas"],
                       focus_alpha=policy.alpha, delta=options["delta"], samples=options["mc_samples"],
                       seed=options["seed"])
    result = simulation_of(policy, matrix)
    oracle = simulate(Oracle(), matrix, policy.alpha)
    metrics = deployed_metrics(analysis, policy.kind if policy.kind in analysis["rules"] else "oracle",
                               result, oracle)
    metrics["retraining_matrix_fit_seconds"] = float(np.nansum(matrix.fit_seconds))
    metrics["retraining_offline_batches"] = matrix.offline
    metrics["retraining_online_batches"] = matrix.horizon
    analysis["deployed"] = {
        "policy": policy.kind, "alpha": policy.alpha, "retrain_points": list(policy.retrains),
        "served_by": result.deployed, "parameters": json_parameters(result.parameters),
        "oracle_retrain_points": oracle.retrains,
        "model_rows": [s["rows"] for s in states],
    }
    return analysis, metrics
