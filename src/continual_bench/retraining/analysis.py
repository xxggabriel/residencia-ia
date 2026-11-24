"""Paper-style analysis of a performance matrix: cost-vs-alpha curves, AUC, Oracle-relative errors,
the Proposition 3.1 bound, robustness to a wrong alpha and forecaster quality.

Everything returned is plain JSON (finite floats or ``None``) so it can travel inside a run report.
"""

from __future__ import annotations

import math

import numpy as np

from .cara_reference import strategy_scpe
from .matrix import PerformanceMatrix
from .rules import UPF, CaraCT, CaraP, CaraT, Frozen, Markov, Oracle, StalenessOracle
from .schedule import (
    alpha_for_retrain_bound,
    alpha_max,
    alpha_max_same_batch,
    lipschitz_bound,
    optimal_schedule,
    optimal_schedule_same_batch,
    retrain_upper_bound,
)
from .simulate import SimResult, simulate

LABELS = {
    "cara": "CARA",
    "cara_cumulative": "CARA cumul.",
    "cara_periodic": "CARA per.",
    "upf": "UPF",
    "oracle": "oracle",
    "frozen": "frozen",
    "markov": "Markov",
    "oracle_staleness": "oracle (CARA)",
}
DEFAULT_RULES = ("cara", "cara_cumulative", "cara_periodic", "upf", "oracle", "frozen")
# Fractions of alpha_max (UPF Table 2 uses alpha = 0.1 and 0.9 on a dataset with alpha_max = 1).
LOW_HIGH = {"low": 0.1, "high": 0.9}


def _num(value):
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def json_parameters(parameters: dict) -> dict:
    """Rule parameters with non-finite values (``inf`` = never retrain) as ``None``."""
    return {
        k: (_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        for k, v in parameters.items()
    }


def make_rule(name: str, *, delta: float = 0.95, samples: int = 100, seed: int = 0, same_batch: bool = False):
    if name == "upf":
        return UPF(delta=delta, samples=samples, seed=seed)
    factories = {
        "cara": CaraT,
        "cara_cumulative": CaraCT,
        "cara_periodic": CaraP,
        "oracle_staleness": StalenessOracle,
    }
    if name in factories:
        return factories[name](same_batch=same_batch)
    return {"oracle": Oracle, "frozen": Frozen, "markov": Markov}[name]()


def trapezoid(xs, ys) -> float | None:
    if len(xs) < 2 or any(y is None for y in ys):
        return None
    return float(sum((b - a) * (v + w) / 2 for (a, v), (b, w) in zip(zip(xs, ys), zip(xs[1:], ys[1:]))))


def forecast_quality(results: list[SimResult], matrix: PerformanceMatrix) -> dict:
    """MAE and bias of the UPF mean forecasts against the entries realised later (UPF appendix A.7)."""
    deltas = []
    for result in results:
        for forecast in result.forecasts:
            deployed = forecast["deployed"]
            for i, j, mean in zip(
                forecast["trained"], forecast["served"], forecast["predicted_new"], strict=True
            ):
                if math.isfinite(matrix.err[i, j]):
                    deltas.append(mean - matrix.err[i, j])
            for j, mean in zip(sorted(set(forecast["served"])), forecast["predicted_old"], strict=True):
                if math.isfinite(matrix.err[deployed, j]):
                    deltas.append(mean - matrix.err[deployed, j])
    if not deltas:
        return {"mae": None, "bias": None, "count": 0}
    deltas = np.asarray(deltas)
    return {"mae": float(np.abs(deltas).mean()), "bias": float(deltas.mean()), "count": len(deltas)}


def forecast_pairs(result: SimResult, matrix: PerformanceMatrix, limit: int = 2000) -> list[list[float]]:
    """``[forecast, realised]`` pairs for the scatter of UPF forecasts against what happened."""
    pairs = []
    for forecast in result.forecasts:
        new = zip(forecast["trained"], forecast["served"], forecast["predicted_new"], strict=True)
        for i, j, mean in new:
            if math.isfinite(matrix.err[i, j]):
                pairs.append([float(mean), float(matrix.err[i, j])])
        for j, mean in zip(sorted(set(forecast["served"])), forecast["predicted_old"], strict=True):
            if math.isfinite(matrix.err[forecast["deployed"], j]):
                pairs.append([float(mean), float(matrix.err[forecast["deployed"], j])])
    stride = max(1, len(pairs) // limit)
    return pairs[::stride]


def analyze(
    matrix: PerformanceMatrix,
    *,
    rules=DEFAULT_RULES,
    n_alphas: int = 10,
    alpha_top: float | None = None,
    alpha_grid: tuple[float, ...] | None = None,
    focus_alpha: float | None = None,
    delta: float = 0.95,
    samples: int = 100,
    seed: int = 0,
    same_batch: bool = False,
) -> dict:
    """``same_batch`` switches the CARA rules and the strategy Oracle to the paper's accounting."""
    first, last = matrix.offline, matrix.batches - 1
    allowed = matrix.admissible_retrains
    decision_candidates = set(range(first, last + 1 if matrix.paper_timing or same_batch else last))
    bound_available = decision_candidates.issubset(allowed)
    if matrix.paper_timing:
        top = alpha_max_same_batch(
            matrix.err.T,
            matrix.initial,
            first,
            last,
            retrain_errors=matrix.err.diagonal(),
            allowed_retrains=allowed,
        )
    else:
        top = alpha_max(matrix.err, matrix.initial, first, last, allowed_retrains=allowed)
    oracle_top = top
    if alpha_top is not None:  # fixed range from the paper (UPF Table 4) instead of the oracle's own
        top = float(alpha_top)
    if alpha_grid is None:
        alphas = [float(a) for a in np.linspace(0.0, top, n_alphas)] if top > 0 else [0.0, 1.0]
    else:
        alphas = [float(a) for a in alpha_grid]
        if (
            len(alphas) < 2
            or alphas[0] != 0
            or not all(math.isfinite(a) for a in alphas)
            or any(a >= b for a, b in zip(alphas, alphas[1:]))
        ):
            raise ValueError("alpha_grid must start at zero and be finite and strictly increasing")
        top = alphas[-1]
    focus = alphas[len(alphas) // 2] if focus_alpha is None else float(focus_alpha)
    all_alphas = sorted({*alphas, focus, *(top * f for f in LOW_HIGH.values())})
    rules = tuple(dict.fromkeys(("oracle", *rules)))
    sims: dict[tuple[str, float], SimResult] = {}
    for name in rules:
        for alpha in all_alphas:
            sims[name, alpha] = simulate(
                make_rule(
                    name,
                    delta=delta,
                    samples=samples,
                    seed=seed,
                    same_batch=same_batch or matrix.paper_timing,
                ),
                matrix,
                alpha,
                same_batch=same_batch and name != "upf" and name != "oracle",
            )
    staleness = matrix.staleness_matrix()
    stale_cost = np.nan_to_num(staleness.T, nan=0.0)
    strategy_oracle = {}
    for alpha in all_alphas:
        if "oracle_staleness" in rules:
            value = sims["oracle_staleness", alpha].strategy_cost(matrix)
        elif same_batch or matrix.paper_timing:
            _, value = optimal_schedule_same_batch(
                stale_cost.T, matrix.initial, first, last, alpha, allowed_retrains=allowed
            )
        else:
            _, value = optimal_schedule(
                stale_cost, matrix.initial, first, last, alpha, allowed_retrains=allowed
            )
        strategy_oracle[alpha] = value
    lipschitz = lipschitz_bound(matrix.err, first)
    horizon = matrix.horizon  # T of Proposition 3.1

    def scpe(cost, reference):
        return None if reference is None or reference <= 0 else 100.0 * (cost - reference) / reference

    def row(name, alpha):
        result = sims[name, alpha]
        oracle = sims["oracle", alpha]
        strategy = result.strategy_cost(matrix)
        return {
            "alpha": alpha,
            "cost": _num(result.cost),
            "retrains": len(result.retrains),
            "retrain_points": result.retrains,
            "average_accuracy": _num(result.average_accuracy),
            "scpe": _num(scpe(result.cost, oracle.cost)),
            "strategy_cost": _num(strategy),
            "error_oracle_gap": _num(result.cost - oracle.cost),
            "strategy_oracle_gap": _num(strategy - strategy_oracle[alpha]),
            "oracle_strategy_cost": _num(strategy_oracle[alpha]),
            "strategy_scpe": _num(strategy_scpe(strategy, strategy_oracle[alpha])),
            "parameters": json_parameters(result.parameters),
        }

    report_rules, tables = {}, {"low": {}, "high": {}}
    for name in rules:
        curve = [row(name, a) for a in alphas]
        report_rules[name] = {
            "label": LABELS[name],
            "curve": curve,
            "auc": _num(trapezoid(alphas, [c["cost"] for c in curve])),
            "mean_scpe": _num(np.mean([c["scpe"] for c in curve if c["scpe"] is not None]))
            if any(c["scpe"] is not None for c in curve)
            else None,
            "mean_strategy_scpe": _num(
                np.mean([c["strategy_scpe"] for c in curve if c["strategy_scpe"] is not None])
            )
            if any(c["strategy_scpe"] is not None for c in curve)
            else None,
            "online_decision_seconds": _num(
                np.mean([t for a in alphas for t in sims[name, a].decision_seconds] or [0.0])
            ),
            "offline_seconds": _num(np.mean([sims[name, a].offline_seconds for a in alphas])),
        }
        for key, fraction in LOW_HIGH.items():
            tables[key][name] = row(name, top * fraction)
    wrong = {}
    for name in ("cara", "upf"):
        if name not in rules:
            continue
        grid = []
        for provided in alphas:
            schedule = sims[name, provided]
            grid.append(
                [
                    None
                    if (best := sims[name, real].cost) <= 0
                    else _num(100.0 * ((real * len(schedule.retrains) + sum(schedule.errors)) - best) / best)
                    for real in alphas
                ]
            )
        wrong[name] = {"provided": alphas, "real": alphas, "pct_increase": grid}
    forecast = forecast_quality([sims["upf", a] for a in alphas], matrix) if "upf" in rules else None
    if forecast is not None:
        forecast["pairs"] = forecast_pairs(sims["upf", focus], matrix)
    cumulative = {}
    for name in rules:
        result = sims[name, focus]
        running, series = 0.0, []
        retrains = set(result.retrains)
        for j, deployed in zip(range(first, matrix.batches), result.deployed, strict=True):
            stale = 0.0 if deployed == j else matrix.staleness(j, deployed)
            running += stale + (focus if j in retrains else 0.0)
            series.append(_num(running))
        cumulative[name] = series
    return {
        "setup": {
            "batches": matrix.batches,
            "offline_batches": matrix.offline,
            "online_batches": matrix.horizon,
            "decision_points": matrix.horizon if matrix.paper_timing else matrix.horizon - 1,
            "oracle_alpha_max": _num(oracle_top),
            "initial_model": matrix.initial,
            "admissible_retrain_points": sorted(decision_candidates.intersection(allowed)),
            "retrain_bound_available": bound_available,
            "retrain_bound_unavailable_reason": None if bound_available else "restricted_candidate_support",
            "alpha_max": _num(top),
            "focus_alpha": focus,
            "lipschitz_L": _num(lipschitz),
            "delta": delta,
            "samples": samples,
            "timing": (
                "UPF decides before batch t; a retrain serves its held-out test split"
                if matrix.paper_timing
                else "CARA accounting: a retrain at t serves t with staleness 0"
                if same_batch
                else "retrain after batch t serves t+1 onward (test-then-train)"
            ),
            "bound_alphas": {
                f"r<={r}": _num(alpha_for_retrain_bound(r, lipschitz, horizon)) if bound_available else None
                for r in range(3)
            },
            "retrain_upper_bound": {
                str(a): _num(retrain_upper_bound(a, lipschitz, horizon)) if bound_available else None
                for a in alphas
            },
        },
        "alphas": alphas,
        "rules": report_rules,
        "table": tables,
        "wrong_alpha": wrong,
        "forecast": forecast,
        "cumulative_strategy_cost": cumulative,
        "staleness_matrix": [[_num(v) for v in r] for r in staleness],
        "performance_matrix": matrix.to_dict(),
        "oracle_schedule_at_focus": sims["oracle", focus].retrains,
    }


def deployed_metrics(analysis: dict, rule: str, result: SimResult, oracle: SimResult) -> dict:
    """Scalar metrics of one deployed schedule, with the curve-derived ones from ``analysis``."""
    curve = analysis["rules"][rule]
    setup = analysis["setup"]
    cost = _num(result.cost)
    out = {
        "retraining_alpha": result.alpha,
        "retraining_c_hat": cost,
        "retraining_online_retrains": len(result.retrains),
        "retraining_average_accuracy": _num(result.average_accuracy),
        "retraining_oracle_c_hat": _num(oracle.cost),
        "retraining_oracle_retrains": len(oracle.retrains),
        "retraining_alpha_max": setup["alpha_max"],
        "retraining_lipschitz_L": setup["lipschitz_L"],
        "retraining_retrain_upper_bound": _num(
            retrain_upper_bound(result.alpha, setup["lipschitz_L"] or 0.0, setup["online_batches"])
        )
        if setup.get("retrain_bound_available", True)
        else None,
        "upf_paper_auc": curve["auc"],
        "retraining_mean_scpe": curve["mean_scpe"],
        "retraining_offline_seconds": curve["offline_seconds"],
        "retraining_online_decision_seconds": curve["online_decision_seconds"],
        "cost_scpe": None if oracle.cost <= 0 else _num(100.0 * (result.cost - oracle.cost) / oracle.cost),
    }
    forecast = analysis.get("forecast")
    out["performance_forecast_mae"] = forecast["mae"] if rule == "upf" and forecast else None
    out["performance_forecast_bias"] = forecast["bias"] if rule == "upf" and forecast else None
    return out


def retraining_reference_metrics(
    analysis: dict,
    matrix: PerformanceMatrix | None = None,
    *,
    focus_alpha: float | None = None,
    rules: tuple[str, ...] = ("cara", "upf", "oracle"),
) -> dict[str, float | int | None]:
    """Per-policy CARA, UPF, and Oracle reference metrics extracted from ``analyze()``."""
    setup = analysis.get("setup") or {}
    target_alpha = setup.get("focus_alpha") if focus_alpha is None else focus_alpha
    out: dict[str, float | int | None] = {
        "retraining_alpha_max": setup.get("alpha_max"),
        "retraining_lipschitz_L": setup.get("lipschitz_L"),
    }
    if matrix is not None:
        out["retraining_offline_batches"] = matrix.offline
        out["retraining_online_batches"] = matrix.horizon
    analysis_rules = analysis.get("rules") or {}
    for name in rules:
        rule = analysis_rules.get(name)
        if not rule:
            continue
        out[f"retraining_{name}_auc"] = rule.get("auc")
        out[f"retraining_{name}_mean_scpe"] = rule.get("mean_scpe")
        curve = rule.get("curve") or []
        if curve:
            focus_row = (
                min(curve, key=lambda row: abs(row["alpha"] - target_alpha))
                if target_alpha is not None
                else curve[len(curve) // 2]
            )
            out[f"retraining_{name}_c_hat"] = focus_row.get("cost")
            out[f"retraining_{name}_retrains"] = focus_row.get("retrains")
            out[f"retraining_{name}_average_accuracy"] = focus_row.get("average_accuracy")
            out[f"retraining_{name}_scpe"] = focus_row.get("scpe")
    forecast = analysis.get("forecast")
    if forecast:
        out["retraining_upf_forecast_mae"] = forecast.get("mae")
        out["retraining_upf_forecast_bias"] = forecast.get("bias")
    return out
