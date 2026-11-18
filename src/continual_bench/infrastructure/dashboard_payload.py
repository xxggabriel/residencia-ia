"""JSON-safe data extraction and normalization for the interactive benchmark HTML dashboard.

Extracts all benchmark telemetry, continual learning / retention matrices, classification curves,
and paper-specific data structures for:
1. Mahadevan & Mathioudakis (2024), "Cost-aware retraining for machine learning" (CARA)
2. Regol et al. (2025), "When to retrain a machine learning model" (UPF)
"""

from __future__ import annotations

import math
from typing import Any


def _finite(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except Exception:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    return None


def _clean_list(values: Any) -> list[float | int | None]:
    if hasattr(values, "tolist") and callable(values.tolist):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return []
    return [_finite(v) for v in values]


def _clean_matrix(matrix: Any) -> list[list[float | int | None]]:
    if hasattr(matrix, "tolist") and callable(matrix.tolist):
        matrix = matrix.tolist()
    if not isinstance(matrix, (list, tuple)):
        return []
    return [_clean_list(row) for row in matrix if isinstance(row, (list, tuple))]


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, bool):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _sanitize_json(obj.item())
        except Exception:
            return None
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return _sanitize_json(obj.tolist())
        except Exception:
            return []
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return str(obj)


def _extract_windows(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw_windows = report.get("windows") or []
    raw_by_step: dict[tuple[str, Any], dict[str, Any]] = {}
    raw_by_end: dict[Any, dict[str, Any]] = {}
    for w in raw_windows:
        if not isinstance(w, dict):
            continue
        step = w.get("end_seq", w.get("step"))
        scope = w.get("scope", "prequential")
        if step is not None:
            raw_by_step[(scope, step)] = w
            raw_by_end.setdefault(step, w)

    obs_windows = (report.get("tracking_observations") or {}).get("windows") or []
    rows: list[dict[str, Any]] = []
    seen_steps: set[tuple[str, Any]] = set()

    if obs_windows:
        for item in obs_windows:
            if not isinstance(item, dict):
                continue
            step = _finite(item.get("step"))
            scope = str(item.get("scope") or "prequential")
            seen_steps.add((scope, step))
            raw_match = raw_by_step.get((scope, step)) or raw_by_end.get(step) or {}
            metrics: dict[str, float | int] = {}
            for k, v in raw_match.items():
                val = _finite(v)
                if k not in ("end_seq", "start_seq", "scope", "step") and val is not None:
                    metrics[str(k)] = val
            for k, v in (item.get("quality") or {}).items():
                val = _finite(v)
                if val is not None:
                    metrics[str(k)] = val
            for k, v in (item.get("metrics") or {}).items():
                val = _finite(v)
                if val is not None:
                    metrics[str(k)] = val
            rows.append(
                {
                    "step": step,
                    "start_seq": _finite(item.get("start_seq", raw_match.get("start_seq"))),
                    "scope": scope,
                    "coordinate": item.get("coordinate", "sequence"),
                    "model_version": item.get("model_version"),
                    "fit_id": item.get("fit_id"),
                    "metrics": metrics,
                }
            )

    for w in raw_windows:
        if not isinstance(w, dict):
            continue
        step = _finite(w.get("end_seq", w.get("step")))
        if step is None:
            continue
        scope = str(w.get("scope") or "prequential")
        if (scope, step) in seen_steps:
            continue
        metrics = {}
        for k, v in w.items():
            if k in ("end_seq", "start_seq", "scope", "step", "model_version", "fit_id", "metrics"):
                continue
            val = _finite(v)
            if val is not None:
                metrics[str(k)] = val
        if isinstance(w.get("metrics"), dict):
            for k, v in w["metrics"].items():
                val = _finite(v)
                if val is not None:
                    metrics[str(k)] = val
        rows.append(
            {
                "step": step,
                "start_seq": _finite(w.get("start_seq")),
                "scope": scope,
                "coordinate": "sequence",
                "model_version": w.get("model_version"),
                "fit_id": w.get("fit_id"),
                "metrics": metrics,
            }
        )
    rows.sort(key=lambda r: (0 if r.get("scope") == "prequential" else 1, r.get("step") or 0))
    return rows


def _extract_all_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(
        group: str,
        name: str,
        value: Any,
        status: str = "available",
        reason: str | None = None,
        unit_or_scope: str | None = None,
    ) -> None:
        key = (group, name)
        if key in seen:
            return
        seen.add(key)
        clean: Any = _finite(value)
        if clean is None and value is not None and isinstance(value, (str, bool)):
            clean = value
        rows.append(
            {
                "group": group,
                "name": name,
                "value": clean,
                "status": status if clean is not None else "unavailable",
                "reason": None if clean is not None else (reason or "unavailable"),
                "unit_or_scope": unit_or_scope,
            }
        )

    tracking = report.get("tracking") or {}
    t_avail = tracking.get("availability") or {}
    for k, v in (tracking.get("metrics") or {}).items():
        info = t_avail.get(k) if isinstance(t_avail.get(k), dict) else {}
        add(
            "tracking",
            str(k),
            v,
            str(info.get("status") or "available"),
            info.get("reason"),
            info.get("scope") or info.get("unit"),
        )

    for k, v in (report.get("metrics") or {}).items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                add(f"summary/{k}", str(sub_k), sub_v, unit_or_scope=str(k))
        else:
            add("summary", str(k), v)

    for k, v in (report.get("holdout") or {}).items():
        add("holdout", str(k), v, unit_or_scope="holdout")

    for task_id, task_vals in (report.get("per_task") or {}).items():
        if isinstance(task_vals, dict):
            for sub_k, sub_v in task_vals.items():
                add(f"per_task/{task_id}", str(sub_k), sub_v, unit_or_scope=str(task_id))

    obs = report.get("tracking_observations") or {}
    for scope_name, scope_vals in (obs.get("scopes") or {}).items():
        if not isinstance(scope_vals, dict):
            continue
        avail = scope_vals.get("availability") or {}
        for k, v in scope_vals.items():
            if k in (
                "availability",
                "confusion_matrix",
                "confusion_counts_sparse",
                "classes",
                "class_support",
                "class_recall",
                "pr_curve",
                "roc_curve",
                "residuals",
            ):
                continue
            info = avail.get(k) if isinstance(avail.get(k), dict) else {}
            add(
                f"scope/{scope_name}",
                str(k),
                v,
                str(info.get("status") or "available"),
                info.get("reason"),
                str(scope_name),
            )

    for section in ("effort", "resources", "retention", "drift"):
        sec = obs.get(section) or {}
        if not isinstance(sec, dict):
            continue
        avail = sec.get("availability") or {}
        scope_hint = sec.get("cost_currency") or sec.get("delay_unit") or section
        for k, v in sec.items():
            if k in (
                "availability",
                "matrix",
                "checkpoints",
                "periods",
                "series",
                "alarms",
                "changes",
                "delays",
                "release_delays",
                "matching",
            ):
                continue
            info = avail.get(k) if isinstance(avail.get(k), dict) else {}
            add(
                section,
                str(k),
                v,
                str(info.get("status") or "available"),
                info.get("reason"),
                str(info.get("scope") or scope_hint),
            )

    costs_obj = tracking.get("costs") or report.get("costs") or {}
    cost_unit = (costs_obj.get("settings") or {}).get("unit") or "units"
    t_costs = costs_obj.get("metrics") or {}
    for k, v in t_costs.items():
        add("costs", str(k), v, unit_or_scope=str(cost_unit))
    aws_costs = (costs_obj.get("aws") or {}).get("metrics") or {}
    for k, v in aws_costs.items():
        add("costs/aws", str(k), v, unit_or_scope="USD")

    task_eval = report.get("task_evaluation") or {}
    if isinstance(task_eval, dict):
        for k, v in task_eval.items():
            if isinstance(v, dict):
                scope_label = str(v.get("scope") or k)
                for sub_k, sub_v in v.items():
                    if sub_k in ("windows", "predictions", "availability", "contract"):
                        continue
                    if isinstance(sub_v, dict):
                        for m_k, m_v in sub_v.items():
                            if _finite(m_v) is not None or isinstance(m_v, (str, bool)):
                                add(f"task_evaluation/{k}", str(m_k), m_v, unit_or_scope=scope_label)
                    elif _finite(sub_v) is not None or isinstance(sub_v, (str, bool)):
                        add(
                            f"task_evaluation/{k}",
                            str(sub_k),
                            sub_v,
                            reason=v.get("reason"),
                            unit_or_scope=scope_label,
                        )
            elif _finite(v) is not None or isinstance(v, (str, bool)):
                add("task_evaluation", str(k), v, unit_or_scope="task_evaluation")

    sys_raw = report.get("system_metrics") or {}
    if isinstance(sys_raw, dict):
        sys_summary = sys_raw.get("summary")
        if not isinstance(sys_summary, dict):
            sys_summary = sys_raw.get("metrics") if isinstance(sys_raw.get("metrics"), dict) else {}
        for k, v in sys_summary.items():
            if isinstance(v, dict):
                add("system", str(k), v.get("mean"), unit_or_scope="system")
                for sub_k, sub_v in v.items():
                    add(f"system/{k}", str(sub_k), sub_v, unit_or_scope="system")
            else:
                add("system", str(k), v, unit_or_scope="system")

    retraining_raw = report.get("retraining") or {}
    if isinstance(retraining_raw, dict) and (retraining_raw.get("setup") or retraining_raw.get("rules")):
        for k, v in (retraining_raw.get("setup") or {}).items():
            if isinstance(v, (int, float, str, bool)):
                add("retraining/setup", str(k), v, unit_or_scope="retraining")
        for rname, rinfo in (retraining_raw.get("rules") or {}).items():
            if isinstance(rinfo, dict):
                add(f"retraining/{rname}", "auc", rinfo.get("auc"), unit_or_scope=str(rname))
                add(f"retraining/{rname}", "mean_scpe", rinfo.get("mean_scpe"), unit_or_scope="%")
                add(
                    f"retraining/{rname}",
                    "mean_strategy_scpe",
                    rinfo.get("mean_strategy_scpe"),
                    unit_or_scope="%",
                )
                add(
                    f"retraining/{rname}",
                    "offline_seconds",
                    rinfo.get("offline_seconds"),
                    unit_or_scope="seconds",
                )
                add(
                    f"retraining/{rname}",
                    "online_decision_seconds",
                    rinfo.get("online_decision_seconds"),
                    unit_or_scope="seconds",
                )

    return rows


def _derive_served_by(
    retrain_points: list[int],
    initial_model: int,
    offline_batches: int,
    batches: int,
    same_batch: bool = False,
) -> list[int]:
    retrains = set(retrain_points)
    deployed = initial_model
    served: list[int] = []
    for j in range(offline_batches, batches):
        if same_batch:
            if j in retrains:
                deployed = j
            served.append(deployed)
        else:
            served.append(deployed)
            if j in retrains:
                deployed = j
    return served


def _compute_empirical_lipschitz(
    err_matrix: list[list[float | int | None]],
    offline_batches: int,
) -> float:
    gaps: list[float] = []
    size = len(err_matrix)
    for j in range(max(0, offline_batches), size):
        for i in range(j):
            if (
                i + 1 < size
                and j < len(err_matrix[i])
                and j < len(err_matrix[i + 1])
                and err_matrix[i][j] is not None
                and err_matrix[i + 1][j] is not None
            ):
                gaps.append(abs(float(err_matrix[i][j]) - float(err_matrix[i + 1][j])))
    return float(max(gaps, default=0.0))


def _extract_retraining(report: dict[str, Any]) -> dict[str, Any]:
    raw = report.get("retraining")
    has_retraining = isinstance(raw, dict) and bool(
        raw.get("rules") or raw.get("performance_matrix") or raw.get("staleness_matrix") or raw.get("setup")
    )
    if not isinstance(raw, dict):
        raw = {}

    setup_raw = raw.get("setup") if isinstance(raw.get("setup"), dict) else {}
    perf_raw_obj = raw.get("performance_matrix")
    if isinstance(perf_raw_obj, dict):
        perf_raw = perf_raw_obj
        err_matrix = _clean_matrix(perf_raw.get("err") or perf_raw.get("matrix") or [])
    elif isinstance(perf_raw_obj, (list, tuple)):
        perf_raw = {}
        err_matrix = _clean_matrix(perf_raw_obj)
    else:
        perf_raw = {}
        err_matrix = []

    batches = int(_finite(setup_raw.get("batches") or perf_raw.get("batches")) or len(err_matrix) or 0)
    offline_val = setup_raw.get("offline_batches")
    if offline_val is None:
        offline_val = setup_raw.get("offline", perf_raw.get("offline"))
    offline_batches = int(_finite(offline_val) or 0)
    online_val = setup_raw.get("online_batches")
    if online_val is None:
        online_val = setup_raw.get("online")
    online_batches = int(
        _finite(online_val) if _finite(online_val) is not None else max(0, batches - offline_batches)
    )
    init_val = setup_raw.get("initial_model")
    if init_val is None:
        init_val = setup_raw.get("initial", perf_raw.get("initial"))
    initial_model = int(_finite(init_val) if _finite(init_val) is not None else max(0, offline_batches - 1))

    paper_timing = bool(perf_raw.get("paper_timing"))
    timing_str = str(setup_raw.get("timing") or "")
    same_batch = paper_timing or ("same_batch" in timing_str or "CARA accounting" in timing_str)

    alphas = _clean_list(raw.get("alphas") or [])
    lipschitz_l = _finite(setup_raw.get("lipschitz_L"))
    if lipschitz_l is None and err_matrix:
        lipschitz_l = _compute_empirical_lipschitz(err_matrix, offline_batches)

    raw_bound_alphas = setup_raw.get("bound_alphas") or {}
    bound_alphas: dict[str, float | int | None] = {str(k): _finite(v) for k, v in raw_bound_alphas.items()}
    for r in range(3):
        key = f"r<={r}"
        if key not in bound_alphas and lipschitz_l is not None:
            bound_alphas[key] = _finite(float(lipschitz_l) * (max(0, online_batches - r) ** 2))

    raw_upper = setup_raw.get("retrain_upper_bound") or {}
    retrain_upper_bound: dict[str, float | int | None] = {str(k): _finite(v) for k, v in raw_upper.items()}
    for a in alphas:
        if a is not None and str(a) not in retrain_upper_bound:
            if lipschitz_l is not None and float(lipschitz_l) > 0 and float(a) >= 0:
                ub = max(0.0, online_batches - math.sqrt(float(a) / float(lipschitz_l)))
                retrain_upper_bound[str(a)] = _finite(ub)
            else:
                retrain_upper_bound[str(a)] = 0.0

    rows_list = _clean_list(perf_raw.get("rows") or setup_raw.get("rows") or [])
    fit_seconds_list = _clean_list(perf_raw.get("fit_seconds") or setup_raw.get("fit_seconds") or [])

    focus_alpha = _finite(setup_raw.get("focus_alpha"))
    if focus_alpha is None and alphas:
        focus_alpha = alphas[len(alphas) // 2]
    alpha_max_val = _finite(setup_raw.get("alpha_max"))
    if alpha_max_val is None and alphas:
        finite_alphas = [float(a) for a in alphas if a is not None]
        alpha_max_val = max(finite_alphas) if finite_alphas else None

    setup = {
        "batches": batches,
        "offline": offline_batches,
        "offline_batches": offline_batches,
        "online": online_batches,
        "online_batches": online_batches,
        "initial": initial_model,
        "initial_model": initial_model,
        "decision_points": _finite(setup_raw.get("decision_points")) or online_batches,
        "oracle_alpha_max": _finite(setup_raw.get("oracle_alpha_max")) or alpha_max_val,
        "alpha_max": alpha_max_val,
        "focus_alpha": focus_alpha,
        "lipschitz_L": lipschitz_l,
        "delta": _finite(setup_raw.get("delta")),
        "samples": _finite(setup_raw.get("samples")),
        "rows": rows_list,
        "fit_seconds": fit_seconds_list,
        "timing": timing_str,
        "bound_alphas": bound_alphas,
        "retrain_upper_bound": retrain_upper_bound,
    }

    rules_out: dict[str, Any] = {}
    for name, info in (raw.get("rules") or {}).items():
        if not isinstance(info, dict):
            continue
        raw_curve = info.get("curve") or []
        curve: list[dict[str, Any]] = []
        for pt in raw_curve:
            if not isinstance(pt, dict):
                continue
            r_pts = [int(x) for x in (pt.get("retrain_points") or []) if _finite(x) is not None]
            served = pt.get("served_by")
            if isinstance(served, (list, tuple)):
                served = [int(x) for x in served if _finite(x) is not None]
            elif batches > offline_batches:
                served = _derive_served_by(
                    r_pts,
                    initial_model,
                    offline_batches,
                    batches,
                    same_batch=same_batch and name not in ("upf", "oracle"),
                )
            else:
                served = []
            params = {
                str(pk): (_finite(pv) if isinstance(pv, (int, float)) else pv)
                for pk, pv in (pt.get("parameters") or {}).items()
            }
            retrains_val = _finite(pt.get("retrains"))
            curve.append(
                {
                    "alpha": _finite(pt.get("alpha")),
                    "cost": _finite(pt.get("cost")),
                    "retrains": int(retrains_val) if retrains_val is not None else len(r_pts),
                    "retrain_points": r_pts,
                    "served_by": served,
                    "average_accuracy": _finite(pt.get("average_accuracy")),
                    "scpe": _finite(pt.get("scpe")),
                    "strategy_cost": _finite(pt.get("strategy_cost")),
                    "error_oracle_gap": _finite(pt.get("error_oracle_gap")),
                    "strategy_oracle_gap": _finite(pt.get("strategy_oracle_gap")),
                    "oracle_strategy_cost": _finite(pt.get("oracle_strategy_cost")),
                    "strategy_scpe": _finite(pt.get("strategy_scpe")),
                    "parameters": params,
                }
            )
        rule_alphas = [c["alpha"] for c in curve] if curve else _clean_list(info.get("alphas") or alphas)
        rule_costs = [c["cost"] for c in curve] if curve else _clean_list(info.get("costs") or [])
        retrains_curve = (
            [c["retrains"] for c in curve] if curve else _clean_list(info.get("retrains_curve") or [])
        )
        accuracies_curve = (
            [c["average_accuracy"] for c in curve]
            if curve
            else _clean_list(info.get("accuracies_curve") or [])
        )
        scpe_curve = [c["scpe"] for c in curve] if curve else _clean_list(info.get("scpe_curve") or [])
        strategy_costs_curve = (
            [c["strategy_cost"] for c in curve]
            if curve
            else _clean_list(info.get("strategy_costs_curve") or [])
        )
        strategy_scpe_curve = (
            [c["strategy_scpe"] for c in curve]
            if curve
            else _clean_list(info.get("strategy_scpe_curve") or [])
        )
        oracle_strategy_costs_curve = [c["oracle_strategy_cost"] for c in curve]
        error_oracle_gap_curve = (
            [c["error_oracle_gap"] for c in curve]
            if curve
            else _clean_list(info.get("error_oracle_gap_curve") or [])
        )
        strategy_oracle_gap_curve = (
            [c["strategy_oracle_gap"] for c in curve]
            if curve
            else _clean_list(info.get("strategy_oracle_gap_curve") or [])
        )
        retrain_points_by_alpha = [c["retrain_points"] for c in curve]
        served_by_alpha = [c["served_by"] for c in curve]
        parameters_by_alpha = [c["parameters"] for c in curve]

        focus_row = (
            min(curve, key=lambda c: abs(float(c["alpha"] or 0.0) - float(focus_alpha)))
            if curve and focus_alpha is not None
            else (curve[len(curve) // 2] if curve else {})
        )
        c_hat_val = _finite(info.get("c_hat"))
        if c_hat_val is None:
            c_hat_val = _finite(focus_row.get("cost"))
        retrains_focus = _finite(info.get("retrains"))
        if retrains_focus is None:
            retrains_focus = _finite(focus_row.get("retrains"))

        rules_out[str(name)] = {
            "label": str(info.get("label") or name),
            "auc": _finite(info.get("auc")),
            "c_hat": c_hat_val,
            "retrains": int(retrains_focus) if retrains_focus is not None else None,
            "average_accuracy": _finite(info.get("average_accuracy", focus_row.get("average_accuracy"))),
            "mean_scpe": _finite(info.get("mean_scpe")),
            "mean_strategy_scpe": _finite(info.get("mean_strategy_scpe")),
            "online_decision_seconds": _finite(info.get("online_decision_seconds")),
            "offline_seconds": _finite(info.get("offline_seconds")),
            "alphas": rule_alphas,
            "costs": rule_costs,
            "retrains_curve": retrains_curve,
            "accuracies_curve": accuracies_curve,
            "scpe_curve": scpe_curve,
            "strategy_costs_curve": strategy_costs_curve,
            "strategy_scpe_curve": strategy_scpe_curve,
            "oracle_strategy_costs_curve": oracle_strategy_costs_curve,
            "error_oracle_gap_curve": error_oracle_gap_curve,
            "strategy_oracle_gap_curve": strategy_oracle_gap_curve,
            "retrain_points_by_alpha": retrain_points_by_alpha,
            "served_by_alpha": served_by_alpha,
            "parameters_by_alpha": parameters_by_alpha,
            "curve": curve,
        }

    if not alphas and rules_out:
        first_rule = next(iter(rules_out.values()))
        alphas = list(first_rule.get("alphas") or [])

    table_out: dict[str, dict[str, Any]] = {}
    raw_table = raw.get("table")
    if isinstance(raw_table, dict):
        for level, by_rule in raw_table.items():
            if not isinstance(by_rule, dict):
                continue
            table_out[str(level)] = {}
            for rname, rcell in by_rule.items():
                if not isinstance(rcell, dict):
                    continue
                r_pts = [int(x) for x in (rcell.get("retrain_points") or []) if _finite(x) is not None]
                ret_cnt = _finite(rcell.get("retrains"))
                table_out[str(level)][str(rname)] = {
                    "alpha": _finite(rcell.get("alpha")),
                    "cost": _finite(rcell.get("cost")),
                    "retrains": int(ret_cnt) if ret_cnt is not None else len(r_pts),
                    "retrain_points": r_pts,
                    "average_accuracy": _finite(rcell.get("average_accuracy")),
                    "scpe": _finite(rcell.get("scpe")),
                    "strategy_cost": _finite(rcell.get("strategy_cost")),
                    "strategy_scpe": _finite(rcell.get("strategy_scpe")),
                }

    for level_name, target_frac in (("low", 0.1), ("focus", 0.5), ("high", 0.9)):
        if level_name not in table_out and rules_out:
            target_a = (
                focus_alpha
                if level_name == "focus" and focus_alpha is not None
                else ((alpha_max_val or 1.0) * target_frac)
            )
            table_out[level_name] = {}
            for rname, rdata in rules_out.items():
                rcurve = rdata.get("curve") or []
                if not rcurve:
                    continue
                chosen = min(rcurve, key=lambda c: abs(float(c["alpha"] or 0.0) - float(target_a)))
                table_out[level_name][rname] = {
                    "alpha": chosen.get("alpha"),
                    "cost": chosen.get("cost"),
                    "retrains": chosen.get("retrains"),
                    "retrain_points": chosen.get("retrain_points") or [],
                    "average_accuracy": chosen.get("average_accuracy"),
                    "scpe": chosen.get("scpe"),
                    "strategy_cost": chosen.get("strategy_cost"),
                    "strategy_scpe": chosen.get("strategy_scpe"),
                }

    summary_table: list[dict[str, Any]] = []
    for rname, rdata in rules_out.items():
        accs = [float(x) for x in (rdata.get("accuracies_curve") or []) if x is not None]
        rets = [float(x) for x in (rdata.get("retrains_curve") or []) if x is not None]
        summary_table.append(
            {
                "rule": rname,
                "label": rdata.get("label") or rname,
                "auc": rdata.get("auc"),
                "mean_scpe": rdata.get("mean_scpe"),
                "mean_strategy_scpe": rdata.get("mean_strategy_scpe"),
                "mean_accuracy": _finite(sum(accs) / len(accs)) if accs else rdata.get("average_accuracy"),
                "mean_retrains": _finite(sum(rets) / len(rets)) if rets else rdata.get("retrains"),
                "c_hat": rdata.get("c_hat"),
                "retrains": rdata.get("retrains"),
                "online_decision_seconds": rdata.get("online_decision_seconds"),
                "offline_seconds": rdata.get("offline_seconds"),
            }
        )

    wrong_out: dict[str, Any] = {}
    for rname, grid in (raw.get("wrong_alpha") or {}).items():
        if isinstance(grid, dict):
            prov = _clean_list(
                grid.get("provided") or grid.get("provided_alphas") or grid.get("alphas") or alphas
            )
            real_a = _clean_list(grid.get("real") or grid.get("real_alphas") or grid.get("alphas") or alphas)
            pct_inc = _clean_matrix(grid.get("pct_increase") or grid.get("percent_increase") or [])
            wrong_out[str(rname)] = {
                "alphas": prov or real_a,
                "provided": prov,
                "provided_alphas": prov,
                "real": real_a,
                "real_alphas": real_a,
                "pct_increase": pct_inc,
                "percent_increase": pct_inc,
            }

    forecast_raw = raw.get("forecast")
    if isinstance(forecast_raw, dict):
        pairs = [
            [_finite(p[0]), _finite(p[1])]
            for p in (forecast_raw.get("pairs") or [])
            if isinstance(p, (list, tuple))
            and len(p) >= 2
            and _finite(p[0]) is not None
            and _finite(p[1]) is not None
        ][:600]
        preds = (
            _clean_list(forecast_raw.get("predictions"))
            if forecast_raw.get("predictions") is not None
            else [p[0] for p in pairs]
        )
        obs_vals = (
            _clean_list(forecast_raw.get("observed") or forecast_raw.get("realised"))
            if (forecast_raw.get("observed") is not None or forecast_raw.get("realised") is not None)
            else [p[1] for p in pairs]
        )
        res_vals = (
            _clean_list(forecast_raw.get("residuals"))
            if forecast_raw.get("residuals") is not None
            else [_finite(float(p[0]) - float(p[1])) for p in pairs if p[0] is not None and p[1] is not None]
        )
        cnt_val = _finite(forecast_raw.get("count"))
        forecast_out: dict[str, Any] = {
            "mae": _finite(forecast_raw.get("mae")),
            "bias": _finite(forecast_raw.get("bias")),
            "count": int(cnt_val) if cnt_val is not None else len(pairs),
            "pairs": pairs,
            "predictions": preds,
            "observed": obs_vals,
            "residuals": res_vals,
        }
    else:
        forecast_out = {
            "mae": None,
            "bias": None,
            "count": 0,
            "pairs": [],
            "predictions": [],
            "observed": [],
            "residuals": [],
        }

    cumulative_out = {str(k): _clean_list(v) for k, v in (raw.get("cumulative_strategy_cost") or {}).items()}
    staleness_matrix = _clean_matrix(raw.get("staleness_matrix") or [])
    acc_matrix = [[_finite(1.0 - float(v)) if v is not None else None for v in row] for row in err_matrix]
    performance_matrix = {
        "err": err_matrix,
        "acc": acc_matrix,
        "matrix": err_matrix,
        "rows": rows_list,
        "fit_seconds": fit_seconds_list,
        "score_seconds": _clean_matrix(perf_raw.get("score_seconds") or []),
        "offline": offline_batches,
        "initial": initial_model,
        "batches": batches,
        "paper_timing": paper_timing,
    }

    oracle_sched_raw = raw.get("oracle_schedule_at_focus")
    if isinstance(oracle_sched_raw, dict):
        oracle_schedule_at_focus = [
            int(x) for x in (oracle_sched_raw.get("retrain_points") or []) if _finite(x) is not None
        ]
    elif isinstance(oracle_sched_raw, (list, tuple)):
        oracle_schedule_at_focus = [int(x) for x in oracle_sched_raw if _finite(x) is not None]
    else:
        oracle_schedule_at_focus = []

    if not oracle_schedule_at_focus and "oracle" in rules_out:
        o_curve = rules_out["oracle"].get("curve") or []
        if o_curve:
            o_pt = (
                min(o_curve, key=lambda c: abs(float(c["alpha"] or 0.0) - float(focus_alpha)))
                if focus_alpha is not None
                else o_curve[len(o_curve) // 2]
            )
            oracle_schedule_at_focus = list(o_pt.get("retrain_points") or [])

    deployed_raw = raw.get("deployed")
    if isinstance(deployed_raw, dict) and deployed_raw.get("served_by"):
        deployed_out = {
            "policy": str(deployed_raw.get("policy") or "cara"),
            "alpha": _finite(deployed_raw.get("alpha")),
            "retrain_points": [
                int(x) for x in (deployed_raw.get("retrain_points") or []) if _finite(x) is not None
            ],
            "served_by": [int(x) for x in (deployed_raw.get("served_by") or []) if _finite(x) is not None],
            "parameters": {
                str(k): (_finite(v) if isinstance(v, (int, float)) else v)
                for k, v in (deployed_raw.get("parameters") or {}).items()
            },
            "oracle_retrain_points": [
                int(x)
                for x in (deployed_raw.get("oracle_retrain_points") or oracle_schedule_at_focus)
                if _finite(x) is not None
            ],
        }
    else:
        default_rule = "cara" if "cara" in rules_out else next(iter(rules_out), "oracle")
        r_curve = (rules_out.get(default_rule) or {}).get("curve") or []
        pt = (
            min(r_curve, key=lambda c: abs(float(c["alpha"] or 0.0) - float(focus_alpha)))
            if r_curve and focus_alpha is not None
            else (r_curve[len(r_curve) // 2] if r_curve else {})
        )
        deployed_out = {
            "policy": default_rule,
            "alpha": pt.get("alpha", focus_alpha),
            "retrain_points": pt.get("retrain_points") or [],
            "served_by": pt.get("served_by") or [],
            "parameters": pt.get("parameters") or {},
            "oracle_retrain_points": oracle_schedule_at_focus,
        }

    upper_bound_curve = [
        retrain_upper_bound.get(str(a))
        if str(a) in retrain_upper_bound
        else (
            _finite(max(0.0, online_batches - math.sqrt(float(a) / float(lipschitz_l))))
            if lipschitz_l and float(lipschitz_l) > 0 and a is not None and float(a) >= 0
            else 0.0
        )
        for a in alphas
    ]
    proposition_bounds = {
        "lipschitz_L": lipschitz_l,
        "L": lipschitz_l,
        "horizon": online_batches,
        "T": online_batches,
        "thresholds": {
            "r<=0": bound_alphas.get("r<=0"),
            "r<=1": bound_alphas.get("r<=1"),
            "r<=2": bound_alphas.get("r<=2"),
        },
        "bound_alphas": bound_alphas,
        "r_le_0": bound_alphas.get("r<=0"),
        "r_le_1": bound_alphas.get("r<=1"),
        "r_le_2": bound_alphas.get("r<=2"),
        "retrain_upper_bound": retrain_upper_bound,
        "upper_bound_curve": upper_bound_curve,
    }

    return {
        "available": has_retraining,
        "setup": setup,
        "alphas": alphas,
        "rules": rules_out,
        "table": table_out,
        "summary_table": summary_table,
        "wrong_alpha": wrong_out,
        "forecast": forecast_out,
        "cumulative_strategy_cost": cumulative_out,
        "staleness_matrix": staleness_matrix,
        "performance_matrix": performance_matrix,
        "oracle_schedule_at_focus": oracle_schedule_at_focus,
        "proposition_bounds": proposition_bounds,
        "deployed": deployed_out,
    }


def _downsample_items(items: list[Any], max_pts: int = 300) -> list[Any]:
    if len(items) <= max_pts:
        return items
    step = (len(items) - 1) / (max_pts - 1)
    idxs = sorted({0, len(items) - 1, *(round(i * step) for i in range(max_pts))})
    return [items[i] for i in idxs]


def _downsample_curve(xs: list[Any], ys: list[Any], max_pts: int = 220) -> tuple[list[float], list[float]]:
    pairs = [
        (float(x), float(y)) for x, y in zip(xs, ys) if _finite(x) is not None and _finite(y) is not None
    ]
    if len(pairs) <= max_pts:
        return [p[0] for p in pairs], [p[1] for p in pairs]
    step = (len(pairs) - 1) / (max_pts - 1)
    idxs = sorted({0, len(pairs) - 1, *(round(i * step) for i in range(max_pts))})
    return [pairs[i][0] for i in idxs], [pairs[i][1] for i in idxs]


def build_dashboard_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Build a comprehensive, JSON-safe payload for the interactive HTML dashboard."""
    obs = report.get("tracking_observations") or {}
    lineage = obs.get("lineage") or {}
    config = report.get("config") or {}
    tracking = report.get("tracking") or {}
    env = report.get("environment") or {}

    def _name(obj: Any) -> Any:
        return obj.get("name") if isinstance(obj, dict) else obj

    dataset_cfg = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    seed_val = lineage.get("seed")
    if seed_val is None:
        seed_val = dataset_cfg.get("seed")
    if seed_val is None and isinstance(report.get("dataset"), dict):
        seed_val = report["dataset"].get("seed")

    duration_val = _finite(
        (obs.get("resources") or {}).get("duration_seconds")
        or report.get("wall_seconds_this_attempt")
        or report.get("compute_seconds")
        or (report.get("compute") or {}).get("elapsed_seconds")
        or report.get("duration_seconds")
    )
    observed_val = _finite(
        report.get("observed")
        if report.get("observed") is not None
        else (obs.get("effort") or {}).get("examples_observed")
    )
    labeled_val = _finite(
        report.get("labeled")
        if report.get("labeled") is not None
        else ((obs.get("effort") or {}).get("evaluated_examples") or (report.get("metrics") or {}).get("n"))
    )
    code_rev = (
        lineage.get("code_revision")
        or env.get("code_sha256")
        or (config.get("environment") or {}).get("code_sha256")
        or report.get("code_revision")
    )

    meta = {
        "run_key": str(report.get("run_key") or report.get("identity") or ""),
        "dataset": _name(lineage.get("dataset") or report.get("dataset") or config.get("dataset")),
        "model": _name(lineage.get("model") or config.get("model")),
        "policy": _name(lineage.get("policy") or config.get("policy")),
        "detector": _name(config.get("detector")),
        "mode": config.get("mode") or report.get("protocol"),
        "protocol": report.get("protocol") or lineage.get("protocol") or config.get("mode"),
        "seed": _finite(seed_val),
        "complete": bool(report.get("complete", True)),
        "code_revision": str(code_rev) if code_rev is not None else None,
        "duration_seconds": duration_val,
        "observed": observed_val,
        "labeled": labeled_val,
        "task": obs.get("task") or config.get("task") or "binary",
        "warmup": _finite(config.get("warmup")),
        "batch_size": _finite(config.get("batch_size")),
        "metric_window": _finite(config.get("metric_window") or config.get("window")),
        "replay_window": _finite(config.get("replay_window")),
    }

    windows = _extract_windows(report)

    epochs = []
    for ep in report.get("epochs") or []:
        if isinstance(ep, dict):
            epochs.append(
                {
                    "epoch": _finite(ep.get("epoch")),
                    "training_seconds": _finite(ep.get("training_seconds")),
                    "prediction_seconds": _finite(ep.get("prediction_seconds")),
                    "metrics": {
                        str(k): _finite(v)
                        for k, v in (ep.get("metrics") or {}).items()
                        if _finite(v) is not None
                    },
                    "costs": {
                        str(k): _finite(v)
                        for k, v in ((ep.get("costs") or {}).get("metrics") or {}).items()
                        if _finite(v) is not None
                    },
                }
            )

    fits = []
    for f in report.get("training_history") or []:
        if not isinstance(f, dict):
            continue
        raw_steps = f.get("observation_steps") or f.get("steps") or []
        step_series = []
        stride = max(1, len(raw_steps) // 120)
        for s in raw_steps[::stride]:
            if isinstance(s, dict):
                s_metrics: dict[str, float | int | None] = {
                    str(k): _finite(v) for k, v in (s.get("metrics") or {}).items() if _finite(v) is not None
                }
                for sc_name, sc_vals in (s.get("scopes") or {}).items():
                    if isinstance(sc_vals, dict):
                        for k, v in sc_vals.items():
                            if _finite(v) is not None:
                                s_metrics[f"{sc_name}/{k}"] = _finite(v)
                step_series.append(
                    {
                        "step": _finite(s.get("step")),
                        "global_step": _finite(s.get("global_step")),
                        "metrics": s_metrics,
                    }
                )
        fit_scopes: dict[str, dict[str, float | int | None]] = {}
        raw_fit_scopes = f.get("observation_scopes") or f.get("scopes") or {}
        for sc_name, sc_vals in raw_fit_scopes.items():
            if isinstance(sc_vals, dict):
                fit_scopes[str(sc_name)] = {
                    str(k): _finite(v) for k, v in sc_vals.items() if _finite(v) is not None
                }
        fit_effort_dict = {
            str(k): _finite(v)
            for k, v in (f.get("observation_effort") or f.get("effort") or {}).items()
            if _finite(v) is not None
        }
        fit_res_dict = {
            str(k): _finite(v)
            for k, v in (f.get("observation_resources") or f.get("resources") or {}).items()
            if _finite(v) is not None
        }
        fits.append(
            {
                "fit_id": f.get("fit_id"),
                "action": f.get("action"),
                "promotion_status": f.get("promotion_status") or f.get("status"),
                "start_seq": _finite(f.get("start_seq")),
                "end_seq": _finite(f.get("end_seq")),
                "promotion_step": _finite(f.get("promotion_step")),
                "train_samples": _finite(f.get("train_samples")),
                "train_seconds": _finite(f.get("train_seconds")),
                "deployed_version": f.get("deployed_version"),
                "training_presentations": _finite(f.get("training_presentations")),
                "effort": fit_effort_dict,
                "observation_effort": fit_effort_dict,
                "resources": fit_res_dict,
                "observation_resources": fit_res_dict,
                "scopes": fit_scopes,
                "observation_scopes": fit_scopes,
                "steps": step_series,
                "observation_steps": step_series,
            }
        )

    def _clean_update_list(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, (list, tuple)):
            return []
        out = []
        for u in items:
            if not isinstance(u, dict):
                continue
            out.append(
                {
                    "seq": _finite(u.get("seq")),
                    "batch": _finite(u.get("batch")),
                    "action": u.get("action"),
                    "reason": u.get("reason"),
                    "train_samples": _finite(u.get("train_samples", u.get("samples"))),
                    "training_presentations": _finite(
                        u.get("training_presentations", u.get("train_samples", u.get("samples")))
                    ),
                    "seconds": _finite(u.get("seconds")),
                    "promoted": bool(u.get("promoted", True)),
                    "candidate_accuracy": _finite(u.get("candidate_accuracy")),
                    "incumbent_accuracy": _finite(u.get("incumbent_accuracy")),
                }
            )
        return out

    updates = _clean_update_list(report.get("updates") or report.get("candidates") or [])
    candidates = _clean_update_list(report.get("candidates") or report.get("updates") or [])

    raw_alarms = report.get("alarms") or ((obs.get("drift") or {}).get("alarms") or [])
    alarms = [
        {"seq": _finite(a.get("seq") if isinstance(a, dict) else a)}
        for a in raw_alarms
        if _finite(a.get("seq") if isinstance(a, dict) else a) is not None
    ]

    dataset_obj = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    raw_changes = dataset_obj.get("changepoints") or ((obs.get("drift") or {}).get("changes") or [])
    changepoints = [_finite(c.get("seq") if isinstance(c, dict) else c) for c in raw_changes]
    changepoints = [c for c in changepoints if c is not None]

    drift_obs = obs.get("drift") or {}
    recovery_raw = tracking.get("recovery") or drift_obs.get("matching") or []
    recovery = [
        {
            "change_seq": _finite(r.get("change_seq")),
            "alarm_seq": _finite(r.get("alarm_seq")),
            "delay": _finite(r.get("delay")),
            "censored": bool(r.get("censored", False)),
        }
        for r in recovery_raw
        if isinstance(r, dict) and _finite(r.get("change_seq")) is not None
    ]
    drift = {
        "alarms": alarms,
        "changepoints": changepoints,
        "recovery": recovery,
        "delays": _clean_list(
            drift_obs.get("delays") or [r["delay"] for r in recovery if r["delay"] is not None]
        ),
        "precision": _finite(
            drift_obs.get("precision") or (tracking.get("metrics") or {}).get("drift_precision")
        ),
        "recall": _finite(drift_obs.get("recall") or (tracking.get("metrics") or {}).get("drift_recall")),
        "f1": _finite(drift_obs.get("f1") or (tracking.get("metrics") or {}).get("drift_f1")),
        "censored_count": _finite(drift_obs.get("censored_count")),
        "release_delay_mean": _finite(drift_obs.get("release_delay_mean")),
    }

    retention_obs = obs.get("retention") or {}
    all_tracking_series: dict[str, list[dict[str, Any]]] = {}
    for k, pts in (tracking.get("series") or {}).items():
        if isinstance(pts, list):
            sampled_pts = _downsample_items(pts, max_pts=300) if len(pts) > 600 else pts
            cleaned_pts = [
                {"step": _finite(p.get("step")), "value": _finite(p.get("value"))}
                for p in sampled_pts
                if isinstance(p, dict) and _finite(p.get("step")) is not None
            ]
            all_tracking_series[str(k)] = _downsample_items(cleaned_pts, max_pts=300)
    t_series = {k: v for k, v in all_tracking_series.items() if k.startswith(("cl_", "retention_"))}

    raw_continual_matrix = ((tracking.get("continual") or {}).get("matrix")) or []
    continual_matrix = []
    for row in raw_continual_matrix:
        if not isinstance(row, dict):
            continue
        continual_matrix.append(
            {
                "task": _finite(row.get("task")),
                "end_seq": _finite(row.get("end_seq")),
                "accuracies": {str(ek): _finite(ev) for ek, ev in (row.get("accuracies") or {}).items()},
                "joint_accuracy": _finite(row.get("joint_accuracy")),
                "single_accuracy": _finite(row.get("single_accuracy")),
                "metrics": {str(mk): _finite(mv) for mk, mv in (row.get("metrics") or {}).items()},
            }
        )

    raw_probes = report.get("retention_probes") or tracking.get("retention_probes") or []
    sampled_probes = _downsample_items(raw_probes, max_pts=500) if isinstance(raw_probes, list) else []
    retention_probes = []
    for p in sampled_probes:
        if isinstance(p, dict):
            retention_probes.append(
                {
                    "concept": p.get("concept"),
                    "observed_seq": _finite(p.get("observed_seq", p.get("step"))),
                    "accuracy": _finite(p.get("accuracy")),
                    "drop_from_best_probe": _finite(p.get("drop_from_best_probe")),
                }
            )

    ret_series = []
    for rs in retention_obs.get("series") or []:
        if isinstance(rs, dict):
            ret_series.append(
                {
                    "step": _finite(rs.get("step")),
                    "model_version": rs.get("model_version"),
                    "absolute_retention": _finite(rs.get("absolute_retention")),
                    "gem_bwt": _finite(rs.get("gem_bwt")),
                    "post_learning_fm": _finite(rs.get("post_learning_fm")),
                }
            )

    near_future = ((report.get("task_evaluation") or {}).get("near_future")) or {}
    retention_payload = {
        "matrix": _clean_matrix(retention_obs.get("matrix") or []),
        "checkpoints": [str(c) for c in (retention_obs.get("checkpoints") or [])],
        "periods": [str(p) for p in (retention_obs.get("periods") or [])],
        "metric": retention_obs.get("metric") or "accuracy",
        "absolute_retention": _finite(retention_obs.get("absolute_retention")),
        "gem_bwt": _finite(retention_obs.get("gem_bwt")),
        "post_learning_fm": _finite(retention_obs.get("post_learning_fm")),
        "series": ret_series,
        "tracking_series": t_series,
        "continual_matrix": continual_matrix,
        "retention_probes": retention_probes,
        "near_future": _sanitize_json(near_future) if isinstance(near_future, dict) else {},
    }

    confusion: dict[str, Any] = {}
    ranking_curves: dict[str, Any] = {}
    class_metrics_by_scope: dict[str, Any] = {}

    for scope_name, scope_vals in (obs.get("scopes") or {}).items():
        if not isinstance(scope_vals, dict):
            continue
        classes_list = [str(c) for c in (scope_vals.get("classes") or [])]
        support_list = _clean_list(scope_vals.get("class_support") or [])
        recall_list = _clean_list(scope_vals.get("class_recall") or [])
        if classes_list or support_list or recall_list:
            class_metrics_by_scope[str(scope_name)] = {
                "classes": classes_list,
                "class_support": support_list,
                "class_recall": recall_list,
                "support": support_list,
                "recall": recall_list,
            }
        if scope_vals.get("confusion_matrix") is not None:
            confusion[str(scope_name)] = {
                "classes": classes_list,
                "matrix": _clean_matrix(scope_vals.get("confusion_matrix") or []),
                "support": support_list,
                "recall": recall_list,
                "class_support": support_list,
                "class_recall": recall_list,
            }
        roc_raw = scope_vals.get("roc_curve") or {}
        pr_raw = scope_vals.get("pr_curve") or {}
        res_raw = scope_vals.get("residuals") or {}
        roc_x, roc_y = _downsample_curve(roc_raw.get("fpr") or [], roc_raw.get("tpr") or [])
        pr_x, pr_y = _downsample_curve(pr_raw.get("recall") or [], pr_raw.get("precision") or [])
        res_x, res_y = _downsample_curve(res_raw.get("prediction") or [], res_raw.get("error") or [])
        roc_obj = (
            {
                "fpr": roc_x,
                "tpr": roc_y,
                "recall_at_fpr_0_01": _finite(scope_vals.get("recall_at_fpr_0_01")),
                "roc_auc": _finite(scope_vals.get("roc_auc") or (report.get("metrics") or {}).get("roc_auc")),
            }
            if roc_x
            else None
        )
        pr_obj = (
            {
                "recall": pr_x,
                "precision": pr_y,
                "average_precision": _finite(
                    scope_vals.get("average_precision")
                    or (report.get("metrics") or {}).get("average_precision")
                ),
                "positive_class": str(scope_vals.get("positive_class")),
                "ranking_method": str(scope_vals.get("ranking_method") or ""),
                "support": _finite(
                    scope_vals.get("ranking_evaluated_examples", scope_vals.get("evaluated_examples"))
                ),
                "positive_support": _finite(scope_vals.get("ranking_positive_examples")),
            }
            if pr_x
            else None
        )
        res_obj = {"prediction": res_x, "error": res_y} if res_x else None
        if roc_obj or pr_obj or res_obj:
            ranking_curves[str(scope_name)] = {
                "roc": roc_obj,
                "roc_curve": roc_obj or {"fpr": [], "tpr": []},
                "pr": pr_obj,
                "pr_curve": pr_obj or {"recall": [], "precision": []},
                "residuals": res_obj,
                "class_metrics": class_metrics_by_scope.get(str(scope_name)),
            }

    if not confusion:
        if report.get("prequential_confusion") is not None:
            confusion["prequential"] = {
                "classes": ["False", "True"],
                "matrix": _clean_matrix(report.get("prequential_confusion")),
                "support": [],
                "recall": [],
                "class_support": [],
                "class_recall": [],
            }
        elif windows and any("tp" in w["metrics"] for w in windows):
            tp = sum(int(w["metrics"].get("tp") or 0) for w in windows)
            fp = sum(int(w["metrics"].get("fp") or 0) for w in windows)
            tn = sum(int(w["metrics"].get("tn") or 0) for w in windows)
            fn = sum(int(w["metrics"].get("fn") or 0) for w in windows)
            if tp + fp + tn + fn > 0:
                neg_sup = tn + fp
                pos_sup = fn + tp
                neg_rec = _finite(tn / neg_sup) if neg_sup > 0 else None
                pos_rec = _finite(tp / pos_sup) if pos_sup > 0 else None
                confusion["prequential"] = {
                    "classes": ["False", "True"],
                    "matrix": [[tn, fp], [fn, tp]],
                    "support": [neg_sup, pos_sup],
                    "recall": [neg_rec, pos_rec],
                    "class_support": [neg_sup, pos_sup],
                    "class_recall": [neg_rec, pos_rec],
                }
                class_metrics_by_scope.setdefault(
                    "prequential",
                    {
                        "classes": ["False", "True"],
                        "class_support": [neg_sup, pos_sup],
                        "class_recall": [neg_rec, pos_rec],
                        "support": [neg_sup, pos_sup],
                        "recall": [neg_rec, pos_rec],
                    },
                )
        if report.get("holdout_confusion") is not None:
            confusion["holdout"] = {
                "classes": ["False", "True"],
                "matrix": _clean_matrix(report.get("holdout_confusion")),
                "support": [],
                "recall": [],
                "class_support": [],
                "class_recall": [],
            }

    primary_scope = (
        "prequential"
        if "prequential" in ranking_curves
        else (next(iter(ranking_curves)) if ranking_curves else "prequential")
    )
    primary_rc = ranking_curves.get(primary_scope) or {}
    primary_cm_scope = (
        "prequential"
        if "prequential" in class_metrics_by_scope
        else (next(iter(class_metrics_by_scope)) if class_metrics_by_scope else None)
    )
    primary_cm = (
        class_metrics_by_scope[primary_cm_scope]
        if primary_cm_scope
        else {
            "classes": [],
            "class_support": [],
            "class_recall": [],
            "support": [],
            "recall": [],
        }
    )
    fading_series = all_tracking_series.get("fading_accuracy") or []
    curves = {
        "roc_curve": primary_rc.get("roc")
        or {
            "fpr": [],
            "tpr": [],
            "recall_at_fpr_0_01": None,
            "roc_auc": _finite((report.get("metrics") or {}).get("roc_auc")),
        },
        "pr_curve": primary_rc.get("pr")
        or {
            "precision": [],
            "recall": [],
            "average_precision": _finite((report.get("metrics") or {}).get("average_precision")),
            "positive_class": "True",
            "ranking_method": "",
            "support": None,
            "positive_support": None,
        },
        "fading_accuracy": fading_series,
        "class_metrics": primary_cm,
        "class_metrics_by_scope": class_metrics_by_scope,
        "by_scope": ranking_curves,
        **ranking_curves,
    }

    retraining = _extract_retraining(report)

    costs_block = tracking.get("costs") or report.get("costs") or {}
    costs = {
        "unit": (costs_block.get("settings") or {}).get("unit")
        or (obs.get("resources") or {}).get("cost_currency")
        or "unidades",
        "metrics": {
            str(k): _finite(v)
            for k, v in (costs_block.get("metrics") or {}).items()
            if _finite(v) is not None
        },
        "aws": {
            str(k): _finite(v)
            for k, v in ((costs_block.get("aws") or {}).get("metrics") or {}).items()
            if _finite(v) is not None
        },
        "sensitivity": [
            {"alpha": _finite(p.get("alpha")), "value": _finite(p.get("value"))}
            for p in (tracking.get("cost_sensitivity") or [])
            if isinstance(p, dict)
            and _finite(p.get("alpha")) is not None
            and _finite(p.get("value")) is not None
        ],
    }

    sys_raw = report.get("system_metrics") or {}
    sys_samples_raw = sys_raw.get("samples") if isinstance(sys_raw, dict) else []
    sys_samples = []
    if isinstance(sys_samples_raw, list) and sys_samples_raw:
        stride = max(1, len(sys_samples_raw) // 180)
        for idx, s in enumerate(sys_samples_raw[::stride]):
            if isinstance(s, dict):
                raw_vals = s.get("values") if isinstance(s.get("values"), dict) else s
                vals = {
                    str(k): _finite(v)
                    for k, v in raw_vals.items()
                    if k not in ("t", "elapsed_seconds", "timestamp_ms", "values") and _finite(v) is not None
                }
                t_val = _finite(s.get("elapsed_seconds", s.get("t", idx * stride)))
                sys_samples.append(
                    {
                        "t": t_val,
                        "elapsed_seconds": t_val,
                        "timestamp_ms": _finite(s.get("timestamp_ms")),
                        "values": vals,
                    }
                )

    sys_summary_flat: dict[str, float | int | None] = {}
    sys_summary_stats: dict[str, dict[str, float | int | None]] = {}
    if isinstance(sys_raw, dict):
        raw_sum = sys_raw.get("summary")
        if not isinstance(raw_sum, dict):
            raw_sum = sys_raw.get("metrics") if isinstance(sys_raw.get("metrics"), dict) else {}
        for k, v in raw_sum.items():
            if isinstance(v, dict):
                mean_v = _finite(v.get("mean"))
                max_v = _finite(v.get("max"))
                min_v = _finite(v.get("min"))
                if mean_v is not None:
                    sys_summary_flat[str(k)] = mean_v
                if max_v is not None:
                    sys_summary_flat[f"{k}_max"] = max_v
                if min_v is not None:
                    sys_summary_flat[f"{k}_min"] = min_v
                sys_summary_stats[str(k)] = {"mean": mean_v, "max": max_v, "min": min_v}
            elif _finite(v) is not None:
                sys_summary_flat[str(k)] = _finite(v)

    system_metrics = {
        "collector": sys_raw.get("collector") if isinstance(sys_raw, dict) else None,
        "summary": sys_summary_flat,
        "summary_stats": sys_summary_stats,
        "samples": sys_samples,
    }

    training_info = report.get("training") if isinstance(report.get("training"), dict) else {}
    training_budget = {
        "budget": _finite(training_info.get("budget") or config.get("training_budget")),
        "used": _finite(training_info.get("used")),
        "blocked_decisions": _finite(training_info.get("blocked_decisions")),
    }

    effort_dict: dict[str, float | int | None] = {
        str(k): _finite(v)
        for k, v in (obs.get("effort") or {}).items()
        if k != "availability" and _finite(v) is not None
    }
    if "examples_observed" not in effort_dict and observed_val is not None:
        effort_dict["examples_observed"] = observed_val
    if "evaluated_examples" not in effort_dict and labeled_val is not None:
        effort_dict["evaluated_examples"] = labeled_val
    if "updates" not in effort_dict:
        effort_dict["updates"] = len(updates)

    resources_dict: dict[str, float | int | None] = {
        str(k): _finite(v)
        for k, v in (obs.get("resources") or {}).items()
        if k != "availability" and _finite(v) is not None
    }
    if "duration_seconds" not in resources_dict and duration_val is not None:
        resources_dict["duration_seconds"] = duration_val
    peak_rss = _finite(
        report.get("peak_rss_bytes_process") or (tracking.get("metrics") or {}).get("process_peak_rss_bytes")
    )
    if "peak_rss_bytes" not in resources_dict and peak_rss is not None:
        resources_dict["peak_rss_bytes"] = peak_rss

    effort_resources_system = {
        "effort": effort_dict,
        "resources": resources_dict,
        "costs": costs,
        "system_metrics": system_metrics,
        "training_budget": training_budget,
    }

    payload = {
        "meta": meta,
        "all_metrics": _extract_all_metrics(report),
        "windows": windows,
        "epochs": epochs,
        "tracking_series": all_tracking_series,
        "fits": fits,
        "updates": updates,
        "candidates": candidates,
        "alarms": alarms,
        "drift": drift,
        "training_budget": training_budget,
        "effort": effort_dict,
        "resources": resources_dict,
        "retention": retention_payload,
        "confusion": confusion,
        "curves": curves,
        "ranking_curves": ranking_curves,
        "retraining": retraining,
        "costs": costs,
        "system_metrics": system_metrics,
        "effort_resources_system": effort_resources_system,
    }
    return _sanitize_json(payload)
