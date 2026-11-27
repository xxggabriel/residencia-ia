"""Tests for CARA & UPF paper plots and telemetry extraction in the benchmark dashboard."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pytest

from continual_bench.application.components import default_registry, make_source
from continual_bench.application.runner import evaluate, prepare
from continual_bench.config import Config, DatasetConfig
from continual_bench.evaluation.typed import run_typed
from continual_bench.infrastructure.dashboard_page import (
    build_dashboard_payload,
    render_dashboard_html,
    write_dashboard_html,
)
from continual_bench.infrastructure.dashboard_payload import (
    build_dashboard_payload as build_payload_direct,
)


def _assert_json_finite(value: Any, path: str = "root") -> None:
    """Recursively verify that every numeric value in the payload is finite (no NaN/Inf)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(value), f"Non-finite number at {path}: {value!r}"
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{idx}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"Non-string dict key at {path}: {k!r}"
            _assert_json_finite(v, f"{path}.{k}")
        return
    raise AssertionError(f"Unexpected non-JSON type at {path}: {type(value)!r}")


def _extract_embedded_json(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script\s+id="benchmark-data"\s+type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None, "Missing <script id='benchmark-data' type='application/json'> tag"
    raw = match.group(1)
    assert "NaN" not in raw and "Infinity" not in raw
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


@pytest.fixture(scope="module")
def incremental_cara_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    state_dir = tmp_path_factory.mktemp("cara_incremental")
    config = Config.model_validate(
        {
            "name": "test-cara-paper-plots",
            "mode": "incremental",
            "dataset": {"name": "sea", "samples": 160, "label_delay": 2, "seed": 42},
            "warmup": 40,
            "metric_window": 40,
            "batch_size": 40,
            "policy": {"name": "cara"},
        }
    )
    manifest = prepare(config, str(state_dir))
    return evaluate(config, manifest, str(state_dir))


@pytest.fixture(scope="module")
def streaming_drift_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    state_dir = tmp_path_factory.mktemp("drift_streaming")
    config = Config.model_validate(
        {
            "name": "test-drift-streaming-plots",
            "mode": "streaming",
            "dataset": {"name": "hyperplane", "samples": 160, "label_delay": 2, "seed": 7},
            "warmup": 40,
            "metric_window": 40,
            "batch_size": 40,
            "policy": {"name": "drift"},
        }
    )
    manifest = prepare(config, str(state_dir))
    return evaluate(config, manifest, str(state_dir))


@pytest.mark.parametrize(
    "mode,policy_cfg",
    [
        ("incremental", {"name": "cara"}),
        ("incremental", {"name": "upf"}),
        ("incremental", {"name": "oracle"}),
        ("streaming", {"name": "drift"}),
        ("streaming", {"name": "online", "action": "learn_one"}),
    ],
)
def test_build_dashboard_payload_extracts_all_sections_across_modes_and_policies(
    mode: str,
    policy_cfg: dict[str, Any],
    tmp_path: Any,
) -> None:
    config = Config.model_validate(
        {
            "name": f"test-{mode}-{policy_cfg['name']}",
            "mode": mode,
            "dataset": {"name": "sea", "samples": 120, "label_delay": 2, "seed": 11},
            "warmup": 30,
            "metric_window": 30,
            "batch_size": 30,
            "policy": policy_cfg,
        }
    )
    manifest = prepare(config, str(tmp_path))
    report = evaluate(config, manifest, str(tmp_path))

    payload = build_dashboard_payload(report)
    assert payload == build_payload_direct(report)
    _assert_json_finite(payload)
    # Must serialize strictly with allow_nan=False
    serialized = json.dumps(payload, allow_nan=False)
    assert serialized

    for top_key in (
        "meta",
        "all_metrics",
        "windows",
        "fits",
        "updates",
        "candidates",
        "alarms",
        "retention",
        "curves",
        "confusion",
        "retraining",
        "effort_resources_system",
    ):
        assert top_key in payload, f"Missing top-level payload key {top_key}"

    meta = payload["meta"]
    assert meta["dataset"] == "sea"
    assert meta["policy"] == policy_cfg["name"]
    assert meta["mode"] == mode
    assert meta["complete"] is True
    assert meta["observed"] == 120
    assert meta["labeled"] is not None and meta["labeled"] > 0

    assert len(payload["all_metrics"]) > 15
    metric_groups = {row["group"] for row in payload["all_metrics"]}
    assert "tracking" in metric_groups
    assert "summary" in metric_groups
    assert any(g.startswith("scope/") for g in metric_groups)
    for row in payload["all_metrics"]:
        assert {"group", "name", "value", "status", "reason", "unit_or_scope"} <= set(row.keys())

    assert len(payload["windows"]) > 0
    for win in payload["windows"]:
        assert "step" in win and "scope" in win and "metrics" in win
        assert isinstance(win["metrics"], dict)

    assert isinstance(payload["fits"], list) and len(payload["fits"]) > 0
    for fit in payload["fits"]:
        assert "fit_id" in fit
        assert "observation_steps" in fit
        assert "observation_scopes" in fit

    assert isinstance(payload["updates"], list)
    assert isinstance(payload["alarms"], list)

    retention = payload["retention"]
    for r_key in (
        "matrix",
        "checkpoints",
        "periods",
        "metric",
        "absolute_retention",
        "gem_bwt",
        "post_learning_fm",
        "series",
        "tracking_series",
        "continual_matrix",
        "retention_probes",
    ):
        assert r_key in retention, f"Missing retention.{r_key}"
    assert len(retention["matrix"]) > 0
    assert retention["absolute_retention"] is not None

    curves = payload["curves"]
    assert "roc_curve" in curves and "fpr" in curves["roc_curve"] and "tpr" in curves["roc_curve"]
    assert "pr_curve" in curves and "precision" in curves["pr_curve"] and "recall" in curves["pr_curve"]
    assert "fading_accuracy" in curves and isinstance(curves["fading_accuracy"], list)
    assert "class_metrics" in curves
    assert {"classes", "class_support", "class_recall"} <= set(curves["class_metrics"].keys())

    confusion = payload["confusion"]
    assert "prequential" in confusion
    assert len(confusion["prequential"]["matrix"]) == 2
    assert len(confusion["prequential"]["classes"]) == 2

    ers = payload["effort_resources_system"]
    assert {"effort", "resources", "costs", "system_metrics", "training_budget"} <= set(ers.keys())
    assert ers["effort"].get("examples_observed") == 120
    assert "samples" in ers["system_metrics"]


def test_typed_online_streaming_payload_and_dashboard_html(tmp_path: Any) -> None:
    source = make_source(default_registry(), DatasetConfig(name="sea", samples=120, label_delay=2, seed=9))
    report = run_typed(
        source,
        tmp_path / "typed-online",
        model={"name": "logistic"},
        policy="online",
        warmup=30,
        window=30,
    )
    payload = build_dashboard_payload(report)
    _assert_json_finite(payload)
    assert payload["meta"]["policy"] == "online"
    assert payload["retention"]["absolute_retention"] is not None
    html = render_dashboard_html(report)
    assert _extract_embedded_json(html) == payload


def test_retraining_paper_structures_extracted_completely(
    incremental_cara_report: dict[str, Any],
) -> None:
    payload = build_dashboard_payload(incremental_cara_report)
    ret = payload["retraining"]
    assert ret is not None
    assert ret["available"] is True

    setup = ret["setup"]
    for s_key in (
        "batches",
        "offline",
        "online",
        "initial",
        "focus_alpha",
        "delta",
        "samples",
        "rows",
        "fit_seconds",
    ):
        assert s_key in setup, f"Missing retraining.setup.{s_key}"
    assert setup["batches"] >= 4
    assert setup["online"] >= 2

    alphas = ret["alphas"]
    assert isinstance(alphas, list) and len(alphas) >= 2

    rules = ret["rules"]
    for rule_name in ("oracle", "cara", "cara_cumulative", "cara_periodic", "upf", "frozen"):
        assert rule_name in rules, f"Missing retraining rule {rule_name}"
        r_info = rules[rule_name]
        for field in (
            "label",
            "auc",
            "mean_scpe",
            "mean_strategy_scpe",
            "online_decision_seconds",
            "offline_seconds",
            "c_hat",
            "retrains",
            "alphas",
            "costs",
            "retrains_curve",
            "accuracies_curve",
            "scpe_curve",
            "strategy_costs_curve",
            "error_oracle_gap_curve",
            "strategy_oracle_gap_curve",
            "retrain_points_by_alpha",
            "parameters_by_alpha",
            "curve",
        ):
            assert field in r_info, f"Missing retraining.rules.{rule_name}.{field}"
        assert len(r_info["alphas"]) == len(alphas)
        assert len(r_info["costs"]) == len(alphas)
        assert len(r_info["retrains_curve"]) == len(alphas)
        assert len(r_info["accuracies_curve"]) == len(alphas)
        assert len(r_info["strategy_costs_curve"]) == len(alphas)
        assert len(r_info["retrain_points_by_alpha"]) == len(alphas)
        assert len(r_info["parameters_by_alpha"]) == len(alphas)

    # Summary table & Low/Focus/High regime table (UPF Table 2 / CARA Tables 2-4)
    assert "table" in ret
    for regime in ("low", "focus", "high"):
        assert regime in ret["table"]
        assert "cara" in ret["table"][regime]
        assert "upf" in ret["table"][regime]
    assert isinstance(ret.get("summary_table"), list) and len(ret["summary_table"]) >= 5

    # Performance Matrix pe_{i,j} (UPF Fig. 1) and Staleness Matrix Psi_bar_{t,t'} (CARA Fig. 3a)
    perf_mat = ret["performance_matrix"]
    assert isinstance(perf_mat, dict)
    assert len(perf_mat["err"]) == setup["batches"]
    assert len(perf_mat["acc"]) == setup["batches"]
    stale_mat = ret["staleness_matrix"]
    assert isinstance(stale_mat, list) and len(stale_mat) == setup["batches"]

    # Cumulative strategy cost over online batches (CARA Fig. B.10)
    cum_cost = ret["cumulative_strategy_cost"]
    assert "cara" in cum_cost and "upf" in cum_cost and "oracle" in cum_cost
    assert len(cum_cost["cara"]) == setup["online"]

    # Wrong-alpha sensitivity heatmap (UPF Fig. 3 Appendix A.2)
    wrong_alpha = ret["wrong_alpha"]
    for r_key in ("cara", "upf"):
        assert r_key in wrong_alpha
        wa = wrong_alpha[r_key]
        assert len(wa["alphas"]) == len(alphas)
        assert len(wa["real_alphas"]) == len(alphas)
        assert len(wa["provided_alphas"]) == len(alphas)
        assert len(wa["percent_increase"]) == len(alphas)
        assert len(wa["percent_increase"][0]) == len(alphas)

    # UPF Forecaster diagnostics (UPF Figs. 5-8)
    forecast = ret["forecast"]
    assert isinstance(forecast, dict)
    for f_key in ("mae", "bias", "pairs", "predictions", "observed", "residuals"):
        assert f_key in forecast
    assert forecast["mae"] is not None
    assert len(forecast["pairs"]) > 0
    assert len(forecast["residuals"]) == len(forecast["pairs"])

    # Oracle schedule at focus & Proposition 3.1 bounds (UPF Fig. 4)
    assert isinstance(ret["oracle_schedule_at_focus"], list)
    prop = ret["proposition_bounds"]
    assert prop["lipschitz_L"] is not None and prop["lipschitz_L"] >= 0.0
    assert "thresholds" in prop
    for bound_key in ("r<=0", "r<=1", "r<=2"):
        assert bound_key in prop["thresholds"]
        assert prop["thresholds"][bound_key] is not None
    assert len(prop["upper_bound_curve"]) == len(alphas)


def test_render_and_write_dashboard_html_contains_all_paper_plots_and_controls(
    incremental_cara_report: dict[str, Any],
    streaming_drift_report: dict[str, Any],
    tmp_path: Any,
) -> None:
    for name, report in (
        ("cara", incremental_cara_report),
        ("drift", streaming_drift_report),
    ):
        out_dir = tmp_path / name
        out_dir.mkdir(parents=True, exist_ok=True)
        html_str = render_dashboard_html(report)
        written_path = write_dashboard_html(out_dir, report)

        assert written_path == out_dir / "dashboard.html"
        assert written_path.is_file()
        file_html = written_path.read_text(encoding="utf-8")
        assert file_html == html_str

        # Embedded #benchmark-data JSON parses cleanly and matches build_dashboard_payload
        embedded_payload = _extract_embedded_json(file_html)
        expected_payload = build_dashboard_payload(report)
        assert embedded_payload == expected_payload

        # Required interactive controls (live retraining slider, alpha slider, smoothing, etc.)
        required_controls = (
            'id="benchmark-data"',
            'id="mode-global"',
            'id="mode-live"',
            'id="timeline-slider"',
            'id="timeline-play"',
            'id="timeline-prev"',
            'id="timeline-next"',
            'id="smooth-strategy"',
            'id="smooth-window"',
            'id="policy-selector"',
            'id="alpha-slider"',
            'id="cost-type-selector"',
            'id="upf-fig1-metric"',
            'id="wrong-alpha-rule"',
            'id="metric-search"',
        )
        for ctrl in required_controls:
            assert ctrl in file_html, f"Missing interactive control {ctrl} in {name} dashboard.html"

        # Smoothing options
        for smooth_opt in ('value="raw"', 'value="sma"', 'value="ema"', 'value="median"', 'value="savgol"'):
            assert smooth_opt in file_html

        # Distinct marker shape definitions (circles, squares, triangles, diamonds, stars, crosses)
        for shape in ("circle", "square", "triangle-up", "diamond", "cross", "star", "triangle-down"):
            assert shape in file_html, f"Missing marker shape {shape!r} in dashboard.html"

        # All CARA & UPF paper plot containers, matrices, curves, tables, and formula guides
        required_containers = (
            'id="upf-fig1-card"',
            'id="upf-fig1-chart"',
            'id="upf-fig1-overview-chart"',
            'id="alpha-curves-card"',
            'id="alpha-cost-chart"',
            'id="alpha-retrains-chart"',
            'id="alpha-accuracy-chart"',
            'id="alpha-scpe-chart"',
            'id="matrices-trajectory-card"',
            'id="staleness-matrix-chart"',
            'id="performance-matrix-chart"',
            'id="cara-params-card"',
            'id="cara-params-chart"',
            'id="cumulative-cost-card"',
            'id="cumulative-cost-chart"',
            'id="tradeoff-pareto-card"',
            'id="tradeoff-pareto-chart"',
            'id="wrong-alpha-card"',
            'id="wrong-alpha-heatmap"',
            'id="forecast-diagnostics-card"',
            'id="forecast-scatter-chart"',
            'id="forecast-hist-chart"',
            'id="paper-summary-tables-card"',
            'id="cara-summary-table"',
            'id="upf-regime-table"',
            'id="auc-bar-chart"',
            'id="prequential-chart"',
            'id="fading-chart"',
            'id="retention-chart"',
            'id="retention-matrix-container"',
            'id="continual-matrix-container"',
            'id="roc-chart"',
            'id="pr-chart"',
            'id="confusion-matrix-container"',
            'id="class-recall-chart"',
            'id="drift-timeline-chart"',
            'id="effort-fits-chart"',
            'id="fit-internal-steps-chart"',
            'id="system-telemetry-chart"',
            'id="costs-breakdown-container"',
            'id="fits-table-container"',
            'id="updates-table-container"',
            'id="all-metrics-container"',
            'id="insights-banner"',
            'id="formulas-guide-card"',
        )
        for cid in required_containers:
            assert cid in file_html, f"Missing plot/table container {cid} in {name} dashboard.html"

        # Paper references and mathematical guide keywords
        for keyword in (
            "Mahadevan &amp; Mathioudakis",
            "Regol et al.",
            "Proposição 3.1",
            "SCPE",
            "Staleness",
            "GEM BWT",
            "cl_forgetting",
            "absolute_retention",
        ):
            assert keyword in file_html, f"Missing interpretability/paper keyword {keyword!r}"


def test_build_dashboard_payload_sanitizes_non_finite_floats() -> None:
    dirty_report: dict[str, Any] = {
        "run_key": "nan-safety-check",
        "dataset": {"name": "synthetic", "seed": 1},
        "config": {"mode": "streaming", "policy": {"name": "drift"}, "warmup": float("nan")},
        "metrics": {"accuracy": 0.85, "bad_nan": float("nan"), "bad_inf": float("inf")},
        "windows": [
            {"end_seq": 10, "accuracy": 0.9, "loss": float("-inf")},
            {"end_seq": 20, "accuracy": float("nan"), "loss": 0.2},
        ],
        "retraining": {
            "setup": {"batches": 2, "offline_batches": 1, "lipschitz_L": float("nan")},
            "alphas": [0.0, float("inf"), 0.5],
            "rules": {
                "cara": {
                    "auc": float("nan"),
                    "curve": [
                        {"alpha": 0.0, "cost": 0.4, "retrains": 1, "retrain_points": [1]},
                        {"alpha": 0.5, "cost": float("inf"), "retrains": 0, "retrain_points": []},
                    ],
                }
            },
            "staleness_matrix": [[0.0, float("nan")], [None, 0.0]],
            "performance_matrix": {"err": [[0.1, 0.2], [float("nan"), 0.15]]},
        },
    }
    payload = build_dashboard_payload(dirty_report)
    _assert_json_finite(payload)
    html = render_dashboard_html(dirty_report)
    embedded = _extract_embedded_json(html)
    assert embedded == payload
