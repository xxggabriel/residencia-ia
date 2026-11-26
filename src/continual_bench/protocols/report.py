"""Benchmark summary and published paper comparison reporting (CSV, JSON, Markdown, and SVG charts)."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from continual_bench.backends.base import ExecutionBackend
from continual_bench.protocols.base import ProtocolResult

_PALETTE = (
    "#2563eb",
    "#0f766e",
    "#d97706",
    "#dc2626",
    "#7c3aed",
    "#64748b",
)


def _canonical_dataset(name: str) -> str:
    lower = name.strip().lower()
    if lower in {"airplanes", "airlines"}:
        return "airlines"
    if lower in {"electricity", "elecnormnew"}:
        return "electricity"
    return lower


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _render_grouped_bar_svg(
    *,
    title: str,
    ylabel: str,
    categories: list[str],
    series: list[tuple[str, list[float], list[float], str]],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_cats = len(categories)
    n_series = max(1, len(series))
    x = np.arange(n_cats)
    total_width = 0.78 if n_series > 1 else 0.56
    bar_width = total_width / n_series

    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * n_cats), 4.6))
    for idx, (label, means, stds, color) in enumerate(series):
        offset = (idx - (n_series - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=3,
            label=label,
            color=color,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if n_series > 1 or (series and series[0][0]):
        ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


def _render_comparison_svg(metric: str, rows: list[dict[str, Any]]) -> str:
    labels = [f"{r['policy']}\n({r['regime']})" for r in rows]
    obs_means = [float(r["observed_mean"]) for r in rows]
    obs_stds = [float(r["observed_std"]) for r in rows]
    pub_vals = [float(r["published_value"]) for r in rows]
    pub_stds = [float(r["published_spread"]) if r["published_spread"] is not None else 0.0 for r in rows]
    return _render_grouped_bar_svg(
        title=f"Observed vs. Published: {metric}",
        ylabel=metric,
        categories=labels,
        series=[
            ("Observed", obs_means, obs_stds, "#2563eb"),
            ("Published", pub_vals, pub_stds, "#64748b"),
        ],
    )


def _render_benchmark_svg(
    metric: str,
    rows: list[dict[str, Any]],
    *,
    protocol: str,
    dataset: str,
) -> str:
    regimes: list[str] = []
    policies: list[str] = []
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        reg = str(row.get("regime", ""))
        pol = str(row.get("policy", ""))
        if reg not in regimes:
            regimes.append(reg)
        if pol not in policies:
            policies.append(pol)
        lookup[(reg, pol)] = (float(row.get("mean", 0.0)), float(row.get("std", 0.0)))

    series: list[tuple[str, list[float], list[float], str]] = []
    for idx, reg in enumerate(regimes):
        means = [lookup.get((reg, pol), (0.0, 0.0))[0] for pol in policies]
        stds = [lookup.get((reg, pol), (0.0, 0.0))[1] for pol in policies]
        color = _PALETTE[idx % len(_PALETTE)]
        series.append((reg or "default", means, stds, color))

    return _render_grouped_bar_svg(
        title=f"{protocol.upper()} Benchmark ({dataset}): {metric}",
        ylabel=metric,
        categories=policies,
        series=series,
    )


def _render_tradeoff_svg(
    points: list[dict[str, Any]],
    *,
    protocol: str,
    dataset: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for idx, pt in enumerate(points):
        color = _PALETTE[idx % len(_PALETTE)]
        ax.errorbar(
            pt["retrains_mean"],
            pt["accuracy_mean"],
            xerr=pt["retrains_std"],
            yerr=pt["accuracy_std"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            alpha=0.9,
        )
        ax.annotate(
            pt["policy"],
            (pt["retrains_mean"], pt["accuracy_mean"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    ax.set_xlabel("Retrains (price_grid mean)")
    ax.set_ylabel("Average Accuracy (price_grid mean)")
    ax.set_title(f"{protocol.upper()} Accuracy vs. Retrains Trade-off ({dataset})")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


def _extract_price_grid_tradeoff_points(result: ProtocolResult) -> list[dict[str, Any]]:
    """Extract (retrains, average_accuracy) per policy on ``price_grid`` from aggregate or seed summaries."""
    by_policy: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in result.aggregate_summary:
        if row.get("regime") != "price_grid":
            continue
        metric = str(row.get("metric", ""))
        if metric in {"retrains", "average_accuracy"}:
            by_policy[str(row["policy"])][metric] = (
                float(row.get("mean", 0.0)),
                float(row.get("std", 0.0)),
            )

    points: list[dict[str, Any]] = []
    for pol, metrics in by_policy.items():
        if "retrains" in metrics and "average_accuracy" in metrics:
            points.append(
                {
                    "policy": pol,
                    "retrains_mean": metrics["retrains"][0],
                    "retrains_std": metrics["retrains"][1],
                    "accuracy_mean": metrics["average_accuracy"][0],
                    "accuracy_std": metrics["average_accuracy"][1],
                }
            )
    if points:
        return points

    # UPF aggregates combined_cost_auc on price_grid; compute tradeoff points from seed_summaries.
    seed_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"retrains": [], "acc": []})
    policy_order: list[str] = []
    for row in result.seed_summaries:
        if row.get("regime", "price_grid") != "price_grid":
            continue
        if "retrains" not in row or "average_accuracy" not in row:
            continue
        pol = str(row["policy"])
        if pol not in policy_order:
            policy_order.append(pol)
        seed_vals[pol]["retrains"].append(float(row["retrains"]))
        seed_vals[pol]["acc"].append(float(row["average_accuracy"]))

    if not policy_order:
        return []

    import numpy as np

    for pol in policy_order:
        ret = seed_vals[pol]["retrains"]
        acc = seed_vals[pol]["acc"]
        if ret and acc:
            points.append(
                {
                    "policy": pol,
                    "retrains_mean": float(np.mean(ret)),
                    "retrains_std": float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0,
                    "accuracy_mean": float(np.mean(acc)),
                    "accuracy_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                }
            )
    return points


def generate_campaign_report(
    result: ProtocolResult,
    backend: ExecutionBackend,
    *,
    run_id: str | None = None,
    regimes: dict[str, float] | None = None,
    grid_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Generate generic summary tables and SVG plots for any dataset and protocol."""
    summary_rows: list[dict[str, Any]] = []
    for agg in result.aggregate_summary:
        summary_rows.append(
            {
                "protocol": str(agg.get("protocol", result.protocol)),
                "dataset": str(agg.get("dataset", result.dataset)),
                "regime": str(agg.get("regime", "")),
                "metric": str(agg.get("metric", "")),
                "policy": str(agg.get("policy", "")),
                "mean": round(float(agg.get("mean", 0.0)), 6),
                "std": round(float(agg.get("std", 0.0)), 6),
                "n_seeds": int(agg.get("n_seeds", 0)),
            }
        )
    summary_rows.sort(key=lambda r: (r["regime"], r["metric"], r["policy"]))

    def _rel(filename: str) -> str:
        return f"{run_id}/{filename}" if run_id else filename

    artifacts: dict[str, str] = {}
    artifacts["benchmark_summary.csv"] = backend.write_text(
        _rel("benchmark_summary.csv"), _rows_to_csv(summary_rows)
    )
    artifacts["benchmark_summary.json"] = backend.write_json(
        _rel("benchmark_summary.json"), summary_rows
    )

    md_lines = [
        f"# Benchmark Summary ({result.protocol} / {result.dataset})",
        "",
    ]
    if grid_bounds is not None:
        md_lines.append(f"- Grid bounds: `[{grid_bounds[0]:.6g}, {grid_bounds[1]:.6g}]`")
    if regimes:
        reg_str = ", ".join(f"`{k}={v}`" for k, v in regimes.items())
        md_lines.append(f"- Regimes: {reg_str}")
    if grid_bounds is not None or regimes:
        md_lines.append("")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        grouped[row["regime"]][row["metric"]].append(row)

    for regime_name in sorted(grouped):
        md_lines.append(f"## Regime: `{regime_name or 'default'}`")
        md_lines.append("")
        for metric_name in sorted(grouped[regime_name]):
            md_lines.append(f"### Metric: `{metric_name}`")
            md_lines.append("")
            md_lines.append("| Policy | Mean | Std | Seeds |")
            md_lines.append("| --- | --- | --- | --- |")
            for row in sorted(grouped[regime_name][metric_name], key=lambda r: r["policy"]):
                md_lines.append(
                    f"| {row['policy']} | {row['mean']} | {row['std']} | {row['n_seeds']} |"
                )
            md_lines.append("")

    artifacts["benchmark_summary.md"] = backend.write_text(
        _rel("benchmark_summary.md"), "\n".join(md_lines)
    )

    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result.aggregate_summary:
        by_metric[str(row.get("metric", ""))].append(row)

    for metric_name, m_rows in by_metric.items():
        if not metric_name:
            continue
        try:
            svg_text = _render_benchmark_svg(
                metric_name,
                m_rows,
                protocol=result.protocol,
                dataset=result.dataset,
            )
            filename = f"benchmark_{metric_name}.svg"
            artifacts[filename] = backend.write_text(_rel(filename), svg_text)
        except Exception:
            # Keep reporting resilient if matplotlib is unavailable in minimal environments.
            pass

    tradeoff_points = _extract_price_grid_tradeoff_points(result)
    if tradeoff_points:
        try:
            svg_text = _render_tradeoff_svg(
                tradeoff_points,
                protocol=result.protocol,
                dataset=result.dataset,
            )
            artifacts["benchmark_tradeoff.svg"] = backend.write_text(
                _rel("benchmark_tradeoff.svg"), svg_text
            )
        except Exception:
            pass

    return {"rows": summary_rows, "artifacts": artifacts}


def compare_with_published_reference(
    aggregate_rows: list[dict[str, Any]],
    paper_id: str,
    dataset: str,
    reference_csv: str | Path,
    backend: ExecutionBackend,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare aggregated benchmark results against published paper reference values."""
    ref_path = Path(reference_csv)
    if not ref_path.exists():
        return {"rows": [], "artifacts": {}}

    canonical_ds = _canonical_dataset(dataset)
    ref_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    with ref_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("paper_id") != paper_id:
                continue
            if _canonical_dataset(row.get("dataset", "")) != canonical_ds:
                continue
            key = (row["policy"], row["metric"], row["regime"])
            ref_index[key] = row

    comparison_rows: list[dict[str, Any]] = []
    for agg in aggregate_rows:
        policy = str(agg["policy"])
        metric = str(agg["metric"])
        regime = str(agg.get("regime", ""))
        ref = ref_index.get((policy, metric, regime))
        if ref is None:
            continue
        obs_mean = float(agg["mean"])
        obs_std = float(agg.get("std", 0.0))
        pub_val = float(ref["published_value"])
        pub_spread_raw = ref.get("published_spread", "")
        pub_spread = float(pub_spread_raw) if pub_spread_raw not in ("", None) else None
        abs_diff = obs_mean - pub_val
        rel_diff_pct = (100.0 * abs_diff / abs(pub_val)) if pub_val != 0.0 else None
        comparison_rows.append(
            {
                "paper_id": paper_id,
                "dataset": canonical_ds,
                "policy": policy,
                "metric": metric,
                "regime": regime,
                "observed_mean": round(obs_mean, 6),
                "observed_std": round(obs_std, 6),
                "published_value": round(pub_val, 6),
                "published_spread": round(pub_spread, 6) if pub_spread is not None else None,
                "abs_diff": round(abs_diff, 6),
                "rel_diff_pct": round(rel_diff_pct, 4) if rel_diff_pct is not None else None,
            }
        )

    if not comparison_rows:
        return {"rows": [], "artifacts": {}}

    def _rel(filename: str) -> str:
        return f"{run_id}/{filename}" if run_id else filename

    artifacts: dict[str, str] = {}
    artifacts["paper_comparison.csv"] = backend.write_text(
        _rel("paper_comparison.csv"), _rows_to_csv(comparison_rows)
    )
    artifacts["paper_comparison.json"] = backend.write_json(
        _rel("paper_comparison.json"), comparison_rows
    )

    md_lines = [
        f"# Paper Comparison ({paper_id} / {canonical_ds})",
        "",
        "| Policy | Metric | Regime | Observed (Mean +/- Std) | Published | Abs Diff | Rel Diff (%) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in comparison_rows:
        pub_str = (
            f"{r['published_value']} +/- {r['published_spread']}"
            if r["published_spread"] is not None
            else str(r["published_value"])
        )
        rel_str = f"{r['rel_diff_pct']}%" if r["rel_diff_pct"] is not None else "N/A"
        md_lines.append(
            f"| {r['policy']} | {r['metric']} | {r['regime']} | "
            f"{r['observed_mean']} +/- {r['observed_std']} | {pub_str} | {r['abs_diff']} | {rel_str} |"
        )
    artifacts["paper_comparison.md"] = backend.write_text(
        _rel("paper_comparison.md"), "\n".join(md_lines) + "\n"
    )

    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in comparison_rows:
        by_metric[r["metric"]].append(r)

    for metric_name, m_rows in by_metric.items():
        try:
            svg_text = _render_comparison_svg(metric_name, m_rows)
            filename = f"paper_comparison_{metric_name}.svg"
            artifacts[filename] = backend.write_text(_rel(filename), svg_text)
        except Exception:
            # Keep reporting resilient if matplotlib is unavailable in minimal environments.
            pass

    return {"rows": comparison_rows, "artifacts": artifacts}
