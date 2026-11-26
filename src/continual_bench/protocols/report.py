"""Published paper comparison reporting (CSV, JSON, Markdown, and SVG charts)."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from continual_bench.backends.base import ExecutionBackend


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


def _render_comparison_svg(metric: str, rows: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{r['policy']}\n({r['regime']})" for r in rows]
    obs_means = [float(r["observed_mean"]) for r in rows]
    obs_stds = [float(r["observed_std"]) for r in rows]
    pub_vals = [float(r["published_value"]) for r in rows]
    pub_stds = [float(r["published_spread"]) if r["published_spread"] is not None else 0.0 for r in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(labels)), 4.5))
    ax.bar(
        x - width / 2, obs_means, width, yerr=obs_stds, capsize=3,
        label="Observed", color="#2563eb", alpha=0.9,
    )
    ax.bar(
        x + width / 2, pub_vals, width, yerr=pub_stds, capsize=3,
        label="Published", color="#64748b", alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"Observed vs. Published: {metric}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


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
