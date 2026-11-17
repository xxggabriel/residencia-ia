import json
import re

import pytest

from continual_bench.evaluation.tracking_observations import batch_quality, observations
from continual_bench.infrastructure.mlflow_figures import _pyplot
from continual_bench.infrastructure.training_figures import (
    MAX_POINTS,
    _draw_panel,
    _fit_quality_panels,
    _fit_series,
    _ranking_panel,
    _sample,
    _window_panels,
    write_observation_figures,
)

pytestmark = pytest.mark.observation_pngs


def _fit(number, scope, quality, status, *, epochs=True):
    all_steps = [{"step": i, "metrics": {"training/native_loss": 1 / i, "training/duration_seconds": 0.2}}
                 for i in range(1, 5)]
    return {
        "fit_id": f"fit-{number:06d}",
        "action": "initial_fit" if number == 1 else "reset_retrain",
        "promotion_status": status,
        "deployed_version": number if status != "rejected" else None,
        "checkpoint": {"sha256": f"sample-{number}", "file": f"checkpoints/{number}.bin"},
        "end_seq": 10 * number,
        "observation_steps": all_steps,
        "observation_epochs": [
            {"epoch": 1, "first_step": 1, "last_step": 2, "scopes": {scope: quality}},
            {"epoch": 2, "first_step": 3, "last_step": 4, "scopes": {scope: quality}},
        ] if epochs else [],
        "best_epoch": 1 if epochs else None,
        "checkpoint_epoch": 1 if epochs else None,
        "last_epoch": 2 if epochs else None,
        "summary": {"early_stopped": epochs},
        "observation_scopes": {scope: quality},
        "observation_effort": {"unique_training_examples": 12, "training_presentations": 24,
                               "steps": 4, "epochs": 2 if epochs else None},
        "observation_resources": {"training_seconds": 0.8, "duration_seconds": 2.0, "peak_rss_bytes": 10**7,
                                  "rss_scope": "process_lifetime_high_water_mark"},
    }


def _binary_report():
    quality = batch_quality(
        [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], classes=[0, 1], positive_class=1,
        scores=[0.1, 0.7, 0.22, 0.22, 0.15], low_fpr=True,
    )
    fits = [_fit(1, "validation", quality, "promoted"), _fit(2, "validation", quality, "promoted"),
            _fit(3, "validation", quality, "rejected", epochs=False)]
    windows = [{"step": i * 10, "scope": "prequential", "quality": quality,
                "fit_id": fits[i - 1]["fit_id"], "model_version": i,
                "checkpoint": f"window-checkpoint-{i}"} for i in (1, 2)]
    return {
        "training_history": fits,
        "tracking_observations": observations(
            "binary", {"prequential": quality}, windows=windows,
            effort={"unique_training_examples": 36, "training_presentations": 72},
            resources={"duration_seconds": 10, "cost_kind": "estimated", "cost_currency": "USD",
                       "cost_total": 0.25, "cost_per_1000_evaluated": 2.5},
            drift={"alarms": [12], "changes": [10, 29], "precision": 1.0, "recall": 1.0, "f1": 1.0,
                   "delays": [2], "delay_unit": "exemplos", "censored_count": 1},
            retention={"matrix": [[0.8, None], [0.75, 0.7]], "checkpoints": ["v1", "v2"],
                       "periods": ["P1", "P2"], "metric": "accuracy",
                       "series": [{"step": 1, "absolute_retention": 0.8}, {"step": 2, "gem_bwt": -0.05}]},
        ),
    }


def test_stage_parent_children_sources_and_explicit_unavailable_epoch(tmp_path):
    image = pytest.importorskip("PIL.Image")
    pytest.importorskip("matplotlib")
    report = _binary_report()
    ids = {fit["fit_id"]: f"run-{i}" for i, fit in enumerate(report["training_history"], 1)}
    manifest = write_observation_figures(tmp_path, report, parent_run_id="main", child_run_ids=ids)
    assert manifest["parent_run_id"] == "main"
    assert set(manifest["children"]) == set(ids)
    entries = {item["filename"]: item for item in manifest["figures"]}
    for name in (
        "overview", "prequential-pr", "prequential-roc", "prequential-confusion", "prequential-class-recall",
        "effort-by-fit", "resources-by-fit", "retention-matrix", "retention", "drift", "detection-delays",
        "detector-quality",
    ):
        entry = entries[f"overview/{name}.png"]
        assert entry["run_id"] == "main" and entry["metric_version"] == "benchmark-observations-v1"
        with image.open(tmp_path / entry["path"]) as png:
            assert png.format == "PNG" and png.width >= 900 and png.height >= 350
            darkest, brightest = png.convert("L").getextrema()
            assert brightest - darkest > 160

    epochs = entries["fits/fit-000002/epochs.png"]
    assert epochs["run_id"] == "run-2"
    assert epochs["checkpoint"]["sha256"] == "sample-2"
    assert epochs["source"]["pointers"] == ["/observation_epochs"]
    assert epochs["artifact_path"] == "tracking-observations/epochs.png"
    assert any(series["name"] == "Macro-F1 · validation" for series in epochs["series"])
    assert any(panel["title"].startswith("Perda ·") for panel in epochs["unavailable_panels"])
    assert entries["fits/fit-000003/steps.png"]["promotion_status"] == "rejected"
    unavailable = entries["fits/fit-000003/epochs.png"]
    assert unavailable["path"] is None and "sem épocas" in unavailable["availability"]["reason"]
    data_path = tmp_path / "fits/fit-000003/data.json"
    assert json.loads(data_path.read_text()) == report["training_history"][-1]

    html = (tmp_path / "overview/overview.html").read_text()
    match = re.search(r'<script type="application/json" id="chart-data">(.*?)</script>', html)
    panels = json.loads(match.group(1))
    trajectory = next(panel for panel in panels if "por janela" in panel["title"])
    assert trajectory["series"][0]["points"][-1]["checkpoint"] == "window-checkpoint-2"
    assert trajectory["markers"][0]["x"] == 20
    comparisons = next(panel for panel in panels if "por ajuste" in panel["title"])
    values = comparisons["series"][0]["points"]
    expected = report["training_history"][0]["observation_scopes"]["validation"]["macro_f1"]
    assert [point["y"] for point in values] == [expected] * 3
    assert values[-1]["support"] == 5 and values[-1]["checkpoint"] == "sample-3"
    assert values[-1]["status"] == "rejected"
    assert 'checkbox.addEventListener' in html and 'pointerenter' in html


@pytest.mark.parametrize("task", ["multiclass", "regression"])
def test_scope_specific_plots_and_availability(tmp_path, task):
    pytest.importorskip("matplotlib")
    if task == "multiclass":
        data = batch_quality([2, 2, 1, 1], [1, 2, 1, 0], classes=[0, 1, 2])
        expected = {"test-confusion.png", "test-class-recall.png"}
    else:
        data = batch_quality([1, 3, 4], [1, 4, 1], target_unit="°C")
        expected = {"test-residuals.png"}
    report = {"tracking_observations": observations(task, {"test": data})}
    manifest = write_observation_figures(tmp_path, report, allow_html=False)
    assert {path.split("/")[-1] for path in manifest["parent"]} >= expected
    assert manifest["children"] == {}
    notices = {fig["filename"]: fig["availability"]["reason"] for fig in manifest["figures"]}
    assert notices["overview/overview.html"] == "HTML indisponível neste publicador"
    assert "históricas" in notices["overview/retention-matrix.png"]
    assert not (tmp_path / "overview/overview.html").exists()


def test_long_plot_retains_extrema_gaps_and_version_boundaries():
    count = MAX_POINTS * 9
    absent = range(633, 707)
    points = ({"x": x, "y": None if x in absent else 400 if x == 2087 else -100 if x == 3123 else x % 37,
               "version": x // 1000, "event": x == 2491} for x in range(count))
    rendered = {p["x"]: p for p in _sample(points, count)}
    assert len(rendered) < MAX_POINTS
    assert {0, 632, 633, 706, 707, 999, 1000, 2087, 2491, 3123, count - 1} <= set(rendered)
    assert rendered[633]["y"] is None and rendered[706]["y"] is None
    assert rendered[2087]["y"] == 400 and rendered[3123]["y"] == -100


def test_precision_recall_uses_each_methods_direction_for_step_interpolation():
    exact = _binary_report()["tracking_observations"]["scopes"]["prequential"]
    recall = exact["pr_curve"]["recall"]
    precision = exact["pr_curve"]["precision"]
    approximate = {
        **exact, "ranking_method": "uniform_histogram_4096_bins",
        "pr_curve": {"recall": recall[::-1], "precision": precision[::-1]},
    }
    descending = _ranking_panel(exact, "binary", "pr", "test")
    ascending = _ranking_panel(approximate, "binary", "pr", "test")
    assert descending.series[0].style == "steps-post" and "Exata" in descending.title
    assert ascending.series[0].style == "steps-pre" and "4096 bins" in ascending.title


def test_dense_adjustment_labels_and_integer_offsets_remain_readable():
    pytest.importorskip("matplotlib")
    plt = _pyplot()
    quality = {"macro_f1": 0.91, "accuracy": 0.911, "evaluated_examples": 125}
    fits = [_fit(i, "validation", quality, "promoted") for i in range(1, 11)]
    run = observations("binary", {"validation": quality}, windows=[
        {"scope": "validation", "step": step, "quality": quality} for step in (9999, 10000)
    ])
    panel = _fit_quality_panels(run, fits)[0]
    windows = _window_panels(run, fits)
    assert windows[0].integer_x
    for coordinate, label in (("epoch", "Época"), ("candidate", "Candidato (ordem do ajuste)")):
        rows = [{**row, "coordinate": coordinate, "step": i} for i, row in enumerate(run["windows"], 1)]
        assert _window_panels({**run, "windows": rows}, fits)[0].x_label == label
    fig, (comparison, trajectory) = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    try:
        _draw_panel(comparison, panel)
        _draw_panel(trajectory, windows[0])
        fig.canvas.draw()
        labels = list(comparison.texts)
        assert [label.get_text() for label in labels] == ["0.91\nn=125"] * 10
        boxes = [label.get_window_extent() for label in labels]
        assert all(not first.overlaps(second) for first, second in zip(boxes, boxes[1:]))
        assert trajectory.xaxis.get_offset_text().get_text() == ""
        assert {"9999", "10000"} <= {label.get_text() for label in trajectory.get_xticklabels()}
    finally:
        plt.close(fig)


def test_ranking_plot_reports_only_examples_with_valid_scores():
    quality = _binary_report()["tracking_observations"]["scopes"]["prequential"]
    quality.update(evaluated_examples=7, ranking_evaluated_examples=2, ranking_positive_examples=1)
    panel = _ranking_panel(quality, "binary", "pr", "test")
    assert "n=2 de 7 avaliados" in panel.title
    assert "n positivo=1" in panel.title
    assert all(point["support"] == 2 for point in panel.series[0].points)


def test_step_and_epoch_panels_use_integer_coordinates_with_unscoped_loss():
    fit = {
        "fit_id": "fit-000001",
        "observation_steps": [{"step": i, "metrics": {"loss": 1 / i}} for i in range(1, 4)],
        "observation_epochs": [{"epoch": i, "metrics": {"loss": 1 / i}} for i in range(1, 4)],
    }
    for dimension in ("step", "epoch"):
        panels = _fit_series(fit, dimension)
        assert panels[0].integer_x
        assert [point["x"] for point in panels[0].series[0].points] == [1, 2, 3]


def test_retraining_pages_keep_duplicate_positions_partial_fits_and_zeros(tmp_path, monkeypatch):
    monkeypatch.setattr("continual_bench.infrastructure.training_figures.matplotlib_available", lambda: False)
    report = _binary_report()
    fits = report["training_history"]
    fits[1]["end_seq"] = fits[0]["end_seq"]
    fits[1]["status"] = "partial"
    fits[1]["observation_steps"] = None
    fits[1]["observation_epochs"] = None
    fits[2]["observation_resources"]["duration_seconds"] = 0
    report["tracking_observations"]["lineage"]["dataset"] = "</script><script>unsafe()</script>"
    manifest = write_observation_figures(tmp_path, report)

    def load(name):
        path = tmp_path / f"{name}.html"
        document = path.read_text()
        assert "<script>unsafe()" not in document
        return json.loads(re.search(
            r'<script type="application/json" id="page-data">(.*?)</script>', document,
        ).group(1))

    timeline = load("retraining-timeline")
    explorer = load("retraining-explorer")
    assert [fit["id"] for fit in timeline["fits"]] == [fit["fit_id"] for fit in fits]
    assert timeline["fits"][0]["x"] == timeline["fits"][1]["x"]
    assert timeline["fits"][1]["status"] == "partial"
    assert explorer["fits"][1]["series"] == {"step": [], "epoch": []}
    assert explorer["fits"][2]["duration"] == 0
    assert "retraining-explorer.html" in manifest["parent"]
    assert "retraining-timeline.html" in manifest["parent"]

    report["training_history"] = []
    write_observation_figures(tmp_path, report)
    assert load("retraining-explorer")["fits"] == []
