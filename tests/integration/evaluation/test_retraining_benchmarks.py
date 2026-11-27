"""Dataset generators and the paper-timing matrix of the UPF/CARA replication."""

from itertools import product

import numpy as np
import pytest

from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.evaluation.retraining_replication import (
    ALPHA_MAX,
    COMPARISON_GRID,
    PAPER_AUC,
    XGBoostLearner,
    replicate,
)
from continual_bench.retraining import build_paper_matrix, simulate
from continual_bench.retraining.rules import UPF, Oracle


def test_generators_follow_the_printed_formulas():
    gauss = data.upf_gauss(3, size=4000, seed=1)
    for t, (x, y) in enumerate(gauss):
        assert x.shape == (4000, 2)
        np.testing.assert_allclose(x.mean(axis=0), [(t + 1) / 100, 0.5 - (t + 1) / 100], atol=0.01)
        np.testing.assert_array_equal(y, ((4 * x[:, 0] - 0.5) ** 2 > x[:, 1]).astype(int))
    circles = data.upf_circles(2, size=2000, seed=1, epsilon=0.5)
    x, y = circles[1]
    assert ((x >= 0) & (x <= 1)).all()
    np.testing.assert_array_equal(y, (((x - 0.22) ** 2).sum(axis=1) <= 0.25).astype(int))
    for make in (data.cara_gauss, data.cara_circle, data.cara_covcon):
        batches = make(12, size=200, seed=2)
        assert len(batches) == 12 and all(len(y) == 200 and set(np.unique(y)) <= {0, 1} for _, y in batches)
    flips = data.cara_covcon(20, size=500, seed=3)
    assert flips[0][1].mean() != pytest.approx(flips[10][1].mean(), abs=1e-9)


def test_electricity_loader_reads_chronological_batches(tmp_path):
    label = ["DOWN", "UP", "UP"]
    rows = "\n".join(f"0,1,0.{i % 10},0.{i % 7},0.1,0.2,0.3,0.4,{label[i % 3]}" for i in range(95))
    path = tmp_path / "e.arff"
    path.write_text("@relation x\n@attribute a numeric\n@data\n" + rows + "\n")
    batches = data.electricity_batches(path, size=30)
    assert len(batches) == 3 and batches[0][0].shape == (30, 6)
    assert batches[0][1][0] == 0 and batches[0][1][1] == 1


def test_paper_timing_trains_own_batch_and_exposes_only_observed_errors():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    batches = [data.split_batch(x, y, rng) for x, y in data.upf_circles(9, size=300, seed=4)]
    matrix, states = build_paper_matrix(batches, XGBoostLearner(0), 4, seed=0)
    assert matrix.paper_timing
    assert len(states) == 9 and np.isfinite(matrix.err[4, 4]) and np.isnan(matrix.err[3, 3])
    assert matrix.err[4, 4] == XGBoostLearner(0).losses(states[4], *batches[4][2:]).mean()
    assert len(matrix.offline_view().observations()) == 6
    with pytest.raises(ValueError, match="not observable"):
        matrix.offline_view().err(4, 4)
    assert matrix.view(4, range(5)).err(4, 4) == matrix.err[4, 4]
    with pytest.raises(ValueError, match="not observable"):
        matrix.view(4, range(4)).err(4, 4)
    assert matrix.horizon == 5 and matrix.initial == 3
    oracle = simulate(Oracle(), matrix, 0.0)
    result = simulate(UPF(samples=20), matrix, 0.0)
    assert oracle.cost <= result.cost + 1e-9
    assert len(result.forecasts) == 5
    assert result.forecasts[0]["trained"][0] == result.forecasts[0]["served"][0] == 4
    assert result.forecasts[-1]["trained"] == result.forecasts[-1]["served"] == [8]
    assert np.isfinite(matrix.staleness(6, 4))

    online = range(matrix.offline, matrix.batches)
    for alpha in (0.0, 0.02, 0.5):
        costs = []
        for switches in product((False, True), repeat=matrix.horizon):
            deployed, cost = matrix.initial, alpha * sum(switches)
            for j, retrain in zip(online, switches, strict=True):
                if retrain:
                    deployed = j
                cost += matrix.err[deployed, j]
            costs.append(cost)
        assert simulate(Oracle(), matrix, alpha).cost == pytest.approx(min(costs))


def test_airlines_csv_preserves_reference_category_codes(tmp_path):
    csv = tmp_path / "airlines.csv"
    csv.write_text(
        "Airline,Flight,AirportFrom,AirportTo,DayOfWeek,Time,Length,Delay\n"
        "4,269,253,135,3,15,205,1\n14,35,0,90,6,10,30,0\n"
    )
    x, y = data.airlines_arrays(str(csv))
    np.testing.assert_array_equal(x, [[4, 269, 253, 135, 3, 15, 205], [14, 35, 0, 90, 6, 10, 30]])
    np.testing.assert_array_equal(y, [1, 0])


def test_paper_reference_values_cover_every_replicated_dataset():
    assert set(ALPHA_MAX) == set(PAPER_AUC) == {"gauss", "circles", "electricity", "airlines"}
    assert all(row["oracle"] < row["upf"] < row["cara_periodic"] for row in PAPER_AUC.values())


def test_replication_exports_exact_off_grid_operating_points():
    pytest.importorskip("xgboost")
    report = replicate("gauss", trials=1, size=64, samples=10, rules=("upf",))
    for level, alpha in (("low", 0.05), ("high", 0.45)):
        assert all(point != pytest.approx(alpha) for point in report["alphas"])
        for rule in ("upf", "oracle"):
            row = report["table"][level][rule]
            assert row["alpha"] == pytest.approx(alpha)
            accuracy = row["average_accuracy"]["mean"]
            retrains = row["retrains"]["mean"]
            assert row["cost"]["mean"] == pytest.approx(alpha * retrains + 8 * (1 - accuracy))
            assert row["cost"]["std"] == 0


def test_electricity_article_figures_include_table_2():
    from continual_bench.evaluation.replication_plots import upf_figures

    metrics = {metric: {"mean": 0.2, "std": 0.01} for metric in ("cost", "retrains", "average_accuracy")}
    report = {
        "dataset": "electricity",
        "trials": 2,
        "summary": {rule: {"auc_mean": 2.0, "auc_std": 0.1} for rule in ("upf", "oracle")},
        "table": {
            level: {rule: {"alpha": alpha, **metrics} for rule in ("upf", "oracle")}
            for level, alpha in (("low", 0.1), ("high", 0.9))
        },
    }
    figures = dict(upf_figures([report]))
    table = figures["upf_table2_electricity.png"]
    assert len(table.axes) == 6
    assert [text.get_text() for text in table.legends[0].get_texts()] == ["Resultado local", "Artigo"]
    assert table.axes[0].lines[3].get_ydata().astype(float) == pytest.approx([2.5, 5.6])


def test_shared_staleness_uses_unlabeled_test_queries():
    class Learner:
        def fit(self, index, x, y):
            return float(x.mean())

        def losses(self, cutoff, x, y):
            return (x[:, 0] > cutoff) != y

    rng = np.random.default_rng(4)
    batches = [data.split_batch(x, y, rng) for x, y in data.upf_gauss(6, size=64, seed=4)]
    flipped = [(x, y, xt, 1 - yt) for x, y, xt, yt in batches]
    original, _ = build_paper_matrix(batches, Learner(), 3, separate_query_split=True)
    modified, _ = build_paper_matrix(flipped, Learner(), 3, separate_query_split=True)
    np.testing.assert_allclose(original.staleness_matrix(), modified.staleness_matrix())
    np.testing.assert_allclose(original.err, 1 - modified.err)


@pytest.mark.parametrize(("timing", "decisions"), [("same-batch", 8), ("next-batch", 7)])
@pytest.mark.parametrize("dataset", ("electricity", "airlines"))
def test_shared_benchmark_scores_both_objectives_with_their_oracles(tmp_path, timing, decisions, dataset):
    pytest.importorskip("xgboost")
    rows = []
    if dataset == "electricity":
        path = tmp_path / "electricity.arff"
        header = "@data"
    else:
        path = tmp_path / "airlines.csv"
        header = "Airline,Flight,AirportFrom,AirportTo,DayOfWeek,Time,Length,Delay"
    for batch, i in product(range(15), range(48)):
        positive = i % 11 < 3 + (batch % 4)
        if dataset == "electricity":
            label = "UP" if positive else "DOWN"
            rows.append(f"0,1,{i / 48},{i % 11 / 11},{batch / 15},0.2,0.3,0.4,{label}")
        else:
            rows.append(f"{i % 3},{i},{i % 11},{batch},{batch % 7},{i * 5},205,{int(positive)}")
    path.write_text(header + "\n" + "\n".join(rows))
    report = replicate(
        dataset,
        trials=1,
        electricity_path=path,
        airlines_path=path,
        size=48,
        samples=10,
        rules=("cara", "upf"),
        compare_oracles=True,
        decision_timing=timing,
    )
    assert report["alphas"] == list(COMPARISON_GRID)
    assert report["alpha_max"] == COMPARISON_GRID[-1]
    assert report["comparison"]["batch_size"] == 48
    assert report["comparison"]["decision_points"] == decisions
    assert report["table"]["high"]["upf"]["alpha"] == pytest.approx(0.9)
    assert report["summary"]["oracle"]["paper_auc"] is None
    curves = report["curves"]
    strategy_reference = curves["oracle_staleness"]["strategy_cost"]
    for rule in curves.values():
        np.testing.assert_allclose(
            rule["error_oracle_gap"], np.asarray(rule["cost"]) - curves["oracle"]["cost"], atol=1e-10
        )
        np.testing.assert_allclose(
            rule["strategy_oracle_gap"], np.asarray(rule["strategy_cost"]) - strategy_reference, atol=1e-10
        )
        assert min(rule["error_oracle_gap"]) >= -1e-10
        assert min(rule["strategy_oracle_gap"]) >= -1e-10
        assert max(rule["retrains"]) <= decisions


def test_fast_forest_prediction_matches_scikit_learn():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    from continual_bench.evaluation.cara_replication import ForestLearner

    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] + x[:, 1] ** 2 > 0.7).astype(int)
    forest = RandomForestClassifier(random_state=0, n_jobs=1).fit(x[:300], y[:300])
    np.testing.assert_array_equal(ForestLearner.predict(forest, x[300:]), forest.predict(x[300:]))
    single = RandomForestClassifier(random_state=0, n_jobs=1).fit(x[:50], np.zeros(50, dtype=int))
    np.testing.assert_array_equal(ForestLearner.predict(single, x), np.zeros(400))


def test_cara_static_queries_use_the_batch_concept():
    queries = data.cara_static_queries("covcon", n_batches=3, n_queries=50, seed=0)
    x, y = queries[0]
    assert x.shape == (50, 2) and abs(x.mean() - 0.5) < 0.02 and set(np.unique(y)) <= {0, 1}


def test_schedule_figure_draws_models_and_retrain_marks():
    from continual_bench.evaluation.replication_plots import schedule_figure

    size = 6
    mean = [[0.9 - 0.02 * abs(i - j) for j in range(size)] for i in range(size)]
    showcase = {
        "dataset": "toy",
        "rule": "cara",
        "seeds": 2,
        "batches": size,
        "offline": 2,
        "alpha": 0.1,
        "accuracy_mean": mean,
        "accuracy_std": [[0.01] * size for _ in range(size)],
        "schedule_mean": [0.8, 0.85, 0.8, 0.82],
        "schedule_std": [0.0] * 4,
        "retrain_batches": [4],
    }
    name, figure = schedule_figure(showcase)
    assert name == "cara_schedule_toy.png"
    axis = figure.axes[0]
    assert len(axis.lines) == size  # size - 1 models + the schedule
    assert any(text.get_text() == "Retreino" for text in axis.texts)
