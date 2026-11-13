"""Dataset generators and the paper-timing matrix of the UPF/CARA replication."""

import numpy as np
import pytest

from continual_bench.datasets import retraining_benchmarks as data
from continual_bench.evaluation.retraining_replication import ALPHA_MAX, PAPER_AUC, XGBoostLearner
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


def test_paper_timing_trains_online_points_on_the_next_batch():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    batches = [data.split_batch(x, y, rng) for x, y in data.upf_circles(9, size=300, seed=4)]
    matrix, states = build_paper_matrix(batches, XGBoostLearner(0), 4, seed=0)
    assert list(matrix.source) == [0, 1, 2, 3, 5, 6, 7, 8, 9]  # point p >= offline trains on p + 1
    assert len(states) == 8 and np.isfinite(matrix.err[4, 5]) and np.isnan(matrix.err[4, 4])
    assert matrix.horizon == 5 and matrix.initial == 3
    oracle = simulate(Oracle(), matrix, 0.0)
    assert oracle.cost <= simulate(UPF(samples=20), matrix, 0.0).cost + 1e-9
    assert np.isfinite(matrix.staleness(6, 4))


def test_paper_reference_values_cover_every_replicated_dataset():
    assert set(ALPHA_MAX) == set(PAPER_AUC) == {"gauss", "circles", "electricity", "airlines"}
    assert all(row["oracle"] < row["upf"] < row["cara_periodic"] for row in PAPER_AUC.values())


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
    showcase = {"dataset": "toy", "rule": "cara", "seeds": 2, "batches": size, "offline": 2, "alpha": 0.1,
                "accuracy_mean": mean, "accuracy_std": [[0.01] * size for _ in range(size)],
                "schedule_mean": [0.8, 0.85, 0.8, 0.82], "schedule_std": [0.0] * 4, "retrain_batches": [4]}
    name, figure = schedule_figure(showcase)
    assert name == "cara_schedule_toy.png"
    axis = figure.axes[0]
    assert len(axis.lines) == size  # size - 1 models + the schedule
    assert any(text.get_text() == "retrain" for text in axis.texts)
