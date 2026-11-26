"""CARA, UPF and Oracle: algorithmic contracts on a synthetic drifting stream (no services)."""

import itertools
import math

import numpy as np
import pytest

from continual_bench.retraining import (
    UPF,
    CaraCT,
    CaraP,
    CaraT,
    ElasticNetCV,
    Frozen,
    Oracle,
    alpha_for_retrain_bound,
    alpha_max,
    build_matrix,
    lipschitz_bound,
    min_error_by_retrains,
    optimal_schedule,
    retrain_upper_bound,
    schedule_cost,
    simulate,
)
from continual_bench.retraining.analysis import analyze, deployed_metrics, make_rule


class ThresholdLearner:
    """1-D threshold classifier trained on the batch it is given (a window of one)."""

    def fit(self, index, x, y):
        grid = np.linspace(-1, 2, 61)
        accuracy = [np.mean((x[:, 0] > g) == y) for g in grid]
        return float(grid[int(np.argmax(accuracy))])

    def losses(self, state, x, y):
        return ((x[:, 0] > state) != y).astype(np.float64)


def drifting_stream(batches=14, rows=240, seed=3):
    rng = np.random.default_rng(seed)
    for b in range(batches):
        boundary = 0.5 + 0.09 * max(0, b - 1)  # gradual concept drift, visible offline already
        x = rng.uniform(0, 2, size=(rows, 2))
        yield x, x[:, 0] > boundary


@pytest.fixture(scope="module")
def matrix():
    built, states = build_matrix(drifting_stream(), ThresholdLearner(), offline=5, seed=1)
    assert len(states) == built.batches - 1
    return built


def test_matrix_is_triangular_and_complete(matrix):
    err = matrix.err
    assert err.shape == (14, 14)
    assert np.isnan(err[np.tril_indices(14)]).all()
    assert all(np.isfinite(err[i, i + 1 :]).all() for i in range(13))
    assert np.isnan(err[13]).all()  # the last batch trains no model
    assert matrix.initial == 4 and matrix.horizon == 9


def test_policy_geometry_fills_missing_features_from_warmup_only():
    class MissingAwareLearner(ThresholdLearner):
        trained_on_missing = False
        scored_on_missing = False

        def fit(self, index, x, y):
            self.trained_on_missing |= np.isnan(x).any()
            return super().fit(index, x, y)

        def losses(self, state, x, y):
            self.scored_on_missing |= np.isnan(x).any()
            return super().losses(state, x, y)

    batches = []
    for index, (signal, labels) in enumerate(drifting_stream(batches=7, rows=40)):
        missing = np.full((len(labels), 2), np.nan)
        missing[0, 0] = 5 if index == 0 else 5 + index
        if index > 0:
            missing[1, 1] = 40
        batches.append((np.column_stack((signal[:, 0], missing)), labels))
    learner = MissingAwareLearner()
    result, _ = build_matrix(batches, learner, offline=3, seed=1)

    assert learner.trained_on_missing and learner.scored_on_missing
    assert np.isnan(batches[0][0][1, 1:]).all()
    np.testing.assert_allclose(result.feature_means[:, 1], 5 + np.arange(7) / 40)
    np.testing.assert_allclose(result.feature_means[:, 2], [0, 1, 1, 1, 1, 1, 1])
    assert all(np.isfinite(rows).all() for rows in result.points + result.queries)
    assert math.isfinite(result.staleness(3, 0))
    assert math.isfinite(simulate(UPF(samples=10), result, alpha=0.1).cost)


def test_views_mask_the_future(matrix):
    view = matrix.view(7, {0, 1, 2, 3, 4, 6})
    assert view.err(6, 7) == matrix.err[6, 7]
    for model, batch in ((6, 8), (5, 7), (7, 8)):
        with pytest.raises(ValueError):
            view.err(model, batch)
    with pytest.raises(ValueError):
        view.shift(8)
    with pytest.raises(ValueError):
        matrix.view(5, {0, 6})
    assert all(i in view.models and j <= 7 for i, j, _ in view.observations())


def test_oracle_dp_matches_exhaustive_search(matrix):
    first, last = matrix.offline, matrix.batches - 1
    points = list(range(first, last))
    for alpha in (0.0, 0.05, 0.3, 2.0):
        best = min(
            (schedule_cost(matrix.err, matrix.initial, first, last, list(subset), alpha), subset)
            for size in range(0, 5)
            for subset in itertools.combinations(points, size)
        )
        schedule, cost = optimal_schedule(matrix.err, matrix.initial, first, last, alpha)
        assert cost == pytest.approx(schedule_cost(matrix.err, matrix.initial, first, last, schedule, alpha))
        if len(schedule) <= 4:
            assert cost == pytest.approx(best[0])


def test_alpha_max_is_the_oracle_zero_retrain_boundary(matrix):
    first, last = matrix.offline, matrix.batches - 1
    top = alpha_max(matrix.err, matrix.initial, first, last)
    assert top > 0
    assert optimal_schedule(matrix.err, matrix.initial, first, last, top * 1.001)[0] == []
    assert optimal_schedule(matrix.err, matrix.initial, first, last, top * 0.99)[0] != []
    errors = min_error_by_retrains(matrix.err, matrix.initial, first, last, 3)
    assert errors[0] == pytest.approx(sum(matrix.err[matrix.initial, first : last + 1]))
    assert errors == sorted(errors, reverse=True)


def test_oracle_lower_bounds_every_rule_and_frozen_never_retrains(matrix):
    for alpha in (0.0, 0.02, 0.1, 0.5):
        oracle = simulate(Oracle(), matrix, alpha)
        for rule in (Frozen(), CaraT(), CaraCT(), CaraP(), UPF(samples=40)):
            result = simulate(rule, matrix, alpha)
            assert result.cost >= oracle.cost - 1e-9, rule.name
        frozen = simulate(Frozen(), matrix, alpha)
        assert frozen.retrains == [] and frozen.cost == pytest.approx(
            sum(matrix.err[matrix.initial, matrix.offline :])
        )


def test_proposition_3_1_bound_holds_for_the_oracle(matrix):
    lipschitz = lipschitz_bound(matrix.err, matrix.offline)
    assert 0 < lipschitz <= 1
    for alpha in np.linspace(0, alpha_max(matrix.err, matrix.initial, matrix.offline, 13) * 1.2, 12):
        retrains = len(simulate(Oracle(), matrix, float(alpha)).retrains)
        assert retrains <= retrain_upper_bound(float(alpha), lipschitz, matrix.horizon) + 1e-9
    assert retrain_upper_bound(alpha_for_retrain_bound(1, lipschitz, 9), lipschitz, 9) == pytest.approx(1)


@pytest.mark.parametrize("make", [CaraT, CaraCT, CaraP, lambda: UPF(samples=30)])
def test_online_rules_ignore_unobserved_entries(matrix, make):
    horizon = 9  # perturb everything observed after batch 9
    twisted = type(matrix)(
        err=matrix.err.copy(),
        offline=matrix.offline,
        points=matrix.points,
        queries=matrix.queries,
        sub_loss={k: (v if k[1] <= horizon else 1 - v) for k, v in matrix.sub_loss.items()},
        feature_means=matrix.feature_means.copy(),
        fit_seconds=matrix.fit_seconds,
        rows=matrix.rows,
    )
    twisted.err[:, horizon + 1 :] = 1 - twisted.err[:, horizon + 1 :]
    twisted.feature_means[horizon + 1 :] += 5
    for alpha in (0.02, 0.2):
        real = simulate(make(), matrix, alpha)
        other = simulate(make(), twisted, alpha)
        assert [t for t in real.retrains if t <= horizon] == [t for t in other.retrains if t <= horizon]


def test_cara_variants_optimise_offline_and_react_to_drift(matrix):
    cara = simulate(CaraT(), matrix, 0.02)
    assert math.isfinite(cara.parameters["tau"]) and cara.retrains  # drift is visible in the staleness
    assert simulate(CaraT(), matrix, 5.0).retrains == []  # retraining too expensive: never retrain
    periodic = simulate(CaraP(), matrix, 0.02)
    gaps = {b - a for a, b in zip(periodic.retrains, periodic.retrains[1:])}
    assert gaps <= {periodic.parameters["period"]}
    assert simulate(CaraCT(tau_cum=0.0), matrix, 0.1).retrains == list(range(matrix.offline, 13))
    assert simulate(CaraT(tau=math.inf), matrix, 0.0).retrains == []


def test_upf_retrains_when_cheap_and_never_when_prohibitive(matrix):
    cheap = simulate(UPF(samples=60), matrix, 0.0)
    assert cheap.retrains  # a free retrain is always worth it once drift is forecast
    assert simulate(UPF(samples=60), matrix, 50.0).retrains == []
    assert (
        simulate(UPF(samples=60, seed=7), matrix, 0.05).retrains
        == simulate(UPF(samples=60, seed=7), matrix, 0.05).retrains
    )  # seeded Monte Carlo is reproducible
    assert all({"quantile_retrain", "quantile_keep"} <= set(f) for f in cheap.forecasts)


def test_upf_quantile_is_more_conservative_than_the_median_only_in_variance(matrix):
    wide = simulate(UPF(delta=0.95, samples=60), matrix, 0.05)
    median = simulate(UPF(delta=0.5, samples=60), matrix, 0.05)
    assert wide.forecasts[0]["quantile_keep"] >= median.forecasts[0]["quantile_keep"] - 1e-9


def test_upf_uses_one_future_decision_across_samples():
    rule = UPF(samples=100, delta=0.95)
    rule.alpha = 0.1
    old = np.full((100, 2), 0.05)
    old[:, 1] = 0.2
    old[:10, 1] = 0.9
    new = np.zeros((100, 2, 2))
    new[:, 0, 0] = 0.05
    new[:, 0, 1] = 0.45
    new[:, 1, 1] = 0.2
    new[9:19, 1, 1] = 0.7
    retrain, keep = rule._branch_costs(new, old)
    np.testing.assert_allclose(retrain, 0.60)
    # At the second step, keeping the original model has q95=.90 vs .80 for retraining.
    # The first keep branch therefore retrains on the second step on every sample.
    np.testing.assert_allclose(keep, old[:, 0] + rule.alpha + new[:, 1, 1])
    assert np.quantile(retrain, rule.delta) < np.quantile(keep, rule.delta)


def test_elastic_net_recovers_a_linear_signal():
    from sklearn.linear_model import ElasticNetCV as SklearnElasticNetCV

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = 0.5 + 0.8 * x[:, 0] - 0.3 * x[:, 2] + rng.normal(scale=0.01, size=200)
    model = ElasticNetCV().fit(x, y)
    reference = SklearnElasticNetCV(cv=5, max_iter=10000, tol=1e-3).fit(x, y)
    np.testing.assert_allclose(model.predict(x), reference.predict(x))
    assert model.alpha_ == pytest.approx(reference.alpha_)
    assert np.abs(model.predict(x) - y).mean() < 0.08
    assert abs(model.coef_[1]) < 0.05 and abs(model.coef_[3]) < 0.05
    assert ElasticNetCV().fit(np.ones((3, 2)), np.array([1.0, 2.0, 3.0])).predict(np.ones((1, 2)))[
        0
    ] == pytest.approx(2.0)


def test_analysis_reports_paper_quantities(matrix):
    report = analyze(matrix, n_alphas=5, samples=30)
    setup = report["setup"]
    assert (
        setup["alpha_max"] > 0
        and report["alphas"][0] == 0
        and report["alphas"][-1] == pytest.approx(setup["alpha_max"])
    )
    assert set(report["rules"]) >= {"cara", "cara_cumulative", "cara_periodic", "upf", "oracle", "frozen"}
    auc = {name: rule["auc"] for name, rule in report["rules"].items()}
    assert auc["oracle"] <= min(auc.values()) + 1e-9
    assert all(row["scpe"] == pytest.approx(0) for row in report["rules"]["oracle"]["curve"])
    grid = report["wrong_alpha"]["upf"]["pct_increase"]
    assert len(grid) == len(grid[0]) == 5
    assert all(grid[i][i] == pytest.approx(0) for i in range(5))  # correct alpha never costs extra
    assert report["forecast"]["count"] > 0 and report["forecast"]["mae"] is not None
    table = report["table"]
    assert len(table["low"]["upf"]["retrain_points"]) >= len(table["high"]["upf"]["retrain_points"])
    metrics = deployed_metrics(
        report, "upf", simulate(UPF(samples=30), matrix, 0.1), simulate(Oracle(), matrix, 0.1)
    )
    assert metrics["cost_scpe"] >= -1e-9 and metrics["performance_forecast_mae"] is not None
    assert metrics["upf_paper_auc"] == auc["upf"]
    assert make_rule("cara").name == "cara"


def test_every_paper_figure_builds_from_an_analysis(matrix):
    pytest.importorskip("matplotlib")
    from continual_bench.infrastructure import mlflow_figures
    from continual_bench.infrastructure.retraining_figures import RETRAINING_BUILDERS

    analysis = analyze(matrix, n_alphas=5, samples=30)
    analysis["deployed"] = {
        "policy": "upf",
        "alpha": 0.1,
        "retrain_points": [6, 9],
        "served_by": [4, 4, 6, 6, 6, 9, 9, 9, 9],
        "parameters": {},
        "oracle_retrain_points": [5, 8],
    }
    report = {"retraining": analysis}
    plt = mlflow_figures._pyplot()
    names = []
    for builder in RETRAINING_BUILDERS:  # not through iter_report_figures: it hides builder errors
        for path, figure in builder(plt, report):
            figure.canvas.draw()
            names.append(path.rsplit("/", 1)[-1])
            plt.close(figure)
    assert set(names) == {
        "retraining_problem.png",
        "cost_and_retrains_vs_alpha.png",
        "retrain_bounds.png",
        "wrong_alpha.png",
        "auc_table.png",
        "low_high_alpha_table.png",
        "forecast_quality.png",
        "cost_matrix.png",
        "cara_parameters.png",
        "strategy_retrains_accuracy.png",
        "cumulative_strategy_cost.png",
    }
    assert not [b for b in RETRAINING_BUILDERS if b(plt, {})]  # reports without the analysis draw nothing
    from continual_bench.evaluation.artifacts import retraining_figures

    figures = retraining_figures(analysis)
    assert {
        "cost_vs_alpha",
        "retrains_vs_alpha",
        "wrong_alpha_upf",
        "staleness_matrix",
        "accuracy_matrix",
        "retrain_bound",
        "auc_by_policy",
    } <= {f.name for f in figures}
    assert all(f.render().startswith("<svg") for f in figures)


def _stale(matrix):
    return np.nan_to_num(matrix.staleness_matrix(), nan=0.0)


def _same_batch_cost(stale, initial, first, last, retrains, alpha):
    deployed, total = initial, 0.0
    for t in range(first, last + 1):
        if t in retrains:
            total, deployed = total + alpha, t
        else:
            total += stale[t, deployed]
    return total


def test_same_batch_dp_matches_exhaustive_search(matrix):
    from continual_bench.retraining.schedule import optimal_schedule_same_batch

    stale = _stale(matrix)
    first, last = matrix.offline, matrix.batches - 1
    points = range(first, last + 1)
    for alpha in (0.0, 0.05, 0.4, 3.0):
        _, value = optimal_schedule_same_batch(stale, matrix.initial, first, last, alpha)
        best = min(
            _same_batch_cost(stale, matrix.initial, first, last, set(subset), alpha)
            for size in range(len(points) + 1)
            for subset in itertools.combinations(points, size)
        )
        assert value == pytest.approx(best)


def test_same_batch_accounting_is_consistent_between_rules_and_oracle(matrix):
    from continual_bench.retraining.schedule import alpha_max_same_batch, optimal_schedule_same_batch
    from continual_bench.retraining.simulate import simulate

    stale = _stale(matrix)
    first, last = matrix.offline, matrix.batches - 1
    top = alpha_max_same_batch(stale, matrix.initial, first, last)
    assert top > 0
    assert optimal_schedule_same_batch(stale, matrix.initial, first, last, top * 1.001)[0] == []
    assert optimal_schedule_same_batch(stale, matrix.initial, first, last, top * 0.9)[0]
    alpha = top / 4
    oracle = simulate(make_rule("oracle_staleness", same_batch=True), matrix, alpha, same_batch=True)
    assert oracle.strategy_cost(matrix) == pytest.approx(
        optimal_schedule_same_batch(stale, matrix.initial, first, last, alpha)[1]
    )
    for name in ("cara", "cara_cumulative", "cara_periodic", "markov", "frozen"):
        result = simulate(make_rule(name, same_batch=True), matrix, alpha, same_batch=True)
        assert result.strategy_cost(matrix) >= oracle.strategy_cost(matrix) - 1e-9
        assert result.strategy_cost(matrix) == pytest.approx(
            _same_batch_cost(stale, matrix.initial, first, last, set(result.retrains), alpha)
        )


def test_cara_periodic_fits_an_offset_and_fires_on_absolute_batches():
    from continual_bench.retraining.rules import CaraP

    assert [t for t in range(1, 12) if CaraP._due(t, 3, 1)] == [4, 7, 10]
    assert not CaraP._due(5, math.inf, 0)
