"""Retraining decision rules: CARA-T/-CT/-P (Mahadevan & Mathioudakis 2024), UPF (Regol et al. 2025)
and the Oracle. A rule sees a :class:`MatrixView` only, so it can never read an unobserved entry.

The operational replay takes decisions after batch ``t`` for batch ``t + 1``. In the UPF paper
replay the decision for batch ``t`` is taken with history through ``t - 1``.
``alpha`` is the retraining cost relative to the cost of one fully misclassified batch (UPF), which is
CARA's ``kappa`` expressed per query.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from .forecaster import ElasticNetCV
from .matrix import MatrixView, PerformanceMatrix
from .schedule import optimal_schedule, optimal_schedule_same_batch

MAX_CANDIDATES = 400


class Rule:
    name: ClassVar[str]
    needs_future: ClassVar[bool] = False  # Only the Oracle may read unobserved entries.

    def fit_offline(self, view: MatrixView, alpha: float) -> None:
        """Choose parameters from the offline phase (``view`` ends at the last offline batch)."""

    def fit_future(self, matrix: PerformanceMatrix, alpha: float) -> None:
        raise NotImplementedError

    def begin_online(self) -> None:
        """Reset per-run state before the first online decision."""

    def decide(self, t: int, deployed: int, view: MatrixView) -> bool:
        raise NotImplementedError

    def parameters(self) -> dict:
        return {}


class Frozen(Rule):
    name = "frozen"

    def decide(self, t, deployed, view):
        return False


def _candidates(values, extra=()):
    unique = np.unique(np.asarray(list(values) + list(extra), dtype=np.float64))
    if len(unique) > MAX_CANDIDATES:
        finite = unique[np.isfinite(unique)]
        sampled = np.quantile(finite, np.linspace(0, 1, MAX_CANDIDATES))
        unique = np.unique(np.concatenate((sampled, unique[~np.isfinite(unique)])))
    return [float(v) for v in unique]


class _CaraRule(Rule):
    """Shared offline optimisation: CARA replays Algorithm 1 over the offline phase and keeps the best."""

    def __init__(self, same_batch: bool = False):
        self.kappa = 0.0
        self._cumulative = 0.0
        self.same_batch = same_batch  # CARA's accounting: a retrain replaces the staleness of its batch

    def _offline_matrix(self, view: MatrixView) -> dict[tuple[int, int], float]:
        return {(t, m): view.staleness(t, m) for t in range(1, view.upto + 1) for m in range(t)}

    def _replay(self, psi, last, trigger) -> float:
        """Offline strategy cost for ``trigger(t, stale, state) -> bool`` (``t`` is the batch index)."""
        deployed, total, state = 0, 0.0, {"cumulative": 0.0}
        for t in range(1, last + 1):
            stale = psi[t, deployed]
            if self.same_batch:
                if trigger(t, stale, state):
                    total += self.kappa
                    deployed, state["cumulative"] = t, 0.0
                else:
                    total += stale
            else:
                total += stale
                if t <= last - 1 and trigger(t, stale, state):
                    total += self.kappa
                    deployed, state["cumulative"] = t, 0.0
        return total

    def fit_offline(self, view, alpha):
        self.kappa = alpha
        psi = self._offline_matrix(view)
        self._optimise(psi, view.upto)

    def begin_online(self):
        self._cumulative = 0.0

    def _optimise(self, psi, last):
        raise NotImplementedError


class CaraT(_CaraRule):
    name = "cara"  # the "standard" CARA of the UPF paper: threshold on the model staleness
    label = "CARA-T"

    def __init__(self, tau: float | None = None, same_batch: bool = False):
        super().__init__(same_batch)
        self.tau, self._fixed = tau, tau is not None

    def _optimise(self, psi, last):
        if self._fixed:
            return
        best = None
        for tau in _candidates(psi.values(), [math.inf]):
            cost = self._replay(psi, last, lambda t, stale, state, tau=tau: stale >= tau)
            if best is None or cost < best[0] - 1e-12 or (abs(cost - best[0]) <= 1e-12 and tau > best[1]):
                best = (cost, tau)
        self.tau = best[1]

    def decide(self, t, deployed, view):
        return view.staleness(t, deployed) >= self.tau

    def parameters(self):
        return {"tau": self.tau, "kappa": self.kappa}


class Markov(Rule):
    """CARA's uncalibrated baseline: retrain when the relative staleness reaches kappa (alpha per query)."""

    name = "markov"

    def __init__(self):
        self.tau = 0.0

    def fit_offline(self, view, alpha):
        self.tau = alpha

    def decide(self, t, deployed, view):
        return view.staleness(t, deployed) >= self.tau

    def parameters(self):
        return {"tau": self.tau}


class CaraCT(_CaraRule):
    name = "cara_cumulative"
    label = "CARA-CT"

    def __init__(self, tau_cum: float | None = None, same_batch: bool = False):
        super().__init__(same_batch)
        self.tau_cum, self._fixed = tau_cum, tau_cum is not None

    @staticmethod
    def _trigger(tau):
        def trigger(t, stale, state):
            state["cumulative"] += stale
            return state["cumulative"] >= tau

        return trigger

    def _optimise(self, psi, last):
        if self._fixed:
            return
        total = sum(max(psi[t, 0], 0.0) for t in range(1, last + 1))
        best = None
        for tau in [*np.linspace(0.0, total, 200).tolist(), math.inf]:
            cost = self._replay(psi, last, self._trigger(tau))
            if best is None or cost < best[0] - 1e-12 or (abs(cost - best[0]) <= 1e-12 and tau > best[1]):
                best = (cost, tau)
        self.tau_cum = best[1]

    def decide(self, t, deployed, view):
        self._cumulative += view.staleness(t, deployed)
        if self._cumulative >= self.tau_cum:
            self._cumulative = 0.0
            return True
        return False

    def parameters(self):
        return {"tau_cum": self.tau_cum, "kappa": self.kappa}


class CaraP(_CaraRule):
    """Retrain at every batch ``t`` with ``(t - offset) % period == 0`` (Eq. 10); both are fitted offline."""

    name = "cara_periodic"
    label = "CARA-P"

    def __init__(self, period: float | None = None, offset: int = 0, same_batch: bool = False):
        super().__init__(same_batch)
        self.period, self.offset, self._fixed = period, offset, period is not None

    @staticmethod
    def _due(t, period, offset):
        return math.isfinite(period) and t > offset and (t - offset) % period == 0

    def _optimise(self, psi, last):
        if self._fixed:
            return
        best = None
        candidates = [(p, a) for p in range(1, max(2, last)) for a in range(p)] + [(math.inf, 0)]
        for period, offset in candidates:
            cost = self._replay(psi, last, lambda t, stale, state, p=period, a=offset: self._due(t, p, a))
            if best is None or cost < best[0] - 1e-12 or (abs(cost - best[0]) <= 1e-12 and period > best[1]):
                best = (cost, period, offset)
        self.period, self.offset = best[1], best[2]

    def decide(self, t, deployed, view):
        return self._due(t, self.period, self.offset)

    def parameters(self):
        return {"period": self.period, "offset": self.offset, "kappa": self.kappa}


class Oracle(Rule):
    """Retrospectively optimal schedule: needs every future entry, so it is a lower bound, not a policy."""

    name = "oracle"
    needs_future = True

    def __init__(self, objective: str = "error", same_batch: bool = False):
        if objective not in ("error", "staleness"):
            raise ValueError("oracle objective must be 'error' or 'staleness'")
        if same_batch and objective != "staleness":
            raise ValueError("same-batch accounting is defined for the staleness objective only")
        self.objective, self.schedule, self.cost, self.same_batch = objective, [], 0.0, same_batch

    def fit_future(self, matrix, alpha):
        if self.same_batch or (matrix.paper_timing and self.objective == "staleness"):
            self.schedule, self.cost = optimal_schedule_same_batch(
                np.nan_to_num(matrix.staleness_matrix(), nan=0.0),
                matrix.initial,
                matrix.offline,
                matrix.batches - 1,
                alpha,
            )
            return
        if self.objective == "error" and matrix.paper_timing:
            self.schedule, self.cost = optimal_schedule_same_batch(
                matrix.err.T,
                matrix.initial,
                matrix.offline,
                matrix.batches - 1,
                alpha,
                retrain_errors=matrix.err.diagonal(),
            )
            return
        if self.objective == "error":
            cost = matrix.err
        else:
            cost = matrix.staleness_matrix().T  # cost[model, batch]
        self.schedule, self.cost = optimal_schedule(
            np.nan_to_num(cost, nan=0.0), matrix.initial, matrix.offline, matrix.batches - 1, alpha
        )

    def decide(self, t, deployed, view):
        return t in self.schedule

    def parameters(self):
        return {"objective": self.objective, "schedule": list(self.schedule)}


class UPF(Rule):
    """Uncertainty-Performance Forecaster: Monte-Carlo quantile test between retraining now and keeping.

    Features ``[i, j, j - i, z_shift]`` feed a linear ElasticNetCV mean; the entries are modelled as
    Beta with a constant residual variance. For each future state, the comparison of cost quantiles
    chooses one continuation shared by all Monte Carlo samples (paper equations 16–22).
    """

    name = "upf"

    def __init__(self, delta: float = 0.95, samples: int = 100, seed: int = 0):
        if not 0 < delta < 1 or samples < 2:
            raise ValueError("invalid UPF quantile or sample count")
        self.delta, self.samples, self.seed = delta, samples, seed
        self.alpha = 0.0
        self.horizon = 0
        self.paper_timing = False
        self.forecasts: list[dict] = []
        self.last_fit: dict = {}

    def fit_offline(self, view, alpha):
        self.alpha = alpha

    def begin_online(self):
        self.forecasts = []

    def bind(self, matrix: PerformanceMatrix):
        self.horizon = matrix.batches
        self.paper_timing = matrix.paper_timing

    @staticmethod
    def _features(i, j, shift):
        return [float(i), float(j), float(j - i), shift]

    def _regressor(self, view: MatrixView):
        rows = view.observations()
        x = [self._features(i, j, view.shift(j)) for i, j, _ in rows]
        model = ElasticNetCV().fit(np.asarray(x), np.asarray([e for _, _, e in rows]))
        return model, max(model.residual_std_, 1e-3)

    def _beta(self, mean, std):
        mean = np.clip(mean, 1e-3, 1 - 1e-3)
        variance = np.minimum(std**2, 0.9 * mean * (1 - mean))
        strength = mean * (1 - mean) / variance - 1
        return mean * strength, (1 - mean) * strength

    def decide(self, t, deployed, view):
        size = self.horizon
        first = t if self.paper_timing else t + 1
        future = size - first
        if future < 1:
            return False
        model, std = self._regressor(view)
        shift = view.shift(view.upto)
        # u indexes the candidate model (0 is "retrain now"), k the served batch.
        u_index, k_index = np.triu_indices(future)
        trained, served = t + u_index, first + k_index
        features = np.column_stack([trained, served, served - trained, np.full(len(trained), shift)])
        mean_new = model.predict(features)
        old_features = np.column_stack(
            [
                np.full(future, deployed),
                first + np.arange(future),
                first + np.arange(future) - deployed,
                np.full(future, shift),
            ]
        )
        mean_old = model.predict(old_features)
        rng = np.random.default_rng([self.seed, view.upto])
        a, b = self._beta(mean_new, std)
        new = np.zeros((self.samples, future, future))
        new[:, u_index, k_index] = rng.beta(a, b, size=(self.samples, len(a)))
        a, b = self._beta(mean_old, std)
        old = rng.beta(a, b, size=(self.samples, future))
        retrain_cost, keep_cost = self._branch_costs(new, old)
        self.forecasts.append(
            {
                "t": t,
                "predicted_new": mean_new.tolist(),
                "predicted_old": mean_old.tolist(),
                "trained": trained.tolist(),
                "served": served.tolist(),
                "deployed": deployed,
                "quantile_retrain": float(np.quantile(retrain_cost, self.delta)),
                "quantile_keep": float(np.quantile(keep_cost, self.delta)),
            }
        )
        return self.forecasts[-1]["quantile_retrain"] < self.forecasts[-1]["quantile_keep"]

    def _branch_costs(self, new: np.ndarray, old: np.ndarray):
        samples, future, _ = new.shape
        # Column 0 keeps the current model. Column u+1 keeps candidate u.
        continuation = np.zeros((samples, future + 1))
        for k in range(future - 1, -1, -1):
            retrain = self.alpha + new[:, k, k] + continuation[:, k + 1]
            errors = np.column_stack((old[:, k], new[:, :k, k]))
            keep = errors + continuation[:, : k + 1]
            choose_retrain = np.quantile(retrain, self.delta) < np.quantile(keep, self.delta, axis=0)
            continuation[:, : k + 1] = np.where(choose_retrain, retrain[:, None], keep)
        return retrain, keep[:, 0]

    def parameters(self):
        return {"delta": self.delta, "samples": self.samples, "alpha": self.alpha}


class StalenessOracle(Oracle):
    """CARA's Oracle (Algorithms 2-3): optimal on the staleness cost matrix instead of the measured error."""

    name = "oracle_staleness"

    def __init__(self, same_batch: bool = False):
        super().__init__("staleness", same_batch)


RULES = {cls.name: cls for cls in (Frozen, CaraT, CaraCT, CaraP, Markov, Oracle, StalenessOracle, UPF)}
