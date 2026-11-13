"""ElasticNetCV in numpy: the UPF performance forecaster mean, without a scikit-learn dependency.

Objective (scikit-learn's convention): ``1/(2n) ||y - Xw - b||^2 + a*l1*||w||_1 + a*(1-l1)/2 ||w||^2``
with coordinate descent along a decreasing ``a`` path and K-fold cross-validated ``a``.
"""

from __future__ import annotations

import numpy as np


def _soft(value, threshold):
    return np.sign(value) * max(abs(value) - threshold, 0.0)


def _descend(x, y, alpha, l1_ratio, weights, iterations=500, tolerance=1e-7):
    n = len(y)
    residual = y - x @ weights
    penalty_l1, penalty_l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    for _ in range(iterations):
        largest = 0.0
        for j in range(x.shape[1]):
            old = weights[j]
            rho = (x[:, j] @ residual) / n + old  # columns are standardized: (1/n) x_j.x_j == 1
            weights[j] = _soft(rho, penalty_l1) / (1 + penalty_l2)
            if weights[j] != old:
                residual -= x[:, j] * (weights[j] - old)
                largest = max(largest, abs(weights[j] - old))
        if largest < tolerance:
            break
    return weights


class ElasticNetCV:
    def __init__(self, l1_ratio: float = 0.5, n_alphas: int = 30, eps: float = 1e-3, folds: int = 5):
        if not 0 < l1_ratio <= 1:
            raise ValueError("l1_ratio must be in (0, 1]")
        self.l1_ratio, self.n_alphas, self.eps, self.folds = l1_ratio, n_alphas, eps, folds
        self.alpha_ = 0.0
        self.coef_: np.ndarray | None = None
        self.intercept_ = 0.0
        self.residual_std_ = 0.0

    def _standardize(self, x):
        return (x - self._mean) / self._scale

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("inconsistent regression dataset")
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self._scale = np.where(scale > 1e-12, scale, 1.0)
        z = self._standardize(x)
        self.intercept_ = float(y.mean())
        centered = y - self.intercept_
        n = len(y)
        if n < 4 or np.allclose(centered, 0):
            self.coef_ = np.zeros(x.shape[1])
        else:
            top = max(float(np.abs(z.T @ centered).max()) / (n * self.l1_ratio), 1e-12)
            alphas = top * np.geomspace(1.0, self.eps, self.n_alphas)
            folds = min(self.folds, n)
            assignment = np.arange(n) % folds
            scores = np.zeros(len(alphas))
            for fold in range(folds):
                train, test = assignment != fold, assignment == fold
                weights = np.zeros(x.shape[1])
                for index, alpha in enumerate(alphas):
                    weights = _descend(z[train], centered[train], alpha, self.l1_ratio, weights)
                    scores[index] += float(np.mean((centered[test] - z[test] @ weights) ** 2))
            self.alpha_ = float(alphas[int(np.argmin(scores))])
            self.coef_ = _descend(z, centered, self.alpha_, self.l1_ratio, np.zeros(x.shape[1]), 2000)
        self.residual_std_ = float(np.sqrt(np.mean((centered - z @ self.coef_) ** 2)))
        return self

    def predict(self, x):
        z = self._standardize(np.asarray(x, dtype=np.float64))
        return self.intercept_ + z @ self.coef_
