"""ElasticNetCV mean and residual scale for the UPF Beta forecast."""

from __future__ import annotations

import numpy as np


class ElasticNetCV:
    def fit(self, x, y):
        from sklearn.linear_model import ElasticNetCV as SklearnElasticNetCV

        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
            raise ValueError("inconsistent regression dataset")
        self.alpha_ = 0.0
        self.coef_ = np.zeros(x.shape[1])
        self.intercept_ = float(y.mean())
        # A tiny operational history may have fewer observations than the default five folds.
        if len(y) >= 2 and np.any(y != y[0]) and np.any(np.ptp(x, axis=0) > 0):
            model = SklearnElasticNetCV(cv=min(5, len(y)), max_iter=10000, tol=1e-3).fit(x, y)
            self.alpha_, self.coef_, self.intercept_ = model.alpha_, model.coef_, model.intercept_
        self.residual_std_ = float(np.sqrt(np.mean((y - self.predict(x)) ** 2)))
        return self

    def predict(self, x):
        return self.intercept_ + np.asarray(x, dtype=np.float64) @ self.coef_
