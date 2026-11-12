"""Shared River feature encoding: incremental StandardScaler plus one-hot nominals.

Predictions never update the scaler; only `learn` does. `transform` mirrors
River's StandardScaler formula without touching its defaultdict state, so an
unseen numeric key at prediction time cannot mutate the saved model.
"""

import numbers


def split_features(x: dict) -> tuple[dict, dict]:
    numeric = {k: float(v) for k, v in x.items() if isinstance(v, numbers.Number)}
    categorical = {f"{k}={v}": 1.0 for k, v in x.items() if not isinstance(v, numbers.Number)}
    return numeric, categorical


def read_only_scale(scaler, numeric: dict) -> dict:
    encoded = {}
    for key, value in numeric.items():
        variance = scaler.vars.get(key, 0.0)
        encoded[key] = (value - scaler.means.get(key, 0.0)) / variance**0.5 if variance else 0.0
    return encoded


class EncodedEstimator:
    """Task-generic wrapper giving any River estimator the shared encoding contract."""

    def __init__(self, estimator):
        from river import preprocessing

        self.scaler = preprocessing.StandardScaler()
        self.estimator = estimator

    def _encode(self, x: dict, *, learn: bool) -> dict:
        numeric, categorical = split_features(x)
        if learn:
            self.scaler.learn_one(numeric)
            return {**self.scaler.transform_one(numeric), **categorical}
        return {**read_only_scale(self.scaler, numeric), **categorical}

    def predict_one(self, x: dict):
        return self.estimator.predict_one(self._encode(x, learn=False))

    def predict_proba_one(self, x: dict):
        return self.estimator.predict_proba_one(self._encode(x, learn=False))

    def learn_one(self, x: dict, y) -> None:
        self.estimator.learn_one(self._encode(x, learn=True), y)


class NativeEstimator:
    """River estimator consuming raw numeric/nominal features (trees)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict_one(self, x: dict):
        return self.estimator.predict_one(x)

    def predict_proba_one(self, x: dict):
        return self.estimator.predict_proba_one(x)

    def learn_one(self, x: dict, y) -> None:
        self.estimator.learn_one(x, y)
