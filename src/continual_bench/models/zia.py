"""One CPU learner per Zia task, with bounded batches and loadable model state."""

import io

import numpy as np
from river import optim
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax


class FraudOrReturnBooster:
    name = "xgboost"
    params = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 5,
              "max_bin": 128, "eta": 0.15, "nthread": 1, "seed": 42}
    initial_rounds = 24
    continued_rounds = 1
    online_params = {"optimizer": "xgboost_refresh", "gradient": "river_log_loss",
                     "minibatch_rows": 8192}

    def __init__(self):
        import xgboost

        self.xgb = xgboost
        self.model = None

    def fit(self, x, y, continuation):
        matrix = self.xgb.QuantileDMatrix(x, label=y, max_bin=self.params["max_bin"], nthread=1)
        rounds = self.continued_rounds if continuation else self.initial_rounds
        self.model = self.xgb.train(
            self.params, matrix, num_boost_round=rounds,
            xgb_model=self.model if continuation else None,
        )
        return self.model.num_boosted_rounds()

    def fit_online(self, x, y):
        """Update the leaves of the existing booster with River's logistic gradient."""
        loss = optim.losses.Log()

        def objective(margins, matrix):
            gradient = loss.gradient(matrix.get_label(), margins.clip(-80, 80))
            p = expit(margins)
            return gradient, np.maximum(p * (1 - p), 1e-16)

        params = {**self.params, "process_type": "update", "updater": "refresh",
                  "refresh_leaf": True, "verbosity": 0}
        rounds = self.model.num_boosted_rounds()
        size = self.online_params["minibatch_rows"]
        steps = 0
        for start in range(0, len(y), size):
            stop = start + size
            matrix = self.xgb.DMatrix(x[start:stop], label=y[start:stop], nthread=1)
            self.model = self.xgb.train(
                params, matrix, num_boost_round=rounds, xgb_model=self.model, obj=objective,
            )
            steps += rounds
        return steps

    def predict(self, x):
        positive = self.model.inplace_predict(x)
        return np.column_stack((1 - positive, positive))

    def payload(self):
        return bytes(self.model.save_raw(raw_format="ubj")), "ubj"

    def load(self, payload):
        self.model = self.xgb.Booster(params={"nthread": 1})
        self.model.load_model(bytearray(payload))

    def explanation(self, features):
        kinds = ("gain", "weight", "cover", "total_gain", "total_cover")
        return {kind: {features[int(key[1:])]: value for key, value in self.model.get_score(
            importance_type=kind).items()} for kind in kinds}


class MultinomialLogistic:
    name = "multinomial_logistic"
    params = {"solver": "L-BFGS-B", "l2": 0.001}
    initial_rounds = 24
    continued_rounds = 5
    online_params = {"optimizer": "river_sgd", "learning_rate": 0.05, "minibatch_rows": 1024}

    def __init__(self, classes):
        self.classes = classes
        self.weights = None
        self._online_optimizer = optim.SGD(self.online_params["learning_rate"])

    def fit(self, x, y, continuation):
        data = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
        shape = data.shape[1], self.classes
        regularization = self.params["l2"]
        index = np.arange(len(y))

        def objective(flat):
            weights = flat.reshape(shape)
            logits = data @ weights
            logits -= logsumexp(logits, axis=1, keepdims=True)
            loss = -logits[index, y].mean() + regularization / 2 * np.square(weights[:-1]).sum()
            residuals = np.exp(logits)
            residuals[index, y] -= 1
            gradient = data.T @ residuals / len(y)
            gradient[:-1] += regularization * weights[:-1]
            return loss, gradient.ravel()

        start = self.weights.ravel() if continuation else np.zeros(np.prod(shape))
        result = minimize(objective, start, jac=True, method=self.params["solver"],
                          options={"maxiter": self.continued_rounds if continuation else self.initial_rounds})
        if not np.isfinite(result.fun) or (not result.success and result.status != 1):
            raise ValueError(f"logistic optimization failed: {result.message}")
        self.weights = result.x.reshape(shape)
        return result.nit

    def fit_online(self, x, y):
        """Apply River SGD to the same softmax weights and intercept fitted at warm-up."""
        size = self.online_params["minibatch_rows"]
        for start in range(0, len(y), size):
            labels = y[start:start + size]
            features = x[start:start + size]
            residuals = self.predict(features)
            residuals[np.arange(len(labels)), labels] -= 1
            gradient = np.empty_like(self.weights)
            gradient[:-1] = features.T @ residuals / len(labels)
            gradient[:-1] += self.params["l2"] * self.weights[:-1]
            gradient[-1] = residuals.mean(axis=0)
            self.weights = self._online_optimizer.step(self.weights, gradient)
        return (len(y) + size - 1) // size

    def predict(self, x):
        return softmax(x @ self.weights[:-1] + self.weights[-1], axis=1)

    def payload(self):
        buffer = io.BytesIO()
        np.save(buffer, self.weights, allow_pickle=False)
        return buffer.getvalue(), "npy"

    def load(self, payload):
        self.weights = np.load(io.BytesIO(payload), allow_pickle=False)

    def explanation(self, features):
        return {"coefficients": {name: row.tolist() for name, row in zip(
            [*features, "intercept"], self.weights, strict=True)}}


def make_zia_model(name, classes):
    if name == "xgboost":
        return FraudOrReturnBooster()
    if name == "multinomial_logistic":
        return MultinomialLogistic(classes)
    raise ValueError(f"unknown Zia model: {name}")


class ZiaPredictor:
    """Load a native checkpoint and score prepared numeric features in training order."""

    def __init__(self, directory):
        import hashlib
        import json
        from pathlib import Path

        directory = Path(directory)
        metadata = json.loads((directory / "model-metadata.json").read_text())
        payload = (directory / metadata["payload"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["state_sha256"]:
            raise ValueError("Zia model checkpoint checksum mismatch")
        self.features = metadata["features"]
        self.model = make_zia_model(metadata["model"], len(metadata["classes"]))
        self.model.load(payload)

    def predict(self, data, params=None):
        if not data.columns.is_unique or set(data.columns) != set(self.features):
            raise ValueError("inference feature schema differs from the Zia model")
        return self.model.predict(data[self.features].to_numpy(dtype=np.float32))


def _load_pyfunc(path):
    return ZiaPredictor(path)
