"""One CPU learner per Zia task, with bounded batches and loadable model state."""

import io

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


class FraudOrReturnBooster:
    name = "xgboost"
    params = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 5,
              "max_bin": 128, "eta": 0.15, "nthread": 1, "seed": 42}
    initial_rounds = 24
    continued_rounds = 1

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

    def __init__(self, classes):
        self.classes = classes
        self.weights = None

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
