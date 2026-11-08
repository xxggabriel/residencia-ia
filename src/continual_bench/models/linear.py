"""Linear River learners: logistic (binary/multiclass) and SGD linear regression."""

from river import linear_model, multiclass, optim, preprocessing

from .encoding import EncodedEstimator, split_features


class RiverModel:
    """Binary logistic regression. Predict is read-only; the scaler learns inside learn()."""

    def __init__(self, learning_rate: float = 0.05):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0,1]")
        self.scaler = preprocessing.StandardScaler()
        self.classifier = linear_model.LogisticRegression(optimizer=optim.SGD(learning_rate))

    def encode(self, x: dict, *, learn: bool) -> dict:
        numeric, categorical = split_features(x)
        if learn:
            self.scaler.learn_one(numeric)
        return {**self.scaler.transform_one(numeric), **categorical}

    def predict(self, x: dict) -> float:
        return float(self.classifier.predict_proba_one(self.encode(x, learn=False)).get(True, 0.5))

    def learn(self, x: dict, y: bool) -> None:
        self.classifier.learn_one(self.encode(x, learn=True), bool(y))


def logistic_multiclass(learning_rate: float = 0.05) -> EncodedEstimator:
    """One-vs-rest logistic regression over the shared scaler/one-hot encoding."""
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0,1]")
    return EncodedEstimator(multiclass.OneVsRestClassifier(
        linear_model.LogisticRegression(optimizer=optim.SGD(learning_rate))))


class LinearRegressor:
    """Standardized SGD linear regression; numeric features only."""

    def __init__(self, learning_rate: float = 0.01):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0,1]")
        self.scaler = preprocessing.StandardScaler()
        self.model = linear_model.LinearRegression(optimizer=optim.SGD(learning_rate))

    def predict_one(self, x):
        return self.model.predict_one(self.scaler.transform_one(x))

    def learn_one(self, x, y):
        self.scaler.learn_one(x)
        self.model.learn_one(self.scaler.transform_one(x), y)
