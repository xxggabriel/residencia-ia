"""Exact KNN over the shared incremental scaler/one-hot encoding.

Stored neighbors retain their encoding at admission time; evolving scaler
statistics can therefore give old/new neighbors different coordinate systems.
This is the usual incremental-pipeline behavior, not a claim of scale invariance.
"""

from river import neighbors, preprocessing

from .encoding import EncodedEstimator, read_only_scale, split_features
from .linear import RiverModel
from .tree import positive_probability


def _validate(n_neighbors, window_size):
    if n_neighbors < 1 or window_size < 1 or n_neighbors > window_size:
        raise ValueError("require 1 <= n_neighbors <= window_size")


class KNNModel(RiverModel):
    """Exact weighted binary KNN, bounded by an explicit FIFO neighbor window.

    There is no random approximate-neighbor graph. Equal-distance neighbors use
    their insertion order in LazySearch. Predictions do not update the scaler.
    """

    def __init__(self, n_neighbors: int = 11, window_size: int = 256, weighted: bool = True):
        _validate(n_neighbors, window_size)
        self.scaler = preprocessing.StandardScaler()
        self.classifier = neighbors.KNNClassifier(
            n_neighbors=n_neighbors,
            engine=neighbors.LazySearch(window_size=window_size),
            weighted=weighted,
            cleanup_every=0,
            softmax=False,
        )

    def encode(self, x: dict, *, learn: bool) -> dict:
        if learn:
            return super().encode(x, learn=True)
        numeric, categorical = split_features(x)
        return {**read_only_scale(self.scaler, numeric), **categorical}

    def predict(self, x: dict) -> float:
        return positive_probability(self.classifier.predict_proba_one(self.encode(x, learn=False)))


def knn_multiclass(n_neighbors: int = 11, window_size: int = 256, weighted: bool = True) -> EncodedEstimator:
    _validate(n_neighbors, window_size)
    return EncodedEstimator(neighbors.KNNClassifier(
        n_neighbors=n_neighbors, engine=neighbors.LazySearch(window_size=window_size),
        weighted=weighted, cleanup_every=0, softmax=False))


def knn_regressor(n_neighbors: int = 11, window_size: int = 256, aggregation_method: str = "weighted_mean"
                  ) -> EncodedEstimator:
    _validate(n_neighbors, window_size)
    return EncodedEstimator(neighbors.KNNRegressor(
        n_neighbors=n_neighbors, engine=neighbors.LazySearch(window_size=window_size),
        aggregation_method=aggregation_method))
