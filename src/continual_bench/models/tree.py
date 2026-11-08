"""Hoeffding trees learning native numeric/nominal inputs, without scaling."""

from river import tree
from river.tree.nodes.branch import DTBranch
from river.tree.splitter.nominal_splitter_classif import NominalSplitterClassif

from .encoding import NativeEstimator


def positive_probability(distribution):
    # A nonempty, single-negative-class distribution means p(True)=0, not 0.5.
    return float(distribution.get(True, 0.0)) if distribution else 0.5


class HoeffdingTreeModel:
    """Hoeffding tree; inferred string categories are kept nominal, without scaling."""

    def __init__(
        self,
        grace_period: int = 50,
        max_depth: int = 8,
        leaf_prediction: str = "nba",
        max_size: float = 8.0,
        memory_estimate_period: int = 100,
        nominal_attributes: list[str] | None = None,
    ):
        if grace_period < 1 or max_depth < 1 or max_size <= 0 or memory_estimate_period < 1:
            raise ValueError("tree growth and memory limits must be positive")
        if leaf_prediction not in {"mc", "nb", "nba"}:
            raise ValueError("leaf_prediction must be mc, nb or nba")
        if nominal_attributes is not None and (
            not isinstance(nominal_attributes, (list, tuple))
            or any(not isinstance(name, str) for name in nominal_attributes)
            or len(set(nominal_attributes)) != len(nominal_attributes)
        ):
            raise ValueError("nominal_attributes must be unique feature names")
        self.classifier = tree.HoeffdingTreeClassifier(
            grace_period=grace_period,
            max_depth=max_depth,
            leaf_prediction=leaf_prediction,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            nominal_attributes=None if nominal_attributes is None else list(nominal_attributes),
        )

    def predict(self, x: dict) -> float:
        root = self.classifier._root
        leaf = root.traverse(x, until_leaf=True) if isinstance(root, DTBranch) else root
        observers = (getattr(leaf, "splitters", None) or {}).values()
        before = [
            (observer._att_dist_per_class, set(observer._att_dist_per_class))
            for observer in observers
            if isinstance(observer, NominalSplitterClassif)
        ]
        try:
            return positive_probability(self.classifier.predict_proba_one(x))
        finally:
            # River 0.26.1 cond_proba indexes a defaultdict for classes that may
            # occur in leaf priors but not in its nominal observer. Remove only
            # the empty entries created by this prediction, preserving priors
            # and all pre-existing state, including pre-existing empty entries.
            unexpected = False
            for distributions, known in before:
                for target in tuple(distributions):
                    if target not in known:
                        values = distributions[target]
                        if isinstance(values, dict) and not values:
                            del distributions[target]
                        else:
                            unexpected = True
            if unexpected:
                raise RuntimeError("tree prediction created nonempty nominal class statistics")

    def learn(self, x: dict, y: bool) -> None:
        self.classifier.learn_one(dict(x), bool(y))


def hoeffding_multiclass(grace_period: int = 25, max_depth: int = 12, max_size: float = 64,
                         **params) -> NativeEstimator:
    return NativeEstimator(tree.HoeffdingTreeClassifier(grace_period=grace_period, max_depth=max_depth,
                                                        max_size=max_size, **params))


def hoeffding_regressor(grace_period: int = 50, max_depth: int = 12, max_size: float = 64,
                        **params) -> NativeEstimator:
    return NativeEstimator(tree.HoeffdingTreeRegressor(grace_period=grace_period, max_depth=max_depth,
                                                       max_size=max_size, **params))
