"""Model families: one algorithm, one builder per supported target type.

Abstract Factory: `MODELS.get("knn").build("regression", n_neighbors=5)` returns a
learner for that task. Binary builders return the engine's probability contract
(`predict`/`learn`); multiclass/regression builders return `predict_one`/`learn_one`.
Builders are import strings, so querying compatibility never imports River.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Mapping

from continual_bench.core.registry import Registry, resolve
from continual_bench.core.tasks import BINARY, MULTICLASS, REGRESSION


class IncompatibleComponent(ValueError):
    """A component cannot serve the dataset's declared target type."""


@dataclass(frozen=True)
class ModelFamily:
    name: str
    builders: Mapping[str, str | Callable]
    description: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.builders)

    def supports(self, task: str) -> bool:
        return task in self.builders

    def factory(self, task: str, **params) -> Callable:
        if not self.supports(task):
            raise IncompatibleComponent(
                f"model {self.name} does not support {task} targets; supported={list(self.tasks)}")
        return partial(resolve(self.builders[task]), **params)

    def build(self, task: str, **params):
        return self.factory(task, **params)()


MODELS: Registry[ModelFamily] = Registry("model")
MODELS.add(ModelFamily("logistic", {
    BINARY: "continual_bench.models.linear:RiverModel",
    MULTICLASS: "continual_bench.models.linear:logistic_multiclass",
}, "SGD logistic regression with incremental scaling and one-hot nominals"))
MODELS.add(ModelFamily("linear_regression", {
    REGRESSION: "continual_bench.models.linear:LinearRegressor",
}, "Standardized SGD linear regression"))
MODELS.add(ModelFamily("hoeffding_tree", {
    BINARY: "continual_bench.models.tree:HoeffdingTreeModel",
    MULTICLASS: "continual_bench.models.tree:hoeffding_multiclass",
    REGRESSION: "continual_bench.models.tree:hoeffding_regressor",
}, "Bounded Hoeffding tree on native numeric/nominal features"))
MODELS.add(ModelFamily("knn", {
    BINARY: "continual_bench.models.neighbors:KNNModel",
    MULTICLASS: "continual_bench.models.neighbors:knn_multiclass",
    REGRESSION: "continual_bench.models.neighbors:knn_regressor",
}, "Exact KNN over a FIFO window with incremental scaling"))

# Historical defaults of the typed/epoch evaluators; binary keeps the engine default.
DEFAULT_MODELS = {BINARY: "logistic", MULTICLASS: "hoeffding_tree", REGRESSION: "linear_regression"}


def default_model(task: str) -> str:
    return DEFAULT_MODELS[task]


def compatible_models(task: str, registry: Registry[ModelFamily] = MODELS) -> tuple[str, ...]:
    return tuple(family.name for family in registry if family.supports(task))
