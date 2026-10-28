import importlib
from functools import partial

from continual_bench.domain.policies import PolicyStrategy


class Registry:
    def __init__(self):
        self.entries = {kind: {} for kind in ("source", "model", "detector", "policy")}

    def register(self, kind, name, factory):
        if name in self.entries[kind]:
            raise ValueError(f"duplicate {kind} registration: {name}")
        self.entries[kind][name] = factory

    def factory(self, kind, name, **params):
        try:
            return partial(self.entries[kind][name], **params)
        except KeyError as e:
            raise ValueError(f"unknown {kind} {name}; available={sorted(self.entries[kind])}") from e


def make_source(registry, config):
    return registry.factory("source", **config.model_dump(exclude={"params"}), **config.params)()


def default_registry(plugins=()):
    from continual_bench.adapters.nonlinear import HoeffdingTreeModel, KNNModel
    from continual_bench.adapters.real_sources import RealSource
    from continual_bench.adapters.river import NullDetector, RiverDetector, RiverModel, SyntheticSource

    registry = Registry()
    for name in ("sea", "agrawal", "stagger"):
        registry.register("source", name, partial(SyntheticSource, name=name))
    for name in ("electricity", "airlines"):
        registry.register("source", name, partial(RealSource, name=name))
    registry.register("model", "logistic", RiverModel)
    registry.register("model", "hoeffding_tree", HoeffdingTreeModel)
    registry.register("model", "knn", KNNModel)
    registry.register("detector", "none", NullDetector)
    for name in ("adwin", "ddm"):
        registry.register("detector", name, partial(RiverDetector, name=name))
    for name in ("none", "periodic", "drift", "online"):
        registry.register("policy", name, partial(PolicyStrategy, kind=name))
    for plugin in plugins:
        module, function = plugin.split(":")
        getattr(importlib.import_module(module), function)(registry)
    return registry
