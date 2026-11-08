"""Component factory over the dataset, model, detector and policy catalogs.

Abstract Factory + plugin hook: every run builds its components through one
`ComponentRegistry`, so engines and evaluators never import concrete adapters.
Plugins receive the registry and call `register(kind, name, factory)`.
"""

import importlib

from continual_bench.core.policies import POLICIES
from continual_bench.core.registry import Registry, Spec
from continual_bench.core.tasks import BINARY
from continual_bench.datasets.catalog import CATALOG, DatasetSpec
from continual_bench.detectors import DETECTORS
from continual_bench.models import MODELS, IncompatibleComponent, ModelFamily

KINDS = ("source", "model", "detector", "policy")


class ComponentRegistry:
    def __init__(self, sources=CATALOG, models=MODELS, detectors=DETECTORS, policies=None):
        if policies is None:
            policies = Registry("policy")
            for name, policy in POLICIES.items():
                policies.add(Spec(name, policy))
        self.catalogs = {"source": sources.copy(), "model": models.copy(),
                         "detector": detectors.copy(), "policy": policies.copy()}

    @property
    def sources(self) -> Registry[DatasetSpec]:
        return self.catalogs["source"]

    @property
    def models(self) -> Registry[ModelFamily]:
        return self.catalogs["model"]

    def _catalog(self, kind):
        if kind not in self.catalogs:
            raise ValueError(f"unknown component kind {kind}; available={list(KINDS)}")
        return self.catalogs[kind]

    def register(self, kind, name, factory, *, tasks=(BINARY,), task=None, **metadata):
        """Plugin API. Models default to binary; sources may declare their target type."""
        catalog = self._catalog(kind)
        if kind == "model":
            catalog.add(ModelFamily(name, {t: factory for t in tasks}, metadata=metadata))
        elif kind == "source":
            catalog.add(DatasetSpec(name, task, "plugin", factory, metadata=metadata))
        else:
            catalog.add(Spec(name, factory, metadata=metadata))

    def factory(self, kind, name, **params):
        """Zero-argument factory with bound parameters (binary target type for models)."""
        if kind == "model":
            return self.model_factory(name, BINARY, **params)
        if kind == "source":
            spec = self.sources.get(name)
            return lambda: spec.open_stream(**params)
        return self._catalog(kind).get(name).partial(**params)

    def model_factory(self, name, task, **params):
        return self.models.get(name).factory(task, **params)

    def source(self, dataset):
        """Instantiate a DatasetConfig-described stream."""
        fields = dataset.model_dump(exclude={"params", "name"})
        return self.sources.get(dataset.name).open_stream(**fields, **dataset.params)

    def ensure_compatible(self, dataset_name, model_name, task=None):
        """Fail before materialization when the model cannot learn the dataset's targets."""
        spec = self.sources.get(dataset_name)
        task = task or spec.task
        if task is not None and not self.models.get(model_name).supports(task):
            raise IncompatibleComponent(
                f"model {model_name} does not support {task} targets of dataset {dataset_name}")


def default_registry(plugins=()):
    registry = ComponentRegistry()
    for plugin in plugins:
        module, function = plugin.split(":")
        getattr(importlib.import_module(module), function)(registry)
    return registry


def make_source(registry, config):
    return registry.source(config)
