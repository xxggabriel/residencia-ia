"""Generic named registry with lazily imported factories.

Factories may be callables or "module:attribute" strings. Strings keep catalogs
cheap to import (configuration validation never imports River or PyArrow) and
let a plugin register components without importing the whole project.
"""

import importlib
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Generic, Iterator, TypeVar


def resolve(factory: str | Callable) -> Callable:
    if callable(factory):
        return factory
    module, _, attribute = factory.partition(":")
    if not module or not attribute:
        raise ValueError(f"factory must be callable or module:attribute, got {factory!r}")
    return getattr(importlib.import_module(module), attribute)


@dataclass(frozen=True)
class Spec:
    """A registered component: factory, fixed keyword arguments and searchable metadata."""

    name: str
    factory: str | Callable
    defaults: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def build(self, **params) -> Any:
        return partial(resolve(self.factory), **self.defaults)(**params)

    def partial(self, **params) -> Callable:
        return partial(resolve(self.factory), **self.defaults, **params)


S = TypeVar("S", bound=Spec)


class Registry(Generic[S]):
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, S] = {}

    def add(self, spec: S) -> S:
        if spec.name in self._entries:
            raise ValueError(f"duplicate {self.kind} registration: {spec.name}")
        self._entries[spec.name] = spec
        return spec

    def get(self, name: str) -> S:
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"unknown {self.kind} {name}; available={sorted(self._entries)}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[S]:
        return iter(self._entries.values())

    def copy(self) -> "Registry[S]":
        clone = type(self)(self.kind)
        clone._entries = dict(self._entries)
        return clone
