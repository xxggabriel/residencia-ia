"""Configuration files with base inheritance and overrides.

Any mapping in a YAML/JSON configuration file may declare `extends` with one
path or a list of paths, relative to the file that declares it. Bases are
merged in order, then the mapping's own keys override them:

- mappings merge recursively, key by key;
- lists and scalars replace the inherited value entirely;
- an explicit `null` overrides an inherited value with null.

`extends` is resolved only when reading trusted local files; validated schemas
forbid the key, so HTTP job payloads must stay self-contained.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

EXTENDS = "extends"
M = TypeVar("M", bound=BaseModel)


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(base[key], value) if key in base else deepcopy(value)
        return merged
    return deepcopy(override)


def _read(path: Path) -> Any:
    with path.open() as stream:
        return yaml.safe_load(stream)  # YAML is a superset of the JSON files used here.


def _resolve(value: Any, directory: Path, chain: tuple[Path, ...]) -> Any:
    if isinstance(value, list):
        return [_resolve(item, directory, chain) for item in value]
    if not isinstance(value, dict):
        return value
    own = {key: _resolve(item, directory, chain) for key, item in value.items() if key != EXTENDS}
    bases = value.get(EXTENDS, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(base, str) and base for base in bases):
        raise ValueError(f"{EXTENDS} must be a path or a list of paths")
    merged: Any = {}
    for base in bases:
        merged = deep_merge(merged, _load(directory / base, chain))
    return deep_merge(merged, own)


def _load(path: Path, chain: tuple[Path, ...] = ()) -> Any:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(p.name) for p in (*chain, path))
        raise ValueError(f"configuration inheritance cycle: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    document = _read(path)
    if not isinstance(document, dict):
        raise ValueError(f"configuration file must contain a mapping: {path}")
    return _resolve(document, path.parent, (*chain, path))


def load_document(path: str | Path) -> dict:
    """Read a configuration file and resolve every `extends` it references."""
    return _load(Path(path))


def load_model(schema: type[M], path: str | Path) -> M:
    """Resolve inheritance, then validate against a configuration schema."""
    return schema.model_validate(load_document(path))
