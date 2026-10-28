"""Immutable identities of complete plugin packages and explicitly declared helpers."""

import hashlib
import importlib
from pathlib import Path


def package_identity(plugins, dependencies=()):
    names = {plugin.split(":")[0].split(".")[0] for plugin in plugins} | {
        name.split(".")[0] for name in dependencies
    }
    result = {}
    for name in sorted(names):
        module = importlib.import_module(name)
        locations = list(getattr(module, "__path__", ()))
        digest = hashlib.sha256()
        if locations:
            for index, location in enumerate(sorted(locations)):
                directory = Path(location)
                for path in sorted(directory.rglob("*")):
                    relative = path.relative_to(directory)
                    if (
                        not path.is_file()
                        or "__pycache__" in relative.parts
                        or path.suffix in {".pyc", ".pyo"}
                        or path.name.startswith("._")
                        or path.name in {".DS_Store", ".env"}
                    ):
                        continue
                    digest.update(f"{index}/{relative.as_posix()}".encode())
                    digest.update(path.read_bytes())
        else:
            path = Path(module.__file__)
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        result[name] = digest.hexdigest()
    return result
