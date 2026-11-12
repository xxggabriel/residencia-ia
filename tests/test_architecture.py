"""Architecture fitness: each layer imports only the layers beneath it."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "continual_bench"
THIRD_PARTY = {"river", "pydantic", "yaml", "zenml", "mlflow", "boto3", "confluent_kafka", "pandas", "numpy",
               "pyarrow", "PIL"}
UPPER = {
    "core": {"config", "datasets", "models", "detectors", "evaluation", "application", "infrastructure",
             "orchestration", "cli"},
    "datasets": {"evaluation", "application", "orchestration", "cli"},
    "models": {"datasets", "evaluation", "application", "infrastructure", "orchestration", "cli"},
    "detectors": {"datasets", "models", "evaluation", "application", "infrastructure", "orchestration",
                  "cli"},
    "config": {"evaluation", "application", "infrastructure", "orchestration", "cli"},
    "retraining": {"config", "datasets", "models", "detectors", "evaluation", "application",
                   "infrastructure", "orchestration", "cli"},
    "infrastructure": {"evaluation", "application", "orchestration", "cli"},
    "evaluation": {"application", "orchestration", "cli"},
    "application": {"orchestration", "cli"},
}


def imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # Relative imports stay inside the same package by construction.
                continue
            yield node.module or ""


@pytest.mark.parametrize("layer", sorted(UPPER))
def test_layer_never_imports_upper_layers(layer):
    violations = []
    for path in sorted((PACKAGE / layer).rglob("*.py")):
        for name in imports(path):
            parts = name.split(".")
            if parts[0] == "continual_bench" and len(parts) > 1 and parts[1] in UPPER[layer]:
                violations.append(f"{path.relative_to(PACKAGE)} -> {name}")
            if layer == "core" and parts[0] in THIRD_PARTY:
                violations.append(f"{path.relative_to(PACKAGE)} -> {name} (third-party in core)")
    assert not violations, "\n".join(violations)
