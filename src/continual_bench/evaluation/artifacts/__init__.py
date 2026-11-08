"""Run artifacts: segmented SVG plots and the JSON data behind them.

Layout inside a run directory (mirrored at the root of the MLflow run artifacts):

    artifacts/plots/<group>/<name>.svg   one chart per figure
    artifacts/data/<group>/<name>.json   the exact data drawn in that chart
    artifacts/data/<dump>.json           complete metric dumps (summary, series, events, system)
    artifacts/data/index.json            every figure with its plot and data paths
"""

import json
from pathlib import Path

from continual_bench.infrastructure.storage import atomic_write

from .figures import Figure, binary_figures, epoch_figures, system_figures, typed_figures

ROOT = "artifacts"
__all__ = ["Figure", "binary_figures", "epoch_figures", "system_figures", "typed_figures", "write_artifacts"]


def _dump(value) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str).encode()


def write_artifacts(directory, figures: list[Figure], dumps: dict[str, object]) -> dict:
    """Render every figure, save its data and the raw dumps; return the index."""
    root = Path(directory) / ROOT
    index = {"version": "run-artifacts-v1", "figures": [], "data": []}
    seen = set()
    for figure in figures:
        key = (figure.group, figure.name)
        if key in seen:
            raise ValueError(f"duplicate figure {figure.group}/{figure.name}")
        seen.add(key)
        plot = f"plots/{figure.group}/{figure.name}.svg"
        data = f"data/{figure.group}/{figure.name}.json"
        atomic_write(root / plot, figure.render().encode())
        atomic_write(
            root / data,
            _dump(
                {
                    "title": figure.title,
                    "group": figure.group,
                    "name": figure.name,
                    "kind": figure.kind,
                    **figure.data,
                }
            ),
        )
        index["figures"].append(
            {"group": figure.group, "name": figure.name, "title": figure.title, "plot": plot, "data": data}
        )
    for name, value in sorted(dumps.items()):
        if value is None:
            continue
        path = f"data/{name}.json"
        atomic_write(root / path, _dump(value))
        index["data"].append(path)
    index["groups"] = sorted({figure.group for figure in figures})
    atomic_write(root / "data/index.json", _dump(index))
    return {
        "root": ROOT,
        "figures": len(index["figures"]),
        "groups": index["groups"],
        "data_files": len(index["data"]) + len(index["figures"]) + 1,
    }
