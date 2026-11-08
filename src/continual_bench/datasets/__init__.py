"""Dataset adapters and the catalog that declares every admitted dataset.

Importing this package is cheap: adapters (River, PyArrow, Pillow) load lazily
through the catalog's import-string factories.
"""

from .catalog import CATALOG, DATASETS, DatasetSpec, dataset_spec, datasets_for, load_dataset

__all__ = ["CATALOG", "DATASETS", "DatasetSpec", "dataset_spec", "datasets_for", "load_dataset"]
