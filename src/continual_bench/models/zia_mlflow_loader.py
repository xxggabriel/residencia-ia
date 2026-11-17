"""Load the Zia predictor from the source archive saved with an MLflow model."""

import sys
from pathlib import Path


def _load_pyfunc(path):
    archive = str(Path(__file__).with_name("continual_bench.zip"))
    sys.path.insert(0, archive)
    try:
        from continual_bench.models.zia import ZiaPredictor

        return ZiaPredictor(path)
    finally:
        sys.path.remove(archive)
