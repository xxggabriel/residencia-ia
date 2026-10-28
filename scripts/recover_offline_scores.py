"""Fresh-process inference from a frozen Registry model and its bundled code only."""

import importlib
import json
import math
import sys
from pathlib import Path


def recover_loaded(request, model, loader, implementation):
    """Validate bundled inference and preserve the requested temporal suffix order."""
    import pandas as pd

    for module in (loader, implementation):
        if Path(module.__file__).resolve().parents[2].name != "code":
            raise ValueError("Registry inference must use bundled code")
    expected = {"logistic": "RiverModel", "hoeffding_tree": "HoeffdingTreeModel", "knn": "KNNModel"}
    if (request["model"] not in expected
            or type(model._model_impl.model).__name__ != expected[request["model"]]):
        raise ValueError("Registry model implementation differs from the requested model")
    rows = request["rows"]
    schema = model.metadata.get_input_schema().inputs
    names = {column.name for column in schema}
    if not rows or any(set(row["features"]) != names for row in rows):
        raise ValueError("Recovery rows must match the frozen model feature schema")
    frame = pd.DataFrame([row["features"] for row in rows])
    for column in schema:
        name = getattr(column.type, "name", None)
        if name in ("double", "float"):
            frame[column.name] = frame[column.name].astype("float64" if name == "double" else "float32")
        elif name in ("long", "integer"):
            if not all(math.isfinite(value) and value == int(value) for value in frame[column.name]):
                raise ValueError("Frozen integer schema cannot represent a fractional feature")
            frame[column.name] = frame[column.name].astype("int64" if name == "long" else "int32")
    probabilities = [float(value) for value in model.predict(frame)]
    if len(probabilities) != len(rows) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Registry returned invalid recovery probabilities")
    return {
        "uri": request["uri"],
        "model": request["model"],
        "seqs": [row["seq"] for row in rows],
        "event_ids": [row["event_id"] for row in rows],
        "probabilities": probabilities,
        "fresh_process": True,
        "bundled_loader": loader.__file__,
        "bundled_model_code": implementation.__file__,
    }


def main():
    request = json.load(sys.stdin)
    if any(name == "continual_bench" or name.startswith("continual_bench.") for name in sys.modules):
        raise RuntimeError("Recovery must start without the current training package imported")
    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(request["uri"])
    loader = importlib.import_module("continual_bench.adapters.model_loader")
    implementation = importlib.import_module(type(model._model_impl.model).__module__)
    print(json.dumps(recover_loaded(request, model, loader, implementation), allow_nan=False))


if __name__ == "__main__":
    main()
