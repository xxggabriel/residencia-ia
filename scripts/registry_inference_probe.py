"""Run in a fresh process: prove Registry inference uses its bundled code."""

import importlib
import json
import sys
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

request = json.load(sys.stdin)
assert not any(key.startswith("continual_bench") for key in sys.modules)
model = mlflow.pyfunc.load_model(request["uri"])
loader = importlib.import_module("continual_bench.adapters.model_loader")
implementation = importlib.import_module(type(model._model_impl.model).__module__)
if request.get("model"):
    expected = {"logistic": "RiverModel", "hoeffding_tree": "HoeffdingTreeModel", "knn": "KNNModel"}
    assert type(model._model_impl.model).__name__ == expected[request["model"]]
for module in (loader, implementation):
    assert Path(module.__file__).resolve().parents[2].name == "code", module.__file__
frame = pd.DataFrame([request["probe"]["features"]])
for column in model.metadata.get_input_schema().inputs:
    if getattr(column.type, "name", None) == "double":
        frame[column.name] = frame[column.name].astype("float64")
probability = model.predict(frame).tolist()
assert probability == [request["probe"]["probability"]]
print(
    json.dumps(
        {
            "dataset": request["dataset"],
            "uri": request["uri"],
            "prediction_identical": True,
            "fresh_process": True,
            "model": request.get("model", "logistic"),
            "bundled_loader": loader.__file__,
            "bundled_model_code": implementation.__file__,
        }
    )
)
