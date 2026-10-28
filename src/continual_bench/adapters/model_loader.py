"""MLflow Python-function flavor for the versioned River model and its preprocessing."""

import hashlib
import json
import pickle
from pathlib import Path


def inference_frame(features):
    import pandas as pd

    frame = pd.DataFrame([features])
    # A numeric generator feature may be int on the first row and float later
    # (e.g. Agrawal commission=0). Do not freeze that accident into the signature.
    for column in frame.select_dtypes(include=["number", "bool"]).columns:
        frame[column] = frame[column].astype("float64")
    return frame


class RiverPredictor:
    def __init__(self, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "model-metadata.json").read_text())
        blob = (directory / "model.pkl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != metadata["state_sha256"]:
            raise ValueError("model artifact checksum mismatch")
        self.model = pickle.loads(blob)  # Only self-produced, authenticated registry artifacts.
        self.features = set(metadata["inference_probe"]["features"])

    def predict(self, data, params=None):
        import numpy as np

        if set(data.columns) != self.features:
            raise ValueError("inference feature schema differs from the trained model")
        return np.asarray([self.model.predict(row) for row in data.to_dict(orient="records")])


def _load_pyfunc(path):
    return RiverPredictor(path)
