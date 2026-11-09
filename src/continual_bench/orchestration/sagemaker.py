"""SageMaker container contract for the `ml` image: `process`, `train` and `serve`.

- `process`: profile a prepared snapshot (rows, class balance per month, feature ranges).
- `train`: fit the project's Zia learner on a chronological train split and score a separate,
  later validation split (the frozen temporal holdout is never read). The objective is printed
  as `validation:log_loss=<value>;` for Automatic Model Tuning.
- `serve`: stdlib HTTP server on :8080 with `/ping` and `/invocations`, scoring through
  `ZiaPredictor`, the same native loader the benchmark validates.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PREFIX = Path(os.environ.get("SM_PREFIX", "/opt/ml"))
HOLDOUT_FRACTION = 0.1


def hyperparameters() -> dict:
    path = PREFIX / "input" / "config" / "hyperparameters.json"
    raw = json.loads(path.read_text()) if path.is_file() else {}

    def parse(value):
        try:
            return json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            return value

    return {k: parse(v) for k, v in raw.items()}


def chronological_splits(rows: int, validation_fraction: float, max_train_rows: int):
    holdout_start = rows - int(rows * HOLDOUT_FRACTION)
    validation_start = holdout_start - int(rows * validation_fraction)
    train_start = max(0, validation_start - max_train_rows)
    if not 0 <= train_start < validation_start < holdout_start:
        raise ValueError("dataset too small for train/validation/holdout splits")
    return (train_start, validation_start), (validation_start, holdout_start), holdout_start


def _read(path, columns, start, stop):
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["label", *columns]).slice(start, stop - start)
    x = np.column_stack([table[c].to_numpy().astype(np.float32) for c in columns])
    return x, table["label"].to_numpy().astype(np.uint8)


def train(channel=None, model_dir=None, params=None) -> dict:

    from continual_bench.datasets.zia import DATASETS
    from continual_bench.evaluation.zia_metrics import ClassificationMetrics
    from continual_bench.models.zia import FraudOrReturnBooster, MultinomialLogistic

    params = params if params is not None else hyperparameters()
    channel = Path(channel or PREFIX / "input" / "data" / "train")
    model_dir = Path(model_dir or PREFIX / "model")
    scope = str(params.get("scope", "smoke"))
    manifest = json.loads((channel / f"{scope}.json").read_text())
    parquet = next(p for p in (channel / scope / "data.parquet", channel / f"{scope}.parquet") if p.is_file())
    dataset = manifest["dataset"]
    classes = DATASETS[dataset]["classes"]
    (train_start, train_stop), (val_start, val_stop), holdout_start = chronological_splits(
        manifest["rows"],
        float(params.get("validation_fraction", 0.1)),
        int(params.get("max_train_rows", 2_000_000)),
    )
    model = MultinomialLogistic(len(classes)) if dataset == "social" else FraudOrReturnBooster()
    if dataset == "social":
        model.params = {**model.params, "l2": float(params.get("l2", model.params["l2"]))}
        model.initial_rounds = int(params.get("max_iterations", model.initial_rounds))
    else:
        model.params = {**model.params, **{k: params[k] for k in ("eta", "max_depth") if k in params}}
        model.initial_rounds = int(params.get("rounds", model.initial_rounds))
    x, y = _read(parquet, manifest["features"], train_start, train_stop)
    model.fit(x, y, False)
    del x, y
    vx, vy = _read(parquet, manifest["features"], val_start, val_stop)
    metrics = ClassificationMetrics(len(classes))
    metrics.add(vy, model.predict(vx))
    summary, _ = metrics.metrics()
    payload, extension = model.payload()
    model_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    (model_dir / f"model.{extension}").write_bytes(payload)
    metadata = {
        "model": model.name,
        "model_params": model.params,
        "features": manifest["features"],
        "classes": classes,
        "dataset_version": manifest["sha256"],
        "state_sha256": hashlib.sha256(payload).hexdigest(),
        "payload": f"model.{extension}",
        "native_path": f"model.{extension}",
        "code_sha256": None,
        "output": "class probabilities in classes order",
        "splits": {
            "train": [train_start, train_stop],
            "validation": [val_start, val_stop],
            "holdout_start_excluded": holdout_start,
        },
    }
    (model_dir / "model-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"validation:log_loss={summary['log_loss']};", flush=True)
    print(f"validation:accuracy={summary['accuracy']};", flush=True)
    output = PREFIX / "output"
    if output.is_dir():
        (output / "metrics.json").write_text(json.dumps({"validation": summary}, default=float))
    return {"validation": summary, "metadata": metadata}


def process(input_dir=None, output_dir=None) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    input_dir = Path(input_dir or PREFIX / "processing" / "input")
    output_dir = Path(output_dir or PREFIX / "processing" / "output")
    profile = {}
    for parquet in sorted(input_dir.glob("*.parquet")):
        months, features = {}, {}
        with pq.ParquetFile(parquet) as source:
            for batch in source.iter_batches(batch_size=262_144, use_threads=False):
                month = (batch["event_time"].to_numpy() // (30 * 86_400_000)).astype(np.int64)
                labels = batch["label"].to_numpy()
                pairs, counts = np.unique(np.column_stack((month, labels)), axis=0, return_counts=True)
                for (m, label), count in zip(pairs, counts, strict=True):
                    bucket = months.setdefault(str(m), {})
                    bucket[str(label)] = bucket.get(str(label), 0) + int(count)
                for name in batch.schema.names[2:]:
                    values = batch[name].to_numpy(zero_copy_only=False).astype(np.float64)
                    current = features.setdefault(name, {"min": np.inf, "max": -np.inf, "nulls": 0})
                    current["min"] = min(current["min"], float(np.nanmin(values)))
                    current["max"] = max(current["max"], float(np.nanmax(values)))
                    current["nulls"] += int(batch[name].null_count)
            profile[parquet.name] = {
                "rows": source.metadata.num_rows,
                "class_by_month": months,
                "features": features,
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2))
    return profile


def load_predictor(model_dir=None):
    from continual_bench.models.zia import ZiaPredictor

    return ZiaPredictor(Path(model_dir or PREFIX / "model"))


def score(predictor, body: bytes, content_type: str):
    import pandas as pd

    if content_type.startswith("text/csv"):
        rows = [[float(v) for v in line.split(",")] for line in body.decode().splitlines() if line.strip()]
        frame = pd.DataFrame(rows, columns=predictor.features)
    else:
        payload = json.loads(body)
        if "instances" in payload:
            frame = pd.DataFrame(payload["instances"], columns=predictor.features)
        else:
            frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    return predictor.predict(frame.astype("float64")).tolist()


def serve(model_dir=None, port=8080):
    predictor = load_predictor(model_dir)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(
                200 if self.path == "/ping" else 404, {"status": "ok" if self.path == "/ping" else "unknown"}
            )

        def do_POST(self):
            if self.path != "/invocations":
                return self._send(404, {"error": "unknown path"})
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                probabilities = score(predictor, body, self.headers.get("Content-Type", "application/json"))
            except (ValueError, KeyError) as error:
                return self._send(400, {"error": str(error)})
            self._send(200, {"probabilities": probabilities})

        def log_message(self, fmt, *args):
            print(json.dumps({"access": fmt % args}), flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(json.dumps({"serving": port}), flush=True)
    server.serve_forever()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if argv else "serve"
    if command == "train":
        print(json.dumps(train()["validation"], default=float))
    elif command == "process":
        process()
    elif command == "serve":
        serve()
    else:
        raise SystemExit(f"unknown SageMaker command: {command}")


if __name__ == "__main__":
    main()
