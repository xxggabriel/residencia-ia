"""SageMaker container contract for the `ml` image: `process`, `train` and `serve`.

- `process`: profile a prepared snapshot (rows, class balance per month, feature ranges).
- `train`: fit the project's Zia learner on a chronological train split and score a separate,
  later validation split before the frozen temporal holdout. The objective is printed
  as `validation:log_loss=<value>;` for Automatic Model Tuning.
- `serve`: stdlib HTTP server on :8080 with `/ping` and `/invocations`, scoring through
  `ZiaPredictor`, the same native loader the benchmark validates.
"""

import gc
import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables, offset = [], 0
    with pq.ParquetFile(path) as source:
        if not 0 <= start < stop <= source.metadata.num_rows:
            raise ValueError("training and validation ranges must be within the prepared snapshot")
        for index in range(source.num_row_groups):
            end = offset + source.metadata.row_group(index).num_rows
            if end > start:
                group_start = max(start, offset)
                table = source.read_row_group(index, columns=["label", *columns], use_threads=False)
                tables.append(table.slice(group_start - offset, min(stop, end) - group_start))
            if end >= stop:
                break
            offset = end
    table = pa.concat_tables(tables)
    x = np.column_stack([table[c].to_numpy().astype(np.float32) for c in columns])
    return x, table["label"].to_numpy().astype(np.uint8)


def _training_manifest(channel, scope):
    manifest = json.loads((Path(channel) / f"{scope}.json").read_text())
    if manifest.get("scope") != scope:
        raise ValueError("prepared training manifest scope differs from the job scope")
    return manifest


def _training_job_name():
    if job_name := os.environ.get("TRAINING_JOB_NAME"):
        return job_name
    if value := os.environ.get("SM_TRAINING_ENV"):
        return json.loads(value).get("job_name")
    return None


def train(channel=None, model_dir=None, params=None, *, tracker=None, finish_tracker=True) -> dict:
    from continual_bench.config.iterative import IterativeTrainingConfig
    from continual_bench.datasets.zia import DATASETS
    from continual_bench.evaluation.environment import encoded, environment
    from continual_bench.evaluation.epoch_history import epoch_fit, write_state
    from continual_bench.evaluation.training_evidence import write_training_artifacts
    from continual_bench.evaluation.zia_metrics import ClassificationMetrics
    from continual_bench.infrastructure.storage import atomic_write
    from continual_bench.models.training.loop import train_iterative
    from continual_bench.models.zia import make_zia_model

    params = params if params is not None else hyperparameters()
    iter_config = IterativeTrainingConfig.from_params(params)
    channel = Path(channel or PREFIX / "input" / "data" / "train")
    model_dir = Path(model_dir or PREFIX / "model")
    scope = str(params.get("scope", "smoke"))
    manifest = _training_manifest(channel, scope)
    parquet = next(p for p in (channel / scope / "data.parquet", channel / f"{scope}.parquet") if p.is_file())
    dataset = manifest["dataset"]
    spec = DATASETS[dataset]
    classes = spec["classes"]
    (train_start, train_stop), (val_start, val_stop), holdout_start = chronological_splits(
        manifest["rows"],
        iter_config.validation_fraction,
        int(params.get("max_train_rows", 2_000_000)),
    )
    threshold = float(params.get("decision_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1")
    model = make_zia_model(spec["model"], classes, config=iter_config, dataset=dataset,
                           decision_threshold=threshold)
    if model.name == "xgboost" and "max_depth" in params:
        model.params["max_depth"] = int(params["max_depth"])

    if tracker is None and (
        os.environ.get("SM_TRAINING_ENV") or os.environ.get("BENCH_SAGEMAKER_EXPERIMENTS") == "true"
    ):
        from continual_bench.infrastructure.aws.experiments import SageMakerExperimentsTracker

        job_name = _training_job_name() or f"cb-train-{dataset}-{scope}"
        tracker = SageMakerExperimentsTracker(
            dataset=dataset,
            policy=str(params.get("policy", "frozen")),
            model_name=model.name,
            run_id=str(params.get("run_id", scope)),
            job_name=job_name,
            emit_stdout=False,
        )
        tracker.start(iter_config)

    x, y = _read(parquet, manifest["features"], train_start, train_stop)
    vx, vy = _read(parquet, manifest["features"], val_start, val_stop)
    loop_result = train_iterative(
        model,
        (x, y),
        val_data=(vx, vy),
        config=iter_config,
        tracker=tracker,
        emit_stdout=True,
    )
    if loop_result["stopped_by_signal"]:
        raise SystemExit(128 + signal.SIGTERM)
    del x, y
    gc.collect()
    metrics = ClassificationMetrics(len(classes), threshold)
    metrics.add(vy, model.predict(vx))
    del vx, vy
    gc.collect()
    summary, _ = metrics.metrics()
    payload, extension = model.payload()
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = write_state(model_dir, f"model.{extension}", payload, format=extension)
    env = environment()
    metadata = {
        "dataset": dataset,
        "model": model.name,
        "model_params": model.params,
        "hyperparameters": iter_config.to_hyperparameters_dict(),
        "training": {
            "epochs_run": loop_result["epochs_run"],
            "best_epoch": loop_result["best_epoch"],
            "early_stopped": loop_result["early_stopped"],
            "best_validation_loss": loop_result["best_validation_loss"],
            "patience_consumed": loop_result["patience_consumed"],
        },
        "features": manifest["features"],
        "classes": classes,
        "dataset_version": manifest["sha256"],
        "state_sha256": checkpoint["sha256"],
        "payload": checkpoint["file"],
        "native_path": checkpoint["file"],
        "code_sha256": env["code_sha256"],
        "decision_threshold": threshold if len(classes) == 2 else None,
        "output": "class probabilities in classes order",
        "splits": {
            "train": [train_start, train_stop],
            "validation": [val_start, val_stop],
            "holdout_start_excluded": holdout_start,
        },
    }
    atomic_write(model_dir / "model-metadata.json", encoded(metadata))
    report = {
        "version": "sagemaker-native-training-v1", "complete": True, "dataset": dataset, "scope": scope,
        "validation": summary, "metadata": metadata, "training": dict(loop_result), "environment": env,
        "training_history": [epoch_fit(
            loop_result["history"], action="initial", start_seq=train_start, end_seq=val_stop,
            train_samples=train_stop - train_start, validation_samples=val_stop - val_start,
            train_seconds=loop_result["training_duration_seconds"], checkpoint=checkpoint, deployed_version=0,
            summary={**metadata["training"], "selection_scope": loop_result["selection_scope"],
                     "selection_metric": loop_result["selection_metric"]},
        )],
    }
    report["artifacts"] = write_training_artifacts(model_dir, report)
    atomic_write(model_dir / "report.json", encoded(report))
    print(f"validation:log_loss={summary['log_loss']};", flush=True)
    print(f"validation:accuracy={summary['accuracy']};", flush=True)
    print(f"validation:f1_macro={summary['f1_macro']};", flush=True)
    output = PREFIX / "output"
    if output.is_dir():
        metrics = {"validation": summary, "training": metadata["training"]}
        atomic_write(output / "metrics.json", encoded(metrics))
    if finish_tracker and tracker is not None and hasattr(tracker, "finish"):
        tracker.finish("Completed")
    return report


def process(input_dir=None, output_dir=None, *, tracker=None) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    input_dir = Path(input_dir or PREFIX / "processing" / "input")
    output_dir = Path(output_dir or PREFIX / "processing" / "output")
    profile = {}
    processed_rows = 0
    step = 0
    for parquet in sorted(input_dir.glob("*.parquet")):
        months, features = {}, {}
        with pq.ParquetFile(parquet) as source:
            batch_size = 4096 if tracker is not None else 262_144
            for batch in source.iter_batches(batch_size=batch_size, use_threads=False):
                month = (batch["event_time"].to_numpy() // (30 * 86_400_000)).astype(np.int64)
                labels = batch["label"].to_numpy()
                pairs, counts = np.unique(np.column_stack((month, labels)), axis=0, return_counts=True)
                for (m, label), count in zip(pairs, counts, strict=True):
                    bucket = months.setdefault(str(m), {})
                    bucket[str(label)] = bucket.get(str(label), 0) + int(count)
                for name in batch.schema.names[2:]:
                    values = batch[name].to_numpy(zero_copy_only=False).astype(np.float64)
                    finite = values[np.isfinite(values)]
                    current = features.setdefault(name, {"min": None, "max": None, "nulls": 0})
                    if len(finite):
                        minimum, maximum = float(finite.min()), float(finite.max())
                        current["min"] = minimum if current["min"] is None else min(current["min"], minimum)
                        current["max"] = maximum if current["max"] is None else max(current["max"], maximum)
                    current["nulls"] += int(batch[name].null_count)
                if tracker is not None:
                    step += 1
                    processed_rows += batch.num_rows
                    tracker.log_epoch_metrics(
                        {"processing_rows": processed_rows, "processing_features": len(features)}, step=step
                    )
            profile[parquet.name] = {
                "rows": source.metadata.num_rows,
                "class_by_month": months,
                "features": features,
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False))
    return profile


def native_pipeline_job(command):
    """Profile and stage verified inputs, or run native training with per-job MLflow tracking."""
    import hashlib
    import shutil

    from continual_bench.datasets.zia import DATASETS
    from continual_bench.infrastructure.aws.native_tracking import track_native_job

    uri = os.environ["MLFLOW_TRACKING_URI"]
    job_name = _training_job_name() if command == "train" else os.environ.get("BENCH_PROCESSING_JOB_NAME")
    if not job_name:
        raise RuntimeError("Missing SageMaker job name")
    execution = os.environ.get("BENCH_NATIVE_EXECUTION_ID", job_name)
    stage = os.environ.get("BENCH_NATIVE_STAGE", "training" if command == "train" else "processing")

    def stopped(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stopped)
    if command == "train":
        params = hyperparameters()
        scope = str(params.get("scope", "smoke"))
        model_dir = PREFIX / "model"
        manifest = _training_manifest(PREFIX / "input" / "data" / "train", scope)
        dataset = manifest["dataset"]
        with track_native_job(uri, execution, stage, job_name, dataset=dataset, scope=scope,
                              model_name=DATASETS[dataset]["model"]) as tracker:
            tracker.log_hyperparameters(params)
            result = train(model_dir=model_dir, params=params, tracker=tracker, finish_tracker=False)
            tracker.publish_training_evidence(result, model_dir)
        return result["validation"]

    scope = os.environ["BENCH_NATIVE_SCOPE"]
    directory = PREFIX / "processing"
    data_path = directory / "input" / "data" / "data.parquet"
    manifest_path = directory / "input" / "manifest" / f"{scope}.json"
    manifest = _training_manifest(manifest_path.parent, scope)
    dataset = manifest["dataset"]
    with data_path.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    if dataset not in DATASETS or manifest["sha256"] != digest:
        raise ValueError("Prepared snapshot manifest does not match the pipeline input")
    with track_native_job(uri, execution, stage, job_name, dataset=dataset, scope=scope) as tracker:
        tracker.log_hyperparameters({"scope": scope, "dataset_sha256": digest})
        report = process(data_path.parent, directory / "output", tracker=tracker)
        if report[data_path.name]["rows"] != manifest["rows"]:
            raise ValueError("Prepared snapshot row count does not match its manifest")
        channel = directory / "channel"
        channel.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, channel / manifest_path.name)
        shutil.copyfile(data_path, channel / f"{scope}.parquet")
        tracker.publish_training_evidence({"profile": report}, directory / "output")
    return report


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
    pipeline = command in {"train", "process"} and os.environ.get("BENCH_NATIVE_EXECUTION_ID")
    managed_training = command == "train" and os.environ.get("MLFLOW_TRACKING_URI") and _training_job_name()
    if pipeline or managed_training:
        result = native_pipeline_job(command)
        if command == "train":
            print(json.dumps(result, default=float))
    elif command == "train":
        print(json.dumps(train()["validation"], default=float))
    elif command == "process":
        process()
    elif command == "serve":
        serve()
    else:
        raise SystemExit(f"unknown SageMaker command: {command}")


if __name__ == "__main__":
    main()
