"""Content-addressed dataset catalog and versioned, loadable MLflow model artifacts."""

import hashlib
import importlib.metadata
import json
import shutil
import zipfile
from pathlib import Path

from .local import atomic_write, publication_lock


def strategy_name(config):
    policy = config["policy"]["name"]
    if policy not in ("none", "online") and config["policy"]["action"] != "reset_retrain":
        policy += "-" + config["policy"]["action"]
    return f"continual-bench/policy-{policy}/detector-{config['detector']['name']}"


def model_family_name(config):
    parts = [config["dataset"]["name"]]
    # Preserve the historical logistic families; nonlinear models have distinct registries.
    if config["model"]["name"] != "logistic":
        parts.append(config["model"]["name"])
    parts += [config["mode"], config["policy"]["name"], config["detector"]["name"]]
    return "continual-bench-" + "-".join(parts)


def put_immutable(objects, key, path):
    from botocore.exceptions import ClientError

    path = Path(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        existing = objects.client.head_object(Bucket=objects.bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
    else:
        if existing.get("Metadata", {}).get("sha256") != sha:
            raise ValueError("immutable catalog key already contains different content")
        return f"s3://{objects.bucket}/{key}"
    objects.client.upload_file(str(path), objects.bucket, key, ExtraArgs={"Metadata": {"sha256": sha}})
    return f"s3://{objects.bucket}/{key}"


def publish_catalog(objects, report, directory):
    directory = Path(directory)
    dataset = report["dataset"]
    version = dataset["sha256"]
    first = json.loads((directory / "events.jsonl").open().readline())
    schema = {key: type(value).__name__ for key, value in first["x"].items()}
    source = {
        k: v
        for k, v in dataset.items()
        if k not in ("path", "raw_path", "run_key", "protocol", "config", "requested_seed")
    }
    # A real source explicitly ignores seed; do not turn run-only metadata into a dataset version.
    if source.get("seed_ignored"):
        source.pop("seed", None)
    raw_uri = None
    if dataset.get("raw_sha256"):
        raw = Path(dataset["raw_path"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != dataset["raw_sha256"]:
            raise ValueError("raw dataset changed after admission")
        with publication_lock(directory.parent, "raw-" + dataset["raw_sha256"]):
            raw_uri = put_immutable(objects, f"datasets/raw/{dataset['raw_sha256']}/source.arff", raw)
    catalog = {
        "dataset_version": version,
        "events_sha256": version,
        "source": source,
        "raw_source_uri": raw_uri,
        "feature_schema": schema,
        "selection": {"kind": "published_prefix", "seq_range": [0, dataset["samples"]]},
    }
    path = directory / "dataset-catalog.json"
    atomic_write(path, json.dumps(catalog, sort_keys=True, indent=2).encode())
    with publication_lock(directory.parent, "dataset-" + version):
        events_uri = put_immutable(objects, f"datasets/{version}/events.jsonl", directory / "events.jsonl")
        catalog_uri = put_immutable(objects, f"datasets/{version}/catalog.json", path)
    versions = report["model_versions"]
    source_zip = directory / "training-source.zip"
    source_root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_root.rglob("*.py")):
            entry = zipfile.ZipInfo(str(path.relative_to(source_root)))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    with publication_lock(directory.parent, "code-" + report["environment"]["code_sha256"]):
        code_uri = put_immutable(
            objects, f"code/{report['environment']['code_sha256']}/source.zip", source_zip
        )
    with publication_lock(directory.parent, "models-" + versions["journal_sha256"]):
        model_uri = put_immutable(
            objects, f"model-states/{versions['journal_sha256']}/models.bin", directory / "models.bin"
        )
        index_uri = put_immutable(
            objects, f"model-indexes/{versions['index_sha256']}/index.json", directory / "model-versions.json"
        )
    return {
        "dataset_version": version,
        "dataset_uri": events_uri,
        "dataset_catalog_uri": catalog_uri,
        "model_states_uri": model_uri,
        "model_index_uri": index_uri,
        "model_version_id": versions["final_version_id"],
        "model_state_sha256": versions["state_sha256"],
        "model_versions_count": versions["count"],
        "training_source_uri": code_uri,
    }


def prepare_mlflow_model(directory):
    import mlflow.pyfunc
    import numpy as np
    from mlflow.models import infer_signature

    directory = Path(directory)
    target = directory / "mlflow-model"
    metadata = json.loads((directory / "model-metadata.json").read_text())
    ready = target / ".ready.json"
    if target.exists():
        if not ready.exists():
            shutil.rmtree(target)  # Only this run's incomplete, self-produced staging folder.
        else:
            if json.loads(ready.read_text())["state_sha256"] != metadata["state_sha256"]:
                raise ValueError("staged MLflow model differs from the versioned state")
            return target
    from .model_loader import RiverPredictor, inference_frame

    predictor = RiverPredictor(directory)
    example = inference_frame(metadata["inference_probe"]["features"])
    signature = infer_signature(example, np.asarray([metadata["inference_probe"]["probability"]]))
    data = directory / "model-data"
    data.mkdir(exist_ok=True)
    for name in ("model.pkl", "model-metadata.json"):
        shutil.copy2(directory / name, data / name)
    dependencies = [
        f"{p}=={importlib.metadata.version(p)}"
        for p in ("river", "numpy", "scipy", "mlflow-skinny", "pandas")
    ]
    # Only inference dependencies go into each model; full training source is versioned once in the catalog.
    code = directory / "inference-code" / "continual_bench"
    for relative in (
        "__init__.py",
        "adapters/__init__.py",
        "adapters/river.py",
        "adapters/nonlinear.py",
        "adapters/model_loader.py",
        "domain/__init__.py",
        "domain/contracts.py",
    ):
        target_file = code / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / relative, target_file)
    mlflow.pyfunc.save_model(
        path=str(target),
        loader_module="continual_bench.adapters.model_loader",
        data_path=str(data),
        code_paths=[str(code)],
        pip_requirements=dependencies,
        signature=signature,
        input_example=example,
        metadata={
            k: metadata[k] for k in ("dataset_version", "final_version_id", "state_sha256", "code_sha256")
        },
    )
    loaded = mlflow.pyfunc.load_model(str(target))
    if loaded.predict(example).tolist() != predictor.predict(example).tolist():
        raise ValueError("MLflow flavor changed model inference")
    # The SDK writes MLmodel before code/dependencies. Its existence is not a commit marker.
    atomic_write(ready, json.dumps({"state_sha256": metadata["state_sha256"]}).encode())
    return target


def register_final_model(client, run, config, report, directory, catalog):
    name = model_family_name(config)
    with publication_lock(Path(directory).parent, "registry-" + name):
        from mlflow.exceptions import MlflowException

        try:
            client.get_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            client.create_registered_model(name, tags={"dataset": config["dataset"]["name"]})
        matches = [v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == run.info.run_id]
        if len(matches) > 1:
            raise RuntimeError("duplicate model registry versions for one benchmark run")
        if matches:
            version = matches[0]
        else:
            model = prepare_mlflow_model(directory)
            client.log_artifacts(run.info.run_id, str(model), artifact_path="model")
            metadata = report["model_versions"]
            version = client.create_model_version(
                name,
                source=run.info.artifact_uri.rstrip("/") + "/model",
                run_id=run.info.run_id,
                tags={
                    "run_key": report["run_key"],
                    "dataset_version": report["dataset"]["sha256"],
                    "model_version_id": metadata["final_version_id"],
                    "model_sha256": metadata["state_sha256"],
                    "code_sha256": report["environment"]["code_sha256"],
                    "model": config["model"]["name"],
                    "model_params": json.dumps(config["model"]["params"], sort_keys=True),
                    **{
                        key: catalog[key]
                        for key in (
                            "dataset_uri",
                            "dataset_catalog_uri",
                            "training_source_uri",
                            "model_states_uri",
                            "model_index_uri",
                        )
                    },
                },
            )
    return {
        "registered_model": name,
        "registered_model_version": str(version.version),
        "model_uri": f"models:/{name}/{version.version}",
    }
