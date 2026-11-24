"""Publish standard Zia run artifacts from retained reports and native weights."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from continual_bench.datasets.streams import load_stream_manifest as load_manifest
from continual_bench.datasets.streams import sha256_file
from continual_bench.evaluation.environment import encoded
from continual_bench.evaluation.zia_artifacts import tracking_report, write_zia_artifacts
from continual_bench.evaluation.zia_predictions import INDEX, PredictionArchive, prediction_manifest
from continual_bench.infrastructure.artifact_policy import metrics_only, verify_metrics_only_files
from continual_bench.infrastructure.metric_tracking import log_metric_points, publish_tracking
from continual_bench.infrastructure.run_artifacts import publish_run_artifacts
from continual_bench.infrastructure.storage import atomic_write, publication_lock
from continual_bench.infrastructure.zia_model_bundle import (
    PYFUNC_LOADER,
    pyfunc_code_paths,
    pyfunc_dependencies,
)
from continual_bench.infrastructure.zia_tracking import publish_run_metrics
from continual_bench.models.zia import ZiaPredictor, make_zia_model


def dataset_source_uri(source):
    """Provenance URI of a prepared stream: the pinned Hugging Face file or the catalog entry."""
    if "repo" not in source:
        return f"catalog://{source.get('catalog', 'unknown')}"
    return (
        f"https://huggingface.co/datasets/ziadatalabs/{source['repo']}"
        f"/resolve/{source['revision']}/{source['file']}"
    )


def run_documents(report, adapted, base_uri):
    """Model metadata, catalog links and JSON documents of a completed Zia run (sink-independent)."""
    versions = report["model_versions"]
    manifest, env = adapted["dataset"], adapted["environment"]
    source = manifest["source"]
    final = versions[-1]
    only_metrics = metrics_only(report)
    metadata = (
        None
        if only_metrics
        else {
            "dataset": report["dataset"],
            "model": report["model"],
            "model_params": report["model_params"],
            "features": manifest["features"],
            "classes": source["classes"],
            "dataset_version": manifest["sha256"],
            "code_sha256": env["code_sha256"],
            "state_sha256": final["sha256"],
            "final_version_id": final["sha256"],
            "native_path": final["file"],
            "payload": Path(final["file"]).name,
            "output": "class probabilities in classes order",
        }
    )
    catalog = {
        "dataset_version": manifest["sha256"],
        "source": manifest,
        "feature_schema": {name: "numeric" for name in manifest["features"]},
        "selection": {"kind": report["scope"], "seq_range": [0, manifest["rows"]]},
    }
    links = {
        "dataset_version": manifest["sha256"],
        "dataset_source_uri": dataset_source_uri(source),
        "dataset_catalog_uri": f"{base_uri}/dataset-catalog.json",
        "model_states_uri": None if only_metrics else f"{base_uri}/models",
        "model_index_uri": f"{base_uri}/model-versions.json",
        "model_state_sha256": final["sha256"],
        "model_versions_count": len(versions),
    }
    if "training_history" in report:
        links["training_history_uri"] = f"{base_uri}/training-history.json"
    files = {
        "manifest.json": manifest,
        "environment.json": env,
        **({} if metadata is None else {"model-metadata.json": metadata}),
        "model-versions.json": {
            "versions": versions,
            "final_version_id": final["sha256"],
            "count": len(versions),
            "storage_bytes": 0 if only_metrics else sum(v["bytes"] for v in versions),
            **(
                {
                    "artifact_mode": "metrics_only",
                    "logical_serialized_bytes": sum(v["bytes"] for v in versions),
                }
                if only_metrics
                else {}
            ),
        },
        "dataset-catalog.json": catalog,
        "catalog-links.json": links,
        "config.json": {
            "dataset": report["dataset"],
            "scope": report["scope"],
            "policy": report["policy"],
            "model": report["model"],
            "model_params": report["model_params"],
            "tracking": adapted["tracking"]["settings"],
            "contract": report["contract"],
        },
    }
    return metadata, links, files


def replay_predictions(directory, report, root):
    if metrics_only(report):
        raise ValueError("metrics_only predictions must be retained, not replayed from model checkpoints")
    from continual_bench.evaluation.zia import batches

    contract = report["contract"]
    manifest = contract["input"]
    path, local_manifest = load_manifest(root, report["dataset"], report["scope"])
    if local_manifest != manifest or sha256_file(path) != manifest["sha256"]:
        raise ValueError("cannot replay predictions on a different prepared dataset")
    classes = manifest["source"]["classes"]
    model = make_zia_model(report["model"], len(classes), dataset=report.get("dataset"))
    versions = iter(report["model_versions"])
    active = next(versions)
    pending = next(versions, None)
    model.load((directory / active["file"]).read_bytes())
    blocks = batches(path, contract["block_rows"], contract["holdout_start"])
    warmup, _, _, _ = next(blocks)
    start = len(warmup)
    with PredictionArchive(
        directory,
        classes,
        manifest["sha256"],
        "replayed_from_checkpoints",
        threshold=contract.get("decision_threshold"),
    ) as archive:
        for (times, features, labels, holdout), window in zip(blocks, report["windows"], strict=True):
            while pending is not None and pending["end_seq"] <= start:
                active, pending = pending, next(versions, None)
                model.load((directory / active["file"]).read_bytes())
            probabilities = model.predict(features)
            errors = archive.add(start, times, labels, probabilities, holdout, active["end_seq"])
            start += len(labels)
            if (
                start != window["end_seq"]
                or errors != window["errors"]
                or holdout != (window["scope"] == "holdout")
            ):
                raise ValueError("prediction replay differs from recorded window")


def publish_predictions(client, run, directory, report, root):
    evidence = prediction_manifest(directory, report)
    if evidence is None:
        replay_predictions(directory, report, root)
        evidence = prediction_manifest(directory, report)
    index = directory / INDEX
    digest = sha256_file(index)
    if run.data.tags.get("zia_predictions_sha256") != digest:
        for part in evidence["parts"]:
            path = directory / "predictions" / part["file"]
            if path.stat().st_size != part["bytes"] or sha256_file(path) != part["sha256"]:
                raise ValueError("prediction part differs from its manifest")
            client.log_artifact(run.info.run_id, str(path), artifact_path="predictions")
        client.log_artifact(run.info.run_id, str(index), artifact_path="predictions")
        client.set_tag(run.info.run_id, "zia_predictions_sha256", digest)
    return evidence


def publish_model(client, run, directory, report, metadata, root):
    import mlflow.pyfunc
    import numpy as np
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature
    from pyarrow.parquet import ParquetFile

    manifest = report["contract"]["input"]
    dataset = report["dataset"]
    name = f"continual-bench-zia-{dataset}-{report['model']}-{report['policy']}"
    if report["scope"] == "smoke":
        name += "-smoke"
    run_id = run.info.run_id
    with publication_lock(directory.parent, "registry-" + name):
        try:
            client.get_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            client.create_registered_model(name, tags={"dataset": dataset})
        versions = [v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == run_id]
        if len(versions) > 1:
            raise ValueError("duplicate model registry versions for one Zia run")
        if versions:
            return versions[0]
        parquet, local_manifest = load_manifest(root, dataset, report["scope"])
        if local_manifest != manifest:
            raise ValueError("inference probe input differs from the Zia run manifest")
        with ParquetFile(parquet) as reader:
            sample = next(reader.iter_batches(batch_size=8, columns=manifest["features"], use_threads=False))
        example = sample.to_pandas().astype("float64")
        with TemporaryDirectory(dir=directory, prefix=".model-") as temp:
            staging = Path(temp)
            data = staging / "state"
            data.mkdir()
            shutil.copy2(directory / "model-metadata.json", data)
            shutil.copy2(directory / metadata["native_path"], data / metadata["payload"])
            predictor = ZiaPredictor(data)
            prediction = predictor.predict(example)
            target = staging / "model"
            mlflow.pyfunc.save_model(
                path=str(target),
                loader_module=PYFUNC_LOADER,
                data_path=str(data),
                code_paths=pyfunc_code_paths(staging),
                pip_requirements=pyfunc_dependencies(report["model"]),
                signature=infer_signature(example, prediction),
                input_example=example,
                metadata={
                    k: metadata[k] for k in ("dataset_version", "state_sha256", "code_sha256", "classes")
                },
            )
            np.testing.assert_array_equal(mlflow.pyfunc.load_model(str(target)).predict(example), prediction)
            client.log_artifacts(run_id, str(target), artifact_path="model")
        return client.create_model_version(
            name,
            source=run.info.artifact_uri.rstrip("/") + "/model",
            run_id=run_id,
            tags={
                "dataset_version": metadata["dataset_version"],
                "model_sha256": metadata["state_sha256"],
                "code_sha256": metadata["code_sha256"],
            },
        )


def publish_artifacts(client, directory, run_id, report, root, *, derived=False, elapsed_seconds=None):
    directory = Path(directory)
    with publication_lock(directory.parent, f"zia-run-{run_id}"):
        return _publish_artifacts(client, directory, run_id, report, root, derived, elapsed_seconds)


def _publish_artifacts(client, directory, run_id, report, root, derived=False, elapsed_seconds=None):
    from mlflow.entities import RunTag

    directory = Path(directory)
    run = client.get_run(run_id)
    if run.data.tags.get("zia_run_key") != directory.name:
        raise ValueError("MLflow run does not match the retained Zia evidence")
    versions = report["model_versions"]
    only_metrics = metrics_only(report)
    if only_metrics:
        verify_metrics_only_files(directory)
        if any(version.get("file") is not None for version in versions):
            raise ValueError("metrics_only model version advertises persisted weights")
    for version in () if only_metrics else versions:
        path = directory / version["file"]
        if path.stat().st_size != version["bytes"] or sha256_file(path) != version["sha256"]:
            raise ValueError("retained model file differs from the Zia report")
    history = directory / "training-history.json"
    if "training_history" in report:
        if json.loads(history.read_bytes()) != {"version": 1, "fits": report["training_history"]}:
            raise ValueError("retained training history differs from the Zia report")
    client.set_tag(run_id, "zia_artifacts_status", "publishing")
    evidence = publish_predictions(client, run, directory, report, root)
    adapted = tracking_report(report, evidence)
    if elapsed_seconds is not None:
        from continual_bench.infrastructure.metric_tracking import EXECUTION_SCOPE

        tracking = adapted["tracking"]
        tracking["metrics"]["pipeline_elapsed_seconds"] = elapsed_seconds
        tracking["availability"]["pipeline_elapsed_seconds"] = {
            "status": "available",
            "reason": None,
            "scope": EXECUTION_SCOPE,
        }
    write_zia_artifacts(directory, report, adapted)
    artifacts = publish_run_artifacts(client, run_id, directory, report, include_checkpoints=False)
    metadata, links, files = run_documents(report, adapted, run.info.artifact_uri.rstrip("/"))
    for name, value in files.items():
        path = directory / name
        atomic_write(path, encoded(value))
        client.log_artifact(run_id, str(path))
    version = None
    if not only_metrics:
        version = publish_model(client, run, directory, report, metadata, root)
        client.log_batch(
            run_id,
            synchronous=True,
            tags=[
                RunTag("registered_model", version.name),
                RunTag("registered_model_version", str(version.version)),
            ],
        )
    else:
        client.set_tag(run_id, "model_retention", "metrics_only_no_registered_model")
    log_metric_points(client, run_id, [("training_used", 0, report["metrics"]["training_used_post_warmup"])])
    tracking = publish_tracking(
        client,
        run_id,
        adapted,
        directory,
        derived=derived,
        **({"max_points_per_key": None} if only_metrics else {}),
    )
    training_rounds = artifacts.get("training_rounds", [])
    client.log_batch(
        run_id,
        tags=[
            RunTag("cb-training-round-count", str(len(training_rounds))),
            RunTag("zia_artifacts_status", "complete"),
        ],
        synchronous=True,
    )
    return {
        "run_artifacts": artifacts,
        "tracking": tracking,
        "training_rounds": training_rounds,
        "registered_model": None if version is None else version.name,
        "registered_model_version": None if version is None else str(version.version),
        **links,
    }


def publish(client, directory, run_id, experiment_id, report, root):
    directory = Path(directory)
    with publication_lock(directory.parent, f"zia-run-{run_id}"):
        client.set_tag(run_id, "benchmark_complete", "false")
        publish_run_metrics(client, directory, run_id, report)
        artifacts = _publish_artifacts(client, directory, run_id, report, root)
        client.set_tag(run_id, "benchmark_complete", "true")
        client.set_terminated(run_id, "FINISHED")
        receipt = {
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": experiment_id,
            "complete": True,
            **artifacts,
        }
        atomic_write(directory / "receipt.json", encoded(receipt))
        return receipt
