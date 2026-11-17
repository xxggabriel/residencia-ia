from importlib.util import find_spec

from continual_bench.infrastructure.aws.mlflow_tracking import MlflowRestError, SageMakerMlflowTracker
from continual_bench.infrastructure.object_store import LocalObjectStore


def _spot_tracker(unit_id, attempt, **metadata):
    tracker = SageMakerMlflowTracker(
        tracking_uri="http://mlflow.example", dataset="social", policy="frozen", model_name="logistic",
        run_id=unit_id, campaign_id="spot-acceptance", git_sha="abcdef1234",
    )
    tracker.bind_spot_attempt(
        f"i-{attempt}-{attempt}", instance_id=f"i-{attempt}", attempt=attempt, **metadata,
    )
    return tracker


def test_spot_reuses_parent_fits_versions_and_artifacts_after_lost_responses(tmp_path, monkeypatch):
    from google.protobuf.json_format import MessageToDict
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag
    from mlflow.exceptions import MlflowException

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    if find_spec("sqlalchemy") and find_spec("alembic"):
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    first = _spot_tracker("unit-id", 1, log_stream="logs/i-1")
    exp = client.create_experiment(first.experiment_name, artifact_location="s3://tracking/artifacts")
    fault = None

    def encode(entity):
        return MessageToDict(entity.to_proto(), preserving_proto_field_name=True)

    def remote(method, path, *, payload=None, params=None):
        nonlocal fault
        request = payload or params or {}
        api = path.removeprefix("/api/2.0/mlflow/")
        if api == "experiments/get-by-name":
            return {"experiment": {"experiment_id": exp}}
        if api == "runs/create":
            tags = {tag["key"]: tag["value"] for tag in request["tags"]}
            return {"run": encode(client.create_run(exp, tags=tags, run_name=request["run_name"]))}
        if api == "runs/search":
            runs = client.search_runs([exp], filter_string=request["filter"])
            return {"runs": [encode(run) for run in runs]}
        if api == "runs/get":
            return {"run": encode(client.get_run(request["run_id"]))}
        if api == "runs/update":
            if request["status"] == "RUNNING":
                client.update_run(request["run_id"], status="RUNNING")
            else:
                client.set_terminated(request["run_id"], request["status"])
            return {}
        if api == "metrics/get-history":
            return {"metrics": [encode(m) for m in client.get_metric_history(
                request["run_id"], request["metric_key"],
            )]}
        if api == "runs/log-batch":
            client.log_batch(
                request["run_id"], metrics=[Metric(**m) for m in request.get("metrics", [])],
                params=[Param(**p) for p in request.get("params", [])],
                tags=[RunTag(**t) for t in request.get("tags", [])], synchronous=True,
            )
            if fault == "progress" and any(m["key"].startswith("progress/") for m in request["metrics"]):
                fault = None
                raise TimeoutError("response lost after logging progress")
            return {}
        if api == "registered-models/get":
            try:
                return {"registered_model": encode(client.get_registered_model(request["name"]))}
            except MlflowException as exc:
                raise MlflowRestError(404, exc.error_code, str(exc)) from exc
        if api == "registered-models/create":
            tags = {tag["key"]: tag["value"] for tag in request["tags"]}
            return {"registered_model": encode(client.create_registered_model(
                request["name"], tags=tags, description=request["description"],
            ))}
        if api == "model-versions/search":
            return {"model_versions": [encode(m) for m in client.search_model_versions(request["filter"])]}
        if api == "model-versions/create":
            tags = {tag["key"]: tag["value"] for tag in request.pop("tags")}
            version = client.create_model_version(**request, tags=tags)
            if fault == "family" and tags["ordinal"] == "final":
                fault = None
                raise TimeoutError("response lost after creating family model version")
            return {"model_version": encode(version)}
        if api == "registered-models/alias" and method == "POST":
            client.set_registered_model_alias(**request)
            return {}
        if api == "registered-models/alias" and method == "GET":
            try:
                return {"model_version": encode(client.get_model_version_by_alias(**request))}
            except MlflowException as exc:
                raise MlflowRestError(404, exc.error_code, str(exc)) from exc
        raise AssertionError(f"Unexpected MLflow request {method} {api}")

    monkeypatch.setattr(first, "_request", remote)
    assert first.start_live()
    root_id = first.mlflow_run_id
    fault = "progress"
    first.progress(12, {"rows": 12, "elapsed": 2})
    assert fault is None

    checkpoint = "s3://tracking/checkpoints/unit/manifest.json"
    second = _spot_tracker("unit-id", 2, resumed_checkpoint=checkpoint,
                           previous_attempt_key="i-1-1", recovery_reason="INSTANCE_LOST")
    monkeypatch.setattr(second, "_request", remote)
    assert second.start_live() and second.mlflow_run_id == root_id
    second.progress(12, {"rows": 12, "elapsed": 3})
    second.progress(20, {"rows": 20})

    store = LocalObjectStore(tmp_path / "store")
    output = tmp_path / "output"
    (output / "model").mkdir(parents=True)
    (output / "model" / "model.tar.gz").write_bytes(b"model")
    (output / "report.json").write_text("{}")
    report = {
        "metrics": {"accuracy": 0.8},
        "training_history": [{
            "fit_id": "fit-000001", "action": "initial", "start_seq": 0, "end_seq": 12,
            "steps": [{"step": 1, "global_step": 1, "kind": "epoch", "metrics": {"loss": 0.5}}],
            "checkpoint": {"file": "model/model.tar.gz", "sha256": "a" * 64},
            "summary": {},
        }],
    }
    fault = "family"
    failed = second.publish_run_report(report, output_dir=output, store=store)
    assert fault is None and not failed["published"]
    assert client.get_run(root_id).data.tags["benchmark_complete"] == "false"
    assert second.record_spot_attempt("TRACKING_FAILED", reason="response lost", checkpoint=checkpoint)
    [fit_before] = client.search_runs([exp], filter_string=f"tags.`mlflow.parentRunId` = '{root_id}'")

    third = _spot_tracker("unit-id", 3, previous_attempt_key="i-2-2", recovery_reason="TRACKING_FAILED")
    monkeypatch.setattr(third, "_request", remote)
    assert third.start_live() and third.mlflow_run_id == root_id
    result = third.publish_run_report(report, output_dir=output, store=store)
    assert result["published"] and result["child_runs"][0]["run_id"] == fit_before.info.run_id
    assert third.record_spot_attempt("SUCCEEDED")

    parent = client.get_run(root_id)
    assert parent.info.status == "FINISHED" and parent.data.tags["benchmark_complete"] == "true"
    assert parent.data.tags["cb-spot-attempt-i-1-1-instance-id"] == "i-1"
    assert parent.data.tags["cb-spot-attempt-i-1-1-recovery-cause"] == "INSTANCE_LOST"
    assert parent.data.tags["cb-spot-attempt-i-2-2-resumed-checkpoint"] == checkpoint
    assert len(client.search_runs([exp])) == 2
    assert [(p.step, p.value) for p in client.get_metric_history(root_id, "progress/rows")] == [
        (12, 12), (20, 20),
    ]
    assert [p.value for p in client.get_metric_history(root_id, "progress/elapsed")] == [2]
    assert [p.value for p in client.get_metric_history(root_id, "summary/accuracy")] == [0.8]
    assert [p.value for p in client.get_metric_history(fit_before.info.run_id, "loss")] == [0.5]
    assert fit_before.data.tags["cb-fit-checkpoint-uri"] == f"{parent.info.artifact_uri}/model/model.tar.gz"
    assert len(client.search_model_versions()) == 2
    assert len(list(store.list("artifacts/"))) == 2
