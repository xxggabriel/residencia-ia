"""Tests for decoupled cloud/container/CLI DI contracts and native SageMaker integrations."""

import json
from pathlib import Path
from runpy import run_path
from unittest.mock import Mock

import boto3
from botocore.stub import ANY, Stubber

from continual_bench.application.cloud import layout, planning
from continual_bench.application.cloud.contracts import BenchmarkWorkload, CampaignRequest
from continual_bench.application.cloud.dependencies import (
    build_cli_runtime,
    build_container_runtime,
    resolve_unit_dataset,
    resolve_unit_model_name,
    resolve_unit_policy,
)
from continual_bench.application.cloud.fixtures import publish_fixture_release
from continual_bench.application.cloud.worker import run_unit
from continual_bench.infrastructure.aws.experiments import SageMakerExperimentsTracker
from continual_bench.infrastructure.aws.model_registry import SageMakerModelRegistry
from continual_bench.infrastructure.object_store import LocalObjectStore, get_json
from continual_bench.orchestration import sagemaker


def test_benchmark_workload_plans_and_executes_via_cli_runtime(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id="generic-sea-test",
        matrix=BenchmarkWorkload(
            evaluator="binary",
            datasets=("sea",),
            models=("logistic",),
            policies=("drift",),
            detectors=("adwin",),
            modes=("incremental",),
            samples=300,
            seeds=(42,),
        ),
    )
    plan = planning.build_plan(request, release, project_root=".")
    published = planning.publish_plan(store, plan)
    assert published["units"] == 1
    item = published["items"][0]

    runtime = build_cli_runtime(tmp_path / "store")
    assert runtime.mode == "cli"
    code = run_unit(
        runtime.store,
        item["unit_uri"],
        tmp_path / "work",
        runtime=runtime,
    )
    assert code == 0
    final = get_json(store, layout.final_key("generic-sea-test", item["run_id"]))
    assert final["kind"] == "matrix_case"
    assert final["labels"]["dataset"] == "sea"
    assert final["labels"]["model"] == "logistic"


def test_container_runtime_resolves_unit_metadata_without_zia(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    release = publish_fixture_release(store)
    request = CampaignRequest(
        campaign_id="container-generic",
        matrix=BenchmarkWorkload(
            evaluator="binary",
            datasets=("agrawal",),
            models=("hoeffding_tree",),
            policies=("periodic",),
            detectors=("none",),
            modes=("incremental",),
            samples=300,
        ),
    )
    plan = planning.build_plan(request, release, project_root=".")
    published = planning.publish_plan(store, plan)
    unit_data = get_json(store, plan["units"][0]["unit_key"])
    from continual_bench.application.cloud.contracts import RunUnit

    unit = RunUnit(**unit_data)
    assert resolve_unit_dataset(unit) == "agrawal"
    assert resolve_unit_model_name(unit) == "hoeffding_tree"
    assert resolve_unit_policy(unit) == "periodic"

    runtime = build_container_runtime(local_root=tmp_path / "store")
    assert runtime.mode == "container"
    assert set(runtime.workloads.kinds()) >= {"zia", "matrix_case"}
    assert len(published["items"]) == 1


def test_sagemaker_train_executes_generic_registered_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(sagemaker, "PREFIX", tmp_path)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    result = sagemaker.train(
        channel=tmp_path / "input" / "data" / "train",
        model_dir=tmp_path / "model",
        params={
            "evaluator": "binary",
            "dataset": "sea",
            "model": "logistic",
            "policy": "drift",
            "detector": "adwin",
            "mode": "incremental",
            "samples": 300,
            "seed": 7,
        },
    )
    assert result["complete"] is True
    assert result["metadata"]["dataset"] == "sea"
    assert result["metadata"]["model"] == "logistic"
    assert (tmp_path / "model" / "model-metadata.json").is_file()
    assert (tmp_path / "output" / "metrics.json").is_file()
    metrics_doc = json.loads((tmp_path / "output" / "metrics.json").read_text())
    assert "validation" in metrics_doc and metrics_doc["validation"]


def test_native_pipeline_includes_experiments_and_register_model_step(monkeypatch):
    scripts = Path(__file__).parents[4] / "scripts" / "aws"
    monkeypatch.syspath_prepend(str(scripts))
    make = run_path(str(scripts / "native_pipeline.py"))["pipeline_definition"]
    definition = make(
        bucket="b",
        role="role",
        runtime_image="runtime",
        ml_image="ml",
        git_sha="a" * 40,
        key_arn="key",
        mlflow_arn="app",
    )
    assert definition["PipelineExperimentConfig"] == {
        "ExperimentName": {"Get": "Parameters.ExperimentName"},
        "TrialName": {"Get": "Execution.PipelineExecutionId"},
    }
    param_names = {p["Name"] for p in definition["Parameters"]}
    assert {
        "DatasetName",
        "ModelName",
        "PolicyName",
        "Evaluator",
        "ExperimentName",
        "ModelPackageGroupName",
    } <= param_names

    steps = {step["Name"]: step for step in definition["Steps"]}
    assert "RegisterBaselineModel" in steps
    reg = steps["RegisterBaselineModel"]
    assert reg["Type"] == "RegisterModel"
    assert reg["DependsOn"] == ["TrainBaseline"]
    assert reg["Arguments"]["ModelMetrics"]["ModelQuality"]["Statistics"]["ContentType"] == "application/json"


def test_sagemaker_model_registry_attaches_model_metrics():
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = []
    client.create_model_package.return_value = {"ModelPackageArn": "arn:aws:sagemaker:us-east-1:1:pkg/1"}

    registry = SageMakerModelRegistry(client=client)
    res = registry.register(
        group="cb-models",
        run_id="run123",
        image="123.dkr.ecr.us-east-1.amazonaws.com/ml:latest",
        model_data_uri="s3://b/model.tar.gz",
        metadata={"dataset": "sea", "model": "logistic"},
        model_metrics_uri="s3://b/metrics.json",
    )
    assert res["created"] is True
    call_kwargs = client.create_model_package.call_args.kwargs
    assert call_kwargs["ModelMetrics"] == {
        "ModelQuality": {"Statistics": {"ContentType": "application/json", "S3Uri": "s3://b/metrics.json"}}
    }


def test_sagemaker_experiments_records_artifacts_and_lineage():
    sm = boto3.client("sagemaker", region_name="us-east-1", aws_access_key_id="t", aws_secret_access_key="t")
    metrics = boto3.client(
        "sagemaker-metrics", region_name="us-east-1", aws_access_key_id="t", aws_secret_access_key="t"
    )
    tracker = SageMakerExperimentsTracker(
        project="cb",
        dataset="sea",
        policy="drift",
        model_name="logistic",
        run_id="r1",
        client=sm,
        metrics_client=metrics,
        input_artifacts={"dataset": {"MediaType": "application/x-parquet", "Value": "s3://b/sea.parquet"}},
        output_artifacts={"model": {"MediaType": "application/x-tar", "Value": "s3://b/model.tar.gz"}},
    )
    tc_arn = "arn:aws:sagemaker:us-east-1:123456789012:experiment-trial-component/tc1"
    ds_arn = "arn:aws:sagemaker:us-east-1:123456789012:artifact/ds1"
    mdl_arn = "arn:aws:sagemaker:us-east-1:123456789012:artifact/mdl1"
    tracker.trial_component_arn = tc_arn

    with Stubber(sm) as stub:
        stub.add_response(
            "create_artifact",
            {"ArtifactArn": ds_arn},
            {
                "ArtifactName": f"{tracker.trial_component_name}-dataset",
                "Source": {"SourceUri": "s3://b/sea.parquet"},
                "ArtifactType": "DataSet",
                "Properties": {"dataset": "sea", "scope": "smoke"},
            },
        )
        stub.add_response(
            "add_association",
            {"SourceArn": ds_arn, "DestinationArn": tc_arn},
            {"SourceArn": ds_arn, "DestinationArn": tc_arn, "AssociationType": "ContributedTo"},
        )
        stub.add_response(
            "create_artifact",
            {"ArtifactArn": mdl_arn},
            {
                "ArtifactName": f"{tracker.trial_component_name}-model",
                "Source": {"SourceUri": "s3://b/model.tar.gz"},
                "ArtifactType": "Model",
                "Properties": {"model": "logistic", "policy": "drift"},
            },
        )
        stub.add_response(
            "add_association",
            {"SourceArn": tc_arn, "DestinationArn": mdl_arn},
            {"SourceArn": tc_arn, "DestinationArn": mdl_arn, "AssociationType": "Produced"},
        )
        linked = tracker.register_lineage(
            dataset_uri="s3://b/sea.parquet",
            model_uri="s3://b/model.tar.gz",
        )
    assert linked == {"dataset_artifact_arn": ds_arn, "model_artifact_arn": mdl_arn}
    assert ANY is not None
