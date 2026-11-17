"""Render a native SageMaker Pipelines definition using the existing job profiles."""

import argparse
import json

from sagemaker_jobs import processing_request, profile, training_request, tuning_request


def get(path):
    return {"Get": path}


def param(name):
    return get(f"Parameters.{name}")


def join_path(*values):
    return {"Std:Join": {"On": "/", "Values": list(values)}}


def as_string(value):
    return {"Std:Join": {"On": "", "Values": [value]}}


def pipeline_definition(*, bucket, role, runtime_image, ml_image, git_sha, key_arn, mlflow_arn):
    process_profile, train_profile, tune_profile = (
        profile("processing"),
        profile("training"),
        profile("tuning"),
    )
    execution = get("Execution.PipelineExecutionId")

    def output_path(stage):
        return join_path(f"s3://{bucket}/sagemaker/native", execution, stage)

    training_channel = get(
        "Steps.ProfileSnapshot.ProcessingOutputConfig.Outputs['TrainingChannel'].S3Output.S3Uri"
    )
    common = {"bucket": bucket, "role": role, "git_sha": git_sha, "kms_key_arn": key_arn}
    processing = processing_request(process_profile, image=runtime_image, **common)
    training = training_request(train_profile, image=ml_image, **common)
    training["HyperParameters"] = dict(training["HyperParameters"])
    tuning = tuning_request(tune_profile, train_profile, image=ml_image, **common)
    for request, name in (
        (processing, "ProcessingJobName"),
        (training, "TrainingJobName"),
        (tuning, "HyperParameterTuningJobName"),
    ):
        del request[name]

    def environment(stage):
        return {
            "MLFLOW_TRACKING_URI": mlflow_arn,
            "BENCH_KMS_KEY_ID": key_arn,
            "BENCH_NATIVE_EXECUTION_ID": execution,
            "BENCH_NATIVE_STAGE": stage,
            "BENCH_NATIVE_SCOPE": param("Scope"),
        }

    data_input = processing["ProcessingInputs"][0]
    data_input["S3Input"].update(S3Uri=param("PreparedParquetUri"), LocalPath="/opt/ml/processing/input/data")
    processing["ProcessingInputs"].append(
        {
            "InputName": "manifest",
            "S3Input": {
                **data_input["S3Input"],
                "S3Uri": param("PreparedManifestUri"),
                "LocalPath": "/opt/ml/processing/input/manifest",
            },
        }
    )
    processing["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"] = output_path("profile")
    processing["ProcessingOutputConfig"]["Outputs"].append(
        {
            "OutputName": "TrainingChannel",
            "S3Output": {
                "S3Uri": output_path("channel"),
                "LocalPath": "/opt/ml/processing/channel",
                "S3UploadMode": "EndOfJob",
            },
        }
    )
    processing["Environment"].update(environment("processing"))
    processing["Environment"]["BENCH_PROCESSING_JOB_NAME"] = get("Execution.ProcessingJobName")

    for definition, stage, hyperparameter_field in (
        (training, "training", "HyperParameters"),
        (tuning["TrainingJobDefinition"], "tuning", "StaticHyperParameters"),
    ):
        definition["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] = training_channel
        definition["OutputDataConfig"]["S3OutputPath"] = output_path(stage)
        definition["Environment"].update(environment(stage))
        definition[hyperparameter_field]["scope"] = param("Scope")
        definition["ResourceConfig"]["InstanceType"] = param("TrainingInstanceType")

    training["HyperParameters"]["max_epochs"] = as_string(param("MaxEpochs"))
    # An explicit max_epochs takes precedence over the legacy max_iterations alias.
    # Tune the parameter actually consumed by the job and avoid a static shadow value.
    tuning["TrainingJobDefinition"]["StaticHyperParameters"].pop("max_epochs", None)
    ranges = tuning["HyperParameterTuningJobConfig"]["ParameterRanges"]["IntegerParameterRanges"]
    for search_range in ranges:
        if search_range["Name"] == "max_iterations":
            search_range.update(Name="max_epochs", MaxValue=as_string(param("TuningMaxEpochs")))

    processing["ProcessingResources"]["ClusterConfig"]["InstanceType"] = param("ProcessingInstanceType")
    tuning["HyperParameterTuningJobConfig"]["ResourceLimits"] = {
        "MaxNumberOfTrainingJobs": param("MaxTuningJobs"),
        "MaxParallelTrainingJobs": param("MaxParallelTuningJobs"),
    }
    return {
        "Version": "2020-12-01",
        "Parameters": [
            {
                "Name": "PreparedParquetUri",
                "Type": "String",
                "DefaultValue": f"s3://{bucket}/{process_profile['input_prefix']}data.parquet",
            },
            {
                "Name": "PreparedManifestUri",
                "Type": "String",
                "DefaultValue": f"s3://{bucket}/{train_profile['train_prefix']}smoke.json",
            },
            {"Name": "Scope", "Type": "String", "DefaultValue": "smoke"},
            {
                "Name": "MaxEpochs",
                "Type": "Integer",
                "DefaultValue": int(train_profile["hyperparameters"]["max_epochs"]),
            },
            {
                "Name": "ProcessingInstanceType",
                "Type": "String",
                "DefaultValue": process_profile["instance_type"],
            },
            {
                "Name": "TrainingInstanceType",
                "Type": "String",
                "DefaultValue": train_profile["instance_type"],
            },
            {"Name": "TuningMaxEpochs", "Type": "Integer", "DefaultValue": 100},
            {"Name": "MaxTuningJobs", "Type": "Integer", "DefaultValue": tune_profile["max_jobs"]},
            {
                "Name": "MaxParallelTuningJobs",
                "Type": "Integer",
                "DefaultValue": tune_profile["max_parallel_jobs"],
            },
        ],
        "Steps": [
            {"Name": "ProfileSnapshot", "Type": "Processing", "Arguments": processing},
            {"Name": "TrainBaseline", "Type": "Training", "Arguments": training},
            {
                "Name": "TuneHyperparameters",
                "Type": "Tuning",
                "DependsOn": ["TrainBaseline"],
                "Arguments": tuning,
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("bucket", "role", "runtime-image", "ml-image", "git-sha", "key-arn", "mlflow-arn"):
        parser.add_argument(f"--{argument}", required=True)
    print(json.dumps(pipeline_definition(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
