"""Build (and optionally start) the SageMaker profiles from infra/sagemaker/profiles/*.json.

Every profile reuses the release images: `runtime` for Processing, `ml` for Training, Tuning
and Inference. `--dry-run` prints the exact API request without calling AWS.

Usage:
  python scripts/aws/sagemaker_jobs.py processing --release latest --dry-run
  python scripts/aws/sagemaker_jobs.py training --release <sha>
  python scripts/aws/sagemaker_jobs.py tuning --release <sha>
  python scripts/aws/sagemaker_jobs.py inference --model-data s3://.../model.tar.gz --delete-after
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def profile(name):
    return json.loads((ROOT / "infra" / "sagemaker" / "profiles" / f"{name}.json").read_text())


def job_name(kind, git_sha):
    return f"cb-{kind}-{git_sha[:8]}-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}"


def processing_request(spec, *, bucket, role, image, git_sha, name=None, kms_key_arn=None):
    name = name or job_name("proc", git_sha)
    return {
        "ProcessingJobName": name,
        "RoleArn": role,
        "AppSpecification": {"ImageUri": image, "ContainerEntrypoint": spec["entrypoint"]},
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": spec["instance_type"],
                "VolumeSizeInGB": spec["volume_gb"],
            }
        },
        "ProcessingInputs": [
            {
                "InputName": "snapshot",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{spec['input_prefix']}",
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        "ProcessingOutputConfig": {
            **({"KmsKeyId": kms_key_arn} if kms_key_arn else {}),
            "Outputs": [
                {
                    "OutputName": "profile",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{spec['output_prefix'].format(name=name)}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": spec["max_runtime_seconds"]},
        "Environment": {"OMP_NUM_THREADS": "1"},
        "Tags": [{"Key": "cb-stack", "Value": "continual-bench"}, {"Key": "cb-git-sha", "Value": git_sha}],
    }


def training_request(spec, *, bucket, role, image, git_sha, name=None, hyperparameters=None,
                     kms_key_arn=None):
    name = name or job_name("train", git_sha)
    return {
        "TrainingJobName": name,
        "RoleArn": role,
        "AlgorithmSpecification": {
            "TrainingImage": image,
            "TrainingInputMode": "File",
            "MetricDefinitions": spec["metric_definitions"],
        },
        "HyperParameters": hyperparameters if hyperparameters is not None else spec["hyperparameters"],
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/{spec['train_prefix']}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/{spec['output_prefix']}",
            **({"KmsKeyId": kms_key_arn} if kms_key_arn else {}),
        },
        "ResourceConfig": {
            "InstanceType": spec["instance_type"],
            "InstanceCount": 1,
            "VolumeSizeInGB": spec["volume_gb"],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": spec["max_runtime_seconds"]},
        "Tags": [{"Key": "cb-stack", "Value": "continual-bench"}, {"Key": "cb-git-sha", "Value": git_sha}],
    }


def tuning_request(spec, training, *, bucket, role, image, git_sha, kms_key_arn=None):
    base = training_request(
        training,
        bucket=bucket,
        role=role,
        image=image,
        git_sha=git_sha,
        kms_key_arn=kms_key_arn,
        hyperparameters=spec["static_hyperparameters"],
    )
    definition = {
        k: base[k]
        for k in (
            "RoleArn",
            "AlgorithmSpecification",
            "InputDataConfig",
            "OutputDataConfig",
            "ResourceConfig",
            "StoppingCondition",
        )
    }
    definition["StaticHyperParameters"] = base["HyperParameters"]
    return {
        "HyperParameterTuningJobName": f"cb-tune-{git_sha[:8]}-{time.strftime('%m%d%H%M%S', time.gmtime())}",
        "HyperParameterTuningJobConfig": {
            "Strategy": spec["strategy"],
            "HyperParameterTuningJobObjective": spec["objective"],
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": spec["max_jobs"],
                "MaxParallelTrainingJobs": spec["max_parallel_jobs"],
            },
            "ParameterRanges": spec["ranges"],
            "TrainingJobEarlyStoppingType": "Off",
        },
        "TrainingJobDefinition": definition,
        "Tags": [{"Key": "cb-stack", "Value": "continual-bench"}, {"Key": "cb-git-sha", "Value": git_sha}],
    }


def inference_requests(spec, *, role, image, model_data, git_sha):
    name = job_name("infer", git_sha)
    return name, {
        "model": {
            "ModelName": name,
            "ExecutionRoleArn": role,
            "PrimaryContainer": {"Image": image, "ModelDataUrl": model_data},
        },
        "config": {
            "EndpointConfigName": name,
            "ProductionVariants": [
                {
                    "VariantName": "candidate",
                    "ModelName": name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": spec["memory_mb"],
                        "MaxConcurrency": spec["max_concurrency"],
                    },
                }
            ],
        },
        "endpoint": {"EndpointName": name, "EndpointConfigName": name},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["processing", "training", "tuning", "inference"])
    parser.add_argument("--release", default="latest")
    parser.add_argument("--bucket")
    parser.add_argument("--role")
    parser.add_argument("--kms-key-arn", help="SSE-KMS key for job outputs (default: the data bucket key)")
    parser.add_argument("--model-data")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-after", action="store_true")
    args = parser.parse_args()
    if args.dry_run and not (args.bucket and args.role):
        args.bucket, args.role = (
            args.bucket or "example-bucket",
            args.role or "arn:aws:iam::123456789012:role/x",
        )
    from continual_bench.application.cloud.planning import resolve_release

    if args.dry_run and args.release == "example":
        git_sha, images = "0" * 40, {"runtime": "runtime@sha256:" + "0" * 64, "ml": "ml@sha256:" + "0" * 64}
    else:
        from continual_bench.infrastructure.aws.s3 import S3ObjectStore

        if not args.bucket or not args.role:
            parser.error("--bucket and --role are required")
        store = S3ObjectStore(args.bucket)
        release = resolve_release(store, args.release)
        git_sha, images = release.git_sha, {k: v.uri for k, v in release.images.items()}
        if args.kind != "inference" and not args.kms_key_arn:
            response = store.client.get_bucket_encryption(Bucket=args.bucket)
            rules = response["ServerSideEncryptionConfiguration"]
            encryption = rules["Rules"][0]["ApplyServerSideEncryptionByDefault"]
            args.kms_key_arn = encryption.get("KMSMasterKeyID")
            if encryption["SSEAlgorithm"] != "aws:kms" or not (args.kms_key_arn or "").startswith("arn:"):
                parser.error("provide the data bucket customer-managed KMS key ARN using --kms-key-arn")
    spec = profile(args.kind)
    common = {"bucket": args.bucket, "role": args.role, "git_sha": git_sha, "kms_key_arn": args.kms_key_arn}
    if args.kind == "processing":
        request = processing_request(spec, image=images["runtime"], **common)
    elif args.kind == "training":
        request = training_request(spec, image=images["ml"], **common)
    elif args.kind == "tuning":
        request = tuning_request(spec, profile("training"), image=images["ml"], **common)
    else:
        if not args.model_data:
            raise SystemExit("--model-data (model.tar.gz of a registered candidate) is required")
        name, request = inference_requests(
            spec, role=args.role, image=images["ml"], model_data=args.model_data, git_sha=git_sha
        )
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return
    import boto3

    sagemaker = boto3.client("sagemaker", region_name="us-east-1")
    if args.kind == "processing":
        print(sagemaker.create_processing_job(**request)["ProcessingJobArn"])
    elif args.kind == "training":
        print(sagemaker.create_training_job(**request)["TrainingJobArn"])
    elif args.kind == "tuning":
        print(sagemaker.create_hyper_parameter_tuning_job(**request)["HyperParameterTuningJobArn"])
    else:
        sagemaker.create_model(**request["model"])
        sagemaker.create_endpoint_config(**request["config"])
        sagemaker.create_endpoint(**request["endpoint"])
        sagemaker.get_waiter("endpoint_in_service").wait(EndpointName=name)
        runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
        response = runtime.invoke_endpoint(
            EndpointName=name, ContentType=spec["content_type"], Body=json.dumps(spec["probe"])
        )
        print(response["Body"].read().decode())
        if args.delete_after:
            sagemaker.delete_endpoint(EndpointName=name)
            sagemaker.delete_endpoint_config(EndpointConfigName=name)
            sagemaker.delete_model(ModelName=name)
            print(f"deleted endpoint {name}")


if __name__ == "__main__":
    main()
