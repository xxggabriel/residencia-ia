"""SageMaker Model Registry: candidates stay PendingManualApproval and point to S3 + ECR digests."""

from .session import client as make_client
from .session import error_code


class SageMakerModelRegistry:
    def __init__(self, *, client=None):
        self.client = client or make_client("sagemaker")

    def ensure_group(self, name: str, description: str, tags: dict) -> str:
        try:
            return self.client.describe_model_package_group(ModelPackageGroupName=name)[
                "ModelPackageGroupArn"
            ]
        except Exception as error:
            if error_code(error) != "ValidationException":
                raise
        try:
            return self.client.create_model_package_group(
                ModelPackageGroupName=name,
                ModelPackageGroupDescription=description[:1024],
                Tags=[{"Key": k, "Value": str(v)} for k, v in tags.items()],
            )["ModelPackageGroupArn"]
        except Exception as error:
            if error_code(error) not in {"ValidationException", "ConflictException"}:
                raise
            return self.client.describe_model_package_group(ModelPackageGroupName=name)[
                "ModelPackageGroupArn"
            ]

    def find(self, group: str, run_id: str) -> str | None:
        """Model package ARN already registered for this run (idempotent retries)."""
        paginator = self.client.get_paginator("list_model_packages")
        for page in paginator.paginate(ModelPackageGroupName=group, ModelPackageType="Versioned"):
            for summary in page.get("ModelPackageSummaryList", []):
                detail = self.client.describe_model_package(ModelPackageName=summary["ModelPackageArn"])
                if detail.get("CustomerMetadataProperties", {}).get("run_id") == run_id:
                    return summary["ModelPackageArn"]
        return None

    def register(
        self,
        *,
        group: str,
        run_id: str,
        image: str,
        model_data_uri: str,
        metadata: dict,
        content_types=("application/json", "text/csv"),
        description: str = "",
    ) -> dict:
        existing = self.find(group, run_id)
        if existing:
            return {"model_package_arn": existing, "created": False}
        properties = {k: str(v)[:256] for k, v in {**metadata, "run_id": run_id}.items()}
        response = self.client.create_model_package(
            ModelPackageGroupName=group,
            ModelPackageDescription=description[:1024],
            ModelApprovalStatus="PendingManualApproval",
            CustomerMetadataProperties=properties,
            InferenceSpecification={
                "Containers": [{"Image": image, "ModelDataUrl": model_data_uri}],
                "SupportedContentTypes": list(content_types),
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
                "SupportedTransformInstanceTypes": ["ml.m5.large"],
            },
        )
        return {"model_package_arn": response["ModelPackageArn"], "created": True}
