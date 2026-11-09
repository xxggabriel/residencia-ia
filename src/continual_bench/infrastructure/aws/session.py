"""Shared boto3 client factory: region, bounded retries and timeouts in one place."""

import os

DEFAULT_REGION = "us-east-1"


def region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION


def client(service: str, *, region_name: str | None = None, max_attempts: int = 8, read_timeout: int = 60):
    import boto3
    from botocore.config import Config

    return boto3.client(
        service,
        region_name=region_name or region(),
        config=Config(
            connect_timeout=10,
            read_timeout=read_timeout,
            retries={"max_attempts": max_attempts, "mode": "adaptive"},
            user_agent_extra="continual-bench",
        ),
    )


def error_code(error) -> str | None:
    response = getattr(error, "response", None) or {}
    return response.get("Error", {}).get("Code")
