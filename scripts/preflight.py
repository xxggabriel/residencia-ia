"""Validate approved external environment; never prints credential values."""

import base64
import ipaddress
import os


def validate(environment):
    ip = ipaddress.ip_address(environment["WEB_BIND_IP"])
    if not ip.is_private or ip.is_unspecified or ip.is_multicast:
        raise ValueError("WEB_BIND_IP must be the approved private host address")
    if str(ip) != "192.168.100.27":
        raise ValueError("this deployment is approved for 192.168.100.27 only")
    if len(environment["MINIO_USER"]) < 3 or len(environment["MINIO_PASSWORD"]) < 8:
        raise ValueError("MinIO requires username >=3 and password >=8 characters")
    for name in (
        "WEB_USER",
        "WEB_PASSWORD",
        "POSTGRES_PASSWORD",
        "AIRFLOW_FERNET_KEY",
        "AIRFLOW_JWT_SECRET",
        "MLFLOW_FLASK_SECRET",
    ):
        if not environment.get(name) or environment[name].startswith("CHANGE_ME"):
            raise ValueError(f"approved value required: {name}")
    if (len(environment.get("MLFLOW_PASSWORD", "")) < 12
            or environment["MLFLOW_PASSWORD"].startswith("CHANGE_ME")):
        raise ValueError("MLflow 3.17.2 requires an admin password of at least 12 characters")
    if len(base64.urlsafe_b64decode(environment["AIRFLOW_FERNET_KEY"])) != 32:
        raise ValueError("AIRFLOW_FERNET_KEY must encode exactly 32 bytes")
    if any(c in environment["POSTGRES_PASSWORD"] for c in "@:/?#%"):
        raise ValueError("Postgres password must be URL-safe for the configured DSN")


if __name__ == "__main__":
    validate(os.environ)
    print("Approved environment structure valid; credential values omitted.")
