import base64
import hashlib
import importlib.util
from pathlib import Path

import pytest

from continual_bench.application.config import Config
from continual_bench.application.runner import run


def test_csv_extension_preserves_real_time_and_purges_unavailable_training_labels(tmp_path):
    file = tmp_path / "real.csv"
    file.write_text("time,value,target\n" + "".join(f"{10000 + i * 10},{i},{i % 2}\n" for i in range(100)))
    cfg = Config.model_validate(
        {
            "name": "csv-contract",
            "mode": "offline",
            "warmup": 10,
            "plugins": ["examples.temporal_csv:register"],
            "policy": {"name": "none"},
            "dataset": {
                "name": "temporal_csv",
                "samples": 100,
                "label_delay": 25,
                "params": {
                    "path": str(file),
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "license_id": "test-fixture-self-generated",
                    "source_url": "fixture://self",
                    "timestamp_column": "time",
                    "label_column": "target",
                    "feature_columns": ["value"],
                },
            },
        }
    )
    result = run(cfg, tmp_path / "state")
    assert result["split"]["train"] == [0, 58]
    assert result["split"]["purged_label_delay"] == [58, 60]
    assert result["dataset"]["changepoints"] is None
    assert result["retention_probes"] == []


def test_preflight_rejects_incompatible_minio_password_and_unapproved_bind():
    path = Path(__file__).parents[1] / "scripts/preflight.py"
    spec = importlib.util.spec_from_file_location("preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = {
        "WEB_BIND_IP": "192.168.100.27",
        "MINIO_USER": "fixture",
        "MINIO_PASSWORD": "fixture8",
        "WEB_USER": "fixture",
        "WEB_PASSWORD": "fixture",
        "POSTGRES_PASSWORD": "fixture",
        "AIRFLOW_FERNET_KEY": base64.urlsafe_b64encode(bytes(32)).decode(),
        "AIRFLOW_JWT_SECRET": "fixture",
        "MLFLOW_FLASK_SECRET": "fixture",
        "MLFLOW_PASSWORD": "fixture-long-enough",
    }
    module.validate(fields)
    with pytest.raises(ValueError, match="MLflow"):
        module.validate({**fields, "MLFLOW_PASSWORD": "short"})
    with pytest.raises(ValueError, match="MinIO"):
        module.validate({**fields, "MINIO_PASSWORD": "short"})
    with pytest.raises(ValueError, match="private"):
        module.validate({**fields, "WEB_BIND_IP": "0.0.0.0"})
