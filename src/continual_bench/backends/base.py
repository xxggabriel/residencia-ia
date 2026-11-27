"""Base abstractions and runtime settings for benchmark storage and execution backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")

CHECKPOINT_EXTENSIONS: frozenset[str] = frozenset(
    {".pkl", ".pickle", ".ubj", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".bin"}
)
VALID_PLATFORMS: frozenset[str] = frozenset({"local", "homelab", "aws"})
DEFAULT_CONFIG_PATH = Path("configs/base/benchmark-defaults.yaml")
ENVIRONMENTS_DIR = Path("configs/environments")


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, str) and "${" in value:
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            return ""
        return expanded
    return value


def _parse_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


class RuntimeSettings(BaseModel):
    """Runtime configuration controlling backend platform, checkpoint policy, and tracking."""

    env_name: str = "local"
    platform: Literal["local", "homelab", "aws"] = "local"
    save_model_checkpoints: bool = False
    generate_plots: bool = False
    flush_artifacts_at_end_only: bool = True
    tracking_enabled: bool = True
    output_dir: Path | str = Field(default_factory=lambda: Path("state/benchmarks"))
    bucket: str = "benchmarks"
    s3_endpoint_url: str = ""
    tracking_uri: str = ""
    experiment_prefix: str = ""
    airflow_url: str = ""
    worker_url: str = ""

    @field_validator("platform", mode="before")
    @classmethod
    def _validate_platform(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in VALID_PLATFORMS:
            raise ValueError(f"Unsupported platform {value!r}; expected one of {sorted(VALID_PLATFORMS)}")
        return normalized

    @field_validator(
        "save_model_checkpoints",
        "generate_plots",
        "flush_artifacts_at_end_only",
        "tracking_enabled",
        mode="before",
    )
    @classmethod
    def _validate_bool(cls, value: Any, info: Any) -> bool:
        return _parse_bool(value, name=info.field_name)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, value: Any) -> Path | str:
        if isinstance(value, Path):
            return value
        text = str(_expand_env_placeholders(value)).strip()
        if not text:
            return Path("state/benchmarks")
        if text.startswith("s3://"):
            return text
        return Path(text)

    @field_validator(
        "env_name",
        "bucket",
        "s3_endpoint_url",
        "tracking_uri",
        "experiment_prefix",
        "airflow_url",
        "worker_url",
        mode="before",
    )
    @classmethod
    def _validate_optional_str(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(_expand_env_placeholders(value)).strip()

    @classmethod
    def from_env(
        cls,
        config_path: Path | None = None,
        overrides: dict[str, Any] | None = None,
        env_name: str | None = None,
    ) -> RuntimeSettings:
        """Load settings from environment YAML, environment variables, and explicit overrides."""
        data: dict[str, Any] = {
            "env_name": "local",
            "platform": "local",
            "save_model_checkpoints": False,
            "generate_plots": False,
            "flush_artifacts_at_end_only": True,
            "tracking_enabled": True,
            "output_dir": Path("state/benchmarks"),
            "bucket": "benchmarks",
            "s3_endpoint_url": "",
            "tracking_uri": "",
            "experiment_prefix": "",
            "airflow_url": "",
            "worker_url": "",
        }

        requested_env = (env_name or os.environ.get("BENCH_ENV", "")).strip().lower()
        env_config_raw = os.environ.get("BENCH_ENV_CONFIG", "").strip()
        if config_path is not None:
            resolved_path = config_path
        elif env_config_raw:
            resolved_path = Path(env_config_raw)
        elif requested_env and (ENVIRONMENTS_DIR / f"{requested_env}.yaml").exists():
            resolved_path = ENVIRONMENTS_DIR / f"{requested_env}.yaml"
            data["env_name"] = requested_env
        else:
            resolved_path = DEFAULT_CONFIG_PATH

        if resolved_path.exists():
            loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected mapping in {resolved_path}, got {type(loaded).__name__}")
            for key in (
                "env_name",
                "platform",
                "save_model_checkpoints",
                "generate_plots",
                "flush_artifacts_at_end_only",
                "tracking_enabled",
                "output_dir",
                "bucket",
                "s3_endpoint_url",
                "tracking_uri",
                "experiment_prefix",
                "airflow_url",
                "worker_url",
            ):
                if key in loaded and loaded[key] is not None:
                    data[key] = _expand_env_placeholders(loaded[key])

        env_map = {
            "BENCH_ENV": "env_name",
            "BENCH_PLATFORM": "platform",
            "BENCH_SAVE_MODEL_CHECKPOINTS": "save_model_checkpoints",
            "BENCH_GENERATE_PLOTS": "generate_plots",
            "BENCH_FLUSH_ARTIFACTS_AT_END_ONLY": "flush_artifacts_at_end_only",
            "BENCH_TRACKING_ENABLED": "tracking_enabled",
            "BENCH_OUTPUT_DIR": "output_dir",
            "BENCH_BUCKET": "bucket",
            "S3_ENDPOINT_URL": "s3_endpoint_url",
            "MLFLOW_TRACKING_URI": "tracking_uri",
            "BENCH_TRACKING_URI": "tracking_uri",
            "BENCH_EXPERIMENT_PREFIX": "experiment_prefix",
            "BENCH_AIRFLOW_URL": "airflow_url",
            "BENCH_WORKER_URL": "worker_url",
        }
        for env_key, field_name in env_map.items():
            if env_key in os.environ:
                env_val = os.environ[env_key]
                if env_val != "" or field_name == "tracking_uri":
                    data[field_name] = env_val

        if env_name:
            data["env_name"] = env_name.strip().lower()

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    data[key] = value

        return cls(**data)


class StorageBackend(ABC):
    """Abstract storage interface for benchmark artifacts with zero-checkpoint enforcement."""

    settings: RuntimeSettings

    @abstractmethod
    def write_bytes(self, rel_path: str, data: bytes) -> str:
        """Write raw bytes to a relative path under the backend output root."""

    @abstractmethod
    def write_json(self, rel_path: str, payload: Any) -> str:
        """Serialize payload as JSON and write it to a relative path."""

    @abstractmethod
    def write_text(self, rel_path: str, text: str) -> str:
        """Write UTF-8 text to a relative path under the backend output root."""

    @abstractmethod
    def save_model_checkpoint(self, rel_path: str, model_obj: Any) -> str | None:
        """Persist a model checkpoint only when settings.save_model_checkpoints is True."""

    @abstractmethod
    def verify_no_checkpoints(self) -> list[str]:
        """Scan the backend output location and return any forbidden checkpoint paths."""


class ExecutionBackend(StorageBackend, ABC):
    """Abstract execution and storage backend for running benchmark tasks."""

    platform_name: str = "base"

    @abstractmethod
    def run_tasks(self, tasks: list[Callable[[], T]], max_workers: int = 1) -> list[T]:
        """Execute a list of zero-argument callables and return results in input order."""
