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

    platform: Literal["local", "homelab", "aws"] = "local"
    save_model_checkpoints: bool = False
    tracking_enabled: bool = True
    output_dir: Path | str = Field(default_factory=lambda: Path("state/benchmarks"))
    tracking_uri: str = ""
    experiment_prefix: str = ""
    env_name: str = "local"
    generate_plots: bool = False

    @field_validator("platform", mode="before")
    @classmethod
    def _validate_platform(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized not in VALID_PLATFORMS:
            raise ValueError(f"Unsupported platform {value!r}; expected one of {sorted(VALID_PLATFORMS)}")
        return normalized

    @field_validator("save_model_checkpoints", "tracking_enabled", "generate_plots", mode="before")
    @classmethod
    def _validate_bool(cls, value: Any, info: Any) -> bool:
        return _parse_bool(value, name=info.field_name)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, value: Any) -> Path | str:
        if isinstance(value, Path):
            return value
        text = str(value).strip()
        if text.startswith("s3://"):
            return text
        return Path(text)

    @field_validator("tracking_uri", mode="before")
    @classmethod
    def _validate_tracking_uri(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @classmethod
    def from_env(
        cls,
        config_path: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> RuntimeSettings:
        """Load settings from YAML defaults, environment variables, and explicit overrides."""
        data: dict[str, Any] = {
            "platform": "local",
            "save_model_checkpoints": False,
            "tracking_enabled": True,
            "output_dir": Path("state/benchmarks"),
            "tracking_uri": "",
        }

        resolved_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
        if resolved_path.exists():
            loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected mapping in {resolved_path}, got {type(loaded).__name__}")
            for key in (
                "platform",
                "save_model_checkpoints",
                "tracking_enabled",
                "output_dir",
                "tracking_uri",
            ):
                if key in loaded and loaded[key] is not None:
                    data[key] = loaded[key]

        env_map = {
            "BENCH_PLATFORM": "platform",
            "BENCH_SAVE_MODEL_CHECKPOINTS": "save_model_checkpoints",
            "BENCH_TRACKING_ENABLED": "tracking_enabled",
            "BENCH_OUTPUT_DIR": "output_dir",
            "MLFLOW_TRACKING_URI": "tracking_uri",
            "BENCH_TRACKING_URI": "tracking_uri",
        }
        for env_key, field_name in env_map.items():
            if env_key in os.environ:
                env_val = os.environ[env_key]
                if env_val != "" or field_name == "tracking_uri":
                    data[field_name] = env_val

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
