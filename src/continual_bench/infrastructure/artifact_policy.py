"""Filesystem retention contract shared by evaluators and evidence publishers."""

from pathlib import Path


def metrics_only(report: dict) -> bool:
    mode = (
        report.get("artifact_mode")
        or (report.get("config") or {}).get("artifact_mode")
        or (report.get("contract") or {}).get("artifact_mode")
        or "full"
    )
    if mode not in {"full", "metrics_only"}:
        raise ValueError("artifact_mode must be full or metrics_only")
    return mode == "metrics_only"


def verify_metrics_only_files(directory) -> None:
    """Fail closed if a metrics-only result contains persisted model state."""
    forbidden = {".pkl", ".pickle", ".ubj", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".bin"}
    for path in Path(directory).rglob("*"):
        if path.is_file() and (
            path.suffix.lower() in forbidden
            or path.name in {"model.tar.gz", "checkpoint.tar.gz"}
        ):
            raise ValueError(f"metrics_only evidence contains model or checkpoint state: {path.name}")
