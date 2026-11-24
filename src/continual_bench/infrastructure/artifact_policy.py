"""Filesystem retention contract shared by evaluators and evidence publishers."""

import json
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
    state_directories = {"model", "models", "candidates", "checkpoints", ".zia-resume"}
    root = Path(directory)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("metrics_only evidence must not contain symlinked artifacts")
        relative = path.relative_to(root)
        # The normal artifact namespace records why model explanations are absent.
        # Only this exact metadata object is allowed; it cannot carry parameters.
        if relative.as_posix() == "artifacts/data/model/explanation.json" and path.is_file():
            if json.loads(path.read_text()) == {
                "status": "unavailable",
                "reason": "metrics_only_does_not_persist_learned_parameters",
            }:
                continue
        if path.is_file() and (
            path.suffix.lower() in forbidden
            or path.name in {"model.tar.gz", "checkpoint.tar.gz"}
            or state_directories.intersection(relative.parts[:-1])
        ):
            raise ValueError(f"metrics_only evidence contains model or checkpoint state: {path.name}")
