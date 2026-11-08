"""Separate finite ZenML pipeline for the datasets' explicit two-pass protocol."""

import json
from pathlib import Path

from zenml import pipeline, step

from continual_bench.config import EpochConfig
from continual_bench.evaluation.epochs import run_epochs


@step(enable_cache=False)
def epoch_evaluate(config_data: dict, root: str, cache: str, real_directory: str, download: bool) -> dict:
    config = EpochConfig.model_validate(config_data)
    return run_epochs(config, root, cache, download=download, real_directory=real_directory)


@step(enable_cache=False)
def epoch_publish(report: dict, root: str) -> dict:
    from continual_bench.infrastructure.epoch_tracking import publish_epochs

    return publish_epochs(report, root)


@pipeline(enable_cache=False)
def dataset_epoch_benchmark(config_data: dict, root: str, cache: str, real_directory: str, download: bool):
    report = epoch_evaluate(config_data, root, cache, real_directory, download)
    epoch_publish(report, root)


def execute_epoch_pipeline(config, root, cache, *, real_directory, download=False):
    from continual_bench.application.readiness import ensure_tracking_ready

    from .zenml_pipeline import configure_local_stack

    ensure_tracking_ready()
    configure_local_stack(root)
    dataset_epoch_benchmark(config.model_dump(), root, cache, real_directory, download)
    # Its run key includes the actual selected dataset hash, not just request parameters.
    matches = []
    for path in Path(root).glob("*/epoch-report.json"):
        report = json.loads(path.read_text())
        if report["contract"]["config"] == config.model_dump():
            matches.append(report)
    if len(matches) != 1:
        raise ValueError("expected exactly one matching epoch report")
    report = matches[0]
    receipt = json.loads((Path(root) / report["run_key"] / "epoch-receipt.json").read_text())
    return {"summary": report, "receipt": receipt}
