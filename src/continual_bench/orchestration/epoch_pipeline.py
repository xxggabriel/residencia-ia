"""Separate finite pipeline for the datasets' explicit two-pass protocol."""

import json
from pathlib import Path

from continual_bench.config import EpochConfig
from continual_bench.evaluation.epochs import run_epochs


def epoch_evaluate(config_data: dict, root: str, cache: str, real_directory: str, download: bool) -> dict:
    config = EpochConfig.model_validate(config_data)
    return run_epochs(config, root, cache, download=download, real_directory=real_directory)


def epoch_publish(report: dict, root: str) -> dict:
    from continual_bench.infrastructure.epoch_tracking import publish_epochs

    return publish_epochs(report, root)


def execute_epoch_pipeline(config, root, cache, *, real_directory, download=False):
    from continual_bench.application.readiness import ensure_tracking_ready

    ensure_tracking_ready()
    report = epoch_evaluate(config.model_dump(), root, cache, real_directory, download)
    epoch_publish(report, root)
    matches = []
    for path in Path(root).glob("*/epoch-report.json"):
        loaded = json.loads(path.read_text())
        if loaded["contract"]["config"] == config.model_dump():
            matches.append(loaded)
    if len(matches) != 1:
        raise ValueError("expected exactly one matching epoch report")
    matched = matches[0]
    receipt = json.loads((Path(root) / matched["run_key"] / "epoch-receipt.json").read_text())
    return {"summary": matched, "receipt": receipt}
