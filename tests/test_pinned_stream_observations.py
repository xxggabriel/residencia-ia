"""Prepared-pair fixtures exercise each real provider's immutable source schema."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from continual_bench.config import TrackingConfig
from continual_bench.datasets.pinned_streams import (
    PINNED_NAMES,
    PREPARATION,
    PinnedRealStreamProvider,
    expected_rows,
    source_contract,
)
from continual_bench.datasets.zia import FORMAT, sha256_file
from continual_bench.evaluation.stream_benchmark import run
from continual_bench.evaluation.zia_artifacts import tracking_report
from continual_bench.evaluation.zia_predictions import prediction_manifest
from continual_bench.evaluation.zia_publication import dataset_source_uri, run_documents


@pytest.mark.parametrize("dataset", PINNED_NAMES)
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_pinned_provider_metadata_supports_complete_observations(tmp_path, dataset, seed):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = expected_rows(dataset, "smoke")
    ids = np.arange(rows)
    labels = ((ids // 500 + ids % 2) % 2).astype(np.uint8)
    path = tmp_path / dataset / "smoke.parquet"
    path.parent.mkdir()
    pq.write_table(
        pa.table({"event_time": ids, "label": labels, "f1": (ids % 2).astype(np.float32)}),
        path,
        row_group_size=83,
    )
    source = {**source_contract(dataset), "classes": ["0", "1"], "model": "xgboost"}
    assert "revision" not in source
    manifest = {
        "format": FORMAT,
        "provider": "pinned_real",
        "preparation_version": PREPARATION,
        "dataset": dataset,
        "scope": "smoke",
        "source": source,
        "selection": {"kind": "published_prefix", "seq_range": [0, rows]},
        "features": ["f1"],
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    path.with_suffix(".json").write_text(json.dumps(manifest))
    before = path.with_suffix(".json").read_bytes()
    assert PinnedRealStreamProvider().load_manifest(tmp_path, dataset, "smoke")[1] == manifest
    tracking = TrackingConfig.model_validate(
        yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    ).model_dump()
    output = tmp_path / "result"
    report = run(
        tmp_path,
        dataset,
        "smoke",
        "frozen",
        output,
        tracking,
        iterative_config={"seed": seed, "max_epochs": 1},
        artifact_mode="metrics_only",
    )
    observed = report["tracking_observations"]
    assert report["complete"] and observed["lineage"]["source_revision"] == source["raw_sha256"]
    assert observed["lineage"]["dataset_version"] == manifest["sha256"]
    assert observed["lineage"]["seed"] == seed
    assert observed["lineage"]["dataset"] == dataset
    assert observed["scopes"]["prequential"]["evaluated_examples"] == report["metrics"]["n"]
    assert observed["scopes"]["holdout"]["evaluated_examples"] == report["holdout"]["n"]
    assert report["metrics"]["n"] + report["holdout"]["n"] + report["metrics"]["initial_train_rows"] == rows
    assert all(fit["seed"] == seed for fit in report["training_history"])
    assert path.with_suffix(".json").read_bytes() == before
    adapted = tracking_report(report, prediction_manifest(output, report))
    _, links, documents = run_documents(report, adapted, "s3://evidence/result")
    assert links["dataset_source_uri"] == source["url"]
    assert documents["manifest.json"] == manifest


def test_pinned_source_uri_keeps_public_origin_instead_of_unknown_catalog():
    source = source_contract(PINNED_NAMES[0])
    assert dataset_source_uri(source) == source["url"]
