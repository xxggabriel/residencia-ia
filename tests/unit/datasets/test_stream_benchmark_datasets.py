from pathlib import Path

import pytest
import yaml

from continual_bench.datasets.streams import (
    CatalogStreamProvider,
    ZiaStreamProvider,
    dataset_model_params,
    load_stream_manifest,
    prepare_stream,
    stream_names,
    stream_spec,
)


def _tracking():
    from continual_bench.config import TrackingConfig

    settings = yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    return TrackingConfig.model_validate(settings).model_dump()


def test_registry_exposes_zia_and_catalog_datasets_with_provider_defaults():
    names = stream_names()
    assert {"fraud", "social", "ecommerce"} <= set(names)
    assert {"sea", "agrawal", "electricity", "insects"} <= set(names)
    assert "friedman_drift" not in names  # regression is not a classification stream
    assert stream_spec("fraud").model_params == {"scale_pos_weight": 39.0}
    assert stream_spec("social").model == "multinomial_logistic"
    assert dataset_model_params("ecommerce") == {"scale_pos_weight": 10.0}
    assert dataset_model_params("sea") == {} and dataset_model_params("unknown") == {}
    sea = stream_spec("sea")
    assert sea.provider == "catalog" and sea.classes == ("0", "1") and sea.model == "xgboost"
    assert sea.block_rows == CatalogStreamProvider.BLOCK_ROWS != ZiaStreamProvider.BLOCK_ROWS


def test_catalog_preparation_is_deterministic_and_validated(tmp_path):
    first = prepare_stream(tmp_path / "a", "agrawal", "smoke")
    second = prepare_stream(tmp_path / "b", "agrawal", "smoke")
    assert first["sha256"] == second["sha256"] and first["rows"] == CatalogStreamProvider.EXAMPLES["smoke"]
    assert first["provider"] == "catalog" and first["source"]["classes"] == ["0", "1"]
    path, manifest = load_stream_manifest(tmp_path / "a", "agrawal", "smoke")
    assert manifest == first and path.name == "smoke.parquet"
    assert prepare_stream(tmp_path / "a", "agrawal", "smoke") == first  # cached pair is reused
    (tmp_path / "a" / "agrawal" / "smoke.parquet").write_bytes(b"corrupt")
    with pytest.raises(Exception):
        load_stream_manifest(tmp_path / "a", "agrawal", "smoke")


@pytest.mark.parametrize("policy", ["frozen", "periodic", "drift"])
def test_benchmark_runs_any_catalog_dataset_with_the_compat_entry_point(tmp_path, policy):
    from continual_bench.evaluation.stream_benchmark import run
    from continual_bench.evaluation.stream_benchmark import run as legacy_run

    assert legacy_run is run
    prepare_stream(tmp_path, "sea", "smoke")
    report = run(tmp_path, "sea", "smoke", policy, tmp_path / policy, _tracking())
    assert report["complete"] and report["dataset"] == "sea" and report["model"] == "xgboost"
    assert report["contract"]["block_rows"] == CatalogStreamProvider.BLOCK_ROWS["smoke"]
    summary, holdout = report["metrics"], report["holdout"]
    rows = summary["initial_train_rows"] + summary["n"] + holdout["n"]
    assert rows == CatalogStreamProvider.EXAMPLES["smoke"]
    assert set(report["class_metrics"]) == {"0", "1"}
    assert "scale_pos_weight" not in report["model_params"]
