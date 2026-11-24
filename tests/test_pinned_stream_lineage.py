"""Publish lineage for already immutable prepared sources without rewriting them."""

import hashlib
import json

import pytest

from continual_bench.datasets.pinned_streams import PINNED_NAMES, source_contract
from continual_bench.evaluation.zia_publication import dataset_source_uri
from continual_bench.infrastructure.zia_tracking import publish_run_metrics


def _report(dataset, seed):
    prepared = hashlib.sha256(f"prepared:{dataset}".encode()).hexdigest()
    return {
        "dataset": dataset,
        "scope": "full",
        "model": "xgboost",
        "model_params": {},
        "policy": "frozen",
        "protocol": "batch-prequential-fixture",
        "artifact_mode": "metrics_only",
        "contract": {
            # Published pairs have license under source, so retain these exact fields.
            "input": {
                "source": source_contract(dataset),
                "sha256": prepared,
                "rows": 10,
                "features": ["feature"],
            },
            "environment": {"code_sha256": "c" * 64, "git_revision": "d" * 40},
            "iterative_training": {"seed": seed},
            "block_rows": 2,
            "holdout_start": 9,
        },
        "metrics": {"n": 7, "accuracy": 0.75},
        "holdout": {"n": 1},
        "windows": [],
        "class_metrics": {},
        "holdout_class_metrics": {},
        "model_versions": [],
        "cost_sensitivity": [],
        "costs": {},
    }


@pytest.mark.parametrize("dataset", PINNED_NAMES)
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_pinned_source_publishes_canonical_license_url_and_raw_revision_without_rewriting_input(
    tmp_path, monkeypatch, dataset, seed
):
    from mlflow import MlflowClient

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    experiment = client.create_experiment("pinned-lineage")
    run_id = client.create_run(experiment).info.run_id
    report = _report(dataset, seed)
    before = json.dumps(report, sort_keys=True)
    directory = tmp_path / "report"
    directory.mkdir()
    path = directory / "report.json"
    path.write_text(before)

    publish_run_metrics(client, directory, run_id, report)
    run = client.get_run(run_id)
    source = source_contract(dataset)
    assert run.data.params["dataset_license"] == source["license"]
    assert run.data.params["dataset_source"] == f"pinned-real/{dataset}"
    assert run.data.params["dataset_source_uri"] == dataset_source_uri(source) == source["url"]
    assert (
        run.data.params["dataset_revision"] == run.data.params["dataset_raw_sha256"] == source["raw_sha256"]
    )
    assert run.data.params["dataset_sha256"] == report["contract"]["input"]["sha256"]
    assert run.data.params["seed"] == str(seed)
    if "source_version" in source:
        assert run.data.params["dataset_source_version"] == source["source_version"]
    inputs = run.inputs.dataset_inputs
    assert len(inputs) == 1
    assert inputs[0].dataset.name == f"pinned-real/{dataset}"
    assert json.loads(inputs[0].dataset.source) == {"url": source["url"]}
    assert json.dumps(report, sort_keys=True) == before
    assert path.read_text() == before
    assert not list(directory.glob("*.npy")) and not list(directory.glob("*.pkl"))


@pytest.mark.parametrize(
    "root_license,source_license", [(None, None), (False, None), ("", None), ("Public", "CC BY 4.0")]
)
def test_publication_refuses_missing_invalid_or_conflicting_license_before_writes(
    root_license, source_license
):
    class NoWrites:
        def __getattr__(self, name):
            pytest.fail(f"unexpected publication before license validation: {name}")

    report = _report("electricity", 42)
    report["contract"]["input"]["source"].pop("license")
    if root_license is not None:
        report["contract"]["input"]["license"] = root_license
    if source_license is not None:
        report["contract"]["input"]["source"]["license"] = source_license
    with pytest.raises(ValueError, match="license"):
        publish_run_metrics(NoWrites(), None, "unwritten", report)


def test_source_uri_keeps_huggingface_and_generated_catalog_contracts():
    source = {"repo": "PinnedDataset", "revision": "a" * 40, "file": "source.parquet"}
    assert dataset_source_uri(source) == (
        "https://huggingface.co/datasets/ziadatalabs/PinnedDataset/resolve/" + "a" * 40 + "/source.parquet"
    )
    assert dataset_source_uri({"catalog": "sea"}) == "catalog://sea"


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("mode,expected", [("metrics_only", 600), ("full", 250)])
def test_metrics_only_keeps_complete_native_series_and_reconciles_republication(
    tmp_path, monkeypatch, seed, mode, expected
):
    from mlflow import MlflowClient

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    experiment = client.create_experiment("pinned-series")
    run_id = client.create_run(experiment).info.run_id
    report = _report("electricity", seed)
    report["artifact_mode"] = mode
    if mode == "full":
        # Legacy prepared manifests stored the license at the root instead.
        report["contract"]["input"]["license"] = report["contract"]["input"]["source"].pop("license")
    report["contract"]["input"]["rows"] = 602
    report["contract"]["holdout_start"] = 601
    report["metrics"]["n"] = 600
    report["windows"] = [
        {"scope": "prequential", "end_seq": i + 2, "n": 1, "accuracy": (i % 5) / 5} for i in range(600)
    ]
    directory = tmp_path / "report"
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps(report))
    for _ in range(2):
        publish_run_metrics(client, directory, run_id, report)
    accuracy = client.get_metric_history(run_id, "prequential/windows/accuracy")
    counts = client.get_metric_history(run_id, "prequential/windows/n")
    assert len(accuracy) == len(counts) == expected
    assert len({p.step for p in accuracy}) == expected
    assert min(p.step for p in accuracy) == 2 and max(p.step for p in accuracy) == 601
    if mode == "metrics_only":
        assert {(p.step, p.value) for p in accuracy} == {(i + 2, (i % 5) / 5) for i in range(600)}
        assert {(p.step, p.value) for p in counts} == {(i + 2, 1.0) for i in range(600)}
