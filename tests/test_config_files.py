"""Configuration inheritance and the versioned configs/ tree."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from continual_bench.application.campaign import Campaign
from continual_bench.application.matrix import ExperimentMatrix
from continual_bench.config import Config, TrackingConfig, TypedConfig, deep_merge, load_document, load_model

CONFIGS = Path(__file__).parents[1] / "configs"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_deep_merge_recurses_mappings_and_replaces_lists_scalars_and_nulls():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1, "c": {"keep": True}}
    merged = deep_merge(base, {"a": {"y": [3], "z": 2}, "b": None})
    assert merged == {"a": {"x": 1, "y": [3], "z": 2}, "b": None, "c": {"keep": True}}
    assert base["a"]["y"] == [1, 2]  # Inputs are never mutated.


def test_extends_chains_lists_and_nested_mappings_relative_to_each_file(tmp_path):
    write(tmp_path / "base/root.yaml", "dataset: {name: sea, samples: 500}\nwarmup: 10\n")
    write(tmp_path / "base/mid.yaml", "extends: root.yaml\nwarmup: 20\npolicy: {name: none}\n")
    write(tmp_path / "base/extra.json", '{"metric_window": 50, "policy": {"cooldown": 5}}')
    write(tmp_path / "base/tracking.yaml", "probe_samples: 4\n")
    path = write(tmp_path / "runs/run.yaml",
                 "extends: [../base/mid.yaml, ../base/extra.json]\n"
                 "dataset: {samples: 300}\n"
                 "tracking: {extends: ../base/tracking.yaml, continual_references: false}\n")
    assert load_document(path) == {
        "dataset": {"name": "sea", "samples": 300}, "warmup": 20, "metric_window": 50,
        "policy": {"name": "none", "cooldown": 5},
        "tracking": {"probe_samples": 4, "continual_references": False}}
    assert load_model(Config, path).tracking.probe_samples == 4


def test_list_items_may_extend_a_base(tmp_path):
    write(tmp_path / "case.yaml", "warmup: 30\ndataset: {samples: 180}\n")
    path = write(tmp_path / "campaign.yaml",
                 "name: c\ncases:\n  - {extends: case.yaml, dataset: {name: sea}}\n"
                 "  - {extends: case.yaml, dataset: {name: agrawal}}\n")
    cases = load_model(Campaign, path).cases
    assert [(c.dataset.name, c.dataset.samples, c.warmup) for c in cases] == [("sea", 180, 30),
                                                                             ("agrawal", 180, 30)]


@pytest.mark.parametrize("files,error,match", [
    ({"a.yaml": "extends: b.yaml\n", "b.yaml": "extends: a.yaml\n"}, ValueError, "cycle"),
    ({"a.yaml": "extends: missing.yaml\n"}, FileNotFoundError, "not found"),
    ({"a.yaml": "extends: [1]\n"}, ValueError, "path or a list"),
    ({"a.yaml": "extends: b.yaml\n", "b.yaml": "- not a mapping\n"}, ValueError, "mapping"),
])
def test_invalid_inheritance_fails_closed(tmp_path, files, error, match):
    for name, text in files.items():
        write(tmp_path / name, text)
    with pytest.raises(error, match=match):
        load_document(tmp_path / "a.yaml")


def test_extends_is_never_accepted_by_a_validated_payload():
    with pytest.raises(ValidationError):
        Config.model_validate({"extends": "../../etc/config.yaml"})


# Identities of the versioned examples; editing a base must not silently change a run.
PINNED_RUNS = {
    "runs/streaming.yaml": "897e16de43642d1d0a9e0e28",
    "runs/incremental.yaml": "044e156dd54baed038c20bd6",
    "runs/offline.yaml": "f60534406896a010d318bdf1",
    "runs/metrics.yaml": "5541472dde9d9a55390db437",
}


@pytest.mark.parametrize("name,identity", sorted(PINNED_RUNS.items()))
def test_versioned_runs_keep_their_identities(name, identity):
    assert load_model(Config, CONFIGS / name).identity() == identity


def test_versioned_campaign_typed_and_matrix_configs_validate():
    campaign = load_model(Campaign, CONFIGS / "campaigns/task-smoke.yaml")
    assert campaign.identity() == "056808872404983207e64a7b"
    assert load_model(TypedConfig, CONFIGS / "typed/friedman-smoke.yaml").dataset.name == "friedman_drift"
    assert load_model(ExperimentMatrix, CONFIGS / "matrices/binary-comparison.yaml").identity() == (
        "380356c50d6234d3b4e3c41f")
    any_model = load_model(ExperimentMatrix, CONFIGS / "matrices/any-model.yaml")
    assert any_model.identity() == "14ad8b68c464cbd8a5c39bb1"


def test_tracking_base_is_a_plain_root_profile_for_the_dags():
    raw = yaml.safe_load((CONFIGS / "base/tracking.yaml").read_text())
    assert "extends" not in raw and TrackingConfig.model_validate(raw).version == "residency-metrics-v1"


def test_every_input_config_resolves_and_inputs_live_only_in_known_folders():
    schemas = {"runs": Config, "typed": TypedConfig, "campaigns": Campaign, "matrices": ExperimentMatrix}
    for folder, schema in schemas.items():
        for path in sorted((CONFIGS / folder).glob("*.yaml")):
            if path.name != "base.yaml":
                load_model(schema, path)
    expected = {"base", "runs", "typed", "campaigns", "matrices", "contracts", "history"}
    assert {p.name for p in CONFIGS.iterdir() if p.is_dir()} == expected
    assert sorted(p.name for p in CONFIGS.iterdir() if p.is_file()) == ["README.md"]
