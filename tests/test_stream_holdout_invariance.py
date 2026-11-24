"""Future holdout labels cannot affect training, schedules, or earlier predictions.

These are controlled binary and multiclass Parquet fixtures, not benchmark
results for the public datasets whose model providers they exercise.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import yaml

from continual_bench.config import TrackingConfig
from continual_bench.datasets import zia
from continual_bench.datasets.streams import ZiaStreamProvider
from continual_bench.evaluation import stream_benchmark as stream
from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files


@pytest.fixture
def ordered_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(zia, "SMOKE_ROWS", 800)
    monkeypatch.setattr(ZiaStreamProvider, "BLOCK_ROWS", {"smoke": 100, "full": 100})
    ids = np.arange(800)
    for dataset in ("fraud", "social"):
        classes = len(zia.DATASETS[dataset]["classes"])
        base = ids % classes
        labels = ((base + ids // 250) % classes).astype(np.uint8)
        # Ensure the holdout label permutation changes class support even if
        # a short training run produces constant predictions.
        labels[-100:] = 0
        path = tmp_path / dataset / "smoke.parquet"
        path.parent.mkdir()
        table = pa.table({"event_time": ids // 3, "label": labels, "feature": base.astype(np.float32)})
        pq.write_table(table, path, row_group_size=71)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "scope": "smoke",
                    "source": zia.DATASETS[dataset],
                    "format": zia.FORMAT,
                    "features": ["feature"],
                    "rows": len(labels),
                    "bytes": path.stat().st_size,
                    "sha256": zia.sha256_file(path),
                }
            )
        )
    settings = TrackingConfig.model_validate(
        yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    ).model_dump()
    settings["retraining"] = {"n_alphas": 2, "mc_samples": 3}
    return tmp_path, settings


def prediction_table(directory):
    return pa.concat_tables(
        [pq.read_table(path) for path in sorted((directory / "predictions").glob("*.parquet"))]
    )


def fit_contract(fit):
    return {
        **{
            key: fit[key]
            for key in (
                "fit_id",
                "action",
                "block_index",
                "start_seq",
                "end_seq",
                "train_samples",
                "validation_samples",
                "training_presentations",
                "seed",
                "deployed_version",
                "promotion_status",
            )
        },
        "checkpoint": fit["checkpoint"],
    }


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("dataset", ["fraud", "social"])
@pytest.mark.parametrize("policy", ["cara", "upf", "oracle"])
def test_future_holdout_labels_do_not_change_learning_or_decisions(ordered_fixture, dataset, policy, seed):
    root, tracking = ordered_fixture
    settings = copy.deepcopy(tracking)
    settings["retraining"]["seed"] = seed

    def evaluate(name):
        # Each evaluation builds its own matrix. No external cache can mask
        # accidental use of the changed future labels.
        return stream.run(
            root,
            dataset,
            "smoke",
            policy,
            root / name,
            settings,
            iterative_config={"seed": seed, "max_epochs": 3, "patience": 3},
            artifact_mode="metrics_only",
        )

    before = evaluate("before")
    boundary = before["contract"]["holdout_start"]
    path = root / dataset / "smoke.parquet"
    original = pq.read_table(path)
    labels = original["label"].to_numpy().copy()
    changed_labels = labels.copy()
    classes = len(zia.DATASETS[dataset]["classes"])
    changed_labels[boundary:] = (changed_labels[boundary:] + 1) % classes
    changed = original.set_column(original.schema.get_field_index("label"), "label", pa.array(changed_labels))
    assert changed.slice(0, boundary).equals(original.slice(0, boundary))
    assert np.all(changed_labels[boundary:] != labels[boundary:])
    pq.write_table(changed, path, row_group_size=71)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(bytes=path.stat().st_size, sha256=zia.sha256_file(path))
    manifest_path.write_text(json.dumps(manifest))

    after = evaluate("after")
    assert after["contract"]["holdout_start"] == boundary
    assert after["contract"]["input"]["sha256"] != before["contract"]["input"]["sha256"]
    assert after["contract"]["iterative_training"] == before["contract"]["iterative_training"]
    assert after["prequential_confusion"] == before["prequential_confusion"]
    assert after["retraining"]["deployed"] == before["retraining"]["deployed"]
    assert after["alarms"] == before["alarms"]
    assert [version["sha256"] for version in after["model_versions"]] == [
        version["sha256"] for version in before["model_versions"]
    ]
    assert [fit_contract(fit) for fit in after["training_history"]] == [
        fit_contract(fit) for fit in before["training_history"]
    ]
    for fit in after["training_history"]:
        start, end = fit["start_seq"], fit["end_seq"]
        assert 0 <= start < end <= boundary
        assert fit["train_samples"] + fit["validation_samples"] == end - start
        # This interval includes both training and temporal validation labels.
        np.testing.assert_array_equal(changed_labels[start:end], labels[start:end])

    before_predictions = prediction_table(root / "before")
    after_predictions = prediction_table(root / "after")
    assert before_predictions.filter(pc.invert(before_predictions["holdout"])).equals(
        after_predictions.filter(pc.invert(after_predictions["holdout"]))
    )
    # The final model also predicts the unchanged future features identically.
    assert before_predictions.drop(["label"]).equals(after_predictions.drop(["label"]))
    assert after["holdout_confusion"] != before["holdout_confusion"]
    assert after["holdout_class_metrics"] != before["holdout_class_metrics"]
    for name in ("before", "after"):
        verify_metrics_only_files(root / name)
