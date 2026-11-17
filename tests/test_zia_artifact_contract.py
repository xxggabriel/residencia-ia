import json
import shutil
import subprocess
import sys

import pytest

from continual_bench.evaluation.artifacts import binary_figures, write_artifacts
from continual_bench.evaluation.zia_artifacts import training_figures, write_zia_campaign_artifacts


def test_training_plots_keep_fit_local_axes_and_global_coordinates(tmp_path):
    def fit(index, kind, start):
        return {
            "fit_id": f"fit-{index:06}",
            "action": "initial" if index == 1 else "online",
            "start_seq": (index - 1) * 100,
            "end_seq": index * 100,
            "block_index": index - 1,
            "checkpoint": {"file": f"models/{index}.npy", "sha256": "abc"},
            "deployed_version": index - 1,
            "steps": [
                {"step": 1, "global_step": start, "kind": kind, "metrics": {"train:log_loss": 0.2}},
                {"step": 2, "global_step": start + 1, "kind": kind, "metrics": {"train:log_loss": 0.1}},
            ],
        }

    write_artifacts(tmp_path, training_figures([fit(1, "epoch", 1), fit(2, "minibatch", 3)]), {})
    root = tmp_path / "artifacts"
    for index, axis, global_step in ((1, "época", 1), (2, "minibatch", 3)):
        loss = json.loads((root / f"data/training/fit-{index:06}/loss.json").read_text())
        assert loss["x_label"] == axis
        assert loss["series"] == [{"name": "train:log_loss", "points": [[1, 0.2], [2, 0.1]]}]
        assert loss["global_steps"] == [[1, global_step], [2, global_step + 1]]
        assert loss["fit"]["checkpoint"]["file"] == f"models/{index}.npy"
        assert "<svg" in (root / f"plots/training/fit-{index:06}/loss.svg").read_text()


def test_multiclass_plots_select_macro_series_and_keep_detector_events():
    figures = {fig.name: fig for fig in binary_figures({
        "classes": ["follow", "friend", "block", "mute"],
        "windows": [{
            "end_seq": 100, "accuracy": 0.8, "macro_roc_auc": 0.9, "macro_pr_auc": 0.7,
            "macro_precision": 0.65, "macro_recall": 0.6,
        }],
        "alarms": [{"seq": 85}, {"seq": 94}],
    })}
    assert figures["ranking"].data["series"] == [
        {"name": "macro_roc_auc", "points": [[100, 0.9]]},
        {"name": "macro_pr_auc", "points": [[100, 0.7]]},
    ]
    assert [line["name"] for line in figures["precision_recall"].data["series"]] == [
        "macro_precision", "macro_recall",
    ]
    assert "class_balance" not in figures
    assert [event["x"] for event in figures["timeline"].data["markers"]] == [85, 94]


def test_campaign_comparison_uses_completed_runs_and_separate_partitions(tmp_path):
    units = [
        {"run_id": "a", "labels": {"policy": "frozen"}, "status": "SUCCEEDED",
         "metrics": {"accuracy": 0.4}, "holdout": {"accuracy": 0.7}},
        {"run_id": "b", "labels": {"policy": "online"}, "status": "SUCCEEDED",
         "metrics": {"accuracy": 0.6}, "holdout": {"accuracy": 0.5}},
        {"run_id": "c", "labels": {"policy": "incremental"}, "status": "FAILED"},
    ]
    write_zia_campaign_artifacts(tmp_path, units)
    root = tmp_path / "artifacts"
    for scope, expected in (("prequential", [0.4, 0.6]), ("holdout", [0.7, 0.5])):
        values = json.loads((root / f"data/comparison/{scope}/accuracy.json").read_text())
        assert values["labels"] == ["frozen", "online"]
        assert values["values"] == expected
    assert json.loads((root / "data/comparison/policies.json").read_text()) == units


def test_pyfunc_loader_imports_from_packaged_archive(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from continual_bench.evaluation.zia_publication import pyfunc_code_paths

    loader, archive = pyfunc_code_paths(tmp_path)
    shutil.copy(loader, tmp_path)
    script = """
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from zia_mlflow_loader import _load_pyfunc

state = Path(sys.argv[1])
checkpoint = state / "weights.npy"
np.save(checkpoint, np.zeros((2, 4)), allow_pickle=False)
(state / "model-metadata.json").write_text(json.dumps({
    "features": ["f"], "classes": ["a", "b", "c", "d"], "model": "multinomial_logistic",
    "payload": checkpoint.name, "state_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
}))
predictor = _load_pyfunc(state)
np.testing.assert_array_equal(predictor.predict(pd.DataFrame({"f": [0.0]})), [[0.25] * 4])
assert sys.argv[2] in sys.modules["continual_bench.models.zia"].__file__
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path), archive], check=True)
