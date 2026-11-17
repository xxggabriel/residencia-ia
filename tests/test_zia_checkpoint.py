import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from continual_bench.config import TrackingConfig
from continual_bench.datasets.zia import DATASETS, FORMAT, sha256_file


class PowerLoss(RuntimeError):
    pass


class Snapshot:
    """Stand-in for a durable store holding the last atomically committed snapshot."""

    def __init__(self, output, durable, *, interval=2, stop_after=None):
        self.output = output
        self.durable = durable
        self.interval = interval
        self.stop_after = stop_after
        self.notice = False
        self.boundaries = 0
        self.last_force = False

    def due(self):
        self.boundaries += 1
        return self.notice or self.boundaries % self.interval == 0

    def requested(self):
        return self.notice

    def save(self, state, *, force=False):
        self.last_force = force
        self.durable["state"] = pickle.dumps(state, protocol=5)
        self.durable["files"] = {
            path.relative_to(self.output): path.read_bytes()
            for path in self.output.rglob("*")
            if path.is_file()
        }
        if self.stop_after is not None and self.stop_after(state):
            raise PowerLoss("instance disappeared after a durable checkpoint")

    def load(self):
        if not self.durable:
            return None
        for key, body in self.durable["files"].items():
            target = self.output / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        return pickle.loads(self.durable["state"])


@pytest.fixture()
def stream_fixture(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    ids = np.arange(10_000)
    timestamps = ids // 13
    for dataset in ("social", "fraud"):
        size = len(DATASETS[dataset]["classes"])
        classes = ids % size
        # The feature/label relation changes twice to exercise stateful drift and retraining.
        label = ((classes + ids // 3300) % size).astype(np.uint8)
        path = tmp_path / dataset / "smoke.parquet"
        path.parent.mkdir()
        pq.write_table(
            pa.table(
                {
                    "event_time": timestamps,
                    "label": label,
                    "feature": classes.astype(np.float32),
                }
            ),
            path,
            row_group_size=701,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "scope": "smoke",
                    "source": DATASETS[dataset],
                    "format": FORMAT,
                    "rows": len(ids),
                    "bytes": path.stat().st_size,
                    "features": ["feature"],
                    "sha256": sha256_file(path),
                }
            )
        )
    settings = TrackingConfig.model_validate(
        yaml.safe_load(Path("configs/base/tracking.yaml").read_text())
    ).model_dump()
    return tmp_path, settings


def deterministic_window(window):
    return {k: v for k, v in window.items() if not k.startswith("prediction_")}


def fit_lineage(fit):
    keys = (
        "fit_id",
        "start_seq",
        "end_seq",
        "action",
        "checkpoint",
        "source_model_version",
        "promotion_status",
        "deployed_version",
        "replaces_model_version",
        "training_presentations",
    )
    return ({k: fit[k] for k in keys if k in fit}, [step["global_step"] for step in fit["steps"]])


def read_predictions(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = json.loads((path / "predictions-manifest.json").read_bytes())
    predictions = pa.concat_tables(
        [pq.read_table(path / "predictions" / part["file"]) for part in manifest["parts"]]
    ).combine_chunks()
    return predictions, manifest["fading_accuracy"]


def test_seek_keeps_timestamp_ties_and_the_same_holdout_split(stream_fixture):
    from continual_bench.evaluation.zia import batches

    root, _ = stream_fixture
    path = root / "social" / "smoke.parquet"
    original = list(batches(path, 1000, 9000))
    start = 0
    for i, block in enumerate(original):
        resumed = list(batches(path, 1000, 9000, start=start, use_threads=True))
        assert len(resumed) == len(original) - i
        for expected, actual in zip(original[i:], resumed, strict=True):
            assert actual[-1] == expected[-1]
            for a, b in zip(actual[:3], expected[:3], strict=True):
                np.testing.assert_array_equal(a, b)
        start += len(block[0])


@pytest.mark.parametrize(
    ("dataset", "policy", "mode"),
    [
        ("social", "periodic", "notice"),
        ("social", "online", "loss"),
        ("social", "drift_adwin", "loss"),
        ("social", "drift_ddm", "loss"),
        ("social", "upf", "matrix_loss"),
        ("social", "upf", "matrix_loss_then_evaluation_loss"),
        ("social", "frozen", "final_loss"),
        ("fraud", "incremental", "loss"),
    ],
)
def test_resumes_with_identical_predictions_metrics_and_lineage(stream_fixture, dataset, policy, mode):
    from continual_bench.evaluation.zia import run

    if dataset == "fraud":
        pytest.importorskip("xgboost")
    root, settings = stream_fixture
    durable = {}
    parameters = (root, dataset, "smoke", policy)
    iterative = {"max_epochs": 4, "patience": 2}
    reference_path = root / "reference"
    reference = run(*parameters, reference_path, settings, iterative_config=iterative)
    interrupted_path = root / "interrupted"
    matrix_loss = mode.startswith("matrix_loss")

    def lose_when_saved(saved):
        if saved["phase"] == "matrix":
            return matrix_loss
        return mode == "final_loss" and saved["state"]["stream"].rows == 10_000

    checkpoint = Snapshot(interrupted_path, durable, stop_after=lose_when_saved)

    def progress(_step, metrics):
        if 4_900 <= metrics.get("rows_processed", -1) < 5_100:
            if mode == "notice":
                checkpoint.notice = True
            elif mode == "loss":
                raise PowerLoss("instance disappeared after an unconfirmed block")

    expected_error = InterruptedError if mode == "notice" else PowerLoss
    with pytest.raises(expected_error):
        run(
            *parameters,
            interrupted_path,
            settings,
            iterative_config=iterative,
            progress=progress,
            checkpoint=checkpoint,
        )
    last = pickle.loads(durable["state"])
    if matrix_loss:
        assert last["phase"] == "matrix"
        assert len(last["state"][0].rows) == 2
    else:
        assert last["phase"] == "evaluation"
        if mode == "final_loss":
            assert last["state"]["stream"].rows == 10_000
        else:
            expected = 5 if mode == "notice" else 4
            assert last["state"]["stream"].next_step == expected
        assert checkpoint.last_force is (mode != "loss")
    if mode == "matrix_loss_then_evaluation_loss":
        next_attempt = root / "retry"
        with pytest.raises(PowerLoss):
            run(
                *parameters,
                next_attempt,
                settings,
                iterative_config=iterative,
                checkpoint=Snapshot(
                    next_attempt,
                    durable,
                    stop_after=lambda saved: saved["phase"] == "evaluation",
                ),
            )
        assert pickle.loads(durable["state"])["state"]["stream"].next_step > 0

    output = root / "resumed"
    resumed_progress = []
    resumed = run(
        *parameters,
        output,
        settings,
        iterative_config=iterative,
        checkpoint=Snapshot(output, durable),
        progress=lambda step, _: resumed_progress.append(step),
    )
    if mode == "final_loss":
        assert resumed_progress == []
    else:
        assert resumed_progress[0] > 0
    assert resumed_progress == sorted(set(resumed_progress))
    assert {sample["segment"] for sample in resumed["system_metrics"]["samples"]} >= {0, 1}
    assert not (output / ".zia-resume").exists()
    for name in ("alarms", "holdout", "prequential_confusion", "holdout_confusion", "class_counts"):
        assert resumed[name] == reference[name]
    if policy.startswith("drift"):
        cursor = last["state"]["stream"].rows
        assert any(alarm["seq"] <= cursor for alarm in reference["alarms"])
        assert any(alarm["seq"] > cursor for alarm in reference["alarms"])
    assert resumed["contract"]["holdout_start"] == reference["contract"]["holdout_start"]
    assert [fit_lineage(fit) for fit in resumed["training_history"]] == [
        fit_lineage(fit) for fit in reference["training_history"]
    ]
    assert [deterministic_window(w) for w in resumed["windows"]] == [
        deterministic_window(w) for w in reference["windows"]
    ]
    actual_pred, actual_fading = read_predictions(output)
    expected_pred, expected_fading = read_predictions(reference_path)
    assert actual_pred.equals(expected_pred)
    assert actual_fading == expected_fading
    assert actual_pred["seq"].to_pylist() == list(range(reference["metrics"]["initial_train_rows"], 10_000))
    for key in ("accuracy", "log_loss", "update_count", "retrain_count", "training_presentations"):
        assert resumed["metrics"][key] == reference["metrics"][key]
    if matrix_loss:
        assert resumed["retraining"]["deployed"] == reference["retraining"]["deployed"]
