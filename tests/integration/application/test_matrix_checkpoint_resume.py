import json
import pickle

import pytest

from continual_bench.application.matrix import _execute_case
from continual_bench.config import Config, EpochConfig, TypedConfig


class InstanceLost(BaseException):
    """Simulate a process disappearing after a durable checkpoint."""


class FixtureCheckpoint:
    """Copy both metadata and evidence as if restoring on a new EC2 instance."""

    def __init__(self, root, *, interrupt_after=None, lose_after=None):
        self.root = root
        self.interrupt_after = interrupt_after
        self.lose_after = lose_after
        self.boundary = 0
        self.archive = None

    def due(self):
        self.boundary += 1
        if self.boundary == self.lose_after:
            raise InstanceLost
        return self.boundary % 15 == 0 or self.requested()

    def requested(self):
        return self.boundary == self.interrupt_after

    def save(self, state, *, force=False):
        assert not self.requested() or force
        self.archive = pickle.dumps(
            (
                state,
                {
                    path.relative_to(self.root): path.read_bytes()
                    for path in self.root.rglob("*")
                    if path.is_file() and path.name != ".lock"
                },
            )
        )

    def load(self):
        if self.archive is None:
            return None
        state, files = pickle.loads(self.archive)
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return state


def evaluate(config, output, tmp_path, checkpoint=None):
    return _execute_case(config, output, str(tmp_path / "cache"), None, checkpoint=checkpoint)


def resumed_case(config, tmp_path, **failure):
    before = tmp_path / "instance-before"
    after = tmp_path / "instance-after"
    first = FixtureCheckpoint(before, **failure)
    with pytest.raises(InterruptedError if "interrupt_after" in failure else InstanceLost):
        evaluate(config, before, tmp_path, first)
    assert first.archive is not None
    saved, _ = pickle.loads(first.archive)
    replacement = FixtureCheckpoint(after)
    replacement.archive = first.archive
    return evaluate(config, after, tmp_path, replacement), saved["state"]


def without_timer(rows):
    return [{key: value for key, value in row.items() if key != "seconds"} for row in rows]


@pytest.mark.parametrize("mode", ["streaming", "incremental", "offline"])
def test_binary_matrix_restores_metrics_delayed_labels_and_model_journal(mode, tmp_path):
    config = Config.model_validate(
        {
            "mode": mode,
            "warmup": 10,
            "metric_window": 10,
            "batch_size": 4,
            "checkpoint_every": 100,
            "dataset": {"name": "sea", "samples": 100, "label_delay": 5},
            "policy": {"name": "none"}
            if mode == "offline"
            else {
                "name": "periodic",
                "every": 10,
                "min_samples": 10,
                "min_window": 10,
            },
        }
    )
    expected = evaluate(config, tmp_path / "reference", tmp_path)
    actual, state = resumed_case(config, tmp_path, interrupt_after=18)
    assert 0 < state.next_seq < actual["observed"]
    assert expected["metrics"] == actual["metrics"]
    assert expected["windows"] == actual["windows"]
    assert without_timer(expected["updates"]) == without_timer(actual["updates"])
    assert expected.get("prediction_digest") == actual.get("prediction_digest")
    assert expected.get("split") == actual.get("split")
    assert actual["model_versions"]["count"] == expected["model_versions"]["count"]
    after = tmp_path / "instance-after" / "binary" / actual["run_key"]
    versions = json.loads((after / "model-versions.json").read_text())["versions"]
    assert [version["ordinal"] for version in versions] == list(range(len(versions)))
    assert [version["parent_version"] for version in versions[1:]] == [
        version["version_id"] for version in versions[:-1]
    ]


def test_typed_matrix_recovers_without_notice_with_pending_labels_and_replay(tmp_path):
    config = TypedConfig.model_validate(
        {
            "policy": "periodic",
            "warmup": 5,
            "every": 5,
            "window": 10,
            "max_samples": 60,
            "dataset": {"name": "sea", "samples": 100, "label_delay": 6},
        }
    )
    expected = evaluate(config, tmp_path / "reference", tmp_path)
    actual, state = resumed_case(config, tmp_path, lose_after=49)
    assert state.pending
    assert state.replay and state.versions
    assert expected["metrics"] == actual["metrics"]
    assert expected["prediction_digest"] == actual["prediction_digest"]
    assert without_timer(expected["updates"]) == without_timer(actual["updates"])
    assert actual["model_versions"]["count"] == expected["model_versions"]["count"]
    reference = tmp_path / "reference" / "typed" / expected["run_key"]
    after = tmp_path / "instance-after" / "typed" / actual["run_key"]
    assert (reference / "predictions.json").read_bytes() == (after / "predictions.json").read_bytes()


def test_epoch_matrix_restores_partial_holdout_and_completed_passes(tmp_path):
    config = EpochConfig(dataset="sea", examples=50, epochs=3)
    expected = evaluate(config, tmp_path / "reference", tmp_path)
    actual, state = resumed_case(config, tmp_path, interrupt_after=95)
    assert len(state.epochs) == 1
    assert state.current.training_offset == len(expected["train_ids"])
    assert state.current.predictions
    for key in ("train_ids", "test_ids"):
        assert expected[key] == actual[key]
    for key in ("epoch", "metrics", "prediction_digest", "train_presentations"):
        assert [epoch[key] for epoch in expected["epochs"]] == [epoch[key] for epoch in actual["epochs"]]
    fit = actual["training_history"][0]
    assert fit["training_presentations"] == 3 * len(expected["train_ids"])
    assert [step["step"] for step in fit["observation_steps"]] == list(
        range(1, fit["training_presentations"] + 1)
    )


def test_binary_checkpoint_rejects_a_new_evaluation_contract(tmp_path):
    config = Config.model_validate(
        {
            "warmup": 10,
            "dataset": {"name": "sea", "samples": 100, "label_delay": 2},
        }
    )
    checkpoint = FixtureCheckpoint(tmp_path / "instance-before", interrupt_after=18)
    with pytest.raises(InterruptedError):
        evaluate(config, checkpoint.root, tmp_path, checkpoint)
    checkpoint.root = tmp_path / "instance-after"
    varied = config.model_copy(update={"metric_window": 9})
    with pytest.raises(ValueError, match="resume environment/config/dataset/transport/plugin mismatch"):
        evaluate(varied, checkpoint.root, tmp_path, checkpoint)
