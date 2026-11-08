import json
from dataclasses import replace

import pytest

from continual_bench.core.contracts import TaskSample
from continual_bench.evaluation.epochs import EpochConfig, run_epochs, split_records


class Fixture:
    def __init__(self, official=False):
        self.official = official

    def manifest(self):
        return {"task": "multiclass", "version": "fixture-only", "source": "synthetic_fixture"}

    def samples_iter(self):
        for seq in range(100):
            yield TaskSample(str(seq), seq, seq, seq + 3, {"x": seq % 2}, str(seq % 2),
                             split="test" if self.official and seq >= 80 else "train")


def test_two_epochs_reuse_only_training_records_and_keep_test_isolated(tmp_path):
    learned = []

    class Model:
        def learn_one(self, x, y):
            learned.append((x, y))

        def predict_one(self, x):
            return str(x["x"])

    config = EpochConfig(dataset="sea")
    report = run_epochs(config, tmp_path, tmp_path, source=Fixture(official=True), model_factory=Model)
    assert report["unique_examples"] == 100 and len(report["epochs"]) == 2
    assert len(learned) == 160
    assert report["train_ids"] == [str(i) for i in range(80)]
    assert report["test_ids"] == [str(i) for i in range(80, 100)]
    assert all(e["metrics"]["accuracy"] == 1 for e in report["epochs"])
    repeated = run_epochs(config, tmp_path, tmp_path, source=Fixture(official=True), model_factory=Model)
    assert repeated == report
    assert len(learned) == 160


def test_stream_purge_and_repeated_training_do_not_count_as_new_stream_events(tmp_path):
    source = Fixture()
    train, test, contract = split_records(list(source.samples_iter()), source.manifest())
    assert len(train) == 77 and len(test) == 20 and contract["purged"] == 3
    assert all(row.available_at < test[0].timestamp for row in train)
    report = run_epochs(EpochConfig(dataset="sea"), tmp_path, tmp_path, source=source)
    assert [e["train_presentations"] for e in report["epochs"]] == [77, 77]
    assert "no prequential claim" in report["scope"]
    assert json.loads((tmp_path / report["run_key"] / "epoch-report.json").read_text()) == report


def test_bad_selection_and_failed_fit_never_create_completed_report(tmp_path):
    with pytest.raises(ValueError, match="exactly 101"):
        run_epochs(EpochConfig(dataset="sea", examples=101), tmp_path, tmp_path, source=Fixture())

    class Broken:
        def learn_one(self, *args):
            raise RuntimeError("injected fit failure")

    with pytest.raises(RuntimeError, match="injected"):
        run_epochs(EpochConfig(dataset="sea"), tmp_path, tmp_path, source=Fixture(), model_factory=Broken)
    assert not list(tmp_path.rglob("epoch-report.json"))


def test_replay_rejects_corrupted_materialized_samples(tmp_path):
    config = EpochConfig(dataset="sea")
    report = run_epochs(config, tmp_path, tmp_path, source=Fixture())
    (tmp_path / report["run_key"] / "samples.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="artifact checksum"):
        run_epochs(config, tmp_path, tmp_path, source=Fixture())


@pytest.mark.parametrize("change", [{"available_at": -1}, {"x": {"x": float("nan")}}, {"y": ""}])
def test_malformed_source_is_rejected_before_training(tmp_path, change):
    fixture = Fixture()
    rows = list(fixture.samples_iter())
    rows[0] = replace(rows[0], **change)
    fixture.samples_iter = lambda: iter(rows)
    with pytest.raises(ValueError):
        run_epochs(EpochConfig(dataset="sea"), tmp_path, tmp_path, source=fixture)
    assert not list(tmp_path.rglob("epoch-report.json"))


def test_runtime_fingerprint_tracks_dependency_versions():
    from continual_bench.evaluation.epochs import dataset_fingerprint

    left = {"code_sha256": "same", "python": "3.12", "packages": {"pillow": "13.0.0"}}
    right = {**left, "packages": {"pillow": "13.0.1"}}
    assert dataset_fingerprint(left) != dataset_fingerprint(right)


def test_failed_all_dataset_repeat_revokes_prior_success(tmp_path, monkeypatch):
    import continual_bench.evaluation.epochs as module

    receipt = tmp_path / "dataset-smoke-complete.json"
    receipt.write_text(json.dumps({"complete": True, "datasets": ["old-success"]}))

    def inaccessible(*args, **kwargs):
        raise FileNotFoundError("official source unavailable")

    monkeypatch.setattr(module, "run_epochs", inaccessible)
    with pytest.raises(FileNotFoundError, match="official source unavailable"):
        module.smoke_all(tmp_path, tmp_path)
    assert json.loads(receipt.read_text())["complete"] is False


def test_epoch_costs_count_repeated_presentations(tmp_path):
    report = run_epochs(EpochConfig(dataset="sea", costs={"train_sample": 2, "update": 3}),
                        tmp_path, tmp_path, source=Fixture(official=True))
    costs = report["costs"]
    assert costs["consumption"]["training_presentations"] == 160
    assert costs["metrics"]["cost_training_presentations"] == 320
    assert costs["metrics"]["cost_update_events"] == 6
    assert costs["metrics"]["cost_total"] == sum(e["costs"]["metrics"]["cost_total"]
                                               for e in report["epochs"])
