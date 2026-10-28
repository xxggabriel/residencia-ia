import copy
import json
import pickle

import pytest

from continual_bench.domain.continual import METRICS, ContinualEvaluation


class ReferenceModel:
    def __init__(self):
        self.learned = []
        self.predict_calls = 0

    def learn(self, x, y):
        self.learned.append((copy.deepcopy(x), y))
        x["mutated_by_learning"] = True

    def predict(self, x):
        self.predict_calls += 1
        index = x["index"]
        x["mutated_by_prediction"] = True
        return 0.9 if index < 5 else 0.1


class TableModel(ReferenceModel):
    def __init__(self):
        super().__init__()
        self.accuracies = {}

    def predict(self, x):
        self.predict_calls += 1
        return 0.9 if x["index"] < round(self.accuracies.get(x["task"], 0) * 10) else 0.1


def probes(tasks=3):
    return {task: [({"task": task, "index": index}, True) for index in range(10)]
            for task in range(tasks)}


def add_episode(subject, model, task, accuracies):
    model.accuracies = dict(enumerate(accuracies))
    start = subject.next_seq
    for seq in range(start, subject.task_ends[task]):
        subject.after_label(seq, {"source_seq": seq}, True, model)


def without_seconds(value):
    if isinstance(value, dict):
        return {key: without_seconds(item) for key, item in value.items()
                if "seconds" not in key}
    if isinstance(value, list):
        return [without_seconds(item) for item in value]
    return value


def test_equations_preserve_negative_forgetting_positive_bwt_and_diagonal_fwt():
    subject = ContinualEvaluation(ReferenceModel, probes(), [2, 4, 6])
    model = TableModel()
    add_episode(subject, model, 0, [0.4])
    first = subject.report()
    assert first["metrics"]["cl_average_accuracy"] == 0.4
    assert first["metrics"]["cl_average_incremental_accuracy"] == 0.4
    assert first["metrics"]["cl_intransigence"] == pytest.approx(0.1)
    for name in ("cl_forgetting", "cl_backward_transfer", "cl_forward_transfer"):
        assert first["metrics"][name] is None
        assert first["availability"][name]["reason"] == "insufficient_tasks"
    add_episode(subject, model, 1, [0.6, 0.8])
    second = subject.report()["metrics"]
    assert second["cl_average_accuracy"] == pytest.approx(0.7)
    assert second["cl_average_incremental_accuracy"] == pytest.approx(0.55)
    assert second["cl_forgetting"] == pytest.approx(-0.2)
    assert second["cl_backward_transfer"] == pytest.approx(0.2)
    assert second["cl_intransigence"] == pytest.approx(-0.3)
    assert second["cl_forward_transfer"] == pytest.approx(0.3)
    add_episode(subject, model, 2, [0.5, 0.9, 0.6])
    result = subject.report()
    assert result["metrics"]["cl_forgetting"] == pytest.approx(0)
    assert result["metrics"]["cl_backward_transfer"] == pytest.approx(0.1)
    assert result["metrics"]["cl_forward_transfer"] == pytest.approx(0.2)
    assert result["metrics"]["cl_average_accuracy"] == pytest.approx(2 / 3)
    assert result["metrics"]["cl_average_incremental_accuracy"] == pytest.approx((0.4 + 0.7 + 2 / 3) / 3)
    assert result["matrix"][2]["accuracies"] == {"0": 0.5, "1": 0.9, "2": 0.6}
    assert result["series"]["cl_forward_transfer"] == [
        {"step": 0, "value": None}, {"step": 1, "value": pytest.approx(0.3)},
        {"step": 2, "value": pytest.approx(0.2)},
    ]
    assert result["reference_train_samples"] == 18
    assert len(subject.reference_models) == 6
    assert result["overhead_seconds"] == result["probe_seconds"] + result["reference_train_seconds"]
    json.dumps(result, allow_nan=False)


def test_reference_budget_is_per_model_and_does_not_partially_train():
    subject = ContinualEvaluation(ReferenceModel, probes(), [2, 4, 6], reference_budget=3)
    model = TableModel()
    for task in range(3):
        add_episode(subject, model, task, [0.6] * (task + 1))
    report = subject.report()
    assert report["metrics"]["cl_intransigence"] is None
    assert report["availability"]["cl_intransigence"]["reason"] == "reference_budget_exceeded"
    assert report["metrics"]["cl_forward_transfer"] == pytest.approx(0.1)
    assert report["matrix"][2]["reference_joint"]["train_samples"] == 0
    assert report["matrix"][2]["reference_joint"]["required_train_samples"] == 6
    assert report["reference_train_samples"] == 8
    assert [(item["name"], item["task"]) for item in subject.reference_models] == [
        ("task_only", 0), ("joint_prefix_one_pass", 0), ("task_only", 1), ("task_only", 2),
    ]
    assert all(item["train_samples"] <= 3 for item in subject.reference_models)


def test_forgetting_maximum_includes_better_prelearning_evaluation_of_an_old_episode():
    subject = ContinualEvaluation(ReferenceModel, probes(), [2, 4, 6])
    model = TableModel()
    add_episode(subject, model, 0, [0.4, 0.9, 0.2])
    first = subject.report()
    assert first["matrix"][0]["accuracies"] == {"0": 0.4, "1": 0.9, "2": 0.2}
    assert first["metrics"]["cl_average_accuracy"] == 0.4  # Future probes are excluded from AA.
    assert first["metrics"]["cl_forward_transfer"] is None  # Future score is not Wang's FWT.
    assert first["released_labels"] == 2
    assert all(item["seq_range"] == [0, 2] for item in subject.reference_models)
    assert all([x["source_seq"] for x, y in item["model"].learned] == [0, 1]
               for item in subject.reference_models)
    add_episode(subject, model, 1, [0.6, 0.7, 0.3])
    add_episode(subject, model, 2, [0.5, 0.6, 0.8])
    report = subject.report()
    # Task 1's maximum is .9 BEFORE its learning, rather than its diagonal .7.
    assert report["metrics"]["cl_forgetting"] == pytest.approx(((0.6 - 0.5) + (0.9 - 0.6)) / 2)
    assert report["metrics"]["cl_backward_transfer"] == pytest.approx(0)
    assert report["metrics"]["cl_forward_transfer"] == pytest.approx(((0.7 - 0.5) + (0.8 - 0.5)) / 2)
    assert report["metrics"]["cl_average_incremental_accuracy"] == pytest.approx(
        (0.4 + 0.65 + (0.5 + 0.6 + 0.8) / 3) / 3
    )
    assert report["contract"]["matrix_scope"] == (
        "all_declared_synthetic_episode_probes; future scores diagnostic only no adaptation"
    )
    assert "before the episode was learned" in report["contract"]["forgetting_max_scope"]


@pytest.mark.parametrize("options,reason", [
    ({"references_enabled": False}, "references_disabled"),
    ({"reference_budget": 0}, "reference_budget_exceeded"),
    ({"reference_budget": 1}, "reference_budget_exceeded"),
])
def test_unavailable_references_are_none_with_reasons_and_no_training(options, reason):
    def forbidden_factory():
        raise AssertionError("unavailable references must not be constructed")

    subject = ContinualEvaluation(forbidden_factory, probes(2), [2, 4], **options)
    model = TableModel()
    add_episode(subject, model, 0, [0.7])
    add_episode(subject, model, 1, [0.7, 0.8])
    report = subject.report()
    assert report["metrics"]["cl_average_accuracy"] == 0.75
    for name in ("cl_intransigence", "cl_forward_transfer"):
        assert report["metrics"][name] is None
        assert report["availability"][name] == {"available": False, "reason": reason}
    assert report["reference_train_samples"] == 0
    assert report["reference_train_seconds"] == 0
    assert not subject.reference_models


def test_recurrent_concept_keeps_distinct_episodes_and_task_training_views():
    repeated = probes()
    repeated[2] = copy.deepcopy(repeated[0])
    subject = ContinualEvaluation(ReferenceModel, repeated, [2, 4, 6])
    model = TableModel()
    for task in range(3):
        add_episode(subject, model, task, [0.8] * (task + 1))
    assert [row["task"] for row in subject.report()["matrix"]] == [0, 1, 2]
    assert set(subject.report()["matrix"][2]["accuracies"]) == {"0", "1", "2"}
    singles = [item for item in subject.reference_models if item["name"] == "task_only"]
    assert [item["seq_range"] for item in singles] == [[0, 2], [2, 4], [4, 6]]
    assert [[x["source_seq"] for x, y in item["model"].learned] for item in singles] == [
        [0, 1], [2, 3], [4, 5],
    ]


def test_probe_and_reference_prediction_leave_real_model_and_input_examples_unchanged():
    probe_data = probes(1)
    original_probes = copy.deepcopy(probe_data)
    subject = ContinualEvaluation(ReferenceModel, probe_data, [2])
    model = ReferenceModel()
    model.learn({"main": True}, True)
    before = pickle.dumps(model)
    x = {"source_seq": 0, "nested": [1]}
    subject.after_label(0, x, True, model)
    x["nested"].append(2)
    subject.after_label(1, {"source_seq": 1}, False, model)
    assert pickle.dumps(model) == before
    assert probe_data == original_probes == subject.probes
    assert subject.examples[0][0]["nested"] == [1]
    assert all(item["model"].predict_calls == 0 for item in subject.reference_models)
    assert all("mutated_by_learning" not in x for x, y in subject.examples)
    report = subject.report()
    report["matrix"][0]["accuracies"]["0"] = 0
    assert subject.report()["matrix"][0]["accuracies"]["0"] == 0.5


def test_only_explicit_released_calls_are_visible_and_references_wait_for_boundary():
    subject = ContinualEvaluation(ReferenceModel, probes(2), [2, 4])
    model = ReferenceModel()
    assert subject.report()["released_labels"] == 0
    assert all(item["reason"] == "no_completed_tasks" for item in subject.report()["availability"].values())
    subject.after_label(0, {"source_seq": 0}, True, model)
    assert subject.report()["released_labels"] == 1
    assert subject.report()["completed_tasks"] == 0
    assert not subject.reference_models
    subject.after_label(1, {"source_seq": 1}, False, model)
    assert subject.report()["completed_tasks"] == 1
    assert all(len(item["model"].learned) == 2 for item in subject.reference_models)
    with pytest.raises(ValueError, match="expected released label"):
        subject.after_label(3, {}, True, model)
    assert subject.report()["released_labels"] == 2


def test_pickle_resume_matches_uninterrupted_scientific_evidence_and_reference_models():
    def run(interrupt):
        subject = ContinualEvaluation(ReferenceModel, probes(), [2, 4, 6])
        model = TableModel()
        for seq in range(6):
            task = seq // 2
            model.accuracies = {j: 0.6 + 0.1 * task for j in range(task + 1)}
            subject.after_label(seq, {"source_seq": seq}, bool(seq % 2), model)
            if interrupt and seq in (0, 2, 4):
                subject, model = pickle.loads(pickle.dumps((subject, model)))
        return subject

    reference, restored = run(False), run(True)
    assert without_seconds(reference.report()) == without_seconds(restored.report())
    assert [item["model"].learned for item in reference.reference_models] == [
        item["model"].learned for item in restored.reference_models
    ]


def test_missing_probe_is_unavailable_without_averaging_only_observed_tasks():
    subject = ContinualEvaluation(ReferenceModel, {0: probes(1)[0]}, [1, 2])
    model = ReferenceModel()
    subject.after_label(0, {}, True, model)
    first = subject.report()
    assert first["matrix"][0]["accuracies"] == {"0": 0.5, "1": None}
    assert first["metrics"]["cl_average_accuracy"] == 0.5  # Missing future probe does not poison AA.
    subject.after_label(1, {}, True, model)
    report = subject.report()
    assert report["matrix"][1]["accuracies"] == {"0": 0.5, "1": None}
    for metric in ("cl_average_accuracy", "cl_average_incremental_accuracy", "cl_intransigence",
                   "cl_forward_transfer"):
        assert report["metrics"][metric] is None
        assert report["availability"][metric]["reason"] == "missing_probe"
    assert report["metrics"]["cl_backward_transfer"] == 0
    assert set(report["metrics"]) == set(METRICS)


@pytest.mark.parametrize("ends,budget", [([], None), ([2, 2], None), ([2, 1], None),
                                       ([0], None), ([True], None), ([1], -1), ([1], 1.5)])
def test_invalid_evaluation_contract_rejected(ends, budget):
    with pytest.raises(ValueError):
        ContinualEvaluation(ReferenceModel, {}, ends, reference_budget=budget)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probe_probabilities_rejected(probability):
    class InvalidModel:
        def predict(self, x):
            return probability

    subject = ContinualEvaluation(ReferenceModel, probes(1), [1], references_enabled=False)
    with pytest.raises(ValueError, match="probe probability"):
        subject.after_label(0, {}, True, InvalidModel())
