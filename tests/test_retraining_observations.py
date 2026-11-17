import numpy as np

from continual_bench.evaluation import cara_replication
from continual_bench.evaluation.retraining_observations import candidate_report


def test_cara_counts_detector_alarms_independently_of_coalesced_retrains(monkeypatch):
    class Detector:
        def update(self, loss):
            self.drift_detected = bool(loss)

    monkeypatch.setattr(cara_replication, "drift_detectors", lambda _: [("adwin", Detector())])
    phase = cara_replication.Phase(
        np.zeros((2, 2)), np.zeros((2, 2)),
        [[np.array([True, False, True])], [np.array([False]), np.array([True])]],
    )
    alarms = []
    assert cara_replication.drift_schedules(phase, "river", alarms=alarms) == {"adwin": [0]}
    assert alarms == [
        {"seq": 1, "offset_in_batch": 1, "detector": "adwin"},
        {"seq": 1, "offset_in_batch": 3, "detector": "adwin"},
    ]


def test_candidate_identity_preserves_promotion_and_detector_evidence(monkeypatch, tmp_path):
    class Learner:
        observation_model = "fixture"
        observation_threshold = 0.5
        observation_step_kind = "fixture_fit"

        def observation_predict(self, state, x, positive):
            return x[:, 0].astype(int), x[:, 0]

        def observation_checkpoint(self, state):
            return bytes([state]), "fixture"

        def observation_durations(self, index, elapsed):
            return [elapsed]

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "stub")
    monkeypatch.setenv("BENCH_STATE", str(tmp_path))
    monkeypatch.setattr("continual_bench.evaluation.training_evidence.write_training_artifacts",
                        lambda directory, report: {})
    monkeypatch.setattr("continual_bench.infrastructure.benchmark_tracking.publish_local_benchmark",
                        lambda directory, report, **kwargs: {"run_id": report["run_key"]})
    x = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    splits = [(x, y, x.copy(), y.copy()) for _ in range(2)]
    alarm = {"seq": 1, "offset_in_batch": 1, "detector": "adwin"}

    def run(initial, *, backend="river", alarms=(), seconds=1.0, gamma=0.2):
        return candidate_report(
            "covcon-d", 0, splits, [1, 2], Learner(), [seconds] * 2,
            protocol="CARA-paper-showcase-candidates",
            initial=initial, evaluation={"rbf_gamma": gamma}, duration_before_tracking=seconds * 2,
            drift={"detector_backend": backend, "alarms": list(alarms), "coordinate": "batch_number"},
        )

    first = run(0)
    assert run(0, seconds=10.0) == first  # Time observations do not rebind a completed identity.
    second = run(1)
    assert first["run_key"] != second["run_key"]
    assert [fit["deployed_version"] for fit in first["training_history"]] == [0, None]
    assert [fit["deployed_version"] for fit in second["training_history"]] == [None, 1]
    assert run(0, backend="skmultiflow")["run_key"] != first["run_key"]
    assert run(0, gamma=0.3)["run_key"] != first["run_key"]
    drifted = run(0, alarms=[alarm])
    assert drifted["run_key"] != first["run_key"]
    assert drifted["tracking_observations"]["drift"]["alarms"] == [alarm]
