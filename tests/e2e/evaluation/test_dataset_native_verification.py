import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

from scripts import verify_dataset_suite as qa

MARKERS = {"tracking_v1_status": "complete", "zia_artifacts_status": "complete", "benchmark_complete": "true"}


class HistoryClient:
    def __init__(self, points, *, markers=None, fault=None, coordinate_first_wave=False):
        self.markers = dict(MARKERS if markers is None else markers)
        self.histories = defaultdict(list)
        for key, step, value in points:
            if value is not None:
                self.histories[key].append(SimpleNamespace(step=step, value=value))
        self.fault = fault
        self.coordinate_first_wave = coordinate_first_wave
        self.lock, self.first_wave = threading.Lock(), threading.Event()
        self.calls, self.active, self.maximum_active = [], 0, 0

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"), data=SimpleNamespace(tags=self.markers)
        )

    def get_metric_history(self, run_id, key):
        with self.lock:
            self.calls.append(key)
            number = len(self.calls)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            if number == 4:
                self.first_wave.set()
        try:
            if self.coordinate_first_wave and number <= 4:
                assert self.first_wave.wait(timeout=5), "the first four history reads did not start together"
            values = list(self.histories[key])
            if self.fault and key == "tracking_v1/dense_0":
                if self.fault == "request_error":
                    raise RuntimeError("history request failed")
                if self.fault == "missing":
                    values.pop(263)
                elif self.fault == "extra":
                    values.append(SimpleNamespace(step=1000, value=0.0))
                elif self.fault == "duplicate":
                    values.append(values[263])
                elif self.fault == "value":
                    values[263] = SimpleNamespace(step=263, value=values[263].value + 1.0)
                elif self.fault == "none":
                    values[263] = SimpleNamespace(step=263, value=None)
                elif self.fault == "nonfinite":
                    values[263] = SimpleNamespace(step=263, value=float("nan"))
            return values
        finally:
            with self.lock:
                self.active -= 1


def native_fixture(tmp_path, monkeypatch, seed):
    tracking = {
        "version": "residency-metrics-v1",
        "metrics": {"accuracy": 0.5, "unavailable": None},
        "series": {
            f"dense_{index}": [
                *({"step": step, "value": (seed + index + step) / 1000} for step in range(600)),
                {"step": 1000, "value": None},
            ]
            for index in range(8)
        },
        "availability": {},
        "settings": {"seed": seed},
    }
    qa.save(tmp_path / "predictions-manifest.json", {})
    monkeypatch.setattr(qa, "tracking_report", lambda report, index: {"tracking": tracking})
    points = qa.metric_points(tracking)
    return {}, {"complete": True, "mlflow_run_id": f"run-{seed}"}, points


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_native_history_verifies_every_point_with_at_most_four_reads_and_committed_markers(
    tmp_path, monkeypatch, seed
):
    report, receipt, points = native_fixture(tmp_path, monkeypatch, seed)
    client = HistoryClient(points, coordinate_first_wave=True)
    result = qa.verify_native(tmp_path, report, receipt, client)
    assert result["points"] == 4801
    assert result["series"] == 9
    assert result["undefined_catalog_points_not_logged_as_numbers"] == 9
    assert client.maximum_active == 4 and client.active == 0
    assert set(client.calls) == set(client.histories)
    assert result["publication_markers"] == MARKERS
    assert result["required_publication_markers"] == MARKERS
    assert result["run_status"] == "FINISHED"
    assert all(result["checks"].values())
    assert result["history_read_workers"] == 4


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("marker", list(MARKERS))
@pytest.mark.parametrize("incomplete", ["publishing", None])
def test_finished_run_with_an_uncommitted_publication_marker_is_rejected_before_history_reads(
    tmp_path, monkeypatch, seed, marker, incomplete
):
    report, receipt, points = native_fixture(tmp_path, monkeypatch, seed)
    client = HistoryClient(points, markers={**MARKERS, marker: incomplete})
    with pytest.raises(ValueError, match=f"publication marker {marker} is incomplete"):
        qa.verify_native(tmp_path, report, receipt, client)
    assert client.calls == [] and client.maximum_active == 0


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize(
    "fault,exception,message",
    [
        ("missing", ValueError, "axis differs"),
        ("extra", ValueError, "axis differs"),
        ("duplicate", ValueError, "coordinate duplicated"),
        ("value", ValueError, "native metric differs"),
        ("none", ValueError, "not finite numeric"),
        ("nonfinite", ValueError, "not finite numeric"),
        ("request_error", RuntimeError, "history request failed"),
    ],
)
def test_parallel_history_readback_does_not_hide_one_bad_point_or_request(
    tmp_path, monkeypatch, seed, fault, exception, message
):
    report, receipt, points = native_fixture(tmp_path, monkeypatch, seed)
    client = HistoryClient(points, fault=fault)
    with pytest.raises(exception, match=message):
        qa.verify_native(tmp_path, report, receipt, client)
    assert client.maximum_active <= 4 and client.active == 0


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_explicit_serial_history_mode_preserves_the_same_complete_checks(tmp_path, monkeypatch, seed):
    report, receipt, points = native_fixture(tmp_path, monkeypatch, seed)
    client = HistoryClient(points)
    result = qa.verify_native(tmp_path, report, receipt, client, workers=1)
    assert result["points"] == 4801 and result["history_read_workers"] == 1
    assert client.maximum_active == 1


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_marker_regression_during_history_reads_blocks_final_acceptance(tmp_path, monkeypatch, seed):
    report, receipt, points = native_fixture(tmp_path, monkeypatch, seed)

    class RegressingClient(HistoryClient):
        snapshots = 0

        def get_run(self, run_id):
            self.snapshots += 1
            if self.snapshots > 1:
                self.markers["tracking_v1_status"] = "publishing"
            return super().get_run(run_id)

    client = RegressingClient(points)
    with pytest.raises(ValueError, match="tracking_v1_status is incomplete"):
        qa.verify_native(tmp_path, report, receipt, client)
    assert client.snapshots == 2 and client.calls


@pytest.mark.parametrize("workers", [0, -1, 5, True, None])
def test_invalid_history_concurrency_is_rejected_before_native_requests(tmp_path, monkeypatch, workers):
    report, receipt, points = native_fixture(tmp_path, monkeypatch, 42)
    client = HistoryClient(points)
    with pytest.raises(ValueError, match="between 1 and 4"):
        qa.verify_native(tmp_path, report, receipt, client, workers=workers)
    assert client.calls == []
