"""JSON model payloads are weights even when their extension is also used by reports."""

import pytest

from continual_bench.infrastructure.artifact_policy import verify_metrics_only_files


@pytest.mark.parametrize(
    "relative",
    [
        "models/model-0001.json",
        "candidates/fit-0001.json",
        "model/native.json",
        ".zia-resume/progress.json",
        "checkpoints/state.json",
    ],
)
def test_metrics_only_rejects_native_json_state(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"learner": {"gradient_booster": {}}}')
    with pytest.raises(ValueError, match="model or checkpoint state"):
        verify_metrics_only_files(tmp_path)


def test_logical_model_metadata_and_metric_cache_are_allowed(tmp_path):
    (tmp_path / "model-versions.json").write_text('[{"file": null, "bytes": 123}]')
    (tmp_path / "model-metadata.json").write_text('{"native_path": null, "payload": null}')
    (tmp_path / "matrix-metrics.json.gz").write_bytes(b"metric evidence")
    verify_metrics_only_files(tmp_path)


def test_metrics_only_rejects_symlinked_payload(tmp_path):
    (tmp_path / "external-model.json").write_text("{}")
    (tmp_path / "report.json").symlink_to(tmp_path / "external-model.json")
    with pytest.raises(ValueError, match="symlinked"):
        verify_metrics_only_files(tmp_path)
