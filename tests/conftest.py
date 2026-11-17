import pytest

_RENDER_MARKERS = ("observation_pngs", "services", "benchmark")


@pytest.fixture(autouse=True)
def render_observation_pngs(request, monkeypatch):
    # Keep full artifacts in dedicated rendering tests and integration runs.
    # The quick tests still generate the observation data, manifests and HTML.
    if not any(request.node.get_closest_marker(name) for name in _RENDER_MARKERS):
        monkeypatch.setattr(
            "continual_bench.infrastructure.training_figures.matplotlib_available",
            lambda: False,
        )
