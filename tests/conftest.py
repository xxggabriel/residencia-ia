import pytest

_RENDER_MARKERS = ("observation_pngs", "services", "benchmark")


def pytest_addoption(parser):
    parser.addoption("--shard", type=int, default=None, help="0-based test shard index for CI parallelism")
    parser.addoption("--num-shards", type=int, default=1, help="Total number of test shards")


def pytest_collection_modifyitems(config, items):
    shard = config.getoption("--shard")
    num_shards = config.getoption("--num-shards")
    if shard is None or num_shards <= 1:
        return
    if not 0 <= shard < num_shards:
        raise pytest.UsageError(f"--shard must be in [0, {num_shards}), got {shard}")
    selected = [item for index, item in enumerate(items) if index % num_shards == shard]
    deselected = [item for index, item in enumerate(items) if index % num_shards != shard]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.fixture(autouse=True)
def render_observation_pngs(request, monkeypatch):
    # Keep full artifacts in dedicated rendering tests and integration runs.
    # The quick tests still generate the observation data, manifests and HTML.
    if not any(request.node.get_closest_marker(name) for name in _RENDER_MARKERS):
        monkeypatch.setattr(
            "continual_bench.infrastructure.training_figures.matplotlib_available",
            lambda: False,
        )
