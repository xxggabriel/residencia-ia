import pickle

import pytest

from continual_bench.config import Config
from continual_bench.core.observability import OnlineObservability


def current_memory():
    return 100


def test_ram_time_excludes_pause_between_checkpoint_attempts(monkeypatch):
    times = iter([100, 101, 500, 502])
    monkeypatch.setattr("continual_bench.core.observability.time.perf_counter", lambda: next(times))
    observer = OnlineObservability({}, memory_sampler=current_memory)
    observer.sample_memory()
    observer.sample_memory()
    assert observer.report()["sampled_rss_byte_seconds"] == 100
    observer = pickle.loads(pickle.dumps(observer))
    observer.sample_memory()
    # 399 seconds paused do not become 39,900 byte-seconds.
    assert observer.report()["sampled_rss_byte_seconds"] == 100
    observer.sample_memory()
    assert observer.report()["sampled_rss_byte_seconds"] == 300


@pytest.mark.parametrize("bad", [float("nan"), -0.1, float("inf")])
def test_metric_settings_reject_nonfinite_or_negative_costs(bad):
    with pytest.raises(ValueError):
        Config.model_validate({"tracking": {"costs": {"train_sample": bad}}})
