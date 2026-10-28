import importlib.util
import pickle
from pathlib import Path

import pytest

from continual_bench.application.config import Config
from continual_bench.domain.observability import OnlineObservability


def current_memory():
    return 100


def test_ram_time_excludes_pause_between_checkpoint_attempts(monkeypatch):
    times = iter([100, 101, 500, 502])
    monkeypatch.setattr("continual_bench.domain.observability.time.perf_counter", lambda: next(times))
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


def jobs():
    spec = importlib.util.spec_from_file_location(
        "metrics_jobs", Path(__file__).parents[1] / "dags/bench_jobs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_suites_are_new_identities_and_bounded_smoke_covers_stack():
    module = jobs()
    params = {"samples": 300, "seeds": [42], "label_delay": 3}
    original = {Config.model_validate(c).identity() for c in module.configurations(params, "nonlinear")}
    cases = [Config.model_validate(c) for c in module.configurations(params, "metrics_smoke")]
    assert len(cases) == 8
    assert not original.intersection(c.identity() for c in cases)
    assert {c.mode for c in cases} == {"offline", "incremental", "streaming"}
    assert len({c.dataset.name for c in cases}) == 5
    assert len({c.model.name for c in cases}) == 3
    assert all(c.tracking.version == "residency-metrics-v1" for c in cases)
    full_params = {**params, "seeds": list(range(42, 47))}
    assert len(module.configurations(full_params, "metrics")) == 867


@pytest.mark.parametrize("bad", [float("nan"), -0.1, float("inf")])
def test_metric_settings_reject_nonfinite_or_negative_costs(bad):
    with pytest.raises(ValueError):
        Config.model_validate({"tracking": {"costs": {"train_sample": bad}}})
