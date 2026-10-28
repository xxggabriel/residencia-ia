"""Regression checks for scientific run identity; no external services are substituted."""

import pytest

from continual_bench.application.config import Config
from continual_bench.application.runner import evaluate, prepare


class UnitTransport:
    def __init__(self):
        self.calls = 0

    def contract(self):
        return {"name": "unit_fixture", "version": 1}

    def events(self, manifest, source, deadline):
        self.calls += 1
        yield from source


def small_config(**extra):
    return Config.model_validate({"warmup": 20, "dataset": {"samples": 140}, **extra})


@pytest.mark.parametrize("complete", [True, False])
def test_local_streaming_cache_cannot_satisfy_another_transport(complete, tmp_path):
    config = small_config()
    manifest = prepare(config, tmp_path)
    evaluate(config, manifest, tmp_path, stop_after=None if complete else 40)
    transport = UnitTransport()
    with pytest.raises(ValueError, match="transport"):
        evaluate(config, manifest, tmp_path, transport)
    assert transport.calls == 0


def test_transport_result_cannot_satisfy_local_cache(tmp_path):
    config = small_config()
    manifest = prepare(config, tmp_path)
    transport = UnitTransport()
    result = evaluate(config, manifest, tmp_path, transport)
    assert result["transport"] == "unit_fixture"
    assert transport.calls == 1
    with pytest.raises(ValueError, match="transport"):
        evaluate(config, manifest, tmp_path)


@pytest.mark.parametrize("changed", ["package_helper", "external_helper", "resource"])
def test_plugin_auxiliary_changes_invalidate_resume(changed, tmp_path, monkeypatch):
    package_name = "bench_plugin_" + changed
    external_name = "bench_helper_" + changed
    package = tmp_path / package_name
    external = tmp_path / external_name
    package.mkdir()
    external.mkdir()
    (package / "__init__.py").write_text("")
    (external / "__init__.py").write_text("")
    (package / "registration.py").write_text(
        f"from {external_name} import helper\ndef register(registry):\n    return None\n"
    )
    (package / "helper.py").write_text("VALUE = 1\n")
    (package / "settings.json").write_text('{"value": 1}')
    (external / "helper.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    config = small_config(plugins=[package_name + ".registration:register"], plugin_packages=[external_name])
    state = tmp_path / "state"
    manifest = prepare(config, state)
    evaluate(config, manifest, state, stop_after=40)
    target = {
        "package_helper": package / "helper.py",
        "external_helper": external / "helper.py",
        "resource": package / "settings.json",
    }[changed]
    original = target.read_text()
    target.write_text(original.replace("1", "2"))
    with pytest.raises(ValueError, match="plugin"):
        evaluate(config, manifest, state)


def test_zenml_definitions_are_importable_sdk_objects():
    pytest.importorskip("zenml")
    from zenml.pipelines.pipeline_definition import Pipeline
    from zenml.steps.base_step import BaseStep

    from continual_bench.adapters import zenml_steps

    assert isinstance(zenml_steps.finite_benchmark, Pipeline)
    for name in ("materialize", "benchmark", "persist"):
        step = getattr(zenml_steps, name)
        assert isinstance(step, BaseStep)
        assert step.__class__.__module__ == zenml_steps.__name__
