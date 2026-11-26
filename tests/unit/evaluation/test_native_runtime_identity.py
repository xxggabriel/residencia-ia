"""Linux CPU wheel names must be present in the recorded software identity."""

from importlib.metadata import PackageNotFoundError

from continual_bench.evaluation import environment as identity


def test_cpu_xgboost_distribution_is_recorded(monkeypatch):
    def version(package):
        if package == "xgboost-cpu":
            return "3.2.0"
        raise PackageNotFoundError(package)

    monkeypatch.setattr(identity.importlib.metadata, "version", version)
    assert identity.environment()["packages"] == {"xgboost-cpu": "3.2.0"}
