"""In-process fakes for external services (S3, MLflow, Kafka) used in E2E tests."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeS3Client:
    """In-memory S3 client backed by a dict of (bucket, key) -> bytes."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes | str) -> dict:
        data = Body.encode() if isinstance(Body, str) else bytes(Body)
        self.objects[(Bucket, Key)] = data
        return {"ETag": hashlib.sha256(data).hexdigest()}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        data = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(data)}


class FakeS3Objects:
    """Drop-in replacement for continual_bench.infrastructure.services.S3Objects."""

    def __init__(self, bucket: str = "fake-bench-bucket") -> None:
        self.bucket = bucket
        self.client = FakeS3Client()

    def seed_from_directory(self, prefix: str, directory: Path, filenames: tuple[str, ...]) -> None:
        for name in filenames:
            path = directory / name
            if path.exists():
                self.client.put_object(Bucket=self.bucket, Key=f"{prefix}/{name}", Body=path.read_bytes())


class FakeMlflowClient:
    """In-memory MLflow client supporting runs, metrics, params, artifacts, and model versions."""

    def __init__(self, tracking_uri: str = "http://fake-mlflow:5000") -> None:
        self.tracking_uri = tracking_uri
        self._experiments: dict[str, SimpleNamespace] = {}
        self._runs: dict[str, SimpleNamespace] = {}
        self._metric_histories: dict[tuple[str, str], list[SimpleNamespace]] = {}
        self._artifacts: dict[str, dict[str, bytes]] = {}
        self._model_versions: dict[tuple[str, str], SimpleNamespace] = {}

    def register_experiment(self, name: str, experiment_id: str = "1") -> SimpleNamespace:
        exp = SimpleNamespace(experiment_id=experiment_id, name=name)
        self._experiments[name] = exp
        self._experiments[experiment_id] = exp
        return exp

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        return self._experiments.get(name)

    def get_experiment(self, experiment_id: str) -> SimpleNamespace:
        return self._experiments[experiment_id]

    def register_run(
        self,
        *,
        run_id: str,
        experiment_name: str,
        params: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        dataset_name: str = "",
        dataset_digest: str = "",
        registered_model: str = "",
        registered_model_version: str = "1",
        model_tags: dict[str, str] | None = None,
        artifacts: dict[str, bytes] | None = None,
    ) -> SimpleNamespace:
        exp = self._experiments.get(experiment_name) or self.register_experiment(
            experiment_name, str(len(self._experiments) + 1)
        )
        run = SimpleNamespace(
            info=SimpleNamespace(
                run_id=run_id,
                experiment_id=exp.experiment_id,
                status="FINISHED",
                artifact_uri=f"mlflow-artifacts:/{exp.experiment_id}/{run_id}/artifacts",
            ),
            data=SimpleNamespace(
                params=dict(params or {}),
                tags=dict(tags or {}),
                metrics=dict(metrics or {}),
            ),
            inputs=SimpleNamespace(
                dataset_inputs=[
                    SimpleNamespace(
                        dataset=SimpleNamespace(name=dataset_name, digest=dataset_digest)
                    )
                ]
            ),
        )
        self._runs[run_id] = run
        self._artifacts[run_id] = dict(artifacts or {"report.json": b"{}"})
        if registered_model:
            self._model_versions[(registered_model, registered_model_version)] = SimpleNamespace(
                name=registered_model,
                version=registered_model_version,
                run_id=run_id,
                tags=dict(model_tags or {}),
            )
        return run

    def record_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self._metric_histories.setdefault((run_id, key), []).append(
            SimpleNamespace(key=key, value=float(value), step=step)
        )
        if run_id in self._runs:
            self._runs[run_id].data.metrics[key] = float(value)

    def get_run(self, run_id: str) -> SimpleNamespace:
        return self._runs[run_id]

    def search_runs(self, experiment_ids: list[str], filter_string: str = "") -> list[SimpleNamespace]:
        results = [r for r in self._runs.values() if r.info.experiment_id in experiment_ids]
        if "tags.run_key =" in filter_string:
            key = filter_string.split("tags.run_key =")[1].strip().strip("'\"")
            results = [r for r in results if r.data.tags.get("run_key") == key]
        return results

    def list_artifacts(self, run_id: str) -> list[SimpleNamespace]:
        return [SimpleNamespace(path=path) for path in self._artifacts.get(run_id, {})]

    def get_model_version(self, name: str, version: str) -> SimpleNamespace:
        return self._model_versions[(name, str(version))]

    def get_metric_history(self, run_id: str, key: str) -> list[SimpleNamespace]:
        return list(self._metric_histories.get((run_id, key), []))

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        target = Path(dst_path) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self._artifacts[run_id][path])
        return str(target)


def install_fake_mlflow(monkeypatch, client: FakeMlflowClient) -> None:
    """Inject a fake mlflow.tracking module returning `client`."""
    mlflow_mod = ModuleType("mlflow")
    tracking_mod = ModuleType("mlflow.tracking")
    tracking_mod.MlflowClient = lambda *args, **kwargs: client
    mlflow_mod.tracking = tracking_mod
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mod)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_mod)


def install_fake_kafka(monkeypatch, samples: int) -> None:
    """Inject a fake confluent_kafka module returning (0, samples) watermarks."""
    kafka_mod = ModuleType("confluent_kafka")

    class _FakeConsumer:
        def __init__(self, _conf) -> None:
            pass

        def get_watermark_offsets(self, _tp, timeout: float = 10) -> tuple[int, int]:
            return (0, samples)

        def close(self) -> None:
            pass

    kafka_mod.Consumer = _FakeConsumer
    kafka_mod.TopicPartition = lambda topic, partition: (topic, partition)
    monkeypatch.setitem(sys.modules, "confluent_kafka", kafka_mod)


def write_fake_receipt(directory: Path, receipt: dict) -> None:
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))


def seed_fake_arff_datasets(tmp_path: Path, monkeypatch, n_rows: int = 220) -> Path:
    """Create minimal valid electricity.arff and airlines.arff files and set BENCH_DATASETS."""
    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    elec_lines = [
        "@relation electricity",
        "@attribute date numeric",
        "@attribute day {1,2,3,4,5,6,7}",
        "@attribute period numeric",
        "@attribute nswprice numeric",
        "@attribute nswdemand numeric",
        "@attribute vicprice numeric",
        "@attribute vicdemand numeric",
        "@attribute transfer numeric",
        "@attribute class {UP,DOWN}",
        "@data",
    ]
    for i in range(n_rows):
        day = (i % 7) + 1
        label = "UP" if i % 2 == 0 else "DOWN"
        elec_lines.append(f"{i * 0.01:.4f},{day},0.5,0.1,0.2,0.1,0.2,0.3,{label}")
    (ds_dir / "electricity.arff").write_text("\n".join(elec_lines) + "\n")

    air_lines = [
        "@relation airlines",
        "@attribute Airline {AA,DL,UA}",
        "@attribute Flight numeric",
        "@attribute AirportFrom {JFK,LAX,ORD}",
        "@attribute AirportTo {ATL,DFW,SFO}",
        "@attribute DayOfWeek {1,2,3,4,5,6,7}",
        "@attribute Time numeric",
        "@attribute Length numeric",
        "@attribute Delay {0,1}",
        "@data",
    ]
    airlines = ("AA", "DL", "UA")
    origins = ("JFK", "LAX", "ORD")
    dests = ("ATL", "DFW", "SFO")
    for i in range(n_rows):
        label = "1" if i % 3 == 0 else "0"
        air_lines.append(
            f"{airlines[i % 3]},{100 + i},{origins[i % 3]},{dests[i % 3]},"
            f"{(i % 7) + 1},{500 + i},{60 + (i % 120)},{label}"
        )
    (ds_dir / "airlines.arff").write_text("\n".join(air_lines) + "\n")

    monkeypatch.setenv("BENCH_DATASETS", str(ds_dir))
    from continual_bench.datasets.arff import file_sha256
    from continual_bench.datasets.openml import REAL_SOURCES

    for name in ("electricity", "airlines"):
        monkeypatch.setitem(REAL_SOURCES[name], "sha256", file_sha256(ds_dir / f"{name}.arff"))
    return ds_dir
