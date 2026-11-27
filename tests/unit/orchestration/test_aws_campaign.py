"""Unit tests for generic parallel AWS Batch campaign orchestration and Spot checkpointing."""

from __future__ import annotations

import signal
from typing import Any

import pytest

from continual_bench.models.training.callbacks import GracefulTerminationHandler
from continual_bench.orchestration.aws_campaign import (
    EXIT_INTERRUPTED,
    config_fingerprint,
    consolidate_aws,
    resolve_campaign_config,
    resolve_campaign_resources,
    run_seed_aws,
    submit_aws_campaign,
    validate_fargate_resources,
)
from continual_bench.protocols.base import CampaignConfig


class _FakeS3Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3Client:
    """In-memory S3 client mock for testing AwsBackend and campaign orchestration."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kwargs: Any) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = bytes(Body)
        return {"ETag": '"fake"'}

    def get_object(self, *, Bucket: str, Key: str, **_kwargs: Any) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise FileNotFoundError(f"NoSuchKey: s3://{Bucket}/{Key}")
        return {"Body": _FakeS3Body(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket: str, Key: str, **_kwargs: Any) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise FileNotFoundError(f"NoSuchKey: s3://{Bucket}/{Key}")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}


class _FakeBatchClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, Any]] = []

    def submit_job(self, **kwargs: Any) -> dict[str, Any]:
        idx = len(self.submissions)
        self.submissions.append(kwargs)
        return {"jobId": f"job-{idx}", "jobName": kwargs["jobName"]}


def test_fargate_resource_validation_and_dynamic_sizing():
    assert validate_fargate_resources("2", 4096) == ("2", 4096)
    assert validate_fargate_resources(4, 8192) == ("4", 8192)
    with pytest.raises(ValueError, match="Invalid Fargate memory"):
        validate_fargate_resources("2", 2048)
    with pytest.raises(ValueError, match="Invalid Fargate vCPU"):
        validate_fargate_resources("3", 4096)

    light = resolve_campaign_config(config_ref="configs/campaigns/cara-electricity.yaml")
    assert resolve_campaign_resources(light) == ("2", 4096)

    heavy = resolve_campaign_config(config_ref="configs/campaigns/cara-airlines.yaml")
    assert resolve_campaign_resources(heavy) == ("4", 8192)
    assert resolve_campaign_resources(heavy, vcpus="8", memory_mib=16384) == ("8", 16384)


def test_resolve_campaign_config_overrides_and_s3_config():
    cfg = resolve_campaign_config(
        config_ref="configs/campaigns/upf-insects-abrupt.yaml",
        seeds="0,1,2",
        policies="upf,cara,oracle",
        model_params='{"nthread": 4}',
        overrides='{"upf_mc_samples": 25}',
    )
    assert cfg.seeds == [0, 1, 2]
    assert cfg.policies == ["upf", "cara", "oracle"]
    assert cfg.model.params["nthread"] == 4
    assert cfg.upf_mc_samples == 25

    s3 = _FakeS3Client()
    s3.put_object(
        Bucket="my-bucket",
        Key="configs/custom.yaml",
        Body=(
            b"name: custom-s3\nprotocol: cara\ndataset:\n  name: synthetic\n  protocol: cara\n"
            b"model:\n  name: logistic\nseeds: [1, 2]\npolicies: [cara, frozen]\n"
        ),
    )
    s3_cfg = resolve_campaign_config(
        config_ref="s3://my-bucket/configs/custom.yaml",
        campaign="custom-override",
        s3_client=s3,
    )
    assert s3_cfg.name == "custom-override"
    assert s3_cfg.seeds == [1, 2]


def test_submit_aws_campaign_dry_run_and_boto3_fanout():
    cfg = resolve_campaign_config(
        config_ref="configs/campaigns/cara-airlines.yaml",
        seeds="0,1",
        output_dir="s3://bench-bucket/runs",
    )
    dry = submit_aws_campaign(
        cfg,
        job_queue="continual-bench-jobs",
        job_definition="continual-bench-benchmark",
        output_dir="s3://bench-bucket/runs",
        vcpus=4,
        memory_mib=8192,
        dry_run=True,
    )
    assert dry["dry_run"] is True
    assert dry["resources"] == {"vcpus": "4", "memory_mib": 8192}
    assert len(dry["seed_jobs"]) == 2
    assert dry["seed_jobs"][0]["containerOverrides"]["resourceRequirements"] == [
        {"type": "VCPU", "value": "4"},
        {"type": "MEMORY", "value": "8192"},
    ]
    assert "consolidate" in dry["consolidate_job"]["containerOverrides"]["command"]

    s3 = _FakeS3Client()
    batch = _FakeBatchClient()
    live = submit_aws_campaign(
        cfg,
        job_queue="continual-bench-jobs",
        job_definition="continual-bench-benchmark",
        output_dir="s3://bench-bucket/runs",
        batch_client=batch,
        s3_client=s3,
    )
    assert live["dry_run"] is False
    assert len(batch.submissions) == 3
    assert batch.submissions[2]["dependsOn"] == [
        {"jobId": "job-0", "type": "SEQUENTIAL"},
        {"jobId": "job-1", "type": "SEQUENTIAL"},
    ]
    assert ("bench-bucket", "runs/campaigns/cara-airlines/campaign_config.json") in s3.objects


@pytest.mark.parametrize("protocol", ["cara", "upf"])
def test_spot_checkpoint_resume_and_s3_consolidation(tmp_path, protocol):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_rows = 240
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n_rows),
            "f2": rng.normal(size=n_rows),
            "label": rng.integers(0, 2, size=n_rows),
        }
    )
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)

    s3 = _FakeS3Client()
    cfg = CampaignConfig(
        name=f"spot-{protocol}",
        protocol=protocol,  # type: ignore[arg-type]
        dataset={
            "name": "synthetic",
            "protocol": protocol,
            "source_path": str(csv_path),
            "num_batches": 6,
            "offline_batches": 3,
            "online_batches": 3,
            "batch_size": 40,
            "split_ratio": 0.8,
            "query_ratio": 0.1,
        },
        model={"name": "random_forest", "params": {"n_estimators": 5}},
        seeds=[0, 1],
        policies=[protocol, "cara_periodic"] if protocol == "upf" else ["cara", "frozen"],
        upf_mc_samples=10,
    )

    # Simulate SIGTERM firing during matrix evaluation (after fitting + 1st batch)
    class _DelayedStopHandler(GracefulTerminationHandler):
        def __init__(self, trigger_after: int) -> None:
            super().__init__()
            self._calls = 0
            self._trigger_after = trigger_after

        @property
        def terminated(self) -> bool:
            self._calls += 1
            if self._calls >= self._trigger_after and not self.received:
                self.trigger(signal.SIGTERM)
            return self.received

    handler = _DelayedStopHandler(trigger_after=8)
    interrupted = run_seed_aws(
        cfg,
        0,
        output_dir="s3://test-bucket/exp",
        s3_client=s3,
        checkpoint_interval_seconds=0.0,
        stop_handler=handler,
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["exit_code"] == EXIT_INTERRUPTED
    fp = config_fingerprint(cfg)
    assert ("test-bucket", f"exp/campaigns/{cfg.name}/{fp}/checkpoints/seed_0_matrix.json.gz") in s3.objects

    # Resume seed 0 and run seed 1
    res0 = run_seed_aws(
        cfg,
        0,
        output_dir="s3://test-bucket/exp",
        s3_client=s3,
        checkpoint_interval_seconds=0.0,
    )
    res1 = run_seed_aws(
        cfg,
        1,
        output_dir="s3://test-bucket/exp",
        s3_client=s3,
        checkpoint_interval_seconds=0.0,
    )
    assert res0["status"] == "completed"
    assert res1["status"] == "completed"

    # Idempotent reuse
    reused = run_seed_aws(cfg, 0, output_dir="s3://test-bucket/exp", s3_client=s3)
    assert reused["status"] == "reused"

    # Changing model params invalidates reuse via config fingerprint
    changed_model = cfg.model.model_copy(update={"params": {"n_estimators": 3}})
    changed_cfg = cfg.model_copy(update={"model": changed_model})
    assert config_fingerprint(changed_cfg) != fp
    res_changed = run_seed_aws(
        changed_cfg,
        0,
        output_dir="s3://test-bucket/exp",
        s3_client=s3,
        workdir=tmp_path / "work",
    )
    assert res_changed["status"] == "completed"

    # Consolidate from S3
    cons = consolidate_aws(
        cfg,
        output_dir="s3://test-bucket/exp",
        s3_client=s3,
    )
    assert cons["campaign"] == cfg.name
    assert cons["seeds"] == [0, 1]
    s3_keys = {k for (_, k) in s3.objects}
    assert any(k.endswith("benchmark_summary.json") for k in s3_keys)
    assert any(k.endswith("notebook_trajectories.json") for k in s3_keys)
    assert any(k.endswith("notebook_report_tables.md") for k in s3_keys)
    # Ensure no ML model checkpoint extensions were written to S3
    assert not any(
        k.endswith((".pkl", ".pickle", ".ubj", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".bin"))
        for k in s3_keys
    )
