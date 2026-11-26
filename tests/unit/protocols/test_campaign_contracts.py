"""Contract tests for paper campaign configs, backend parity, and benchmark CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from continual_bench import cli
from continual_bench.backends.aws import AwsBackend
from continual_bench.backends.base import CHECKPOINT_EXTENSIONS
from continual_bench.datasets.streams_unified import StreamDatasetSpec, register_stream_source
from continual_bench.models.adapters import ModelConfig
from continual_bench.protocols import CampaignConfig, load_campaign_config, run_campaign


def _synthetic_parity_stream(spec: StreamDatasetSpec):
    rng = np.random.default_rng(spec.seed)
    n_rows = 1200
    X = rng.normal(size=(n_rows, 4))
    shift = np.linspace(-1.0, 1.0, n_rows)
    y = ((X[:, 0] + 0.5 * shift * X[:, 1]) > 0.0).astype(np.int64)
    return X, y


class _FakeS3Client:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, Any]:
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body})
        return {"ETag": "fake-etag"}


@pytest.mark.parametrize(
    ("config_path", "dataset_name"),
    [
        ("configs/campaigns/cara-electricity.yaml", "electricity"),
        ("configs/campaigns/cara-airlines.yaml", "airlines"),
    ],
)
def test_cara_paper_campaign_configs(config_path: str, dataset_name: str) -> None:
    cfg = load_campaign_config(config_path)
    assert isinstance(cfg, CampaignConfig)
    assert cfg.protocol == "cara"
    assert cfg.dataset.name == dataset_name
    assert cfg.dataset.protocol == "cara"
    assert cfg.dataset.num_batches == 100
    assert cfg.dataset.offline_batches == 25
    assert cfg.dataset.online_batches == 75
    assert cfg.dataset.split_ratio == 0.9
    assert cfg.dataset.query_ratio == 0.1
    assert cfg.model.name == "random_forest"
    assert cfg.model.params["n_estimators"] == 100
    assert cfg.seeds == [0, 1, 2, 3, 4]
    assert cfg.policies == [
        "cara",
        "cara_cumulative",
        "cara_periodic",
        "oracle_staleness",
        "frozen",
        "markov",
        "adwin",
        "ddm",
    ]
    assert cfg.staleness_mode == "relative"


@pytest.mark.parametrize(
    ("config_path", "dataset_name", "expected_batch_size"),
    [
        ("configs/campaigns/upf-electricity.yaml", "electricity", 2000),
        ("configs/campaigns/upf-airlines.yaml", "airlines", 3000),
    ],
)
def test_upf_paper_campaign_configs(
    config_path: str, dataset_name: str, expected_batch_size: int
) -> None:
    cfg = load_campaign_config(config_path)
    assert isinstance(cfg, CampaignConfig)
    assert cfg.protocol == "upf"
    assert cfg.dataset.name == dataset_name
    assert cfg.dataset.protocol == "upf"
    assert cfg.dataset.batch_size == expected_batch_size
    assert cfg.dataset.num_batches == 15
    assert cfg.dataset.offline_batches == 7
    assert cfg.dataset.online_batches == 8
    assert cfg.dataset.split_ratio == 0.8
    assert cfg.model.name == "xgboost"
    assert cfg.model.params == {
        "eta": 0.3,
        "objective": "binary:logistic",
        "num_boost_round": 100,
        "nthread": 2,
    }
    assert cfg.seeds == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert cfg.policies == [
        "upf",
        "cara",
        "cara_cumulative",
        "cara_periodic",
        "oracle",
        "adwin_5",
        "adwin_50",
        "fhddm_5",
        "fhddm_50",
        "kswin_5",
        "kswin_50",
    ]
    assert cfg.staleness_mode == "absolute"
    assert cfg.regimes == {"low": 0.1, "high": 0.9}


def test_homelab_and_aws_backend_parity_and_zero_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_stream_source("contract_parity_synth", _synthetic_parity_stream)
    monkeypatch.setenv("BENCH_SAVE_MODEL_CHECKPOINTS", "false")
    monkeypatch.chdir(tmp_path)

    fake_s3 = _FakeS3Client()
    monkeypatch.setattr(AwsBackend, "_get_s3_client", lambda self: fake_s3)

    base_cfg = CampaignConfig(
        name="parity_check",
        protocol="cara",
        dataset=StreamDatasetSpec(
            name="contract_parity_synth",
            protocol="cara",
            num_batches=8,
            offline_batches=3,
            online_batches=5,
            split_ratio=0.8,
            query_ratio=0.1,
            seed=13,
        ),
        model=ModelConfig(name="random_forest", params={"n_estimators": 10, "n_jobs": 1}, seed=0),
        seeds=[0],
        policies=["cara", "cara_periodic", "frozen"],
        kappa_grid=[0.1, 1.0],
        staleness_mode="relative",
    )

    homelab_out = tmp_path / "homelab_out"
    homelab_result = run_campaign(
        base_cfg,
        platform="homelab",
        output_dir=str(homelab_out),
    )

    aws_result = run_campaign(
        base_cfg,
        platform="aws",
        output_dir="s3://mock-bench-bucket/parity",
    )

    assert len(homelab_result.aggregate_summary) == len(aws_result.aggregate_summary) > 0
    for row_h, row_a in zip(homelab_result.aggregate_summary, aws_result.aggregate_summary):
        assert row_h.keys() == row_a.keys()
        for key in row_h:
            val_h, val_a = row_h[key], row_a[key]
            if isinstance(val_h, (int, float)) and isinstance(val_a, (int, float)):
                assert val_h == pytest.approx(val_a)
            else:
                assert val_h == val_a

    # Verify zero model checkpoint files on both backends
    homelab_ckpts = [
        p for p in homelab_out.rglob("*") if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS
    ]
    assert homelab_ckpts == []

    aws_staging = tmp_path / "state" / "benchmarks" / "aws_staging"
    aws_ckpts = [
        p for p in aws_staging.rglob("*") if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS
    ]
    assert aws_ckpts == []
    assert len(fake_s3.puts) > 0
    assert all(
        Path(put["Key"]).suffix.lower() not in CHECKPOINT_EXTENSIONS for put in fake_s3.puts
    )


def test_cli_benchmark_subcommand_executes_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_stream_source("contract_cli_synth", _synthetic_parity_stream)
    cfg_path = tmp_path / "cli_campaign.yaml"
    payload = {
        "name": "cli_benchmark_test",
        "protocol": "cara",
        "dataset": {
            "name": "contract_cli_synth",
            "protocol": "cara",
            "num_batches": 8,
            "offline_batches": 3,
            "online_batches": 5,
            "split_ratio": 0.8,
            "query_ratio": 0.1,
            "seed": 5,
        },
        "model": {"name": "random_forest", "params": {"n_estimators": 10, "n_jobs": 1}, "seed": 0},
        "seeds": [0],
        "policies": ["cara", "frozen"],
        "kappa_grid": [0.1, 1.0],
        "staleness_mode": "relative",
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert "benchmark" in cli.COMMANDS
    out_dir = tmp_path / "cli_out"
    cli.main(
        [
            "benchmark",
            str(cfg_path),
            "--platform",
            "local",
            "--output-dir",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    emitted = json.loads(captured.out)
    assert emitted["protocol"] == "cara"
    assert emitted["dataset"] == "contract_cli_synth"
    assert isinstance(emitted["aggregate_summary"], list)
    assert len(emitted["aggregate_summary"]) > 0
    assert "metadata.json" in emitted["artifacts"]
    assert "aggregate_summary.json" in emitted["artifacts"]
