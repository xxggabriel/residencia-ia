"""Tests for the continual_bench.protocols layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from continual_bench.datasets.streams_unified import (
    _STREAM_SOURCES,
    StreamDatasetSpec,
    register_stream_source,
)
from continual_bench.models.adapters import ModelConfig
from continual_bench.protocols import (
    CampaignConfig,
    ProtocolResult,
    load_campaign_config,
    run_campaign,
)


def _synthetic_cara_stream(spec: StreamDatasetSpec):
    rng = np.random.default_rng(spec.seed)
    n_rows = 2400
    X = rng.normal(size=(n_rows, 4))
    shift = np.linspace(-1.0, 1.0, n_rows)
    y = ((X[:, 0] + 0.5 * shift * X[:, 1]) > 0.0).astype(np.int64)
    return X, y


def _synthetic_upf_stream(spec: StreamDatasetSpec):
    rng = np.random.default_rng(spec.seed)
    n_rows = 1500
    X = rng.normal(size=(n_rows, 4))
    shift = np.linspace(-0.8, 0.8, n_rows)
    y = ((X[:, 0] - 0.6 * shift + 0.3 * X[:, 2]) > 0.0).astype(np.int64)
    return X, y


def test_cara_protocol_end_to_end(tmp_path: Path):
    register_stream_source("synth_cara", _synthetic_cara_stream)
    cfg = CampaignConfig(
        name="test_cara_synth",
        protocol="cara",
        dataset=StreamDatasetSpec(
            name="synth_cara",
            protocol="cara",
            num_batches=12,
            offline_batches=4,
            online_batches=8,
            split_ratio=0.8,
            query_ratio=0.15,
            seed=7,
        ),
        model=ModelConfig(name="random_forest", params={"n_estimators": 10}, seed=0),
        seeds=[0, 1],
        policies=[
            "cara",
            "cara_cumulative",
            "cara_periodic",
            "oracle_staleness",
            "frozen",
            "markov",
            "adwin",
            "ddm",
        ],
        # Default 99-kappa grid when kappa_grid=None
        output_dir=str(tmp_path / "out"),
    )
    result = run_campaign(cfg, platform="local", save_model_checkpoints=False)
    assert isinstance(result, ProtocolResult)
    assert result.protocol == "cara"
    assert len(result.seed_summaries) == 2 * 8
    # 2 seeds * 99 kappas * 8 policies = 1584 regime records
    assert len(result.regime_records) == 2 * 99 * 8
    assert len(result.aggregate_summary) == 8 * 4
    assert "metadata.json" in result.artifacts
    assert "aggregate_summary.csv" in result.artifacts


def test_upf_protocol_end_to_end_with_paper_comparison(tmp_path: Path):
    register_stream_source("electricity_synth_upf", _synthetic_upf_stream)
    cfg = CampaignConfig(
        name="test_upf_synth",
        protocol="upf",
        dataset=StreamDatasetSpec(
            name="electricity_synth_upf",
            protocol="upf",
            batch_size=80,
            seed=11,
        ),
        model=ModelConfig(name="xgboost", params={"eta": 0.3, "num_boost_round": 10}, seed=0),
        seeds=[0, 1],
        policies=[
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
        ],
        upf_mc_samples=20,
        output_dir=str(tmp_path / "out"),
    )
    result = run_campaign(cfg, platform="local", save_model_checkpoints=False)
    assert result.protocol == "upf"
    assert len(result.seed_summaries) == 2 * 11
    # 2 seeds * 11 policies * (10 alpha_grid + 2 point regimes) = 264
    assert len(result.regime_records) == 2 * 11 * 12
    # 11 policies * (1 price_grid AUC + 2 regimes * 3 metrics) = 77
    assert len(result.aggregate_summary) == 11 * 7
    assert "aggregate_summary.json" in result.artifacts


def test_load_campaign_yaml_and_paper_comparison_svg(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(_STREAM_SOURCES, "electricity", _synthetic_upf_stream)
    yaml_path = tmp_path / "campaign.yaml"
    payload = {
        "name": "yaml_upf_test",
        "protocol": "upf",
        "dataset": {
            "name": "electricity",
            "protocol": "upf",
            "batch_size": 80,
            "seed": 3,
        },
        "model": {"name": "xgboost", "params": {"eta": 0.3, "num_boost_round": 8}, "seed": 0},
        "seeds": [0],
        "policies": ["upf", "cara", "oracle"],
        "upf_mc_samples": 10,
        "output_dir": str(tmp_path / "out"),
    }
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_campaign_config(yaml_path)
    assert loaded.name == "yaml_upf_test"

    result = run_campaign(yaml_path, platform="local", save_model_checkpoints=False)
    assert "paper_comparison.csv" in result.artifacts
    assert "paper_comparison.json" in result.artifacts
    assert "paper_comparison.md" in result.artifacts
    svg_keys = [k for k in result.artifacts if k.endswith(".svg")]
    assert len(svg_keys) >= 1
    for svg_key in svg_keys:
        svg_path = Path(result.artifacts[svg_key])
        assert svg_path.exists()
        assert "<svg" in svg_path.read_text(encoding="utf-8")
