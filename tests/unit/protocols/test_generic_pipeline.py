"""Unit and contract tests for generic multi-dataset & INSECTS benchmark pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from continual_bench.backends.base import CHECKPOINT_EXTENSIONS
from continual_bench.datasets.streams_unified import (
    _STREAM_SOURCES,
    StreamDatasetSpec,
    _dataframe_to_xy,
    load_stream_batches,
    resolve_dataset_path,
)
from continual_bench.models.adapters import (
    ModelConfig,
    RandomForestAdapter,
    XGBoostAdapter,
    build_model_adapter,
)
from continual_bench.protocols import CampaignConfig, load_campaign_config, run_campaign


def _make_insects_dataframe(
    n_rows: int = 360,
    n_features: int = 33,
    *,
    seed: int = 0,
    string_labels: bool = False,
) -> pd.DataFrame:
    """Create a simulated INSECTS DataFrame with metadata/leakage columns and multiclass labels."""
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {
        "event_time": np.arange(n_rows, dtype=np.int64) * 10,
        "seq": np.arange(n_rows, dtype=np.int64),
        "timestamp": np.linspace(1_700_000_000.0, 1_700_001_000.0, n_rows),
        "available_at": np.arange(n_rows, dtype=np.int64) * 10 + 5,
        "concept": ["abrupt_a" if i < n_rows // 2 else "abrupt_b" for i in range(n_rows)],
    }
    for j in range(1, n_features + 1):
        data[f"f{j}"] = rng.normal(loc=0.0, scale=1.0, size=n_rows)

    # 3-class imbalanced labels with concept drift across the stream.
    shift = np.linspace(-1.5, 1.5, n_rows)
    score = data["f1"] + 0.8 * shift * data["f2"]
    y_int = np.where(score > 0.9, 2, np.where(score > -0.3, 1, 0)).astype(np.int64)
    if string_labels:
        label_map = {0: "Aedes_aegypti", 1: "Culex_quinquefasciatus", 2: "Musca_domestica"}
        data["label"] = [label_map[int(v)] for v in y_int]
    else:
        data["label"] = y_int
    return pd.DataFrame(data)


def _drifting_binary_stream(spec: StreamDatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    n_rows = 1500
    X = rng.normal(size=(n_rows, 6))
    y = np.zeros(n_rows, dtype=np.int64)
    half = n_rows // 2
    y[:half] = (X[:half, 0] > 0.0).astype(np.int64)
    y[half:] = (X[half:, 0] < 0.0).astype(np.int64)
    return X, y


def _drifting_multiclass_stream(spec: StreamDatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    n_rows = 1200
    X = rng.normal(size=(n_rows, 5))
    shift = np.linspace(-1.5, 1.5, n_rows)
    score = X[:, 0] + shift * X[:, 1]
    y = np.where(score > 0.8, 2, np.where(score > -0.4, 1, 0)).astype(np.int64)
    return X, y


def test_resolve_dataset_path_finds_prepared_full_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_name = "insects_abrupt_imbalanced"
    ds_dir = tmp_path / "datasets_root"
    parquet_path = ds_dir / ds_name / "full.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    _make_insects_dataframe(n_rows=30).to_parquet(parquet_path, index=False)

    # Located via explicit cache_dir
    resolved_cache = resolve_dataset_path(ds_name, cache_dir=ds_dir)
    assert resolved_cache == parquet_path

    # Located via $BENCH_DATASETS
    monkeypatch.setenv("BENCH_DATASETS", str(ds_dir))
    resolved_env = resolve_dataset_path(ds_name)
    assert resolved_env == parquet_path

    # Located via alias (abrupt_imbalanced -> insects_abrupt_imbalanced)
    resolved_alias = resolve_dataset_path("abrupt_imbalanced")
    assert resolved_alias == parquet_path


def test_dataframe_to_xy_and_stream_loading_strip_leakage_handle_multiclass_and_cara_mixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df_int = _make_insects_dataframe(n_rows=360, n_features=33, seed=4, string_labels=False)
    X_int, y_int = _dataframe_to_xy(df_int)
    assert X_int.shape == (360, 33)
    assert X_int.dtype == np.float64
    assert y_int.dtype == np.int64
    assert set(np.unique(y_int).tolist()) == {0, 1, 2}

    # Multiclass string labels and non-contiguous integer labels map to contiguous 0..K-1
    df_str = _make_insects_dataframe(n_rows=120, n_features=8, seed=5, string_labels=True)
    X_str, y_str = _dataframe_to_xy(df_str)
    assert X_str.shape == (120, 8)
    assert y_str.dtype == np.int64
    assert set(np.unique(y_str).tolist()) == {0, 1, 2}

    df_noncontig = pd.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 2.0, 3.0, 4.0],
            "label": [10, 30, 20, 30],
        }
    )
    X_nc, y_nc = _dataframe_to_xy(df_noncontig)
    assert X_nc.shape == (4, 2)
    assert y_nc.tolist() == [0, 2, 1, 2]

    # Simulated INSECTS full.parquet loaded via $BENCH_DATASETS produces exact 33 features
    ds_root = tmp_path / "bench_data"
    insects_parquet = ds_root / "insects_abrupt_imbalanced" / "full.parquet"
    insects_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_int.to_parquet(insects_parquet, index=False)
    monkeypatch.setenv("BENCH_DATASETS", str(ds_root))

    upf_batches = load_stream_batches(
        StreamDatasetSpec(
            name="insects_abrupt_imbalanced",
            protocol="upf",
            batch_size=24,
            num_batches=15,
            offline_batches=7,
            online_batches=8,
            split_ratio=0.8,
            seed=0,
        )
    )
    assert len(upf_batches) == 15
    assert upf_batches[0].X_train.shape[1] == 33
    assert upf_batches[0].X_test.shape[1] == 33
    assert set(np.unique(np.concatenate([b.y_train for b in upf_batches])).tolist()) == {0, 1, 2}

    # CARA numeric-only generic stream applies MinMaxScaler
    cara_batches = load_stream_batches(
        StreamDatasetSpec(
            name="insects_abrupt_imbalanced",
            protocol="cara",
            num_batches=10,
            offline_batches=3,
            online_batches=7,
            split_ratio=0.9,
            query_ratio=0.1,
            seed=0,
        )
    )
    assert len(cara_batches) == 10
    assert cara_batches[0].X_train.shape[1] == 33
    all_train = np.vstack([b.X_train for b in cara_batches])
    assert float(all_train.min()) >= -1e-9
    assert float(all_train.max()) <= 1.0 + 1e-9

    # Generic mixed-type stream in CARA applies OneHotEncoder + MaxAbsScaler (sparse CSR)
    mixed_csv = tmp_path / "mixed_stream.csv"
    mixed_df = pd.DataFrame(
        {
            "event_time": np.arange(100),
            "concept": ["c0"] * 50 + ["c1"] * 50,
            "region": ["north", "south", "east", "west"] * 25,
            "tier": ["a", "b"] * 50,
            "amount": np.linspace(10.0, 200.0, 100),
            "label": ["low", "mid", "high", "low"] * 25,
        }
    )
    mixed_df.to_csv(mixed_csv, index=False)
    mixed_batches = load_stream_batches(
        StreamDatasetSpec(
            name="mixed_stream",
            source_path=str(mixed_csv),
            protocol="cara",
            num_batches=5,
            offline_batches=2,
            online_batches=3,
            split_ratio=0.8,
            query_ratio=0.2,
            causal_preprocessing=True,
            seed=1,
        )
    )
    assert len(mixed_batches) == 5
    assert issparse(mixed_batches[0].X_train)
    # 4 region OHE + 2 tier OHE + 1 numeric amount = 7 columns (event_time & concept stripped)
    assert mixed_batches[0].X_train.shape[1] == 7


def test_adapters_handle_3class_imbalanced_and_missing_class_batches() -> None:
    rng = np.random.default_rng(42)
    X_full = rng.normal(size=(120, 6))
    # Imbalanced 3-class: 70% class 0, 20% class 1, 10% class 2
    y_full = np.array([0] * 84 + [1] * 24 + [2] * 12, dtype=np.int64)
    perm = rng.permutation(len(y_full))
    X_full, y_full = X_full[perm], y_full[perm]

    # Batch missing middle class (only classes {0, 2}) and single-class batch (only {2})
    X_missing_mid = rng.normal(size=(40, 6))
    y_missing_mid = np.array([0] * 32 + [2] * 8, dtype=np.int64)
    X_single = rng.normal(size=(15, 6))
    y_single = np.full(15, 2, dtype=np.int64)

    X_eval = rng.normal(size=(25, 6))

    for adapter_name, params in (
        ("random_forest", {"n_estimators": 10, "n_jobs": 1}),
        # Even with binary:logistic (INSECTS UPF config), XGBoostAdapter auto-switches to multi:softprob.
        ("xgboost", {"eta": 0.3, "objective": "binary:logistic", "num_boost_round": 8, "nthread": 1}),
    ):
        full_model = build_model_adapter(ModelConfig(name=adapter_name, params=params, seed=0))
        full_model.fit(X_full, y_full)
        proba_full = full_model.predict_proba(X_eval)
        preds_full = full_model.predict(X_eval)
        assert proba_full.shape == (25, 3)
        np.testing.assert_allclose(proba_full.sum(axis=1), 1.0, atol=1e-6)
        assert set(preds_full.tolist()) <= {0, 1, 2}

        missing_model = full_model.clone(seed=1)
        missing_model.fit(X_missing_mid, y_missing_mid)
        proba_missing = missing_model.predict_proba(X_eval)
        preds_missing = missing_model.predict(X_eval)
        assert proba_missing.shape == (25, 3)
        np.testing.assert_allclose(proba_missing[:, 1], 0.0)
        np.testing.assert_allclose(proba_missing.sum(axis=1), 1.0, atol=1e-6)
        assert set(preds_missing.tolist()) <= {0, 2}

    # Explicit num_class=3 handles even a single-class batch cleanly
    xgb_single = XGBoostAdapter(params={"num_boost_round": 5, "num_class": 3}, seed=0)
    xgb_single.fit(X_single, y_single)
    proba_single = xgb_single.predict_proba(X_eval)
    preds_single = xgb_single.predict(X_eval)
    assert proba_single.shape == (25, 3)
    np.testing.assert_allclose(proba_single[:, 2], 1.0)
    assert np.all(preds_single == 2)

    rf_check = RandomForestAdapter(params={"n_estimators": 5}, seed=0).fit(X_missing_mid, y_missing_mid)
    assert rf_check.predict_proba(X_eval).shape == (25, 3)


def test_upf_protocol_dynamic_alpha_max_regimes_and_generic_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(_STREAM_SOURCES, "custom_insects_upf_synth", _drifting_binary_stream)
    out_dir = tmp_path / "upf_generic_out"
    cfg = CampaignConfig(
        name="custom_upf_dynamic",
        protocol="upf",
        dataset=StreamDatasetSpec(
            name="custom_insects_upf_synth",
            protocol="upf",
            batch_size=80,
            num_batches=15,
            offline_batches=7,
            online_batches=8,
            split_ratio=0.8,
            seed=3,
        ),
        model=ModelConfig(name="xgboost", params={"eta": 0.3, "num_boost_round": 8, "nthread": 1}, seed=0),
        seeds=[0],
        policies=["upf", "cara", "cara_periodic", "oracle", "adwin_5"],
        upf_mc_samples=10,
        output_dir=str(out_dir),
    )
    result = run_campaign(cfg, platform="local", save_model_checkpoints=False)

    meta = json.loads(Path(result.artifacts["metadata.json"]).read_text(encoding="utf-8"))
    alpha_max = float(meta["alpha_max"])
    alpha_grid = [float(v) for v in meta["alpha_grid"]]
    regimes = {k: float(v) for k, v in meta["regimes"].items()}

    assert alpha_max > 0.0
    assert len(alpha_grid) == 10
    assert alpha_grid[0] == pytest.approx(0.0)
    assert alpha_grid[-1] == pytest.approx(alpha_max)
    assert regimes["low"] == pytest.approx(round(0.1 * alpha_max, 6))
    assert regimes["high"] == pytest.approx(round(0.9 * alpha_max, 6))

    # Oracle reaches 0 retrains at alpha_max
    oracle_at_max = [
        r["retrains"]
        for r in result.regime_records
        if r["policy"] == "oracle"
        and r["regime"] == "price_grid"
        and np.isclose(r["alpha"], alpha_max)
    ]
    assert oracle_at_max == [0.0]

    # Proportional low/high regimes recorded for every policy
    for pol in cfg.policies:
        pol_regimes = {
            r["regime"]: r["alpha"]
            for r in result.regime_records
            if r["policy"] == pol and r["regime"] in {"low", "high"}
        }
        assert pol_regimes == {"low": pytest.approx(regimes["low"]), "high": pytest.approx(regimes["high"])}

    # Generic benchmark reports generated cleanly; no SVG/PNG plots
    for fname in ("benchmark_summary.csv", "benchmark_summary.json", "benchmark_summary.md"):
        assert fname in result.artifacts
        assert Path(result.artifacts[fname]).exists()

    svg_keys = [k for k in result.artifacts if k.endswith((".svg", ".png"))]
    assert svg_keys == []

    # Zero checkpoints
    ckpt_files = [p for p in out_dir.rglob("*") if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS]
    assert ckpt_files == []


def test_cara_protocol_multiclass_dynamic_kappa_grid_all_policies_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(_STREAM_SOURCES, "custom_insects_cara_synth", _drifting_multiclass_stream)
    out_dir = tmp_path / "cara_multiclass_out"
    policies = [
        "cara",
        "cara_cumulative",
        "cara_periodic",
        "oracle_staleness",
        "frozen",
        "markov",
        "adwin",
        "ddm",
    ]
    cfg = CampaignConfig(
        name="custom_cara_multiclass",
        protocol="cara",
        dataset=StreamDatasetSpec(
            name="custom_insects_cara_synth",
            protocol="cara",
            num_batches=10,
            offline_batches=3,
            online_batches=7,
            split_ratio=0.9,
            query_ratio=0.1,
            seed=9,
        ),
        model=ModelConfig(name="random_forest", params={"n_estimators": 8, "n_jobs": 1}, seed=0),
        seeds=[0],
        policies=policies,
        staleness_mode="relative",
        output_dir=str(out_dir),
    )
    result = run_campaign(cfg, platform="local", save_model_checkpoints=False)

    assert len(result.seed_summaries) == len(policies)
    # 1 seed * 99 dynamic kappa points * 8 policies = 792 records
    assert len(result.regime_records) == 99 * len(policies)
    assert len(result.aggregate_summary) == 4 * len(policies)

    meta = json.loads(Path(result.artifacts["metadata.json"]).read_text(encoding="utf-8"))
    assert float(meta["kappa_min"]) > 0.0
    assert float(meta["kappa_max"]) > float(meta["kappa_min"])

    for fname in ("benchmark_summary.csv", "benchmark_summary.json", "benchmark_summary.md"):
        assert fname in result.artifacts
        assert Path(result.artifacts[fname]).exists()

    svg_keys = [k for k in result.artifacts if k.endswith((".svg", ".png"))]
    assert svg_keys == []

    ckpt_files = [p for p in out_dir.rglob("*") if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS]
    assert ckpt_files == []


@pytest.mark.parametrize(
    ("config_path", "dataset_name"),
    [
        ("configs/campaigns/cara-insects-abrupt.yaml", "insects_abrupt_imbalanced"),
        (
            "configs/campaigns/cara-insects-reoccurring.yaml",
            "insects_incremental_abrupt_reoccurring_imbalanced",
        ),
    ],
)
def test_cara_insects_campaign_configs(config_path: str, dataset_name: str) -> None:
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
    assert cfg.model.params == {"n_estimators": 100, "n_jobs": 2}
    assert cfg.seeds == [0, 1, 2, 3, 4]
    assert cfg.max_workers == 5
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
    assert cfg.kappa_grid is None


@pytest.mark.parametrize(
    ("config_path", "dataset_name"),
    [
        ("configs/campaigns/upf-insects-abrupt.yaml", "insects_abrupt_imbalanced"),
        (
            "configs/campaigns/upf-insects-reoccurring.yaml",
            "insects_incremental_abrupt_reoccurring_imbalanced",
        ),
    ],
)
def test_upf_insects_campaign_configs(config_path: str, dataset_name: str) -> None:
    cfg = load_campaign_config(config_path)
    assert isinstance(cfg, CampaignConfig)
    assert cfg.protocol == "upf"
    assert cfg.dataset.name == dataset_name
    assert cfg.dataset.protocol == "upf"
    assert cfg.dataset.batch_size == 10000
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
    assert cfg.max_workers == 5
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
    assert cfg.alpha_grid is None
    assert cfg.regimes is None
