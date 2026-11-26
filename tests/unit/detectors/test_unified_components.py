"""Tests for FHDDM detector, unified stream datasets, and model adapters."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from continual_bench.datasets.streams_unified import (
    BatchSplit,
    StreamDatasetSpec,
    load_stream_batches,
    register_stream_source,
    resolve_dataset_path,
)
from continual_bench.detectors import DETECTORS, make_detector
from continual_bench.detectors.fhddm import FHDDM
from continual_bench.models.adapters import (
    MODEL_ADAPTERS,
    ModelAdapter,
    ModelConfig,
    RandomForestAdapter,
    SklearnAdapter,
    XGBoostAdapter,
    build_model_adapter,
    register_model_adapter,
)


def test_fhddm_detector_and_registry():
    assert "fhddm" in DETECTORS
    det = make_detector("fhddm", delta=0.05)
    assert isinstance(det, FHDDM)
    assert det.window_size == 25
    assert det.delta == 0.05

    # 25 correct predictions -> p_max = 1.0, no drift
    for _ in range(25):
        assert det.update(0) is False
    assert det.p_max == pytest.approx(1.0)

    # Feed errors until Hoeffding bound exceeded
    detected = False
    for _ in range(25):
        if det.update(1):
            detected = True
            assert det.drift_detected is True
            break
    assert detected

    # Reset clears state
    det.reset()
    assert det.p_max == 0.0
    assert det.p_current == 0.0
    assert det.drift_detected is False


def test_electricity_cara_and_upf_presets(tmp_path):
    arff_path = tmp_path / "electricity.arff"
    header = (
        "@relation electricity\n"
        "@attribute date numeric\n"
        "@attribute day {1,2,3,4,5,6,7}\n"
        "@attribute period numeric\n"
        "@attribute nswprice numeric\n"
        "@attribute nswdemand numeric\n"
        "@attribute vicprice numeric\n"
        "@attribute vicdemand numeric\n"
        "@attribute transfer numeric\n"
        "@attribute class {UP,DOWN}\n"
        "@data\n"
    )
    rows = []
    for i in range(400):
        lbl = "UP" if i % 2 == 0 else "DOWN"
        rows.append(f"0.0,{1 + (i % 7)},{i * 0.01},0.05,0.4,0.01,0.3,0.2,{lbl}")
    arff_path.write_text(header + "\n".join(rows) + "\n")

    # CARA mode
    cara_spec = StreamDatasetSpec(
        name="electricity",
        protocol="cara",
        source_path=str(arff_path),
        num_batches=10,
        offline_batches=3,
        seed=42,
    )
    cara_batches = load_stream_batches(cara_spec)
    assert len(cara_batches) == 10
    assert all(isinstance(b, BatchSplit) for b in cara_batches)
    assert sum(b.is_offline for b in cara_batches) == 3
    assert cara_batches[0].X_train.shape[1] == 6
    assert cara_batches[0].X_train.shape[0] == 36
    assert cara_batches[0].X_test.shape[0] == 4
    assert cara_batches[0].X_query.shape[0] == 4

    # UPF mode with smaller batch_size to fit 400 rows
    upf_spec = StreamDatasetSpec(
        name="electricity",
        protocol="upf",
        source_path=str(arff_path),
        batch_size=20,
        seed=7,
    )
    upf_batches = load_stream_batches(upf_spec)
    assert len(upf_batches) == 15
    assert sum(b.is_offline for b in upf_batches) == 7
    assert upf_batches[0].X_train.shape == (16, 6)
    assert upf_batches[0].X_test.shape == (4, 6)


def test_airlines_cara_and_upf_presets(tmp_path):
    csv_path = tmp_path / "airlines.csv"
    lines = ["Airline,Flight,AirportFrom,AirportTo,DayOfWeek,Time,Length,Delay"]
    airlines = ["AA", "DL", "UA"]
    airports = ["JFK", "LAX", "ORD", "SFO"]
    for i in range(300):
        lines.append(
            f"{airlines[i % 3]},{100 + (i % 10)},{airports[i % 4]},{airports[(i + 1) % 4]},"
            f"{1 + (i % 7)},{60 + i},{120 + (i % 30)},{i % 2}"
        )
    csv_path.write_text("\n".join(lines) + "\n")

    # Alias "airplanes" resolves airlines.csv in cache_dir
    resolved = resolve_dataset_path("airplanes", cache_dir=tmp_path)
    assert resolved == csv_path

    cara_spec = StreamDatasetSpec(
        name="airplanes",
        protocol="cara",
        num_batches=10,
        offline_batches=2,
        causal_preprocessing=True,
        seed=42,
    )
    cara_batches = load_stream_batches(cara_spec, cache_dir=tmp_path)
    assert len(cara_batches) == 10
    assert issparse(cara_batches[0].X_train)
    # 3 airlines + 10 flights + 4 origins + 4 dests + 7 days (limited by first 60 rows) + 2 numeric
    assert cara_batches[0].X_train.shape[1] >= 10

    upf_spec = StreamDatasetSpec(
        name="airlines",
        protocol="upf",
        batch_size=15,
        seed=11,
    )
    upf_batches = load_stream_batches(upf_spec, cache_dir=tmp_path)
    assert len(upf_batches) == 15
    assert upf_batches[0].X_train.dtype == np.float32
    assert upf_batches[0].X_train.shape == (12, 7)
    assert upf_batches[0].X_test.shape == (3, 7)


def test_custom_loaders_and_registered_source(tmp_path):
    npz_path = tmp_path / "toy.npz"
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    np.savez(npz_path, X=X, y=y)

    spec = StreamDatasetSpec(
        name="toy",
        protocol="custom",
        source_path=str(npz_path),
        num_batches=5,
        offline_batches=2,
        split_ratio=0.8,
        query_ratio=0.2,
    )
    batches = load_stream_batches(spec)
    assert len(batches) == 5
    assert batches[0].X_train.shape == (16, 4)
    assert batches[0].X_test.shape == (4, 4)
    assert batches[0].X_query.shape == (4, 4)

    register_stream_source("synthetic_registered", lambda s: (X, y))
    reg_batches = load_stream_batches(
        StreamDatasetSpec(name="synthetic_registered", protocol="custom", num_batches=5, offline_batches=1)
    )
    assert len(reg_batches) == 5
    assert sum(b.is_offline for b in reg_batches) == 1


def test_model_adapters_dense_and_sparse():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    X_sparse = csr_matrix(X)

    for name, params in (
        ("random_forest", {"n_estimators": 10}),
        ("xgboost", {"num_boost_round": 10}),
        ("sklearn", {}),
    ):
        adapter = build_model_adapter(ModelConfig(name=name, params=params, seed=7))
        assert isinstance(adapter, ModelAdapter)

        adapter.fit(X, y)
        preds = adapter.predict(X)
        proba = adapter.predict_proba(X)
        assert preds.shape == (120,)
        assert proba.shape == (120, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

        # Sparse support
        cloned = adapter.clone(seed=9)
        cloned.fit(X_sparse, y)
        assert cloned.predict_proba(X_sparse).shape == (120, 2)

    assert {"random_forest", "xgboost", "sklearn"} <= set(MODEL_ADAPTERS)
    register_model_adapter("rf_alias", RandomForestAdapter)
    rf_alias = build_model_adapter(ModelConfig(name="rf_alias", params={"n_estimators": 5}))
    assert isinstance(rf_alias, RandomForestAdapter)
    dt_adapter = build_model_adapter(
        ModelConfig(name="sklearn", params={"estimator": "sklearn.tree:DecisionTreeClassifier"})
    )
    assert isinstance(dt_adapter, SklearnAdapter)
    assert isinstance(build_model_adapter(ModelConfig(name="xgboost")), XGBoostAdapter)


def test_resolve_prepared_parquet_aliases_and_metadata_stripping(tmp_path):
    import pandas as pd

    ds_dir = tmp_path / "insects_abrupt_imbalanced"
    ds_dir.mkdir()
    parquet_path = ds_dir / "full.parquet"
    n_rows = 300
    rng = np.random.default_rng(123)
    data = {
        "event_time": np.arange(n_rows, dtype=np.int64),
        "Timestamp": np.arange(n_rows, dtype=np.int64),
        "available_at": np.arange(n_rows, dtype=np.int64),
        "label_available_at": np.arange(n_rows, dtype=np.int64),
        "point_available_at": np.arange(n_rows, dtype=np.int64),
        "seq": np.arange(n_rows, dtype=np.int64),
        "row_id": np.arange(n_rows, dtype=np.int64),
        "sample_id": np.arange(n_rows, dtype=np.int64),
        "concept": ["c1"] * n_rows,
        "regime": np.zeros(n_rows, dtype=np.int64),
        "epoch": np.zeros(n_rows, dtype=np.int64),
        "batch_index": np.zeros(n_rows, dtype=np.int64),
        "label": (np.arange(n_rows) % 2).astype(np.uint8),
    }
    for j in range(1, 34):
        data[f"f{j}"] = rng.normal(size=n_rows).astype(np.float32)
    pd.DataFrame(data).to_parquet(parquet_path, index=False)

    assert resolve_dataset_path("insects_abrupt_imbalanced", cache_dir=tmp_path) == parquet_path
    assert resolve_dataset_path("abrupt_imbalanced", cache_dir=tmp_path) == parquet_path
    assert resolve_dataset_path("insects_abrupt_imbalanced", source_path=ds_dir) == parquet_path
    assert resolve_dataset_path("insects_abrupt_imbalanced", source_path=tmp_path) == parquet_path

    # Alias for incremental_abrupt_imbalanced -> insects_incremental_abrupt_reoccurring_imbalanced
    reocc_dir = tmp_path / "insects_incremental_abrupt_reoccurring_imbalanced"
    reocc_dir.mkdir()
    reocc_csv = reocc_dir / "data.csv"
    pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4] * 25, "label": [0, 1, 0, 1] * 25}).to_csv(
        reocc_csv, index=False
    )
    assert resolve_dataset_path("incremental_abrupt_imbalanced", cache_dir=tmp_path) == reocc_csv

    cara_batches = load_stream_batches(
        StreamDatasetSpec(
            name="insects_abrupt_imbalanced",
            protocol="cara",
            num_batches=10,
            offline_batches=3,
            seed=42,
        ),
        cache_dir=tmp_path,
    )
    assert len(cara_batches) == 10
    assert cara_batches[0].X_train.shape[1] == 33
    assert set(np.unique(cara_batches[0].y_train).tolist()) <= {0, 1}

    upf_batches = load_stream_batches(
        StreamDatasetSpec(
            name="insects_abrupt_imbalanced",
            protocol="upf",
            batch_size=20,
            seed=7,
        ),
        cache_dir=tmp_path,
    )
    assert len(upf_batches) == 15
    assert upf_batches[0].X_train.shape == (16, 33)


def test_generic_multiclass_mixed_categorical_and_dirty_values(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(7)
    n_rows = 300
    cats = np.array(["red", "green", "blue"])
    shapes = np.array(["circle", "square"])
    num1 = rng.normal(size=n_rows)
    num2 = rng.uniform(size=n_rows)
    num1[0] = np.nan
    num2[1] = np.inf
    labels = np.array(["alpha", "beta", "gamma"])[np.arange(n_rows) % 3]

    csv_path = tmp_path / "mixed_multi.csv"
    pd.DataFrame(
        {
            "event_time": np.arange(n_rows),
            "color": cats[np.arange(n_rows) % 3],
            "shape": shapes[np.arange(n_rows) % 2],
            "x1": num1,
            "x2": num2,
            "label": labels,
        }
    ).to_csv(csv_path, index=False)

    cara_batches = load_stream_batches(
        StreamDatasetSpec(
            name="mixed_multi",
            protocol="cara",
            source_path=str(csv_path),
            num_batches=6,
            offline_batches=2,
            causal_preprocessing=True,
            seed=11,
        )
    )
    assert len(cara_batches) == 6
    assert issparse(cara_batches[0].X_train)
    assert np.all(np.isfinite(cara_batches[0].X_train.toarray()))
    assert set(np.unique(cara_batches[0].y_train).tolist()) == {0, 1, 2}

    upf_batches = load_stream_batches(
        StreamDatasetSpec(
            name="mixed_multi",
            protocol="upf",
            source_path=str(csv_path),
            batch_size=20,
            seed=11,
        )
    )
    assert len(upf_batches) == 15
    assert not issparse(upf_batches[0].X_train)
    assert upf_batches[0].X_train.shape == (16, 4)
    assert np.all(np.isfinite(upf_batches[0].X_train))

    # Multiclass ARFF with non-numeric labels
    arff_path = tmp_path / "multi.arff"
    arff_lines = [
        "@relation multi",
        "@attribute f1 numeric",
        "@attribute cat {a,b,c}",
        "@attribute class {low,mid,high}",
        "@data",
    ]
    cls_names = ["low", "mid", "high"]
    cat_names = ["a", "b", "c"]
    for i in range(150):
        arff_lines.append(f"{i * 0.1},{cat_names[i % 3]},{cls_names[i % 3]}")
    arff_path.write_text("\n".join(arff_lines) + "\n", encoding="utf-8")

    arff_batches = load_stream_batches(
        StreamDatasetSpec(
            name="multi",
            protocol="cara",
            source_path=str(arff_path),
            num_batches=5,
            offline_batches=2,
            seed=3,
        )
    )
    assert len(arff_batches) == 5
    assert issparse(arff_batches[0].X_train)
    assert set(np.unique(arff_batches[0].y_train).tolist()) == {0, 1, 2}


def test_multiclass_and_subset_batch_model_adapters(tmp_path):
    from continual_bench.protocols import CampaignConfig, run_campaign

    rng = np.random.default_rng(99)
    X = rng.normal(size=(150, 4))
    y_full = (np.arange(150) % 3).astype(np.int64)
    X_sparse = csr_matrix(X)

    for name, params in (
        ("xgboost", {"num_boost_round": 8}),
        ("random_forest", {"n_estimators": 8}),
        ("sklearn", {}),
    ):
        # Full 3-class fit (dense and sparse)
        adapter = build_model_adapter(ModelConfig(name=name, params=params, seed=5))
        adapter.fit(X, y_full)
        proba = adapter.predict_proba(X)
        preds = adapter.predict(X)
        assert proba.shape == (150, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert preds.shape == (150,)
        assert set(preds.tolist()) <= {0, 1, 2}

        sparse_clone = adapter.clone(seed=6)
        sparse_clone.fit(X_sparse, y_full)
        assert sparse_clone.predict_proba(X_sparse).shape == (150, 3)

        # Subset batch missing class 1 ({0, 2})
        mask_02 = y_full != 1
        sub_adapter = adapter.clone(seed=7)
        sub_adapter.fit(X[mask_02], y_full[mask_02])
        sub_proba = sub_adapter.predict_proba(X[:10])
        sub_preds = sub_adapter.predict(X[:10])
        assert sub_proba.shape == (10, 3)
        np.testing.assert_allclose(sub_proba.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(sub_proba[:, 1], 0.0)
        assert set(sub_preds.tolist()) <= {0, 2}

        # Subset batch not starting at 0 ({1, 2})
        mask_12 = y_full != 0
        sub12 = adapter.clone(seed=8)
        sub12.fit(X[mask_12], y_full[mask_12])
        sub12_proba = sub12.predict_proba(X[:10])
        sub12_preds = sub12.predict(X[:10])
        assert sub12_proba.shape == (10, 3)
        np.testing.assert_allclose(sub12_proba[:, 0], 0.0)
        assert set(sub12_preds.tolist()) <= {1, 2}

        # Single-class batch in multiclass ({2})
        mask_2 = y_full == 2
        single_adapter = adapter.clone(seed=9)
        single_adapter.fit(X[mask_2], y_full[mask_2])
        single_proba = single_adapter.predict_proba(X[:5])
        single_preds = single_adapter.predict(X[:5])
        assert single_proba.shape == (5, 3)
        np.testing.assert_allclose(single_proba[:, 2], 1.0)
        assert np.all(single_preds == 2)

    # End-to-end CARA & UPF on multiclass parquet with XGBoost
    import pandas as pd

    n_rows = 600
    Xm = rng.normal(size=(n_rows, 4))
    ym = np.where(Xm[:, 0] > 0.5, 2, np.where(Xm[:, 0] < -0.5, 0, 1))
    pq_path = tmp_path / "multi_stream.parquet"
    df_m = pd.DataFrame(Xm, columns=["f1", "f2", "f3", "f4"])
    df_m["event_time"] = np.arange(n_rows)
    df_m["label"] = ym
    df_m.to_parquet(pq_path, index=False)

    cara_res = run_campaign(
        CampaignConfig(
            name="multi_cara",
            protocol="cara",
            dataset=StreamDatasetSpec(
                name="multi_stream",
                protocol="cara",
                source_path=str(pq_path),
                num_batches=6,
                offline_batches=2,
                online_batches=4,
                seed=4,
            ),
            model=ModelConfig(name="xgboost", params={"num_boost_round": 6}, seed=0),
            seeds=[0],
            policies=["cara", "frozen"],
            kappa_grid=[0.5],
            output_dir=str(tmp_path / "cara_multi_out"),
        ),
        platform="local",
        save_model_checkpoints=False,
    )
    assert len(cara_res.seed_summaries) == 2

    upf_res = run_campaign(
        CampaignConfig(
            name="multi_upf",
            protocol="upf",
            dataset=StreamDatasetSpec(
                name="multi_stream",
                protocol="upf",
                source_path=str(pq_path),
                batch_size=40,
                seed=4,
            ),
            model=ModelConfig(name="xgboost", params={"num_boost_round": 6}, seed=0),
            seeds=[0],
            policies=["upf", "cara", "oracle"],
            alpha_grid=[0.2],
            regimes={"low": 0.1, "high": 0.9},
            upf_mc_samples=10,
            output_dir=str(tmp_path / "upf_multi_out"),
        ),
        platform="local",
        save_model_checkpoints=False,
    )
    assert len(upf_res.seed_summaries) == 3
