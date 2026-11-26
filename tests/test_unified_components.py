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
