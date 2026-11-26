"""Strict methodological and contract tests for CARA (2024) and UPF (2025) protocols."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import continual_bench.protocols.cara as cara_mod
import continual_bench.protocols.upf as upf_mod
from continual_bench.backends.base import CHECKPOINT_EXTENSIONS
from continual_bench.datasets.streams_unified import (
    _STREAM_SOURCES,
    BatchSplit,
    StreamDatasetSpec,
    load_stream_batches,
)
from continual_bench.detectors import make_detector
from continual_bench.detectors.fhddm import FHDDM
from continual_bench.detectors.skmultiflow import DDM, KSWIN, SkmultiflowDetector
from continual_bench.models.adapters import ModelConfig
from continual_bench.protocols import CampaignConfig, run_campaign
from continual_bench.protocols.common import run_error_detector_schedule
from continual_bench.retraining.rules import CaraP


def _synthetic_stream(n_rows: int = 1500, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    shift = np.linspace(-1.0, 1.0, n_rows)
    y = ((X[:, 0] + 0.8 * shift * X[:, 1]) > 0.0).astype(np.int64)
    return X, y


def test_skmultiflow_adwin_ddm_kswin_and_fhddm_contracts() -> None:
    # ADWIN via SkmultiflowDetector
    adwin = make_detector("adwin", backend="skmultiflow", delta=0.05)
    assert isinstance(adwin, SkmultiflowDetector)
    assert adwin.drift_detected is False
    with pytest.raises(ValueError, match="binary prediction error"):
        adwin.update(2)

    for _ in range(80):
        adwin.update(0)
    triggered = any(adwin.update(1) for _ in range(120))
    assert triggered
    adwin.reset()
    assert adwin.drift_detected is False
    adwin.warm_start([0] * 40)
    assert adwin.drift_detected is False

    # DDM internal reset behavior (sample_count=1, min_num_instances=30)
    ddm = DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)
    assert ddm.sample_count == 1
    assert ddm.min_instances == 30
    for _ in range(28):
        ddm.add_element(0)
        assert ddm.detected_change() is False

    for _ in range(60):
        ddm.add_element(0)
    ddm_triggered = False
    for _ in range(80):
        ddm.add_element(1)
        if ddm.detected_change():
            ddm_triggered = True
            break
    assert ddm_triggered
    # Next element after in_concept_change=True triggers internal DDM.reset() (sample_count=1 -> 2)
    ddm.add_element(0)
    assert ddm.sample_count == 2
    assert ddm.detected_change() is False

    ddm_adapter = make_detector("ddm", backend="skmultiflow")
    assert isinstance(ddm_adapter, SkmultiflowDetector)
    ddm_adapter.warm_start([0] * 35)
    assert ddm_adapter.drift_detected is False

    # KSWIN exact mode, deterministic seed, and warm_start
    src = inspect.getsource(KSWIN.add_element)
    assert 'mode="exact"' in src

    kswin_a = make_detector("kswin", backend="skmultiflow", alpha=0.05, window_size=60, stat_size=30, seed=7)
    kswin_b = make_detector("kswin", backend="skmultiflow", alpha=0.05, window_size=60, stat_size=30, seed=7)
    assert isinstance(kswin_a, SkmultiflowDetector)
    kswin_a.warm_start([0] * 80)
    kswin_b.warm_start([0] * 80)
    assert kswin_a.drift_detected is False
    assert len(kswin_a.inner.window) == 60

    seq = [1] * 45
    out_a = [kswin_a.update(v) for v in seq]
    out_b = [kswin_b.update(v) for v in seq]
    assert out_a == out_b
    assert any(out_a)

    # FHDDM warm_start and reset
    fhddm = make_detector("fhddm", backend="skmultiflow", window_size=25, delta=0.05)
    assert isinstance(fhddm, FHDDM)
    fhddm.warm_start([0] * 25)
    assert len(fhddm._window) == 25
    assert fhddm.p_current == pytest.approx(1.0)
    assert fhddm.p_max == pytest.approx(1.0)
    assert fhddm.drift_detected is False
    assert any(fhddm.update(1) for _ in range(25))
    fhddm.reset()
    assert len(fhddm._window) == 0
    assert fhddm.p_max == 0.0
    assert fhddm.p_current == 0.0
    assert fhddm.drift_detected is False


def test_cara_stream_loading_disjoint_train_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    def _row_id_loader(spec: StreamDatasetSpec):
        n_rows = 600
        row_ids = np.arange(n_rows, dtype=np.float64)[:, None]
        X = np.hstack([row_ids, np.ones((n_rows, 3), dtype=np.float64)])
        y = (np.arange(n_rows) % 2).astype(np.int64)
        return X, y

    monkeypatch.setitem(_STREAM_SOURCES, "cara_disjoint_source", _row_id_loader)
    default_spec = StreamDatasetSpec(
        name="cara_disjoint_source",
        protocol="cara",
        num_batches=6,
        offline_batches=2,
        online_batches=4,
        split_ratio=0.8,
        seed=123,
    )
    batches = load_stream_batches(default_spec)
    assert len(batches) == 6
    for batch in batches:
        train_ids = set(batch.X_train[:, 0].tolist())
        test_ids = set(batch.X_test[:, 0].tolist())
        query_ids = set(batch.X_query[:, 0].tolist())
        assert train_ids.isdisjoint(query_ids)
        assert train_ids.isdisjoint(test_ids)
        np.testing.assert_array_equal(batch.X_query, batch.X_test)
        np.testing.assert_array_equal(batch.y_query, batch.y_test)

    # Explicit smaller query_ratio still samples queries from held-out test split (never X_train)
    sub_query_spec = default_spec.model_copy(update={"query_ratio": 0.1})
    sub_batches = load_stream_batches(sub_query_spec)
    for batch in sub_batches:
        train_ids = set(batch.X_train[:, 0].tolist())
        test_ids = set(batch.X_test[:, 0].tolist())
        query_ids = set(batch.X_query[:, 0].tolist())
        assert train_ids.isdisjoint(query_ids)
        assert query_ids <= test_ids


def test_cara_protocol_online_slice_frozen_model_oracle_reverse_and_electricity_gamma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Verify online = batches[offline:], frozen model = batches[offline], and default gamma=20.0
    def _phase_marker_stream(spec: StreamDatasetSpec):
        rng = np.random.default_rng(spec.seed)
        n_batches = 8
        rows_per_batch = 60
        X = rng.normal(size=(n_batches * rows_per_batch, 4))
        y = np.zeros(n_batches * rows_per_batch, dtype=np.int64)
        # Offline batches [0, 1, 2] have label 0; online batches [3..7] have label 1.
        y[3 * rows_per_batch :] = 1
        return X, y

    monkeypatch.setitem(_STREAM_SOURCES, "electricity", _phase_marker_stream)

    seen_gammas: list[float] = []
    orig_rbf = cara_mod._rbf_kernel

    def _spy_rbf(Q: Any, X: Any, gamma: float) -> np.ndarray:
        seen_gammas.append(gamma)
        return orig_rbf(Q, X, gamma)

    monkeypatch.setattr(cara_mod, "_rbf_kernel", _spy_rbf)

    oracle_shapes: list[tuple[int, int]] = []
    orig_oracle = cara_mod.oracle_schedule

    def _spy_oracle(cost: np.ndarray, kappa: float) -> list[int]:
        oracle_shapes.append(cost.shape)
        return orig_oracle(cost, kappa)

    monkeypatch.setattr(cara_mod, "oracle_schedule", _spy_oracle)

    cfg = CampaignConfig(
        name="cara_contract_check",
        protocol="cara",
        dataset=StreamDatasetSpec(
            name="electricity",
            protocol="cara",
            num_batches=8,
            offline_batches=3,
            online_batches=5,
            split_ratio=0.8,
            seed=9,
        ),
        model=ModelConfig(name="random_forest", params={"n_estimators": 8, "n_jobs": 1}, seed=0),
        seeds=[0],
        policies=["cara", "oracle_staleness", "frozen"],
        kappa_grid=[0.5, 2.0],
        output_dir=str(tmp_path / "cara_out"),
    )
    result = run_campaign(cfg, platform="local", save_model_checkpoints=False)

    assert seen_gammas and all(g == pytest.approx(20.0) for g in seen_gammas)
    assert oracle_shapes == [(5, 5), (5, 5)]  # T - offline = 8 - 3 = 5

    # Because batches[0..2] are all 0 and batches[3..7] are all 1,
    # frozen accuracy = 1.0 iff frozen uses models[offline] (batches[3]), not batches[2].
    frozen_acc = [
        r["mean"]
        for r in result.aggregate_summary
        if r["policy"] == "frozen" and r["metric"] == "average_accuracy"
    ]
    assert frozen_acc == [pytest.approx(1.0)]

    # 2. Verify oracle_staleness reverse-order accuracy clipping semantics
    acc_online = np.array([[0.9, 0.5, 0.0], [0.0, 0.8, 0.6], [0.0, 0.0, 0.3]])
    assert cara_mod._average_accuracy_for_schedule(
        acc_online, [0, 1, 2], reverse_order=False
    ) == pytest.approx((0.9 + 0.5 + 0.3) / 3.0)
    assert cara_mod._average_accuracy_for_schedule(
        acc_online, [0, 1, 2], reverse_order=True
    ) == pytest.approx((0.9 + 0.5 + 0.6) / 3.0)

    acc_calls: list[bool] = []
    orig_avg_acc = cara_mod._average_accuracy_for_schedule

    def _spy_avg_acc(
        accuracy_online: np.ndarray, retrains: list[int], *, reverse_order: bool = False
    ) -> float:
        acc_calls.append(reverse_order)
        return orig_avg_acc(accuracy_online, retrains, reverse_order=reverse_order)

    monkeypatch.setattr(cara_mod, "_average_accuracy_for_schedule", _spy_avg_acc)
    run_campaign(cfg, platform="local", save_model_checkpoints=False)
    # 2 kappas * 3 policies ("cara", "oracle_staleness", "frozen"):
    # only oracle_staleness uses reverse_order=True
    assert acc_calls == [False, True, False, False, True, False]


def test_upf_protocol_carap_gamma_skmultiflow_reset_warm_start_and_zero_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_SAVE_MODEL_CHECKPOINTS", "false")
    monkeypatch.setitem(
        _STREAM_SOURCES,
        "upf_compliance_synth",
        lambda spec: _synthetic_stream(n_rows=1500, n_features=4, seed=spec.seed),
    )

    carap_inits: list[dict[str, Any]] = []
    orig_carap = upf_mod.CaraP

    def _spy_carap(*args: Any, **kwargs: Any) -> CaraP:
        carap_inits.append({"args": args, "kwargs": kwargs})
        return orig_carap(*args, **kwargs)

    monkeypatch.setattr(upf_mod, "CaraP", _spy_carap)

    detector_calls: list[dict[str, Any]] = []
    orig_det_sched = upf_mod.run_error_detector_schedule

    def _spy_det_sched(
        detector_name: str, batches: Any, states: Any, offline: int, **kwargs: Any
    ) -> list[int]:
        detector_calls.append({"detector_name": detector_name, "offline": offline, **kwargs})
        return orig_det_sched(detector_name, batches, states, offline, **kwargs)

    monkeypatch.setattr(upf_mod, "run_error_detector_schedule", _spy_det_sched)

    gammas_passed: list[float | None] = []
    mat_gammas: list[float] = []
    orig_to_abs = upf_mod._to_absolute_staleness_matrix

    def _spy_to_abs(matrix: Any, gamma: float | None = None) -> Any:
        gammas_passed.append(gamma)
        mat = orig_to_abs(matrix, gamma=gamma)
        mat_gammas.append(mat.kernel_gamma())
        return mat

    monkeypatch.setattr(upf_mod, "_to_absolute_staleness_matrix", _spy_to_abs)

    out_dir = tmp_path / "upf_out"
    cfg = CampaignConfig(
        name="upf_compliance_test",
        protocol="upf",
        dataset=StreamDatasetSpec(
            name="upf_compliance_synth",
            protocol="upf",
            batch_size=80,
            seed=17,
        ),
        model=ModelConfig(name="xgboost", params={"eta": 0.3, "num_boost_round": 8, "nthread": 1}, seed=0),
        seeds=[3],
        policies=["upf", "cara", "cara_periodic", "oracle", "adwin_5", "fhddm_5", "kswin_5"],
        alpha_grid=[0.1, 0.9],
        regimes={"low": 0.1, "high": 0.9},
        kernel_gamma=20.0,
        upf_mc_samples=10,
        output_dir=str(out_dir),
    )
    result = run_campaign(cfg, platform="local")

    # 1. CaraP called with allow_never=False, max_period=offline (7)
    assert len(carap_inits) > 0
    for call in carap_inits:
        assert call["kwargs"]["same_batch"] is True
        assert call["kwargs"]["allow_never"] is False
        assert call["kwargs"]["max_period"] == 7
        assert call["kwargs"]["phase_offset"] == 7

    # Even at high cost (alpha=0.9), CaraP with allow_never=False searches finite periods [1, 7]
    high_carap_retrains = [
        r["retrains"]
        for r in result.regime_records
        if r["policy"] == "cara_periodic" and r["regime"] == "high"
    ]
    assert len(high_carap_retrains) == 1
    assert high_carap_retrains[0] >= 1.0

    # 2. kernel_gamma=20.0 honored by _to_absolute_staleness_matrix
    assert gammas_passed == [20.0]
    assert mat_gammas == [20.0]

    # 3. skmultiflow detectors invoked with reset_on_drift=True, initial_model=6
    assert len(detector_calls) == 3
    by_det = {c["detector_name"]: c for c in detector_calls}
    for name in ("adwin", "fhddm", "kswin"):
        c = by_det[name]
        assert c["backend"] == "skmultiflow"
        assert c["reset_on_drift"] is True
        assert c["warm_start_on_reset"] is False
        assert c["initial_model"] == 6
        assert c["eval_split"] == "test"
        assert c["include_offline_in_schedule"] is True
    assert by_det["fhddm"]["detector_params"]["window_size"] == 25
    assert by_det["kswin"]["detector_params"]["window_size"] == 25
    assert by_det["kswin"]["detector_params"]["stat_size"] == 12
    assert by_det["kswin"]["detector_params"]["seed"] == 3

    # 4. Zero model checkpoints saved
    ckpt_files = [p for p in out_dir.rglob("*") if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS]
    assert ckpt_files == []


def test_run_error_detector_schedule_reset_and_warm_start_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, Any]] = []

    class _FakeDetector:
        def update(self, err: int) -> bool:
            events.append(("update", err))
            return err == 1

        def reset(self) -> None:
            events.append(("reset", None))

        def warm_start(self, values: np.ndarray) -> None:
            events.append(("warm_start", values.tolist()))

    monkeypatch.setattr("continual_bench.protocols.common._build_detector", lambda *a, **kw: _FakeDetector())

    class _DummyState:
        def __init__(self, pred: int) -> None:
            self.pred = pred

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), self.pred, dtype=np.int64)

    # 4 batches (offline=2, online=[2, 3]); batch 2 triggers on test label=1 for model 1 (pred=0)
    fake_batches = [
        BatchSplit(
            0,
            np.zeros((2, 2)),
            np.array([0, 0]),
            np.zeros((2, 2)),
            np.array([0, 0]),
            np.zeros((0, 2)),
            np.array([]),
            True,
        ),
        BatchSplit(
            1,
            np.zeros((2, 2)),
            np.array([0, 0]),
            np.zeros((2, 2)),
            np.array([0, 0]),
            np.zeros((0, 2)),
            np.array([]),
            True,
        ),
        BatchSplit(
            2,
            np.zeros((2, 2)),
            np.array([1, 0]),
            np.zeros((2, 2)),
            np.array([1, 0]),
            np.zeros((0, 2)),
            np.array([]),
            False,
        ),
        BatchSplit(
            3,
            np.zeros((2, 2)),
            np.array([1, 1]),
            np.zeros((2, 2)),
            np.array([1, 1]),
            np.zeros((0, 2)),
            np.array([]),
            False,
        ),
    ]
    states = [_DummyState(0), _DummyState(0), _DummyState(1), _DummyState(1)]
    retrains = run_error_detector_schedule(
        "adwin",
        fake_batches,
        states,
        offline=2,
        backend="skmultiflow",
        reset_on_drift=True,
        warm_start_on_reset=True,
        initial_model=1,
        eval_split="test",
        warm_start_split="both",
        include_offline_in_schedule=True,
    )
    assert retrains == [2]
    assert events == [
        ("update", 1),
        ("reset", None),
        ("warm_start", [0, 1, 0, 1]),  # states[2] (pred=1) errors on batch[2] (train=[1,0], test=[1,0])
        ("update", 0),
        ("update", 0),
    ]


def test_local_vs_homelab_backend_consistency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        _STREAM_SOURCES,
        "backend_parity_synth",
        lambda spec: _synthetic_stream(n_rows=1500, n_features=4, seed=spec.seed),
    )

    for protocol, ds_kwargs, model_cfg, policies, extra in (
        (
            "cara",
            {"num_batches": 8, "offline_batches": 3, "online_batches": 5, "split_ratio": 0.8, "seed": 5},
            ModelConfig(name="random_forest", params={"n_estimators": 8, "n_jobs": 1}, seed=0),
            ["cara", "cara_periodic", "oracle_staleness", "frozen", "adwin", "ddm"],
            {"kappa_grid": [0.2, 1.5], "staleness_mode": "relative"},
        ),
        (
            "upf",
            {"batch_size": 80, "seed": 5},
            ModelConfig(name="xgboost", params={"eta": 0.3, "num_boost_round": 8, "nthread": 1}, seed=0),
            ["upf", "cara", "cara_periodic", "oracle", "adwin_5", "fhddm_5", "kswin_5"],
            {
                "alpha_grid": [0.1, 0.9],
                "staleness_mode": "absolute",
                "kernel_gamma": 20.0,
                "upf_mc_samples": 10,
            },
        ),
    ):
        cfg = CampaignConfig(
            name=f"parity_{protocol}",
            protocol=protocol,
            dataset=StreamDatasetSpec(name="backend_parity_synth", protocol=protocol, **ds_kwargs),
            model=model_cfg,
            seeds=[0, 1],
            max_workers=2,
            policies=policies,
            **extra,
        )
        res_local = run_campaign(
            cfg,
            platform="local",
            output_dir=str(tmp_path / f"{protocol}_local"),
            save_model_checkpoints=False,
        )
        res_homelab = run_campaign(
            cfg,
            platform="homelab",
            output_dir=str(tmp_path / f"{protocol}_homelab"),
            save_model_checkpoints=False,
        )

        assert len(res_local.aggregate_summary) == len(res_homelab.aggregate_summary) > 0
        for row_l, row_h in zip(res_local.aggregate_summary, res_homelab.aggregate_summary, strict=True):
            assert row_l.keys() == row_h.keys()
            for key in row_l:
                if isinstance(row_l[key], float):
                    assert row_l[key] == pytest.approx(row_h[key])
                else:
                    assert row_l[key] == row_h[key]
        assert res_local.seed_summaries == res_homelab.seed_summaries
        assert res_local.regime_records == res_homelab.regime_records
