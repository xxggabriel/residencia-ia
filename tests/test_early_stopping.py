"""Unit tests for IterativeTrainingConfig, EarlyStoppingStrategy, IterativeTrainingLoop and Zia models."""

import os
import signal
import struct

import numpy as np
import pytest

from continual_bench.config.iterative import (
    IterativeConfig,
    IterativeHyperparameters,
    IterativeTrainingConfig,
)
from continual_bench.models.training import (
    EarlyStoppingStrategy,
    GracefulTerminationHandler,
    IterativeTrainableModel,
    IterativeTrainingLoop,
    chronological_train_val_split,
    train_iterative,
)
from continual_bench.models.zia import FraudOrReturnBooster, MultinomialLogistic


class SyntheticScheduledModel:
    """Deterministic model whose validation log-loss follows a prescribed schedule per epoch."""

    def __init__(self, val_losses: list[float], train_losses: list[float] | None = None):
        self.val_losses = val_losses
        self.train_losses = train_losses or [max(0.01, v - 0.02) for v in val_losses]
        self.epoch_cursor = 0
        self.weight_epoch = 0

    def train_epoch(self, train_data) -> dict[str, float]:
        idx = min(self.epoch_cursor, len(self.train_losses) - 1)
        self.epoch_cursor += 1
        self.weight_epoch = self.epoch_cursor
        loss = float(self.train_losses[idx])
        return {"log_loss": loss, "accuracy": float(max(0.0, 1.0 - loss * 0.5))}

    def evaluate_split(self, val_data) -> dict[str, float]:
        idx = min(max(0, self.epoch_cursor - 1), len(self.val_losses) - 1)
        loss = float(self.val_losses[idx])
        acc = float(max(0.0, 1.0 - loss * 0.5))
        return {"log_loss": loss, "accuracy": acc, "f1_macro": acc * 0.95}

    def get_state(self) -> bytes:
        return struct.pack("<i", self.weight_epoch)

    def restore_state(self, state: bytes) -> None:
        (self.weight_epoch,) = struct.unpack("<i", state)


class SpyTracker:
    def __init__(self):
        self.epochs: list[dict[str, float]] = []
        self.final_summary: dict | None = None

    def log_epoch(self, metrics: dict[str, float]) -> None:
        self.epochs.append(dict(metrics))

    def finalize(self, summary: dict) -> None:
        self.final_summary = dict(summary)


def test_iterative_config_defaults_and_mappings():
    config = IterativeTrainingConfig()
    assert config.max_epochs == 100
    assert config.patience == 10
    assert config.min_delta == pytest.approx(1e-4)
    assert config.eval_metric == "validation:log_loss"
    assert config.learning_rate == pytest.approx(0.01)
    assert config.l2_regularization == pytest.approx(0.001)
    assert config.validation_fraction == pytest.approx(0.1)
    assert config.seed == 42
    assert config.restore_best_weights is True
    assert IterativeConfig is IterativeTrainingConfig
    assert IterativeHyperparameters is IterativeTrainingConfig

    xgb_params = config.to_xgboost_params()
    assert xgb_params["eta"] == 0.01
    assert xgb_params["reg_lambda"] == 0.001
    assert xgb_params["seed"] == 42
    assert xgb_params["nthread"] == 1
    log_params = config.to_logistic_params()
    assert log_params["learning_rate"] == 0.01
    assert log_params["l2"] == 0.001
    assert log_params["seed"] == 42

    parsed = IterativeTrainingConfig.from_params(
        {"l2": "0.005", "max_iterations": "40", "eta": "0.02", "patience": "8"}
    )
    assert parsed.l2_regularization == pytest.approx(0.005)
    assert parsed.max_epochs == 40
    assert parsed.learning_rate == pytest.approx(0.02)
    assert parsed.patience == 8


def test_early_stopping_triggers_after_10_epochs_without_min_delta_and_restores_best_weights():
    # Losses decrease up to epoch 15 (0.50 -> 0.36), then plateau / degrade from epoch 16..100
    val_losses = [0.50 - 0.01 * e for e in range(15)] + [0.36 + 0.002 * e for e in range(85)]
    model = SyntheticScheduledModel(val_losses)
    assert isinstance(model, IterativeTrainableModel)
    tracker = SpyTracker()
    config = IterativeTrainingConfig(max_epochs=100, patience=10, min_delta=1e-4, restore_best_weights=True)

    result = train_iterative(model, train_data="train", val_data="val", config=config, tracker=tracker)

    assert result["early_stopped"] is True
    assert result["best_epoch"] == 15
    assert result["epochs_trained"] == 25  # 15 + 10 patience epochs
    assert result["patience_consumed"] == 10
    assert result["best_validation_loss"] == pytest.approx(0.36)
    assert model.weight_epoch == 15  # restored from epoch 15 checkpoint, not epoch 25!
    assert len(tracker.epochs) == 25
    assert tracker.epochs[-1]["early_stopped"] == 1.0
    assert tracker.epochs[-1]["best_epoch"] == 15.0
    assert tracker.epochs[-1]["patience_consumed"] == 10.0
    assert tracker.final_summary is not None and tracker.final_summary["best_epoch"] == 15


def test_early_stopping_treats_sub_min_delta_improvements_as_no_progress():
    strategy = EarlyStoppingStrategy(patience=10, min_delta=1e-4, restore_best_weights=True)
    assert strategy.step(1, 1.0, b"state-1") is False
    # Improvements of 5e-5 (< 1e-4 min_delta) do not reset patience
    for epoch in range(2, 11):
        stopped = strategy.step(epoch, 1.0 - 5e-5, f"state-{epoch}".encode())
        assert stopped is False
        assert strategy.best_epoch == 1
    assert strategy.step(11, 1.0 - 5e-5, b"state-11") is True
    assert strategy.early_stopped is True
    assert strategy.best_epoch == 1
    assert strategy.patience_counter == 10
    assert strategy.best_state == b"state-1"


def test_training_loop_runs_full_100_epochs_when_loss_continually_improves():
    val_losses = [1.0 - 0.005 * e for e in range(100)]
    model = SyntheticScheduledModel(val_losses)
    config = IterativeTrainingConfig(max_epochs=100, patience=10, min_delta=1e-4)
    loop = IterativeTrainingLoop(config=config)
    result = loop.run(model, "train", "val")

    assert result["early_stopped"] is False
    assert result["epochs_trained"] == 100
    assert result["best_epoch"] == 100
    assert result["patience_consumed"] == 0
    assert model.weight_epoch == 100


def test_chronological_split_uses_zero_copy_views_and_never_touches_holdout():
    rng = np.random.default_rng(42)
    full_x = rng.normal(size=(200, 4)).astype(np.float64)
    full_y = rng.integers(0, 2, size=200).astype(np.int32)
    holdout_start = 150

    # Sentinel values in holdout [150:] that would corrupt training if ever read
    full_x[holdout_start:] = np.nan
    full_y[holdout_start:] = -999

    pre_holdout_x = full_x[:holdout_start]
    pre_holdout_y = full_y[:holdout_start]
    (train_x, train_y), (val_x, val_y) = chronological_train_val_split(
        pre_holdout_x, pre_holdout_y, validation_fraction=0.1
    )

    assert len(train_x) == 135 and len(val_x) == 15
    assert np.shares_memory(train_x, full_x) and np.shares_memory(val_x, full_x)
    assert np.shares_memory(train_y, full_y) and np.shares_memory(val_y, full_y)
    assert not np.isnan(train_x).any() and not np.isnan(val_x).any()
    assert (train_y >= 0).all() and (val_y >= 0).all()


def test_multinomial_logistic_and_fraud_booster_implement_iterative_protocol():
    rng = np.random.default_rng(42)
    x_multi = rng.normal(size=(120, 3)).astype(np.float64)
    y_multi = ((x_multi[:, 0] > 0).astype(int) + (x_multi[:, 1] > 0).astype(int)).astype(np.int64)

    logistic = MultinomialLogistic(classes=3)
    assert isinstance(logistic, IterativeTrainableModel)
    logistic.fit(x_multi, y_multi)
    assert 1 <= logistic.training_summary["epochs_trained"] <= 100
    assert 1 <= logistic.training_summary["best_epoch"] <= logistic.training_summary["epochs_trained"]
    probs = logistic.predict_proba(x_multi[:10])
    assert probs.shape == (10, 3) and np.allclose(probs.sum(axis=1), 1.0)

    saved_state = logistic.get_state()
    mutated = MultinomialLogistic(classes=3)
    mutated.restore_state(saved_state)
    np.testing.assert_allclose(mutated.predict_proba(x_multi[:10]), probs)

    x_bin = rng.normal(size=(160, 4)).astype(np.float32)
    y_bin = (x_bin[:, 0] + 0.5 * x_bin[:, 1] > 0).astype(np.int32)
    booster = FraudOrReturnBooster()
    assert isinstance(booster, IterativeTrainableModel)
    booster.fit(x_bin, y_bin)
    assert 1 <= booster.training_summary["epochs_trained"] <= 100
    assert booster.rounds == booster.training_summary["best_epoch"]
    b_probs = booster.predict_proba(x_bin[:10])
    assert b_probs.shape == (10, 2) and np.allclose(b_probs.sum(axis=1), 1.0)


def test_graceful_sigterm_handler_saves_partial_state_to_s3(tmp_path):
    from continual_bench.application.cloud.worker import flush_partial_state
    from continual_bench.infrastructure.object_store import LocalObjectStore, get_json

    store = LocalObjectStore(tmp_path / "store")
    run_output = tmp_path / "work" / "run"
    run_output.mkdir(parents=True)
    (run_output / "checkpoint.bin").write_bytes(b"partial-weights-v1")

    status_key = "campaigns/c1/runs/r1/attempts/job-1/status.json"
    prefix = "campaigns/c1/runs/r1/attempts/job-1"
    status = {"run_id": "r1", "campaign_id": "c1", "attempt_key": "job-1"}

    with GracefulTerminationHandler(
        on_terminate=lambda _sig: flush_partial_state(
            store, run_output, prefix, status_key, status, started=0.0, clock=lambda: 123.0
        )
    ) as handler:
        os.kill(os.getpid(), signal.SIGTERM)
        assert handler.terminated is True

    persisted_status = get_json(store, status_key)
    assert persisted_status["state"] == "INTERRUPTED"
    assert persisted_status["partial_files"] == 1
    assert store.get_bytes(f"{prefix}/partial/checkpoint.bin") == b"partial-weights-v1"
