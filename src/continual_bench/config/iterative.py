"""Canonical iterative training hyperparameters for Zia batch/iterative learners."""

from typing import Any

from pydantic import Field, model_validator

from .benchmark import Strict


class IterativeTrainingConfig(Strict):
    """Unified hyperparameter contract across Zia iterative learners."""

    max_epochs: int = Field(default=100, ge=1, le=100)
    patience: int = Field(default=10, ge=1, le=100)
    min_delta: float = Field(default=1e-4, ge=0.0, allow_inf_nan=False)
    eval_metric: str = Field(default="validation:log_loss", min_length=1)
    learning_rate: float = Field(default=0.01, gt=0.0, le=10.0, allow_inf_nan=False)
    l2_regularization: float = Field(default=0.001, ge=0.0, allow_inf_nan=False)
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=0.5, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    restore_best_weights: bool = True

    @model_validator(mode="after")
    def patience_within_max_epochs(self):
        if self.patience > self.max_epochs and self.max_epochs > 10:
            raise ValueError("patience cannot exceed max_epochs when max_epochs > 10")
        return self

    def to_xgboost_params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 5,
            "max_bin": 128,
            "eta": self.learning_rate,
            "reg_lambda": self.l2_regularization,
            "nthread": 1,
            "seed": self.seed,
        }
        if extra:
            base.update(extra)
            base["nthread"] = 1
        return base

    def to_logistic_params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "solver": "L-BFGS-B",
            "l2": self.l2_regularization,
            "l2_regularization": self.l2_regularization,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }
        if extra:
            base.update(extra)
        return base

    def to_hyperparameters_dict(self) -> dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "eval_metric": self.eval_metric,
            "learning_rate": self.learning_rate,
            "l2": self.l2_regularization,
            "l2_regularization": self.l2_regularization,
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
            "restore_best_weights": self.restore_best_weights,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any] | None = None) -> "IterativeTrainingConfig":
        if not params:
            return cls()
        if isinstance(params, cls):
            return params
        mapped: dict[str, Any] = {}
        if "max_epochs" in params:
            mapped["max_epochs"] = min(100, max(1, int(params["max_epochs"])))
        elif "max_iterations" in params:
            mapped["max_epochs"] = min(100, max(1, int(params["max_iterations"])))
        elif "rounds" in params:
            mapped["max_epochs"] = min(100, max(1, int(params["rounds"])))
        if "patience" in params:
            mapped["patience"] = int(params["patience"])
        if "min_delta" in params:
            mapped["min_delta"] = float(params["min_delta"])
        if "eval_metric" in params:
            mapped["eval_metric"] = str(params["eval_metric"])
        if "learning_rate" in params:
            mapped["learning_rate"] = float(params["learning_rate"])
        elif "eta" in params:
            mapped["learning_rate"] = float(params["eta"])
        if "l2_regularization" in params:
            mapped["l2_regularization"] = float(params["l2_regularization"])
        elif "l2" in params:
            mapped["l2_regularization"] = float(params["l2"])
        elif "reg_lambda" in params:
            mapped["l2_regularization"] = float(params["reg_lambda"])
        if "validation_fraction" in params:
            mapped["validation_fraction"] = float(params["validation_fraction"])
        if "seed" in params:
            mapped["seed"] = int(params["seed"])
        if "restore_best_weights" in params:
            val = params["restore_best_weights"]
            mapped["restore_best_weights"] = (
                val.strip().lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)
            )
        if "max_epochs" in mapped and "patience" not in mapped and mapped["max_epochs"] < 10:
            mapped["patience"] = min(10, mapped["max_epochs"])
        return cls(**mapped)


IterativeHyperparameters = IterativeTrainingConfig
IterativeConfig = IterativeTrainingConfig

__all__ = ["IterativeConfig", "IterativeHyperparameters", "IterativeTrainingConfig"]
