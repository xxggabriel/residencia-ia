"""Declarative experiment matrices: datasets × models × strategies × modes × seeds.

Builder pattern: `ExperimentMatrix.plan()` expands the Cartesian product into
validated case configurations for one evaluator and reports every combination
it skipped (with the reason) instead of silently dropping it. `execute_matrix`
runs the plan with a resumable journal and writes a flat metrics summary for
cross-strategy comparison.
"""

import hashlib
import json
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from continual_bench.config import Component, Config, EpochConfig, PolicyConfig, Strict, TypedConfig
from continual_bench.core.tasks import BINARY, MULTICLASS
from continual_bench.datasets.catalog import CATALOG
from continual_bench.infrastructure.storage import atomic_write, run_lock
from continual_bench.models import MODELS

MAX_CASES = 4096


class Strategy(Strict):
    """One adaptation strategy: a drift detector paired with a retraining policy."""

    detector: Component = Field(default_factory=lambda: Component(name="none"))
    policy: PolicyConfig = Field(default_factory=lambda: PolicyConfig(name="none"))


class MatrixDataset(Strict):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    samples: int | None = Field(default=None, ge=10, le=100_000)
    label_delay: int | None = Field(default=None, ge=0, le=10_000)
    params: dict = Field(default_factory=dict)


class ExperimentMatrix(Strict):
    version: Literal["matrix-v1"] = "matrix-v1"
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    evaluator: Literal["binary", "typed", "epochs"] = "binary"
    datasets: tuple[MatrixDataset, ...] = Field(min_length=1)
    models: tuple[Component, ...] = Field(default=(Component(name="logistic"),), min_length=1)
    strategies: tuple[Strategy, ...] = Field(default=(Strategy(),), min_length=1)
    modes: tuple[Literal["streaming", "incremental", "offline"], ...] = ("streaming",)
    typed_policies: tuple[Literal["none", "online", "periodic"], ...] = ("online",)
    seeds: tuple[int, ...] = Field(default=(42,), min_length=1)
    # Shared fields merged into every case (e.g. warmup, metric_window, tracking, epochs).
    base: dict = Field(default_factory=dict)
    on_incompatible: Literal["skip", "error"] = "skip"

    @model_validator(mode="after")
    def bounded(self):
        size = len(self.datasets) * len(self.models) * len(self.seeds) * (
            len(self.modes) * len(self.strategies) if self.evaluator == "binary"
            else len(self.typed_policies) if self.evaluator == "typed" else 1)
        if size > MAX_CASES:
            raise ValueError(f"matrix expands to {size} combinations; limit is {MAX_CASES}")
        return self

    def identity(self) -> str:
        raw = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    @staticmethod
    def task_for(evaluator: str, dataset: str) -> str | None:
        """Target type the model must support; binary streams become two classes off-engine."""
        task = CATALOG.get(dataset).task if dataset in CATALOG else None
        if evaluator != "binary" and task == BINARY:
            return MULTICLASS
        return task

    def _incompatibility(self, dataset: str, model: str) -> str | None:
        task = self.task_for(self.evaluator, dataset)
        if self.evaluator == "binary" and task not in (None, BINARY):
            return f"dataset {dataset} has {task} targets; binary engine requires binary"
        if task is not None and model in MODELS and not MODELS.get(model).supports(task):
            return f"model {model} does not support {task}"
        return None

    def _dataset(self, entry: MatrixDataset, seed: int) -> dict:
        dataset = {"name": entry.name, "seed": seed, "params": entry.params}
        if entry.samples is not None:
            dataset["samples"] = entry.samples
        if entry.label_delay is not None:
            dataset["label_delay"] = entry.label_delay
        return dataset

    def _binary(self, entry, model, seed):
        for mode, strategy in product(self.modes, self.strategies):
            if mode == "offline" and strategy.policy.name != "none":
                continue  # Offline is a frozen holdout; adaptive strategies do not apply.
            yield {**deepcopy(self.base), "name": self.name, "mode": mode, "model": model.model_dump(),
                   "dataset": self._dataset(entry, seed), "detector": strategy.detector.model_dump(),
                   "policy": strategy.policy.model_dump()}, Config

    def _typed(self, entry, model, seed):
        for policy in self.typed_policies:
            yield {**deepcopy(self.base), "policy": policy, "model": model.model_dump(),
                   "dataset": self._dataset(entry, seed)}, TypedConfig

    def _epochs(self, entry, model, seed):
        case = {**deepcopy(self.base), "dataset": entry.name, "seed": seed, "model": model.model_dump()}
        if entry.samples is not None:
            case["examples"] = entry.samples
        yield case, EpochConfig

    def plan(self) -> dict:
        builder = {"binary": self._binary, "typed": self._typed, "epochs": self._epochs}[self.evaluator]
        cases, skipped, keys = [], [], set()
        for entry, model, seed in product(self.datasets, self.models, self.seeds):
            reason = self._incompatibility(entry.name, model.name)
            if reason is not None:
                if self.on_incompatible == "error":
                    raise ValueError(reason)
                skipped.append({"dataset": entry.name, "model": model.name, "seed": seed, "reason": reason})
                continue
            for data, schema in builder(entry, model, seed):
                try:
                    config = schema.model_validate(data)
                except ValidationError as error:
                    if self.on_incompatible == "error":
                        raise
                    skipped.append({"dataset": entry.name, "model": model.name, "seed": seed,
                                    "reason": error.errors()[0]["msg"]})
                    continue
                key = case_key(config)
                if key not in keys:  # Offline cases repeat across strategies.
                    keys.add(key)
                    cases.append(config)
        return {"matrix": self.identity(), "evaluator": self.evaluator, "cases": cases, "skipped": skipped}


def case_key(config) -> str:
    if isinstance(config, Config):
        return config.identity()
    raw = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def describe(config) -> dict:
    """Comparison coordinates of a case, independent of its evaluator."""
    if isinstance(config, Config):
        return {"dataset": config.dataset.name, "seed": config.dataset.seed, "model": config.model.name,
                "mode": config.mode, "detector": config.detector.name, "policy": config.policy.name}
    if isinstance(config, TypedConfig):
        return {"dataset": config.dataset.name, "seed": config.dataset.seed,
                "model": config.model.name if config.model else None, "mode": "typed_prequential",
                "detector": None, "policy": config.policy}
    return {"dataset": config.dataset, "seed": config.seed,
            "model": config.model.name if config.model else None, "mode": "epochs", "detector": None,
            "policy": None}


def _execute_case(config, root: Path, cache: str, real_directory: str | None):
    if isinstance(config, Config):
        from .runner import run

        return run(config, str(root / "binary"))
    if isinstance(config, TypedConfig):
        from continual_bench.evaluation.typed import run_typed

        from .components import default_registry, make_source

        source = make_source(default_registry(), config.dataset)
        return run_typed(source, str(root / "typed"), **config.model_dump(exclude={"dataset"}))
    from continual_bench.evaluation.epochs import run_epochs

    return run_epochs(config, str(root / "epochs"), cache, real_directory=real_directory)


def _headline(report) -> dict:
    """Comparable final metrics: binary aggregate, typed test/train split or last epoch."""
    if "epochs" in report:
        return report["epochs"][-1]["metrics"] | {"cost_total": report["costs"]["metrics"]["cost_total"]}
    if "per_task" in report:
        split = next((s for s in ("test", "validation", "train") if report["metrics"][s]["n"]), "train")
        return report["metrics"][split] | {"split": split,
                                            "cost_total": report["costs"]["metrics"]["cost_total"]}
    summary = {k: report["metrics"].get(k) for k in ("accuracy", "f1", "brier", "log_loss", "roc_auc")}
    return summary | {"updates": len(report.get("updates", [])), "alarms": len(report.get("alarms", []))}


def execute_matrix(matrix: ExperimentMatrix, root, *, cache="state/datasets", real_directory=None,
                   max_seconds=86_400):
    """Run every planned case once; a rerun resumes, reusing completed reports."""
    plan = matrix.plan()
    directory = Path(root) / matrix.identity()
    journal_path = directory / "matrix.json"
    with run_lock(root, "matrix-" + matrix.identity()):
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            "matrix": matrix.model_dump(mode="json"), "skipped": plan["skipped"], "cases": {},
            "complete": False}
        started = time.monotonic()
        for config in plan["cases"]:
            key = case_key(config)
            if journal["cases"].get(key, {}).get("status") == "passed":
                continue
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("matrix deadline reached before next case")
            journal["cases"][key] = {"status": "running", **describe(config)}
            atomic_write(journal_path, json.dumps(journal, indent=2).encode())
            try:
                report = _execute_case(config, directory, cache, real_directory)
            except Exception as error:
                # Store type only: third-party error strings can contain credential URLs.
                journal["cases"][key] = {"status": "failed", "error_type": type(error).__name__,
                                         **describe(config)}
                atomic_write(journal_path, json.dumps(journal, indent=2).encode())
                raise
            journal["cases"][key] = {"status": "passed", **describe(config), "run_key": report["run_key"],
                                     "metrics": _headline(report)}
            atomic_write(journal_path, json.dumps(journal, indent=2).encode())
        journal["complete"] = True
        atomic_write(journal_path, json.dumps(journal, indent=2).encode())
        rows = [{"case": key, **row} for key, row in journal["cases"].items()]
        atomic_write(directory / "summary.json", json.dumps(rows, indent=2).encode())
        return journal
