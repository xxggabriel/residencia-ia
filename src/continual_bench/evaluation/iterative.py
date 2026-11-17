"""Continual updates with up to 100 epochs each and early stopping (AWS/Zia training contract).

The ordered stream is cut into sequential updates (one per task when the source declares at most 20
contiguous tasks, otherwise 10 equal blocks). Each update is first scored test-then-train with the
current model, then trained for up to `max_epochs` shuffled passes; the chronological 10% tail of the
block is the validation set. Training stops after `patience` epochs without improving validation loss
and the best weights are restored (same rules as `models.training.callbacks.EarlyStoppingStrategy`).
A cgroup-aware memory guard turns an impending OOM kill into a recorded failure.
"""

import copy
import gc
import json
import math
import random
import time
from itertools import batched
from pathlib import Path

from continual_bench.core.tasks import BINARY, REGRESSION, task_kind
from continual_bench.infrastructure.storage import atomic_write
from continual_bench.models import MODELS, default_model

from .environment import environment
from .epoch_history import epoch_fit, pickle_state, training_step, write_state
from .iterative_scores import ClassificationScorer, RegressionScorer, argmax_label, score_rows
from .stream_observations import adapted_binary_positive_class, cost_resources, declared_class_labels
from .tracking_observations import observations, process_peak_rss_bytes, quality_scope

CLASSIFIER_PARAMS = {
    "grace_period": 50,
    "max_depth": 8,
    "max_size": 8,
    "memory_estimate_period": 100,
    "leaf_prediction": "nba",
}
STEP_ROWS = 512
OBSERVATION_SCORES = (
    "evaluated_examples",
    "macro_f1",
    "accuracy",
    "positive_precision",
    "positive_recall",
    "average_precision",
    "mae",
    "rmse",
)


class MemoryBudgetExceeded(MemoryError):
    pass


class MemoryGuard:
    """Reads cgroup v2/v1 usage (fallback: process RSS) against a fraction of the container limit."""

    def __init__(self, budget_bytes=None, fraction=0.9):
        limit = budget_bytes or self._cgroup_limit() or 2 * 1024**3
        self.limit = int(limit * fraction)

    @staticmethod
    def _cgroup_limit():
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                value = Path(path).read_text().strip()
                if value.isdigit() and int(value) < 1 << 50:
                    return int(value)
            except OSError:
                pass
        return None

    @staticmethod
    def usage():
        for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            try:
                return int(Path(path).read_text())
            except OSError:
                pass
        try:
            import resource
            import sys

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak if sys.platform == "darwin" else peak * 1024
        except Exception:
            return 0

    def check(self):
        used = self.usage()
        if used > self.limit:
            gc.collect()
            used = self.usage()
            if used > self.limit:
                raise MemoryBudgetExceeded(
                    f"memory {used / 2**20:.0f} MiB above budget {self.limit / 2**20:.0f} MiB"
                )
        return used


def update_bounds(tasks, n):
    runs, start = [], 0
    for i in range(1, n + 1):
        if i == n or tasks[i] != tasks[start]:
            runs.append((start, i))
            start = i
    if tasks[0] is not None and 2 <= len(runs) <= 20:
        return runs
    size = math.ceil(n / 10)
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def make_model(dataset, task, name=None):
    family = name or default_model(task)
    params = {}
    if family == "hoeffding_tree":
        params = dict(CLASSIFIER_PARAMS)
        if dataset in {"agrawal", "stagger"}:
            params["nominal_attributes"] = {
                "agrawal": ["elevel", "car", "zipcode"],
                "stagger": ["size", "color", "shape"],
            }[dataset]
    return MODELS.get(family).factory(task, **params)(), {"family": family, "params": params}


def _epoch_metrics(prefix, scorer):
    return {f"{prefix}:{key}": value for key, value in scorer.result().items()}


def _eval_loss(scorer, is_regression):
    return (scorer.abs if is_regression else scorer.loss) / scorer.n


def _figures(report):
    from .artifacts.figures import _bar, _line, system_figures

    updates, history = report["windows"], report["epoch_history"]
    names = sorted({k for row in history for k in row if k.startswith(("train:", "validation:"))})
    figures = [
        _line(
            "performance",
            "metrics_by_epoch",
            "Métricas por época (eixo global)",
            [{"name": n, "points": [[r["global_epoch"], r.get(n)] for r in history]} for n in names],
            x_label="época global",
            y_label="valor",
        ),
        _line(
            "adaptation",
            "epochs_by_update",
            "Épocas executadas e melhor época por update",
            [
                {"name": key, "points": [[u["update"], u[key]] for u in report["updates"]]}
                for key in ("epochs_run", "best_epoch", "patience_consumed")
            ],
            x_label="update",
            y_label="épocas",
        ),
        _line(
            "compute",
            "seconds_by_update",
            "Tempo de treino por update",
            [
                {
                    "name": "train_seconds",
                    "points": [[v["update"], v["train_seconds"]] for v in report["model_versions"]],
                }
            ],
            x_label="update",
            y_label="s",
        ),
        _bar("costs", "total", "Custo de cenário total", report["costs"]["metrics"]),
    ]
    keys = sorted(
        {
            k
            for w in updates
            for k, v in w.items()
            if k not in {"scope", "end_seq", "n"} and isinstance(v, (int, float))
        }
    )
    figures.append(
        _line(
            "performance",
            "prequential_by_update",
            "Prequential por update",
            [{"name": k, "points": [[w["end_seq"], w.get(k)] for w in updates]} for k in keys],
            x_label="amostras",
            y_label="valor",
        )
    )
    aws = report["costs"].get("aws")
    if aws:
        figures.append(
            _bar("costs", "aws_usd", "Estimativa de infraestrutura AWS", aws["metrics"], x_label="USD")
        )
    return figures + system_figures(report.get("system_metrics"))


def run_iterative(
    data,
    out_dir,
    *,
    max_epochs=100,
    patience=10,
    min_delta=0.0,
    validation_fraction=0.1,
    seed=42,
    model_name=None,
    guard=None,
    log=print,
    tracker=None,
    train_probe=1000,
):
    from continual_bench.config.benchmark import CostConfig

    from .artifacts import write_artifacts
    from .costs import account
    from .system_metrics import SystemMetricsSampler
    from .training_evidence import write_training_artifacts

    if not len(data) or max_epochs < 1 or patience < 1 or train_probe < 1:
        raise ValueError("iterative training requires positive samples, epochs, patience and probe size")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    guard = guard or MemoryGuard()
    is_regression = data.task == REGRESSION
    kind, task = task_kind(data.task), data.task
    model, descriptor = make_model(data.name, task, model_name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    started = time.monotonic()
    updates, windows, versions, history, peak = [], [], [], [], 0
    fits, observed_windows = [], []
    bounds = update_bounds(data.tasks, len(data))
    manifest = getattr(data, "meta", None) or {}
    positive = adapted_binary_positive_class(manifest) if not is_regression else None
    observed_task = BINARY if positive else task
    target_unit = manifest.get("target_unit") or "target_unit"
    declared = declared_class_labels(manifest)
    source_labels = getattr(data, "ys", None)
    if not is_regression and source_labels is None:
        source_labels = (data.row(index)[1] for index in range(len(data)))
    classes = () if is_regression else (*declared, *sorted(set(source_labels) - set(declared)))

    def scorer(*, retain_residuals=False):
        return (
            RegressionScorer(retain_residuals=retain_residuals)
            if is_regression
            else ClassificationScorer(observed_positive_class=positive)
        )

    def collect(score, *, curves=False):
        return score.quality(target_unit) if is_regression else score.quality(classes, curves=curves)

    overall = RegressionScorer(retain_residuals=True) if is_regression else scorer()
    totals = {"errors": 0.0, "presentations": 0, "train_seconds": 0.0, "predict_seconds": 0.0}
    inference_seconds = quality_seconds = 0.0
    costs_config = CostConfig(train_sample=0.001, update=0.05).model_dump()
    sampler = SystemMetricsSampler(out)
    sampler.__enter__()
    try:
        for number, (start, stop) in enumerate(bounds, 1):
            scores = scorer()
            before = time.perf_counter()
            fit_inference_seconds = 0.0
            for index in range(start, stop):
                x, y = data.row(index)
                tick = time.perf_counter()
                if is_regression:
                    prediction = model.predict_one(x)
                    fit_inference_seconds += time.perf_counter() - tick
                    scores.update(y, prediction)
                    overall.update(y, prediction)
                    totals["errors"] += kind.error_units(
                        float(y), kind.normalize_prediction(prediction), costs_config
                    )
                else:
                    probabilities = model.predict_proba_one(x) or {}
                    fit_inference_seconds += time.perf_counter() - tick
                    scores.update(y, probabilities)
                    overall.update(y, probabilities)
                    totals["errors"] += kind.error_units(y, argmax_label(probabilities), costs_config)
                if index % 2000 == 0:
                    peak = max(peak, guard.check())
            totals["predict_seconds"] += time.perf_counter() - before
            inference_seconds += fit_inference_seconds
            quality_seconds += scores.tracking_seconds
            tick = time.perf_counter()
            observed_windows.append(
                {
                    "step": stop,
                    "start_seq": start,
                    "scope": "prequential",
                    "model_version": fits[-1]["deployed_version"] if fits else None,
                    "fit_id": fits[-1]["fit_id"] if fits else None,
                    "quality": collect(scores),
                }
            )
            quality_seconds += time.perf_counter() - tick
            size = stop - start
            val_start = stop - max(1, int(size * validation_fraction)) if size > 1 else stop
            train_ids = list(range(start, val_start))
            validation = [data.row(i) for i in range(val_start, stop)]
            stride = max(1, len(train_ids) // train_probe)
            probe_ids = train_ids[::stride][:train_probe]
            best, best_state, best_epoch, wait, epochs, stopped = math.inf, None, 0, 0, [], False
            rows, observed_steps, observed_epochs = [], [], []
            fit_train_seconds = 0.0
            cumulative = 0
            best_scopes = {}
            began = time.monotonic()
            for epoch in range(1, max_epochs + 1):
                epoch_started = time.perf_counter()
                rng.shuffle(train_ids)
                tick = time.perf_counter()
                first_step = len(observed_steps) + 1
                count = 0
                for group in batched(train_ids, STEP_ROWS):
                    group_started = time.perf_counter()
                    for index in group:
                        x, y = data.row(index)
                        model.learn_one(x, float(y) if is_regression else y)
                        if count % 2000 == 0:
                            peak = max(peak, guard.check())
                        count += 1
                    group_duration = time.perf_counter() - group_started
                    observed_started = time.perf_counter()
                    cumulative += len(group)
                    repeated_epochs = max(0, cumulative - len(train_ids))
                    point = training_step(
                        len(observed_steps) + 1,
                        examples=len(group),
                        cumulative=cumulative,
                        duration_seconds=group_duration,
                        metrics={
                            "training/replay_presentations_cumulative": 0,
                            "training/repeated_epoch_presentations_cumulative": repeated_epochs,
                        },
                    )
                    observed_steps.append(point)
                    quality_seconds += time.perf_counter() - observed_started
                train_duration = time.perf_counter() - tick
                fit_train_seconds += train_duration
                totals["train_seconds"] += train_duration
                totals["presentations"] += len(train_ids)
                tick = time.perf_counter()
                val_scores = (
                    score_rows(model, is_regression, validation, scorer(retain_residuals=True))
                    if validation
                    else None
                )
                val_duration = time.perf_counter() - tick if validation else 0.0
                train_scores = score_rows(
                    model, is_regression, (data.row(i) for i in probe_ids), scorer(retain_residuals=True)
                )
                loss = _eval_loss(val_scores if validation else train_scores, is_regression)
                improved = loss < best - min_delta
                if improved:
                    best, best_epoch, wait = loss, epoch, 0
                    best_state = copy.deepcopy(model)
                else:
                    wait += 1
                stopped = wait >= patience
                tick = time.perf_counter()
                scopes = {"train_probe": collect(train_scores, curves=improved)}
                if validation:
                    scopes["validation"] = collect(val_scores, curves=improved)
                if improved:
                    best_scopes = {
                        scope: quality_scope(observed_task, values) for scope, values in scopes.items()
                    }
                metrics = {
                    f"{scope}/{key}": values.get(key)
                    for scope, values in scopes.items()
                    for key in OBSERVATION_SCORES
                }
                metrics.update(
                    {
                        "train_probe/evaluation_loss": _eval_loss(train_scores, is_regression),
                        "validation/evaluation_loss": loss if validation else None,
                        "training/examples_processed": len(train_ids),
                        "training/presentations_cumulative": cumulative,
                        "training/duration_seconds": train_duration,
                        "validation/duration_seconds": val_duration,
                        "epoch/duration_seconds": time.perf_counter() - epoch_started,
                    }
                )
                observed_epochs.append(
                    {
                        "epoch": epoch,
                        "first_step": first_step,
                        "last_step": len(observed_steps),
                        "best": improved,
                        "early_stopped": stopped,
                        "metrics": metrics,
                    }
                )
                quality_seconds += time.perf_counter() - tick + train_scores.tracking_seconds
                if val_scores is not None:
                    quality_seconds += val_scores.tracking_seconds
                row = {
                    "epoch": epoch,
                    **_epoch_metrics("train", train_scores),
                    **(_epoch_metrics("validation", val_scores) if validation else {}),
                    "early_stopped": int(stopped),
                    "best_epoch": best_epoch,
                    "patience_consumed": wait,
                    "training_duration_seconds": train_duration,
                    "validation_duration_seconds": val_duration,
                    "train_samples": len(train_ids),
                    "validation_samples": len(validation),
                    "train_probe_samples": len(probe_ids),
                    "peak_rss_mb": guard.usage() / 2**20,
                    "best_validation_loss": best if validation else None,
                    "best_score": best,
                }
                rows.append(row)
                history.append({"global_epoch": len(history) + 1, "update": number, **row})
                epochs.append(
                    {
                        "epoch": epoch,
                        "validation_loss": loss if validation else None,
                        "best": improved,
                    }
                )
                if tracker is not None:
                    tracker.log_epoch(row)
                scope = "val" if validation else "train"
                log(
                    f"[{data.name}] update {number} epoch {epoch} "
                    f"{scope}_loss={loss:.5f} best_epoch={best_epoch}"
                )
                if stopped:
                    break
                gc.collect()
            if best_state is not None:
                model = best_state
            prequential = scores.result()
            windows.append(
                {
                    "scope": "prequential",
                    "end_seq": stop,
                    **prequential,
                    "update_count": number,
                    "epochs_run": len(epochs),
                    "best_epoch": best_epoch,
                }
            )
            filename = "model.pkl" if number == len(bounds) else f"models/fit-{number:06d}.pkl"
            checkpoint = write_state(out, filename, pickle_state(model), format="pickle")
            summary_round = {
                "epochs_trained": len(epochs),
                "best_epoch": best_epoch,
                "early_stopped": int(stopped),
                "best_validation_loss": best if validation else None,
                "best_score": best,
            }
            selection_metric = "mae" if is_regression else "log_loss"
            selection_scope = "validation" if validation else "train"
            fit = epoch_fit(
                rows,
                number=number,
                action="initial" if number == 1 else "continue",
                previous_steps=len(history) - len(rows),
                start_seq=start,
                end_seq=stop,
                block_index=number - 1,
                train_samples=len(train_ids),
                validation_samples=len(validation),
                train_seconds=fit_train_seconds,
                checkpoint=checkpoint,
                deployed_version=number - 1,
                summary={
                    **summary_round,
                    "selection_scope": selection_scope,
                    "selection_metric": selection_metric,
                },
                observation_steps=observed_steps,
                observation_epochs=observed_epochs,
                observed_step_kind=f"batch_of_up_to_{STEP_ROWS}_sequential_learner_calls",
                source_model_version=number - 2 if number > 1 else None,
                promotion_status="promoted",
                observation_effort={
                    "unique_training_examples": len(train_ids),
                    "training_presentations": cumulative,
                    "replay_presentations": 0,
                    "repeated_epoch_presentations": cumulative - len(train_ids),
                    "steps": len(observed_steps),
                    "epochs": len(rows),
                },
                observation_resources={
                    "training_seconds": fit_train_seconds,
                    "duration_seconds": time.monotonic() - began,
                },
                observation_scopes=best_scopes,
            )
            fit.update(
                best_epoch=best_epoch,
                last_epoch=len(rows),
                checkpoint_epoch=best_epoch,
                native_loss_availability={
                    "status": "unavailable",
                    "reason": "learner_does_not_expose_loss",
                },
            )
            fits.append(fit)
            versions.append(
                {
                    "update": number,
                    "end_seq": stop,
                    "train_samples": len(train_ids),
                    "train_seconds": time.monotonic() - began,
                    "iterations": len(epochs),
                    "fit_id": fit["fit_id"],
                    **checkpoint,
                }
            )
            if tracker is not None:
                tracker.finalize_round(summary_round)
            updates.append(
                {
                    "update": number,
                    "range": [start, stop],
                    "samples": size,
                    "train": len(train_ids),
                    "validation": len(validation),
                    "prequential": prequential,
                    "epochs_run": len(epochs),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best if validation else None,
                    "early_stopped": stopped,
                    "patience_consumed": wait,
                    "seconds": time.monotonic() - began,
                    "epochs": epochs,
                }
            )
            atomic_write(out / "progress.json", json.dumps({"updates": updates}).encode())
            peak = max(peak, guard.check())
    finally:
        sampler.__exit__(None, None, None)
    summary = {
        **overall.result(),
        "update_count": len(updates),
        "retrain_count": len(updates),
        "epochs_total": len(history),
        "early_stopped_updates": sum(u["early_stopped"] for u in updates),
    }
    report = {
        "dataset": data.name,
        "task": task,
        "samples": len(data),
        "model": descriptor,
        "dataset_manifest": getattr(data, "meta", None),
        "environment": environment(),
        "config": {
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "eval_metric": "validation:mae" if is_regression else "validation:log_loss",
            "train_metrics_probe_rows": train_probe,
        },
        "updates": updates,
        "windows": windows,
        "model_versions": versions,
        "summary": summary,
        "class_metrics": overall.per_class(),
        "epoch_history": history,
        "training_history": fits,
        "seconds": time.monotonic() - started,
        "peak_memory_bytes": peak,
        "memory_limit_bytes": guard.limit,
        "complete": True,
    }
    report["costs"] = account(
        costs_config,
        errors=totals["errors"],
        presentations=totals["presentations"],
        updates=len(updates),
        train_seconds=totals["train_seconds"],
        predict_seconds=totals["predict_seconds"],
        model_storage_bytes=sum(version["bytes"] for version in versions),
    )
    report["system_metrics"] = sampler.report()
    tick = time.perf_counter()
    evaluated = len(data)
    quality = collect(overall, curves=True)
    train_unique = sum(update["train"] for update in updates)
    report["tracking_observations"] = observations(
        observed_task,
        {"prequential": quality},
        windows=observed_windows,
        effort={
            "examples_observed": evaluated,
            "evaluated_examples": evaluated,
            "unique_training_examples": train_unique,
            "training_presentations": totals["presentations"],
            "replay_presentations": 0,
            "repeated_epoch_presentations": totals["presentations"] - train_unique,
            "steps": sum(len(fit["observation_steps"]) for fit in fits),
            "epochs": len(history),
            "updates": len(fits) - 1,
            "promoted_versions": len(versions),
            "initial_unique_training_examples": updates[0]["train"],
            "initial_training_presentations": fits[0]["training_presentations"],
            "initial_steps": len(fits[0]["observation_steps"]),
            "initial_epochs": len(fits[0]["steps"]),
            "update_training_presentations": totals["presentations"] - fits[0]["training_presentations"],
            "update_steps": sum(len(fit["observation_steps"]) for fit in fits[1:]),
            "update_epochs": len(history) - len(fits[0]["steps"]),
        },
        resources={
            "duration_seconds": report["seconds"],
            "training_seconds": totals["train_seconds"],
            "inference_seconds": inference_seconds,
            "inference_examples": evaluated,
            "inference_scope": "prequential before each update; excludes train probe and validation",
            "predictions_per_second": evaluated / inference_seconds if inference_seconds else None,
            "peak_rss_bytes": process_peak_rss_bytes(),
            "rss_scope": "process lifetime high-water mark",
            **cost_resources(report["costs"], evaluated),
        },
        lineage={
            "dataset": data.name,
            "dataset_version": manifest.get("version"),
            "model": descriptor,
            "policy": "ordered_continuation_with_early_stopping",
            "seed": seed,
            "environment": report["environment"],
            "code_revision": report["environment"]["code_sha256"],
            "protocol": "prequential windows with separate chronological validation for each fit",
            "metric_implementation_revisions": {
                "r2": "welford-central-moment-v2",
                "macro_f1": "whole-dataset-target-universe-v1",
                "average_precision": "explicit-binary-task-and-observed-scores-v1",
            },
        },
    )
    measured = report["tracking_observations"]["resources"]
    measured["tracking_overhead_seconds"] = (
        quality_seconds + overall.tracking_seconds + time.perf_counter() - tick
    )
    measured["availability"].pop("tracking_overhead_seconds", None)
    report["artifacts"] = write_artifacts(
        out,
        _figures(report),
        {
            "summary/epochs": history,
            "summary/prequential": windows,
            "costs/scenario": report["costs"],
            "system/samples": report["system_metrics"],
        },
    )
    report["artifacts"] = write_training_artifacts(out, report)
    atomic_write(out / "result.json", json.dumps(report, default=str).encode())
    if tracker is not None:
        try:
            report["mlflow"] = tracker.publish(report, out)
        except Exception as error:  # noqa: BLE001 - evidence stays on disk for republication
            report["mlflow"] = {
                "published": False,
                "error": f"{type(error).__name__}: {str(error)[:500]}",
                "tracker_errors": list(tracker.errors),
            }
            tracker.fail(report["mlflow"]["error"])
        atomic_write(out / "result.json", json.dumps(report, default=str).encode())
    return report
