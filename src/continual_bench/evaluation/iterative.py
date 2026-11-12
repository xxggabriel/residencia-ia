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
import pickle
import random
import time
from pathlib import Path

from continual_bench.core.tasks import REGRESSION, task_kind
from continual_bench.infrastructure.storage import atomic_write
from continual_bench.models import MODELS, default_model

from .iterative_scores import ClassificationScorer, RegressionScorer, argmax_label, score_rows

CLASSIFIER_PARAMS = {"grace_period": 50, "max_depth": 8, "max_size": 8,
                     "memory_estimate_period": 100, "leaf_prediction": "nba"}


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
                    f"memory {used / 2**20:.0f} MiB above budget {self.limit / 2**20:.0f} MiB")
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
            params["nominal_attributes"] = {"agrawal": ["elevel", "car", "zipcode"],
                                            "stagger": ["size", "color", "shape"]}[dataset]
    return MODELS.get(family).factory(task, **params)(), {"family": family, "params": params}


def _epoch_metrics(prefix, scorer):
    names = {"log_loss", "accuracy", "balanced_accuracy", "f1_macro", "pr_auc", "mae", "rmse", "r2"}
    return {f"{prefix}:{k}": v for k, v in scorer.result().items() if k in names}


def _eval_loss(scorer, is_regression):
    result = scorer.result()
    return result["mae"] if is_regression else result["log_loss"]


def _figures(report):
    from .artifacts.figures import _bar, _line, system_figures

    updates, history = report["windows"], report["epoch_history"]
    names = sorted({k for row in history for k in row if k.startswith(("train:", "validation:"))})
    figures = [
        _line("performance", "metrics_by_epoch", "Métricas por época (eixo global)",
              [{"name": n, "points": [[r["global_epoch"], r.get(n)] for r in history]} for n in names],
              x_label="época global", y_label="valor"),
        _line("adaptation", "epochs_by_update", "Épocas executadas e melhor época por update",
              [{"name": key, "points": [[u["update"], u[key]] for u in report["updates"]]}
               for key in ("epochs_run", "best_epoch", "patience_consumed")],
              x_label="update", y_label="épocas"),
        _line("compute", "seconds_by_update", "Tempo de treino por update",
              [{"name": "train_seconds", "points": [[v["update"], v["train_seconds"]]
                                                       for v in report["model_versions"]]}],
              x_label="update", y_label="s"),
        _bar("costs", "total", "Custo de cenário total", report["costs"]["metrics"]),
    ]
    keys = sorted({k for w in updates for k, v in w.items() if k not in {"scope", "end_seq", "n"}
                   and isinstance(v, (int, float))})
    figures.append(_line("performance", "prequential_by_update", "Prequential por update",
                         [{"name": k, "points": [[w["end_seq"], w.get(k)] for w in updates]} for k in keys],
                         x_label="amostras", y_label="valor"))
    aws = report["costs"].get("aws")
    if aws:
        figures.append(_bar("costs", "aws_usd", "Estimativa de infraestrutura AWS", aws["metrics"],
                            x_label="USD"))
    return figures + system_figures(report.get("system_metrics"))


def run_iterative(data, out_dir, *, max_epochs=100, patience=10, min_delta=0.0, validation_fraction=0.1,
                  seed=42, model_name=None, guard=None, log=print, tracker=None, train_probe=1000):
    from continual_bench.config.benchmark import CostConfig

    from .artifacts import write_artifacts
    from .costs import account
    from .system_metrics import SystemMetricsSampler

    guard = guard or MemoryGuard()
    is_regression = data.task == REGRESSION
    kind, task = task_kind(data.task), data.task
    model, descriptor = make_model(data.name, task, model_name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    started = time.monotonic()
    updates, windows, versions, history, peak = [], [], [], [], 0
    overall = RegressionScorer() if is_regression else ClassificationScorer()
    totals = {"errors": 0.0, "presentations": 0, "train_seconds": 0.0, "predict_seconds": 0.0}
    costs_config = CostConfig(train_sample=0.001, update=0.05).model_dump()
    sampler = SystemMetricsSampler(out)
    sampler.__enter__()
    try:
        for number, (start, stop) in enumerate(update_bounds(data.tasks, len(data)), 1):
            scores = RegressionScorer() if is_regression else ClassificationScorer()
            before = time.perf_counter()
            for index in range(start, stop):
                x, y = data.row(index)
                if is_regression:
                    prediction = model.predict_one(x)
                    scores.update(y, prediction)
                    overall.update(y, prediction)
                    totals["errors"] += kind.error_units(float(y), kind.normalize_prediction(prediction),
                                                         costs_config)
                else:
                    probabilities = model.predict_proba_one(x) or {}
                    scores.update(y, probabilities)
                    overall.update(y, probabilities)
                    totals["errors"] += kind.error_units(y, argmax_label(probabilities), costs_config)
                if index % 2000 == 0:
                    peak = max(peak, guard.check())
            totals["predict_seconds"] += time.perf_counter() - before
            size = stop - start
            val_start = stop - max(1, int(size * validation_fraction))
            train_ids = list(range(start, val_start)) or [start]
            validation = [data.row(i) for i in range(val_start, stop)]
            stride = max(1, len(train_ids) // train_probe)
            probe_ids = train_ids[::stride][:train_probe]
            best, best_state, best_epoch, wait, epochs, stopped = math.inf, None, 0, 0, [], False
            began = time.monotonic()
            for epoch in range(1, max_epochs + 1):
                rng.shuffle(train_ids)
                tick = time.perf_counter()
                for count, index in enumerate(train_ids):
                    x, y = data.row(index)
                    model.learn_one(x, float(y) if is_regression else y)
                    if count % 2000 == 0:
                        peak = max(peak, guard.check())
                train_duration = time.perf_counter() - tick
                totals["train_seconds"] += train_duration
                totals["presentations"] += len(train_ids)
                tick = time.perf_counter()
                val_scores = score_rows(model, is_regression, validation)
                val_duration = time.perf_counter() - tick
                train_scores = score_rows(model, is_regression, (data.row(i) for i in probe_ids))
                loss = _eval_loss(val_scores, is_regression)
                improved = loss < best - min_delta
                if improved:
                    best, best_epoch, wait = loss, epoch, 0
                    best_state = copy.deepcopy(model)
                else:
                    wait += 1
                stopped = wait >= patience
                row = {"epoch": epoch, **_epoch_metrics("train", train_scores),
                       **_epoch_metrics("validation", val_scores),
                       "early_stopped": int(stopped), "best_epoch": best_epoch, "patience_consumed": wait,
                       "training_duration_seconds": train_duration,
                       "validation_duration_seconds": val_duration,
                       "peak_rss_mb": guard.usage() / 2**20, "best_validation_loss": best}
                if is_regression:
                    row["validation:log_loss"] = loss
                history.append({"global_epoch": len(history) + 1, "update": number, **row})
                epochs.append({"epoch": epoch, "validation_loss": loss, "best": improved})
                if tracker is not None:
                    tracker.log_epoch(row)
                log(f"[{data.name}] update {number} epoch {epoch} "
                    f"val_loss={loss:.5f} best_epoch={best_epoch}")
                if stopped:
                    break
                gc.collect()
            if best_state is not None:
                model = best_state
            prequential = scores.result()
            windows.append({"scope": "prequential", "end_seq": stop, **prequential,
                            "update_count": number, "epochs_run": len(epochs), "best_epoch": best_epoch})
            versions.append({"update": number, "end_seq": stop, "train_samples": len(train_ids),
                             "train_seconds": time.monotonic() - began, "iterations": len(epochs),
                             "bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))})
            summary_round = {"epochs_trained": len(epochs), "best_epoch": best_epoch,
                             "early_stopped": int(stopped), "best_validation_loss": best}
            if tracker is not None:
                tracker.finalize_round(summary_round)
            updates.append({"update": number, "range": [start, stop], "samples": size,
                            "train": len(train_ids), "validation": len(validation),
                            "prequential": prequential, "epochs_run": len(epochs), "best_epoch": best_epoch,
                            "best_validation_loss": best, "early_stopped": stopped,
                            "patience_consumed": wait, "seconds": time.monotonic() - began,
                            "epochs": epochs})
            atomic_write(out / "progress.json", json.dumps({"updates": updates}).encode())
            peak = max(peak, guard.check())
    finally:
        sampler.__exit__(None, None, None)
    (out / "model.pkl").write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    summary = {**overall.result(), "update_count": len(updates), "retrain_count": len(updates),
               "epochs_total": len(history),
               "early_stopped_updates": sum(u["early_stopped"] for u in updates)}
    report = {"dataset": data.name, "task": task, "samples": len(data), "model": descriptor,
              "config": {"max_epochs": max_epochs, "patience": patience, "min_delta": min_delta,
                         "validation_fraction": validation_fraction, "seed": seed,
                         "eval_metric": "validation:mae" if is_regression else "validation:log_loss",
                         "train_metrics_probe_rows": train_probe},
              "updates": updates, "windows": windows,
              "model_versions": versions, "summary": summary, "class_metrics": overall.per_class(),
              "epoch_history": history, "seconds": time.monotonic() - started,
              "peak_memory_bytes": peak, "memory_limit_bytes": guard.limit, "complete": True}
    report["costs"] = account(costs_config, errors=totals["errors"], presentations=totals["presentations"],
                              updates=len(updates), train_seconds=totals["train_seconds"],
                              predict_seconds=totals["predict_seconds"],
                              model_storage_bytes=versions[-1]["bytes"] if versions else 0)
    report["system_metrics"] = sampler.report()
    report["artifacts"] = write_artifacts(
        out, _figures(report),
        {"summary/epochs": history, "summary/prequential": windows, "costs/scenario": report["costs"],
         "system/samples": report["system_metrics"]})
    atomic_write(out / "result.json", json.dumps(report, default=str).encode())
    if tracker is not None:
        try:
            report["mlflow"] = tracker.publish(report, out)
        except Exception as error:  # noqa: BLE001 - evidence stays on disk for republication
            report["mlflow"] = {"published": False, "error": f"{type(error).__name__}: {str(error)[:500]}",
                                "tracker_errors": list(tracker.errors)}
            tracker.fail(report["mlflow"]["error"])
        atomic_write(out / "result.json", json.dumps(report, default=str).encode())
    return report
