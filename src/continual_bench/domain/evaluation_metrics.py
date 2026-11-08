"""Versioned supplemental evaluation; diagnostic results never drive adaptation."""

import copy
import hashlib
import json
import math


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _accuracy(value):
    if value is not None and (type(value) not in (int, float)
                              or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("accuracy must be finite in [0,1] or None")
    return value


def task_matrix_metrics(matrix, initial=None):
    """Summarize completed task rows without changing historical Wang-style metrics.

    GEM BWT/FWT use Eq. 3/4; forgetting uses only postlearning earlier rows.
    Missing required cells propagate rather than averaging observed tasks only.
    """
    if not matrix:
        return {"available": False, "reason": "no_completed_tasks", "metrics": {}}
    width = len(matrix[0])
    if not width or len(matrix) > width or any(len(row) != width for row in matrix):
        raise ValueError("task matrix must be rectangular with at most one row per task")
    for row in matrix:
        for value in row:
            _accuracy(value)
    if initial is not None:
        if len(initial) != width:
            raise ValueError("initial baseline must have one entry per task")
        for value in initial:
            _accuracy(value)

    def mean(values):
        return sum(values) / len(values) if values and all(v is not None for v in values) else None

    last = len(matrix) - 1
    backward, forgetting, retention = [], [], []
    for task in range(last):
        current, diagonal = matrix[last][task], matrix[task][task]
        history = [matrix[row][task] for row in range(task, last)]
        backward.append(current - diagonal if current is not None and diagonal is not None else None)
        forgetting.append(max(history) - current
                          if current is not None and all(v is not None for v in history) else None)
        retention.append(current)
    forward = [matrix[task - 1][task] - initial[task]
               if initial is not None and matrix[task - 1][task] is not None and initial[task] is not None
               else None for task in range(1, last + 1)]
    values = {
        "gem_average_accuracy": mean(matrix[last][:last + 1]),
        "gem_backward_transfer": mean(backward),
        "gem_forward_transfer": mean(forward),
        "postlearning_forgetting": mean(forgetting),
        "past_task_retention_accuracy": mean(retention),
    }
    return {
        "available": True,
        "metrics": values,
        "availability": {name: {"available": value is not None,
                                "reason": None if value is not None else (
                                    "insufficient_tasks" if not last else
                                    "missing_initial_baseline" if name == "gem_forward_transfer"
                                    and initial is None else "missing_required_cell")}
                         for name, value in values.items()},
        "completed_tasks": len(matrix),
        "declared_tasks": width,
        "contract": {
            "average": "completed tasks only; equal task weights",
            "bwt": "mean R[last,j]-R[j,j] for j<last; GEM Eq.3",
            "fwt": "mean R[j-1,j]-initial[j] for 1<=j<=last; GEM Eq.4",
            "forgetting": "mean max(R[i,j] for j<=i<last)-R[last,j]; unclipped",
            "retention": "mean final accuracy over previous tasks; no ratio to initial performance",
            "historical_metrics": "supplement; does not replace residency-metrics-v1 definitions",
        },
    }


def validate_temporal_holdout(train, test, *, cutoff):
    """Validate explicit Event partitions and delayed labels at the fit cutoff.

    Duplicate/group leakage outside event IDs requires dataset-specific manifests.
    Chronological row order alone cannot establish a natural time axis.
    """
    _integer(cutoff, "cutoff")
    if not train or not test:
        raise ValueError("both train and test partitions must be nonempty")
    identities = set()
    for name, partition in (("train", train), ("test", test)):
        last_seq = last_stamp = -1
        for event in partition:
            if event.seq <= last_seq or event.timestamp <= last_stamp:
                raise ValueError(f"{name} partition must be strictly chronological")
            if event.event_id in identities:
                raise ValueError("duplicate event ID within or across partitions")
            if event.available_at < event.timestamp:
                raise ValueError("label precedes features")
            if name == "train" and (event.timestamp >= cutoff or event.available_at >= cutoff):
                raise ValueError("training features and labels must be available before cutoff")
            if name == "test" and event.timestamp < cutoff:
                raise ValueError("test features precede cutoff")
            identities.add(event.event_id)
            last_seq, last_stamp = event.seq, event.timestamp
    if train[-1].seq >= test[0].seq:
        raise ValueError("training sequence overlaps test sequence")
    return {"train_samples": len(train), "test_samples": len(test), "cutoff": cutoff,
            "contract": "train features and labels < cutoff <= test features; IDs disjoint"}


class NearFutureEvaluation:
    """Bounded frozen-model horizon windows; no labels consumed in feature hook.

    The gap is explicit and must be chosen on development data. This operational
    binary adaptation of NFA does not reproduce a published visual benchmark.
    """

    def __init__(self, *, warmup, horizon=32, gap=0, max_windows=16):
        self.warmup = _integer(warmup, "warmup", 1)
        self.horizon = _integer(horizon, "horizon", 1)
        self.gap = _integer(gap, "gap")
        self.max_windows = _integer(max_windows, "max_windows", 1)
        if horizon > 1024 or max_windows > 128:
            raise ValueError("near-future evaluation is bounded at 1024 samples and 128 windows")
        self.windows = []
        self.next_seq = warmup
        self.last_timestamp = -1

    def before_prediction(self, event, model, labeled, prediction_seconds=0):
        _integer(event.seq, "event seq")
        if event.seq < self.warmup:
            return
        if event.seq < self.next_seq:
            # Replay is harmless only if feature metadata has not changed.
            for window in self.windows:
                for row in window["predictions"].values():
                    if row["seq"] != event.seq:
                        continue
                    if row["fingerprint"] != self._fingerprint(event):
                        raise ValueError("conflicting replayed near-future event")
            return
        if event.seq != self.next_seq:
            raise ValueError("near-future features must be contiguous")
        _integer(event.timestamp, "event timestamp")
        _integer(event.available_at, "label availability")
        if event.timestamp <= self.last_timestamp:
            raise ValueError("near-future event timestamps must increase")
        _integer(labeled, "labeled")
        if labeled > event.seq:
            raise ValueError("model has labels from the prediction sample or the future")
        if event.available_at < event.timestamp:
            raise ValueError("label precedes features")
        fingerprint = self._fingerprint(event)
        if any(event.event_id in window["predictions"] for window in self.windows):
            raise ValueError("duplicate event ID at a new source sequence")
        period = self.gap + self.horizon
        new_window = None
        if (event.seq - self.warmup) % period == 0 and len(self.windows) < self.max_windows:
            new_window = {
                "anchor_seq": event.seq - 1,
                "captured_at": event.timestamp,
                "trained_label_count": labeled,
                "start_seq": event.seq + self.gap,
                "stop_seq": event.seq + period,
                "model": copy.deepcopy(model),
                "predictions": {},
            }
        windows = self.windows + ([new_window] if new_window is not None else [])
        additions = []
        for window in windows:
            if window["start_seq"] <= event.seq < window["stop_seq"]:
                # Copying also isolates accidental mutation during predict().
                probability = copy.deepcopy(window["model"]).predict(copy.deepcopy(event.x))
                _accuracy(probability)
                if probability is None:
                    raise ValueError("frozen-model probability cannot be None")
                additions.append((window, {
                    "event_id": event.event_id, "seq": event.seq, "timestamp": event.timestamp,
                    "available_at": event.available_at,
                    "probability": probability, "label": None,
                    "released_at": None, "fingerprint": fingerprint,
                }))
        # Commit only once all prediction calls and contracts have succeeded.
        if new_window is not None:
            self.windows.append(new_window)
        for window, row in additions:
            window["predictions"][event.event_id] = row
            if event.seq + 1 == window["stop_seq"]:
                window["model"] = None  # Pending labels require predictions, not the model object.
        self.next_seq += 1
        self.last_timestamp = event.timestamp

    @staticmethod
    def _fingerprint(event):
        raw = [event.event_id, event.seq, event.timestamp, event.available_at, event.x]
        return hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def after_label(self, event, probability, trace, released_at, model, evaluated):
        if not evaluated:
            return
        _integer(released_at, "released_at")
        if type(event.y) is not bool:
            raise ValueError("near-future metric requires explicit binary labels")
        for window in self.windows:
            row = window["predictions"].get(event.event_id)
            if row is None:
                continue
            if row["fingerprint"] != self._fingerprint(event):
                raise ValueError("released label does not match predicted features")
            if released_at < row["available_at"]:
                raise ValueError("near-future label is not available yet")
            if row["label"] is not None:
                if row["label"] != event.y:
                    raise ValueError("conflicting replayed near-future label")
                continue
            row.update(label=event.y, released_at=released_at)

    def report(self):
        windows = []
        for window in self.windows:
            predictions = list(window["predictions"].values())
            released = [row for row in predictions if row["label"] is not None]
            correct = sum((row["probability"] >= 0.5) == row["label"] for row in released)
            complete = len(released) == self.horizon
            windows.append({
                **{key: value for key, value in window.items() if key not in ("model", "predictions")},
                "predicted": len(predictions), "released": len(released), "complete": complete,
                "near_future_accuracy": correct / self.horizon if complete else None,
                "observed_accuracy": correct / len(released) if released else None,
                "reason": None if complete else "incomplete_horizon_or_unreleased_labels",
                "predictions": copy.deepcopy(predictions),
            })
        completed = [row["near_future_accuracy"] for row in windows if row["complete"]]
        return {"version": "semana09-near-future-v1", "windows": windows,
                "near_future_accuracy": sum(completed) / len(completed) if completed else None,
                "completed_windows": len(completed), "scheduled_windows": len(windows),
                "horizon": self.horizon, "gap": self.gap, "max_windows": self.max_windows,
                "reason": None if completed else "no_complete_frozen_horizon",
                "contract": "equal complete-window weights; frozen before gap; labels scored after release",
                "scope": "bounded binary operational NFA; not UPF performance forecasting"}
