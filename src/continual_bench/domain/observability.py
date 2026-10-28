"""Read-only evaluation instrumentation; never participates in policy decisions."""

import copy
import math
import time


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class OnlineObservability:
    def __init__(self, settings, continual=None, memory_sampler=None):
        self.settings = settings
        self.continual = continual
        self.memory_sampler = memory_sampler
        self.predictions = []
        self.shadow = None
        self.shadow_id = None
        self.comparisons = []
        self.instrumentation_seconds = 0.0
        self.warmup_train_seconds = 0.0
        self.warmup_train_samples = 0
        self.fading_correct = self.fading_weight = 0.0
        self.fading_series = []
        self.memory_samples = []
        self.monitor_seconds = 0.0
        self.memory_breaks = set()
        self.start_memory_segment = True

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.start_memory_segment = True

    def before_prediction(self, event, model, labeled, prediction_seconds):
        started = time.perf_counter()
        old_probability = self.shadow.predict(dict(event.x)) if self.shadow is not None else None
        trace = {
            "prediction_seconds": prediction_seconds,
            "trained_labels_at_prediction": labeled,
            "comparison_id": self.shadow_id,
            "previous_probability": old_probability,
        }
        self.instrumentation_seconds += time.perf_counter() - started
        return trace

    def before_update(self, model):
        started = time.perf_counter()
        previous = copy.deepcopy(model) if self.settings.get("adaptation_shadows") else None
        self.instrumentation_seconds += time.perf_counter() - started
        return previous

    def after_update(self, previous, update):
        if previous is not None:
            self.shadow = previous
            self.shadow_id = len(self.comparisons)
            self.comparisons.append(
                {
                    "id": self.shadow_id,
                    **update,
                    "evaluated": 0,
                    "old_errors": 0,
                    "new_errors": 0,
                    "old_weighted_loss": 0.0,
                    "new_weighted_loss": 0.0,
                }
            )

    def after_label(self, event, probability, trace, released_at, model, evaluated):
        started = time.perf_counter()
        if evaluated:
            row = {
                "event_id": event.event_id,
                "seq": event.seq,
                "timestamp": event.timestamp,
                "available_at": event.available_at,
                "released_at": released_at,
                "probability": probability,
                "label": bool(event.y),
                **trace,
            }
            self.predictions.append(row)
            factor = self.settings["fading_factor"]
            self.fading_weight = factor * self.fading_weight + 1
            self.fading_correct = factor * self.fading_correct + ((probability >= 0.5) == event.y)
            self.fading_series.append({"step": event.seq, "value": self.fading_correct / self.fading_weight})
            comparison = trace.get("comparison_id")
            if comparison is not None:
                point = self.comparisons[comparison]
                old = trace["previous_probability"] >= 0.5
                new = probability >= 0.5
                point["evaluated"] += 1
                point["old_errors"] += old != event.y
                point["new_errors"] += new != event.y
                costs = self.settings.get("costs") or {"false_positive": 1, "false_negative": 1}
                weight = costs["false_negative"] if event.y else costs["false_positive"]
                point["old_weighted_loss"] += weight * (old != event.y)
                point["new_weighted_loss"] += weight * (new != event.y)
        if self.continual is not None:
            self.continual.after_label(event.seq, event.x, event.y, model)
        self.instrumentation_seconds += time.perf_counter() - started

    def sample_memory(self):
        if self.memory_sampler is not None:
            value = self.memory_sampler()
            if value is not None:
                if self.start_memory_segment:
                    self.memory_breaks.add(len(self.memory_samples))
                    self.start_memory_segment = False
                self.memory_samples.append((time.perf_counter(), value))

    def report(self):
        timings = [row["prediction_seconds"] for row in self.predictions]
        count = len(timings)
        intervals = [
            (a, b)
            for index, (a, b) in enumerate(zip(self.memory_samples, self.memory_samples[1:]))
            if index + 1 not in self.memory_breaks
        ]
        memory_seconds = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in intervals)
        return {
            "prediction_seconds": sum(timings) if count else None,
            "prediction_latency_mean_seconds": sum(timings) / count if count else None,
            "prediction_latency_p50_seconds": quantile(timings, 0.5),
            "prediction_latency_p95_seconds": quantile(timings, 0.95),
            "prediction_latency_p99_seconds": quantile(timings, 0.99),
            "prediction_throughput_per_second": count / sum(timings) if sum(timings) else None,
            "fading_accuracy": self.fading_correct / self.fading_weight if self.fading_weight else None,
            "fading_series": self.fading_series,
            "warmup_train_seconds": self.warmup_train_seconds,
            "warmup_train_samples": self.warmup_train_samples,
            "instrumentation_seconds": self.instrumentation_seconds,
            "monitor_seconds": self.monitor_seconds,
            "sampled_rss_byte_seconds": memory_seconds if intervals else None,
            "memory_samples": len(self.memory_samples),
            "memory_sampling": "trapezoidal current RSS integral by attempt; checkpoint pauses excluded",
            "comparisons": self.comparisons,
        }


def relative_cost_error(cost, oracle):
    if not math.isfinite(cost) or not math.isfinite(oracle) or cost < 0 or oracle < 0:
        raise ValueError("costs must be finite and nonnegative")
    return 100 * (cost - oracle) / oracle if oracle else None


def first_target(curve, target):
    return next(
        (point["step"] for point in curve if point["value"] is not None and point["value"] >= target), None
    )


def learning_speed_ratio(method, baseline, baseline_target, ratio=1.0):
    before = first_target(baseline, baseline_target)
    after = first_target(method, ratio * baseline_target)
    return before / after if before is not None and after is not None and after > 0 else None
