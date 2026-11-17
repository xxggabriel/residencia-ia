"""Streaming classification/regression scores for the iterative runner (single pass, bounded memory)."""

import math
import time
from collections import Counter

from continual_bench.core.tasks import UNTRAINED

from .tracking_observations import binary_ranking, classification_quality

LOSS_FLOOR = 1e-9


def argmax_label(probabilities):
    return max(probabilities, key=probabilities.get) if probabilities else UNTRAINED


def average_precision(labels, scores):
    """Non-interpolated AP over tied-score groups; None without positives."""
    positives = sum(labels)
    if not positives:
        return None
    order = sorted(range(len(labels)), key=lambda i: -scores[i])
    seen = hits = 0
    total, index = 0.0, 0
    while index < len(order):
        stop = index
        while stop < len(order) and scores[order[stop]] == scores[order[index]]:
            stop += 1
        group_hits = sum(labels[order[k]] for k in range(index, stop))
        seen += stop - index
        hits += group_hits
        total += group_hits * hits / seen
        index = stop
    return total / positives


class ClassificationScorer:
    def __init__(self, positive="1", *, observed_positive_class=None):
        self.positive = positive
        self.n = self.correct = 0
        self.loss = 0.0
        self.tp, self.fp, self.fn = {}, {}, {}
        self.truth, self.scores, self.binary = [], [], True
        self.labels = set()
        self.confusion = Counter()
        self.observed_positive_class = observed_positive_class
        self.ranked_labels, self.ranked_scores = [], []
        self.tracking_seconds = 0.0

    def update(self, y, probabilities):
        prediction = argmax_label(probabilities)
        self.n += 1
        self.labels.add(y)
        self.loss -= math.log(min(1.0, max(float(probabilities.get(y, 0.0)), LOSS_FLOOR)))
        if prediction == y:
            self.correct += 1
            self.tp[y] = self.tp.get(y, 0) + 1
        else:
            self.fn[y] = self.fn.get(y, 0) + 1
            self.fp[prediction] = self.fp.get(prediction, 0) + 1
        if self.binary:
            if len(self.labels) > 2 or (self.labels - {"0", "1"}):
                self.binary, self.truth, self.scores = False, [], []
            else:
                self.truth.append(1 if y == self.positive else 0)
                self.scores.append(float(probabilities.get(self.positive, 0.0)))
        tick = time.perf_counter()
        self.confusion[y, prediction] += 1
        if self.observed_positive_class is not None and probabilities:
            score = float(probabilities.get(self.observed_positive_class, 0.0))
            if math.isfinite(score) and 0 <= score <= 1:
                self.ranked_labels.append(y == self.observed_positive_class)
                self.ranked_scores.append(score)
        self.tracking_seconds += time.perf_counter() - tick

    def quality(self, classes, *, curves=False):
        labels = list(dict.fromkeys((*classes, UNTRAINED, *sorted({p for _, p in self.confusion}))))
        ranking = binary_ranking(self.ranked_labels, self.ranked_scores, curves=curves)
        result = classification_quality(
            [[self.confusion[actual, predicted] for predicted in labels] for actual in labels],
            labels, target_classes=classes, positive_class=self.observed_positive_class, **ranking,
        )
        if self.observed_positive_class is not None:
            result.update(ranking_evaluated_examples=len(self.ranked_labels),
                          ranking_positive_examples=sum(self.ranked_labels),
                          ranking_scope="examples_with_observed_positive_probability")
        return result

    def per_class(self):
        rows = {}
        for label in sorted(self.labels):
            tp, fp, fn = self.tp.get(label, 0), self.fp.get(label, 0), self.fn.get(label, 0)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        return rows

    def result(self):
        if not self.n:
            return {"n": 0}
        rows = self.per_class().values()
        count = len(self.labels)
        result = {"n": self.n, "accuracy": self.correct / self.n, "error_rate": 1 - self.correct / self.n,
                  "log_loss": self.loss / self.n,
                  "balanced_accuracy": sum(r["recall"] for r in rows) / count,
                  "f1_macro": sum(r["f1"] for r in rows) / count,
                  "precision_macro": sum(r["precision"] for r in rows) / count,
                  "recall_macro": sum(r["recall"] for r in rows) / count}
        if self.binary and len(self.labels) == 2:
            pr_auc = average_precision(self.truth, self.scores)
            if pr_auc is not None:
                result["pr_auc"] = pr_auc
        return result


class RegressionScorer:
    def __init__(self, *, retain_residuals=False):
        self.n, self.abs, self.sq, self.mean_y, self.m2 = 0, 0.0, 0.0, 0.0, 0.0
        self.residuals = {"prediction": [], "error": []} if retain_residuals else None
        self.tracking_seconds = 0.0

    def update(self, y, prediction):
        value = float(y)
        prediction = 0.0 if prediction is None else float(prediction)
        error = value - prediction
        self.n += 1
        self.abs += abs(error)
        self.sq += error * error
        # Welford's second central moment avoids cancellation for a large target offset.
        delta = value - self.mean_y
        self.mean_y += delta / self.n
        self.m2 += delta * (value - self.mean_y)
        if self.residuals is not None:
            tick = time.perf_counter()
            self.residuals["prediction"].append(prediction)
            self.residuals["error"].append(error)
            self.tracking_seconds += time.perf_counter() - tick

    def per_class(self):
        return {}

    def result(self):
        if not self.n:
            return {"n": 0}
        return {"n": self.n, "mae": self.abs / self.n, "rmse": math.sqrt(self.sq / self.n),
                "r2": 1 - self.sq / self.m2 if self.m2 > 0 else 0.0}

    def quality(self, target_unit):
        result = self.result()
        return {"evaluated_examples": self.n, "mae": result.get("mae"), "rmse": result.get("rmse"),
                "target_unit": target_unit, "residuals": self.residuals}


def score_rows(model, task_is_regression, rows, scorer=None):
    """Score rows without learning; returns the scorer."""
    scorer = scorer or (RegressionScorer() if task_is_regression else ClassificationScorer())
    for x, y in rows:
        if task_is_regression:
            scorer.update(y, model.predict_one(x))
        else:
            scorer.update(y, model.predict_proba_one(x) or {})
    return scorer
