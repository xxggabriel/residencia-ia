"""Streaming classification/regression scores for the iterative runner (single pass, bounded memory)."""

import math

from continual_bench.core.tasks import UNTRAINED

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
    def __init__(self, positive="1"):
        self.positive = positive
        self.n = self.correct = 0
        self.loss = 0.0
        self.tp, self.fp, self.fn = {}, {}, {}
        self.truth, self.scores, self.binary = [], [], True
        self.labels = set()

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
    def __init__(self):
        self.n, self.abs, self.sq, self.sum_y, self.sum_y2 = 0, 0.0, 0.0, 0.0, 0.0

    def update(self, y, prediction):
        error = float(y) - (0.0 if prediction is None else float(prediction))
        self.n += 1
        self.abs += abs(error)
        self.sq += error * error
        self.sum_y += float(y)
        self.sum_y2 += float(y) ** 2

    def per_class(self):
        return {}

    def result(self):
        if not self.n:
            return {"n": 0}
        variance = self.sum_y2 / self.n - (self.sum_y / self.n) ** 2
        return {"n": self.n, "mae": self.abs / self.n, "rmse": math.sqrt(self.sq / self.n),
                "r2": 1 - (self.sq / self.n) / variance if variance > 0 else 0.0}


def score_rows(model, task_is_regression, rows, scorer=None):
    """Score rows without learning; returns the scorer."""
    scorer = scorer or (RegressionScorer() if task_is_regression else ClassificationScorer())
    for x, y in rows:
        if task_is_regression:
            scorer.update(y, model.predict_one(x))
        else:
            scorer.update(y, model.predict_proba_one(x) or {})
    return scorer
