"""Mergeable classification scores with fixed memory, independent of stream length."""

import numpy as np

from continual_bench.models.classification_metrics import (
    SCORE_BINS as SCORE_BINS,
)
from continual_bench.models.classification_metrics import (
    ClassificationCounts,
    average_precision,
)
from continual_bench.models.classification_metrics import (
    decide as decide,
)
from continual_bench.models.classification_metrics import (
    ranking_histograms as ranking_histograms,
)


def ratio(a, b):
    return float(a / b) if b else None


def decision_threshold(costs):
    """Bayes threshold for calibrated binary scores: predict positive when p >= c_FP / (c_FP + c_FN)."""
    false_positive, false_negative = float(costs["false_positive"]), float(costs["false_negative"])
    total = false_positive + false_negative
    return false_positive / total if total else 0.5


class ClassificationMetrics(ClassificationCounts):
    def __init__(self, classes, threshold=0.5):
        super().__init__(classes, threshold)
        self.calibration = np.zeros((classes, 3, 10))
        self.squared_error = self.absolute_error = 0.0

    def add(self, labels, probabilities):
        observed = super().add(labels, probabilities)
        self.absolute_error += float((1 - observed).sum())
        squared = np.square(probabilities, dtype=np.float64).sum() - 2 * observed.sum() + len(labels)
        self.squared_error += max(0.0, float(squared)) / (2 if self.classes == 2 else 1)
        for i in range(self.classes):
            mask = labels == i
            calibration_bin = np.minimum((probabilities[:, i] * 10).astype(np.int32), 9)
            self.calibration[i, 0] += np.bincount(calibration_bin, minlength=10)
            self.calibration[i, 1] += np.bincount(calibration_bin, weights=probabilities[:, i], minlength=10)
            self.calibration[i, 2] += np.bincount(calibration_bin, weights=mask, minlength=10)

    def merge(self, other):
        super().merge(other)
        self.calibration += other.calibration
        self.squared_error += other.squared_error
        self.absolute_error += other.absolute_error

    def metrics(self):
        total = int(self.confusion.sum())
        actual = self.confusion.sum(axis=1)
        predicted = self.confusion.sum(axis=0)
        correct = self.confusion.diagonal()
        rows = []
        for i in range(self.classes):
            p, n = self.positive[i], self.negative[i]
            counts, probability, positives = self.calibration[i]
            gaps = np.abs(probability - positives)
            tp = p[::-1].cumsum()
            fp = n[::-1].cumsum()
            recall_curve = tp / actual[i] if actual[i] else np.zeros_like(tp)
            rows.append({
                "support": int(actual[i]),
                "precision": ratio(correct[i], predicted[i]),
                "recall": ratio(correct[i], actual[i]),
                "f1": ratio(2 * correct[i], predicted[i] + actual[i]),
                "pr_auc": average_precision(p, n),
                "roc_auc": ratio(np.dot(p, n.cumsum() - n / 2), actual[i] * n.sum()),
                "ece": ratio(gaps.sum(), total),
                "mce": float(np.max(np.divide(gaps, counts, out=np.zeros(10), where=counts > 0))) if total
                else None,
                "recall_at_fpr_0_01": float(recall_curve[fp <= 0.01 * n.sum()].max(initial=0))
                if actual[i] and n.sum() else None,
            })
        result = {
            "n": total, "accuracy": ratio(correct.sum(), total),
            "errors": total - int(correct.sum()), "error_rate": ratio(total - correct.sum(), total),
            "log_loss": ratio(self.loss, total), "brier": ratio(self.squared_error, total),
            "mae_probability": ratio(self.absolute_error, total),
        }
        for name in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
            values = [row[name] for row in rows if row[name] is not None]
            result["macro_" + name] = ratio(sum(values), len(values))
        result["balanced_accuracy"] = result["macro_recall"]
        expected = sum(int(a) * int(p) for a, p in zip(actual, predicted, strict=True))
        result["cohen_kappa"] = ratio(total * int(correct.sum()) - expected, total**2 - expected)
        result["f1_macro"] = result["macro_f1"]
        result["f1_weighted"] = ratio(sum(r["support"] * (r["f1"] or 0) for r in rows), total)
        if self.classes == 2:
            tn, fp, fn, tp = map(int, self.confusion.ravel())
            result.update(rows[1], tp=tp, tn=tn, fp=fp, fn=fn, false_positive_rate=ratio(fp, fp + tn),
                          false_negative_rate=ratio(fn, fn + tp), specificity=ratio(tn, tn + fp),
                          negative_predictive_value=ratio(tn, tn + fn), positive_rate=ratio(tp + fn, total),
                          predicted_positive_rate=ratio(tp + fp, total),
                          mcc=ratio(tp * tn - fp * fn, ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5))
            result["average_precision"] = result["pr_auc"]
            result["mse_probability"] = result["brier"]
        return result, rows
