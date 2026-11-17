"""Vectorized classification rules shared by training and temporal evaluation."""

import numpy as np

from continual_bench.models.observations import classification_quality

SCORE_BINS = 4096


def decide(probabilities, threshold=0.5):
    """Binary: p_1 >= threshold. Multiclass: argmax, ties favor the lower class ID."""
    if probabilities.shape[1] == 2:
        return (probabilities[:, 1] >= threshold).astype(np.int32)
    return probabilities.argmax(axis=1)


def ranking_histograms(scores, positive):
    bins = np.minimum((scores * SCORE_BINS).astype(np.int32), SCORE_BINS - 1)
    return (
        np.bincount(bins[positive], minlength=SCORE_BINS),
        np.bincount(bins[~positive], minlength=SCORE_BINS),
    )


def average_precision(positive, negative):
    tp, fp = positive[::-1].cumsum(), negative[::-1].cumsum()
    if not tp[-1]:
        return None
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
    return float(np.dot(positive[::-1], precision) / tp[-1])


class ClassificationCounts:
    """Confusion counts and uniform score histograms that can be merged across batches."""

    def __init__(self, classes, threshold):
        self.classes = classes
        self.threshold = threshold
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.positive = np.zeros((classes, SCORE_BINS), dtype=np.int64)
        self.negative = np.zeros_like(self.positive)
        self.loss = 0.0

    def add(self, labels, probabilities):
        if (
            probabilities.shape != (len(labels), self.classes)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("invalid classification probabilities")
        predicted = decide(probabilities, self.threshold)
        self.confusion += np.bincount(
            self.classes * labels.astype(np.int32, copy=False) + predicted, minlength=self.classes**2
        ).reshape(self.confusion.shape)
        observed = probabilities[np.arange(len(labels)), labels].astype(np.float64, copy=False)
        self.loss -= float(np.log(observed.clip(1e-15, 1)).sum())
        for i in range(self.classes):
            positive, negative = ranking_histograms(probabilities[:, i], labels == i)
            self.positive[i] += positive
            self.negative[i] += negative
        return observed

    def merge(self, other):
        if self.classes != other.classes:
            raise ValueError("cannot merge metrics computed with different classes")
        if self.threshold != other.threshold:
            raise ValueError("cannot merge metrics computed with different decision thresholds")
        self.confusion += other.confusion
        self.positive += other.positive
        self.negative += other.negative
        self.loss += other.loss

    def quality(self, classes, *, curves=False):
        if self.classes != 2:
            return classification_quality(self.confusion.tolist(), classes)
        positive, negative = self.positive[1], self.negative[1]
        tp, fp = positive[::-1].cumsum(), negative[::-1].cumsum()
        ap, low_fpr = average_precision(positive, negative), None
        roc, pr = None, None
        if tp[-1] and fp[-1]:
            low_fpr = float(tp[fp <= 0.01 * fp[-1]].max(initial=0) / tp[-1])
        if curves and tp[-1]:
            nonempty = (positive + negative)[::-1] > 0
            tp, fp = tp[nonempty], fp[nonempty]
            recall = [0.0, *(tp / tp[-1]).tolist()]
            pr = {"recall": recall, "precision": [1.0, *(tp / (tp + fp)).tolist()]}
            if fp[-1]:
                roc = {"fpr": [0.0, *(fp / fp[-1]).tolist()], "tpr": recall}
        return classification_quality(
            self.confusion.tolist(),
            classes,
            positive_class=classes[1],
            decision_threshold=self.threshold,
            average_precision=ap,
            recall_at_fpr=low_fpr,
            ranking_method="uniform_histogram_4096_bins",
            pr_curve=pr,
            roc_curve=roc,
        )
