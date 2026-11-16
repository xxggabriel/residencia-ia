"""Vectorized classification rules shared by training and temporal evaluation."""

import numpy as np

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
