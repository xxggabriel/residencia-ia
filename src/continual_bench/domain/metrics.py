import math
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """Binary metrics at threshold 0.5, with exact score ranking metrics.

    Undefined precision/recall and related confusion ratios use zero; macro F1
    always averages the two binary classes. AUROC requires both classes, AP at
    least one positive, and balanced accuracy both true classes. Degenerate MCC
    and kappa return None. ECE/MCE compare mean positive probability with positive
    frequency in ten equal-width bins [i/10, (i+1)/10), including 1 in the last.
    These measure positive-class calibration, not confidence calibration.

    Score storage is O(n); the runner's sample cap bounds n at 100,000. Ranking
    calculations sort once in O(n log n) when values() is requested. No score is
    rounded, clipped, binned, or subsampled for AUROC/AP.
    """

    n: int = 0
    correct: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    brier: float = 0.0
    loss: float = 0.0
    _scores: list[tuple[float, bool]] = field(default_factory=list, repr=False)

    def add(self, y: bool, p: float) -> None:
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("model probability must be finite and in [0,1]")
        pred = p >= 0.5
        self.n += 1
        self.correct += int(pred == y)
        self.tp += int(pred and y)
        self.fp += int(pred and not y)
        self.fn += int(not pred and y)
        self.brier += (int(y) - p) ** 2
        q = min(max(p, 1e-15), 1 - 1e-15)
        self.loss -= math.log(q if y else 1 - q)
        self._scores.append((p, bool(y)))

    def _ranking(self, positives: int, negatives: int) -> tuple[float | None, float | None]:
        """Pairwise AUROC and non-interpolated AP, grouping tied scores."""
        if not positives:
            return None, None
        ranked = sorted(self._scores)
        concordant = 0.0
        ap = 0.0
        positives_below = negatives_below = start = 0
        while start < len(ranked):
            end = start
            group_positive = 0
            score = ranked[start][0]
            while end < len(ranked) and ranked[end][0] == score:
                group_positive += int(ranked[end][1])
                end += 1
            group_negative = end - start - group_positive
            concordant += group_positive * (negatives_below + 0.5 * group_negative)
            # Threshold at this score includes this entire group and all above.
            precision = (positives - positives_below) / (self.n - start)
            ap += group_positive * precision / positives
            positives_below += group_positive
            negatives_below += group_negative
            start = end
        return concordant / (positives * negatives) if negatives else None, ap

    def _calibration(self) -> tuple[float, float]:
        counts = [0] * 10
        probabilities = [0.0] * 10
        positives = [0] * 10
        for p, y in self._scores:
            index = min(int(p * 10), 9)
            counts[index] += 1
            probabilities[index] += p
            positives[index] += int(y)
        ece = mce = 0.0
        for count, probability, positive in zip(counts, probabilities, positives):
            if count:
                gap = abs(probability / count - positive / count)
                ece += count * gap / self.n
                mce = max(mce, gap)
        return ece, mce

    def values(self) -> dict:
        denom = 2 * self.tp + self.fp + self.fn
        tn = self.correct - self.tp
        positives = self.tp + self.fn
        negatives = tn + self.fp
        predicted_positives = self.tp + self.fp
        predicted_negatives = tn + self.fn
        precision = self.tp / predicted_positives if predicted_positives else 0.0
        recall = self.tp / positives if positives else 0.0
        specificity = tn / negatives if negatives else 0.0
        f1 = 2 * self.tp / denom if denom else 0.0
        negative_denom = 2 * tn + self.fp + self.fn
        negative_f1 = 2 * tn / negative_denom if negative_denom else 0.0
        mcc_denom = predicted_positives * predicted_negatives * positives * negatives
        # Integer kappa algebra avoids subtracting nearly equal probabilities.
        agreement_expected = predicted_positives * positives + predicted_negatives * negatives
        kappa_denom = self.n * self.n - agreement_expected
        roc_auc, average_precision = self._ranking(positives, negatives)
        ece, mce = self._calibration()
        return {
            "n": self.n,
            "accuracy": self.correct / self.n if self.n else 0.0,
            "f1": f1,
            "brier": self.brier / self.n if self.n else 0.0,
            "log_loss": self.loss / self.n if self.n else 0.0,
            "tp": self.tp,
            "fp": self.fp,
            "tn": tn,
            "fn": self.fn,
            "errors": self.fp + self.fn,
            "error_rate": (self.fp + self.fn) / self.n if self.n else 0.0,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "false_positive_rate": self.fp / negatives if negatives else 0.0,
            "false_negative_rate": self.fn / positives if positives else 0.0,
            "negative_predictive_value": tn / predicted_negatives if predicted_negatives else 0.0,
            "balanced_accuracy": (recall + specificity) / 2 if positives and negatives else None,
            "f1_macro": (f1 + negative_f1) / 2,
            "f1_weighted": (positives * f1 + negatives * negative_f1) / self.n if self.n else 0.0,
            "mcc": (self.tp * tn - self.fp * self.fn) / math.sqrt(mcc_denom) if mcc_denom else None,
            "cohen_kappa": (self.n * self.correct - agreement_expected) / kappa_denom
            if kappa_denom else None,
            "positive_rate": positives / self.n if self.n else 0.0,
            "predicted_positive_rate": predicted_positives / self.n if self.n else 0.0,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "ece": ece,
            "mce": mce,
        }
