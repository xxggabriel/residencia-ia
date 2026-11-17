"""Quality and step statistics shared between trainers and evaluators."""

from itertools import chain

import numpy as np

QUALITY = (
    "evaluated_examples", "macro_f1", "accuracy", "positive_precision", "positive_recall",
    "average_precision", "recall_at_fpr_0_01", "confusion_matrix", "classes", "class_support",
    "class_recall", "positive_class", "decision_threshold", "ranking_method", "pr_curve", "roc_curve",
    "mae", "rmse", "target_unit", "residuals",
)


def classification_quality(
    matrix, classes, *, positive_class=None, decision_threshold=None, average_precision=None,
    recall_at_fpr=None, ranking_method=None, pr_curve=None, roc_curve=None, target_classes=None,
):
    """Summarize a confusion matrix with rows for actual and columns for predicted."""
    counts = np.asarray(matrix, dtype=np.int64)
    result = classification_count_quality(
        counts.sum(axis=1), counts.sum(axis=0), counts.diagonal(), classes,
        positive_class=positive_class, decision_threshold=decision_threshold, target_classes=target_classes,
    )
    result.update(confusion_matrix=counts.tolist(), average_precision=average_precision,
                  recall_at_fpr_0_01=recall_at_fpr, ranking_method=ranking_method,
                  pr_curve=pr_curve, roc_curve=roc_curve)
    return result


def classification_count_quality(
    support, predicted, correct, classes, *,
    positive_class=None, decision_threshold=None, target_classes=None,
):
    """Summarize aligned class counts without constructing a dense confusion matrix."""
    classes = list(classes)
    support, predicted, correct = (
        np.asarray(values, dtype=np.int64) for values in (support, predicted, correct)
    )
    total = int(support.sum())
    denominator = support + predicted
    f1 = np.divide(2 * correct, denominator, out=np.zeros(len(classes)), where=denominator > 0)
    recall = np.divide(correct, support, out=np.zeros(len(classes)), where=support > 0)
    universe = list(classes if target_classes is None else target_classes)
    class_index = {label: index for index, label in enumerate(classes)}
    indices = [class_index[label] for label in universe]
    positive = class_index.get(positive_class) if positive_class is not None else None
    return {
        "evaluated_examples": total,
        "macro_f1": float(f1[indices].mean()) if total and indices else None,
        "accuracy": float(correct.sum() / total) if total else None,
        "positive_precision": (
            float(correct[positive] / predicted[positive])
            if positive is not None and predicted[positive] else None
        ),
        "positive_recall": float(recall[positive]) if positive is not None and support[positive] else None,
        "classes": list(classes),
        "target_classes": universe,
        "class_support": support.tolist(),
        "class_recall": [float(value) if size else None for value, size in zip(recall, support, strict=True)],
        "positive_class": positive_class,
        "decision_threshold": decision_threshold,
    }


def binary_ranking(labels, scores, *, curves=True, low_fpr=False):
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )

    binary = np.asarray(list(labels), dtype=bool)
    values = np.asarray(list(scores), dtype=float)
    if values.shape != binary.shape:
        raise ValueError("ranking metrics require one score per label")
    if not binary.any() or not np.isfinite(values).all():
        return {}
    result = {
        "average_precision": float(average_precision_score(binary, values)),
        "ranking_method": "exact_ties_grouped",
    }
    if curves:
        precision, recall, _ = precision_recall_curve(binary, values)
        result["pr_curve"] = {"recall": recall.tolist(), "precision": precision.tolist()}
    if (curves or low_fpr) and not binary.all():
        fpr, tpr, _ = roc_curve(binary, values, drop_intermediate=False)
        if low_fpr:
            result["recall_at_fpr"] = float(max(tpr[fpr <= 0.01]))
        if curves:
            result["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return result


def batch_quality(
    y_true, predictions, *, classes=None, positive_class=None, scores=None, decision_threshold=None,
    low_fpr=False, target_unit=None, curves=True, confusion_classes=(),
):
    """Exact reference for retained, aligned predictions in a bounded evaluation scope."""
    from sklearn.metrics import confusion_matrix, mean_absolute_error, root_mean_squared_error

    actual, predicted = list(y_true), list(predictions)
    if len(actual) != len(predicted):
        raise ValueError("quality requires one prediction per label")
    if classes is None:
        result = {"evaluated_examples": len(actual), "mae": None, "rmse": None, "target_unit": target_unit}
        if actual:
            result.update(mae=float(mean_absolute_error(actual, predicted)),
                          rmse=float(root_mean_squared_error(actual, predicted)))
            if curves:
                result["residuals"] = {
                    "prediction": [float(value) for value in predicted],
                    "error": [float(a - p) for a, p in zip(actual, predicted, strict=True)],
                }
        return result

    target_classes = list(classes)
    labels = list(dict.fromkeys(chain(target_classes, confusion_classes, actual, predicted)))
    if not labels:
        return {"evaluated_examples": 0}
    if not actual:
        empty = np.zeros((len(labels), len(labels)), dtype=np.int64)
        return classification_quality(empty, labels, positive_class=positive_class,
                                      decision_threshold=decision_threshold, target_classes=target_classes)
    # Encoding keeps distinct labels such as integer 1 and string "1" distinct.
    encoding = {label: index for index, label in enumerate(labels)}
    matrix = confusion_matrix(
        [encoding[label] for label in actual], [encoding[label] for label in predicted],
        labels=range(len(labels)),
    )
    ranking = {}
    if positive_class is not None and scores is not None and actual:
        ranking = binary_ranking((label == positive_class for label in actual), scores, curves=curves,
                                 low_fpr=low_fpr)
    return classification_quality(
        matrix, labels, positive_class=positive_class, decision_threshold=decision_threshold,
        target_classes=target_classes, **ranking,
    )


def with_availability(keys, values, reason, overrides=None):
    result = dict.fromkeys(keys)
    result.update(values or {})
    result["availability"] = {
        key: {"status": "unavailable", "reason": (overrides or {}).get(key, reason)}
        for key, value in result.items()
        if value is None
    }
    return result


def quality_scope(task, values):
    classification = {"macro_f1", "accuracy", "confusion_matrix", "classes", "class_support", "class_recall"}
    binary = {
        "positive_precision", "positive_recall", "average_precision", "recall_at_fpr_0_01",
        "positive_class", "decision_threshold", "ranking_method", "pr_curve", "roc_curve",
    }
    regression = {"mae", "rmse", "target_unit", "residuals"}
    applicable = {"binary": classification | binary, "multiclass": classification, "regression": regression}
    reasons = {key: "not_applicable_to_task" for key in classification | binary | regression
               if task in applicable and key not in applicable[task]}
    reasons.update({
        key: reason for key, reason in {
            "positive_precision": "no_positive_predictions_or_no_confusion_statistics",
            "positive_recall": "no_positive_labels_or_no_confusion_statistics",
            "average_precision": "valid_positive_scores_and_positive_labels_required",
            "recall_at_fpr_0_01": "requires_low_fpr_protocol_scores_and_both_classes",
            "pr_curve": "ranking_curve_not_retained_or_positive_scores_unavailable",
            "roc_curve": "ranking_curve_not_retained_or_scores_for_both_classes_unavailable",
            "residuals": "aligned_numeric_predictions_and_targets_not_retained",
        }.items() if key not in reasons
    })
    return with_availability(QUALITY, values, "labeled_predictions_or_statistics_unavailable", reasons)


def training_step(step, *, examples, cumulative, duration_seconds, native_loss=None, metrics=None):
    """One observed trainer call. Counts and metrics belong to the current local adjustment."""
    point = {
        "step": step,
        "metrics": {
            "training/batch_size": examples,
            "training/examples_processed": examples,
            "training/presentations_cumulative": cumulative,
            "training/duration_seconds": duration_seconds,
            "training/native_loss": native_loss,
            **(metrics or {}),
        },
    }
    if native_loss is None:
        point["availability"] = {
            "training/native_loss": {"status": "unavailable", "reason": "trainer_does_not_emit_native_loss"},
        }
    return point
