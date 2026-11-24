"""Unified regression and classification metric implementations.

Every Arcora model family is scored through these functions so metrics are computed
identically everywhere. Metrics that are mathematically undefined for a given input (for
example ROC-AUC when only one class is present) are returned as ``None`` rather than a
misleading number, and are listed by :func:`undefined_metrics`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from arcora.models.modeling_strategy import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_TASK,
    CLASSIFICATION_THRESHOLD,
    PROBABILITY_CLIP_EPSILON,
    REGRESSION_METRICS,
    REGRESSION_TASK,
)

MetricValue = float | None
MetricDict = dict[str, MetricValue]

REGRESSION_METRIC_NAMES: tuple[str, ...] = tuple(m.name for m in REGRESSION_METRICS)
CLASSIFICATION_METRIC_NAMES: tuple[str, ...] = tuple(m.name for m in CLASSIFICATION_METRICS)


def _as_1d_float(values: Any, label: str) -> npt.NDArray[np.float64]:
    arr = np.asarray(values, dtype="float64").reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or infinite values")
    return arr


def _validate_pair(y_true: Any, y_pred: Any) -> tuple[npt.NDArray[np.float64], ...]:
    true = _as_1d_float(y_true, "y_true")
    pred = _as_1d_float(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    return true, pred


def _empty(names: tuple[str, ...]) -> MetricDict:
    return dict.fromkeys(names)


def regression_metrics(y_true: Any, y_pred: Any) -> MetricDict:
    """Compute MAE, RMSE, and R² for point predictions."""
    true, pred = _validate_pair(y_true, y_pred)
    if true.size == 0:
        return _empty(REGRESSION_METRIC_NAMES)
    r2: MetricValue = None
    if true.size >= 2 and float(np.var(true)) > 0.0:
        r2 = float(r2_score(true, pred))
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(true, pred))),
        "r2": r2,
    }


def probabilities_to_labels(
    y_prob: Any, threshold: float = CLASSIFICATION_THRESHOLD
) -> npt.NDArray[np.int64]:
    """Convert positive-class probabilities to 0/1 labels using ``p >= threshold``."""
    prob = _as_1d_float(y_prob, "y_prob")
    return (prob >= threshold).astype("int64")


def classification_metrics(
    y_true: Any,
    y_prob: Any,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> MetricDict:
    """Compute label and probability metrics for a binary classifier.

    ``y_prob`` is the predicted probability of the positive class (home win). Labels are
    derived with ``p >= threshold``.
    """
    true, prob = _validate_pair(y_true, y_prob)
    if true.size == 0:
        return _empty(CLASSIFICATION_METRIC_NAMES)
    if not np.isin(true, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only 0/1 labels")
    if (prob < 0.0).any() or (prob > 1.0).any():
        raise ValueError("y_prob must contain probabilities in [0, 1]")

    labels_true = true.astype("int64")
    labels_pred = probabilities_to_labels(prob, threshold)
    tp = int(((labels_pred == 1) & (labels_true == 1)).sum())
    fp = int(((labels_pred == 1) & (labels_true == 0)).sum())
    fn = int(((labels_pred == 0) & (labels_true == 1)).sum())
    tn = int(((labels_pred == 0) & (labels_true == 0)).sum())
    both_classes = 0 < labels_true.sum() < labels_true.size

    clipped = np.clip(prob, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll = float(log_loss(labels_true, clipped, labels=[0, 1]))
        brier = float(brier_score_loss(labels_true, prob, pos_label=1))

    return {
        "accuracy": float(accuracy_score(labels_true, labels_pred)),
        "balanced_accuracy": (
            float((tp / (tp + fn) + tn / (tn + fp)) / 2.0) if both_classes else None
        ),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        "roc_auc": float(roc_auc_score(labels_true, prob)) if both_classes else None,
        "log_loss": ll,
        "brier": brier,
    }


def compute_metrics(task: str, y_true: Any, y_pred: Any) -> MetricDict:
    """Dispatch to the regression or classification metric suite.

    For classification, ``y_pred`` must be positive-class probabilities.
    """
    if task == REGRESSION_TASK:
        return regression_metrics(y_true, y_pred)
    if task == CLASSIFICATION_TASK:
        return classification_metrics(y_true, y_pred)
    raise ValueError(f"Unknown task '{task}'")


def undefined_metrics(metrics: Mapping[str, MetricValue]) -> list[str]:
    """Return the names of metrics that were undefined (``None``) for an evaluation."""
    return [name for name, value in metrics.items() if value is None]
