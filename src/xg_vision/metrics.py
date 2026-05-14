from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    true = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob).astype(float)
    pred = (prob >= threshold).astype(int)

    if np.unique(true).size < 2:
        auc = float("nan")
        avg_precision = float("nan")
        balanced_accuracy = float("nan")
    else:
        auc = float(roc_auc_score(true, prob))
        avg_precision = float(average_precision_score(true, prob))
        balanced_accuracy = float(balanced_accuracy_score(true, pred))

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=labels).ravel()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
        "auc_roc": auc,
        "average_precision": avg_precision,
        "brier": float(brier_score_loss(true, prob)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_rate": float(true.mean()) if len(true) else 0.0,
        "predicted_positive_rate": float(pred.mean()) if len(pred) else 0.0,
    }


def best_threshold(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    metric: str = "f1",
) -> tuple[float, dict[str, Any]]:
    """Choose a classification threshold on validation predictions."""
    true = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob).astype(float)
    if len(prob) == 0:
        return 0.5, binary_metrics(true, prob, threshold=0.5)

    candidate_thresholds = np.unique(np.concatenate([prob, np.linspace(0.05, 0.95, 91), np.array([0.5])]))
    best_value = float("-inf")
    best_t = 0.5
    best_metrics = binary_metrics(true, prob, threshold=best_t)
    metric = metric.lower()

    for threshold in candidate_thresholds:
        metrics = binary_metrics(true, prob, threshold=float(threshold))
        if metric == "youden":
            value = metrics["recall"] + (metrics["tn"] / max(metrics["tn"] + metrics["fp"], 1)) - 1.0
        else:
            value = float(metrics.get(metric, float("-inf")))
        if np.isnan(value):
            continue
        if value > best_value:
            best_value = value
            best_t = float(threshold)
            best_metrics = metrics
    return best_t, best_metrics


def metric_for_selection(metrics: dict[str, Any], metric_name: str = "auc_roc") -> float:
    value = metrics.get(metric_name)
    if value is not None and not np.isnan(float(value)):
        return float(value)
    auc = metrics.get("auc_roc")
    if auc is not None and not np.isnan(float(auc)):
        return float(auc)
    return float(metrics.get("f1", 0.0))
