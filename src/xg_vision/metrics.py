from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
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
    else:
        auc = float(roc_auc_score(true, prob))

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=labels).ravel()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
        "auc_roc": auc,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_rate": float(true.mean()) if len(true) else 0.0,
        "predicted_positive_rate": float(pred.mean()) if len(pred) else 0.0,
    }


def metric_for_selection(metrics: dict[str, Any]) -> float:
    auc = metrics.get("auc_roc")
    if auc is not None and not np.isnan(float(auc)):
        return float(auc)
    return float(metrics.get("f1", 0.0))
