"""Метрики бінарної класифікації без додаткових залежностей."""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(y_true, dtype=int).reshape(-1)
    pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if true.size == 0 or true.shape != pred.shape:
        raise ValueError("y_true та y_pred мають бути непорожніми масивами однакової довжини.")
    if np.any((true != 0) & (true != 1)) or np.any((pred != 0) & (pred != 1)):
        raise ValueError("Для бінарної класифікації мітки мають бути 0 або 1.")
    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / true.size,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
