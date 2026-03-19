from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class MetricResult:
    roc_auc: float
    accuracy: float
    f1: float
    precision: float
    recall: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }


def compute_binary_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> MetricResult:
    y_pred = (y_proba >= threshold).astype(int)
    roc_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    return MetricResult(
        roc_auc=roc_auc,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
    )
