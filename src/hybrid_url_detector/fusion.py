from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from hybrid_url_detector.metrics import MetricResult, compute_binary_metrics


@dataclass(frozen=True)
class FusionTrainResult:
    model: LogisticRegression
    val_metrics: MetricResult


def train_fusion_logreg(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> FusionTrainResult:
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    val_proba = model.predict_proba(x_val)[:, 1]
    metrics = compute_binary_metrics(y_val, val_proba)
    return FusionTrainResult(model=model, val_metrics=metrics)


def save_fusion_model(model: LogisticRegression, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def load_fusion_model(path: Path) -> LogisticRegression:
    return joblib.load(path)


def predict_fusion_proba(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]
