from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from hybrid_url_detector.metrics import compute_binary_metrics, MetricResult
from hybrid_url_detector.url_features import features_and_labels, extract_features


@dataclass(frozen=True)
class MetadataModelResult:
    model: RandomForestClassifier
    val_metrics: MetricResult


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    url_col: str = "url",
    label_col: str = "label",
    n_estimators: int = 400,
    max_depth: Optional[int] = None,
) -> MetadataModelResult:
    x_train, y_train = features_and_labels(train_df, url_col=url_col, label_col=label_col)
    x_val, y_val = features_and_labels(val_df, url_col=url_col, label_col=label_col)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=1337,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    val_proba = model.predict_proba(x_val)[:, 1]
    val_metrics = compute_binary_metrics(y_val, val_proba)
    return MetadataModelResult(model=model, val_metrics=val_metrics)


def save_metadata_model(model: RandomForestClassifier, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def load_metadata_model(path: Path) -> RandomForestClassifier:
    return joblib.load(path)


def predict_metadata_proba(model: RandomForestClassifier, urls: list[str]) -> np.ndarray:
    x = extract_features(urls)
    return model.predict_proba(x)[:, 1]
