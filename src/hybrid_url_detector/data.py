from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from hybrid_url_detector.config import DEFAULT_LABEL_COL, DEFAULT_URL_COL, RANDOM_SEED


@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    val_csv: Path
    test_csv: Path


def load_url_csv(path: Path, url_col: str = DEFAULT_URL_COL, label_col: str = DEFAULT_LABEL_COL) -> pd.DataFrame:
    df = pd.read_csv(path)
    if url_col not in df.columns:
        raise ValueError(f"Missing url column '{url_col}'. Found: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'. Found: {list(df.columns)}")

    df = df[[url_col, label_col]].copy()
    df[url_col] = df[url_col].astype(str).str.strip()
    df = df[df[url_col].str.len() > 0]

    return df.reset_index(drop=True)


def apply_label_map(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    label_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Converts labels into integer class ids.

    - If label_map is provided, values are mapped via `str(value)` lookup.
    - Otherwise, attempts numeric coercion.
    """
    out = df.copy()
    if label_map is not None:
        mapped = out[label_col].astype(str).map(label_map)
        if mapped.isna().any():
            unknown = sorted(set(out.loc[mapped.isna(), label_col].astype(str).unique().tolist()))
            raise ValueError(f"Unknown label values in '{label_col}': {unknown}. Provide a label_map.")
        out[label_col] = mapped.astype(int)
        return out.reset_index(drop=True)

    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    out = out.dropna(subset=[label_col])
    out[label_col] = out[label_col].astype(int)
    return out.reset_index(drop=True)


def enforce_binary_labels(df: pd.DataFrame, label_col: str = DEFAULT_LABEL_COL) -> pd.DataFrame:
    out = df.copy()
    if not set(out[label_col].unique()).issubset({0, 1}):
        raise ValueError(f"Label column '{label_col}' must be binary 0/1. Found: {sorted(out[label_col].unique())}")
    return out.reset_index(drop=True)

def downsample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=RANDOM_SEED).reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df[label_col],
    )
    # val_size is relative to remaining train portion
    val_rel = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_rel,
        random_state=RANDOM_SEED,
        stratify=train_df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> SplitPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return SplitPaths(train_csv=train_csv, val_csv=val_csv, test_csv=test_csv)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
