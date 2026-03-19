from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import DEFAULT_LABEL_COL, DEFAULT_URL_COL, Paths
from hybrid_url_detector.data import (
    apply_label_map,
    downsample_rows,
    enforce_binary_labels,
    load_url_csv,
    save_splits,
    split_train_val_test,
)


def _parse_label_map(text: str) -> dict[str, int]:
    """
    Parses: "benign=0,phishing=1,malware=1,defacement=1"
    """
    mapping: dict[str, int] = {}
    if not text.strip():
        return mapping
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --label-map entry: '{part}'. Expected key=value.")
        k, v = part.split("=", 1)
        mapping[k.strip()] = int(v.strip())
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess and split URL dataset into train/val/test CSVs.")
    parser.add_argument("--input", type=str, default="data/raw/urls.csv", help="Input CSV with columns url,label")
    parser.add_argument("--url-col", type=str, default=DEFAULT_URL_COL)
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument(
        "--label-map",
        type=str,
        default="",
        help="Optional mapping for string labels, e.g. benign=0,phishing=1,malware=1,defacement=1",
    )
    parser.add_argument(
        "--kaggle-malicious-phish",
        action="store_true",
        help="Convenience: sets url_col=url label_col=type label_map benign=0,others=1",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional random downsample (0 means all)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    in_path = (repo_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)

    url_col = args.url_col
    label_col = args.label_col
    label_map = _parse_label_map(args.label_map)
    if args.kaggle_malicious_phish:
        url_col = "url"
        label_col = "type"
        label_map = {"benign": 0, "defacement": 1, "phishing": 1, "malware": 1}

    df = load_url_csv(in_path, url_col=url_col, label_col=label_col)
    df = apply_label_map(df, label_col=label_col, label_map=label_map or None)
    df = enforce_binary_labels(df, label_col=label_col)
    df = downsample_rows(df, args.max_rows)

    train_df, val_df, test_df = split_train_val_test(
        df, label_col=label_col, test_size=args.test_size, val_size=args.val_size
    )
    # Write with canonical column names expected by downstream scripts.
    train_df = train_df.rename(columns={url_col: "url", label_col: "label"})
    val_df = val_df.rename(columns={url_col: "url", label_col: "label"})
    test_df = test_df.rename(columns={url_col: "url", label_col: "label"})

    split_paths = save_splits(train_df, val_df, test_df, paths.data_processed)

    print(f"Wrote: {split_paths.train_csv}")
    print(f"Wrote: {split_paths.val_csv}")
    print(f"Wrote: {split_paths.test_csv}")
    print(f"Counts: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
