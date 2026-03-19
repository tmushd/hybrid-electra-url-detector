from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import Paths
from hybrid_url_detector.data import load_split
from hybrid_url_detector.url_features import extract_features


def featurize_split(split_csv: Path, out_csv: Path, url_col: str = "url") -> None:
    df = load_split(split_csv)
    x = extract_features(df[url_col].astype(str).tolist())
    out = pd.concat([df.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute CTI-inspired structured features for each split.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    processed_dir = (repo_root / args.processed_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    for split in ["train", "val", "test"]:
        in_csv = processed_dir / f"{split}.csv"
        out_csv = out_dir / f"{split}_features.csv"
        if not in_csv.exists():
            raise FileNotFoundError(f"Missing split file: {in_csv}. Run preprocess.py first.")
        featurize_split(in_csv, out_csv)
        print(f"Wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

