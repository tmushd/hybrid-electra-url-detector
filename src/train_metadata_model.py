from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import Paths
from hybrid_url_detector.data import load_split
from hybrid_url_detector.metadata_branch import save_metadata_model, train_random_forest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train metadata (CTI-inspired structured features) branch.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--model-out", type=str, default="models/metadata.joblib")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means None")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    processed = (repo_root / args.processed_dir).resolve()
    train_csv = processed / "train.csv"
    val_csv = processed / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Missing train/val splits. Run src/preprocess.py first.")

    train_df = load_split(train_csv)
    val_df = load_split(val_csv)

    result = train_random_forest(
        train_df,
        val_df,
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth == 0 else args.max_depth,
    )

    model_out = (repo_root / args.model_out).resolve()
    save_metadata_model(result.model, model_out)

    print(f"Saved metadata model: {model_out}")
    print("Val metrics:", result.val_metrics.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

