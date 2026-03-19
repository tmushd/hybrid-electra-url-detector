from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import Paths
from hybrid_url_detector.data import load_split
from hybrid_url_detector.electra_branch import train_electra_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune ELECTRA on raw URL text.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--model", type=str, default="google/electra-small-discriminator")
    parser.add_argument("--out-dir", type=str, default="models/electra")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--fp16", action="store_true")
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

    out_dir = (repo_root / args.out_dir).resolve()
    result = train_electra_classifier(
        train_df=train_df,
        val_df=val_df,
        out_dir=out_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        max_train_samples=None if args.max_train_samples == 0 else args.max_train_samples,
        fp16=args.fp16,
    )

    print(f"Saved ELECTRA model: {result.model_dir}")
    print("Val metrics:", result.val_metrics.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

