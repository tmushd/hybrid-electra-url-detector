from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import Paths
from hybrid_url_detector.data import load_split
from hybrid_url_detector.electra_branch import load_electra
from hybrid_url_detector.uncertainty import is_uncertain, mc_dropout_predict_proba


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ELECTRA MC-dropout uncertainty for a split.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--electra-dir", type=str, default="models/electra")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ELECTRA inference (MC-dropout)")
    parser.add_argument("--std-threshold", type=float, default=0.15)
    parser.add_argument("--out", type=str, default="results/electra_uncertainty.csv")
    parser.add_argument("--progress", action="store_true", help="Print MC-dropout progress")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    split_csv = (repo_root / args.processed_dir / f"{args.split}.csv").resolve()
    df = load_split(split_csv)

    loaded = load_electra((repo_root / args.electra_dir).resolve())
    res = mc_dropout_predict_proba(
        loaded=loaded,
        urls=df["url"].astype(str).tolist(),
        n_passes=args.mc_passes,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    uncertain = is_uncertain(res.ci_low, res.ci_high, res.p_std, std_threshold=args.std_threshold)

    out_df = df.copy()
    out_df["p_mean"] = res.p_mean
    out_df["p_std"] = res.p_std
    out_df["ci_low"] = res.ci_low
    out_df["ci_high"] = res.ci_high
    out_df["uncertain"] = uncertain.astype(int)

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(f"Uncertain fraction: {uncertain.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
