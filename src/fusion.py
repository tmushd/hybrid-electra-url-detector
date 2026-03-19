from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import Paths
from hybrid_url_detector.data import load_split
from hybrid_url_detector.electra_branch import load_electra
from hybrid_url_detector.fusion import save_fusion_model, train_fusion_logreg
from hybrid_url_detector.metadata_branch import load_metadata_model, predict_metadata_proba
from hybrid_url_detector.uncertainty import mc_dropout_predict_proba


def _fusion_features(p_electra_mean: np.ndarray, p_electra_std: np.ndarray, p_meta: np.ndarray) -> np.ndarray:
    return np.stack([p_electra_mean, p_electra_std, p_meta], axis=1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fusion model on validation split features.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--electra-dir", type=str, default="models/electra")
    parser.add_argument("--metadata-model", type=str, default="models/metadata.joblib")
    parser.add_argument("--fusion-out", type=str, default="models/fusion.joblib")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ELECTRA inference (MC-dropout)")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--progress", action="store_true", help="Print MC-dropout progress (useful on MPS/CPU)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    train_df = load_split((repo_root / args.processed_dir / "train.csv").resolve())
    val_df = load_split((repo_root / args.processed_dir / "val.csv").resolve())
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(n=args.max_train_samples, random_state=1337).reset_index(drop=True)
    if args.max_val_samples and len(val_df) > args.max_val_samples:
        val_df = val_df.sample(n=args.max_val_samples, random_state=1337).reset_index(drop=True)

    loaded = load_electra((repo_root / args.electra_dir).resolve())
    print(f"Using ELECTRA device: {loaded.device}")
    meta = load_metadata_model((repo_root / args.metadata_model).resolve())

    # ELECTRA uncertainty features for train+val
    train_mc = mc_dropout_predict_proba(
        loaded,
        train_df["url"].astype(str).tolist(),
        n_passes=args.mc_passes,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    val_mc = mc_dropout_predict_proba(
        loaded,
        val_df["url"].astype(str).tolist(),
        n_passes=args.mc_passes,
        batch_size=args.batch_size,
        progress=args.progress,
    )

    p_meta_train = predict_metadata_proba(meta, train_df["url"].astype(str).tolist())
    p_meta_val = predict_metadata_proba(meta, val_df["url"].astype(str).tolist())

    x_train = _fusion_features(train_mc.p_mean, train_mc.p_std, p_meta_train)
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = _fusion_features(val_mc.p_mean, val_mc.p_std, p_meta_val)
    y_val = val_df["label"].astype(int).to_numpy()

    result = train_fusion_logreg(x_train, y_train, x_val, y_val)
    out_path = (repo_root / args.fusion_out).resolve()
    save_fusion_model(result.model, out_path)

    print(f"Saved fusion model: {out_path}")
    print("Val metrics:", result.val_metrics.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
