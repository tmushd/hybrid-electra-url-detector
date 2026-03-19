from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import Paths
from hybrid_url_detector.data import load_split
from hybrid_url_detector.electra_branch import load_electra, predict_electra_proba
from hybrid_url_detector.fusion import load_fusion_model, predict_fusion_proba
from hybrid_url_detector.metadata_branch import load_metadata_model, predict_metadata_proba
from hybrid_url_detector.metrics import compute_binary_metrics
from hybrid_url_detector.uncertainty import is_uncertain, mc_dropout_predict_proba


def _fusion_features(p_electra_mean: np.ndarray, p_electra_std: np.ndarray, p_meta: np.ndarray) -> np.ndarray:
    return np.stack([p_electra_mean, p_electra_std, p_meta], axis=1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate metadata, ELECTRA, and fusion models on test split.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--electra-dir", type=str, default="models/electra")
    parser.add_argument("--metadata-model", type=str, default="models/metadata.joblib")
    parser.add_argument("--fusion-model", type=str, default="models/fusion.joblib")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ELECTRA inference (MC-dropout)")
    parser.add_argument("--std-threshold", type=float, default=0.15)
    parser.add_argument("--progress", action="store_true", help="Print MC-dropout progress")
    parser.add_argument("--out-json", type=str, default="results/metrics.json")
    parser.add_argument("--preds-csv", type=str, default="results/predictions_test.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    test_df = load_split((repo_root / args.processed_dir / "test.csv").resolve())
    y = test_df["label"].astype(int).to_numpy()
    urls = test_df["url"].astype(str).tolist()

    # Load models
    meta = load_metadata_model((repo_root / args.metadata_model).resolve())
    electra = load_electra((repo_root / args.electra_dir).resolve())
    print(f"Using ELECTRA device: {electra.device}")

    p_meta = predict_metadata_proba(meta, urls)
    meta_metrics = compute_binary_metrics(y, p_meta)

    # Deterministic ELECTRA
    p_electra = predict_electra_proba(electra, urls)
    electra_metrics = compute_binary_metrics(y, p_electra)

    # MC-dropout ELECTRA
    mc = mc_dropout_predict_proba(
        electra, urls, n_passes=args.mc_passes, batch_size=args.batch_size, progress=args.progress
    )
    uncertain_mask = is_uncertain(mc.ci_low, mc.ci_high, mc.p_std, std_threshold=args.std_threshold)
    electra_mc_metrics = compute_binary_metrics(y, mc.p_mean)

    # Fusion
    fusion_path = (repo_root / args.fusion_model).resolve()
    fusion_metrics = None
    p_fusion = None
    if fusion_path.exists():
        fusion = load_fusion_model(fusion_path)
        x = _fusion_features(mc.p_mean, mc.p_std, p_meta)
        p_fusion = predict_fusion_proba(fusion, x)
        fusion_metrics = compute_binary_metrics(y, p_fusion)

    metrics_out = {
        "metadata": meta_metrics.as_dict(),
        "electra_deterministic": electra_metrics.as_dict(),
        "electra_mc_dropout": electra_mc_metrics.as_dict(),
        "uncertain_fraction": float(uncertain_mask.mean()),
    }
    if fusion_metrics is not None:
        metrics_out["fusion"] = fusion_metrics.as_dict()

    out_json = (repo_root / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics_out, indent=2))
    print(f"Wrote: {out_json}")
    print(json.dumps(metrics_out, indent=2))

    preds = test_df.copy()
    preds["p_meta"] = p_meta
    preds["p_electra"] = p_electra
    preds["p_electra_mean"] = mc.p_mean
    preds["p_electra_std"] = mc.p_std
    preds["ci_low"] = mc.ci_low
    preds["ci_high"] = mc.ci_high
    preds["uncertain"] = uncertain_mask.astype(int)
    if p_fusion is not None:
        preds["p_fusion"] = p_fusion

    out_csv = (repo_root / args.preds_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
