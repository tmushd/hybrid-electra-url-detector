from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_url_detector.config import Paths, RANDOM_SEED
from hybrid_url_detector.data import (
    apply_label_map,
    downsample_rows,
    enforce_binary_labels,
    load_url_csv,
    save_splits,
    split_train_val_test,
)
from hybrid_url_detector.data import load_split
from hybrid_url_detector.electra_branch import load_electra, train_electra_classifier
from hybrid_url_detector.fusion import save_fusion_model, train_fusion_logreg
from hybrid_url_detector.metadata_branch import save_metadata_model, train_random_forest, predict_metadata_proba, load_metadata_model
from hybrid_url_detector.uncertainty import mc_dropout_predict_proba
from hybrid_url_detector.metrics import compute_binary_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command reproduction of the hybrid URL detector pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Input dataset (CSV or .zip containing CSV).")
    parser.add_argument("--kaggle-malicious-phish", action="store_true", help="Treat input as malicious_phish url/type.")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)

    parser.add_argument("--electra-model", type=str, default="google/electra-small-discriminator")
    parser.add_argument("--electra-epochs", type=int, default=1)
    parser.add_argument("--electra-max-train-samples", type=int, default=20000)
    parser.add_argument("--electra-batch-size", type=int, default=16)

    parser.add_argument("--mc-passes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fusion-train-cap", type=int, default=8000)
    parser.add_argument("--fusion-val-cap", type=int, default=3000)
    parser.add_argument(
        "--no-print-json",
        action="store_false",
        dest="print_json",
        help="Disable printing a JSON summary to stdout at the end of the run.",
    )
    parser.set_defaults(print_json=True)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    in_path = Path(args.input).expanduser()
    df = load_url_csv(in_path, url_col="url" if args.kaggle_malicious_phish else "url", label_col="type" if args.kaggle_malicious_phish else "label")

    if args.kaggle_malicious_phish:
        label_map = {"benign": 0, "defacement": 1, "phishing": 1, "malware": 1}
        df = apply_label_map(df, label_col="type", label_map=label_map)
        df = df.rename(columns={"type": "label"})
    else:
        df = df.rename(columns={"label": "label"})

    df = enforce_binary_labels(df, label_col="label")
    df = downsample_rows(df, args.max_rows)

    train_df, val_df, test_df = split_train_val_test(
        df,
        label_col="label",
        test_size=args.test_size,
        val_size=args.val_size,
    )
    save_splits(train_df, val_df, test_df, paths.data_processed)

    # Train metadata branch
    meta_res = train_random_forest(train_df, val_df)
    save_metadata_model(meta_res.model, paths.models / "metadata.joblib")

    # Train ELECTRA
    train_electra_classifier(
        train_df=train_df,
        val_df=val_df,
        out_dir=paths.models / "electra",
        model_name=args.electra_model,
        epochs=args.electra_epochs,
        batch_size=args.electra_batch_size,
        max_train_samples=args.electra_max_train_samples,
    )

    # Train fusion (uses MC dropout features)
    electra = load_electra(paths.models / "electra")
    meta = load_metadata_model(paths.models / "metadata.joblib")

    train_sub = train_df
    val_sub = val_df
    if args.fusion_train_cap and len(train_sub) > args.fusion_train_cap:
        train_sub = train_sub.sample(n=args.fusion_train_cap, random_state=RANDOM_SEED).reset_index(drop=True)
    if args.fusion_val_cap and len(val_sub) > args.fusion_val_cap:
        val_sub = val_sub.sample(n=args.fusion_val_cap, random_state=RANDOM_SEED).reset_index(drop=True)

    train_mc = mc_dropout_predict_proba(
        electra, train_sub["url"].astype(str).tolist(), n_passes=args.mc_passes, batch_size=args.batch_size, progress=True
    )
    val_mc = mc_dropout_predict_proba(
        electra, val_sub["url"].astype(str).tolist(), n_passes=args.mc_passes, batch_size=args.batch_size, progress=True
    )
    p_meta_train = predict_metadata_proba(meta, train_sub["url"].astype(str).tolist())
    p_meta_val = predict_metadata_proba(meta, val_sub["url"].astype(str).tolist())

    x_train = np.stack([train_mc.p_mean, train_mc.p_std, p_meta_train], axis=1).astype(np.float32)
    y_train = train_sub["label"].astype(int).to_numpy()
    x_val = np.stack([val_mc.p_mean, val_mc.p_std, p_meta_val], axis=1).astype(np.float32)
    y_val = val_sub["label"].astype(int).to_numpy()
    fusion_res = train_fusion_logreg(x_train, y_train, x_val, y_val)
    save_fusion_model(fusion_res.model, paths.models / "fusion.joblib")

    # Evaluate on test
    test_urls = test_df["url"].astype(str).tolist()
    y = test_df["label"].astype(int).to_numpy()
    p_meta = predict_metadata_proba(meta, test_urls)
    mc = mc_dropout_predict_proba(electra, test_urls, n_passes=args.mc_passes, batch_size=args.batch_size, progress=True)
    x = np.stack([mc.p_mean, mc.p_std, p_meta], axis=1).astype(np.float32)
    p_fusion = fusion_res.model.predict_proba(x)[:, 1]

    metrics_out = {
        "metadata": compute_binary_metrics(y, p_meta).as_dict(),
        "electra_mc_dropout": compute_binary_metrics(y, mc.p_mean).as_dict(),
        "fusion": compute_binary_metrics(y, p_fusion).as_dict(),
    }
    out_path = paths.results / "metrics_reproduce.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"Wrote: {out_path}")
    if args.print_json:
        print(json.dumps(metrics_out, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
