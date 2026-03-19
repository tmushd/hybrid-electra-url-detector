from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def defang(url: str) -> str:
    """
    Reduces the chance someone accidentally clicks a malicious URL in slides/README.
    """
    s = str(url)
    s = s.replace("http://", "hxxp://").replace("https://", "hxxps://")
    s = s.replace(".", "[.]")
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick representative qualitative examples from predictions_test.csv.")
    parser.add_argument("--preds", type=str, default="results/predictions_test.csv")
    parser.add_argument("--out-md", type=str, default="results/examples.md")
    parser.add_argument("--out-csv", type=str, default="results/examples.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    preds_path = (repo_root / args.preds).resolve()
    df = pd.read_csv(preds_path)

    thr = float(args.threshold)
    df["pred_electra"] = (df["p_electra_mean"] >= thr).astype(int)
    df["pred_fusion"] = (df["p_fusion"] >= thr).astype(int)

    # Helper views
    correct_mal = df[(df["label"] == 1) & (df["pred_electra"] == 1)].copy()
    correct_ben = df[(df["label"] == 0) & (df["pred_electra"] == 0)].copy()
    fp = df[(df["label"] == 0) & (df["pred_electra"] == 1)].copy()
    fn = df[(df["label"] == 1) & (df["pred_electra"] == 0)].copy()
    uncertain = df[df["uncertain"] == 1].copy()

    picks = []
    if len(correct_mal):
        picks.append(("Correct malicious (ELECTRA)", correct_mal.sort_values("p_electra_mean", ascending=False).head(1)))
    if len(correct_ben):
        picks.append(("Correct benign (ELECTRA)", correct_ben.sort_values("p_electra_mean", ascending=True).head(1)))
    if len(fp):
        picks.append(("False positive (ELECTRA)", fp.sort_values("p_electra_mean", ascending=False).head(1)))
    if len(fn):
        picks.append(("False negative (ELECTRA)", fn.sort_values("p_electra_mean", ascending=True).head(1)))
    if len(uncertain):
        picks.append(("Uncertain example", uncertain.sort_values("p_electra_std", ascending=False).head(1)))

    out_rows = []
    for title, sub in picks:
        row = sub.iloc[0].to_dict()
        row["example_type"] = title
        row["url_defanged"] = defang(row.get("url", ""))
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)[
        [
            "example_type",
            "url_defanged",
            "label",
            "p_meta",
            "p_electra_mean",
            "p_electra_std",
            "ci_low",
            "ci_high",
            "uncertain",
            "p_fusion",
        ]
    ]

    out_csv = (repo_root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # Markdown output
    out_md = (repo_root / args.out_md).resolve()
    lines = []
    lines.append("# Qualitative Examples (Defanged)")
    lines.append("")
    lines.append(f"Source: `{preds_path}`. Threshold: `{thr}`.")
    lines.append("")
    lines.append(
        "| example | url (defanged) | label | p_meta | p_electra_mean | p_electra_std | CI | uncertain | p_fusion |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in out_df.iterrows():
        ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        lines.append(
            f"| {r['example_type']} | `{r['url_defanged']}` | {int(r['label'])} | {r['p_meta']:.3f} | "
            f"{r['p_electra_mean']:.3f} | {r['p_electra_std']:.3f} | {ci} | {int(r['uncertain'])} | {r['p_fusion']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

