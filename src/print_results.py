from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _sanitize_for_json(obj):
    # Ensure strict JSON output (no NaN/Infinity).
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Print saved results to the terminal (no training).")
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="",
        help="Path to a metrics JSON file. Defaults to results/metrics_reproduce.json if present, else results/metrics.json.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    if args.metrics_path:
        metrics_path = Path(args.metrics_path).expanduser()
        if not metrics_path.is_absolute():
            metrics_path = repo_root / metrics_path
    else:
        metrics_path = results_dir / "metrics_reproduce.json"
        if not metrics_path.exists():
            metrics_path = results_dir / "metrics.json"

    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_safe = _sanitize_for_json(metrics)

    print("=== Metrics Summary ===")
    if isinstance(metrics_safe, dict):
        for k, v in metrics_safe.items():
            if isinstance(v, dict):
                parts = []
                for metric_name in ("roc_auc", "accuracy", "f1", "precision", "recall"):
                    if metric_name in v:
                        parts.append(f"{metric_name}={v.get(metric_name)}")
                if parts:
                    print(f"{k}: " + " ".join(parts))
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: {v}")
    else:
        print(metrics_safe)

    print("\n=== Artifacts ===")
    print(str(metrics_path))
    for p in (
        results_dir / "run_config.json",
        results_dir / "predictions_test.csv",
        results_dir / "examples.md",
        results_dir / "examples.csv",
        results_dir / "fusion_probability_histogram.png",
    ):
        if p.exists():
            print(str(p))

    print("\n=== JSON ===")
    print(json.dumps(metrics_safe, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

