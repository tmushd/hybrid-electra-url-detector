from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import time

from hybrid_url_detector.electra_branch import ElectraLoaded


@dataclass(frozen=True)
class McDropoutResult:
    p_mean: np.ndarray
    p_std: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def mc_dropout_predict_proba(
    loaded: ElectraLoaded,
    urls: list[str],
    n_passes: int = 30,
    max_length: int = 128,
    batch_size: int = 32,
    ci_z: float = 1.96,
    progress: bool = False,
    cache_on_device: bool = True,
) -> McDropoutResult:
    """
    Monte Carlo dropout for uncertainty estimation.

    Implementation detail: enable dropout by setting model.train(), but keep torch.no_grad().
    CI is computed for the *mean* estimate: mean ± z * (std / sqrt(n_passes)).
    """
    if n_passes < 2:
        raise ValueError("n_passes must be >= 2 for uncertainty estimation.")

    all_passes = []
    was_training = loaded.model.training
    try:
        loaded.model.train()
        with torch.no_grad():
            start = time.time()
            # Tokenize once, reuse for all MC passes.
            cpu_batches = []
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i : i + batch_size]
                enc = loaded.tokenizer(
                    batch_urls,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                cpu_batches.append(enc)

            if cache_on_device and loaded.device.type in {"cuda", "mps"}:
                batches = [_move_batch_to_device(b, loaded.device) for b in cpu_batches]
            else:
                batches = cpu_batches

            for _ in range(n_passes):
                pass_start = time.time()
                probs = []
                for enc in batches:
                    if not cache_on_device or loaded.device.type == "cpu":
                        enc = _move_batch_to_device(enc, loaded.device)
                    logits = loaded.model(**enc).logits
                    p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                    probs.extend(p.tolist())
                all_passes.append(np.asarray(probs, dtype=np.float32))
                if progress:
                    done = len(all_passes)
                    elapsed = time.time() - start
                    per_pass = elapsed / max(done, 1)
                    eta = per_pass * (n_passes - done)
                    print(
                        f"[mc-dropout] pass {done}/{n_passes} done "
                        f"({time.time() - pass_start:.1f}s, eta {eta/60:.1f}m)"
                    )
    finally:
        if not was_training:
            loaded.model.eval()

    stacked = np.stack(all_passes, axis=0)  # [T, N]
    p_mean = stacked.mean(axis=0)
    p_std = stacked.std(axis=0, ddof=1)
    se = p_std / np.sqrt(float(n_passes))
    ci_low = np.clip(p_mean - ci_z * se, 0.0, 1.0)
    ci_high = np.clip(p_mean + ci_z * se, 0.0, 1.0)
    return McDropoutResult(p_mean=p_mean, p_std=p_std, ci_low=ci_low, ci_high=ci_high)


def is_uncertain(ci_low: np.ndarray, ci_high: np.ndarray, std: np.ndarray, std_threshold: float = 0.15) -> np.ndarray:
    crosses = (ci_low <= 0.5) & (ci_high >= 0.5)
    high_std = std >= std_threshold
    return crosses | high_std
