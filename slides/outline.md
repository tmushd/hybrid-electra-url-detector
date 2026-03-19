# Slide Outline (8–10 slides, ~10 min)

## Slide 1 — Title

**Uncertainty-Aware Hybrid ELECTRA + CTI-Inspired Framework for Malicious URL Detection**  
Name • Course • Date

Speaker notes:
- Goal: show a real prototype that combines 3 paper ideas.

## Slide 2 — Problem

- Malicious URLs (phishing/malware/defacement) are common and costly.
- Text-only URL detection can be brittle; URL-only signals can be manipulated.
- Confident wrong predictions are dangerous → we want **uncertainty awareness**.

## Slide 3 — Paper: Probabilistic vs Deterministic (uncertainty)

- Deterministic models output a fixed prediction, no confidence.
- Probabilistic / repeated inference provides uncertainty (e.g., confidence intervals).
- Motivation: flag ambiguous URLs instead of pretending certainty.

## Slide 4 — Paper: CTI Ensemble (multi-source fusion)

- Uses multiple information sources (URL, Whois, Google/CTI).
- Separate predictors per feature source; combine outputs for final decision.
- Motivation: “attacker-independent” signals can reduce false positives.

## Slide 5 — Paper: ELECTRA / Transformers (strong backbone)

- Transformer-based modeling of URL text is very strong.
- ELECTRA-style pretraining can outperform older baselines in similar setups.

## Slide 6 — My Approach (bridge)

- Paper 1 → uncertainty via repeated inference (MC dropout).
- Paper 2 → separate branches + fusion logic.
- Paper 3 → ELECTRA as the URL-text backbone.

Goal:
- Build a **demoable** hybrid prototype (not full industrial CTI scraping).

## Slide 7 — Methodology (diagram)

Show the diagram from `slides/architecture.mmd`.

- Branch A: ELECTRA on raw URL → `p_electra_mean`, `p_electra_std`, CI
- Branch B: RF on CTI-inspired structured features → `p_meta`
- Fusion: logistic regression on `[p_electra_mean, p_electra_std, p_meta]`
- Uncertain rule: CI crosses 0.5 OR std above threshold

## Slide 8 — Results (frozen run)

Dataset setup:
- `malicious_phish` (binary): benign vs {defacement, phishing, malware}
- 50k subset → 30k train / 10k val / 10k test

Test metrics (from `results/metrics.json`):
- Metadata (RF): ROC-AUC 0.9764, Acc 0.9308
- ELECTRA: ROC-AUC 0.9866, Acc 0.9600
- ELECTRA + MC dropout mean: ROC-AUC 0.9850, Acc 0.9601
- Uncertain fraction: 3.61%
- Fusion: ROC-AUC 0.9829, Acc 0.9421

Key message:
- Fusion does **not** beat ELECTRA yet → that’s a normal first-prototype result.

## Slide 9 — Qualitative examples

Use 3–5 rows from `results/examples.md` (URLs are defanged).

Talk track:
- Example of confident correct
- One false positive / false negative (dataset noise + limitations)
- One uncertain example (CI crosses threshold)

## Slide 10 — Limitations + Future work

Limitations (be honest):
- CTI branch is “CTI-inspired” (no live Google/Whois pipeline yet).
- Fusion is simple (logreg); needs tuning/calibration.
- Evaluation is preliminary (single dataset, limited ablations).

Future work:
- Add Whois/domain-age/reputation signals and safe CTI enrichment.
- Improve fusion (calibrated stacking, MLP, per-class fusion, thresholds).
- Calibration/uncertainty analysis and robustness testing.
