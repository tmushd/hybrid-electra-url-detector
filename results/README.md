# Results

## Files

- `metrics.json`: headline metrics on the **test** split.
- `predictions_test.csv`: per-example predictions on the **test** split.

## `predictions_test.csv` columns

- `url`: raw URL string (do not click)
- `label`: ground-truth label (`0` benign, `1` malicious)
- `p_meta`: metadata/RF probability of malicious
- `p_electra`: deterministic ELECTRA probability of malicious
- `p_electra_mean`: MC-dropout mean probability
- `p_electra_std`: MC-dropout std dev (uncertainty proxy)
- `ci_low`, `ci_high`: 95% CI for the mean estimate
- `uncertain`: 1 if CI crosses 0.5 **or** std exceeds threshold
- `p_fusion`: fusion probability (logreg on `[p_electra_mean, p_electra_std, p_meta]`)
