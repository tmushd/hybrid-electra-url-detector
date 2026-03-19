# Dataset

## Source

This project uses the **`malicious_phish`** dataset (commonly distributed as `malicious_phish.csv.zip`).

Expected raw schema:
- `url` (string)
- `type` (string; one of `benign`, `defacement`, `phishing`, `malware`)

## Labels used in this prototype (binary)

For this midterm prototype we convert the 4-class `type` into a binary target:
- `label = 0` for `benign`
- `label = 1` for `{defacement, phishing, malware}` (“malicious”)

This mapping is applied by:
`/Users/vayu/Documents/Playground/hybrid_url_detector/src/preprocess.py` with `--kaggle-malicious-phish`.

## Subset and split (frozen run)

The reported metrics in `results/metrics.json` were produced with:
- Randomly sampled subset size: **50,000 rows** (`--max-rows 50000`)
- Split: **60/20/20** into train/val/test
  - train: 30,000
  - val: 10,000
  - test: 10,000
- Seed: `1337` (see `src/hybrid_url_detector/config.py`)

## Notes / limitations

- URLs may be live/malicious. Handle carefully and avoid clicking them.
- This repo does **not** include the original raw dataset by default; it includes scripts to reproduce the splits and results.
