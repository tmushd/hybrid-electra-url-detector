# Grader / Repro Instructions

## Quick start (no Kaggle account needed)

This repo includes a **frozen 50k subset** split already:
- `data/processed/train.csv` (30k)
- `data/processed/val.csv` (10k)
- `data/processed/test.csv` (10k)

So you can run the full pipeline without downloading the original Kaggle zip.

### 1) Clone

```bash
git clone https://github.com/tmushd/hybrid-electra-url-detector.git
cd hybrid-electra-url-detector
```

### 2) Install deps

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PyTorch fails to install, install it first via:
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

### 3) Train models + evaluate

```bash
python src/train_metadata_model.py
python src/train_electra.py --model google/electra-small-discriminator --epochs 1 --max-train-samples 20000
python src/fusion.py --mc-passes 10 --max-train-samples 8000 --max-val-samples 3000
python src/evaluate.py --mc-passes 10 --progress
python src/pick_examples.py
```

Expected outputs:
- `results/metrics.json`
- `results/predictions_test.csv`
- `results/examples.md`

## Full dataset (optional)

If you want to run preprocessing from the original dataset:

1. Download `malicious_phish.csv.zip` from Kaggle (file contains columns `url,type`).
2. Run:
```bash
python src/preprocess.py --input /path/to/malicious_phish.csv.zip --kaggle-malicious-phish --max-rows 50000
```

If the dataset is already present in this repo at `data/raw/malicious_phish.csv.zip`, you can run:
```bash
python src/preprocess.py --input data/raw/malicious_phish.csv.zip --kaggle-malicious-phish --max-rows 50000
```

## Safety note

The datasets contain real URLs that may be malicious. Do not click them.
