# Hybrid Malicious URL Detector (Midterm Prototype)

Prototype of an **uncertainty-aware hybrid malicious URL detector** inspired by:
- ELECTRA-style transformer backbone for raw URL text
- probabilistic/uncertainty via repeated inference (Monte Carlo dropout)
- CTI-inspired ensemble fusion (separate predictors + final combiner)

This repo is intentionally scoped to a **one-week, demoable midterm**:
- **Branch A (Text):** ELECTRA fine-tuned on raw URL strings
- **Branch B (Structured):** RandomForest on CTI-inspired URL/domain features
- **Uncertainty:** MC-dropout on ELECTRA (`mean`, `std`, `95% CI`)
- **Fusion:** Logistic regression on `[p_electra_mean, p_electra_std, p_meta]`

## TL;DR (What I built)

I implemented a hybrid malicious URL detection prototype consisting of:
- a **Random Forest metadata branch** trained on CTI-inspired URL/domain features,
- an **ELECTRA text branch** trained on raw URL strings,
- **Monte Carlo dropout** on ELECTRA to estimate uncertainty (mean/std/95% CI),
- a **logistic-regression fusion layer** using ELECTRA mean score, ELECTRA uncertainty, and metadata score.

## 1) Setup

### macOS / Linux

Create a venv and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows (cmd.exe)

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Reproducibility notes

- First run needs **internet** to download the pretrained model from Hugging Face.
- `torch` installation can differ by OS/CPU/GPU. If `pip install -r requirements.txt` fails on `torch`,
  install PyTorch first using the official selector, then install the rest:
  [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
- Train/test split and training seeds are fixed (`1337`), but exact metrics can still vary slightly across hardware
  (CPU vs GPU vs Apple MPS).

## 2) Data format

Put a CSV at `data/raw/urls.csv` with at least:
- `url` (string)
- `label` (`0` = benign, `1` = malicious)

Example:
```csv
url,label
https://example.com,0
http://secure-login-paypal.com/update,1
```

### Using the ELECTRA paper dataset (`malicious_phish.csv.zip`)

That dataset typically has columns `url,type` where `type` is one of:
`benign`, `defacement`, `phishing`, `malware`.

You can preprocess it directly from the zip like this:
```bash
python src/preprocess.py --input /path/to/malicious_phish.csv.zip --kaggle-malicious-phish
```

Optional downsample for faster training:
```bash
python src/preprocess.py --input /path/to/malicious_phish.csv.zip --kaggle-malicious-phish --max-rows 50000
```

## 3) Run (suggested order)

Preprocess / split:
```bash
python src/preprocess.py --input data/raw/urls.csv
```

If `data/processed/train.csv` / `val.csv` / `test.csv` are already present (e.g., the frozen 50k split),
you can skip preprocessing and start training from the next steps.

Train metadata (structured) branch:
```bash
python src/train_metadata_model.py
```

Fine-tune ELECTRA branch:
```bash
python src/train_electra.py --model google/electra-small-discriminator
```

Train fusion model:
```bash
python src/fusion.py --mc-passes 30
```

Evaluate on test split:
```bash
python src/evaluate.py --mc-passes 30
```

### Reproduce the frozen run (the numbers in `results/metrics.json`)

```bash
python src/preprocess.py --input /path/to/malicious_phish.csv.zip --kaggle-malicious-phish --max-rows 50000
python src/train_metadata_model.py
python src/train_electra.py --model google/electra-small-discriminator --epochs 1 --max-train-samples 20000
python src/fusion.py --mc-passes 10 --max-train-samples 8000 --max-val-samples 3000
python src/evaluate.py --mc-passes 10
python src/pick_examples.py
```

### One-command reproduction (optional)

This runs the pipeline end-to-end:
```bash
python src/reproduce.py --input /path/to/malicious_phish.csv.zip --kaggle-malicious-phish
```

## 4) What “CTI” means here (honest scope)

This midterm implements a **lightweight CTI-inspired feature branch** using structured URL/domain signals
(lexical counts, URL patterns, keyword flags, entropy-ish randomness). Live Google CTI and Whois pipelines are
left as future work.

## 5) Outputs

- Processed splits: `data/processed/{train,val,test}.csv`
- Trained models: `models/metadata.joblib`, `models/fusion.joblib`, `models/electra/`
- Metrics + predictions: `results/`

## 6) Results (frozen run)

See:
- `results/metrics.json`
- `results/predictions_test.csv`
- `results/examples.md` (defanged qualitative examples)
- `results/run_config.json` (frozen run settings)
- `data/DATASET.md` (dataset + label mapping)
- `slides/outline.md` (ready-to-copy slide content)

Headline metrics on the test split:
- Metadata (RF): ROC-AUC **0.9764**, Accuracy **0.9308**
- ELECTRA (deterministic): ROC-AUC **0.9866**, Accuracy **0.9600**
- ELECTRA (MC-dropout mean): ROC-AUC **0.9850**, Accuracy **0.9601**
- Uncertainty: **3.61%** flagged uncertain
- Fusion (logreg): ROC-AUC **0.9829**, Accuracy **0.9421**

Interpretation (midterm-appropriate):
- ELECTRA is the strongest standalone model (consistent with the transformer motivation).
- Metadata features carry useful signal but are simplified “CTI-inspired” features.
- Uncertainty adds a practical safety layer by flagging ambiguous cases.
- Fusion is a working first prototype but does not yet outperform ELECTRA.

## 7) Notes

- ELECTRA training can be slow on CPU; for a quick demo, reduce epochs via `--epochs 1` and/or sample data via `--max-train-samples`.
- MC-dropout uncertainty: mark a prediction as **uncertain** if the 95% CI crosses `0.5` or if `std` exceeds a threshold.

## Repo structure

Key paths:
- `src/preprocess.py`: loads raw dataset, maps labels, creates splits
- `src/train_metadata_model.py`: trains RandomForest on structured features
- `src/train_electra.py`: fine-tunes ELECTRA on URL text
- `src/fusion.py`: trains fusion model
- `src/evaluate.py`: evaluates models and writes `results/*`
- `src/pick_examples.py`: extracts qualitative examples into `results/examples.*`
- `src/hybrid_url_detector/url_features.py`: CTI-inspired feature extraction
- `src/hybrid_url_detector/uncertainty.py`: MC-dropout mean/std/CI + uncertainty rule
