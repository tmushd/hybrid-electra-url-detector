# Hybrid Malicious URL Detector

## Repo

Uncertainty-aware hybrid malicious URL detection prototype:
- Text branch: ELECTRA fine-tuned on raw URL strings
- Structured branch: Random Forest on URL/domain features
- Uncertainty: Monte Carlo dropout on ELECTRA (mean/std over repeated inference)
- Fusion: Logistic regression on `[p_electra_mean, p_electra_std, p_meta]`

Dataset included in this repo:
- `data/raw/malicious_phish.csv.zip` (Kaggle “malicious_phish”; mapped to binary labels)

Main entrypoint:
- `src/reproduce.py` (one-command end-to-end pipeline; writes metrics to `results/`)

## Run

### 1) Clone

```bash
git clone https://github.com/tmushd/hybrid-electra-url-detector.git
cd hybrid-electra-url-detector
```

### 2) Environment setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 2) Environment setup (Windows PowerShell)

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
pip install -r requirements.txt
```

Notes:
- First run requires internet (downloads the pretrained ELECTRA weights from Hugging Face).
- If `torch` fails to install from `requirements.txt`, install PyTorch first (per your CPU/GPU) and then re-run `pip install -r requirements.txt`.
- The pipeline uses a fixed seed (`1337`), but exact metrics can still vary slightly across hardware (CPU vs GPU vs Apple MPS).

### 3) Quick run (downsized, single command)

```bash
python src/reproduce.py --input data/raw/malicious_phish.csv.zip --kaggle-malicious-phish --max-rows 5000 --electra-epochs 1 --electra-max-train-samples 2000 --mc-passes 3 --fusion-train-cap 2000 --fusion-val-cap 1000
```

### 4) Reproduce our run (exact command + code version)

Code version (commit): `37f648f66fd731b56bfe41e0f9e56c30dd51292c`

These settings mirror `results/run_config.json`.

```bash
python src/reproduce.py --input data/raw/malicious_phish.csv.zip --kaggle-malicious-phish --max-rows 50000 --test-size 0.2 --val-size 0.2 --electra-model google/electra-small-discriminator --electra-epochs 1 --electra-max-train-samples 20000 --electra-batch-size 16 --mc-passes 10 --batch-size 64 --fusion-train-cap 8000 --fusion-val-cap 3000
```

### Outputs

After either run completes:
- Metrics: `results/metrics_reproduce.json`
- Processed splits: `data/processed/{train,val,test}.csv`
- Trained models: `models/metadata.joblib`, `models/fusion.joblib`, `models/electra/`
