# Hybrid Malicious URL Detector

## Repo

This repository is an end-to-end hybrid malicious URL detector that combines a transformer text model with structured URL/domain features, then adds uncertainty estimation and a lightweight fusion model.

### At A Glance

| Area | What this project does |
| --- | --- |
| Problem | Detects malicious URLs with a hybrid text + metadata approach |
| Dataset | Kaggle malicious URLs dataset included in the repo |
| Text branch | ELECTRA fine-tuned on raw URL strings |
| Structured branch | Random Forest on CTI-inspired URL and domain features |
| Uncertainty | Monte Carlo dropout on ELECTRA with mean, std, and confidence intervals |
| Fusion | Logistic regression combining text score, uncertainty, and metadata score |
| Output | End-to-end training, evaluation, fusion, and example selection scripts |
| Scope | Deliberately sized as a demoable, reproducible prototype |

Dataset included in this repo:
- `data/raw/malicious_phish.csv.zip` (Kaggle "malicious_phish"; mapped to binary labels)

Main entrypoint:
- `src/reproduce.py` (one-command end-to-end pipeline; writes metrics to `results/`)

Engineering highlights:
- Single command pipeline (`src/reproduce.py`) that: preprocesses, splits, trains, fuses, and evaluates
- Repeatable configuration captured in `results/run_config.json` (dataset path, label mapping, split sizes, and training caps)
- Concrete artifacts for review: metrics JSON, predictions CSV, trained model files, and example outputs under `results/`

### Why This Is Useful (Professional Framing)

This repo demonstrates applied ML patterns that translate well to production work:
- Multi-signal modeling: combines text understanding (transformer) with structured signals (engineered features)
- Operational uncertainty: adds a practical "flag ambiguous cases" layer via MC-dropout statistics
- Reproducibility-first: a single script produces models + metrics + artifacts from a fresh checkout

### Key Artifacts

After running, check `results/` for:
- `metrics_reproduce.json`: metrics from the one-command pipeline run
- `metrics.json`: metrics from the frozen/project run already included in the repo
- `predictions_test.csv`: per-example scores and predictions (from the frozen/project run)
- `examples.md` / `examples.csv`: qualitative examples for quick review
- `run_config.json`: the exact settings used for the frozen/project run

## Run

### 0) Install prerequisites (macOS / Windows)

You need:
- Git
- Python 3.10+ (with `pip` and `venv`)

macOS:
- Install Git (via Xcode Command Line Tools): `xcode-select --install`
- Install Python 3.10+: download and install from https://www.python.org/downloads/

Windows:
- Install Git for Windows: https://git-scm.com/download/win
- Install Python 3.10+: download and install from https://www.python.org/downloads/ (check "Add python.exe to PATH")

Verify installs:
- `git --version`
- macOS: `python3 --version`
- Windows: `py --version`

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

Terminal output:
- The script prints a metrics summary + artifact paths, then a JSON blob at the end.
- To disable JSON printing (e.g., for scripting), add `--no-print-json`.

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
