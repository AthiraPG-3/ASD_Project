# Autism Scan Classification (MLOps Project)

MSc Data Science final semester project: an end-to-end **MLOps pipeline** to classify preprocessed brain scan images into **Autism** vs **Control** using **classical ML** with robust **feature extraction**, **experiment tracking**, **data/model versioning**, **CI/CD**, **model registry**, **monitoring**, and a simple **inference service**.

> **Disclaimer:** This project is for academic/research use only and is **not** a medical diagnostic device.

## 1) Project structure

```
.
├─ data/
│  ├─ scans/                  # (DVC-tracked) train/val/test folders with autism/control
│  └─ features/               # extracted features (DVC output)
├─ models/                    # serialized best model (DVC output)
├─ reports/                   # metrics, plots, drift report (DVC output)
├─ scripts/                   # pipeline entrypoints (called by DVC/CI)
├─ service/                   # FastAPI inference service
└─ src/autism_mlops/          # reusable library code
```

## 2) Dataset layout (expected)

Place your dataset under `data/scans/` (recommended):

```
data/scans/
  train/
    autism/   *.png|*.jpg|*.jpeg|*.bmp|*.tif|*.tiff
    control/  *.png|...
  val/  (or "validation/")
    autism/
    control/
  test/
    autism/
    control/
```

Labels:
- `control` → 0
- `autism` → 1

### If your data stays outside the repo

Option A (recommended for DVC): add it as an external DVC dependency:

```powershell
dvc add --external "E:\\path\\to\\your\\scans"
```

Option B (quick local run): set `data.scans_dir` in `params.yaml` to an absolute path and run the scripts directly (but DVC won’t version your raw data changes).

## 3) Setup (local)

1) Create and activate a venv, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Initialize Git + DVC:

```powershell
git init
dvc init
```

3) Track your data with DVC (recommended):

```powershell
dvc add data/scans
git add data/scans.dvc .gitignore
git commit -m "Track scans with DVC"
```

## 4) Run the pipeline (DVC)

From the repo root:

```powershell
dvc repro
```

Outputs:
- `data/features/features.npz`
- `models/best_model.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.png`
- `reports/drift_report.html`

## 5) Experiment tracking (MLflow + optional W&B)

### MLflow (local default)
Runs are stored under `./mlruns/` by default.

### MLflow on DagsHub
Create a DagsHub repo, then set:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<user>"
$env:MLFLOW_TRACKING_PASSWORD="<dagshub_token>"
```

### Weights & Biases (optional)

```powershell
$env:WANDB_API_KEY="<your_key>"
# optional (useful for labs / no-internet environments)
$env:WANDB_MODE="offline"
```

## 6) DVC remote on DagsHub (data versioning)

After creating your DagsHub repo, configure a DVC remote and push:

```powershell
dvc remote add -d dagshub https://dagshub.com/<user>/<repo>.dvc
dvc remote modify dagshub auth basic
dvc remote modify dagshub user <user>
dvc remote modify dagshub password <dagshub_token>

dvc push
git add .dvc/config
git commit -m "Configure DagsHub DVC remote"
```

## 7) Serve the model (FastAPI)

1) Train once (via `dvc repro`) so `models/best_model.joblib` exists.
2) Start API:

```powershell
uvicorn service.main:app --reload
```

Then POST an image file to `http://127.0.0.1:8000/predict`.

## 8) CI/CD (GitHub Actions)

Workflows are included in `.github/workflows/`:
- `ci.yml`: lint + unit tests
- `train.yml`: (manual) run the DVC pipeline (requires DVC remote + secrets)

## 9) Model registry (MLflow)

After training, register the logged MLflow model:

```powershell
python scripts/register_model.py --model-name autism-scan-classifier --stage Staging
```

## 10) Notes on “promotion workflow”

Suggested stages using **MLflow Model Registry**:
1. **Train** → log run + artifacts to MLflow
2. **Register** best run as `autism-scan-classifier`
3. **Promote to Staging** after metric checks
4. **Promote to Production** with manual approval (GitHub Environments / reviewer sign-off)
5. **Monitor** drift/performance; rollback if necessary
