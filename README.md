# PCOD Detection System

A desktop-ready PCOS prediction platform built with Django, machine learning, and a packaged Windows executable.

Desktop app name: `PCOS AI Clinical Support Platform`

It supports:
- `Basic Screening` for symptom-based risk estimation
- `Clinical Assessment` for lab and follicle-based prediction
- explainable AI outputs for model-driven contributors
- analytics charts, doctor lookup, education content, and desktop packaging

## Windows Download

Download the latest desktop app here:

- [Download `pcod.exe`](https://github.com/sangeethsanthosh-git/PCOD-Detection-System/releases/latest/download/pcod.exe)
- [View all releases](https://github.com/sangeethsanthosh-git/PCOD-Detection-System/releases/latest)

The packaged executable uses the icon from `assets/favicon.ico` and runs the Django app inside a native `pywebview` desktop window.

## Current Model Snapshot

Latest saved clinical model metrics from [model_metrics.csv](C:/pcod/results/model_metrics.csv):

| Model | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| LogisticRegression | 94.50% | 0.917 | 0.963 |
| XGBoost | 93.58% | 0.896 | 0.948 |
| StackingEnsemble | 91.74% | 0.883 | 0.959 |

The current best production model is `LogisticRegression`, with coefficient-based local explanations supported in the app.

## Features

- Dual prediction workflow for public screening and doctor-facing clinical assessment
- Trained clinical model artifacts synced into the Django app
- Feature importance, ROC curve, confusion matrix, and SHAP-based reporting
- Native Windows `.exe` packaging with persisted local database storage
- Search, education, doctor discovery, and analysis dashboards in one interface

## Project Structure

```text
PCOD-Detection-System/
|- assets/                     # icon assets
|- dataset/                    # training data
|- models/                     # exported model artifacts
|- pcos_project/               # Django app + desktop launcher
|- results/                    # metrics, charts, feature importance outputs
|- src/                        # training pipeline modules
`- main.py                     # end-to-end model training entrypoint
```

## Local Setup

### 1. Activate the existing virtual environment

```powershell
C:\PCOD-Detection-System\venv\Scripts\Activate.ps1
```

### 2. Recreate dependencies if needed

This repo currently uses the existing local `venv` and does not yet ship a pinned `requirements.txt`.

If you are rebuilding the environment from scratch, install the main packages used by the app and training pipeline:
- `django`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `shap`
- `pywebview`
- `pyinstaller`

### 3. Run the Django app

```powershell
cd C:\PCOD-Detection-System\pcos_project
C:\PCOD-Detection-System\venv\Scripts\python.exe manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Retrain the Clinical Model

Run the full training pipeline from the repo root:

```powershell
cd C:\PCOD-Detection-System
C:\PCOD-Detection-System\venv\Scripts\python.exe main.py
```

This regenerates:
- `models/*.pkl`
- `pcos_project/models/*.pkl`
- `results/model_metrics.csv`
- `results/cv_summary.csv`
- `results/feature_scores.csv`
- plots in `results/`

## Build the Windows Executable

From `C:\PCOD-Detection-System\pcos_project`:

```powershell
C:\PCOD-Detection-System\venv\Scripts\pyinstaller.exe --noconfirm --clean launch_app.spec
```

The output executable is:

```text
pcos_project\dist\pcod.exe
```

Detailed packaging notes are in [BUILD_EXE.md](C:/PCOD-Detection-System/pcos_project/BUILD_EXE.md).

## Notes on External Data

I checked additional public PCOS sources and documented the useful ones in [DATASET_SOURCES.md](C:/PCOD-Detection-System/DATASET_SOURCES.md).

Important takeaway:
- the clinical dataset family already in this repo remains the best trustworthy source for the current tabular model
- several public datasets are image-based, omics-based, duplicated, or summary-only and are not safe direct merges into the current training table

## Tech Stack

- Python
- Django
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- PyInstaller
- pywebview

## Repository

- GitHub: <https://github.com/sangeethsanthosh-git/PCOD-Detection-System>
