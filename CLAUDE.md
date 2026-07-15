# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multiclass classification datathon. Predict student performance class (0, 1, 2, 3) from academic, behavioral, and demographic features. Roughly balanced classes (~800 each) across 3200 training samples.

## Data

- `data/train.csv` — 3200 rows, 42 features + `target` (4 classes)
- `data/test.csv` — 800 rows, 42 features, no target
- `data/sample_submission.csv` — 800 rows, format: `id,target`

### Feature Groups

| Group | Features | Count |
|---|---|---|
| Weekly scores | `nilai_minggu_01` — `nilai_minggu_12` | 12 |
| Daily activity | `aktivitas_hari_01` — `aktivitas_hari_16` | 16 |
| Task completion | `tugas_selesai`, `tugas_diberikan` | 2 |
| Exams | `kelas`, `urutan_ujian`, `skor_tryout` | 3 |
| Behavioral | `skor_motivasi`, `skor_kedisiplinan`, `skor_ekstrakurikuler`, `indeks_kehadiran`, `skor_literasi`, `skor_minat_belajar` | 6 |
| Demographic | `jarak_rumah_km`, `jumlah_saudara` | 2 |
| ID | `id` | 1 |

No missing values. 38 float features, 5 int features (`id`, `tugas_selesai`, `tugas_diberikan`, `kelas`, `target`).

## Environment

- Python 3.13.4
- Key libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, jupyter
- No virtual environment file (no .venv/ or requirements.txt yet)

## Common Commands

```bash
# Explore or prototype
jupyter notebook                       # Launch notebook server
jupyter lab                            # Launch JupyterLab

# Run a training script
python train.py                        # When a training script exists

# If this becomes a git repo (recommended)
git init                               # Initialize repo
```

## Typical Workflow

1. EDA in a notebook or script — examine feature distributions, correlations, target relationships
2. Feature engineering — weekly-score trends, activity patterns, interactions
3. Model training — start with sklearn classifiers (RandomForest, LogisticRegression), then XGBoost/LightGBM
4. Hyperparameter tuning — GridSearchCV or Optuna
5. Submission — predict on test.csv, format as sample_submission.csv
