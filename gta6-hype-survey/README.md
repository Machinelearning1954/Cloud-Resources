# GTA 6 Fan Expectations — Preorder Intent & Fan Personas

A compact, reproducible ML project on the Kaggle **"GTA 6 Fan Expectations & Hype
Survey 2026"** dataset (synthetic survey data, 6,000 rows × ~20 columns). Two tasks:

1. **Preorder-intent classification** (supervised) — predict whether a respondent
   intends to preorder, from demographics, platform, loyalty, hype and engagement.
2. **Fan-persona segmentation** (unsupervised KMeans) — group respondents into
   marketing personas.

## Why this dataset

Clean tabular data that fits a standard sklearn pipeline — the same stack used
across this portfolio (scikit-learn, joblib, argparse CLI). Good for EDA,
classification, and clustering without GPU infrastructure.

## Run

```bash
pip install pandas scikit-learn numpy joblib

# with the real Kaggle CSV (recommended for a portfolio submission)
python gta6_hype_analysis.py --train --data gta6_hype_survey.csv --output artifacts/

# no CSV on hand? a labeled synthetic stand-in is generated so it runs anyway
python gta6_hype_analysis.py --train
```

Download the real CSV from Kaggle ("GTA 6 Fan Expectations & Hype Survey 2026").
If your target column isn't named `preorder_intent`, rename it before training.

## Pipeline

- `ColumnTransformer`: median-impute + standardize numerics; most-frequent-impute +
  one-hot encode categoricals — all inside the model pipeline (no leakage).
- **Classifier**: `RandomForestClassifier` (300 trees), reported against a
  5-fold cross-validated `LogisticRegression` AUC baseline.
- **Personas**: `KMeans` (k configurable via `--personas`), scored by silhouette.
- Models persisted with `joblib` to `--output`.

## Results (synthetic stand-in, seed 42)

| Metric | RandomForest |
|---|---|
| Accuracy | 0.84 |
| Precision | 0.71 |
| Recall | 0.49 |
| F1 | 0.58 |
| ROC-AUC | 0.876 |
| LogReg 5-fold CV AUC (baseline) | 0.894 |

Personas: k=4, silhouette ≈ 0.12. Numbers on the real Kaggle CSV will differ —
these confirm the pipeline is correct and reproducible.

## Note on the data

The Kaggle dataset is itself **synthetic** (simulated responses). When no CSV is
supplied, this script generates its own clearly-labeled synthetic stand-in so the
pipeline is runnable end to end; that stand-in is a scaffold for demonstration,
not a replacement for the real dataset in a submission.
