"""
GTA 6 Fan Expectations & Hype Survey — Analysis and Modeling

Trains two models on the Kaggle "GTA 6 Fan Expectations & Hype Survey" dataset:
  1. Preorder-intent classifier (supervised) — will a respondent preorder?
  2. Fan-persona segmentation (unsupervised KMeans) — cluster fans into personas.

The dataset is synthetic (simulated survey responses). If the real Kaggle CSV is
not supplied via --data, a schema-matching synthetic stand-in is generated so the
pipeline runs end to end; the stand-in is clearly labeled and is NOT a substitute
for the real dataset in a portfolio submission.

CLI:
    python gta6_hype_analysis.py --train --data gta6_hype_survey.csv --output artifacts/
    python gta6_hype_analysis.py --train            # uses synthetic fallback
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, silhouette_score,
)
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "preorder_intent"
PLATFORMS = ["PlayStation 5", "Xbox Series X/S", "PC", "Undecided"]
REGIONS = ["North America", "Europe", "Latin America", "Asia", "Africa", "Oceania"]
FEATURES_WANTED = ["Map size", "Online/multiplayer", "Story/characters",
                   "Graphics/realism", "Modding", "Heists"]


class HypeSurveyModel:
    """Trains and evaluates preorder-intent and fan-persona models."""

    def __init__(self, seed: int = 42, n_personas: int = 4):
        self.seed = seed
        self.n_personas = n_personas
        self.clf: Pipeline | None = None
        self.persona: Pipeline | None = None
        self.feature_cols: list[str] = []

    # ---- data ------------------------------------------------------------
    def make_synthetic(self, n: int = 6000) -> pd.DataFrame:
        """Schema-matching synthetic stand-in (labeled). Mirrors the Kaggle set's
        realistic relationships, e.g. long-time GTA V fans skew more hyped."""
        rng = np.random.default_rng(self.seed)
        age = rng.integers(16, 55, n)
        years_playing = rng.integers(0, 25, n)
        platform = rng.choice(PLATFORMS, n, p=[0.42, 0.23, 0.30, 0.05])
        region = rng.choice(REGIONS, n, p=[0.34, 0.30, 0.14, 0.14, 0.05, 0.03])
        top_feature = rng.choice(FEATURES_WANTED, n)
        played_gtav = rng.random(n) < 0.82
        willingness_pay = np.clip(rng.normal(70, 22, n) + played_gtav * 8, 0, 200).round(2)
        # hype: driven by loyalty, willingness to pay, youth
        hype = np.clip(
            3.0 + 0.09 * years_playing + 0.015 * (willingness_pay - 70)
            - 0.02 * (age - 30) + played_gtav * 0.8 + rng.normal(0, 1.0, n),
            1, 10,
        ).round(1)
        engagement = np.clip(rng.normal(5, 2, n) + (hype - 5) * 0.5, 0, 10).round(1)
        # preorder intent: logistic-ish in hype + willingness + engagement
        z = -6 + 0.55 * hype + 0.02 * willingness_pay + 0.18 * engagement + rng.normal(0, 0.8, n)
        preorder = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
        return pd.DataFrame({
            "age": age, "region": region, "platform": platform,
            "years_playing_gta": years_playing, "played_gta_v": played_gtav,
            "top_feature": top_feature, "willingness_to_pay_usd": willingness_pay,
            "hype_level": hype, "engagement_score": engagement,
            TARGET: preorder,
        })

    def load(self, path: str | None) -> pd.DataFrame:
        if path and Path(path).exists():
            logger.info("Loading real dataset: %s", path)
            df = pd.read_csv(path)
            if TARGET not in df.columns:
                raise ValueError(
                    f"'{TARGET}' column not found. Columns: {list(df.columns)}. "
                    "Map your target column to 'preorder_intent' before training."
                )
            return df
        logger.warning("No --data CSV found; generating labeled SYNTHETIC stand-in "
                       "(not for portfolio submission — download the real Kaggle CSV).")
        return self.make_synthetic()

    # ---- modeling --------------------------------------------------------
    def _preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat = [c for c in X.columns if c not in num]  # pandas-3 safe (object/string/category)
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ])

    def train_classifier(self, df: pd.DataFrame) -> dict:
        y = df[TARGET].astype(int)
        X = df.drop(columns=[TARGET])
        self.feature_cols = X.columns.tolist()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y)
        pre = self._preprocessor(X)
        self.clf = Pipeline([("pre", pre), ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=self.seed, n_jobs=-1))])
        self.clf.fit(Xtr, ytr)
        proba = self.clf.predict_proba(Xte)[:, 1]
        pred = (proba > 0.5).astype(int)
        metrics = {
            "accuracy": round(accuracy_score(yte, pred), 4),
            "precision": round(precision_score(yte, pred, zero_division=0), 4),
            "recall": round(recall_score(yte, pred, zero_division=0), 4),
            "f1": round(f1_score(yte, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(yte, proba), 4),
        }
        # baseline for context
        base = LogisticRegression(max_iter=1000)
        base_scores = cross_val_score(
            Pipeline([("pre", self._preprocessor(X)), ("lr", base)]),
            X, y, cv=5, scoring="roc_auc")
        metrics["logreg_cv_auc"] = round(float(base_scores.mean()), 4)
        logger.info("Classifier metrics: %s", metrics)
        logger.info("\n%s", classification_report(yte, pred, zero_division=0))
        return metrics

    def train_personas(self, df: pd.DataFrame) -> dict:
        X = df.drop(columns=[TARGET]) if TARGET in df.columns else df.copy()
        pre = self._preprocessor(X)
        Xt = pre.fit_transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        km = KMeans(n_clusters=self.n_personas, random_state=self.seed, n_init=10)
        labels = km.fit_predict(Xt)
        self.persona = Pipeline([("pre", pre), ("km", km)])
        sil = round(float(silhouette_score(Xt, labels)), 4)
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        logger.info("Personas: k=%d, silhouette=%.4f, sizes=%s",
                    self.n_personas, sil, sizes)
        return {"silhouette": sil, "cluster_sizes": {int(k): int(v) for k, v in sizes.items()}}

    def save(self, out: str):
        Path(out).mkdir(parents=True, exist_ok=True)
        if self.clf:
            joblib.dump(self.clf, Path(out) / "preorder_classifier.joblib")
        if self.persona:
            joblib.dump(self.persona, Path(out) / "fan_persona_kmeans.joblib")
        logger.info("Saved models to %s", out)


def main():
    ap = argparse.ArgumentParser(description="GTA 6 Hype Survey analysis")
    ap.add_argument("--train", action="store_true", help="train models")
    ap.add_argument("--data", default=None, help="path to the Kaggle survey CSV")
    ap.add_argument("--output", default="artifacts", help="output dir for models")
    ap.add_argument("--personas", type=int, default=4, help="number of fan personas")
    args = ap.parse_args()

    model = HypeSurveyModel(n_personas=args.personas)
    df = model.load(args.data)
    logger.info("Dataset shape: %s | preorder rate: %.1f%%",
                df.shape, 100 * df[TARGET].mean())

    if args.train:
        clf_metrics = model.train_classifier(df)
        persona_metrics = model.train_personas(df)
        model.save(args.output)
        print("\n=== SUMMARY ===")
        print("Preorder classifier:", clf_metrics)
        print("Fan personas:", persona_metrics)


if __name__ == "__main__":
    main()
