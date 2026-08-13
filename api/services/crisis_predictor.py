"""
Crisis predictor — Random Forest trained on 60 years of macro panel data.

Honest framing:
    This model does NOT predict the calendar date of the next crisis.
    It predicts the probability that a country's current macro pattern
    matches the pattern of countries historically classified as 'in crisis'
    (i.e., during a registered hito_historico window).

    High probability => the country's economy LOOKS LIKE a typical
    pre-crisis or crisis economy, statistically speaking.
    Treat output as a risk score, not as a forecast.

Methodology:
    - Train on (country, year) panel 1965-2014
    - Test on 2015-2024 (true out-of-sample, time-series split)
    - Class balance: in_crisis ≈ 54% positive on the training panel (class_weight='balanced')
      NOTE: the 12 historical shock windows cover a large share of the 1965-2024
      period, so "in crisis" is not a rare label. Read the metrics with that in
      mind — a majority-class baseline is already ~54%, not ~50%.
    - Features: 52 after one-hot encoding the archetype
    - Model: RandomForestClassifier (200 trees, max_depth=12)

Observed metrics (out-of-sample 2015-2024): ROC-AUC 0.868, precision 1.00,
recall 0.29, F1 0.45. The precision/recall asymmetry is the honest headline:
when the model flags a country it has been right, but it stays silent about
most crisis-like country-years. It is a conservative screen, not a detector.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


# Numeric features used for training (excludes IDs, categoricals, targets)
NUMERIC_FEATURES = [
    "gdp_pcap", "gdp_growth", "population", "life_exp", "urban_pct",
    "unemp", "inflation", "edu_pct",
    "srv_va", "ind_va", "agr_va",
    "srv_empl", "ind_empl", "agr_empl",
    "trade_pct", "exports_pct", "imports_pct", "curr_acc",
    "fuel_exp_pct", "manuf_exp_pct",
    "govt_debt", "ext_debt", "reserves_mo", "fdi_pct", "capital_form",
    "fertility", "dep_ratio", "gini", "rd_pct", "internet_pct",
    "renewables_pct", "co2_pcap",
    "brent", "vix", "fed", "gpr", "stress",
    "gdp_pcap_yoy", "fx_yoy",
    "gdp_growth_3y", "inflation_3y", "curr_acc_3y", "ext_debt_3y",
]
CATEGORICAL_FEATURES = ["archetype"]
TARGET = "in_crisis"
TRAIN_END_YEAR = 2014  # train 1965-2014, test 2015-2024


class CrisisPredictor:
    """Trains once at startup; serves predictions for all subsequent requests."""

    def __init__(self):
        self.model:    Optional[RandomForestClassifier] = None
        self.features: list[str]   = []
        self.metrics:  dict        = {}
        self.medians:  pd.Series   = None  # for NaN imputation at predict time

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> None:
        """Fit the Random Forest on (country, year) panel."""
        if df.empty:
            logger.warning("Empty training dataframe — predictor disabled")
            return

        # Drop rows without target
        df = df.dropna(subset=[TARGET]).copy()
        df[TARGET] = df[TARGET].astype(int)

        # Coerce numeric features to float
        for c in NUMERIC_FEATURES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # One-hot encode archetype
        df_arch = pd.get_dummies(df["archetype"].fillna("Unknown"), prefix="arch").astype(int)
        df_full = pd.concat([df[NUMERIC_FEATURES + ["year", TARGET]], df_arch], axis=1)

        # Impute NaN with column medians (computed on train set only — fold-safe)
        train_mask = df["year"] <= TRAIN_END_YEAR
        self.medians = df_full.loc[train_mask, NUMERIC_FEATURES].median()
        df_full[NUMERIC_FEATURES] = df_full[NUMERIC_FEATURES].fillna(self.medians)

        feature_cols = NUMERIC_FEATURES + list(df_arch.columns)
        self.features = feature_cols

        X_train = df_full.loc[train_mask, feature_cols]
        y_train = df_full.loc[train_mask, TARGET]
        X_test  = df_full.loc[~train_mask, feature_cols]
        y_test  = df_full.loc[~train_mask, TARGET]

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate on holdout
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        self.metrics = {
            "train_period":      f"1965-{TRAIN_END_YEAR}",
            "test_period":       f"{TRAIN_END_YEAR + 1}-2024",
            "train_samples":     int(len(y_train)),
            "test_samples":      int(len(y_test)),
            "train_positive_rate":  round(float(y_train.mean()), 3),
            "test_positive_rate":   round(float(y_test.mean()), 3),
            "auc_roc":           round(float(roc_auc_score(y_test, y_proba)), 3),
            "f1_score":          round(float(f1_score(y_test, y_pred)), 3),
            "precision":         round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
            "recall":            round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
            "n_features":        len(feature_cols),
            "model":             "RandomForestClassifier(200 trees, max_depth=12)",
        }
        logger.info("CrisisPredictor trained — AUC=%.3f F1=%.3f", self.metrics["auc_roc"], self.metrics["f1_score"])

    # ── Inference ─────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self.model is not None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same preprocessing as training."""
        for c in NUMERIC_FEATURES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df_arch = pd.get_dummies(df["archetype"].fillna("Unknown"), prefix="arch").astype(int)
        df_full = pd.concat([df[NUMERIC_FEATURES], df_arch], axis=1)
        # Add any missing one-hot columns (archetypes not seen in inference batch)
        for col in self.features:
            if col not in df_full.columns:
                df_full[col] = 0
        df_full[NUMERIC_FEATURES] = df_full[NUMERIC_FEATURES].fillna(self.medians)
        return df_full[self.features]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns array of crisis probabilities for each row."""
        if not self.is_ready():
            return np.array([])
        X = self._prepare(df)
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self, top_n: int = 15) -> list[dict]:
        """Top-N most important features sorted by Gini importance."""
        if not self.is_ready():
            return []
        imps = list(zip(self.features, self.model.feature_importances_))
        imps.sort(key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(i), 4)} for f, i in imps[:top_n]]
