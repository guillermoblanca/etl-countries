"""
Country clustering analyzer — K-Means on 16 economic/structural features.

Goal: find data-driven groups of countries with similar economic patterns,
complementing the rules-based archetype classification.

Methodology:
    - 27 features: GDP, growth, inflation, unemployment, sector VA/empl,
      trade balance, debt, investment, inequality, digitalisation, CO2,
      plus engineered volatility features and eurozone membership
    - StandardScaler (zero mean, unit variance)
    - K-Means with automatic K selection by silhouette score (range 3-8)
    - PCA to 2D for visualisation
    - Cluster characterisation: top distinguishing features per cluster

Observed fit: K=3 over 180 countries with silhouette 0.185. That is a weak
score — the economies form a continuum rather than well-separated groups, and
the clusters should be read as a coarse ordering, not as discovered species.
Reported rather than tuned away, since raising the score would mean dropping
features until the data agreed with the method.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

# np used for inf handling below
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Features chosen to capture economic structure + behaviour
CLUSTER_FEATURES = [
    # Original 16 — macro structure
    "gdp_pcap",          # income level
    "gdp_growth_3y",     # growth trend
    "inflation_3y",      # price stability
    "unemp",             # labour market
    "srv_va",            # services share
    "ind_va",            # industry share
    "agr_va",            # agriculture share
    "srv_empl",          # services employment
    "exports_pct",       # trade openness
    "imports_pct",
    "curr_acc",          # external balance
    "ext_debt",          # external vulnerability
    "capital_form",      # investment intensity
    "gini",              # inequality
    "internet_pct",      # digital development
    "co2_pcap",          # energy intensity
    # NEW: trade specialisation
    "fuel_exp_pct",      # petrostate signature
    "manuf_exp_pct",     # manufacturing signature
    # NEW: demographics
    "dep_ratio",         # aging vs young
    "fertility",         # demographic momentum
    # NEW: innovation & openness
    "rd_pct",            # innovation intensity
    "renewables_pct",    # energy transition
    "fdi_pct",           # capital openness
    # NEW: monetary regime (binary, added in fit())
    "is_eurozone",       # currency union membership
    # NEW: stability metrics (volatility, added in fit())
    "gdp_growth_vol",    # economic stability (stddev 10y)
    "inflation_vol",     # price predictability
    "fx_vol",            # currency stability
]


class ClusterAnalyzer:
    """K-Means clustering trained once at startup."""

    def __init__(self):
        self.kmeans:   Optional[KMeans]       = None
        self.scaler:   Optional[StandardScaler] = None
        self.pca:      Optional[PCA]          = None
        self.k:        int                    = 0
        self.silhouette: float                = 0.0
        self.assignments: dict[str, dict]     = {}  # cca2 → {cluster, pca_x, pca_y}
        self.centroids: list[dict]            = []  # one dict per cluster
        self.features: list[str]              = CLUSTER_FEATURES

    def fit(self, df: pd.DataFrame, eurozone_set: Optional[set] = None) -> None:
        """Train clustering with engineered features on latest row per country."""
        if df.empty:
            logger.warning("Empty df, clustering disabled")
            return

        # ── Engineer features that need the full time series ────────────────
        # Rolling 10-year volatility (stddev) per country
        for src_col, vol_col in [
            ("gdp_growth", "gdp_growth_vol"),
            ("inflation",  "inflation_vol"),
            ("fx_yoy",     "fx_vol"),
        ]:
            if src_col in df.columns:
                df[vol_col] = df.groupby("cca2")[src_col].transform(
                    lambda x: x.rolling(10, min_periods=3).std()
                )

        # Eurozone membership (binary)
        eurozone_set = eurozone_set or set()
        df["is_eurozone"] = df["cca2"].isin(eurozone_set).astype(int)

        # ── Keep latest row per country ─────────────────────────────────────
        df = df.sort_values(["cca2", "year"]).groupby("cca2").tail(1).copy()

        # Coerce numeric features
        for c in CLUSTER_FEATURES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop rows with too many NaNs (>10 missing of 27 features)
        df = df.dropna(subset=CLUSTER_FEATURES, thresh=len(CLUSTER_FEATURES) - 10)
        # Fill NaN with column medians; if a column is entirely NaN, use 0
        col_medians = df[CLUSTER_FEATURES].median()
        col_medians = col_medians.fillna(0)
        df[CLUSTER_FEATURES] = df[CLUSTER_FEATURES].fillna(col_medians)
        # Final safety: replace any remaining NaN/inf with 0
        df[CLUSTER_FEATURES] = df[CLUSTER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

        if len(df) < 20:
            logger.warning("Only %d countries with enough data for clustering", len(df))
            return

        X = df[CLUSTER_FEATURES].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Select K by silhouette score (3-8)
        best_k = 4
        best_score = -1.0
        for k in range(3, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            score = silhouette_score(X_scaled, km.labels_)
            logger.info("Cluster K=%d silhouette=%.3f", k, score)
            if score > best_score:
                best_k = k
                best_score = score
        self.k = best_k
        self.silhouette = round(float(best_score), 3)

        # Final model
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_scaled)
        labels = self.kmeans.labels_

        # PCA for 2D visualization
        self.pca = PCA(n_components=2, random_state=42).fit(X_scaled)
        pca_coords = self.pca.transform(X_scaled)

        # Build assignments
        df_out = df.assign(cluster=labels, pca_x=pca_coords[:, 0], pca_y=pca_coords[:, 1])
        for _, row in df_out.iterrows():
            self.assignments[row["cca2"]] = {
                "cluster": int(row["cluster"]),
                "pca_x":   float(row["pca_x"]),
                "pca_y":   float(row["pca_y"]),
            }

        # Characterise each cluster: feature means in original units + label hint
        centroids_scaled = self.kmeans.cluster_centers_
        centroids_orig = self.scaler.inverse_transform(centroids_scaled)
        # Mean feature values per cluster
        df_groups = df_out.groupby("cluster")
        for cid in range(best_k):
            members = df_out[df_out["cluster"] == cid]
            # Distinguishing features (Z-scores furthest from 0)
            z = pd.Series(centroids_scaled[cid], index=CLUSTER_FEATURES)
            top_high = z.nlargest(3)
            top_low  = z.nsmallest(3)
            features_dict = {f: round(float(centroids_orig[cid][i]), 2) for i, f in enumerate(CLUSTER_FEATURES)}
            self.centroids.append({
                "cluster_id":      int(cid),
                "n_members":       int(len(members)),
                "label":           self._auto_label(features_dict, cid),
                "centroid":        features_dict,
                "distinguishing_high": [{"feature": f, "z_score": round(float(z[f]), 2)} for f in top_high.index],
                "distinguishing_low":  [{"feature": f, "z_score": round(float(z[f]), 2)} for f in top_low.index],
            })
        logger.info("ClusterAnalyzer trained — K=%d, silhouette=%.3f, %d countries", self.k, self.silhouette, len(df))

    @staticmethod
    def _auto_label(features: dict, cid: int) -> str:
        """Auto-label cluster based on dominant features. Heuristic."""
        gdp        = features.get("gdp_pcap", 0)
        srv        = features.get("srv_va", 0)
        ind        = features.get("ind_va", 0)
        agr        = features.get("agr_va", 0)
        ext_debt   = features.get("ext_debt", 0)
        growth     = features.get("gdp_growth_3y", 0)
        inflation  = features.get("inflation_3y", 0)
        gini       = features.get("gini", 35)
        fuel       = features.get("fuel_exp_pct", 0)
        manuf      = features.get("manuf_exp_pct", 0)
        is_eurozone= features.get("is_eurozone", 0) >= 0.5
        infl_vol   = features.get("inflation_vol", 5)
        fx_vol     = features.get("fx_vol", 5)
        renew      = features.get("renewables_pct", 0)

        # Most specific labels first
        if fuel >= 35:
            return "Petrostate exporters"
        if inflation >= 20 or infl_vol >= 25:
            return "Hyper-volatile economies"
        if is_eurozone and gdp >= 25000:
            return "Eurozone core (currency-locked)"
        if is_eurozone:
            return "Eurozone periphery"
        if gdp >= 50000 and srv >= 70:
            return "High-income service economies"
        if gdp >= 30000 and manuf >= 60:
            return "Manufacturing advanced exporters"
        if gdp >= 30000 and ind >= 22:
            return "Industrial advanced economies"
        if gdp >= 15000 and growth >= 4:
            return "Emerging fast-growing"
        if agr >= 18:
            return "Agriculture-heavy economies"
        if ext_debt >= 50:
            return "Indebted emerging"
        if renew >= 50:
            return "Renewables-led developing"
        if 15000 <= gdp < 35000 and srv >= 55:
            return "Mid-income service-oriented"
        if 5000 <= gdp < 15000:
            return "Lower-middle income mixed"
        if gdp < 5000:
            return "Low-income developing"
        return f"Cluster {cid}"

    def is_ready(self) -> bool:
        return self.kmeans is not None and bool(self.assignments)

    def get_country(self, cca2: str) -> Optional[dict]:
        return self.assignments.get(cca2.upper())

    def get_summary(self) -> dict:
        return {
            "k":          self.k,
            "silhouette": self.silhouette,
            "n_countries":len(self.assignments),
            "features":   self.features,
            "centroids":  self.centroids,
            "pca_explained_variance": [round(float(v), 3) for v in self.pca.explained_variance_ratio_] if self.pca is not None else [],
        }
