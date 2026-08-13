"""
Analytics Service — Historical Milestone Correlation
=====================================================
Pure pandas/numpy layer: no database calls, no FastAPI imports.
The endpoint handler (api/main.py) feeds raw query results; this
module computes and returns structured dicts.

Public API
----------
compute_milestone_correlation(country_code, hito, ind_rows, ctx_rows)
    → dict  (ready for JSON serialisation)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Label maps ────────────────────────────────────────────────────────────────

# Human-readable column names used in the correlation output
INDICATOR_DISPLAY: dict[str, str] = {
    "NY.GDP.PCAP.CD":    "GDP_per_capita_USD",
    "NY.GDP.MKTP.KD.ZG": "GDP_growth_pct",
    "SL.UEM.TOTL.ZS":    "Unemployment_pct",
    "FP.CPI.TOTL.ZG":    "Inflation_pct",
    "NE.TRD.GNFS.ZS":    "Trade_pct_GDP",
    "SP.DYN.LE00.IN":    "Life_expectancy",
    "SP.URB.TOTL.IN.ZS": "Urban_pop_pct",
    "SE.XPD.TOTL.GD.ZS": "Education_pct_GDP",
    "SP.POP.TOTL":       "Population",
    "SL.SRV.EMPL.ZS":    "Empl_Services_pct",
    "SL.IND.EMPL.ZS":    "Empl_Industry_pct",
    "SL.AGR.EMPL.ZS":    "Empl_Agriculture_pct",
    "NV.SRV.TOTL.ZS":    "VA_Services_pct_GDP",
    "NV.IND.TOTL.ZS":    "VA_Industry_pct_GDP",
    "NV.AGR.TOTL.ZS":    "VA_Agriculture_pct_GDP",
}

CONTEXT_DISPLAY: dict[str, str] = {
    "precio_brent_avg":      "Brent_USD",
    "indice_vix_avg":        "VIX_Fear_Index",
    "fed_funds_rate":        "FedFunds_pct",
    "geopolitical_risk_idx": "GPR_Index",
    "stress_financiero":     "FinancialStress_Idx",
}

# Indicators for which we compute YoY % change instead of using raw values
# (avoids spurious unit-level correlations)
YOY_CODES = {"NY.GDP.PCAP.CD", "SL.UEM.TOTL.ZS", "SP.POP.TOTL"}


# ── Feature builders ──────────────────────────────────────────────────────────

def _build_country_features(ind_rows: list[dict]) -> pd.DataFrame:
    """
    Pivot indicator rows to wide format and apply YoY % change for
    selected codes.  Returns DataFrame indexed by year.
    """
    if not ind_rows:
        return pd.DataFrame()

    df_long = pd.DataFrame(ind_rows)
    # Pivot: rows = years, columns = indicator codes
    wide = df_long.pivot_table(
        index="year", columns="indicator_code", values="value", aggfunc="mean"
    )
    wide.index.name = "anio"

    features: dict[str, pd.Series] = {}
    for code in wide.columns:
        label = INDICATOR_DISPLAY.get(code, code)
        if code in YOY_CODES:
            # Year-over-year percentage change highlights structural shifts
            features[f"Δ%_{label}"] = wide[code].pct_change() * 100
        else:
            features[label] = wide[code]

    return pd.DataFrame(features, index=wide.index)


def _build_context_features(ctx_rows: list[dict]) -> pd.DataFrame:
    """
    Convert global context rows to a wide DataFrame indexed by year.
    Renames columns to human-readable display labels.
    """
    if not ctx_rows:
        return pd.DataFrame()

    df = pd.DataFrame(ctx_rows).set_index("anio")
    df = df.rename(columns=CONTEXT_DISPLAY)
    # Drop the created_at column if it leaked through
    df = df[[c for c in df.columns if c in CONTEXT_DISPLAY.values()]]
    return df


# ── Correlation engine ────────────────────────────────────────────────────────

def _pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix; NaN columns are dropped before computation."""
    df_clean = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df_clean.corr(method="pearson")


def _top_cross_correlations(
    corr: pd.DataFrame,
    country_vars: list[str],
    context_vars: list[str],
    top_n: int = 10,
) -> list[dict]:
    """
    Extracts cross-correlations between country variables and global context
    variables, sorted by absolute value descending.
    """
    results: list[dict] = []
    for cv in country_vars:
        for gv in context_vars:
            if cv not in corr.index or gv not in corr.columns:
                continue
            r = corr.loc[cv, gv]
            if pd.isna(r):
                continue
            results.append({
                "variable_pais":   cv,
                "variable_global": gv,
                "pearson_r":       round(float(r), 4),
                "abs_r":           round(abs(float(r)), 4),
                "interpretacion":  _interpret_r(r),
            })
    results.sort(key=lambda x: x["abs_r"], reverse=True)
    # Remove helper field before returning
    for row in results:
        del row["abs_r"]
    return results[:top_n]


def _interpret_r(r: float) -> str:
    """Simple magnitude label for correlation coefficient."""
    a = abs(r)
    if a >= 0.8:
        direction = "muy fuerte"
    elif a >= 0.6:
        direction = "fuerte"
    elif a >= 0.4:
        direction = "moderada"
    elif a >= 0.2:
        direction = "débil"
    else:
        direction = "muy débil o nula"
    sign = "positiva" if r >= 0 else "negativa"
    return f"Correlación {direction} {sign}"


def _matrix_to_dict(corr: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Serialise a correlation matrix to a nested dict (NaN → None)."""
    return {
        col: {
            idx: (round(float(v), 4) if not pd.isna(v) else None)
            for idx, v in corr[col].items()
        }
        for col in corr.columns
    }


# ── Public entry point ────────────────────────────────────────────────────────

def compute_milestone_correlation(
    country_code: str,
    hito: dict,
    ind_rows: list[dict],
    ctx_rows: list[dict],
) -> dict[str, Any]:
    """
    Computes a Pearson correlation matrix between macroeconomic country
    variables and global context variables, scoped to the hito's date range.

    Parameters
    ----------
    country_code : str
        ISO 2-letter code (e.g. 'ES', 'SA').
    hito : dict
        Row from hitos_historicos: id, anio_inicio, anio_fin,
        nombre_evento, tipo_shock, descripcion.
    ind_rows : list[dict]
        Rows from country_indicators pre-filtered to [anio_inicio, anio_fin].
    ctx_rows : list[dict]
        Rows from contexto_global_anual pre-filtered to [anio_inicio, anio_fin].

    Returns
    -------
    dict ready for JSON serialisation.
    """
    year_start: int = hito["anio_inicio"]
    year_end:   int = hito["anio_fin"]

    # ── Build feature frames ──────────────────────────────────────────────────
    country_df = _build_country_features(ind_rows)
    context_df = _build_context_features(ctx_rows)

    if country_df.empty:
        raise ValueError(
            f"No indicator data for {country_code} in {year_start}–{year_end}"
        )

    # ── Merge on year index ───────────────────────────────────────────────────
    if not context_df.empty:
        combined = country_df.join(context_df, how="left")
    else:
        combined = country_df.copy()
        logger.warning("No global context data for period %d–%d", year_start, year_end)

    combined = (
        combined
        .dropna(axis=1, how="all")  # drop columns with zero observations
        .dropna(axis=0, how="all")  # drop years with no data at all
    )

    n_years = len(combined)
    if n_years < 2:
        raise ValueError(
            f"Insufficient observations ({n_years}) for period {year_start}–{year_end}. "
            "Correlation requires at least 2 data points."
        )

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr_matrix = _pearson_matrix(combined)

    # Classify columns into country vs context
    all_context_labels = set(CONTEXT_DISPLAY.values())
    country_vars = [c for c in corr_matrix.columns if c not in all_context_labels]
    context_vars = [c for c in corr_matrix.columns if c in all_context_labels]

    top_cross = _top_cross_correlations(corr_matrix, country_vars, context_vars)

    # Coverage: how many non-NaN values per variable
    coverage = {
        col: int(combined[col].notna().sum())
        for col in combined.columns
        if col in corr_matrix.columns
    }

    logger.info(
        "Correlation computed — country=%s hito=%d years=%d vars=%d cross_pairs=%d",
        country_code, hito["id"], n_years,
        len(corr_matrix.columns), len(top_cross),
    )

    return {
        "hito": {
            "id":           hito["id"],
            "nombre":       hito["nombre_evento"],
            "periodo":      f"{year_start}–{year_end}",
            "tipo_shock":   hito["tipo_shock"],
            "descripcion":  hito["descripcion"],
        },
        "pais":                  country_code.upper(),
        "n_anios_analizados":    n_years,
        "anios":                 sorted(combined.index.tolist()),
        "variables_pais":        country_vars,
        "variables_globales":    context_vars,
        "cobertura_datos":       coverage,
        "top_correlaciones":     top_cross,
        "matriz_correlacion":    _matrix_to_dict(corr_matrix),
    }
