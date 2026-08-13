"""
Global macro context extraction — v2
=====================================
Local-first strategy:
  1. Read from /app/data/{series}.csv  (mounted host volume — fastest, no network)
  2. Fall back to FRED HTTPS download  (if file not present)
  3. Graceful NaN on any failure       (pipeline never blocked by external sources)

Series downloaded:
  FRED DCOILBRENTEU → precio_brent_avg      (daily oil prices → annual mean)
  FRED VIXCLS       → indice_vix_avg        (daily volatility → annual mean)
  FRED FEDFUNDS     → fed_funds_rate        (monthly rate → annual mean)
  FRED STLFSI4      → stress_financiero     (weekly stress index → annual mean)
  GPR (Caldara & Iacoviello) → geopolitical_risk_idx (monthly → annual mean)

Pre-download with:
  python scripts/download_fred.py   ← run from HOST, not inside Docker
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import requests
import pandas as pd

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "etl-countries-pipeline/2.0 (educational/research)"}
START_YEAR = 1965
DATA_DIR = Path(os.getenv("CONTEXT_DATA_DIR", "/app/data"))

# ── FRED ──────────────────────────────────────────────────────────────────────

FRED_SERIES: dict[str, tuple[str, str]] = {
    # col_name: (local_filename, FRED_URL)
    "precio_brent_avg": (
        "DCOILBRENTEU.csv",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU",
    ),
    "indice_vix_avg": (
        "VIXCLS.csv",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
    ),
    "fed_funds_rate": (
        "FEDFUNDS.csv",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
    ),
    "stress_financiero": (
        "STLFSI4.csv",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4",
    ),
}

GPR_LOCAL_FILENAME = "gpr_monthly.xls"
GPR_REMOTE_URLS = [
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_monthly_recent.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_web.xls",
]


# ── Core helpers ──────────────────────────────────────────────────────────────

def _parse_fred_csv(text: str, col_name: str) -> pd.Series:
    """
    Parse a FRED CSV string (DATE, VALUE) → annual mean Series indexed by year.
    Missing values encoded as '.' are converted to NaN.
    """
    df = pd.read_csv(
        io.StringIO(text),
        na_values=[".", ""],
        dtype_backend="numpy_nullable",
    )
    # FRED header: DATE,<SERIES_ID> — normalise second column
    df.columns = ["date", "value"]
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).set_index("date")

    annual = df["value"].resample("YE").mean()
    annual.index = annual.index.year
    annual.index.name = "anio"
    annual.name = col_name
    return annual[annual.index >= START_YEAR].dropna()


def _fred_series(col_name: str, local_file: str, remote_url: str) -> pd.Series:
    """Local-first: read file if present, else HTTP GET, else return empty Series."""
    local_path = DATA_DIR / local_file

    # ── 1. Local file ─────────────────────────────────────────────────────────
    if local_path.exists():
        try:
            text = local_path.read_text(encoding="utf-8")
            s = _parse_fred_csv(text, col_name)
            logger.info("FRED %-25s ← local (%d years)", col_name, len(s))
            return s
        except Exception as exc:
            logger.warning("Local file %s unreadable: %s — trying network", local_file, exc)

    # ── 2. Network fallback ───────────────────────────────────────────────────
    try:
        logger.info("FRED %-25s ← network %s", col_name, remote_url)
        resp = requests.get(remote_url, timeout=25, headers=HEADERS)
        resp.raise_for_status()
        s = _parse_fred_csv(resp.text, col_name)
        logger.info("FRED %-25s  → %d years", col_name, len(s))
        return s
    except Exception as exc:
        logger.warning(
            "FRED '%s' unavailable (%s). Run scripts/download_fred.py on host.",
            col_name, exc,
        )
        return pd.Series(dtype=float, name=col_name)


# ── GPR ───────────────────────────────────────────────────────────────────────

def _parse_gpr_excel(raw_bytes: bytes) -> pd.Series:
    """
    Parse GPR Excel bytes → annual mean Series.
    Handles two known file formats:
      A) Column 'month' (date-like) + headline GPR column
      B) Columns YEAR (int) + MONTH (int) + GPR column
    """
    df = pd.read_excel(io.BytesIO(raw_bytes))
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Reconstruct date index
    if "month" in df.columns and df["month"].dtype == object:
        df["_date"] = pd.to_datetime(df["month"], errors="coerce")
    elif "year" in df.columns and "month" in df.columns:
        df["_date"] = pd.to_datetime(
            df["year"].astype(str) + "-"
            + df["month"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    elif "date" in df.columns:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["_date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    # Locate headline GPR column
    # Priority: exact 'gpr' → 'gprh' → any col starting with 'gpr'
    if "gpr" in df.columns:
        gpr_col = "gpr"
    elif "gprh" in df.columns:
        gpr_col = "gprh"
    else:
        candidates = [c for c in df.columns if c.startswith("gpr")]
        if not candidates:
            raise ValueError(f"No GPR column found — available: {list(df.columns)}")
        gpr_col = candidates[0]

    df[gpr_col] = pd.to_numeric(df[gpr_col], errors="coerce")
    df = df.dropna(subset=["_date", gpr_col]).set_index("_date").sort_index()

    annual = df[gpr_col].resample("YE").mean()
    annual.index = annual.index.year
    annual.index.name = "anio"
    annual.name = "geopolitical_risk_idx"
    return annual[annual.index >= START_YEAR].dropna()


def _gpr_series() -> pd.Series:
    """Local-first GPR fetch with multi-URL network fallback."""
    local_path = DATA_DIR / GPR_LOCAL_FILENAME

    # ── 1. Local file ─────────────────────────────────────────────────────────
    if local_path.exists():
        try:
            s = _parse_gpr_excel(local_path.read_bytes())
            logger.info("GPR %-25s ← local (%d years)", "geopolitical_risk_idx", len(s))
            return s
        except Exception as exc:
            logger.warning("Local GPR file unreadable: %s — trying network", exc)

    # ── 2. Network fallback (multiple candidate URLs) ─────────────────────────
    for url in GPR_REMOTE_URLS:
        try:
            logger.info("GPR ← %s", url.split("/")[-1])
            resp = requests.get(url, timeout=30, headers=HEADERS)
            resp.raise_for_status()
            s = _parse_gpr_excel(resp.content)
            logger.info("GPR → %d annual observations", len(s))
            return s
        except Exception as exc:
            logger.debug("GPR candidate %s failed: %s", url, exc)

    logger.warning(
        "GPR Index unavailable from all sources. "
        "Run scripts/download_fred.py on host to pre-download."
    )
    return pd.Series(dtype=float, name="geopolitical_risk_idx")


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_fred_context() -> pd.DataFrame:
    """
    Fetches all FRED series and returns a wide DataFrame.
    columns: anio, precio_brent_avg, indice_vix_avg, fed_funds_rate, stress_financiero
    """
    series: dict[str, pd.Series] = {}
    for col, (local_file, remote_url) in FRED_SERIES.items():
        series[col] = _fred_series(col, local_file, remote_url)

    df = pd.DataFrame(series)
    df.index.name = "anio"
    return df[df.index >= START_YEAR].reset_index()


def fetch_gpr_index() -> pd.Series:
    return _gpr_series()


def fetch_global_context() -> pd.DataFrame:
    """
    Main entry point. Returns clean DataFrame for transform_global_context():
      columns: anio, precio_brent_avg, indice_vix_avg, fed_funds_rate,
               geopolitical_risk_idx, stress_financiero
    """
    fred_df = fetch_fred_context()
    gpr_s   = fetch_gpr_index()

    df = fred_df.set_index("anio")
    df["geopolitical_risk_idx"] = gpr_s

    ordered_cols = [
        "precio_brent_avg",
        "indice_vix_avg",
        "fed_funds_rate",
        "geopolitical_risk_idx",
        "stress_financiero",
    ]
    df = df.reindex(columns=ordered_cols)
    df = df[df.index >= START_YEAR].copy()

    coverage = {c: int(df[c].notna().sum()) for c in ordered_cols}
    logger.info(
        "Global context assembled: %d years | %s",
        len(df),
        "  ".join(f"{k.split('_')[0]}={v}" for k, v in coverage.items()),
    )
    return df.reset_index()
