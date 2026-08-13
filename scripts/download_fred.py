"""
download_fred.py — run ONCE from your HOST machine (not inside Docker).

Downloads FRED series (Brent, VIX, Fed Funds, Stress Index) and the
Caldara & Iacoviello GPR Index to the local data/ directory.
The ETL container reads from there via the ./data:/app/data volume mount.

Usage:
    cd D:\\DEV\\etl_countries
    python scripts/download_fred.py

Requirements: requests, pandas  (pip install requests pandas openpyxl)
"""

import os
import sys
import requests
import pandas as pd
from io import StringIO, BytesIO

# ── Config ────────────────────────────────────────────────────────────────────

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

FRED_SERIES: dict[str, str] = {
    "DCOILBRENTEU.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU",
    "VIXCLS.csv":       "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
    "FEDFUNDS.csv":     "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
    "STLFSI4.csv":      "https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4",
}

# Known URLs for GPR dataset — tried in order until one succeeds
GPR_CANDIDATE_URLS = [
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_monthly_recent.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_web.xls",
]


def sep(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def download_fred() -> int:
    sep("FRED Series (Brent · VIX · Fed Funds · Stress)")
    ok = 0
    for filename, url in FRED_SERIES.items():
        dest = os.path.join(DATA_DIR, filename)
        try:
            print(f"  ↓  {filename:35s} ", end="", flush=True)
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            with open(dest, "w", encoding="utf-8") as f:
                f.write(r.text)
            # Quick validation: should have at least 100 data rows
            df = pd.read_csv(StringIO(r.text), na_values=[".", ""])
            rows = df.shape[0]
            print(f"OK  ({rows:,} rows)")
            ok += 1
        except Exception as e:
            print(f"FAIL — {e}")
    return ok


def download_gpr() -> bool:
    sep("GPR Index — Caldara & Iacoviello")
    dest = os.path.join(DATA_DIR, "gpr_monthly.xls")
    for url in GPR_CANDIDATE_URLS:
        try:
            short = url.split("/")[-1]
            print(f"  ↓  {short:35s} ", end="", flush=True)
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            # Validate: readable as Excel with enough rows
            df = pd.read_excel(BytesIO(r.content))
            print(f"OK  ({df.shape[0]:,} rows, {df.shape[1]} cols)")
            print(f"      columns: {list(df.columns[:6])}")
            return True
        except Exception as e:
            print(f"FAIL — {e}")
    return False


def main() -> None:
    print("\n🌍  FRED + GPR Context Downloader")
    print(f"    Output directory: {DATA_DIR}")

    fred_ok = download_fred()
    gpr_ok  = download_gpr()

    sep("Summary")
    print(f"  FRED series:  {fred_ok}/{len(FRED_SERIES)} downloaded")
    print(f"  GPR index:    {'OK' if gpr_ok else 'FAILED (will run without GPR)'}")

    if fred_ok == 0 and not gpr_ok:
        print("\n  ⚠  Nothing downloaded. Check your internet connection.")
        sys.exit(1)

    print(f"\n  Files in data/:")
    for f in sorted(os.listdir(DATA_DIR)):
        size = os.path.getsize(os.path.join(DATA_DIR, f))
        print(f"    {f:40s}  {size/1024:6.1f} KB")

    print("\n  Next step: docker compose down -v && docker compose up --build")
    print("  The ETL will read these files from /app/data/ inside the container.\n")


if __name__ == "__main__":
    main()
