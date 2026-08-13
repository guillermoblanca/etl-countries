"""
Standalone loader for PA.NUS.FCRF (exchange rate LCU/USD).
Run inside the etl container — bypasses REST Countries when it's down.

Usage:
  docker compose run --rm -v ${PWD}/scripts:/app/scripts etl python scripts/load_currency.py
"""
import os
import sys
import time
import logging
import requests
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("currency")

INDICATORS = {
    # Currency + PPP (existing)
    "PA.NUS.FCRF":       "Exchange rate (LCU per USD, annual avg)",
    "NY.GDP.PCAP.PP.CD": "GDP per capita PPP (current intl $)",
    "NE.CON.PRVT.PP.KD": "Household consumption per capita PPP (constant 2021 intl $)",
    "NY.GNP.PCAP.PP.CD": "GNI per capita PPP (current intl $)",
    "NY.GDP.PCAP.PP.KD": "GDP per capita PPP (constant 2021 intl $)",
    # NEW: Specific investment categories
    "MS.MIL.XPND.GD.ZS": "Military expenditure (% GDP)",
    "SH.XPD.GHED.GD.ZS": "Government health expenditure (% GDP)",
    "TX.VAL.TECH.MF.ZS": "High-technology exports (% of manufactured exports)",
    "IP.PAT.RESD":       "Patent applications, residents (count)",
    "IT.CEL.SETS.P2":    "Mobile cellular subscriptions (per 100 people)",
    # NEW: Services breakdown (% of commercial service exports)
    "BX.GSR.TRVL.ZS":    "Travel services exports (% commercial)",
    "BX.GSR.TRAN.ZS":    "Transport services exports (% commercial)",
    "BX.GSR.INSF.ZS":    "Insurance & financial services exports (% commercial)",
    "BX.GSR.GNFS.CD":    "Service exports total (BoP, current US$)",
    "BX.GSR.NFSV.CD":    "Commercial service exports (current US$)",
}
WB_URL_TPL = "https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=25000&mrv=60"


def fetch(code: str):
    url = WB_URL_TPL.format(code=code)
    logger.info("Fetching %s …", code)
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=180,
                                headers={"User-Agent": "etl-countries-pipeline/2.0"})
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2:
                raise RuntimeError(f"Unexpected response: {str(payload)[:120]}")
            return payload[1] or []
        except Exception as exc:
            logger.warning("  attempt %d/3 failed: %s", attempt + 1, exc)
            if attempt == 2:
                raise
            time.sleep(5)


def load_all():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )
    cur = conn.cursor()

    for code, name in INDICATORS.items():
        records = fetch(code)
        rows = []
        for entry in records:
            if entry.get("value") is None:
                continue
            iso2 = (entry.get("country") or {}).get("id", "")
            if len(iso2) != 2 or not iso2.isalpha():
                continue
            rows.append((iso2.upper(), code, name, int(entry["date"]), float(entry["value"])))

        cur.execute("DELETE FROM country_indicators WHERE indicator_code = %s", (code,))
        deleted = cur.rowcount
        cur.executemany(
            "INSERT INTO country_indicators (cca2, indicator_code, indicator_name, year, value) "
            "VALUES (%s, %s, %s, %s, %s)",
            rows,
        )
        logger.info("  %s → -%d / +%d rows", code, deleted, len(rows))
        conn.commit()

    cur.close()
    conn.close()
    logger.info("All %d indicators loaded.", len(INDICATORS))


if __name__ == "__main__":
    load_all()
