import requests
import logging
import pandas as pd

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "etl-countries-pipeline/2.0"}

# ── Country reference data ────────────────────────────────────────────────────
# Source: mledoze/countries — the open dataset that the REST Countries API is
# built on top of. Read directly from the repo at a pinned tag instead of
# through the API, which gives us:
#   - no API key and no rate limit (v3.1 was deprecated in favour of a keyed v5)
#   - reproducible builds: the same tag always yields the same 250 countries
# Data licensed ODbL-1.0; attribution in README.
COUNTRIES_TAG = "v4.1.1"
COUNTRIES_URL = (
    f"https://raw.githubusercontent.com/mledoze/countries/{COUNTRIES_TAG}/countries.json"
)

# Fail loudly below this many records: the previous version of this pipeline
# silently accepted a 3-key API error payload as if it were the dataset.
MIN_COUNTRIES = 200
# Note: population and gini are deliberately NOT expected here. The deprecated API
# enriched the dataset with them; we take both from the World Bank instead, which
# makes them properly dated series rather than undated snapshots.
REQUIRED_COUNTRY_FIELDS = {"name", "cca2", "region", "subregion", "area", "flag"}

# ── World Bank ────────────────────────────────────────────────────────────────
WB_BASE = "https://api.worldbank.org/v2/country/all/indicator"
WB_YEARS = 60

WB_INDICATORS = {
    # ── Macro / welfare ───────────────────────────────────────────────────────
    "SP.POP.TOTL":       "Population total",
    "NY.GDP.PCAP.CD":    "GDP per capita (USD)",
    "NY.GDP.MKTP.KD.ZG": "GDP annual growth (%)",
    "SP.DYN.LE00.IN":    "Life expectancy at birth",
    "SL.UEM.TOTL.ZS":    "Unemployment (%)",
    "SP.URB.TOTL.IN.ZS": "Urban population (%)",
    "SE.XPD.TOTL.GD.ZS": "Education expenditure (% GDP)",
    "NE.TRD.GNFS.ZS":    "Trade (% GDP)",
    "FP.CPI.TOTL.ZG":    "Inflation CPI (%)",
    # ── Sectoral employment (% of total employment) ───────────────────────────
    "SL.SRV.EMPL.ZS":    "Employment in services (% total)",
    "SL.IND.EMPL.ZS":    "Employment in industry (% total)",
    "SL.AGR.EMPL.ZS":    "Employment in agriculture (% total)",
    # ── Sectoral value added (% of GDP) ──────────────────────────────────────
    "NV.SRV.TOTL.ZS":    "Services value added (% GDP)",
    "NV.IND.TOTL.ZS":    "Industry value added (% GDP)",
    "NV.AGR.TOTL.ZS":    "Agriculture value added (% GDP)",
    # ── Trade structure (producer vs consumer signals) ────────────────────────
    "NE.EXP.GNFS.ZS":    "Exports (% GDP)",
    "NE.IMP.GNFS.ZS":    "Imports (% GDP)",
    "BN.CAB.XOKA.GD.ZS": "Current account balance (% GDP)",
    "TX.VAL.FUEL.ZS.UN": "Fuel exports (% merchandise)",
    "TX.VAL.MANF.ZS.UN": "Manufactures exports (% merchandise)",
    # ── Financial strength (debt, reserves, FDI, investment) ──────────────────
    "GC.DOD.TOTL.GD.ZS": "Central govt debt (% GDP)",
    "DT.DOD.DECT.GN.ZS": "External debt stocks (% GNI)",
    "FI.RES.TOTL.MO":    "Reserves (months of imports)",
    "BX.KLT.DINV.WD.GD.ZS": "FDI net inflows (% GDP)",
    "NE.GDI.TOTL.ZS":    "Gross capital formation (% GDP)",
    # ── Innovation & development ──────────────────────────────────────────────
    "GB.XPD.RSDV.GD.ZS": "R&D expenditure (% GDP)",
    "IT.NET.USER.ZS":    "Internet users (% population)",
    # ── Demographics & inequality ─────────────────────────────────────────────
    "SP.DYN.TFRT.IN":    "Fertility rate (births/woman)",
    "SP.POP.DPND":       "Age dependency ratio (%)",
    "SI.POV.GINI":       "Gini index",
    # ── Energy & sustainability ───────────────────────────────────────────────
    "EG.FEC.RNEW.ZS":    "Renewable energy (% final consumption)",
    "EN.GHG.CO2.PC.CE.AR5": "CO2 emissions per capita (tons)",
    # ── Currency ──────────────────────────────────────────────────────────────
    "PA.NUS.FCRF":       "Exchange rate (LCU per USD, annual avg)",
    # ── Purchasing Power Parity (real living standards) ───────────────────────
    "NY.GDP.PCAP.PP.CD": "GDP per capita PPP (current intl $)",
    "NE.CON.PRVT.PP.KD": "Household consumption per capita PPP (constant 2021 intl $)",
    "NY.GNP.PCAP.PP.CD": "GNI per capita PPP (current intl $)",
    "NY.GDP.PCAP.PP.KD": "GDP per capita PPP (constant 2021 intl $)",
}


def fetch_countries() -> list[dict]:
    """
    Fetch country reference data and validate it before letting it into the pipeline.

    Raises ValueError rather than returning degraded data: a pipeline that loads
    3 countries without complaining is worse than one that stops.
    """
    logger.info("Fetching countries dataset: %s", COUNTRIES_URL)
    resp = requests.get(COUNTRIES_URL, timeout=60, headers=HEADERS)
    if not resp.ok:
        logger.error("Countries dataset error %s: %s", resp.status_code, resp.text[:300])
    resp.raise_for_status()
    data = resp.json()

    validate_countries(data)
    logger.info("Countries dataset → %d records (tag %s)", len(data), COUNTRIES_TAG)
    return data


def validate_countries(data) -> None:
    """Shape and volume checks on the raw country payload. Raises ValueError on failure."""
    if not isinstance(data, list):
        # An error envelope like {"success": false, "errors": [...]} is a dict, and
        # len() on it returns its key count — which is how a dead API once passed
        # for "3 records". Reject anything that is not a list outright.
        raise ValueError(
            f"Expected a list of countries, got {type(data).__name__}: "
            f"{str(data)[:200]}"
        )

    if len(data) < MIN_COUNTRIES:
        raise ValueError(
            f"Only {len(data)} countries returned, expected at least {MIN_COUNTRIES}. "
            "Upstream source is likely broken or truncated."
        )

    missing = REQUIRED_COUNTRY_FIELDS - set(data[0])
    if missing:
        raise ValueError(f"Country records are missing required fields: {sorted(missing)}")

    no_code = sum(1 for c in data if not c.get("cca2"))
    if no_code:
        raise ValueError(f"{no_code} country records have no cca2 code — cannot key the panel")


def fetch_indicators() -> pd.DataFrame:
    rows = []
    for code, name in WB_INDICATORS.items():
        url = f"{WB_BASE}/{code}?format=json&per_page=20000&mrv={WB_YEARS}"
        logger.info("Fetching World Bank: %s", code)
        # Retry with backoff — 60y windows are heavier
        last_exc = None
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=180, headers=HEADERS)
                resp.raise_for_status()
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                logger.warning("  attempt %d/3 timed out for %s — retrying", attempt + 1, code)
        else:
            logger.error("  giving up on %s after 3 attempts: %s", code, last_exc)
            continue

        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2:
            logger.warning("Unexpected response for %s", code)
            continue

        before = len(rows)
        for entry in (payload[1] or []):
            if entry.get("value") is None:
                continue
            iso2 = entry.get("country", {}).get("id", "")
            if len(iso2) != 2 or not iso2.isalpha():
                continue
            rows.append({
                "cca2":           iso2.upper(),
                "indicator_code": code,
                "indicator_name": name,
                "year":           int(entry["date"]),
                "value":          float(entry["value"]),
            })
        logger.info("  → %d rows for %s", len(rows) - before, code)

    df = pd.DataFrame(rows)
    logger.info("World Bank total → %d rows across %d indicators", len(df), len(WB_INDICATORS))
    return df
