"""Shared fixtures. Tests run against pure functions — no network, no database."""
import os
import sys

import pandas as pd
import pytest

# etl/ is a plain directory of modules, not an installed package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "etl"))


def make_country(cca2: str, name: str, region: str, subregion: str, area: float) -> dict:
    """A country record shaped like the mledoze/countries dataset."""
    return {
        "name": {"common": name, "official": f"Republic of {name}"},
        "cca2": cca2,
        "region": region,
        "subregion": subregion,
        "capital": [f"{name} City"],
        "area": area,
        "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
        "flag": "\U0001F1EA\U0001F1F8",
    }


@pytest.fixture
def valid_countries() -> list[dict]:
    """250 synthetic countries, enough to clear MIN_COUNTRIES."""
    regions = [("Europe", "Southern Europe"), ("Asia", "Eastern Asia"),
               ("Africa", "Western Africa"), ("Americas", "South America")]
    out = []
    for i in range(250):
        region, subregion = regions[i % len(regions)]
        # cca2 codes AA..JZ
        code = chr(ord("A") + i // 26) + chr(ord("A") + i % 26)
        out.append(make_country(code, f"Country{i}", region, subregion, area=1000.0 * (i + 1)))
    return out


@pytest.fixture
def indicators_df(valid_countries) -> pd.DataFrame:
    """
    A World Bank panel covering the fixture countries.

    30 years per country so that the correlation stage clears its minimum
    observation thresholds (30 per region, 20 per indicator pair).
    """
    codes = {
        "SP.POP.TOTL": "Population total",
        "SI.POV.GINI": "Gini index",
        "NY.GDP.PCAP.CD": "GDP per capita (USD)",
        "NY.GDP.MKTP.KD.ZG": "GDP annual growth (%)",
        "SP.DYN.LE00.IN": "Life expectancy at birth",
        "SL.UEM.TOTL.ZS": "Unemployment (%)",
        "SP.URB.TOTL.IN.ZS": "Urban population (%)",
        "SE.XPD.TOTL.GD.ZS": "Education expenditure (% GDP)",
        "NE.TRD.GNFS.ZS": "Trade (% GDP)",
        "FP.CPI.TOTL.ZG": "Inflation CPI (%)",
    }
    rows = []
    for i, c in enumerate(valid_countries):
        for year in range(1994, 2024):
            offset = year - 1994
            for code, label in codes.items():
                if code == "SP.POP.TOTL":
                    value = 1_000_000 * (i + 1) + 10_000 * offset
                elif code == "SI.POV.GINI":
                    value = 25.0 + (i % 20)
                else:
                    value = float((i % 50) + offset)
                rows.append({
                    "cca2": c["cca2"],
                    "indicator_code": code,
                    "indicator_name": label,
                    "year": year,
                    "value": value,
                })
    return pd.DataFrame(rows)
