import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

INDICATOR_LABELS = {
    # Macro / welfare
    "SP.POP.TOTL":       "Population",
    "NY.GDP.PCAP.CD":    "GDP/capita",
    "NY.GDP.MKTP.KD.ZG": "GDP growth %",
    "SP.DYN.LE00.IN":    "Life expect.",
    "SL.UEM.TOTL.ZS":    "Unemployment",
    "SP.URB.TOTL.IN.ZS": "Urban pop %",
    "SE.XPD.TOTL.GD.ZS": "Education %GDP",
    "NE.TRD.GNFS.ZS":    "Trade %GDP",
    "FP.CPI.TOTL.ZG":    "Inflation %",
    # Sectoral employment
    "SL.SRV.EMPL.ZS":    "Empl. Services",
    "SL.IND.EMPL.ZS":    "Empl. Industry",
    "SL.AGR.EMPL.ZS":    "Empl. Agric.",
    # Sectoral value added
    "NV.SRV.TOTL.ZS":    "VA Services",
    "NV.IND.TOTL.ZS":    "VA Industry",
    "NV.AGR.TOTL.ZS":    "VA Agric.",
    # Trade structure
    "NE.EXP.GNFS.ZS":    "Exports %GDP",
    "NE.IMP.GNFS.ZS":    "Imports %GDP",
    "BN.CAB.XOKA.GD.ZS": "Curr. acc. %GDP",
    "TX.VAL.FUEL.ZS.UN": "Fuel exports %",
    "TX.VAL.MANF.ZS.UN": "Manuf exports %",
    # Financial strength
    "GC.DOD.TOTL.GD.ZS": "Govt debt %GDP",
    "DT.DOD.DECT.GN.ZS": "Ext debt %GNI",
    "FI.RES.TOTL.MO":    "Reserves mo.imp",
    "BX.KLT.DINV.WD.GD.ZS": "FDI in %GDP",
    "NE.GDI.TOTL.ZS":    "Gross capital %GDP",
    # Innovation
    "GB.XPD.RSDV.GD.ZS": "R&D %GDP",
    "IT.NET.USER.ZS":    "Internet %pop",
    # Demographics / inequality
    "SP.DYN.TFRT.IN":    "Fertility",
    "SP.POP.DPND":       "Dependency %",
    "SI.POV.GINI":       "Gini",
    # Energy
    "EG.FEC.RNEW.ZS":    "Renewables %",
    "EN.GHG.CO2.PC.CE.AR5": "CO2/capita",
    # Currency
    "PA.NUS.FCRF":       "Exchange rate LCU/USD",
    # PPP / Purchasing power
    "NY.GDP.PCAP.PP.CD": "GDP/cap PPP",
    "NE.CON.PRVT.PP.KD": "Consumption/cap PPP",
    "NY.GNP.PCAP.PP.CD": "GNI/cap PPP",
    "NY.GDP.PCAP.PP.KD": "GDP/cap PPP (2021$)",
}

# Indicators that go through YoY % change in analytics (not stored as-is)
YOY_INDICATORS = {"NY.GDP.PCAP.CD", "SL.UEM.TOTL.ZS", "SP.POP.TOTL"}


def _get_currencies(country: dict) -> str:
    return ", ".join(
        f"{v.get('name', '')} ({code})"
        for code, v in country.get("currencies", {}).items()
    )


def _wb_summary(indicators_df: pd.DataFrame, valid_cca2: set) -> pd.DataFrame:
    """Latest value per country for selected indicators, plus population growth over 10y."""
    df = indicators_df[indicators_df["cca2"].isin(valid_cca2)].copy()

    def latest(code: str) -> pd.Series:
        sub = df[df["indicator_code"] == code]
        return sub.sort_values("year", ascending=False).groupby("cca2")["value"].first()

    # Population and Gini come from the World Bank rather than the country
    # reference dataset, which carries neither.
    pop_now  = latest("SP.POP.TOTL").rename("population")
    gini     = latest("SI.POV.GINI").rename("gini")
    gdp      = latest("NY.GDP.PCAP.CD").rename("latest_gdp_per_capita")
    gdp_gr   = latest("NY.GDP.MKTP.KD.ZG").rename("latest_gdp_growth")
    life     = latest("SP.DYN.LE00.IN").rename("latest_life_expectancy")
    unemp    = latest("SL.UEM.TOTL.ZS").rename("latest_unemployment")
    urban    = latest("SP.URB.TOTL.IN.ZS").rename("latest_urban_pct")
    educ     = latest("SE.XPD.TOTL.GD.ZS").rename("latest_education_pct")
    trade    = latest("NE.TRD.GNFS.ZS").rename("latest_trade_pct")
    inflat   = latest("FP.CPI.TOTL.ZG").rename("latest_inflation")

    # Population growth over 10 years
    pop = df[df["indicator_code"] == "SP.POP.TOTL"]
    if not pop.empty:
        max_yr = int(pop["year"].max())
        old_yr = max_yr - 10
        p_now  = pop[pop["year"] == max_yr].set_index("cca2")["value"]
        p_old  = pop[pop["year"] == old_yr].set_index("cca2")["value"]
        growth = ((p_now - p_old) / p_old * 100).round(2)
        growth.name = "population_growth_10y"
    else:
        growth = pd.Series(dtype=float, name="population_growth_10y")

    summary = pd.concat(
        [pop_now, gini, gdp, gdp_gr, life, unemp, urban, educ, trade, inflat, growth],
        axis=1,
    ).reset_index()
    summary.columns = [
        "cca2", "population", "gini",
        "latest_gdp_per_capita", "latest_gdp_growth",
        "latest_life_expectancy", "latest_unemployment",
        "latest_urban_pct", "latest_education_pct",
        "latest_trade_pct", "latest_inflation",
        "population_growth_10y",
    ]
    return summary


def _correlations(indicators_df: pd.DataFrame, countries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r between every pair of indicators within each region.
    Each (country, year) observation is one row — avoids spurious trend-vs-trend
    correlations caused by averaging series before computing r.
    """
    df = indicators_df.merge(countries_df[["cca2", "region"]], on="cca2", how="left")
    codes = list(INDICATOR_LABELS.keys())
    rows  = []

    for region in df["region"].dropna().unique():
        rdf = df[df["region"] == region]

        # One row per (country, year): correlate cross-sectional + time pooled
        pivot = (
            rdf.pivot_table(
                index=["cca2", "year"],
                columns="indicator_code",
                values="value",
                aggfunc="first",
            )
            .reindex(columns=codes)
        )

        if len(pivot) < 30:  # need enough country-year points
            continue

        corr = pivot.corr(method="pearson", min_periods=20)

        for i, ind_x in enumerate(codes):
            for ind_y in codes[i + 1:]:
                if ind_x not in corr.columns or ind_y not in corr.columns:
                    continue
                r = corr.loc[ind_x, ind_y]
                if pd.isna(r):
                    continue
                n = pivot[[ind_x, ind_y]].dropna().shape[0]
                rows.append({
                    "region":       region,
                    "indicator_x":  ind_x,
                    "label_x":      INDICATOR_LABELS[ind_x],
                    "indicator_y":  ind_y,
                    "label_y":      INDICATOR_LABELS[ind_y],
                    "correlation":  round(float(r), 6),
                    "n_obs":        int(n),
                })

    df_corr = pd.DataFrame(rows)
    logger.info("Correlations computed: %d pairs across %d regions", len(df_corr), df_corr["region"].nunique() if not df_corr.empty else 0)
    return df_corr


def transform(
    raw_countries: list[dict],
    indicators_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        countries_df    — one row per country
        region_stats_df — one row per region
        indicators_df   — filtered time series
        correlations_df — Pearson r pairs per region
    """
    # ── Countries base ────────────────────────────────────────────────────────
    rows = []
    for c in raw_countries:
        rows.append({
            "name":       c.get("name", {}).get("common", "Unknown"),
            "cca2":       c.get("cca2", "").upper(),
            "region":     c.get("region", "Unknown") or "Unknown",
            "subregion":  c.get("subregion", "") or "",
            "capital":    (c.get("capital") or [None])[0],
            "area":       c.get("area"),
            "currencies": _get_currencies(c),
            "flag_emoji": c.get("flag", ""),
        })

    df = pd.DataFrame(rows)

    # ── Cross-join World Bank summary ─────────────────────────────────────────
    # Brings in population and gini as well as the "latest_*" snapshot columns.
    wb = _wb_summary(indicators_df, set(df["cca2"]))
    df = df.merge(wb, on="cca2", how="left")
    df["population"] = df["population"].fillna(0).astype("int64")

    # Density needs both sources joined, so it is derived after the merge.
    df["population_density"] = np.where(
        df["area"].notna() & (df["area"] > 0),
        (df["population"] / df["area"]).round(4),
        np.nan,
    )

    # ── Rankings ──────────────────────────────────────────────────────────────
    df["global_population_rank"] = df["population"].rank(ascending=False, method="min").astype(int)
    df["global_density_rank"]    = df["population_density"].rank(ascending=False, method="min", na_option="bottom").astype(int)
    df["population_rank_region"] = df.groupby("region")["population"].rank(ascending=False, method="min").astype(int)
    df["density_rank_region"]    = df.groupby("region")["population_density"].rank(ascending=False, method="min", na_option="bottom").astype(int)

    # ── Region stats ──────────────────────────────────────────────────────────
    region_stats = (
        df.groupby("region")
        .agg(
            country_count=("name", "count"),
            total_population=("population", "sum"),
            total_area=("area", "sum"),
            avg_density=("population_density", "mean"),
            avg_gini=("gini", "mean"),
            avg_gdp_per_capita=("latest_gdp_per_capita", "mean"),
            avg_life_expectancy=("latest_life_expectancy", "mean"),
            avg_unemployment=("latest_unemployment", "mean"),
            avg_urban_pct=("latest_urban_pct", "mean"),
        )
        .reset_index()
    )
    for col in region_stats.select_dtypes("float").columns:
        region_stats[col] = region_stats[col].round(2)

    # ── Correlations ──────────────────────────────────────────────────────────
    known = set(df["cca2"])
    indicators_clean = indicators_df[indicators_df["cca2"].isin(known)].copy()
    correlations_df  = _correlations(indicators_clean, df[["cca2", "region"]])

    logger.info("Transform complete: %d countries, %d regions, %d indicator rows, %d correlation pairs",
                len(df), len(region_stats), len(indicators_clean), len(correlations_df))
    return df, region_stats, indicators_clean, correlations_df


def transform_global_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans the raw global context DataFrame produced by
    extract_global.fetch_global_context().

    - Coerces all numeric columns
    - Ensures 'anio' is integer
    - Rounds to 4 decimal places
    - Sorts by year ascending
    - Drops rows where every metric is NaN (year with no data at all)
    """
    df = df.copy()
    numeric_cols = [
        "precio_brent_avg",
        "indice_vix_avg",
        "fed_funds_rate",
        "geopolitical_risk_idx",
        "stress_financiero",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
        else:
            df[col] = np.nan

    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["anio"])
    df["anio"] = df["anio"].astype(int)

    # Drop rows where ALL metrics are missing
    df = df.dropna(subset=numeric_cols, how="all")
    df = df.sort_values("anio").reset_index(drop=True)

    logger.info(
        "Global context transformed: %d years — "
        "Brent=%d VIX=%d Fed=%d GPR=%d Stress=%d observations",
        len(df),
        df["precio_brent_avg"].notna().sum(),
        df["indice_vix_avg"].notna().sum(),
        df["fed_funds_rate"].notna().sum(),
        df["geopolitical_risk_idx"].notna().sum(),
        df["stress_financiero"].notna().sum(),
    )
    return df
