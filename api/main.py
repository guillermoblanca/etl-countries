import os
import sys
import csv
import io
import decimal
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np
import psycopg2
import psycopg2.extras

# Allow importing from sibling services/ package
sys.path.insert(0, os.path.dirname(__file__))
from services.analytics import compute_milestone_correlation
from services.crisis_predictor import CrisisPredictor
from services.cluster_analyzer import ClusterAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Countries ETL API", version="4.0.0")

# ── ML models: trained once at startup ───────────────────────────────────────
predictor = CrisisPredictor()
clusterer = ClusterAnalyzer()


@app.on_event("startup")
def train_ml_models():
    """Train all ML models once when API boots."""
    try:
        import pandas as pd
        conn = _connect()
        df = pd.read_sql("SELECT * FROM ml_features_wide", conn)
        # Eurozone membership set for clustering feature
        cur = conn.cursor()
        cur.execute("SELECT cca2 FROM euro_adoption")
        eurozone_set = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
        # Crisis predictor
        predictor.train(df)
        logger.info("Crisis predictor ready. Metrics: %s", predictor.metrics)
        # Country clustering (with eurozone info + engineered volatility features)
        clusterer.fit(df, eurozone_set=eurozone_set)
        logger.info("Cluster analyzer ready. K=%d silhouette=%.3f (features=%d)",
                    clusterer.k, clusterer.silhouette, len(clusterer.features))
    except Exception as exc:
        logger.exception("ML model training failed: %s", exc)


# ── DB ────────────────────────────────────────────────────────────────────────

def _connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )


def _clean(row: dict) -> dict:
    return {k: float(v) if isinstance(v, decimal.Decimal) else v for k, v in row.items()}


def query(sql: str, params=None) -> list[dict]:
    conn = _connect()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params or ())
    rows = [_clean(dict(r)) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


# ── FX rate (for client-side currency conversion) ────────────────────────────

@app.get("/api/fx/eur-usd")
def fx_eur_usd():
    """Latest EUR/USD rate from any eurozone country's PA.NUS.FCRF series.
    Returns {rate: <EUR per 1 USD>, year: <int>}."""
    rows = query(
        """
        SELECT ci.year, ci.value AS rate
        FROM   country_indicators ci
        WHERE  ci.cca2 = 'ES'
          AND  ci.indicator_code = 'PA.NUS.FCRF'
          AND  ci.year >= 2000
          AND  ci.value IS NOT NULL
        ORDER  BY ci.year DESC
        LIMIT 1
        """
    )
    if not rows:
        raise HTTPException(500, "No FX data available")
    return {"rate": float(rows[0]["rate"]), "year": rows[0]["year"]}


# ── Countries ─────────────────────────────────────────────────────────────────

@app.get("/api/countries")
def get_countries(limit: int = 250):
    return query("SELECT * FROM countries ORDER BY global_population_rank LIMIT %s", (limit,))


@app.get("/api/countries/{cca2}")
def get_country(cca2: str):
    rows = query("SELECT * FROM countries WHERE cca2 = %s", (cca2.upper(),))
    if not rows:
        raise HTTPException(404, "Country not found")
    return rows[0]


# ── Regions ───────────────────────────────────────────────────────────────────

@app.get("/api/regions")
def get_regions():
    return query("SELECT * FROM region_stats ORDER BY total_population DESC")


@app.get("/api/regions/{region}/countries")
def get_region_countries(region: str):
    return query(
        "SELECT * FROM countries WHERE LOWER(region)=LOWER(%s) ORDER BY population_rank_region",
        (region,),
    )


# ── Static rankings ───────────────────────────────────────────────────────────

@app.get("/api/stats/top-population")
def top_population(limit: int = 10):
    return query("SELECT name,flag_emoji,region,population,global_population_rank FROM countries ORDER BY population DESC LIMIT %s", (limit,))


@app.get("/api/stats/top-density")
def top_density(limit: int = 10):
    return query("SELECT name,flag_emoji,region,population_density FROM countries WHERE population_density IS NOT NULL AND area>1 ORDER BY population_density DESC LIMIT %s", (limit,))


@app.get("/api/stats/top-gdp")
def top_gdp(limit: int = 10):
    return query("SELECT name,flag_emoji,region,latest_gdp_per_capita FROM countries WHERE latest_gdp_per_capita IS NOT NULL ORDER BY latest_gdp_per_capita DESC LIMIT %s", (limit,))


@app.get("/api/stats/top-life-expectancy")
def top_life_expectancy(limit: int = 10):
    return query("SELECT name,flag_emoji,region,latest_life_expectancy FROM countries WHERE latest_life_expectancy IS NOT NULL ORDER BY latest_life_expectancy DESC LIMIT %s", (limit,))


@app.get("/api/stats/population-growth")
def population_growth(limit: int = 10):
    return query("SELECT name,flag_emoji,region,population_growth_10y FROM countries WHERE population_growth_10y IS NOT NULL ORDER BY population_growth_10y DESC LIMIT %s", (limit,))


# ── Country ranking ───────────────────────────────────────────────────────────

_RANK_INDS = {
    "NY.GDP.PCAP.CD":    {"label": "GDP per capita", "unit": "USD", "higher_is_better": True},
    "NY.GDP.MKTP.KD.ZG": {"label": "GDP growth",    "unit": "%",   "higher_is_better": True},
    "SL.UEM.TOTL.ZS":    {"label": "Unemployment",  "unit": "%",   "higher_is_better": False},
    "FP.CPI.TOTL.ZG":    {"label": "Inflation",     "unit": "%",   "higher_is_better": False},
}


@app.get("/api/country/{cca2}/ranking")
def country_ranking(cca2: str):
    """Global and regional rank + % vs world/regional avg for key indicators."""
    result = {}
    for code, meta in _RANK_INDS.items():
        order = "DESC" if meta["higher_is_better"] else "ASC"
        rows = query(
            f"""
            WITH latest AS (
                SELECT MAX(year) AS yr FROM country_indicators WHERE indicator_code = %s
            ),
            ranked AS (
                SELECT
                    ci.cca2, ci.value, ci.year, c.region,
                    RANK() OVER (ORDER BY ci.value {order} NULLS LAST)                       AS global_rank,
                    COUNT(*) OVER ()                                                          AS global_total,
                    AVG(ci.value) OVER ()                                                     AS world_avg,
                    RANK() OVER (PARTITION BY c.region ORDER BY ci.value {order} NULLS LAST) AS region_rank,
                    COUNT(*) OVER (PARTITION BY c.region)                                     AS region_total,
                    AVG(ci.value) OVER (PARTITION BY c.region)                                AS region_avg
                FROM country_indicators ci
                JOIN countries c ON c.cca2 = ci.cca2
                WHERE ci.indicator_code = %s
                  AND ci.year = (SELECT yr FROM latest)
            )
            SELECT * FROM ranked WHERE cca2 = %s
            """,
            (code, code, cca2.upper()),
        )
        if rows:
            r = rows[0]
            val       = float(r["value"])      if r["value"]      is not None else None
            world_avg = float(r["world_avg"])  if r["world_avg"]  is not None else None
            reg_avg   = float(r["region_avg"]) if r["region_avg"] is not None else None
            result[code] = {
                **meta,
                "value":        val,
                "year":         r["year"],
                "global_rank":  int(r["global_rank"]),
                "global_total": int(r["global_total"]),
                "region_rank":  int(r["region_rank"]),
                "region_total": int(r["region_total"]),
                "region":       r["region"],
                "world_avg":    world_avg,
                "region_avg":   reg_avg,
                "pct_vs_world":  round((val / world_avg - 1) * 100, 1) if val and world_avg else None,
                "pct_vs_region": round((val / reg_avg   - 1) * 100, 1) if val and reg_avg   else None,
            }
    return result


# ── Country timeseries (all indicators, one call) ─────────────────────────────

@app.get("/api/country/{cca2}/timeseries")
def country_timeseries_all(cca2: str):
    rows = query(
        "SELECT indicator_code, indicator_name, year, value FROM country_indicators WHERE cca2=%s ORDER BY indicator_code, year",
        (cca2.upper(),),
    )
    result: dict = {}
    for row in rows:
        code = row["indicator_code"]
        result.setdefault(code, {"name": row["indicator_name"], "series": []})
        result[code]["series"].append({"year": row["year"], "value": row["value"]})
    return result


# ── Time series ───────────────────────────────────────────────────────────────

@app.get("/api/timeseries/regions/{indicator_code}")
def timeseries_regions(indicator_code: str):
    agg = "SUM" if indicator_code == "SP.POP.TOTL" else "AVG"
    rows = query(
        f"""
        SELECT c.region, ci.year, {agg}(ci.value) AS value
        FROM country_indicators ci
        JOIN countries c ON c.cca2 = ci.cca2
        WHERE ci.indicator_code = %s
        GROUP BY c.region, ci.year
        ORDER BY c.region, ci.year
        """,
        (indicator_code,),
    )
    result: dict = {}
    for row in rows:
        r = row["region"]
        result.setdefault(r, []).append({"year": row["year"], "value": row["value"]})
    return result


@app.get("/api/timeseries/world/{indicator_code}")
def timeseries_world(indicator_code: str):
    """Global annual average across all countries for an indicator."""
    agg = "SUM" if indicator_code == "SP.POP.TOTL" else "AVG"
    rows = query(
        f"SELECT year, {agg}(value) AS value FROM country_indicators WHERE indicator_code=%s GROUP BY year ORDER BY year",
        (indicator_code,),
    )
    return rows


@app.get("/api/timeseries/{cca2}/{indicator_code}")
def timeseries_country(cca2: str, indicator_code: str):
    rows = query(
        "SELECT year,value,indicator_name FROM country_indicators WHERE cca2=%s AND indicator_code=%s ORDER BY year",
        (cca2.upper(), indicator_code),
    )
    if not rows:
        raise HTTPException(404, "No data found")
    return rows


# ── Correlation & scatter ─────────────────────────────────────────────────────

@app.get("/api/correlations")
def get_correlations(region: str = Query(default=None)):
    if region:
        return query(
            "SELECT * FROM region_correlations WHERE LOWER(region)=LOWER(%s) ORDER BY ABS(correlation) DESC",
            (region,),
        )
    return query("SELECT * FROM region_correlations ORDER BY region, ABS(correlation) DESC")


@app.get("/api/correlations/matrix/{region}")
def correlation_matrix(region: str):
    """Returns a dict ready for heatmap rendering: {label_x: {label_y: r}}"""
    rows = query(
        "SELECT label_x, label_y, correlation FROM region_correlations WHERE LOWER(region)=LOWER(%s)",
        (region,),
    )
    matrix: dict = {}
    for row in rows:
        lx, ly, r = row["label_x"], row["label_y"], row["correlation"]
        matrix.setdefault(lx, {})[ly] = r
        matrix.setdefault(ly, {})[lx] = r
        matrix.setdefault(lx, {})[lx] = 1.0
        matrix.setdefault(ly, {})[ly] = 1.0
    return matrix


@app.get("/api/scatter")
def scatter(
    x: str = Query(default="SP.POP.TOTL"),
    y: str = Query(default="NY.GDP.PCAP.CD"),
):
    """Per-country data for scatter plot: latest value of two indicators + metadata."""
    col_map = {
        "SP.POP.TOTL":       "population",
        "NY.GDP.PCAP.CD":    "latest_gdp_per_capita",
        "NY.GDP.MKTP.KD.ZG": "latest_gdp_growth",
        "SP.DYN.LE00.IN":    "latest_life_expectancy",
        "SL.UEM.TOTL.ZS":    "latest_unemployment",
        "SP.URB.TOTL.IN.ZS": "latest_urban_pct",
        "SE.XPD.TOTL.GD.ZS": "latest_education_pct",
        "NE.TRD.GNFS.ZS":    "latest_trade_pct",
        "FP.CPI.TOTL.ZG":    "latest_inflation",
    }
    col_x = col_map.get(x)
    col_y = col_map.get(y)
    if not col_x or not col_y:
        raise HTTPException(400, "Unknown indicator code")

    return query(
        f"""
        SELECT name, flag_emoji, region, cca2,
               {col_x} AS x,
               {col_y} AS y
        FROM countries
        WHERE {col_x} IS NOT NULL AND {col_y} IS NOT NULL
        ORDER BY region, name
        """,
    )


# ── Analytics: crisis impact (country × hito) ────────────────────────────────

@app.get("/analytics/crisis-impact/{cca2}")
def crisis_impact(cca2: str):
    """
    Per-country impact summary for all historical milestones.
    Joins recovery metrics: pct_drop, years_to_recover, recovered flag.
    """
    rows = query(
        """
        SELECT ci.hito_id, ci.nombre_evento, ci.tipo_shock, ci.anio_inicio, ci.anio_fin,
               ci.gdp_growth_min, ci.gdp_growth_avg,
               ci.unemployment_max, ci.unemployment_avg,
               ci.inflation_max, ci.inflation_avg,
               ci.life_exp_delta, ci.urban_delta,
               cr.pre_shock_gdp, cr.trough_gdp, cr.pct_drop,
               cr.trough_year, cr.recovery_year, cr.years_to_recover, cr.recovered
        FROM   country_crisis_impact ci
        LEFT   JOIN country_crisis_recovery cr ON cr.cca2 = ci.cca2 AND cr.hito_id = ci.hito_id
        WHERE  ci.cca2 = %s
        ORDER  BY ci.anio_inicio
        """,
        (cca2.upper(),),
    )
    if not rows:
        raise HTTPException(404, f"No crisis impact data for country '{cca2}'")
    return rows


_DRILL_INDICATORS = [
    ("NY.GDP.MKTP.KD.ZG", "gdp_growth"),
    ("NY.GDP.PCAP.CD",    "gdp_per_capita"),
    ("SL.UEM.TOTL.ZS",    "unemployment"),
    ("FP.CPI.TOTL.ZG",    "inflation"),
]


@app.get("/analytics/crisis-impact/{cca2}/{hito_id}/detail")
def crisis_drill_down(cca2: str, hito_id: int):
    """
    Full drill-down for a single (country, milestone) pair.
    Returns:
      - hito metadata
      - country metadata
      - comparison: country vs region avg vs world avg (4 metrics + recovery)
      - rank_in_shock: country's position within affected countries
      - series: per-indicator yearly data for country, region, world + macro context
      - year_range: window for the chart (hito start-2 to end+2)
    """
    cca2 = cca2.upper()

    # ── Hito metadata ─────────────────────────────────────────────────────────
    hito_rows = query("SELECT * FROM hitos_historicos WHERE id = %s", (hito_id,))
    if not hito_rows:
        raise HTTPException(404, f"Hito {hito_id} not found")
    hito = hito_rows[0]
    y_start, y_end = hito["anio_inicio"], hito["anio_fin"]
    win_start, win_end = y_start - 2, y_end + 2

    # ── Country metadata ──────────────────────────────────────────────────────
    cou_rows = query(
        "SELECT cca2, name, region, flag_emoji, subregion, population FROM countries WHERE cca2 = %s",
        (cca2,),
    )
    if not cou_rows:
        raise HTTPException(404, f"Country '{cca2}' not found")
    country = cou_rows[0]
    region = country["region"]

    # ── Comparison: country impact ────────────────────────────────────────────
    cou_impact = query(
        "SELECT * FROM country_crisis_impact WHERE cca2 = %s AND hito_id = %s",
        (cca2, hito_id),
    )
    cou_recov = query(
        "SELECT * FROM country_crisis_recovery WHERE cca2 = %s AND hito_id = %s",
        (cca2, hito_id),
    )
    country_metrics = {
        **(cou_impact[0] if cou_impact else {}),
        **(cou_recov[0]  if cou_recov  else {}),
    } if (cou_impact or cou_recov) else None

    # ── Comparison: regional + world averages ─────────────────────────────────
    region_metrics = query(
        """
        SELECT AVG(ci.gdp_growth_min)     AS gdp_growth_min,
               AVG(ci.unemployment_max)   AS unemployment_max,
               AVG(ci.inflation_max)      AS inflation_max,
               AVG(cr.pct_drop)           AS pct_drop,
               AVG(cr.years_to_recover)   AS years_to_recover,
               COUNT(DISTINCT ci.cca2)    AS n_countries
        FROM   country_crisis_impact   ci
        LEFT   JOIN country_crisis_recovery cr ON cr.cca2 = ci.cca2 AND cr.hito_id = ci.hito_id
        WHERE  ci.hito_id = %s AND ci.region = %s AND ci.cca2 != %s
        """,
        (hito_id, region, cca2),
    )[0]
    world_metrics = query(
        """
        SELECT AVG(ci.gdp_growth_min)     AS gdp_growth_min,
               AVG(ci.unemployment_max)   AS unemployment_max,
               AVG(ci.inflation_max)      AS inflation_max,
               AVG(cr.pct_drop)           AS pct_drop,
               AVG(cr.years_to_recover)   AS years_to_recover,
               COUNT(DISTINCT ci.cca2)    AS n_countries
        FROM   country_crisis_impact   ci
        LEFT   JOIN country_crisis_recovery cr ON cr.cca2 = ci.cca2 AND cr.hito_id = ci.hito_id
        WHERE  ci.hito_id = %s
        """,
        (hito_id,),
    )[0]

    # ── Rank in shock (worst GDP drop, lowest recovery, etc.) ─────────────────
    rank = {}
    rank_rows = query(
        """
        WITH ranked AS (
            SELECT ci.cca2,
                   RANK() OVER (ORDER BY ci.gdp_growth_min ASC NULLS LAST)        AS rank_gdp,
                   RANK() OVER (ORDER BY ci.unemployment_max DESC NULLS LAST)     AS rank_unemp,
                   RANK() OVER (ORDER BY cr.years_to_recover DESC NULLS LAST)     AS rank_slow_recov,
                   COUNT(*) OVER ()                                                AS total
            FROM   country_crisis_impact   ci
            LEFT   JOIN country_crisis_recovery cr ON cr.cca2 = ci.cca2 AND cr.hito_id = ci.hito_id
            WHERE  ci.hito_id = %s
        )
        SELECT * FROM ranked WHERE cca2 = %s
        """,
        (hito_id, cca2),
    )
    if rank_rows:
        r = rank_rows[0]
        rank = {
            "worst_gdp_drop":      {"position": int(r["rank_gdp"]),        "total": int(r["total"])},
            "highest_unemployment":{"position": int(r["rank_unemp"]),      "total": int(r["total"])},
            "slowest_recovery":    {"position": int(r["rank_slow_recov"]), "total": int(r["total"])},
        }

    # ── Series: country / region / world per indicator over window ────────────
    series = {"country": {}, "region": {}, "world": {}}
    for code, key in _DRILL_INDICATORS:
        cou = query(
            "SELECT year, value FROM country_indicators WHERE cca2 = %s AND indicator_code = %s "
            "AND year BETWEEN %s AND %s ORDER BY year",
            (cca2, code, win_start, win_end),
        )
        reg = query(
            """
            SELECT ci.year, AVG(ci.value) AS value
            FROM   country_indicators ci JOIN countries c ON c.cca2 = ci.cca2
            WHERE  c.region = %s AND ci.indicator_code = %s
              AND  ci.year BETWEEN %s AND %s
            GROUP  BY ci.year ORDER BY ci.year
            """,
            (region, code, win_start, win_end),
        )
        wor = query(
            """
            SELECT year, AVG(value) AS value
            FROM   country_indicators
            WHERE  indicator_code = %s AND year BETWEEN %s AND %s
            GROUP  BY year ORDER BY year
            """,
            (code, win_start, win_end),
        )
        series["country"][key] = cou
        series["region"][key]  = reg
        series["world"][key]   = wor

    # ── Macro context for window ──────────────────────────────────────────────
    macro = query(
        "SELECT anio, precio_brent_avg, indice_vix_avg, fed_funds_rate, "
        "geopolitical_risk_idx, stress_financiero "
        "FROM contexto_global_anual WHERE anio BETWEEN %s AND %s ORDER BY anio",
        (win_start, win_end),
    )

    return {
        "hito":     hito,
        "country":  country,
        "comparison": {
            "country": country_metrics,
            "region":  region_metrics,
            "world":   world_metrics,
        },
        "rank_in_shock": rank,
        "series":   series,
        "macro_context": macro,
        "year_range": [win_start, win_end],
    }


@app.get("/analytics/crisis-impact/hito/{hito_id}")
def crisis_impact_by_hito(hito_id: int, limit: int = 20):
    """Countries most affected by a specific milestone (worst GDP drop)."""
    return query(
        """
        SELECT cca2, name, region, flag_emoji,
               gdp_growth_min, unemployment_max, inflation_max
        FROM   country_crisis_impact
        WHERE  hito_id = %s
        ORDER  BY gdp_growth_min ASC NULLS LAST
        LIMIT  %s
        """,
        (hito_id, limit),
    )


# ── Analytics: economic strength score ───────────────────────────────────────

@app.get("/analytics/strength")
def strength_ranking(
    region: str = Query(default=None),
    limit:  int = Query(default=250),
    order:  str = Query(default="desc", regex="^(asc|desc)$"),
):
    """
    Composite economic strength ranking. 4 drivers: level, stability,
    resilience, diversification — weighted 35/25/20/20.
    """
    direction = "DESC" if order == "desc" else "ASC"
    sql = f"""
        SELECT cca2, name, region, flag_emoji, latest_gdp_per_capita,
               driver_level, driver_stability, driver_resilience, driver_diversification,
               strength_score
        FROM   country_economic_strength
        {{where}}
        ORDER  BY strength_score {direction} NULLS LAST
        LIMIT  %s
    """
    if region:
        return query(sql.format(where="WHERE LOWER(region) = LOWER(%s)"), (region, limit))
    return query(sql.format(where=""), (limit,))


@app.get("/analytics/strength/{cca2}")
def strength_country(cca2: str):
    """Full strength breakdown for a single country."""
    rows = query(
        """
        SELECT cca2, name, region, flag_emoji, latest_gdp_per_capita,
               driver_level, driver_stability, driver_resilience, driver_diversification,
               strength_score,
               (SELECT COUNT(*) FROM country_economic_strength) AS total,
               (SELECT COUNT(*)
                FROM   country_economic_strength s2
                WHERE  s2.strength_score > es.strength_score) + 1 AS global_rank
        FROM   country_economic_strength es
        WHERE  cca2 = %s
        """,
        (cca2.upper(),),
    )
    if not rows:
        raise HTTPException(404, f"No strength score for country '{cca2}'")
    return rows[0]


# ── Analytics: macro-story (Etapa D1) ────────────────────────────────────────

@app.get("/analytics/macro-story/{hito_id}")
def macro_story(hito_id: int, n: int = Query(default=10, ge=1, le=30)):
    """
    Global narrative for a historical milestone:
    - hito metadata
    - top-N winners & losers (ranking change in world GDP/cap)
    - breakdown by archetype: which economic types gained / lost on average
    - macro context averages during the window (Brent, VIX, GPR, Fed, Stress)
    """
    hito = query("SELECT * FROM hitos_historicos WHERE id = %s", (hito_id,))
    if not hito:
        raise HTTPException(404, f"Hito {hito_id} not found")
    h = hito[0]

    # Genuine winners only — exclude rebounds, low-income noise and micro-states (<500k pop)
    winners = query(
        """
        SELECT w.cca2, w.name, w.region, w.flag_emoji,
               w.rank_pre, w.rank_post, w.rank_change, w.gdp_change_pct, w.verdict,
               w.pct_change_5y_prior, w.is_rebound, w.is_low_income
        FROM   crisis_winners_losers w
        JOIN   countries c ON c.cca2 = w.cca2
        WHERE  w.hito_id = %s
          AND  w.verdict IN ('Big Winner', 'Winner')
          AND  c.population >= 500000
        ORDER  BY w.rank_change DESC
        LIMIT  %s
        """,
        (hito_id, n),
    )
    losers = query(
        """
        SELECT w.cca2, w.name, w.region, w.flag_emoji,
               w.rank_pre, w.rank_post, w.rank_change, w.gdp_change_pct, w.verdict,
               w.pct_change_5y_prior, w.is_rebound, w.is_low_income
        FROM   crisis_winners_losers w
        JOIN   countries c ON c.cca2 = w.cca2
        WHERE  w.hito_id = %s
          AND  w.verdict IN ('Big Loser', 'Loser')
          AND  c.population >= 500000
        ORDER  BY w.rank_change ASC
        LIMIT  %s
        """,
        (hito_id, n),
    )
    # Rebounds shown separately for transparency
    rebounds = query(
        """
        SELECT cca2, name, region, flag_emoji,
               rank_pre, rank_post, rank_change, gdp_change_pct,
               pct_change_5y_prior, gdp_pre, gdp_5y_before_pre
        FROM   crisis_winners_losers
        WHERE  hito_id = %s AND verdict = 'Rebound'
        ORDER  BY rank_change DESC
        LIMIT  %s
        """,
        (hito_id, n),
    )

    by_archetype = query(
        """
        SELECT a.archetype,
               COUNT(*)                          AS n_countries,
               ROUND(AVG(w.rank_change)::numeric, 1)   AS avg_rank_change,
               ROUND(AVG(w.gdp_change_pct)::numeric, 1) AS avg_gdp_change_pct
        FROM   crisis_winners_losers w
        JOIN   country_archetype     a ON a.cca2 = w.cca2
        WHERE  w.hito_id = %s
        GROUP  BY a.archetype
        ORDER  BY avg_rank_change DESC NULLS LAST
        """,
        (hito_id,),
    )

    by_region = query(
        """
        SELECT region,
               COUNT(*)                                 AS n_countries,
               ROUND(AVG(rank_change)::numeric, 1)      AS avg_rank_change,
               ROUND(AVG(gdp_change_pct)::numeric, 1)   AS avg_gdp_change_pct
        FROM   crisis_winners_losers
        WHERE  hito_id = %s
        GROUP  BY region
        ORDER  BY avg_rank_change DESC NULLS LAST
        """,
        (hito_id,),
    )

    macro = query(
        """
        SELECT ROUND(AVG(precio_brent_avg)::numeric, 1)      AS brent_avg,
               ROUND(AVG(indice_vix_avg)::numeric, 1)        AS vix_avg,
               ROUND(AVG(fed_funds_rate)::numeric, 2)        AS fed_avg,
               ROUND(AVG(geopolitical_risk_idx)::numeric, 1) AS gpr_avg,
               ROUND(AVG(stress_financiero)::numeric, 2)     AS stress_avg
        FROM   contexto_global_anual
        WHERE  anio BETWEEN %s AND %s
        """,
        (h["anio_inicio"], h["anio_fin"]),
    )

    summary = query(
        """
        SELECT COUNT(*)                                                    AS total,
               COUNT(*) FILTER (WHERE verdict IN ('Winner','Big Winner'))  AS winners_count,
               COUNT(*) FILTER (WHERE verdict IN ('Loser','Big Loser'))    AS losers_count,
               COUNT(*) FILTER (WHERE verdict = 'Stable')                  AS stable_count
        FROM   crisis_winners_losers
        WHERE  hito_id = %s
        """,
        (hito_id,),
    )

    return {
        "hito":         h,
        "winners":      winners,
        "losers":       losers,
        "rebounds":     rebounds,
        "by_archetype": by_archetype,
        "by_region":    by_region,
        "macro_avg":    macro[0] if macro else None,
        "summary":      summary[0] if summary else None,
    }


# ── Analytics: country profile (Etapa D2) ────────────────────────────────────

@app.get("/analytics/country/{cca2}/profile")
def country_profile(cca2: str):
    """Full country profile: archetype, convergence trajectory, vulnerability."""
    cca2 = cca2.upper()
    arche = query("SELECT * FROM country_archetype WHERE cca2 = %s", (cca2,))
    conv  = query("SELECT * FROM convergence_metrics WHERE cca2 = %s", (cca2,))
    vuln  = query("SELECT * FROM external_vulnerability_index WHERE cca2 = %s", (cca2,))
    decades = query(
        "SELECT decade, gdp_per_capita, gdp_growth, unemployment, inflation, "
        "services_va, industry_va, current_account, gini, govt_debt "
        "FROM decade_summary WHERE cca2 = %s ORDER BY decade",
        (cca2,),
    )
    return {
        "archetype":     arche[0]   if arche else None,
        "convergence":   conv[0]    if conv  else None,
        "vulnerability": vuln[0]    if vuln  else None,
        "decades":       decades,
    }


# ── ML Lab: ready-to-use datasets and analyses ───────────────────────────────

@app.get("/ml/features.csv")
def ml_features_csv(
    start_year: int = Query(default=1965, ge=1965, le=2024),
    end_year:   int = Query(default=2024, ge=1965, le=2024),
    region:     str = Query(default=None),
):
    """Stream wide-format CSV ready for pandas. ~40 numeric features + target labels."""
    sql = "SELECT * FROM ml_features_wide WHERE year BETWEEN %s AND %s"
    params = [start_year, end_year]
    if region:
        sql += " AND region = %s"; params.append(region)
    sql += " ORDER BY cca2, year"

    conn = _connect()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close(); conn.close()

    if not rows:
        raise HTTPException(404, "No data for those filters")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ('' if v is None else (float(v) if isinstance(v, decimal.Decimal) else v)) for k,v in r.items()})
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="ml_features_{start_year}_{end_year}.csv"'},
    )


@app.get("/ml/features/schema")
def ml_features_schema():
    """Documents every column in ml_features_wide for ML practitioners."""
    rows = query(
        """
        SELECT column_name, data_type
        FROM   information_schema.columns
        WHERE  table_name = 'ml_features_wide'
        ORDER  BY ordinal_position
        """
    )
    descriptions = {
        "cca2":"ISO-2 country code (primary key part 1)", "year":"Year (primary key part 2)",
        "gdp_pcap":"GDP per capita, USD nominal","gdp_growth":"GDP annual growth (%)",
        "population":"Total population","life_exp":"Life expectancy at birth",
        "urban_pct":"Urban population (% of total)","unemp":"Unemployment rate (%)",
        "inflation":"Consumer price inflation YoY (%)","edu_pct":"Education spending (% GDP)",
        "srv_va":"Services value-added (% GDP)","ind_va":"Industry value-added (% GDP)","agr_va":"Agriculture value-added (% GDP)",
        "srv_empl":"Services employment (% of total)","ind_empl":"Industry employment (%)","agr_empl":"Agriculture employment (%)",
        "trade_pct":"Trade (% GDP)","exports_pct":"Exports (% GDP)","imports_pct":"Imports (% GDP)",
        "curr_acc":"Current account (% GDP)","fuel_exp_pct":"Fuel exports (% merchandise)","manuf_exp_pct":"Manufactures exports (% merch)",
        "govt_debt":"Central government debt (% GDP)","ext_debt":"External debt stocks (% GNI)",
        "reserves_mo":"Reserves in months of imports","fdi_pct":"FDI net inflows (% GDP)","capital_form":"Gross capital formation (% GDP)",
        "fertility":"Fertility rate (births/woman)","dep_ratio":"Age dependency ratio (%)",
        "gini":"Gini index of inequality","rd_pct":"R&D spending (% GDP)","internet_pct":"Internet users (%)",
        "renewables_pct":"Renewable energy (% consumption)","co2_pcap":"CO2 emissions per capita (tonnes)",
        "fx_lcu_usd":"Exchange rate LCU per USD",
        "brent":"Brent crude price avg ($/bbl) — global context","vix":"VIX volatility index avg","fed":"Fed Funds rate avg (%)",
        "gpr":"Geopolitical Risk Index avg","stress":"St. Louis Financial Stress Index avg",
        "gdp_pcap_yoy":"Engineered: GDP per capita YoY % change",
        "fx_yoy":"Engineered: exchange rate YoY % change (>15% = devaluation)",
        "gdp_growth_3y":"Engineered: 3-year rolling mean of GDP growth",
        "inflation_3y":"Engineered: 3-year rolling mean of inflation",
        "curr_acc_3y":"Engineered: 3-year rolling mean of current account",
        "ext_debt_3y":"Engineered: 3-year rolling mean of external debt",
        "region":"Geographic region (categorical)","subregion":"Subregion (categorical)",
        "archetype":"Economic archetype derived (categorical)",
        "in_crisis":"TARGET: True if year is within any historical milestone window",
        "crisis_start_year":"TARGET: True if year is the start year of any milestone",
        "active_hito_id":"TARGET: ID of the active historical milestone (NULL = no crisis)",
    }
    return [{**r, "description": descriptions.get(r["column_name"], "—")} for r in rows]


@app.get("/ml/recovery-trends")
def ml_recovery_trends():
    """Time-series of crisis recovery time + linear trend analysis."""
    rows = query("SELECT * FROM ml_recovery_trend ORDER BY year")
    # Linear regression: is recovery getting faster?
    trend = query(
        """
        SELECT
            ROUND(REGR_SLOPE(avg_years_to_recover, year)::numeric, 4)     AS slope,
            ROUND(REGR_INTERCEPT(avg_years_to_recover, year)::numeric, 2) AS intercept,
            ROUND(REGR_R2(avg_years_to_recover, year)::numeric, 3)        AS r_squared,
            ROUND(AVG(avg_years_to_recover) FILTER (WHERE year < 2000)::numeric, 2)  AS pre_2000_avg,
            ROUND(AVG(avg_years_to_recover) FILTER (WHERE year >= 2000)::numeric, 2) AS post_2000_avg,
            COUNT(*) AS n_crises
        FROM ml_recovery_trend
        """
    )[0]
    return {"crises": rows, "trend": trend}


@app.get("/ml/crisis-predictors")
def ml_crisis_predictors(hito_id: int = Query(6)):
    """Pearson correlations between pre-crisis indicators (5y window) and crisis severity."""
    hito = query("SELECT * FROM hitos_historicos WHERE id = %s", (hito_id,))
    if not hito:
        raise HTTPException(404, "Hito not found")
    h = hito[0]
    pre_start = h["anio_inicio"] - 5
    pre_end   = h["anio_inicio"] - 1

    # 14 candidate predictors
    correlations = query(
        """
        WITH pre AS (
            SELECT cca2,
                AVG(value) FILTER (WHERE indicator_code='DT.DOD.DECT.GN.ZS') AS ext_debt,
                AVG(value) FILTER (WHERE indicator_code='GC.DOD.TOTL.GD.ZS') AS govt_debt,
                AVG(value) FILTER (WHERE indicator_code='BN.CAB.XOKA.GD.ZS') AS curr_acc,
                AVG(value) FILTER (WHERE indicator_code='FI.RES.TOTL.MO')    AS reserves,
                AVG(value) FILTER (WHERE indicator_code='NE.TRD.GNFS.ZS')    AS trade_open,
                AVG(value) FILTER (WHERE indicator_code='FP.CPI.TOTL.ZG')    AS inflation,
                AVG(value) FILTER (WHERE indicator_code='NY.GDP.MKTP.KD.ZG') AS gdp_growth,
                AVG(value) FILTER (WHERE indicator_code='NV.SRV.TOTL.ZS')    AS srv_va,
                AVG(value) FILTER (WHERE indicator_code='NV.IND.TOTL.ZS')    AS ind_va,
                AVG(value) FILTER (WHERE indicator_code='SL.UEM.TOTL.ZS')    AS unemp,
                AVG(value) FILTER (WHERE indicator_code='NE.GDI.TOTL.ZS')    AS capital_form,
                AVG(value) FILTER (WHERE indicator_code='BX.KLT.DINV.WD.GD.ZS') AS fdi,
                AVG(value) FILTER (WHERE indicator_code='TX.VAL.FUEL.ZS.UN') AS fuel_exp,
                AVG(value) FILTER (WHERE indicator_code='TX.VAL.MANF.ZS.UN') AS manuf_exp
            FROM country_indicators
            WHERE year BETWEEN %s AND %s
            GROUP BY cca2
        ),
        target AS (
            SELECT cca2, gdp_growth_min, unemployment_max
            FROM country_crisis_impact WHERE hito_id = %s
        )
        SELECT
            ROUND(CORR(pre.ext_debt,     t.gdp_growth_min)::numeric, 3) AS ext_debt,
            ROUND(CORR(pre.govt_debt,    t.gdp_growth_min)::numeric, 3) AS govt_debt,
            ROUND(CORR(pre.curr_acc,     t.gdp_growth_min)::numeric, 3) AS curr_acc,
            ROUND(CORR(pre.reserves,     t.gdp_growth_min)::numeric, 3) AS reserves,
            ROUND(CORR(pre.trade_open,   t.gdp_growth_min)::numeric, 3) AS trade_open,
            ROUND(CORR(pre.inflation,    t.gdp_growth_min)::numeric, 3) AS inflation,
            ROUND(CORR(pre.gdp_growth,   t.gdp_growth_min)::numeric, 3) AS gdp_growth_pre,
            ROUND(CORR(pre.srv_va,       t.gdp_growth_min)::numeric, 3) AS srv_va,
            ROUND(CORR(pre.ind_va,       t.gdp_growth_min)::numeric, 3) AS ind_va,
            ROUND(CORR(pre.unemp,        t.gdp_growth_min)::numeric, 3) AS unemp,
            ROUND(CORR(pre.capital_form, t.gdp_growth_min)::numeric, 3) AS capital_form,
            ROUND(CORR(pre.fdi,          t.gdp_growth_min)::numeric, 3) AS fdi,
            ROUND(CORR(pre.fuel_exp,     t.gdp_growth_min)::numeric, 3) AS fuel_exp,
            ROUND(CORR(pre.manuf_exp,    t.gdp_growth_min)::numeric, 3) AS manuf_exp
        FROM pre JOIN target t ON t.cca2 = pre.cca2
        """,
        (pre_start, pre_end, hito_id),
    )[0]
    # Sort by |r|
    items = [(k, float(v) if v else 0.0) for k, v in correlations.items() if v is not None]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return {
        "hito":           h,
        "pre_crisis_window": [pre_start, pre_end],
        "target":         "gdp_growth_min (worse drop = more negative)",
        "predictors_ranked": [{"feature": k, "pearson_r": v, "strength": "strong" if abs(v) >= 0.3 else "moderate" if abs(v) >= 0.15 else "weak"} for k, v in items],
    }


@app.get("/ml/correlation-matrix")
def ml_correlation_matrix(year_start: int = Query(2000), year_end: int = Query(2024)):
    """Full correlation matrix between numeric features (Pearson r) for a year window."""
    # 25 columns × 25 rows = 625 correlations
    cols = [
        "gdp_pcap","gdp_growth","inflation","unemp","life_exp","urban_pct","edu_pct",
        "srv_va","ind_va","agr_va","trade_pct","exports_pct","curr_acc",
        "govt_debt","ext_debt","reserves_mo","fdi_pct","capital_form",
        "fertility","dep_ratio","gini","rd_pct","internet_pct","co2_pcap","fx_yoy",
    ]
    matrix = {}
    for c1 in cols:
        matrix[c1] = {}
        for c2 in cols:
            if c1 == c2:
                matrix[c1][c2] = 1.0
                continue
            r = query(
                f"SELECT ROUND(CORR({c1}, {c2})::numeric, 3) AS r FROM ml_features_wide "
                f"WHERE year BETWEEN %s AND %s AND {c1} IS NOT NULL AND {c2} IS NOT NULL",
                (year_start, year_end),
            )
            matrix[c1][c2] = float(r[0]["r"]) if r and r[0]["r"] is not None else None
    return {"window": [year_start, year_end], "features": cols, "matrix": matrix}


# ── ML Predict: crisis vulnerability inference ───────────────────────────────

@app.get("/ml/predict/model-info")
def ml_model_info():
    """Returns training metrics and metadata of the Random Forest model."""
    if not predictor.is_ready():
        raise HTTPException(503, "Model not trained yet")
    return {
        "metrics":             predictor.metrics,
        "feature_importances": predictor.feature_importances(top_n=20),
        "interpretation": {
            "auc_roc":   "Area under ROC. >0.7 = useful, >0.8 = strong, >0.9 = exceptional",
            "f1_score":  "Harmonic mean of precision/recall on minority class",
            "warning":   "Model predicts risk-of-being-in-crisis-pattern, NOT calendar date of next crisis",
        },
    }


@app.get("/ml/predict/ranking")
def ml_predict_ranking(limit: int = Query(default=30, ge=5, le=100)):
    """
    Ranks ALL countries by their current crisis-pattern probability.
    Uses most recent (per country) row of ml_features_wide.
    """
    if not predictor.is_ready():
        raise HTTPException(503, "Model not trained yet")
    import pandas as pd
    conn = _connect()
    df = pd.read_sql(
        """
        SELECT DISTINCT ON (cca2) m.*, c.name, c.flag_emoji
        FROM   ml_features_wide m
        JOIN   countries c ON c.cca2 = m.cca2
        ORDER  BY cca2, year DESC
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return []

    probs = predictor.predict_proba(df)
    df["crisis_proba"] = probs

    # Sort high to low + return top N as JSON
    df = df.sort_values("crisis_proba", ascending=False).head(limit)
    return [{
        "cca2":          row["cca2"],
        "name":          row["name"],
        "flag_emoji":    row["flag_emoji"],
        "year_data":     int(row["year"]),
        "archetype":     row.get("archetype"),
        "region":        row.get("region"),
        "crisis_proba":  round(float(row["crisis_proba"]), 3),
        "in_crisis_actual": bool(row["in_crisis"]) if pd.notna(row["in_crisis"]) else None,
        "gdp_pcap":      float(row["gdp_pcap"]) if pd.notna(row["gdp_pcap"]) else None,
        "gdp_growth":    float(row["gdp_growth"]) if pd.notna(row["gdp_growth"]) else None,
        "inflation":     float(row["inflation"]) if pd.notna(row["inflation"]) else None,
        "unemp":         float(row["unemp"]) if pd.notna(row["unemp"]) else None,
        "ext_debt":      float(row["ext_debt"]) if pd.notna(row["ext_debt"]) else None,
    } for _, row in df.iterrows()]


@app.get("/ml/predict/{cca2}")
def ml_predict_country(cca2: str):
    """Crisis probability + history series for a single country."""
    if not predictor.is_ready():
        raise HTTPException(503, "Model not trained yet")
    cca2 = cca2.upper()
    import pandas as pd
    conn = _connect()
    df = pd.read_sql(
        "SELECT * FROM ml_features_wide WHERE cca2 = %(cca2)s ORDER BY year",
        conn, params={"cca2": cca2},
    )
    conn.close()
    if df.empty:
        raise HTTPException(404, f"No feature data for {cca2}")

    probs = predictor.predict_proba(df)
    df["crisis_proba"] = probs

    history = df[["year", "crisis_proba", "in_crisis", "gdp_growth", "unemp", "inflation"]].copy()
    history = history.replace({np.nan: None})
    return {
        "country":   cca2,
        "history":   history.to_dict(orient="records"),
        "current":   {
            "year":          int(df["year"].iloc[-1]),
            "crisis_proba":  round(float(probs[-1]), 3),
            "interpretation": "high (>0.6)" if probs[-1] > 0.6 else "moderate (0.3-0.6)" if probs[-1] > 0.3 else "low (<0.3)",
        },
    }


# ── ML Risk Profile: master endpoint combining all models ────────────────────

@app.get("/ml/risk-profile/{cca2}")
def ml_risk_profile(cca2: str):
    """
    Master risk profile combining 4 models:
      - Crisis predictor (Random Forest)
      - Data-driven cluster (K-Means)
      - Rules-based archetype
      - Composite scores (strength, vulnerability)
    Plus stress signals, currency regime, peers and a composite risk score.
    """
    cca2 = cca2.upper()

    # Country metadata
    country_rows = query(
        "SELECT cca2, name, region, subregion, flag_emoji, population, "
        "latest_gdp_per_capita, latest_gdp_growth, latest_unemployment, latest_inflation "
        "FROM countries WHERE cca2 = %s", (cca2,)
    )
    if not country_rows:
        raise HTTPException(404, f"Country '{cca2}' not found")
    country = country_rows[0]

    # ── Crisis probability (from RF model) ────────────────────────────────────
    crisis_proba: Optional[float] = None
    if predictor.is_ready():
        try:
            import pandas as pd
            conn = _connect()
            df = pd.read_sql(
                "SELECT * FROM ml_features_wide WHERE cca2=%(cca2)s ORDER BY year DESC LIMIT 1",
                conn, params={"cca2": cca2},
            )
            conn.close()
            if not df.empty:
                proba = predictor.predict_proba(df)
                crisis_proba = round(float(proba[0]), 3)
        except Exception as exc:
            logger.warning("Crisis proba failed for %s: %s", cca2, exc)

    # ── Cluster assignment ────────────────────────────────────────────────────
    cluster_info = None
    if clusterer.is_ready():
        cl = clusterer.get_country(cca2)
        if cl:
            centroid = clusterer.centroids[cl["cluster"]]
            cluster_info = {
                "cluster_id": cl["cluster"],
                "label":      centroid["label"],
                "n_members":  centroid["n_members"],
            }

    # ── Other derived scores ──────────────────────────────────────────────────
    archetype = query("SELECT * FROM country_archetype WHERE cca2 = %s", (cca2,))
    archetype = archetype[0] if archetype else None

    strength = query("SELECT * FROM country_economic_strength WHERE cca2 = %s", (cca2,))
    strength = strength[0] if strength else None

    vulnerability = query("SELECT * FROM external_vulnerability_index WHERE cca2 = %s", (cca2,))
    vulnerability = vulnerability[0] if vulnerability else None

    convergence = query("SELECT * FROM convergence_metrics WHERE cca2 = %s", (cca2,))
    convergence = convergence[0] if convergence else None

    # ── Currency regime ───────────────────────────────────────────────────────
    euro_row = query("SELECT adoption_year FROM euro_adoption WHERE cca2 = %s", (cca2,))
    currency_regime = "Eurozone" if euro_row else ("USD-pegged" if cca2 in ("EC","SV","PA","TL","ZW") else "Own currency")

    # ── Crisis recovery history ───────────────────────────────────────────────
    crises = query(
        """
        SELECT cri.hito_id, cri.nombre_evento, cri.gdp_growth_min,
               cr.years_to_recover, cr.pct_drop, cr.recovered
        FROM   country_crisis_impact cri
        LEFT   JOIN country_crisis_recovery cr ON cr.cca2 = cri.cca2 AND cr.hito_id = cri.hito_id
        WHERE  cri.cca2 = %s
        ORDER  BY cri.anio_inicio DESC LIMIT 5
        """, (cca2,)
    )

    # ── Stress signals (latest available indicators) ─────────────────────────
    latest = query(
        """
        SELECT
            MAX(value) FILTER (WHERE indicator_code='DT.DOD.DECT.GN.ZS' AND year >= 2018) AS ext_debt_recent,
            MAX(value) FILTER (WHERE indicator_code='GC.DOD.TOTL.GD.ZS' AND year >= 2018) AS govt_debt_recent,
            MAX(value) FILTER (WHERE indicator_code='BN.CAB.XOKA.GD.ZS' AND year >= 2020) AS curr_acc_recent,
            MAX(value) FILTER (WHERE indicator_code='FI.RES.TOTL.MO'    AND year >= 2020) AS reserves_recent,
            MAX(value) FILTER (WHERE indicator_code='FP.CPI.TOTL.ZG'    AND year >= 2022) AS inflation_recent,
            MAX(value) FILTER (WHERE indicator_code='SL.UEM.TOTL.ZS'    AND year >= 2022) AS unemp_recent
        FROM country_indicators WHERE cca2 = %s
        """, (cca2,)
    )
    latest = latest[0] if latest else {}
    stress_signals = []
    if latest.get("ext_debt_recent") and float(latest["ext_debt_recent"]) >= 60:
        stress_signals.append({"signal": "High external debt", "value": f"{float(latest['ext_debt_recent']):.0f}% of GNI", "severity": "high" if float(latest['ext_debt_recent']) >= 100 else "moderate"})
    if latest.get("govt_debt_recent") and float(latest["govt_debt_recent"]) >= 80:
        stress_signals.append({"signal": "High government debt", "value": f"{float(latest['govt_debt_recent']):.0f}% of GDP", "severity": "high" if float(latest['govt_debt_recent']) >= 120 else "moderate"})
    if latest.get("curr_acc_recent") and float(latest["curr_acc_recent"]) <= -4:
        stress_signals.append({"signal": "Current account deficit", "value": f"{float(latest['curr_acc_recent']):.1f}% of GDP", "severity": "high" if float(latest['curr_acc_recent']) <= -8 else "moderate"})
    if latest.get("reserves_recent") and float(latest["reserves_recent"]) < 3:
        stress_signals.append({"signal": "Low FX reserves", "value": f"{float(latest['reserves_recent']):.1f} months imports", "severity": "high"})
    if latest.get("inflation_recent") and float(latest["inflation_recent"]) >= 8:
        stress_signals.append({"signal": "High inflation", "value": f"{float(latest['inflation_recent']):.1f}%", "severity": "high" if float(latest['inflation_recent']) >= 20 else "moderate"})
    if latest.get("unemp_recent") and float(latest["unemp_recent"]) >= 10:
        stress_signals.append({"signal": "Elevated unemployment", "value": f"{float(latest['unemp_recent']):.1f}%", "severity": "high" if float(latest['unemp_recent']) >= 20 else "moderate"})

    # ── Composite risk score (0-100) ──────────────────────────────────────────
    # Weighted combination of available signals; missing components ignored.
    components = []
    if crisis_proba is not None:
        components.append(("crisis_pattern", crisis_proba * 100, 0.30))
    if vulnerability and vulnerability.get("vulnerability_score") is not None:
        components.append(("vulnerability", float(vulnerability["vulnerability_score"]), 0.25))
    if strength and strength.get("strength_score") is not None:
        components.append(("strength_inverse", 100 - float(strength["strength_score"]), 0.20))
    # Stress signals contribution
    high_signals = sum(1 for s in stress_signals if s["severity"] == "high")
    mod_signals  = sum(1 for s in stress_signals if s["severity"] == "moderate")
    stress_score = min(high_signals * 25 + mod_signals * 12, 100)
    components.append(("stress_signals", stress_score, 0.25))

    if components:
        total_weight = sum(w for _, _, w in components)
        composite = sum(v * w for _, v, w in components) / total_weight
        composite = round(min(max(composite, 0), 100), 1)
    else:
        composite = None

    if composite is None:
        verdict = "Unknown"
    elif composite >= 65:
        verdict = "High risk"
    elif composite >= 45:
        verdict = "Elevated risk"
    elif composite >= 25:
        verdict = "Moderate risk"
    else:
        verdict = "Low risk"

    # ── Peers in same cluster (top 5 most similar by GDP/cap proximity) ──────
    peers = []
    if cluster_info:
        peer_codes = [c2 for c2, info in clusterer.assignments.items()
                      if info["cluster"] == cluster_info["cluster_id"] and c2 != cca2]
        if peer_codes:
            placeholders = ",".join(["%s"] * len(peer_codes))
            peers = query(
                f"""
                SELECT cca2, name, flag_emoji, region, latest_gdp_per_capita
                FROM countries WHERE cca2 IN ({placeholders})
                ORDER BY ABS(latest_gdp_per_capita - %s) NULLS LAST
                LIMIT 6
                """,
                peer_codes + [country["latest_gdp_per_capita"] or 0],
            )

    return {
        "country":         country,
        "composite":       {
            "score":      composite,
            "verdict":    verdict,
            "components": [{"name": n, "value": round(v, 1), "weight": w} for n, v, w in components],
        },
        "crisis_proba":    crisis_proba,
        "cluster":         cluster_info,
        "archetype":       archetype,
        "strength":        strength,
        "vulnerability":   vulnerability,
        "convergence":     convergence,
        "currency_regime": currency_regime,
        "stress_signals":  stress_signals,
        "recent_crises":   crises,
        "cluster_peers":   peers,
    }


# ── ML Clusters: data-driven country grouping ───────────────────────────────

@app.get("/ml/clusters/info")
def ml_clusters_info():
    """K, silhouette, cluster centroids and characterisation."""
    if not clusterer.is_ready():
        raise HTTPException(503, "Cluster model not ready")
    return clusterer.get_summary()


@app.get("/ml/clusters/countries")
def ml_clusters_countries():
    """All clustered countries with cluster_id, PCA coords and metadata."""
    if not clusterer.is_ready():
        raise HTTPException(503, "Cluster model not ready")
    # Join with country metadata
    rows = query(
        """
        SELECT c.cca2, c.name, c.region, c.subregion, c.flag_emoji,
               c.latest_gdp_per_capita,
               a.archetype
        FROM   countries_clean c
        LEFT   JOIN country_archetype a ON a.cca2 = c.cca2
        """
    )
    out = []
    for r in rows:
        cl = clusterer.get_country(r["cca2"])
        if not cl:
            continue
        out.append({
            **r,
            "cluster_id":  cl["cluster"],
            "pca_x":       cl["pca_x"],
            "pca_y":       cl["pca_y"],
        })
    return out


@app.get("/ml/clusters/{cluster_id}")
def ml_clusters_members(cluster_id: int):
    """All countries belonging to a specific cluster + centroid info."""
    if not clusterer.is_ready():
        raise HTTPException(503, "Cluster model not ready")
    if cluster_id < 0 or cluster_id >= clusterer.k:
        raise HTTPException(404, f"Cluster {cluster_id} not found (K={clusterer.k})")
    centroid_info = clusterer.centroids[cluster_id]
    # Get members with metadata
    member_codes = [cca2 for cca2, info in clusterer.assignments.items() if info["cluster"] == cluster_id]
    if not member_codes:
        return {"centroid": centroid_info, "members": []}
    placeholders = ",".join(["%s"] * len(member_codes))
    members = query(
        f"""
        SELECT c.cca2, c.name, c.flag_emoji, c.region, c.subregion,
               c.latest_gdp_per_capita, c.latest_gdp_growth,
               c.latest_unemployment, c.latest_inflation,
               a.archetype
        FROM   countries_clean c
        LEFT   JOIN country_archetype a ON a.cca2 = c.cca2
        WHERE  c.cca2 IN ({placeholders})
        ORDER  BY c.latest_gdp_per_capita DESC NULLS LAST
        """,
        member_codes,
    )
    return {"centroid": centroid_info, "members": members}


@app.get("/ml/clusters/compare-archetypes")
def ml_clusters_compare():
    """Cross-tab: data-driven cluster vs rules-based archetype."""
    if not clusterer.is_ready():
        raise HTTPException(503, "Cluster model not ready")
    # Get all countries with both cluster_id and archetype
    rows = query(
        "SELECT cca2, archetype FROM country_archetype"
    )
    cross = {}  # cluster_id → archetype → count
    for r in rows:
        cl = clusterer.get_country(r["cca2"])
        if not cl:
            continue
        cid = cl["cluster"]
        arch = r["archetype"] or "Unknown"
        cross.setdefault(cid, {}).setdefault(arch, 0)
        cross[cid][arch] += 1
    return {"matrix": cross, "k": clusterer.k}


# ── Methodology: structured documentation ────────────────────────────────────

@app.get("/methodology")
def methodology():
    """Structured documentation: data sources, derived views, score formulas, caveats."""
    # All indicators with their World Bank codes, names, coverage stats
    indicators = query(
        """
        SELECT indicator_code, indicator_name,
               MIN(year) AS first_year, MAX(year) AS last_year,
               COUNT(DISTINCT cca2) AS n_countries,
               COUNT(*) AS n_obs
        FROM   country_indicators
        WHERE  value IS NOT NULL
        GROUP  BY indicator_code, indicator_name
        ORDER  BY indicator_code
        """
    )
    # Historical milestones (12)
    hitos = query(
        "SELECT id, anio_inicio, anio_fin, nombre_evento, descripcion, tipo_shock "
        "FROM hitos_historicos ORDER BY anio_inicio"
    )
    # Materialised views: row counts as integrity proof
    views_data = query(
        """
        SELECT 'countries_clean' AS view_name, COUNT(*) AS rows FROM countries_clean
        UNION ALL SELECT 'country_archetype',         COUNT(*) FROM country_archetype
        UNION ALL SELECT 'country_economic_strength', COUNT(*) FROM country_economic_strength
        UNION ALL SELECT 'country_crisis_impact',     COUNT(*) FROM country_crisis_impact
        UNION ALL SELECT 'country_crisis_recovery',   COUNT(*) FROM country_crisis_recovery
        UNION ALL SELECT 'crisis_winners_losers',     COUNT(*) FROM crisis_winners_losers
        UNION ALL SELECT 'convergence_metrics',       COUNT(*) FROM convergence_metrics
        UNION ALL SELECT 'external_vulnerability_index', COUNT(*) FROM external_vulnerability_index
        UNION ALL SELECT 'country_purchasing_power',  COUNT(*) FROM country_purchasing_power
        UNION ALL SELECT 'currency_history',          COUNT(*) FROM currency_history
        UNION ALL SELECT 'subregion_stats',           COUNT(*) FROM subregion_stats
        UNION ALL SELECT 'country_peers',             COUNT(*) FROM country_peers
        UNION ALL SELECT 'decade_summary',            COUNT(*) FROM decade_summary
        UNION ALL SELECT 'archetype_shock_matrix',    COUNT(*) FROM archetype_shock_matrix
        UNION ALL SELECT 'household_budget_breakdown', COUNT(*) FROM household_budget_breakdown
        ORDER BY 1
        """
    )
    return {
        "indicators":   indicators,
        "hitos":        hitos,
        "views":        views_data,
        "indicator_groups": {
            "macro_welfare": ["NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","SP.DYN.LE00.IN","SL.UEM.TOTL.ZS","SP.URB.TOTL.IN.ZS","SE.XPD.TOTL.GD.ZS","FP.CPI.TOTL.ZG"],
            "sectoral":      ["NV.SRV.TOTL.ZS","NV.IND.TOTL.ZS","NV.AGR.TOTL.ZS","SL.SRV.EMPL.ZS","SL.IND.EMPL.ZS","SL.AGR.EMPL.ZS"],
            "trade":         ["NE.TRD.GNFS.ZS","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","BN.CAB.XOKA.GD.ZS","TX.VAL.FUEL.ZS.UN","TX.VAL.MANF.ZS.UN"],
            "financial":     ["GC.DOD.TOTL.GD.ZS","DT.DOD.DECT.GN.ZS","FI.RES.TOTL.MO","BX.KLT.DINV.WD.GD.ZS","NE.GDI.TOTL.ZS"],
            "innovation":    ["GB.XPD.RSDV.GD.ZS","IT.NET.USER.ZS"],
            "demographic":   ["SP.DYN.TFRT.IN","SP.POP.DPND","SI.POV.GINI"],
            "energy_environ":["EG.FEC.RNEW.ZS","EN.GHG.CO2.PC.CE.AR5"],
            "currency_ppp":  ["PA.NUS.FCRF","NY.GDP.PCAP.PP.CD","NE.CON.PRVT.PP.KD","NY.GNP.PCAP.PP.CD","NY.GDP.PCAP.PP.KD"],
        },
        "formulas": [
            {"name": "strength_score", "expression": "0.35·level + 0.25·stability + 0.20·resilience + 0.20·diversification",
             "components": {
                "level":          "PERCENT_RANK(GDP_per_capita) × 100",
                "stability":      "100 − PERCENT_RANK(STDDEV(GDP_growth_30y)) × 100",
                "resilience":     "PERCENT_RANK(AVG(GDP_growth in 2010-12 + 2021-23)) × 100",
                "diversification":"100 − PERCENT_RANK(sector_HHI) × 100",
             }},
            {"name": "vulnerability_score", "expression": "0.40·concentration + 0.30·debt + 0.30·openness",
             "components": {
                 "concentration": "MAX(fuel_exports%, manuf_exports% × 0.7)",
                 "debt":          "LEAST(external_debt/GNI/150, 1.0) × 100",
                 "openness":      "LEAST(trade/GDP/200, 1.0) × 100",
             }},
            {"name": "cost_of_living_index", "expression": "(GDP_nominal_per_capita / GDP_PPP_per_capita) × 100",
             "components": {"US baseline": "≈ 100", "Higher → more expensive than US": "<60 = very cheap, >100 = expensive"}},
            {"name": "archetype rules (sequential)", "expression": "First match wins",
             "components": {
                 "Petrostate": "fuel_exports ≥ 40%",
                 "Manufacturing exporter": "manuf_exports ≥ 60% AND industry_VA ≥ 22% OR manuf ≥ 70%",
                 "Agricultural economy": "agriculture_VA ≥ 15%",
                 "Advanced services": "services_VA ≥ 75% AND GDP/cap ≥ $25k",
                 "Service-oriented": "services_VA ≥ 65% AND GDP/cap ≥ $15k",
                 "Industrial economy": "industry_VA ≥ 28%",
                 "Diversified advanced": "sector_HHI < 0.45 AND GDP/cap ≥ $30k",
                 "Diversified emerging": "sector_HHI < 0.45",
                 "Mixed / Other": "fallback",
             }},
            {"name": "crisis_winners_losers verdict", "expression": "rank_change classification with quality flags",
             "components": {
                 "Big Winner": "rank_change ≥ 10, GDP/cap ≥ $1.5k, no rebound",
                 "Rebound":    "rank_change ≥ 10 BUT GDP collapsed > 25% in prior 5 years",
                 "Volatile low-income": "rank_change ≥ 10 BUT GDP/cap < $1.5k pre-shock",
                 "Big Loser":  "rank_change ≤ −10",
             }},
        ],
        "caveats": [
            {"area": "Sector data heterogeneity",
             "detail": "WB harmonised under SNA08 retropolated differently per country. France: from 1965 · Spain: from 1995 · USA: from 1997. Tooltips in UI show coverage per country."},
            {"area": "Employment sector series",
             "detail": "ILO modelled estimates start in 1991. Older years not available from this source."},
            {"area": "PPP indicators",
             "detail": "World Bank PPP series start in 1990 (no earlier data)."},
            {"area": "Budget breakdown",
             "detail": "Curated table — only 31 top economies (sourced from Eurostat HBS, BLS CES, INEGI ENIGH, NBS, etc.). Not available for micro-states or some emerging."},
            {"area": "Currency data errors",
             "detail": "Iraq 1971-1990 and Zimbabwe pre-1979 excluded from currency_history due to WB FCRF denomination jumps. Captured in currency_data_errors table."},
            {"area": "Crisis winners/losers — rebound detection",
             "detail": "Countries whose GDP collapsed > 25% in 5 years BEFORE a crisis are tagged as 'Rebound' (not real winners). Avoids classifying Venezuela 2022-24 as a winner."},
            {"area": "Micro-states filter",
             "detail": "Macro-story endpoints exclude countries with population < 500k to avoid ranking noise (Nauru, San Marino, etc.)."},
        ],
        "data_sources": [
            {"name": "World Bank Open Data", "url": "data.worldbank.org", "indicators": 37, "license": "CC-BY-4.0"},
            {"name": "mledoze/countries (tag v4.1.1)", "url": "github.com/mledoze/countries", "purpose": "Country metadata (flags, regions, subregions, area, currencies)", "license": "ODbL-1.0",
             "note": "Read directly from the source dataset at a pinned tag. Previously consumed via the REST Countries API, which was deprecated in favour of a key-gated v5."},
            {"name": "FRED (St. Louis Fed)", "url": "fred.stlouisfed.org", "purpose": "Macro context: Brent, VIX, Fed Funds, Stress Index"},
            {"name": "Caldara & Iacoviello GPR Index", "url": "matteoiacoviello.com/gpr.htm", "purpose": "Geopolitical Risk Index (monthly)"},
            {"name": "Eurostat HBS / BLS CES / INEGI ENIGH / NBS / etc.", "url": "various", "purpose": "Household budget breakdown by category (31 countries curated)"},
        ],
    }


# ── Studies: macro storytelling endpoints ────────────────────────────────────

@app.get("/studies/shock-anatomy")
def study_shock_anatomy():
    """Matrix: how each archetype reacted to each historical crisis."""
    matrix = query(
        """
        SELECT archetype, hito_id, nombre_evento, tipo_shock,
               anio_inicio, anio_fin, n_countries,
               avg_rank_change, avg_gdp_change_pct, avg_gdp_growth_min,
               avg_unemp_max, avg_inflation_max
        FROM archetype_shock_matrix
        ORDER BY hito_id, avg_rank_change DESC NULLS LAST
        """
    )
    archetypes = query("SELECT DISTINCT archetype FROM country_archetype ORDER BY archetype")
    hitos      = query("SELECT id, nombre_evento, tipo_shock, anio_inicio, anio_fin FROM hitos_historicos ORDER BY anio_inicio")
    return {"matrix": matrix, "archetypes": [a["archetype"] for a in archetypes], "hitos": hitos}


@app.get("/studies/crisis-predictors")
def study_crisis_predictors():
    """Pre-crisis external debt predicts COVID impact (and similar shocks)."""
    # COVID
    covid = query(
        """
        WITH covid_impact AS (
            SELECT ci.cca2, ci.gdp_growth_min,
                   RANK() OVER (ORDER BY ci.gdp_growth_min ASC NULLS LAST) AS worst_rank,
                   COUNT(*) OVER ()                                         AS total
            FROM country_crisis_impact ci JOIN countries_clean cc ON cc.cca2 = ci.cca2
            WHERE ci.hito_id = 6
        ),
        debt_pre AS (
            SELECT cca2, AVG(value) AS avg_debt
            FROM country_indicators
            WHERE indicator_code = 'DT.DOD.DECT.GN.ZS' AND year BETWEEN 2015 AND 2019
            GROUP BY cca2
        )
        SELECT ci.cca2, c.name, c.flag_emoji, ci.gdp_growth_min,
               ROUND(dp.avg_debt::numeric, 1) AS ext_debt_pre,
               CASE WHEN ci.worst_rank <= ci.total * 0.15 THEN 'most_hit'
                    WHEN ci.worst_rank >= ci.total * 0.85 THEN 'least_hit'
                    ELSE 'mid' END AS tier
        FROM covid_impact ci
        JOIN debt_pre dp   ON dp.cca2 = ci.cca2
        JOIN countries c   ON c.cca2  = ci.cca2
        WHERE dp.avg_debt IS NOT NULL
        ORDER BY ci.gdp_growth_min ASC
        """
    )
    # Aggregate tiers
    tiers = query(
        """
        WITH covid_impact AS (
            SELECT ci.cca2, ci.gdp_growth_min,
                   RANK() OVER (ORDER BY ci.gdp_growth_min ASC NULLS LAST) AS worst_rank,
                   COUNT(*) OVER ()                                         AS total
            FROM country_crisis_impact ci
            WHERE ci.hito_id = 6
        ),
        debt_pre AS (
            SELECT cca2, AVG(value) AS avg_debt FROM country_indicators
            WHERE indicator_code='DT.DOD.DECT.GN.ZS' AND year BETWEEN 2015 AND 2019 GROUP BY cca2
        )
        SELECT
          CASE WHEN ci.worst_rank <= ci.total * 0.15 THEN 'most_hit'
               WHEN ci.worst_rank >= ci.total * 0.85 THEN 'least_hit'
               ELSE 'mid' END AS tier,
          COUNT(*) AS n,
          ROUND(AVG(dp.avg_debt)::numeric, 1) AS avg_ext_debt,
          ROUND(AVG(ci.gdp_growth_min)::numeric, 1) AS avg_gdp_min
        FROM covid_impact ci JOIN debt_pre dp ON dp.cca2 = ci.cca2
        GROUP BY tier ORDER BY tier
        """
    )
    return {"countries": covid, "tier_aggregates": tiers}


@app.get("/studies/demographic-destiny")
def study_demographic_destiny():
    """Dependency ratio (aging) vs 30/60-year economic convergence."""
    rows = query(
        """
        WITH dep AS (
            SELECT cca2, AVG(value) AS avg_dep FROM country_indicators
            WHERE indicator_code = 'SP.POP.DPND' AND year BETWEEN 2020 AND 2024
            GROUP BY cca2
        )
        SELECT c.cca2, c.name, c.region, c.flag_emoji,
               ROUND(d.avg_dep::numeric, 1)  AS dependency_ratio,
               cm.pct_us_now,
               cm.change_30y, cm.change_60y, cm.trajectory
        FROM countries_clean c
        JOIN dep d                    ON d.cca2  = c.cca2
        JOIN convergence_metrics cm   ON cm.cca2 = c.cca2
        WHERE cm.change_60y IS NOT NULL
        ORDER BY d.avg_dep
        """
    )
    aggregates = query(
        """
        WITH dep AS (
            SELECT cca2, AVG(value) AS avg_dep FROM country_indicators
            WHERE indicator_code='SP.POP.DPND' AND year BETWEEN 2020 AND 2024 GROUP BY cca2
        )
        SELECT
          CASE WHEN d.avg_dep < 50 THEN 'young (<50)'
               WHEN d.avg_dep < 60 THEN 'mid (50-60)'
               ELSE 'aged (>60)' END AS tier,
          COUNT(*) AS n,
          ROUND(AVG(cm.change_30y)::numeric, 1) AS avg_conv_30y,
          ROUND(AVG(cm.change_60y)::numeric, 1) AS avg_conv_60y
        FROM dep d JOIN convergence_metrics cm ON cm.cca2 = d.cca2
        WHERE cm.change_60y IS NOT NULL
        GROUP BY tier ORDER BY tier
        """
    )
    return {"countries": rows, "tier_aggregates": aggregates}


@app.get("/studies/cost-vs-sectors")
def study_cost_vs_sectors():
    """Cost-of-living index strongly correlates with services-heavy structure."""
    rows = query(
        """
        SELECT pp.cca2, c.name, c.region, c.flag_emoji,
               pp.cost_of_living_index, pp.cheaper_than_us_pct,
               a.services_va_pct, a.industry_va_pct, a.agriculture_va_pct,
               a.archetype
        FROM country_purchasing_power pp
        JOIN country_archetype a ON a.cca2 = pp.cca2
        JOIN countries c         ON c.cca2 = pp.cca2
        WHERE pp.cost_of_living_index IS NOT NULL
          AND a.services_va_pct IS NOT NULL
        ORDER BY pp.cost_of_living_index DESC
        """
    )
    aggregates = query(
        """
        SELECT
          CASE WHEN pp.cost_of_living_index < 50 THEN 'cheap (<50)'
               WHEN pp.cost_of_living_index < 80 THEN 'mid (50-80)'
               ELSE 'expensive (>80)' END AS tier,
          COUNT(*) AS n,
          ROUND(AVG(a.services_va_pct)::numeric, 1)    AS avg_srv,
          ROUND(AVG(a.industry_va_pct)::numeric, 1)    AS avg_ind,
          ROUND(AVG(a.agriculture_va_pct)::numeric, 1) AS avg_agr
        FROM country_purchasing_power pp
        JOIN country_archetype a ON a.cca2 = pp.cca2
        GROUP BY tier ORDER BY tier
        """
    )
    return {"countries": rows, "tier_aggregates": aggregates}


@app.get("/studies/diversification-resilience")
def study_diversification_resilience():
    """Lower sector concentration (HHI) = faster crisis recovery."""
    rows = query(
        """
        SELECT a.cca2, c.name, c.flag_emoji,
               ROUND(a.sector_hhi::numeric, 3) AS sector_hhi,
               cr.years_to_recover, cr.pct_drop, cr.recovered,
               h.nombre_evento, h.id AS hito_id
        FROM country_archetype       a
        JOIN country_crisis_recovery cr ON cr.cca2 = a.cca2
        JOIN hitos_historicos        h  ON h.id   = cr.hito_id
        JOIN countries c               ON c.cca2 = a.cca2
        WHERE a.sector_hhi IS NOT NULL AND cr.years_to_recover IS NOT NULL
        """
    )
    aggregates = query(
        """
        SELECT
          h.id AS hito_id, h.nombre_evento,
          CASE WHEN a.sector_hhi >= 0.6  THEN 'high concentration'
               WHEN a.sector_hhi >= 0.45 THEN 'mid'
               ELSE 'diversified' END AS tier,
          COUNT(*) AS n,
          ROUND(AVG(cr.years_to_recover)::numeric, 1) AS avg_yrs_recover,
          ROUND(AVG(cr.pct_drop)::numeric, 1)         AS avg_drop
        FROM country_archetype       a
        JOIN country_crisis_recovery cr ON cr.cca2 = a.cca2
        JOIN hitos_historicos        h  ON h.id   = cr.hito_id
        WHERE a.sector_hhi IS NOT NULL AND cr.years_to_recover IS NOT NULL
        GROUP BY h.id, h.nombre_evento, tier
        ORDER BY h.id, tier
        """
    )
    return {"countries": rows, "tier_aggregates": aggregates}


@app.get("/studies/petrostates-cycle")
def study_petrostates_cycle():
    """50-year boom-bust cycle of petrostates across all energy-related shocks."""
    cycle = query(
        """
        SELECT h.id AS hito_id, h.nombre_evento, h.tipo_shock,
               h.anio_inicio, h.anio_fin,
               m.avg_rank_change, m.avg_gdp_change_pct, m.n_countries
        FROM archetype_shock_matrix m
        JOIN hitos_historicos       h ON h.id = m.hito_id
        WHERE m.archetype = 'Petrostate'
        ORDER BY h.anio_inicio
        """
    )
    top_petros = query(
        """
        SELECT cca2, name, region, flag_emoji,
               fuel_exports_pct, gdp_per_capita
        FROM country_archetype
        WHERE archetype = 'Petrostate'
        ORDER BY fuel_exports_pct DESC LIMIT 15
        """
    )
    # Brent context per hito window
    brent = query(
        """
        SELECT h.id AS hito_id,
               ROUND(AVG(cg.precio_brent_avg)::numeric, 1) AS avg_brent
        FROM hitos_historicos h
        LEFT JOIN contexto_global_anual cg
          ON cg.anio BETWEEN h.anio_inicio AND h.anio_fin
        GROUP BY h.id ORDER BY h.id
        """
    )
    return {"cycle": cycle, "top_petrostates": top_petros, "brent_context": brent}


@app.get("/studies/lehman-north-vs-south")
def study_lehman_north_vs_south():
    """Lehman 2008-09: a North/rich-country crisis, emerging markets escaped."""
    rows = query(
        """
        WITH lehman AS (
            SELECT cca2, gdp_growth_min, unemployment_max
            FROM country_crisis_impact WHERE hito_id = 3
        ),
        infl_pre AS (
            SELECT cca2, AVG(value) AS avg_infl FROM country_indicators
            WHERE indicator_code = 'FP.CPI.TOTL.ZG' AND year BETWEEN 2005 AND 2007
            GROUP BY cca2
        ),
        gdp_pre AS (
            SELECT DISTINCT ON (cca2) cca2, value AS gdp_pcap
            FROM country_indicators
            WHERE indicator_code = 'NY.GDP.PCAP.CD' AND year = 2007
            ORDER BY cca2, year DESC
        )
        SELECT l.cca2, c.name, c.region, c.flag_emoji,
               l.gdp_growth_min, l.unemployment_max,
               ROUND(ip.avg_infl::numeric, 1) AS infl_pre,
               ROUND(gp.gdp_pcap::numeric, 0) AS gdp_pcap_2007,
               CASE WHEN gp.gdp_pcap >= 25000 THEN 'North'
                    WHEN gp.gdp_pcap >= 5000  THEN 'Mid'
                    ELSE 'South' END AS tier
        FROM lehman l
        JOIN countries_clean c ON c.cca2 = l.cca2
        LEFT JOIN infl_pre ip  ON ip.cca2 = l.cca2
        LEFT JOIN gdp_pre  gp  ON gp.cca2 = l.cca2
        WHERE gp.gdp_pcap IS NOT NULL
        ORDER BY l.gdp_growth_min
        """
    )
    aggregates = query(
        """
        WITH lehman AS (
            SELECT cca2, gdp_growth_min FROM country_crisis_impact WHERE hito_id = 3
        ),
        gdp_pre AS (
            SELECT DISTINCT ON (cca2) cca2, value AS gdp_pcap
            FROM country_indicators
            WHERE indicator_code = 'NY.GDP.PCAP.CD' AND year = 2007
            ORDER BY cca2, year DESC
        )
        SELECT
          CASE WHEN gp.gdp_pcap >= 25000 THEN 'North'
               WHEN gp.gdp_pcap >= 5000  THEN 'Mid'
               ELSE 'South' END AS tier,
          COUNT(*) AS n,
          ROUND(AVG(l.gdp_growth_min)::numeric, 1) AS avg_gdp_min
        FROM lehman l JOIN gdp_pre gp ON gp.cca2 = l.cca2
        GROUP BY tier ORDER BY tier
        """
    )
    return {"countries": rows, "tier_aggregates": aggregates}


@app.get("/studies/hyperinflations")
def study_hyperinflations():
    """Compare documented hyperinflations: Zimbabwe, Venezuela, Bolivia, Argentina, Peru."""
    cases = query(
        """
        SELECT ch.cca2, c.name, c.flag_emoji,
               ch.year, ch.lcu_per_usd, ch.yoy_pct
        FROM   currency_history ch
        JOIN   countries c ON c.cca2 = ch.cca2
        WHERE  ch.cca2 IN ('ZW','VE','BO','AR','PE','HU','DE','MM','SR')
          AND  ch.is_extreme_jump = TRUE
        ORDER  BY ch.cca2, ch.year
        """
    )
    # Cumulative devaluation per country (max LCU/USD vs min)
    cumulative = query(
        """
        SELECT ch.cca2, c.name, c.flag_emoji,
               MIN(ch.lcu_per_usd) AS min_rate,
               MAX(ch.lcu_per_usd) AS max_rate,
               MIN(ch.year) AS first_y,
               MAX(ch.year) AS last_y,
               ROUND((MAX(ch.lcu_per_usd) / NULLIF(MIN(ch.lcu_per_usd), 0))::numeric, 0) AS multiplier
        FROM   currency_history ch
        JOIN   countries c ON c.cca2 = ch.cca2
        WHERE  ch.cca2 IN ('ZW','VE','BO','AR','PE','MM','SR')
        GROUP  BY ch.cca2, c.name, c.flag_emoji
        ORDER  BY multiplier DESC NULLS LAST
        """
    )
    # Inflation context (CPI) for same countries during their hyperinflation
    inflation = query(
        """
        SELECT ci.cca2, ci.year, ci.value AS cpi_yoy
        FROM   country_indicators ci
        WHERE  ci.indicator_code = 'FP.CPI.TOTL.ZG'
          AND  ci.cca2 IN ('ZW','VE','BO','AR','PE','MM')
          AND  ci.value > 100
        ORDER  BY ci.cca2, ci.year
        """
    )
    return {"cases": cases, "cumulative": cumulative, "inflation": inflation}


@app.get("/studies/asian-miracle")
def study_asian_miracle():
    """The Asian miracle: convergence cases since 1965."""
    cases = query(
        """
        SELECT c.cca2, c.name, c.flag_emoji,
               cm.pct_us_60y_ago, cm.pct_us_30y_ago, cm.pct_us_now,
               cm.change_30y, cm.change_60y, cm.trajectory,
               a.archetype,
               es.strength_score
        FROM   convergence_metrics cm
        JOIN   countries_clean c ON c.cca2 = cm.cca2
        LEFT   JOIN country_archetype a  ON a.cca2 = c.cca2
        LEFT   JOIN country_economic_strength es ON es.cca2 = c.cca2
        WHERE  cm.change_60y > 8  -- substantial convergence
          AND  c.region IN ('Asia','Europe')
        ORDER  BY cm.change_60y DESC NULLS LAST
        LIMIT  20
        """
    )
    # Top Asian convergers across multiple indicators
    asian_focus = query(
        """
        SELECT c.cca2, c.name, c.flag_emoji,
               c.population, c.latest_gdp_per_capita,
               cm.change_60y, a.archetype,
               es.strength_score
        FROM   countries_clean c
        JOIN   convergence_metrics cm ON cm.cca2 = c.cca2
        LEFT   JOIN country_archetype a  ON a.cca2 = c.cca2
        LEFT   JOIN country_economic_strength es ON es.cca2 = c.cca2
        WHERE  c.region = 'Asia'
        ORDER  BY cm.change_60y DESC NULLS LAST
        LIMIT  10
        """
    )
    # Common factor analysis: what indicators are correlated with strong convergence
    # (Industry VA, exports %, education spending averaged for top convergers)
    common = query(
        """
        WITH top_conv AS (
            SELECT cca2 FROM convergence_metrics WHERE change_60y > 10
        ),
        bot_conv AS (
            SELECT cca2 FROM convergence_metrics WHERE change_60y < -5
        )
        SELECT
            'top_convergers' AS group_name,
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='NV.IND.TOTL.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 1) AS avg_industry_va,
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='NE.EXP.GNFS.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 1) AS avg_exports,
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='SE.XPD.TOTL.GD.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 2) AS avg_edu,
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='NE.GDI.TOTL.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 1) AS avg_capital_form
        FROM country_indicators ci JOIN top_conv ON top_conv.cca2 = ci.cca2
        UNION ALL
        SELECT
            'bottom_divergers',
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='NV.IND.TOTL.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 1),
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='NE.EXP.GNFS.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 1),
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='SE.XPD.TOTL.GD.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 2),
            ROUND(AVG(value) FILTER (WHERE ci.indicator_code='NE.GDI.TOTL.ZS' AND ci.year BETWEEN 2015 AND 2024)::numeric, 1)
        FROM country_indicators ci JOIN bot_conv ON bot_conv.cca2 = ci.cca2
        """
    )
    return {"top_convergers_global": cases, "asia_focus": asian_focus, "factors": common}


@app.get("/studies/euro-impact")
def study_euro_impact():
    """
    Did giving up your currency help or hurt? Pre/post-adoption metrics
    for eurozone members vs a matched control group of European countries
    that kept their own currency (UK, SE, CH, PL, NO, CZ, DK, HU).
    """
    rows = query("SELECT * FROM euro_impact_analysis ORDER BY in_eurozone DESC, cca2")
    # Aggregate averages by group
    aggs = query(
        """
        SELECT in_eurozone,
               COUNT(*) AS n,
               ROUND(AVG(inflation_pre  - inflation_post)::numeric, 2)  AS infl_drop,
               ROUND(AVG(gdp_growth_pre - gdp_growth_post)::numeric, 2) AS growth_drop,
               ROUND(AVG(govt_debt_post - govt_debt_pre)::numeric, 1)   AS govt_debt_increase,
               ROUND(AVG(curr_acc_post  - curr_acc_pre)::numeric, 2)    AS curr_acc_change,
               ROUND(AVG(unemp_post     - unemp_pre)::numeric, 2)       AS unemp_change
        FROM euro_impact_analysis
        WHERE inflation_pre IS NOT NULL AND inflation_post IS NOT NULL
        GROUP BY in_eurozone
        """
    )
    return {"countries": rows, "aggregates": aggs}


@app.get("/studies/structural-transformation")
def study_structural_transformation():
    """
    Petty-Clark / Kuznets: how economies shift from agriculture to industry to services.
    The 60-year transformation pattern shown for landmark countries.
    """
    # Sectoral VA over 60 years for landmark countries
    cases = ['KR','CN','IN','ES','US','DE','BR','NG','VN','ID','TR']
    placeholders = ",".join(["%s"] * len(cases))
    series = query(
        f"""
        SELECT cca2, year,
               MAX(value) FILTER (WHERE indicator_code='NV.AGR.TOTL.ZS') AS agr,
               MAX(value) FILTER (WHERE indicator_code='NV.IND.TOTL.ZS') AS ind,
               MAX(value) FILTER (WHERE indicator_code='NV.SRV.TOTL.ZS') AS srv
        FROM   country_indicators
        WHERE  cca2 IN ({placeholders})
          AND  indicator_code IN ('NV.AGR.TOTL.ZS','NV.IND.TOTL.ZS','NV.SRV.TOTL.ZS')
          AND  year BETWEEN 1965 AND 2024
        GROUP  BY cca2, year ORDER BY cca2, year
        """,
        cases,
    )
    # Decade snapshots aggregated globally — average sectoral share per decade
    decades = query(
        """
        SELECT
          CASE WHEN year < 1975 THEN '1965-1974' WHEN year < 1985 THEN '1975-1984'
               WHEN year < 1995 THEN '1985-1994' WHEN year < 2005 THEN '1995-2004'
               WHEN year < 2015 THEN '2005-2014' ELSE '2015-2024' END AS decade,
          ROUND(AVG(value) FILTER (WHERE indicator_code='NV.AGR.TOTL.ZS')::numeric, 1) AS agr,
          ROUND(AVG(value) FILTER (WHERE indicator_code='NV.IND.TOTL.ZS')::numeric, 1) AS ind,
          ROUND(AVG(value) FILTER (WHERE indicator_code='NV.SRV.TOTL.ZS')::numeric, 1) AS srv
        FROM country_indicators
        WHERE indicator_code IN ('NV.AGR.TOTL.ZS','NV.IND.TOTL.ZS','NV.SRV.TOTL.ZS')
          AND year BETWEEN 1965 AND 2024
        GROUP BY decade ORDER BY decade
        """
    )
    # Country names for legend
    names = query(
        f"SELECT cca2, name, flag_emoji FROM countries WHERE cca2 IN ({placeholders})",
        cases,
    )
    return {"series": series, "decades_global": decades, "countries": names}


@app.get("/analytics/country/{cca2}/services-breakdown")
def services_breakdown(cca2: str):
    """Breakdown of services exports: Travel, Transport, Finance, ICT, Other."""
    cca2 = cca2.upper()
    rows = query(
        """
        SELECT year,
               MAX(value) FILTER (WHERE indicator_code='BX.GSR.TRVL.ZS') AS travel_pct,
               MAX(value) FILTER (WHERE indicator_code='BX.GSR.TRAN.ZS') AS transport_pct,
               MAX(value) FILTER (WHERE indicator_code='BX.GSR.INSF.ZS') AS finance_pct,
               MAX(value) FILTER (WHERE indicator_code='BX.GSR.CCIS.ZS') AS ict_pct,
               MAX(value) FILTER (WHERE indicator_code='BX.GSR.GNFS.CD') AS total_services_usd
        FROM   country_indicators
        WHERE  cca2 = %s
          AND  indicator_code IN ('BX.GSR.TRVL.ZS','BX.GSR.TRAN.ZS','BX.GSR.INSF.ZS','BX.GSR.CCIS.ZS','BX.GSR.GNFS.CD')
        GROUP  BY year ORDER BY year
        """, (cca2,)
    )
    return {"series": rows}


@app.get("/analytics/country/{cca2}/investment-profile")
def investment_profile(cca2: str):
    """Country's investment composition: military, health, R&D, FDI, capital formation, high-tech."""
    cca2 = cca2.upper()
    codes = ['MS.MIL.XPND.GD.ZS','SH.XPD.GHED.GD.ZS','SE.XPD.TOTL.GD.ZS','GB.XPD.RSDV.GD.ZS',
             'TX.VAL.TECH.MF.ZS','IP.PAT.RESD','IT.CEL.SETS.P2','IT.NET.USER.ZS',
             'NE.GDI.TOTL.ZS','BX.KLT.DINV.WD.GD.ZS']
    investments = []
    for code in codes:
        v = query(
            "SELECT indicator_name, year, value FROM country_indicators "
            "WHERE cca2=%s AND indicator_code=%s AND value IS NOT NULL "
            "ORDER BY year DESC LIMIT 1",
            (cca2, code),
        )
        if v:
            investments.append({
                "indicator": code,
                "name":      v[0]["indicator_name"],
                "year":      v[0]["year"],
                "value":     float(v[0]["value"]),
            })
    return {"investments": investments}


@app.get("/case-study/spain")
def case_study_spain():
    """Master endpoint for the full Spain case study (long-form report)."""
    cca2 = "ES"
    # 1. Identity
    country = query("SELECT * FROM countries WHERE cca2='ES'")[0]
    # 2. Composite scores
    archetype     = query("SELECT * FROM country_archetype WHERE cca2='ES'")
    strength      = query("SELECT * FROM country_economic_strength WHERE cca2='ES'")
    vulnerability = query("SELECT * FROM external_vulnerability_index WHERE cca2='ES'")
    convergence   = query("SELECT * FROM convergence_metrics WHERE cca2='ES'")
    pp            = query("SELECT * FROM country_purchasing_power WHERE cca2='ES'")
    zone          = query("SELECT * FROM country_zone_position WHERE cca2='ES'")
    # 3. Crisis history
    crises = query(
        """
        SELECT w.hito_id, h.nombre_evento, h.tipo_shock, h.anio_inicio, h.anio_fin,
               w.rank_pre, w.rank_post, w.rank_change, w.gdp_change_pct, w.verdict,
               ci.gdp_growth_min, ci.unemployment_max, ci.inflation_max,
               cr.years_to_recover, cr.pct_drop
        FROM   crisis_winners_losers w
        JOIN   hitos_historicos h ON h.id=w.hito_id
        JOIN   country_crisis_impact ci ON ci.cca2=w.cca2 AND ci.hito_id=w.hito_id
        LEFT   JOIN country_crisis_recovery cr ON cr.cca2=w.cca2 AND cr.hito_id=w.hito_id
        WHERE  w.cca2='ES' ORDER BY h.anio_inicio
        """
    )
    # 4. Euro impact (Spain row from euro_impact_analysis + averages)
    euro_es     = query("SELECT * FROM euro_impact_analysis WHERE cca2='ES'")
    euro_others = query(
        """
        SELECT cca2, name, flag_emoji, in_eurozone, inflation_pre, inflation_post,
               gdp_growth_pre, gdp_growth_post, curr_acc_pre, curr_acc_post,
               govt_debt_pre, govt_debt_post, unemp_pre, unemp_post
        FROM euro_impact_analysis WHERE cca2 IN ('ES','DE','FR','IT','PT','GR','GB','SE','PL','CH')
        ORDER BY in_eurozone DESC, cca2
        """
    )
    # 5. North vs South Europe comparison (latest)
    europe_comparison = query(
        """
        SELECT c.cca2, c.name, c.flag_emoji, c.subregion,
               c.latest_gdp_per_capita, c.latest_unemployment, c.latest_inflation,
               c.latest_gdp_growth,
               es.strength_score, v.vulnerability_score,
               pp.cost_of_living_index,
               (SELECT value FROM country_indicators WHERE cca2=c.cca2 AND indicator_code='GC.DOD.TOTL.GD.ZS' AND value IS NOT NULL ORDER BY year DESC LIMIT 1) AS govt_debt
        FROM   countries_clean c
        LEFT   JOIN country_economic_strength es ON es.cca2=c.cca2
        LEFT   JOIN external_vulnerability_index v ON v.cca2=c.cca2
        LEFT   JOIN country_purchasing_power pp ON pp.cca2=c.cca2
        WHERE  c.cca2 IN ('ES','PT','IT','GR','FR','DE','NL','SE','DK','FI','NO','BE','AT')
        ORDER  BY c.latest_gdp_per_capita DESC
        """
    )
    # 6. Recent trends (3-year)
    recent = []
    indicators_map = {
        "SP.POP.TOTL":"Población","NY.GDP.PCAP.CD":"PIB/cápita USD","NY.GDP.PCAP.PP.CD":"PIB/cápita PPP",
        "NY.GDP.MKTP.KD.ZG":"Crecimiento PIB %","SL.UEM.TOTL.ZS":"Paro %","FP.CPI.TOTL.ZG":"Inflación %",
        "BN.CAB.XOKA.GD.ZS":"Cuenta corriente %","GC.DOD.TOTL.GD.ZS":"Deuda gob. %","IT.NET.USER.ZS":"Internet %",
        "NE.EXP.GNFS.ZS":"Exportaciones %",
    }
    for code, label in indicators_map.items():
        rows = query(
            "SELECT year, value FROM country_indicators WHERE cca2='ES' AND indicator_code=%s "
            "AND value IS NOT NULL ORDER BY year DESC LIMIT 5", (code,),
        )
        if rows:
            rows = sorted(rows, key=lambda r: r["year"])
            three_y_ago = next((r for r in rows if r["year"] == rows[-1]["year"] - 3), None)
            delta_pct = ((rows[-1]["value"] / three_y_ago["value"]) - 1) * 100 if three_y_ago and three_y_ago["value"] != 0 else None
            recent.append({
                "code": code, "label": label,
                "latest_year": rows[-1]["year"], "latest_value": float(rows[-1]["value"]),
                "delta_pct_3y": round(delta_pct, 1) if delta_pct is not None else None,
                "series": [{"year": r["year"], "value": float(r["value"])} for r in rows],
            })
    # 7. ML risk profile
    risk = None
    if predictor.is_ready():
        try:
            import pandas as pd
            conn = _connect()
            df = pd.read_sql(
                "SELECT * FROM ml_features_wide WHERE cca2='ES' ORDER BY year DESC LIMIT 1", conn,
            )
            conn.close()
            if not df.empty:
                proba = float(predictor.predict_proba(df)[0])
                risk = {"crisis_proba": round(proba, 3),
                        "interpretation": "low" if proba < 0.3 else "moderate" if proba < 0.6 else "high"}
        except Exception:
            pass
    # 8. Currency history (selected episodes)
    currency = query(
        """
        SELECT year, lcu_per_usd, yoy_pct, is_devaluation, euro_adoption_year
        FROM currency_history WHERE cca2='ES' AND (is_devaluation OR is_appreciation OR year IN (1973,1986,1999,2008,2020))
        ORDER BY year
        """
    )
    # 9. Sector composition latest
    sectors = query(
        """
        SELECT year,
               MAX(value) FILTER (WHERE indicator_code='NV.SRV.TOTL.ZS') AS services,
               MAX(value) FILTER (WHERE indicator_code='NV.IND.TOTL.ZS') AS industry,
               MAX(value) FILTER (WHERE indicator_code='NV.AGR.TOTL.ZS') AS agriculture
        FROM country_indicators WHERE cca2='ES'
          AND indicator_code IN ('NV.SRV.TOTL.ZS','NV.IND.TOTL.ZS','NV.AGR.TOTL.ZS')
        GROUP BY year ORDER BY year
        """
    )
    # 10. Cluster info
    cluster_info = None
    if clusterer.is_ready():
        cl = clusterer.get_country(cca2)
        if cl:
            centroid = clusterer.centroids[cl["cluster"]]
            cluster_info = {"id": cl["cluster"], "label": centroid["label"], "n_members": centroid["n_members"]}

    return {
        "country":           country,
        "archetype":         archetype[0] if archetype else None,
        "strength":          strength[0] if strength else None,
        "vulnerability":     vulnerability[0] if vulnerability else None,
        "convergence":       convergence[0] if convergence else None,
        "purchasing_power":  pp[0] if pp else None,
        "zone":              zone[0] if zone else None,
        "crises":            crises,
        "euro_self":         euro_es[0] if euro_es else None,
        "euro_others":       euro_others,
        "europe_comparison": europe_comparison,
        "recent_trends":     recent,
        "ml_risk":           risk,
        "cluster":           cluster_info,
        "currency_events":   currency,
        "sectors_history":   sectors,
    }


@app.get("/studies/spain-case")
def study_spain_case():
    """Unified narrative: Spain across all dimensions and crises."""
    cca2 = "ES"
    return {
        "country":    query("SELECT * FROM countries WHERE cca2='ES'")[0],
        "archetype":  query("SELECT * FROM country_archetype WHERE cca2='ES'")[0] if query("SELECT 1 FROM country_archetype WHERE cca2='ES'") else None,
        "convergence":query("SELECT * FROM convergence_metrics WHERE cca2='ES'")[0] if query("SELECT 1 FROM convergence_metrics WHERE cca2='ES'") else None,
        "vulnerability": query("SELECT * FROM external_vulnerability_index WHERE cca2='ES'")[0] if query("SELECT 1 FROM external_vulnerability_index WHERE cca2='ES'") else None,
        "strength":   query("SELECT * FROM country_economic_strength WHERE cca2='ES'")[0] if query("SELECT 1 FROM country_economic_strength WHERE cca2='ES'") else None,
        "purchasing_power": query("SELECT * FROM country_purchasing_power WHERE cca2='ES'")[0] if query("SELECT 1 FROM country_purchasing_power WHERE cca2='ES'") else None,
        "zone":       query("SELECT * FROM country_zone_position WHERE cca2='ES'")[0] if query("SELECT 1 FROM country_zone_position WHERE cca2='ES'") else None,
        "peers":      query("SELECT * FROM country_peers WHERE cca2='ES' ORDER BY similarity_distance LIMIT 5"),
        "crises":     query("SELECT w.*, h.tipo_shock FROM crisis_winners_losers w JOIN hitos_historicos h ON h.id=w.hito_id WHERE w.cca2='ES' ORDER BY w.hito_id"),
        "budget":     query("SELECT * FROM household_budget_breakdown WHERE cca2='ES'")[0] if query("SELECT 1 FROM household_budget_breakdown WHERE cca2='ES'") else None,
    }


# ── Analytics: global hero stats + factbox ───────────────────────────────────

@app.get("/analytics/global-stats")
def global_stats():
    """Aggregated world stats + auto-generated facts for the Global Overview hero."""
    # Latest world GDP totals (sum of nominal GDP per country × pop, proxied by GDP per cap × pop)
    gdp_totals = query(
        """
        WITH latest_gdp AS (
            SELECT DISTINCT ON (cca2) cca2, value AS gdp_pcap
            FROM   country_indicators
            WHERE  indicator_code = 'NY.GDP.PCAP.CD' AND value IS NOT NULL
            ORDER  BY cca2, year DESC
        )
        SELECT
            SUM(c.population * lg.gdp_pcap)::numeric           AS world_gdp,
            SUM(c.population * lg.gdp_pcap) FILTER (WHERE c.cca2='US')::numeric AS us_gdp,
            SUM(c.population * lg.gdp_pcap) FILTER (WHERE c.cca2='CN')::numeric AS cn_gdp,
            SUM(c.population * lg.gdp_pcap) FILTER (WHERE c.region='Europe')::numeric AS eu_region_gdp,
            SUM(c.population) FILTER (WHERE c.cca2 IN ('AT','BE','CY','EE','FI','FR','DE','GR','IE','IT','LV','LT','LU','MT','NL','PT','SK','SI','ES','HR'))::numeric AS eurozone_pop,
            SUM(c.population)::numeric                          AS world_pop
        FROM   countries c
        LEFT   JOIN latest_gdp lg ON lg.cca2 = c.cca2
        WHERE  lg.gdp_pcap IS NOT NULL
        """
    )[0]

    # Counts
    n_countries  = query("SELECT COUNT(*) AS n FROM countries_clean")[0]["n"]
    n_hitos      = query("SELECT COUNT(*) AS n FROM hitos_historicos")[0]["n"]
    n_archetypes = query("SELECT COUNT(DISTINCT archetype) AS n FROM country_archetype")[0]["n"]

    # ── Facts (auto-generated) ────────────────────────────────────────────────
    facts = []

    # Highest savings rate
    r = query("SELECT cca2, food_pct, housing_pct, savings_pct FROM household_budget_breakdown ORDER BY savings_pct DESC LIMIT 1")
    if r:
        c = query("SELECT name, flag_emoji FROM countries WHERE cca2 = %s", (r[0]["cca2"],))[0]
        facts.append({
            "icon": "💰", "label_es": "Mayor tasa de ahorro",
            "label_en": "Highest savings rate",
            "value":    f"{c['flag_emoji']} {c['name']} — {r[0]['savings_pct']}%",
            "extra_es": "del ingreso disponible",
            "extra_en": "of disposable income",
        })

    # Largest CONFIRMED hyperinflation devaluation (excludes data errors / denomination changes)
    r = query("""
        SELECT cca2, year, yoy_pct
        FROM   currency_history
        WHERE  yoy_pct IS NOT NULL
          AND  is_extreme_jump = FALSE  -- exclude denomination errors
        ORDER  BY yoy_pct DESC LIMIT 1
    """)
    # If no normal devaluation found, fall back to extreme jump but mark as such
    if not r:
        r = query("""
            SELECT cca2, year, yoy_pct
            FROM   currency_history
            WHERE  yoy_pct IS NOT NULL
            ORDER  BY yoy_pct DESC LIMIT 1
        """)
    # Also fetch top REAL hyperinflations (confirmed cases) for richer fact
    hyper = query("""
        SELECT ch.cca2, c.name, c.flag_emoji, ch.year, ch.yoy_pct
        FROM   currency_history ch JOIN countries c ON c.cca2=ch.cca2
        WHERE  ch.yoy_pct > 1000 AND ch.is_extreme_jump = TRUE
        ORDER  BY ch.yoy_pct DESC LIMIT 1
    """)
    if r:
        c = query("SELECT name, flag_emoji FROM countries WHERE cca2 = %s", (r[0]["cca2"],))[0]
        facts.append({
            "icon": "📉", "label_es": "Mayor devaluación anual",
            "label_en": "Largest annual devaluation",
            "value":    f"{c['flag_emoji']} {c['name']} ({r[0]['year']}) — +{r[0]['yoy_pct']:.0f}%",
            "extra_es": "moneda local vs USD",
            "extra_en": "local currency vs USD",
        })
    if hyper:
        facts.append({
            "icon": "💥", "label_es": "Mayor hiperinflación documentada",
            "label_en": "Worst documented hyperinflation",
            "value":    f"{hyper[0]['flag_emoji']} {hyper[0]['name']} ({hyper[0]['year']}) — +{hyper[0]['yoy_pct']:.0f}%",
            "extra_es": "salto > 100x en un año",
            "extra_en": "currency lost > 99% value in one year",
        })

    # Most expensive country
    r = query("SELECT cca2, name, flag_emoji, cost_of_living_index FROM country_purchasing_power ORDER BY cost_of_living_index DESC LIMIT 1")
    if r:
        facts.append({
            "icon": "🏔", "label_es": "País más caro",
            "label_en": "Most expensive country",
            "value":    f"{r[0]['flag_emoji']} {r[0]['name']} — {r[0]['cost_of_living_index']:.0f} (US=100)",
            "extra_es": "índice coste de vida",
            "extra_en": "cost-of-living index",
        })

    # Cheapest country
    r = query("SELECT cca2, name, flag_emoji, cost_of_living_index FROM country_purchasing_power ORDER BY cost_of_living_index ASC LIMIT 1")
    if r:
        facts.append({
            "icon": "🪙", "label_es": "País más barato",
            "label_en": "Cheapest country",
            "value":    f"{r[0]['flag_emoji']} {r[0]['name']} — {r[0]['cost_of_living_index']:.0f} (US=100)",
            "extra_es": "índice coste de vida",
            "extra_en": "cost-of-living index",
        })

    # Strongest converger (60y)
    r = query("SELECT cca2, name, flag_emoji, change_60y FROM convergence_metrics WHERE change_60y IS NOT NULL ORDER BY change_60y DESC LIMIT 1")
    if r:
        facts.append({
            "icon": "🚀", "label_es": "Mayor convergencia 60y",
            "label_en": "Top converger 60y",
            "value":    f"{r[0]['flag_emoji']} {r[0]['name']} — +{r[0]['change_60y']:.0f} puntos",
            "extra_es": "% PIB EE.UU. ganados",
            "extra_en": "% US GDP gained",
        })

    # Strongest diverger
    r = query("SELECT cca2, name, flag_emoji, change_60y FROM convergence_metrics WHERE change_60y IS NOT NULL ORDER BY change_60y ASC LIMIT 1")
    if r and r[0]["change_60y"] < 0:
        facts.append({
            "icon": "📉", "label_es": "Mayor divergencia 60y",
            "label_en": "Top diverger 60y",
            "value":    f"{r[0]['flag_emoji']} {r[0]['name']} — {r[0]['change_60y']:.0f} puntos",
            "extra_es": "% PIB EE.UU. perdidos",
            "extra_en": "% US GDP lost",
        })

    # Strongest economy by score
    r = query("SELECT cca2, name, flag_emoji, strength_score FROM country_economic_strength ORDER BY strength_score DESC LIMIT 1")
    if r:
        facts.append({
            "icon": "💪", "label_es": "Economía más fuerte (score)",
            "label_en": "Strongest economy (score)",
            "value":    f"{r[0]['flag_emoji']} {r[0]['name']} — {r[0]['strength_score']}/100",
            "extra_es": "nivel+estabilidad+resiliencia+diversificación",
            "extra_en": "level+stability+resilience+diversification",
        })

    # Highest food share (poorest by Engel's law)
    r = query("SELECT cca2, food_pct FROM household_budget_breakdown ORDER BY food_pct DESC LIMIT 1")
    if r:
        c = query("SELECT name, flag_emoji FROM countries WHERE cca2 = %s", (r[0]["cca2"],))[0]
        facts.append({
            "icon": "🍞", "label_es": "Mayor % gasto en alimentos",
            "label_en": "Highest food share",
            "value":    f"{c['flag_emoji']} {c['name']} — {r[0]['food_pct']}%",
            "extra_es": "ley de Engel: a menor renta, más % en comida",
            "extra_en": "Engel's law: lower income → higher food share",
        })

    return {
        "counts": {
            "countries":   n_countries,
            "indicators":  32,
            "years":       60,
            "hitos":       n_hitos,
            "archetypes":  n_archetypes,
        },
        "world": {
            "world_gdp":    float(gdp_totals["world_gdp"])    if gdp_totals["world_gdp"]    else None,
            "us_pct":       float(gdp_totals["us_gdp"])  / float(gdp_totals["world_gdp"]) * 100 if gdp_totals["us_gdp"] and gdp_totals["world_gdp"] else None,
            "cn_pct":       float(gdp_totals["cn_gdp"])  / float(gdp_totals["world_gdp"]) * 100 if gdp_totals["cn_gdp"] and gdp_totals["world_gdp"] else None,
            "eu_pct":       float(gdp_totals["eu_region_gdp"]) / float(gdp_totals["world_gdp"]) * 100 if gdp_totals["eu_region_gdp"] and gdp_totals["world_gdp"] else None,
            "world_pop":    int(gdp_totals["world_pop"])     if gdp_totals["world_pop"]    else None,
            "eurozone_pop": int(gdp_totals["eurozone_pop"])  if gdp_totals["eurozone_pop"] else None,
        },
        "facts":  facts,
    }


# ── Analytics: country deep view (Etapa F) ───────────────────────────────────

@app.get("/analytics/country/{cca2}/peers")
def country_peers(cca2: str):
    """Top 5 most similar countries by archetype + GDP/cap + region."""
    return query(
        """
        SELECT peer_cca2, peer_name, peer_region, peer_subregion,
               peer_gdp_per_capita, peer_archetype, peer_strength,
               similarity_pct, peer_flag
        FROM   country_peers
        WHERE  cca2 = %s
        ORDER  BY similarity_distance ASC
        """,
        (cca2.upper(),),
    )


@app.get("/analytics/country/{cca2}/zone")
def country_zone(cca2: str):
    """
    Subregion context: country's rank within zone + subregion averages +
    all peers in the same subregion.
    """
    cca2 = cca2.upper()
    pos = query("SELECT * FROM country_zone_position WHERE cca2 = %s", (cca2,))
    if not pos:
        raise HTTPException(404, f"No zone data for '{cca2}'")
    p = pos[0]
    sub = query("SELECT * FROM subregion_stats WHERE subregion = %s", (p["subregion"],))
    members = query(
        """
        SELECT c.cca2, c.name, c.flag_emoji,
               c.latest_gdp_per_capita, c.latest_gdp_growth,
               c.latest_unemployment, c.latest_inflation,
               c.latest_life_expectancy, c.population,
               a.archetype, es.strength_score, v.vulnerability_score
        FROM   countries_clean c
        LEFT   JOIN country_archetype           a  ON a.cca2  = c.cca2
        LEFT   JOIN country_economic_strength   es ON es.cca2 = c.cca2
        LEFT   JOIN external_vulnerability_index v ON v.cca2  = c.cca2
        WHERE  c.subregion = %s
        ORDER  BY c.latest_gdp_per_capita DESC NULLS LAST
        """,
        (p["subregion"],),
    )
    return {
        "position":         pos[0],
        "subregion_stats":  sub[0] if sub else None,
        "members":          members,
    }


@app.get("/analytics/country/{cca2}/trajectory")
def country_trajectory(cca2: str):
    """
    Long-term historical trajectory: GDP/cap series + decade summaries +
    sector composition evolution + key inflection metrics.
    """
    cca2 = cca2.upper()
    gdp = query(
        """
        SELECT year, value
        FROM   country_indicators
        WHERE  cca2 = %s AND indicator_code = 'NY.GDP.PCAP.CD'
        ORDER  BY year
        """,
        (cca2,),
    )
    decades = query(
        """
        SELECT decade, gdp_per_capita, gdp_growth, unemployment, inflation,
               services_va, industry_va, current_account, gini, govt_debt
        FROM   decade_summary WHERE cca2 = %s ORDER BY decade
        """,
        (cca2,),
    )
    # Sector composition over time
    sectors = query(
        """
        SELECT year,
               MAX(CASE WHEN indicator_code='NV.SRV.TOTL.ZS' THEN value END) AS services,
               MAX(CASE WHEN indicator_code='NV.IND.TOTL.ZS' THEN value END) AS industry,
               MAX(CASE WHEN indicator_code='NV.AGR.TOTL.ZS' THEN value END) AS agriculture
        FROM   country_indicators
        WHERE  cca2 = %s
          AND  indicator_code IN ('NV.SRV.TOTL.ZS','NV.IND.TOTL.ZS','NV.AGR.TOTL.ZS')
        GROUP  BY year ORDER BY year
        """,
        (cca2,),
    )
    # Coverage info per indicator (transparency for heterogeneous WB data)
    coverage = query(
        """
        SELECT indicator_code, indicator_name, first_year, last_year, span_years, n_observations
        FROM   country_indicator_coverage
        WHERE  cca2 = %s
          AND  indicator_code IN (
            'NY.GDP.PCAP.CD','NV.SRV.TOTL.ZS','NV.IND.TOTL.ZS','NV.AGR.TOTL.ZS',
            'SL.SRV.EMPL.ZS','SL.IND.EMPL.ZS','SL.AGR.EMPL.ZS','SL.UEM.TOTL.ZS',
            'FP.CPI.TOTL.ZG','SP.POP.TOTL','SP.DYN.LE00.IN','SP.URB.TOTL.IN.ZS',
            'NE.EXP.GNFS.ZS','NE.IMP.GNFS.ZS','BN.CAB.XOKA.GD.ZS',
            'GC.DOD.TOTL.GD.ZS','SI.POV.GINI'
          )
        """,
        (cca2,),
    )

    return {
        "gdp_per_capita": gdp,
        "decades":        decades,
        "sectors":        sectors,
        "coverage":       {row["indicator_code"]: row for row in coverage},
    }


@app.get("/analytics/country/{cca2}/currency")
def country_currency(cca2: str):
    """
    Historical exchange rate (LCU/USD) with YoY% and detected devaluation events.
    For eurozone members, includes euro adoption metadata.
    """
    cca2 = cca2.upper()
    rows = query(
        """
        SELECT year, lcu_per_usd, yoy_pct, is_devaluation, is_appreciation, euro_adoption_year
        FROM   currency_history
        WHERE  cca2 = %s
        ORDER  BY year
        """,
        (cca2,),
    )
    if not rows:
        raise HTTPException(404, f"No currency data for '{cca2}'")

    # Country metadata (currency code from countries table)
    meta_rows = query(
        "SELECT name, currencies, flag_emoji FROM countries WHERE cca2 = %s",
        (cca2,),
    )
    meta = meta_rows[0] if meta_rows else {}

    # All FX shocks in chronological order, enriched with historical milestone
    # context (which crisis was happening that year). Lets the user infer cause.
    hitos = query(
        "SELECT id, nombre_evento, tipo_shock, anio_inicio, anio_fin "
        "FROM hitos_historicos ORDER BY anio_inicio"
    )

    def find_hito(year: int):
        for h in hitos:
            if h["anio_inicio"] <= year <= h["anio_fin"]:
                return {
                    "id":   h["id"],
                    "name": h["nombre_evento"],
                    "type": h["tipo_shock"],
                }
        return None

    fx_events = []
    for r in rows:
        if not (r["is_devaluation"] or r["is_appreciation"]):
            continue
        fx_events.append({
            **r,
            "event_type": "devaluation" if r["is_devaluation"] else "appreciation",
            "hito":        find_hito(r["year"]),
        })
    fx_events.sort(key=lambda e: e["year"])

    # Keep legacy keys for backwards-compat with current UI
    devaluations  = [e for e in fx_events if e["event_type"] == "devaluation"]
    appreciations = [e for e in fx_events if e["event_type"] == "appreciation"]

    euro_year = rows[0].get("euro_adoption_year") if rows else None

    return {
        "country":            {**meta, "cca2": cca2},
        "series":             rows,
        "fx_events":          fx_events,
        "devaluations":       devaluations,
        "appreciations":      appreciations,
        "euro_adoption_year": euro_year,
        "years_covered":      [rows[0]["year"], rows[-1]["year"]] if rows else None,
    }


@app.get("/analytics/country/{cca2}/purchasing-power")
def country_purchasing_power(cca2: str):
    """
    Real living standards in PPP-adjusted terms.
    Returns GDP/cap nominal & PPP, household consumption per capita PPP,
    plus ratios vs USA and PPP premium (how much cheaper than USA).
    """
    cca2 = cca2.upper()
    rows = query(
        "SELECT * FROM country_purchasing_power WHERE cca2 = %s",
        (cca2,),
    )
    if not rows:
        raise HTTPException(404, f"No purchasing power data for '{cca2}'")
    me = rows[0]
    peers = query(
        """
        SELECT cca2, name, flag_emoji,
               gdp_per_capita_ppp,
               household_consumption_per_capita_ppp,
               pct_us_gdp_ppp, pct_us_consumption
        FROM   country_purchasing_power
        WHERE  region = %s AND cca2 != %s
        ORDER  BY gdp_per_capita_ppp DESC NULLS LAST
        LIMIT 10
        """,
        (me["region"], cca2),
    )
    # Budget breakdown (curated table — coverage limited to top economies)
    breakdown_rows = query(
        "SELECT * FROM household_budget_breakdown WHERE cca2 = %s",
        (cca2,),
    )
    breakdown = breakdown_rows[0] if breakdown_rows else None
    return {"me": me, "regional_peers": peers, "budget_breakdown": breakdown}


@app.get("/analytics/purchasing-power/ranking")
def purchasing_power_ranking(
    metric: str = Query("gdp_ppp", regex="^(gdp_ppp|consumption|gni_ppp)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    limit: int = 25,
):
    """Global ranking by purchasing power metric."""
    col = {
        "gdp_ppp":     "gdp_per_capita_ppp",
        "consumption": "household_consumption_per_capita_ppp",
        "gni_ppp":     "gni_per_capita_ppp",
    }[metric]
    direction = "DESC" if order == "desc" else "ASC"
    return query(
        f"""
        SELECT cca2, name, region, flag_emoji,
               gdp_per_capita_nominal, gdp_per_capita_ppp,
               household_consumption_per_capita_ppp, gni_per_capita_ppp,
               pct_us_gdp_ppp, pct_us_consumption, ppp_premium_pct
        FROM   country_purchasing_power
        WHERE  {col} IS NOT NULL
        ORDER  BY {col} {direction}
        LIMIT  %s
        """,
        (limit,),
    )


@app.get("/analytics/country/{cca2}/recent-trends")
def country_recent_trends(cca2: str):
    """
    Last 5 years of key indicators per country + 3-year delta + trend direction.
    Used in the country Overview to show recent momentum.
    """
    cca2 = cca2.upper()
    # 10 key indicators (mix: macro + sectoral + financial + digital)
    indicators = {
        "SP.POP.TOTL":       {"label_es": "Población",         "label_en": "Population",       "higher_is_better": True,  "unit": "people"},
        "NY.GDP.PCAP.CD":    {"label_es": "PIB/cápita (USD)",   "label_en": "GDP/capita (USD)", "higher_is_better": True,  "unit": "usd"},
        "NY.GDP.PCAP.PP.CD": {"label_es": "PIB/cápita PPP",     "label_en": "GDP/capita PPP",   "higher_is_better": True,  "unit": "usd"},
        "NY.GDP.MKTP.KD.ZG": {"label_es": "Crecimiento PIB %",  "label_en": "GDP growth %",     "higher_is_better": True,  "unit": "%"},
        "SL.UEM.TOTL.ZS":    {"label_es": "Paro %",             "label_en": "Unemployment %",   "higher_is_better": False, "unit": "%"},
        "FP.CPI.TOTL.ZG":    {"label_es": "Inflación %",        "label_en": "Inflation %",      "higher_is_better": False, "unit": "%"},
        "BN.CAB.XOKA.GD.ZS": {"label_es": "Cuenta corr. %PIB",  "label_en": "Current acc. %GDP","higher_is_better": True,  "unit": "%"},
        "GC.DOD.TOTL.GD.ZS": {"label_es": "Deuda gob. %PIB",    "label_en": "Govt debt %GDP",   "higher_is_better": False, "unit": "%"},
        "IT.NET.USER.ZS":    {"label_es": "Usuarios internet %", "label_en": "Internet users %","higher_is_better": True,  "unit": "%"},
        "NE.EXP.GNFS.ZS":    {"label_es": "Exportaciones %PIB", "label_en": "Exports %GDP",     "higher_is_better": True,  "unit": "%"},
    }

    trends = []
    for code, meta in indicators.items():
        rows = query(
            "SELECT year, value FROM country_indicators "
            "WHERE cca2=%s AND indicator_code=%s AND value IS NOT NULL "
            "ORDER BY year DESC LIMIT 5",
            (cca2, code),
        )
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["year"])  # ascending
        series = [{"year": r["year"], "value": float(r["value"])} for r in rows]
        latest = series[-1]
        # 3-year delta: compare latest with value from 3 years ago (if available)
        three_y_ago = next((r for r in series if r["year"] == latest["year"] - 3), None)
        if three_y_ago and three_y_ago["value"] != 0:
            delta_abs = latest["value"] - three_y_ago["value"]
            delta_pct = (latest["value"] / three_y_ago["value"] - 1) * 100 if three_y_ago["value"] != 0 else None
        else:
            delta_abs = None
            delta_pct = None
        # Direction: trend over the whole series
        trend = None
        if len(series) >= 2:
            slope = (series[-1]["value"] - series[0]["value"]) / max(1, series[-1]["year"] - series[0]["year"])
            if abs(slope) < 0.001 * abs(series[-1]["value"]):
                trend = "flat"
            elif slope > 0:
                trend = "up"
            else:
                trend = "down"

        # Use 3-year delta direction (not linear trend) for good/bad classification.
        # Linear trend can mislead when there's an intermediate spike (inflation 2022).
        good_trend = None
        if delta_abs is not None:
            if meta["higher_is_better"]:
                good_trend = delta_abs > 0
            else:
                good_trend = delta_abs < 0

        trends.append({
            "indicator": code,
            "label_es":  meta["label_es"],
            "label_en":  meta["label_en"],
            "unit":      meta["unit"],
            "higher_is_better": meta["higher_is_better"],
            "series":    series,
            "latest":    latest,
            "delta_abs": round(delta_abs, 2) if delta_abs is not None else None,
            "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
            "trend":     trend,
            "good_trend":good_trend,
        })

    return {"country": cca2, "indicators": trends}


@app.get("/analytics/country/{cca2}/coverage")
def country_coverage(cca2: str):
    """Temporal coverage (first/last year) per indicator for a country.
    Useful for tooltips and heterogeneity transparency."""
    rows = query(
        """
        SELECT indicator_code, indicator_name, first_year, last_year, span_years, n_observations
        FROM   country_indicator_coverage
        WHERE  cca2 = %s
        ORDER  BY first_year, indicator_code
        """,
        (cca2.upper(),),
    )
    if not rows:
        raise HTTPException(404, f"No coverage data for '{cca2}'")
    return rows


@app.get("/analytics/subregion/{subregion}/stats")
def subregion_stats_endpoint(subregion: str):
    rows = query("SELECT * FROM subregion_stats WHERE subregion = %s", (subregion,))
    if not rows:
        raise HTTPException(404, f"Subregion '{subregion}' not found")
    return rows[0]


# ── Analytics: archetypes (listing & filters) ────────────────────────────────

@app.get("/analytics/archetypes")
def list_archetypes():
    """Distribution of archetypes worldwide."""
    return query(
        """
        SELECT archetype, COUNT(*) AS n_countries
        FROM   country_archetype
        GROUP  BY archetype
        ORDER  BY n_countries DESC
        """
    )


@app.get("/analytics/archetypes/{archetype}/countries")
def countries_by_archetype(archetype: str):
    """All countries of a given archetype."""
    return query(
        """
        SELECT a.cca2, a.name, a.region, a.flag_emoji,
               a.gdp_per_capita, a.services_va_pct, a.industry_va_pct, a.agriculture_va_pct,
               a.fuel_exports_pct, a.manuf_exports_pct, a.current_account_pct,
               c.pct_us_now, c.trajectory,
               v.vulnerability_score
        FROM   country_archetype a
        LEFT   JOIN convergence_metrics c ON c.cca2 = a.cca2
        LEFT   JOIN external_vulnerability_index v ON v.cca2 = a.cca2
        WHERE  a.archetype = %s
        ORDER  BY a.gdp_per_capita DESC NULLS LAST
        """,
        (archetype,),
    )


# ── Analytics: trade balance (producers vs consumers) — Etapa D4 ─────────────

@app.get("/analytics/trade-balance")
def trade_balance(order: str = Query("producers", regex="^(producers|consumers)$"), limit: int = 20):
    """
    Top producers (positive current account) or consumers (negative) by % GDP.
    Producers run trade surplus; consumers run deficit.
    """
    direction = "DESC" if order == "producers" else "ASC"
    return query(
        f"""
        SELECT a.cca2, a.name, a.region, a.flag_emoji, a.archetype,
               a.current_account_pct, a.fuel_exports_pct, a.manuf_exports_pct,
               a.gdp_per_capita
        FROM   country_archetype a
        WHERE  a.current_account_pct IS NOT NULL
        ORDER  BY a.current_account_pct {direction} NULLS LAST
        LIMIT  %s
        """,
        (limit,),
    )


# ── Analytics: cohorts (predefined country groups) — Etapa D3 ────────────────

COHORTS = {
    "piigs":         {"name": "PIIGS",         "members": ["PT","IT","IE","GR","ES"]},
    "asian_tigers":  {"name": "Asian Tigers",  "members": ["HK","SG","KR","TW"]},
    "brics":         {"name": "BRICS",         "members": ["BR","RU","IN","CN","ZA"]},
    "gcc":           {"name": "GCC Oil",       "members": ["SA","AE","QA","KW","BH","OM"]},
    "nordics":       {"name": "Nordics",       "members": ["DK","FI","IS","NO","SE"]},
    "g7":            {"name": "G7",            "members": ["US","UK","DE","FR","IT","CA","JP"]},
    "next_eleven":   {"name": "Next 11",       "members": ["BD","EG","ID","IR","MX","NG","PK","PH","TR","KR","VN"]},
}


@app.get("/analytics/cohorts")
def list_cohorts():
    """Lists predefined country groups."""
    return [{"id": k, "name": v["name"], "members": v["members"]} for k, v in COHORTS.items()]


@app.get("/analytics/cohorts/{cohort_id}")
def cohort_data(cohort_id: str):
    """Aggregate data for a cohort: members + strength + convergence + archetype."""
    cohort = COHORTS.get(cohort_id)
    if not cohort:
        raise HTTPException(404, f"Cohort '{cohort_id}' not found")
    placeholders = ",".join(["%s"] * len(cohort["members"]))
    rows = query(
        f"""
        SELECT c.cca2, c.name, c.region, c.flag_emoji, c.latest_gdp_per_capita,
               a.archetype, a.current_account_pct,
               cm.pct_us_now, cm.trajectory,
               v.vulnerability_score,
               es.strength_score, es.driver_level, es.driver_stability,
               es.driver_resilience, es.driver_diversification
        FROM   countries_clean c
        LEFT   JOIN country_archetype           a  ON a.cca2  = c.cca2
        LEFT   JOIN convergence_metrics         cm ON cm.cca2 = c.cca2
        LEFT   JOIN external_vulnerability_index v ON v.cca2  = c.cca2
        LEFT   JOIN country_economic_strength   es ON es.cca2 = c.cca2
        WHERE  c.cca2 IN ({placeholders})
        ORDER  BY es.strength_score DESC NULLS LAST
        """,
        cohort["members"],
    )
    return {"id": cohort_id, "name": cohort["name"], "members": rows}


# ── Analytics: hitos & global context ────────────────────────────────────────

@app.get("/analytics/hitos")
def list_hitos():
    """All historical milestones ordered chronologically."""
    return query("SELECT * FROM hitos_historicos ORDER BY anio_inicio")


@app.get("/analytics/hitos/{hito_id}")
def get_hito(hito_id: int):
    rows = query("SELECT * FROM hitos_historicos WHERE id = %s", (hito_id,))
    if not rows:
        raise HTTPException(404, f"Hito {hito_id} not found")
    return rows[0]


@app.get("/analytics/global-context")
def global_context(anio_inicio: int = Query(default=1995), anio_fin: int = Query(default=2030)):
    """Annual global macro context variables, optionally filtered by year range."""
    return query(
        "SELECT * FROM contexto_global_anual WHERE anio BETWEEN %s AND %s ORDER BY anio",
        (anio_inicio, anio_fin),
    )


@app.get("/analytics/global-context/{anio}")
def global_context_year(anio: int):
    rows = query("SELECT * FROM contexto_global_anual WHERE anio = %s", (anio,))
    if not rows:
        raise HTTPException(404, f"No global context data for year {anio}")
    return rows[0]


# ── Analytics: milestone correlation engine ───────────────────────────────────

@app.get("/analytics/correlation")
def milestone_correlation(
    country_code: str = Query(..., description="ISO 2-letter country code (e.g. ES, SA, US)"),
    hito_id:      int = Query(..., description="ID from hitos_historicos"),
):
    """
    Pearson correlation matrix between a country's macroeconomic variables
    and global context variables (Brent, VIX, Fed Funds, GPR, FinStress),
    scoped exclusively to the historical milestone's date range.

    Example:
      /analytics/correlation?country_code=SA&hito_id=5
      → How Saudi Arabia's economy correlated with oil price during the 2014-2016 crude shock.
    """
    # 1 ── Fetch milestone
    hito_rows = query(
        "SELECT id, anio_inicio, anio_fin, nombre_evento, tipo_shock, descripcion "
        "FROM hitos_historicos WHERE id = %s",
        (hito_id,),
    )
    if not hito_rows:
        raise HTTPException(404, f"Hito {hito_id} not found")
    hito = hito_rows[0]
    year_start, year_end = hito["anio_inicio"], hito["anio_fin"]

    # 2 ── Country indicators (scoped to hito period)
    ind_rows = query(
        """
        SELECT indicator_code, indicator_name, year, value
        FROM   country_indicators
        WHERE  cca2 = %s
          AND  year BETWEEN %s AND %s
        ORDER  BY indicator_code, year
        """,
        (country_code.upper(), year_start, year_end),
    )
    if not ind_rows:
        raise HTTPException(
            404,
            f"No indicator data for country '{country_code}' "
            f"in period {year_start}–{year_end}. "
            "Check the country code or try a different hito.",
        )

    # 3 ── Global macro context (scoped to hito period)
    ctx_rows = query(
        """
        SELECT anio, precio_brent_avg, indice_vix_avg, fed_funds_rate,
               geopolitical_risk_idx, stress_financiero
        FROM   contexto_global_anual
        WHERE  anio BETWEEN %s AND %s
        ORDER  BY anio
        """,
        (year_start, year_end),
    )
    if not ctx_rows:
        logger.warning(
            "No global context data for %d–%d — correlation will use country variables only",
            year_start, year_end,
        )

    # 4 ── Delegate to analytics service (pure pandas, no DB calls)
    try:
        result = compute_milestone_correlation(
            country_code=country_code,
            hito=hito,
            ind_rows=ind_rows,
            ctx_rows=ctx_rows,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return result


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/")
def dashboard():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
