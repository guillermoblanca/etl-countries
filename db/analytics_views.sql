-- =============================================================================
-- Analytics Views — applied after ETL load (not in init.sql)
-- =============================================================================
-- These views require populated data. Re-run after each ETL to refresh.
-- Executed by etl/main.py at the end of the pipeline.
-- =============================================================================


-- ── countries_clean ──────────────────────────────────────────────────────────
-- Excludes territories without economic coverage (Antarctic, micro-states).
-- Threshold: ≥ 8 distinct indicators with data.
-- Use this in place of `countries` for rankings, averages, top-N queries.
DROP VIEW IF EXISTS countries_clean CASCADE;
CREATE VIEW countries_clean AS
SELECT c.*
FROM   countries c
WHERE  EXISTS (
    SELECT 1
    FROM   country_indicators ci
    WHERE  ci.cca2 = c.cca2
    GROUP  BY ci.cca2
    HAVING COUNT(DISTINCT ci.indicator_code) >= 8
);


-- ── country_yoy ──────────────────────────────────────────────────────────────
-- Year-over-year change per country × indicator. Speeds up shock analysis.
DROP VIEW IF EXISTS country_yoy CASCADE;
CREATE VIEW country_yoy AS
SELECT
    cca2,
    indicator_code,
    indicator_name,
    year,
    value,
    LAG(value)  OVER (PARTITION BY cca2, indicator_code ORDER BY year) AS prev_value,
    value - LAG(value)  OVER (PARTITION BY cca2, indicator_code ORDER BY year) AS abs_change,
    CASE
        WHEN LAG(value) OVER (PARTITION BY cca2, indicator_code ORDER BY year) IS NULL
          OR LAG(value) OVER (PARTITION BY cca2, indicator_code ORDER BY year) = 0
        THEN NULL
        ELSE ROUND(
            ((value / LAG(value) OVER (PARTITION BY cca2, indicator_code ORDER BY year)) - 1) * 100,
            2
        )
    END AS yoy_pct
FROM country_indicators;


-- ── country_crisis_impact ────────────────────────────────────────────────────
-- For each (country, historical milestone), summarises the shock magnitude.
DROP MATERIALIZED VIEW IF EXISTS country_crisis_impact CASCADE;
CREATE MATERIALIZED VIEW country_crisis_impact AS
WITH base AS (
    SELECT
        c.cca2, c.name, c.region, c.flag_emoji,
        h.id   AS hito_id,
        h.nombre_evento,
        h.tipo_shock,
        h.anio_inicio,
        h.anio_fin,
        ci.indicator_code,
        ci.year,
        ci.value
    FROM   countries_clean c
    CROSS  JOIN hitos_historicos h
    JOIN   country_indicators ci
      ON   ci.cca2 = c.cca2
     AND   ci.year BETWEEN h.anio_inicio AND h.anio_fin
)
SELECT
    cca2, name, region, flag_emoji,
    hito_id, nombre_evento, tipo_shock,
    anio_inicio, anio_fin,
    -- GDP growth: min (caída más fuerte) durante el período
    MIN(value) FILTER (WHERE indicator_code = 'NY.GDP.MKTP.KD.ZG')   AS gdp_growth_min,
    AVG(value) FILTER (WHERE indicator_code = 'NY.GDP.MKTP.KD.ZG')   AS gdp_growth_avg,
    -- Unemployment: max (peor pico)
    MAX(value) FILTER (WHERE indicator_code = 'SL.UEM.TOTL.ZS')      AS unemployment_max,
    AVG(value) FILTER (WHERE indicator_code = 'SL.UEM.TOTL.ZS')      AS unemployment_avg,
    -- Inflation: max (pico inflacionario)
    MAX(value) FILTER (WHERE indicator_code = 'FP.CPI.TOTL.ZG')      AS inflation_max,
    AVG(value) FILTER (WHERE indicator_code = 'FP.CPI.TOTL.ZG')      AS inflation_avg,
    -- Life expectancy & urban: change endpoint-to-endpoint
    (MAX(value) FILTER (WHERE indicator_code = 'SP.DYN.LE00.IN' AND year = anio_fin))
      - (MAX(value) FILTER (WHERE indicator_code = 'SP.DYN.LE00.IN' AND year = anio_inicio))
                                                                     AS life_exp_delta,
    (MAX(value) FILTER (WHERE indicator_code = 'SP.URB.TOTL.IN.ZS' AND year = anio_fin))
      - (MAX(value) FILTER (WHERE indicator_code = 'SP.URB.TOTL.IN.ZS' AND year = anio_inicio))
                                                                     AS urban_delta
FROM   base
GROUP  BY cca2, name, region, flag_emoji,
         hito_id, nombre_evento, tipo_shock, anio_inicio, anio_fin;

CREATE INDEX IF NOT EXISTS idx_crisis_impact_cca2 ON country_crisis_impact (cca2);
CREATE INDEX IF NOT EXISTS idx_crisis_impact_hito ON country_crisis_impact (hito_id);


-- ── country_economic_strength ────────────────────────────────────────────────
-- Composite economic strength score (0-100), 4 explainable drivers.
DROP MATERIALIZED VIEW IF EXISTS country_economic_strength CASCADE;
CREATE MATERIALIZED VIEW country_economic_strength AS
WITH
-- ── Driver 1: LEVEL — GDP per capita percentile rank (latest available year)
gdp_pctile AS (
    SELECT cca2,
           PERCENT_RANK() OVER (ORDER BY latest_gdp_per_capita NULLS FIRST) * 100 AS pct
    FROM   countries_clean
    WHERE  latest_gdp_per_capita IS NOT NULL
),
-- ── Driver 2: STABILITY — inverse of GDP growth volatility (30-year stddev)
gdp_vol AS (
    SELECT cca2, STDDEV(value) AS sigma, COUNT(*) AS n
    FROM   country_indicators
    WHERE  indicator_code = 'NY.GDP.MKTP.KD.ZG'
    GROUP  BY cca2
    HAVING COUNT(*) >= 20
),
stability AS (
    SELECT cca2,
           -- Lower σ = higher rank = higher score (invert PERCENT_RANK)
           100 - (PERCENT_RANK() OVER (ORDER BY sigma) * 100) AS pct
    FROM   gdp_vol
),
-- ── Driver 3: RESILIENCE — recovery speed after 2009 + 2020 shocks
-- Measured as: avg of 3 years post-shock growth, normalised by country baseline
resilience AS (
    SELECT
        ci.cca2,
        AVG(
            CASE
                WHEN ci.year BETWEEN 2010 AND 2012 THEN ci.value
                WHEN ci.year BETWEEN 2021 AND 2023 THEN ci.value
            END
        ) AS post_shock_growth
    FROM country_indicators ci
    WHERE ci.indicator_code = 'NY.GDP.MKTP.KD.ZG'
      AND ci.year IN (2010, 2011, 2012, 2021, 2022, 2023)
    GROUP BY ci.cca2
),
resilience_pct AS (
    SELECT cca2,
           PERCENT_RANK() OVER (ORDER BY post_shock_growth NULLS FIRST) * 100 AS pct
    FROM   resilience
    WHERE  post_shock_growth IS NOT NULL
),
-- ── Driver 4: DIVERSIFICATION — 1 - Herfindahl-Hirschman of 3 sectoral VAs
-- Higher = more balanced economy across services/industry/agriculture
diversification AS (
    SELECT
        cca2,
        (latest_srv * latest_srv + latest_ind * latest_ind + latest_agr * latest_agr) / 10000.0 AS hhi
    FROM (
        SELECT
            cca2,
            MAX(CASE WHEN indicator_code = 'NV.SRV.TOTL.ZS' THEN value END) AS latest_srv,
            MAX(CASE WHEN indicator_code = 'NV.IND.TOTL.ZS' THEN value END) AS latest_ind,
            MAX(CASE WHEN indicator_code = 'NV.AGR.TOTL.ZS' THEN value END) AS latest_agr
        FROM (
            SELECT DISTINCT ON (cca2, indicator_code)
                   cca2, indicator_code, value
            FROM   country_indicators
            WHERE  indicator_code IN ('NV.SRV.TOTL.ZS','NV.IND.TOTL.ZS','NV.AGR.TOTL.ZS')
            ORDER  BY cca2, indicator_code, year DESC
        ) latest_per_sector
        GROUP BY cca2
    ) sector_pivot
    WHERE latest_srv IS NOT NULL AND latest_ind IS NOT NULL AND latest_agr IS NOT NULL
),
diversification_pct AS (
    SELECT cca2,
           -- Lower HHI = more diversified = higher score
           100 - (PERCENT_RANK() OVER (ORDER BY hhi) * 100) AS pct
    FROM   diversification
)
SELECT
    c.cca2, c.name, c.region, c.flag_emoji,
    c.latest_gdp_per_capita,
    ROUND(g.pct::numeric, 1)  AS driver_level,
    ROUND(s.pct::numeric, 1)  AS driver_stability,
    ROUND(r.pct::numeric, 1)  AS driver_resilience,
    ROUND(d.pct::numeric, 1)  AS driver_diversification,
    ROUND(
        (COALESCE(g.pct, 0) * 0.35
       + COALESCE(s.pct, 0) * 0.25
       + COALESCE(r.pct, 0) * 0.20
       + COALESCE(d.pct, 0) * 0.20)::numeric, 1
    ) AS strength_score
FROM   countries_clean c
LEFT   JOIN gdp_pctile          g ON g.cca2 = c.cca2
LEFT   JOIN stability           s ON s.cca2 = c.cca2
LEFT   JOIN resilience_pct      r ON r.cca2 = c.cca2
LEFT   JOIN diversification_pct d ON d.cca2 = c.cca2
WHERE  g.pct IS NOT NULL OR s.pct IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_strength_score ON country_economic_strength (strength_score DESC);
CREATE INDEX IF NOT EXISTS idx_strength_cca2  ON country_economic_strength (cca2);


-- ── country_crisis_recovery ──────────────────────────────────────────────────
-- Recovery analytics per (country, crisis): pre-shock GDP/cap, trough,
-- % drop and years until GDP/cap returns to pre-shock level.
DROP MATERIALIZED VIEW IF EXISTS country_crisis_recovery CASCADE;
CREATE MATERIALIZED VIEW country_crisis_recovery AS
WITH gdp AS (
    SELECT cca2, year, value AS gdp_cap
    FROM   country_indicators
    WHERE  indicator_code = 'NY.GDP.PCAP.CD'
      AND  value IS NOT NULL
),
windows AS (
    SELECT c.cca2, c.name, c.region, c.flag_emoji,
           h.id AS hito_id, h.nombre_evento, h.tipo_shock,
           h.anio_inicio, h.anio_fin
    FROM   countries_clean c
    CROSS  JOIN hitos_historicos h
),
pre AS (
    SELECT w.*, g.gdp_cap AS pre_shock_gdp
    FROM   windows w
    LEFT   JOIN gdp g ON g.cca2 = w.cca2 AND g.year = w.anio_inicio - 1
),
trough AS (
    SELECT w.cca2, w.hito_id,
           MIN(g.gdp_cap) AS trough_gdp,
           (ARRAY_AGG(g.year ORDER BY g.gdp_cap ASC))[1] AS trough_year
    FROM   windows w
    JOIN   gdp g ON g.cca2 = w.cca2
              AND g.year BETWEEN w.anio_inicio AND w.anio_fin
    GROUP  BY w.cca2, w.hito_id
),
recovery AS (
    SELECT p.cca2, p.hito_id, MIN(g.year) AS recovery_year
    FROM   pre p
    JOIN   gdp g ON g.cca2 = p.cca2
              AND g.year >= p.anio_inicio
              AND g.gdp_cap >= p.pre_shock_gdp
    WHERE  p.pre_shock_gdp IS NOT NULL
    GROUP  BY p.cca2, p.hito_id
)
SELECT
    p.cca2, p.name, p.region, p.flag_emoji,
    p.hito_id, p.nombre_evento, p.tipo_shock,
    p.anio_inicio, p.anio_fin,
    (p.anio_inicio - 1)::int                          AS pre_shock_year,
    ROUND(p.pre_shock_gdp::numeric, 2)                AS pre_shock_gdp,
    t.trough_year,
    ROUND(t.trough_gdp::numeric, 2)                   AS trough_gdp,
    CASE WHEN p.pre_shock_gdp > 0 AND t.trough_gdp IS NOT NULL
         THEN ROUND(((t.trough_gdp / p.pre_shock_gdp - 1) * 100)::numeric, 2)
         ELSE NULL END                                AS pct_drop,
    r.recovery_year,
    CASE WHEN r.recovery_year IS NOT NULL
         THEN r.recovery_year - (p.anio_inicio - 1)
         ELSE NULL END                                AS years_to_recover,
    (r.recovery_year IS NOT NULL)                     AS recovered
FROM   pre p
LEFT   JOIN trough   t ON t.cca2 = p.cca2 AND t.hito_id = p.hito_id
LEFT   JOIN recovery r ON r.cca2 = p.cca2 AND r.hito_id = p.hito_id;

CREATE INDEX IF NOT EXISTS idx_recovery_cca2 ON country_crisis_recovery (cca2);
CREATE INDEX IF NOT EXISTS idx_recovery_hito ON country_crisis_recovery (hito_id);


-- ════════════════════════════════════════════════════════════════════════════
-- ETAPA C — Capas analíticas derivadas
-- ════════════════════════════════════════════════════════════════════════════


-- ── C1: country_archetype ────────────────────────────────────────────────────
-- Classifies each country by economic structure based on latest available data
-- from trade composition, sectoral value-added and income level.
DROP MATERIALIZED VIEW IF EXISTS country_archetype CASCADE;
CREATE MATERIALIZED VIEW country_archetype AS
WITH latest AS (
    SELECT DISTINCT ON (cca2, indicator_code)
        cca2, indicator_code, value, year
    FROM   country_indicators
    WHERE  indicator_code IN (
        'TX.VAL.FUEL.ZS.UN','TX.VAL.MANF.ZS.UN',
        'NV.SRV.TOTL.ZS','NV.IND.TOTL.ZS','NV.AGR.TOTL.ZS',
        'NY.GDP.PCAP.CD','NE.EXP.GNFS.ZS','NE.IMP.GNFS.ZS',
        'BN.CAB.XOKA.GD.ZS'
    )
    AND value IS NOT NULL
    ORDER BY cca2, indicator_code, year DESC
),
pivoted AS (
    SELECT
        cca2,
        MAX(CASE WHEN indicator_code='TX.VAL.FUEL.ZS.UN' THEN value END) AS fuel_exp,
        MAX(CASE WHEN indicator_code='TX.VAL.MANF.ZS.UN' THEN value END) AS manuf_exp,
        MAX(CASE WHEN indicator_code='NV.SRV.TOTL.ZS'    THEN value END) AS va_srv,
        MAX(CASE WHEN indicator_code='NV.IND.TOTL.ZS'    THEN value END) AS va_ind,
        MAX(CASE WHEN indicator_code='NV.AGR.TOTL.ZS'    THEN value END) AS va_agr,
        MAX(CASE WHEN indicator_code='NY.GDP.PCAP.CD'    THEN value END) AS gdp_cap,
        MAX(CASE WHEN indicator_code='NE.EXP.GNFS.ZS'    THEN value END) AS exports,
        MAX(CASE WHEN indicator_code='NE.IMP.GNFS.ZS'    THEN value END) AS imports,
        MAX(CASE WHEN indicator_code='BN.CAB.XOKA.GD.ZS' THEN value END) AS curr_acc
    FROM   latest
    GROUP  BY cca2
),
scored AS (
    SELECT
        p.*,
        -- Sectoral concentration (Herfindahl)
        CASE WHEN va_srv IS NOT NULL AND va_ind IS NOT NULL AND va_agr IS NOT NULL
             THEN ((va_srv/100.0)^2 + (va_ind/100.0)^2 + (va_agr/100.0)^2)
             ELSE NULL
        END AS sector_hhi,
        -- Trade openness
        COALESCE(exports,0) + COALESCE(imports,0) AS openness
    FROM pivoted p
)
SELECT
    c.cca2, c.name, c.region, c.flag_emoji,
    ROUND(s.fuel_exp::numeric,1)   AS fuel_exports_pct,
    ROUND(s.manuf_exp::numeric,1)  AS manuf_exports_pct,
    ROUND(s.va_srv::numeric,1)     AS services_va_pct,
    ROUND(s.va_ind::numeric,1)     AS industry_va_pct,
    ROUND(s.va_agr::numeric,1)     AS agriculture_va_pct,
    ROUND(s.gdp_cap::numeric,0)    AS gdp_per_capita,
    ROUND(s.curr_acc::numeric,1)   AS current_account_pct,
    ROUND(s.openness::numeric,1)   AS trade_openness,
    ROUND(s.sector_hhi::numeric,3) AS sector_hhi,
    CASE
        -- Rules applied top-down (most specific first)
        WHEN s.fuel_exp >= 40 THEN 'Petrostate'
        WHEN (s.manuf_exp >= 60 AND s.va_ind >= 22)
          OR (s.manuf_exp >= 70)                            THEN 'Manufacturing exporter'
        WHEN s.va_agr   >= 15                               THEN 'Agricultural economy'
        WHEN s.va_srv   >= 75 AND s.gdp_cap >= 25000        THEN 'Advanced services'
        WHEN s.va_srv   >= 65 AND s.gdp_cap >= 15000        THEN 'Service-oriented'
        WHEN s.va_ind   >= 28                               THEN 'Industrial economy'
        WHEN s.sector_hhi < 0.45 AND s.gdp_cap >= 30000     THEN 'Diversified advanced'
        WHEN s.sector_hhi < 0.45                            THEN 'Diversified emerging'
        ELSE 'Mixed / Other'
    END AS archetype
FROM   countries_clean c
JOIN   scored s ON s.cca2 = c.cca2;

CREATE INDEX IF NOT EXISTS idx_archetype_cca2 ON country_archetype (cca2);
CREATE INDEX IF NOT EXISTS idx_archetype_type ON country_archetype (archetype);


-- ── C2: crisis_winners_losers ────────────────────────────────────────────────
-- Change in world GDP/capita ranking between pre-shock and post-shock years.
-- Distinguishes:
--   · Genuine winners (climbed rank from a stable base)
--   · Rebound (climbed rank but had collapsed in prior 5 years — not real improvement)
--   · Volatile low-income (gdp_pre < $1500 — rank moves not meaningful)
-- This avoids classifying post-collapse bounces (Venezuela, Lebanon, Zimbabwe) as winners.
DROP MATERIALIZED VIEW IF EXISTS crisis_winners_losers CASCADE;
CREATE MATERIALIZED VIEW crisis_winners_losers AS
WITH gdp_ranked AS (
    SELECT cca2, year, value,
           RANK() OVER (PARTITION BY year ORDER BY value DESC NULLS LAST) AS world_rank,
           COUNT(*) FILTER (WHERE value IS NOT NULL) OVER (PARTITION BY year) AS world_total
    FROM   country_indicators
    WHERE  indicator_code = 'NY.GDP.PCAP.CD'
),
gdp_with_lag AS (
    SELECT cca2, year, value,
           LAG(value, 5) OVER (PARTITION BY cca2 ORDER BY year) AS gdp_5y_ago
    FROM   country_indicators
    WHERE  indicator_code = 'NY.GDP.PCAP.CD'
),
pre_post AS (
    SELECT
        c.cca2, c.name, c.region, c.flag_emoji,
        h.id AS hito_id, h.nombre_evento, h.tipo_shock,
        h.anio_inicio, h.anio_fin,
        (h.anio_inicio - 1) AS pre_year,
        LEAST(h.anio_fin + 3, (SELECT MAX(year) FROM gdp_ranked)) AS post_year,
        pre.world_rank  AS rank_pre,
        post.world_rank AS rank_post,
        pre.value       AS gdp_pre,
        post.value      AS gdp_post,
        lag5.gdp_5y_ago AS gdp_5y_before_pre
    FROM   countries_clean c
    CROSS  JOIN hitos_historicos h
    LEFT   JOIN gdp_ranked   pre  ON pre.cca2  = c.cca2 AND pre.year  = h.anio_inicio - 1
    LEFT   JOIN gdp_ranked   post ON post.cca2 = c.cca2
                                 AND post.year = LEAST(h.anio_fin + 3, (SELECT MAX(year) FROM gdp_ranked))
    LEFT   JOIN gdp_with_lag lag5 ON lag5.cca2 = c.cca2 AND lag5.year = h.anio_inicio - 1
)
SELECT
    cca2, name, region, flag_emoji,
    hito_id, nombre_evento, tipo_shock,
    pre_year, post_year,
    rank_pre, rank_post,
    (rank_pre - rank_post)                       AS rank_change,
    ROUND(gdp_pre::numeric,  0)                  AS gdp_pre,
    ROUND(gdp_post::numeric, 0)                  AS gdp_post,
    CASE WHEN gdp_pre > 0
         THEN ROUND(((gdp_post / gdp_pre - 1) * 100)::numeric, 1)
         ELSE NULL END                           AS gdp_change_pct,
    ROUND(gdp_5y_before_pre::numeric, 0)         AS gdp_5y_before_pre,
    CASE WHEN gdp_5y_before_pre > 0
         THEN ROUND(((gdp_pre / gdp_5y_before_pre - 1) * 100)::numeric, 0)
         ELSE NULL END                           AS pct_change_5y_prior,
    -- Quality flags
    (gdp_5y_before_pre IS NOT NULL
     AND gdp_5y_before_pre > 0
     AND gdp_pre / gdp_5y_before_pre < 0.75)     AS is_rebound,
    (COALESCE(gdp_pre, 0) < 1500)                AS is_low_income,
    -- Verdict with quality awareness
    CASE
        -- Winners that are actually rebounds (collapsed before, now bouncing)
        WHEN (rank_pre - rank_post) >= 10
             AND gdp_5y_before_pre > 0
             AND gdp_pre / gdp_5y_before_pre < 0.75       THEN 'Rebound'
        -- Winners that are low-income (rank movement noisy)
        WHEN (rank_pre - rank_post) >= 10
             AND COALESCE(gdp_pre, 0) < 1500              THEN 'Volatile low-income'
        WHEN (rank_pre - rank_post) >=  10 THEN 'Big Winner'
        WHEN (rank_pre - rank_post) >=   3 THEN 'Winner'
        WHEN (rank_pre - rank_post) <= -10 THEN 'Big Loser'
        WHEN (rank_pre - rank_post) <=  -3 THEN 'Loser'
        ELSE 'Stable'
    END AS verdict
FROM   pre_post
WHERE  rank_pre IS NOT NULL AND rank_post IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_winlose_cca2 ON crisis_winners_losers (cca2);
CREATE INDEX IF NOT EXISTS idx_winlose_hito ON crisis_winners_losers (hito_id);
CREATE INDEX IF NOT EXISTS idx_winlose_chg  ON crisis_winners_losers (rank_change DESC);
CREATE INDEX IF NOT EXISTS idx_winlose_verdict ON crisis_winners_losers (verdict);


-- ── C3: convergence_metrics ──────────────────────────────────────────────────
-- GDP/capita as % of US benchmark over time. Tracks emerging-market catch-up.
DROP MATERIALIZED VIEW IF EXISTS convergence_metrics CASCADE;
CREATE MATERIALIZED VIEW convergence_metrics AS
WITH us_gdp AS (
    SELECT year, value AS us_value
    FROM   country_indicators
    WHERE  cca2 = 'US' AND indicator_code = 'NY.GDP.PCAP.CD'
),
country_gdp AS (
    SELECT ci.cca2, ci.year, ci.value
    FROM   country_indicators ci
    WHERE  ci.indicator_code = 'NY.GDP.PCAP.CD'
      AND  ci.value IS NOT NULL
),
ratios AS (
    SELECT
        c.cca2, c.year,
        ROUND((c.value / u.us_value * 100)::numeric, 2) AS pct_of_us
    FROM   country_gdp c
    JOIN   us_gdp u ON u.year = c.year
    WHERE  u.us_value > 0
),
trends AS (
    -- Long-horizon convergence: latest decade vs 50+ years ago
    SELECT
        r.cca2,
        AVG(CASE WHEN r.year BETWEEN 2015 AND 2024 THEN r.pct_of_us END) AS pct_now,
        AVG(CASE WHEN r.year BETWEEN 2005 AND 2014 THEN r.pct_of_us END) AS pct_10y_ago,
        AVG(CASE WHEN r.year BETWEEN 1995 AND 2004 THEN r.pct_of_us END) AS pct_30y_ago,
        AVG(CASE WHEN r.year BETWEEN 1965 AND 1974 THEN r.pct_of_us END) AS pct_60y_ago
    FROM ratios r
    GROUP BY r.cca2
)
SELECT
    c.cca2, c.name, c.region, c.flag_emoji,
    ROUND(t.pct_now::numeric,1)       AS pct_us_now,
    ROUND(t.pct_10y_ago::numeric,1)   AS pct_us_10y_ago,
    ROUND(t.pct_30y_ago::numeric,1)   AS pct_us_30y_ago,
    ROUND(t.pct_60y_ago::numeric,1)   AS pct_us_60y_ago,
    ROUND((t.pct_now - t.pct_10y_ago)::numeric, 1) AS change_10y,
    ROUND((t.pct_now - t.pct_30y_ago)::numeric, 1) AS change_30y,
    ROUND((t.pct_now - t.pct_60y_ago)::numeric, 1) AS change_60y,
    CASE
        WHEN t.pct_now - COALESCE(t.pct_60y_ago, t.pct_30y_ago) >  10 THEN 'Strong convergence'
        WHEN t.pct_now - COALESCE(t.pct_60y_ago, t.pct_30y_ago) >   3 THEN 'Converging'
        WHEN t.pct_now - COALESCE(t.pct_60y_ago, t.pct_30y_ago) < -10 THEN 'Strong divergence'
        WHEN t.pct_now - COALESCE(t.pct_60y_ago, t.pct_30y_ago) <  -3 THEN 'Diverging'
        ELSE 'Stable'
    END AS trajectory
FROM   countries_clean c
JOIN   trends t ON t.cca2 = c.cca2
WHERE  t.pct_now IS NOT NULL AND COALESCE(t.pct_60y_ago, t.pct_30y_ago) IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_convergence_cca2 ON convergence_metrics (cca2);


-- ── C4: external_vulnerability_index ─────────────────────────────────────────
-- Composite 0-100 of openness × commodity concentration × debt burden.
-- Higher = more exposed to external shocks.
DROP MATERIALIZED VIEW IF EXISTS external_vulnerability_index CASCADE;
CREATE MATERIALIZED VIEW external_vulnerability_index AS
WITH base AS (
    SELECT cca2,
           MAX(CASE WHEN indicator_code='NE.TRD.GNFS.ZS' THEN value END)    AS trade,
           MAX(CASE WHEN indicator_code='TX.VAL.FUEL.ZS.UN' THEN value END) AS fuel_exp,
           MAX(CASE WHEN indicator_code='TX.VAL.MANF.ZS.UN' THEN value END) AS manuf_exp,
           MAX(CASE WHEN indicator_code='DT.DOD.DECT.GN.ZS' THEN value END) AS ext_debt
    FROM (
        SELECT DISTINCT ON (cca2, indicator_code)
               cca2, indicator_code, value
        FROM   country_indicators
        WHERE  value IS NOT NULL
        ORDER  BY cca2, indicator_code, year DESC
    ) latest_per
    GROUP BY cca2
),
scored AS (
    SELECT
        cca2,
        trade, fuel_exp, manuf_exp, ext_debt,
        -- Openness: trade/GDP capped at 200, normalised 0-100
        LEAST(COALESCE(trade,0)/200.0, 1.0) * 100 AS openness_score,
        -- Concentration: max share of fuel or manuf exports (one-sector dependence)
        GREATEST(COALESCE(fuel_exp,0), COALESCE(manuf_exp,0)*0.7) AS concentration_score,
        -- Debt: ext_debt/GNI capped at 150
        LEAST(COALESCE(ext_debt,0)/150.0, 1.0) * 100 AS debt_score
    FROM base
)
SELECT
    c.cca2, c.name, c.region, c.flag_emoji,
    ROUND(s.openness_score::numeric,1)      AS openness_score,
    ROUND(s.concentration_score::numeric,1) AS concentration_score,
    ROUND(s.debt_score::numeric,1)          AS debt_score,
    -- Composite: 40% concentration, 30% debt, 30% openness
    ROUND(
        (s.concentration_score * 0.4
       + s.debt_score          * 0.3
       + s.openness_score      * 0.3)::numeric, 1
    ) AS vulnerability_score
FROM   countries_clean c
JOIN   scored s ON s.cca2 = c.cca2
WHERE  s.trade IS NOT NULL OR s.fuel_exp IS NOT NULL OR s.ext_debt IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_vuln_cca2 ON external_vulnerability_index (cca2);
CREATE INDEX IF NOT EXISTS idx_vuln_score ON external_vulnerability_index (vulnerability_score DESC);


-- ── C5: decade_summary ───────────────────────────────────────────────────────
-- Per (country, decade): averages of key macro indicators. Spots structural shifts.
DROP MATERIALIZED VIEW IF EXISTS decade_summary CASCADE;
CREATE MATERIALIZED VIEW decade_summary AS
WITH decades AS (
    SELECT cca2, indicator_code, value,
           CASE
               WHEN year BETWEEN 1965 AND 1974 THEN '1965-1974'
               WHEN year BETWEEN 1975 AND 1984 THEN '1975-1984'
               WHEN year BETWEEN 1985 AND 1994 THEN '1985-1994'
               WHEN year BETWEEN 1995 AND 2004 THEN '1995-2004'
               WHEN year BETWEEN 2005 AND 2014 THEN '2005-2014'
               WHEN year BETWEEN 2015 AND 2024 THEN '2015-2024'
           END AS decade
    FROM country_indicators
    WHERE year BETWEEN 1965 AND 2024
)
SELECT
    c.cca2, c.name, c.region, c.flag_emoji,
    d.decade,
    ROUND(AVG(CASE WHEN d.indicator_code='NY.GDP.PCAP.CD'    THEN d.value END)::numeric, 0) AS gdp_per_capita,
    ROUND(AVG(CASE WHEN d.indicator_code='NY.GDP.MKTP.KD.ZG' THEN d.value END)::numeric, 2) AS gdp_growth,
    ROUND(AVG(CASE WHEN d.indicator_code='SL.UEM.TOTL.ZS'    THEN d.value END)::numeric, 1) AS unemployment,
    ROUND(AVG(CASE WHEN d.indicator_code='FP.CPI.TOTL.ZG'    THEN d.value END)::numeric, 1) AS inflation,
    ROUND(AVG(CASE WHEN d.indicator_code='NV.SRV.TOTL.ZS'    THEN d.value END)::numeric, 1) AS services_va,
    ROUND(AVG(CASE WHEN d.indicator_code='NV.IND.TOTL.ZS'    THEN d.value END)::numeric, 1) AS industry_va,
    ROUND(AVG(CASE WHEN d.indicator_code='BN.CAB.XOKA.GD.ZS' THEN d.value END)::numeric, 2) AS current_account,
    ROUND(AVG(CASE WHEN d.indicator_code='SI.POV.GINI'       THEN d.value END)::numeric, 1) AS gini,
    ROUND(AVG(CASE WHEN d.indicator_code='GC.DOD.TOTL.GD.ZS' THEN d.value END)::numeric, 1) AS govt_debt
FROM   countries_clean c
JOIN   decades d ON d.cca2 = c.cca2 AND d.decade IS NOT NULL
GROUP  BY c.cca2, c.name, c.region, c.flag_emoji, d.decade;

CREATE INDEX IF NOT EXISTS idx_decade_cca2_dec ON decade_summary (cca2, decade);


-- ════════════════════════════════════════════════════════════════════════════
-- ETAPA F — Country Deep View (zone, peers, trajectory)
-- ════════════════════════════════════════════════════════════════════════════


-- ── F1: subregion_stats ──────────────────────────────────────────────────────
-- Aggregates by subregion (more granular than region). Crucial for context:
-- Spain belongs to "Southern Europe", not just "Europe" as a whole.
DROP MATERIALIZED VIEW IF EXISTS subregion_stats CASCADE;
CREATE MATERIALIZED VIEW subregion_stats AS
SELECT
    c.region, c.subregion,
    COUNT(*)                                        AS n_countries,
    SUM(c.population)                               AS total_population,
    ROUND(AVG(c.latest_gdp_per_capita)::numeric, 0)    AS avg_gdp_per_capita,
    ROUND(AVG(c.latest_gdp_growth)::numeric, 2)         AS avg_gdp_growth,
    ROUND(AVG(c.latest_unemployment)::numeric, 1)       AS avg_unemployment,
    ROUND(AVG(c.latest_inflation)::numeric, 1)          AS avg_inflation,
    ROUND(AVG(c.latest_life_expectancy)::numeric, 1)    AS avg_life_expectancy,
    ROUND(AVG(c.latest_urban_pct)::numeric, 1)          AS avg_urban_pct,
    ROUND(AVG(a.current_account_pct)::numeric, 1)       AS avg_current_account,
    ROUND(AVG(a.services_va_pct)::numeric, 1)           AS avg_services_va,
    ROUND(AVG(a.industry_va_pct)::numeric, 1)           AS avg_industry_va,
    ROUND(AVG(v.vulnerability_score)::numeric, 1)       AS avg_vulnerability,
    ROUND(AVG(es.strength_score)::numeric, 1)           AS avg_strength_score
FROM   countries_clean c
LEFT   JOIN country_archetype           a  ON a.cca2 = c.cca2
LEFT   JOIN external_vulnerability_index v ON v.cca2 = c.cca2
LEFT   JOIN country_economic_strength   es ON es.cca2 = c.cca2
WHERE  c.subregion IS NOT NULL
GROUP  BY c.region, c.subregion;

CREATE INDEX IF NOT EXISTS idx_subregion_name ON subregion_stats (subregion);


-- ── F2: country_peers ────────────────────────────────────────────────────────
-- Top-5 most similar countries for each country.
-- Similarity = same archetype + GDP/cap within ±50% + same region (preferred).
DROP MATERIALIZED VIEW IF EXISTS country_peers CASCADE;
CREATE MATERIALIZED VIEW country_peers AS
WITH base AS (
    SELECT c.cca2, c.name, c.region, c.subregion,
           c.latest_gdp_per_capita AS gdp_cap,
           a.archetype, a.services_va_pct, a.industry_va_pct,
           es.strength_score
    FROM   countries_clean c
    LEFT   JOIN country_archetype         a  ON a.cca2  = c.cca2
    LEFT   JOIN country_economic_strength es ON es.cca2 = c.cca2
),
pairs AS (
    SELECT
        a.cca2 AS cca2,
        a.name AS name,
        b.cca2 AS peer_cca2,
        b.name AS peer_name,
        b.region AS peer_region,
        b.subregion AS peer_subregion,
        b.gdp_cap AS peer_gdp,
        b.archetype AS peer_archetype,
        b.strength_score AS peer_strength,
        -- Similarity score (lower = more similar)
        -- Components: archetype match, GDP/cap distance (log), region/subregion bonus
        CASE WHEN a.archetype = b.archetype THEN 0 ELSE 30 END
        + ABS(LN(GREATEST(a.gdp_cap,1)) - LN(GREATEST(b.gdp_cap,1))) * 15
        + CASE WHEN a.subregion = b.subregion THEN 0
               WHEN a.region    = b.region    THEN 8
               ELSE 18 END
        + CASE WHEN a.strength_score IS NOT NULL AND b.strength_score IS NOT NULL
               THEN ABS(a.strength_score - b.strength_score) * 0.3
               ELSE 0 END
        AS distance
    FROM base a
    JOIN base b ON b.cca2 <> a.cca2
    WHERE a.gdp_cap IS NOT NULL AND b.gdp_cap IS NOT NULL
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY cca2 ORDER BY distance ASC) AS rk
    FROM pairs
)
SELECT
    r.cca2, r.name,
    r.peer_cca2, r.peer_name, r.peer_region, r.peer_subregion,
    ROUND(r.peer_gdp::numeric, 0)     AS peer_gdp_per_capita,
    r.peer_archetype,
    r.peer_strength,
    ROUND(r.distance::numeric, 1)     AS similarity_distance,
    -- Convert distance (0-100+) to similarity % (100 = identical)
    ROUND((100 - LEAST(r.distance, 100))::numeric, 0)::int AS similarity_pct,
    cou.flag_emoji AS peer_flag
FROM   ranked r
JOIN   countries cou ON cou.cca2 = r.peer_cca2
WHERE  r.rk <= 5;

CREATE INDEX IF NOT EXISTS idx_peers_cca2 ON country_peers (cca2, similarity_distance);


-- ── F1: country_zone_position ────────────────────────────────────────────────
-- For each country, its rank within its subregion across key metrics.
DROP MATERIALIZED VIEW IF EXISTS country_zone_position CASCADE;
CREATE MATERIALIZED VIEW country_zone_position AS
WITH ranked AS (
    SELECT
        c.cca2, c.name, c.subregion,
        COUNT(*) OVER (PARTITION BY c.subregion) AS zone_size,
        RANK() OVER (PARTITION BY c.subregion ORDER BY c.latest_gdp_per_capita DESC NULLS LAST) AS rk_gdp_cap,
        RANK() OVER (PARTITION BY c.subregion ORDER BY c.population DESC NULLS LAST)           AS rk_population,
        RANK() OVER (PARTITION BY c.subregion ORDER BY c.latest_gdp_growth DESC NULLS LAST)    AS rk_growth,
        RANK() OVER (PARTITION BY c.subregion ORDER BY c.latest_unemployment ASC NULLS LAST)   AS rk_unemp_best,
        RANK() OVER (PARTITION BY c.subregion ORDER BY c.latest_inflation ASC NULLS LAST)      AS rk_infl_best,
        RANK() OVER (PARTITION BY c.subregion ORDER BY c.latest_life_expectancy DESC NULLS LAST) AS rk_life,
        RANK() OVER (PARTITION BY c.subregion ORDER BY v.vulnerability_score ASC NULLS LAST)   AS rk_vuln_best,
        RANK() OVER (PARTITION BY c.subregion ORDER BY es.strength_score DESC NULLS LAST)      AS rk_strength,
        -- Shares within subregion
        c.population::numeric / NULLIF(SUM(c.population) OVER (PARTITION BY c.subregion), 0) * 100
            AS pop_share_subregion,
        c.population::numeric / NULLIF(SUM(c.population) OVER (PARTITION BY c.region),    0) * 100
            AS pop_share_region
    FROM   countries_clean c
    LEFT   JOIN external_vulnerability_index v ON v.cca2 = c.cca2
    LEFT   JOIN country_economic_strength   es ON es.cca2 = c.cca2
    WHERE  c.subregion IS NOT NULL
)
SELECT
    cca2, name, subregion, zone_size,
    rk_gdp_cap, rk_population, rk_growth,
    rk_unemp_best, rk_infl_best, rk_life,
    rk_vuln_best, rk_strength,
    ROUND(pop_share_subregion::numeric, 1) AS pop_share_subregion,
    ROUND(pop_share_region::numeric, 1)    AS pop_share_region
FROM   ranked;

CREATE INDEX IF NOT EXISTS idx_zonepos_cca2 ON country_zone_position (cca2);


-- ── country_indicator_coverage ───────────────────────────────────────────────
-- Temporal coverage per (country, indicator). Powers UI transparency:
-- e.g. "Sector data available from 1995 for Spain".
DROP MATERIALIZED VIEW IF EXISTS country_indicator_coverage CASCADE;
CREATE MATERIALIZED VIEW country_indicator_coverage AS
SELECT
    cca2,
    indicator_code,
    MAX(indicator_name) AS indicator_name,
    MIN(year)           AS first_year,
    MAX(year)           AS last_year,
    COUNT(*)            AS n_observations,
    MAX(year) - MIN(year) + 1 AS span_years
FROM   country_indicators
WHERE  value IS NOT NULL
GROUP  BY cca2, indicator_code;

CREATE INDEX IF NOT EXISTS idx_coverage_cca2 ON country_indicator_coverage (cca2, indicator_code);


-- ── euro_adoption (reference table for currency narrative) ───────────────────
-- Eurozone members with year of euro adoption.
DROP TABLE IF EXISTS euro_adoption CASCADE;
CREATE TABLE euro_adoption (
    cca2          CHAR(2) PRIMARY KEY,
    adoption_year INTEGER NOT NULL,
    conversion_rate NUMERIC(12,6)  -- old currency units per 1 EUR
);
INSERT INTO euro_adoption VALUES
    ('AT', 1999,    13.7603),  -- ATS
    ('BE', 1999,    40.3399),  -- BEF
    ('FI', 1999,     5.94573), -- FIM
    ('FR', 1999,     6.55957), -- FRF
    ('DE', 1999,     1.95583), -- DEM
    ('IE', 1999,     0.787564),-- IEP
    ('IT', 1999,  1936.27),    -- ITL
    ('LU', 1999,    40.3399),  -- LUF
    ('NL', 1999,     2.20371), -- NLG
    ('PT', 1999,   200.482),   -- PTE
    ('ES', 1999,   166.386),   -- ESP
    ('GR', 2001,   340.750),   -- GRD
    ('SI', 2007,   239.640),   -- SIT
    ('CY', 2008,     0.585274),-- CYP
    ('MT', 2008,     0.4293),  -- MTL
    ('SK', 2009,    30.126),   -- SKK
    ('EE', 2011,    15.6466),  -- EEK
    ('LV', 2014,     0.702804),-- LVL
    ('LT', 2015,     3.4528),  -- LTL
    ('HR', 2023,     7.53450)  -- HRK
ON CONFLICT (cca2) DO NOTHING;


-- ── Data quality: known data errors in WB FCRF ───────────────────────────────
-- Curated list of (cca2, year_range) entries where WB has known issues with
-- currency denomination/redenomination not properly harmonised.
DROP TABLE IF EXISTS currency_data_errors CASCADE;
CREATE TABLE currency_data_errors (
    cca2       CHAR(2),
    year_start INTEGER,
    year_end   INTEGER,
    reason     VARCHAR(200),
    PRIMARY KEY (cca2, year_start)
);
INSERT INTO currency_data_errors (cca2, year_start, year_end, reason) VALUES
    -- Iraq: pre-1990 rates show impossible jump (0.36 → 935 in 1971), no real
    -- devaluation per IMF records. Iraq pegged dinar to USD until 1990 Gulf War.
    ('IQ', 1971, 1990, 'WB FCRF series error: denomination jump 1971, not real devaluation'),
    -- Add other known errors here as detected
    ('ZW', 1965, 1979, 'Pre-independence Rhodesia currency change');


-- ── archetype_shock_matrix (Studies §1) ──────────────────────────────────────
-- Pre-computed 8×12 matrix: how each economic archetype reacted to each
-- historical crisis. Used by /studies/shock-anatomy.
DROP MATERIALIZED VIEW IF EXISTS archetype_shock_matrix CASCADE;
CREATE MATERIALIZED VIEW archetype_shock_matrix AS
SELECT
    a.archetype,
    h.id              AS hito_id,
    h.nombre_evento,
    h.tipo_shock,
    h.anio_inicio,
    h.anio_fin,
    COUNT(*)          AS n_countries,
    ROUND(AVG(w.rank_change)::numeric, 1)    AS avg_rank_change,
    ROUND(AVG(w.gdp_change_pct)::numeric, 1) AS avg_gdp_change_pct,
    ROUND(AVG(cri.gdp_growth_min)::numeric, 1)   AS avg_gdp_growth_min,
    ROUND(AVG(cri.unemployment_max)::numeric, 1) AS avg_unemp_max,
    ROUND(AVG(cri.inflation_max)::numeric, 1)    AS avg_inflation_max
FROM   crisis_winners_losers w
JOIN   country_archetype     a   ON a.cca2 = w.cca2
JOIN   country_crisis_impact cri ON cri.cca2 = w.cca2 AND cri.hito_id = w.hito_id
JOIN   hitos_historicos      h   ON h.id   = w.hito_id
GROUP  BY a.archetype, h.id, h.nombre_evento, h.tipo_shock, h.anio_inicio, h.anio_fin
HAVING COUNT(*) >= 3;

CREATE INDEX IF NOT EXISTS idx_shock_matrix ON archetype_shock_matrix (hito_id, archetype);


-- ── euro_impact_analysis ─────────────────────────────────────────────────────
-- Pre/post-adoption metrics for eurozone members + matched control group.
-- Powers Study #11: "Did giving up your currency help or hurt?"
DROP MATERIALIZED VIEW IF EXISTS euro_impact_analysis CASCADE;
CREATE MATERIALIZED VIEW euro_impact_analysis AS
WITH groups AS (
    -- Eurozone members with their euro adoption year
    SELECT cca2, adoption_year AS pivot_year, TRUE AS in_eurozone FROM euro_adoption
    UNION ALL
    -- Control group: countries similar in development that kept their currency
    SELECT cca2, 2002 AS pivot_year, FALSE AS in_eurozone
    FROM   (VALUES ('GB'), ('SE'), ('CH'), ('PL'), ('NO'), ('CZ'), ('DK'), ('HU')) v(cca2)
)
SELECT
    g.cca2,
    c.name,
    c.flag_emoji,
    g.in_eurozone,
    g.pivot_year,
    -- Inflation (10y pre vs 10y post)
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year-10 AND g.pivot_year-1 AND ci.indicator_code='FP.CPI.TOTL.ZG')::numeric, 2) AS inflation_pre,
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year AND g.pivot_year+10 AND ci.indicator_code='FP.CPI.TOTL.ZG')::numeric, 2) AS inflation_post,
    -- GDP growth (10y pre vs 10y post)
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year-10 AND g.pivot_year-1 AND ci.indicator_code='NY.GDP.MKTP.KD.ZG')::numeric, 2) AS gdp_growth_pre,
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year AND g.pivot_year+10 AND ci.indicator_code='NY.GDP.MKTP.KD.ZG')::numeric, 2) AS gdp_growth_post,
    -- Government debt % GDP (5y pre vs 10-20y post — debt is slow-moving)
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year-5 AND g.pivot_year-1 AND ci.indicator_code='GC.DOD.TOTL.GD.ZS')::numeric, 1) AS govt_debt_pre,
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year+10 AND g.pivot_year+20 AND ci.indicator_code='GC.DOD.TOTL.GD.ZS')::numeric, 1) AS govt_debt_post,
    -- Current account % GDP (5y pre vs 5-15y post)
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year-5 AND g.pivot_year-1 AND ci.indicator_code='BN.CAB.XOKA.GD.ZS')::numeric, 2) AS curr_acc_pre,
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year+5 AND g.pivot_year+15 AND ci.indicator_code='BN.CAB.XOKA.GD.ZS')::numeric, 2) AS curr_acc_post,
    -- Unemployment 5y pre vs 5-15y post
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year-5 AND g.pivot_year-1 AND ci.indicator_code='SL.UEM.TOTL.ZS')::numeric, 2) AS unemp_pre,
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year+5 AND g.pivot_year+15 AND ci.indicator_code='SL.UEM.TOTL.ZS')::numeric, 2) AS unemp_post,
    -- External debt
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year-5 AND g.pivot_year-1 AND ci.indicator_code='DT.DOD.DECT.GN.ZS')::numeric, 1) AS ext_debt_pre,
    ROUND(AVG(ci.value) FILTER (WHERE ci.year BETWEEN g.pivot_year+5 AND g.pivot_year+15 AND ci.indicator_code='DT.DOD.DECT.GN.ZS')::numeric, 1) AS ext_debt_post
FROM   groups g
JOIN   country_indicators ci ON ci.cca2 = g.cca2
JOIN   countries c          ON c.cca2 = g.cca2
GROUP  BY g.cca2, c.name, c.flag_emoji, g.in_eurozone, g.pivot_year;

CREATE INDEX IF NOT EXISTS idx_euro_impact ON euro_impact_analysis (in_eurozone, cca2);


-- ════════════════════════════════════════════════════════════════════════════
-- ML-READY DATASETS
-- ════════════════════════════════════════════════════════════════════════════

-- ── ml_features_wide ─────────────────────────────────────────────────────────
-- Wide-format panel data: (country, year) × all indicators.
-- Ready for pandas read_csv → feature engineering → train.
-- Includes engineered features (3y rolling avg, YoY delta) for key variables.
DROP MATERIALIZED VIEW IF EXISTS ml_features_wide CASCADE;
CREATE MATERIALIZED VIEW ml_features_wide AS
WITH base AS (
    SELECT cca2, year, indicator_code, value
    FROM   country_indicators
    WHERE  year BETWEEN 1965 AND 2024 AND value IS NOT NULL
),
pivoted AS (
    SELECT cca2, year,
        -- Core macro
        MAX(value) FILTER (WHERE indicator_code='NY.GDP.PCAP.CD')    AS gdp_pcap,
        MAX(value) FILTER (WHERE indicator_code='NY.GDP.MKTP.KD.ZG') AS gdp_growth,
        MAX(value) FILTER (WHERE indicator_code='SP.POP.TOTL')       AS population,
        MAX(value) FILTER (WHERE indicator_code='SP.DYN.LE00.IN')    AS life_exp,
        MAX(value) FILTER (WHERE indicator_code='SP.URB.TOTL.IN.ZS') AS urban_pct,
        MAX(value) FILTER (WHERE indicator_code='SL.UEM.TOTL.ZS')    AS unemp,
        MAX(value) FILTER (WHERE indicator_code='FP.CPI.TOTL.ZG')    AS inflation,
        MAX(value) FILTER (WHERE indicator_code='SE.XPD.TOTL.GD.ZS') AS edu_pct,
        -- Sectoral
        MAX(value) FILTER (WHERE indicator_code='NV.SRV.TOTL.ZS')    AS srv_va,
        MAX(value) FILTER (WHERE indicator_code='NV.IND.TOTL.ZS')    AS ind_va,
        MAX(value) FILTER (WHERE indicator_code='NV.AGR.TOTL.ZS')    AS agr_va,
        MAX(value) FILTER (WHERE indicator_code='SL.SRV.EMPL.ZS')    AS srv_empl,
        MAX(value) FILTER (WHERE indicator_code='SL.IND.EMPL.ZS')    AS ind_empl,
        MAX(value) FILTER (WHERE indicator_code='SL.AGR.EMPL.ZS')    AS agr_empl,
        -- Trade
        MAX(value) FILTER (WHERE indicator_code='NE.TRD.GNFS.ZS')    AS trade_pct,
        MAX(value) FILTER (WHERE indicator_code='NE.EXP.GNFS.ZS')    AS exports_pct,
        MAX(value) FILTER (WHERE indicator_code='NE.IMP.GNFS.ZS')    AS imports_pct,
        MAX(value) FILTER (WHERE indicator_code='BN.CAB.XOKA.GD.ZS') AS curr_acc,
        MAX(value) FILTER (WHERE indicator_code='TX.VAL.FUEL.ZS.UN') AS fuel_exp_pct,
        MAX(value) FILTER (WHERE indicator_code='TX.VAL.MANF.ZS.UN') AS manuf_exp_pct,
        -- Financial
        MAX(value) FILTER (WHERE indicator_code='GC.DOD.TOTL.GD.ZS') AS govt_debt,
        MAX(value) FILTER (WHERE indicator_code='DT.DOD.DECT.GN.ZS') AS ext_debt,
        MAX(value) FILTER (WHERE indicator_code='FI.RES.TOTL.MO')    AS reserves_mo,
        MAX(value) FILTER (WHERE indicator_code='BX.KLT.DINV.WD.GD.ZS') AS fdi_pct,
        MAX(value) FILTER (WHERE indicator_code='NE.GDI.TOTL.ZS')    AS capital_form,
        -- Demographic & development
        MAX(value) FILTER (WHERE indicator_code='SP.DYN.TFRT.IN')    AS fertility,
        MAX(value) FILTER (WHERE indicator_code='SP.POP.DPND')       AS dep_ratio,
        MAX(value) FILTER (WHERE indicator_code='SI.POV.GINI')       AS gini,
        MAX(value) FILTER (WHERE indicator_code='GB.XPD.RSDV.GD.ZS') AS rd_pct,
        MAX(value) FILTER (WHERE indicator_code='IT.NET.USER.ZS')    AS internet_pct,
        -- Energy
        MAX(value) FILTER (WHERE indicator_code='EG.FEC.RNEW.ZS')    AS renewables_pct,
        MAX(value) FILTER (WHERE indicator_code='EN.GHG.CO2.PC.CE.AR5') AS co2_pcap,
        -- Currency
        MAX(value) FILTER (WHERE indicator_code='PA.NUS.FCRF')       AS fx_lcu_usd
    FROM base GROUP BY cca2, year
),
with_macro AS (
    SELECT p.*,
           cg.precio_brent_avg AS brent,
           cg.indice_vix_avg   AS vix,
           cg.fed_funds_rate   AS fed,
           cg.geopolitical_risk_idx AS gpr,
           cg.stress_financiero AS stress
    FROM pivoted p
    LEFT JOIN contexto_global_anual cg ON cg.anio = p.year
),
with_lags AS (
    SELECT *,
        -- YoY % changes for key macro variables (lag features)
        ROUND(((gdp_pcap - LAG(gdp_pcap, 1) OVER (PARTITION BY cca2 ORDER BY year))
              / NULLIF(LAG(gdp_pcap, 1) OVER (PARTITION BY cca2 ORDER BY year), 0) * 100)::numeric, 2) AS gdp_pcap_yoy,
        ROUND((fx_lcu_usd - LAG(fx_lcu_usd, 1) OVER (PARTITION BY cca2 ORDER BY year))
              / NULLIF(LAG(fx_lcu_usd, 1) OVER (PARTITION BY cca2 ORDER BY year), 0)::numeric * 100, 2) AS fx_yoy,
        -- Rolling 3-year averages (smoothed signal)
        ROUND(AVG(gdp_growth) OVER (PARTITION BY cca2 ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2) AS gdp_growth_3y,
        ROUND(AVG(inflation)  OVER (PARTITION BY cca2 ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2) AS inflation_3y,
        ROUND(AVG(curr_acc)   OVER (PARTITION BY cca2 ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2) AS curr_acc_3y,
        ROUND(AVG(ext_debt)   OVER (PARTITION BY cca2 ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2) AS ext_debt_3y
    FROM with_macro
)
SELECT
    wl.*,
    c.region, c.subregion,
    a.archetype,
    -- Target labels for ML
    EXISTS (
        SELECT 1 FROM hitos_historicos h
        WHERE wl.year BETWEEN h.anio_inicio AND h.anio_fin
    ) AS in_crisis,
    EXISTS (
        SELECT 1 FROM hitos_historicos h
        WHERE wl.year = h.anio_inicio
    ) AS crisis_start_year,
    (SELECT id FROM hitos_historicos h WHERE wl.year BETWEEN h.anio_inicio AND h.anio_fin LIMIT 1) AS active_hito_id
FROM   with_lags wl
LEFT   JOIN countries          c ON c.cca2 = wl.cca2
LEFT   JOIN country_archetype  a ON a.cca2 = wl.cca2;

CREATE INDEX IF NOT EXISTS idx_ml_features_cca2_year ON ml_features_wide (cca2, year);
CREATE INDEX IF NOT EXISTS idx_ml_features_crisis ON ml_features_wide (in_crisis, year);


-- ── ml_recovery_trend ────────────────────────────────────────────────────────
-- Time-series of recovery time per crisis. Use to answer "are we faster?".
DROP MATERIALIZED VIEW IF EXISTS ml_recovery_trend CASCADE;
CREATE MATERIALIZED VIEW ml_recovery_trend AS
SELECT
    h.id AS hito_id,
    h.anio_inicio AS year,
    h.nombre_evento,
    h.tipo_shock,
    COUNT(*) FILTER (WHERE cr.years_to_recover IS NOT NULL) AS n_recovered,
    COUNT(*) FILTER (WHERE NOT cr.recovered) AS n_not_recovered,
    ROUND(AVG(cr.years_to_recover)::numeric, 2)            AS avg_years_to_recover,
    ROUND(STDDEV(cr.years_to_recover)::numeric, 2)         AS std_years_to_recover,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cr.years_to_recover)::numeric, 2) AS median_years,
    ROUND(AVG(cr.pct_drop)::numeric, 2)                     AS avg_pct_drop
FROM   hitos_historicos    h
LEFT   JOIN country_crisis_recovery cr ON cr.hito_id = h.id
GROUP  BY h.id, h.anio_inicio, h.nombre_evento, h.tipo_shock
ORDER  BY h.anio_inicio;


-- ── currency_history ─────────────────────────────────────────────────────────
-- Per (country, year): exchange rate LCU/USD + YoY change + devaluation flag.
-- Excludes known data errors (currency_data_errors) and auto-detects extreme
-- jumps (>100x year-over-year) that are likely denomination changes, not
-- real devaluations. True hyperinflations (Zimbabwe 2008, Venezuela 2020,
-- Bolivia 1985, Argentina 1989) are preserved.
DROP MATERIALIZED VIEW IF EXISTS currency_history CASCADE;
CREATE MATERIALIZED VIEW currency_history AS
WITH fx_raw AS (
    SELECT ci.cca2, ci.year, ci.value AS lcu_per_usd
    FROM   country_indicators ci
    WHERE  ci.indicator_code = 'PA.NUS.FCRF'
      AND  ci.value IS NOT NULL AND ci.value > 0
),
fx_clean AS (
    -- Strip known errors first
    SELECT r.*
    FROM   fx_raw r
    LEFT   JOIN currency_data_errors e
      ON   e.cca2 = r.cca2 AND r.year BETWEEN e.year_start AND e.year_end
    WHERE  e.cca2 IS NULL
),
fx AS (
    SELECT cca2, year, lcu_per_usd,
           LAG(lcu_per_usd) OVER (PARTITION BY cca2 ORDER BY year) AS prev_value
    FROM   fx_clean
)
SELECT
    f.cca2, f.year,
    f.lcu_per_usd,
    f.prev_value,
    CASE WHEN f.prev_value IS NULL OR f.prev_value = 0 THEN NULL
         ELSE ROUND(((f.lcu_per_usd / f.prev_value - 1) * 100)::numeric, 2)
    END AS yoy_pct,
    -- Devaluation: LCU depreciated >15% in one year (LCU/USD rose >15%)
    CASE WHEN f.prev_value > 0 AND f.lcu_per_usd / f.prev_value > 1.15 THEN TRUE
         ELSE FALSE END AS is_devaluation,
    -- Strong appreciation
    CASE WHEN f.prev_value > 0 AND f.lcu_per_usd / f.prev_value < 0.85 THEN TRUE
         ELSE FALSE END AS is_appreciation,
    -- Auto-flag: jumps >10000% likely denomination changes not real devaluation
    -- (excluded from "biggest devaluation" facts unless confirmed hyperinflation)
    CASE WHEN f.prev_value > 0 AND f.lcu_per_usd / f.prev_value > 100 THEN TRUE
         ELSE FALSE END AS is_extreme_jump,
    -- Euro context
    e.adoption_year AS euro_adoption_year
FROM   fx f
LEFT   JOIN euro_adoption e ON e.cca2 = f.cca2;

CREATE INDEX IF NOT EXISTS idx_currency_cca2 ON currency_history (cca2, year);


-- ── country_purchasing_power ─────────────────────────────────────────────────
-- Real living standards via PPP-adjusted indicators. Compares with USA and
-- regional averages — answers "how much can the average citizen actually buy?".
DROP MATERIALIZED VIEW IF EXISTS country_purchasing_power CASCADE;
CREATE MATERIALIZED VIEW country_purchasing_power AS
WITH latest_ppp AS (
    -- Take latest available value per country × PPP indicator
    SELECT DISTINCT ON (cca2, indicator_code)
        cca2, indicator_code, value, year
    FROM   country_indicators
    WHERE  indicator_code IN (
        'NY.GDP.PCAP.PP.CD','NE.CON.PRVT.PP.KD',
        'NY.GNP.PCAP.PP.CD','NY.GDP.PCAP.PP.KD',
        'NY.GDP.PCAP.CD'  -- nominal for gap comparison
    ) AND value IS NOT NULL
    ORDER BY cca2, indicator_code, year DESC
),
pivoted AS (
    SELECT cca2,
           MAX(CASE WHEN indicator_code='NY.GDP.PCAP.PP.CD'  THEN value END) AS gdp_ppp,
           MAX(CASE WHEN indicator_code='NE.CON.PRVT.PP.KD'  THEN value END) AS cons_ppp,
           MAX(CASE WHEN indicator_code='NY.GNP.PCAP.PP.CD'  THEN value END) AS gni_ppp,
           MAX(CASE WHEN indicator_code='NY.GDP.PCAP.PP.KD'  THEN value END) AS gdp_ppp_const,
           MAX(CASE WHEN indicator_code='NY.GDP.PCAP.CD'     THEN value END) AS gdp_nominal,
           MAX(CASE WHEN indicator_code='NY.GDP.PCAP.PP.CD'  THEN year  END) AS gdp_ppp_year
    FROM   latest_ppp
    GROUP  BY cca2
),
us_bench AS (
    SELECT gdp_ppp AS us_gdp_ppp, cons_ppp AS us_cons_ppp, gni_ppp AS us_gni_ppp
    FROM pivoted WHERE cca2 = 'US'
)
SELECT
    c.cca2, c.name, c.region, c.subregion, c.flag_emoji,
    ROUND(p.gdp_nominal::numeric, 0)         AS gdp_per_capita_nominal,
    ROUND(p.gdp_ppp::numeric, 0)             AS gdp_per_capita_ppp,
    -- Household consumption: WB series is total, divide by population to get per capita
    ROUND((p.cons_ppp / NULLIF(c.population, 0))::numeric, 0)
                                             AS household_consumption_per_capita_ppp,
    ROUND(p.gni_ppp::numeric, 0)             AS gni_per_capita_ppp,
    p.gdp_ppp_year                           AS year_latest,
    -- Ratios vs USA (% of US value) — proper per-capita
    ROUND((p.gdp_ppp  / NULLIF(u.us_gdp_ppp,  0) * 100)::numeric, 1) AS pct_us_gdp_ppp,
    ROUND(
        ((p.cons_ppp / NULLIF(c.population, 0))
         / NULLIF(u.us_cons_ppp / NULLIF((SELECT population FROM countries WHERE cca2='US'), 0), 0)
         * 100)::numeric, 1
    )                                                                AS pct_us_consumption,
    ROUND((p.gni_ppp  / NULLIF(u.us_gni_ppp,  0) * 100)::numeric, 1) AS pct_us_gni_ppp,
    -- Nominal vs PPP gap (positive = PPP higher = cheaper country / lower cost of living)
    ROUND(((p.gdp_ppp - p.gdp_nominal) / NULLIF(p.gdp_nominal, 0) * 100)::numeric, 1)
                                                                     AS ppp_premium_pct,
    -- Cost of living index (US = 100). <100 cheaper than US, >100 more expensive.
    -- Derived from price level = nominal/PPP. A loaf that costs $1 in US costs
    -- (index/100) dollars locally.
    ROUND((p.gdp_nominal / NULLIF(p.gdp_ppp, 0) * 100)::numeric, 1)
                                                                     AS cost_of_living_index,
    -- "Cheaper by X%" (intuitive metric for narrative)
    ROUND(((1 - p.gdp_nominal / NULLIF(p.gdp_ppp, 0)) * 100)::numeric, 1)
                                                                     AS cheaper_than_us_pct
FROM   countries_clean c
LEFT   JOIN pivoted   p ON p.cca2 = c.cca2
CROSS  JOIN us_bench  u
WHERE  p.gdp_ppp IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pp_cca2 ON country_purchasing_power (cca2);
CREATE INDEX IF NOT EXISTS idx_pp_gdp  ON country_purchasing_power (gdp_per_capita_ppp DESC);


-- ── household_budget_breakdown ───────────────────────────────────────────────
-- Curated breakdown of average household spending + savings.
-- Sources (latest available, typically 2022-2023):
--   Europe: Eurostat HBS (Household Budget Survey)
--   USA:    BLS Consumer Expenditure Survey + Personal Savings Rate
--   Japan:  FIES (Family Income and Expenditure Survey)
--   Mexico: INEGI ENIGH
--   China:  NBS Household Survey
--   Korea:  Statistics Korea
-- All values are % of disposable income. food + housing + transport + other + savings ≈ 100.
DROP TABLE IF EXISTS household_budget_breakdown CASCADE;
CREATE TABLE household_budget_breakdown (
    cca2         CHAR(2) PRIMARY KEY,
    year         INTEGER,
    food_pct     NUMERIC(5,2),
    housing_pct  NUMERIC(5,2),  -- includes utilities (water/electricity/gas)
    transport_pct NUMERIC(5,2),
    other_pct    NUMERIC(5,2),  -- health, leisure, restaurants, clothing, etc.
    savings_pct  NUMERIC(5,2),
    source       VARCHAR(80)
);

INSERT INTO household_budget_breakdown
    (cca2, year, food_pct, housing_pct, transport_pct, other_pct, savings_pct, source) VALUES
    -- Eurozone & EU
    ('ES', 2022, 15.5, 31.5, 11.8, 31.0, 10.2, 'Eurostat HBS'),
    ('PT', 2022, 17.5, 27.5, 11.5, 32.0, 11.5, 'Eurostat HBS'),
    ('GR', 2022, 20.5, 30.5, 12.5, 30.0,  6.5, 'Eurostat HBS'),
    ('IT', 2022, 16.5, 26.5, 11.0, 35.0, 11.0, 'Eurostat HBS'),
    ('FR', 2022, 13.2, 28.0, 12.6, 32.0, 14.2, 'Eurostat HBS'),
    ('DE', 2022, 12.5, 32.0, 13.5, 31.8, 10.2, 'Eurostat HBS'),
    ('NL', 2022, 13.0, 25.0, 12.0, 35.0, 15.0, 'Eurostat HBS'),
    ('BE', 2022, 13.5, 27.0, 12.5, 33.0, 14.0, 'Eurostat HBS'),
    ('IE', 2022, 12.5, 28.0, 13.0, 34.0, 12.5, 'Eurostat HBS'),
    ('AT', 2022, 13.0, 28.5, 13.5, 33.0, 12.0, 'Eurostat HBS'),
    ('FI', 2022, 12.0, 29.0, 13.5, 33.5, 12.0, 'Eurostat HBS'),
    -- Nordics & non-euro EU
    ('SE', 2022, 12.5, 27.0, 12.0, 32.5, 16.0, 'SCB / Eurostat'),
    ('DK', 2022, 11.5, 28.5, 12.5, 33.5, 14.0, 'Eurostat HBS'),
    ('NO', 2022, 11.5, 26.5, 12.5, 34.5, 15.0, 'SSB'),
    ('PL', 2022, 17.0, 22.0, 11.0, 39.0, 11.0, 'Eurostat HBS'),
    -- Anglosphere
    ('US', 2022, 12.8, 33.3, 16.8, 33.1,  4.0, 'BLS CES + BEA savings rate'),
    ('GB', 2022, 11.3, 28.2, 14.8, 38.3,  7.4, 'ONS Family Spending'),
    ('CA', 2022, 11.5, 30.0, 14.5, 35.0,  9.0, 'StatsCan / OECD'),
    ('AU', 2022, 11.0, 27.0, 14.5, 36.5, 11.0, 'ABS / OECD'),
    -- Asia developed
    ('JP', 2022, 25.0, 23.0, 13.0, 32.4,  6.6, 'FIES (Japan)'),
    ('KR', 2022, 14.0, 16.0, 12.0, 33.0, 25.0, 'Statistics Korea'),
    -- Emerging
    ('CN', 2021, 30.0, 23.0, 14.0, 18.0, 15.0, 'NBS Household Survey'),
    ('IN', 2022, 30.0, 14.0, 12.0, 40.0,  4.0, 'NSO Consumer Expenditure'),
    ('BR', 2018, 18.0, 18.0, 18.0, 41.0,  5.0, 'POF IBGE'),
    ('MX', 2022, 30.0, 14.0, 17.0, 31.0,  8.0, 'INEGI ENIGH'),
    ('AR', 2022, 24.0, 17.0, 14.0, 40.0,  5.0, 'INDEC EPH'),
    ('TR', 2022, 22.0, 22.0, 18.0, 35.0,  3.0, 'TUIK Household Budget'),
    ('RU', 2022, 35.0, 17.0, 13.0, 30.0,  5.0, 'Rosstat'),
    ('ZA', 2022, 17.0, 25.0, 14.0, 39.0,  5.0, 'StatsSA / OECD'),
    ('SA', 2022, 17.0, 25.0, 15.0, 38.0,  5.0, 'GASTAT Saudi'),
    -- Switzerland (non-EU)
    ('CH', 2022, 11.0, 28.0, 13.5, 31.5, 16.0, 'OFS / SNB');


-- ── region_correlations (rebuilt) ────────────────────────────────────────────
-- The ETL's pandas version correlated regional-averaged time series, producing
-- spurious near-1 correlations between any two monotonic trends.
-- This SQL replacement correlates pooled (country, year) observations within
-- each region — methodologically correct cross-sectional + temporal pooling.
TRUNCATE region_correlations RESTART IDENTITY;

WITH ind_labels AS (
    SELECT indicator_code, MAX(indicator_name) AS indicator_name
    FROM   country_indicators
    GROUP  BY indicator_code
),
pair_obs AS (
    SELECT
        c.region,
        ci_x.indicator_code AS ind_x,
        ci_y.indicator_code AS ind_y,
        ci_x.value          AS vx,
        ci_y.value          AS vy
    FROM   country_indicators ci_x
    JOIN   country_indicators ci_y
      ON   ci_y.cca2 = ci_x.cca2
     AND   ci_y.year = ci_x.year
     AND   ci_y.indicator_code > ci_x.indicator_code
    JOIN   countries_clean c ON c.cca2 = ci_x.cca2
    WHERE  ci_x.value IS NOT NULL AND ci_y.value IS NOT NULL
),
agg AS (
    SELECT region, ind_x, ind_y,
           corr(vx, vy)  AS r,
           COUNT(*)      AS n
    FROM   pair_obs
    GROUP  BY region, ind_x, ind_y
    HAVING COUNT(*) >= 30 AND corr(vx, vy) IS NOT NULL
)
INSERT INTO region_correlations (region, indicator_x, label_x, indicator_y, label_y, correlation, n_obs)
SELECT
    a.region,
    a.ind_x, lx.indicator_name,
    a.ind_y, ly.indicator_name,
    ROUND(a.r::numeric, 6),
    a.n
FROM   agg a
JOIN   ind_labels lx ON lx.indicator_code = a.ind_x
JOIN   ind_labels ly ON ly.indicator_code = a.ind_y;
