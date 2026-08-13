-- =============================================================================
-- Countries ETL — Database Schema v2
-- =============================================================================

-- ── Core tables ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS countries (
    id                       SERIAL PRIMARY KEY,
    name                     VARCHAR(100) NOT NULL,
    cca2                     CHAR(2),
    region                   VARCHAR(50),
    subregion                VARCHAR(100),
    capital                  VARCHAR(100),
    population               BIGINT,
    area                     DECIMAL(14,2),
    population_density       DECIMAL(12,4),
    currencies               TEXT,
    gini                     DECIMAL(5,2),
    flag_emoji               VARCHAR(10),
    -- World Bank: latest snapshot values
    latest_gdp_per_capita    DECIMAL(14,2),
    latest_gdp_growth        DECIMAL(8,4),
    latest_life_expectancy   DECIMAL(5,2),
    latest_unemployment      DECIMAL(7,4),
    latest_urban_pct         DECIMAL(7,4),
    latest_education_pct     DECIMAL(7,4),
    latest_trade_pct         DECIMAL(10,4),
    latest_inflation         DECIMAL(10,4),
    population_growth_10y    DECIMAL(8,2),
    -- Rankings
    population_rank_region   INTEGER,
    density_rank_region      INTEGER,
    global_population_rank   INTEGER,
    global_density_rank      INTEGER,
    created_at               TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS region_stats (
    id                   SERIAL PRIMARY KEY,
    region               VARCHAR(50) NOT NULL,
    country_count        INTEGER,
    total_population     BIGINT,
    total_area           DECIMAL(16,2),
    avg_density          DECIMAL(12,4),
    avg_gini             DECIMAL(5,2),
    avg_gdp_per_capita   DECIMAL(14,2),
    avg_life_expectancy  DECIMAL(5,2),
    avg_unemployment     DECIMAL(7,4),
    avg_urban_pct        DECIMAL(7,4),
    created_at           TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS country_indicators (
    id              SERIAL PRIMARY KEY,
    cca2            CHAR(2)      NOT NULL,
    indicator_code  VARCHAR(30)  NOT NULL,
    indicator_name  VARCHAR(120),
    year            INTEGER      NOT NULL,
    value           DECIMAL(20,4),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS region_correlations (
    id            SERIAL PRIMARY KEY,
    region        VARCHAR(50)  NOT NULL,
    indicator_x   VARCHAR(30)  NOT NULL,
    label_x       VARCHAR(60),
    indicator_y   VARCHAR(30)  NOT NULL,
    label_y       VARCHAR(60),
    correlation   DECIMAL(8,6) NOT NULL,
    n_obs         INTEGER,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- ── Global macro context ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS contexto_global_anual (
    id                    SERIAL PRIMARY KEY,
    anio                  INTEGER      NOT NULL UNIQUE,
    precio_brent_avg      DECIMAL(10,4),   -- USD/barrel annual avg (FRED DCOILBRENTEU)
    indice_vix_avg        DECIMAL(10,4),   -- CBOE VIX annual avg (FRED VIXCLS)
    fed_funds_rate        DECIMAL(8,4),    -- Federal Funds Rate monthly avg (FRED FEDFUNDS)
    geopolitical_risk_idx DECIMAL(10,4),   -- GPR Index (Caldara & Iacoviello)
    stress_financiero     DECIMAL(10,4),   -- St. Louis Fed Financial Stress Index (STLFSI4)
    created_at            TIMESTAMP DEFAULT NOW()
);

-- ── Historical milestones / shocks ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS hitos_historicos (
    id            SERIAL PRIMARY KEY,
    anio_inicio   INTEGER      NOT NULL,
    anio_fin      INTEGER      NOT NULL,
    nombre_evento VARCHAR(200) NOT NULL,
    descripcion   TEXT,
    tipo_shock    VARCHAR(80),             -- 'Financiero', 'Energético', 'Sanitario/Económico', etc.
    created_at    TIMESTAMP DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_ind_cca2    ON country_indicators(cca2);
CREATE INDEX IF NOT EXISTS idx_ind_code    ON country_indicators(indicator_code);
CREATE INDEX IF NOT EXISTS idx_ind_year    ON country_indicators(year);
CREATE INDEX IF NOT EXISTS idx_ind_cca2_code ON country_indicators(cca2, indicator_code);
CREATE INDEX IF NOT EXISTS idx_corr_region ON region_correlations(region);
CREATE INDEX IF NOT EXISTS idx_ctx_anio    ON contexto_global_anual(anio);
CREATE INDEX IF NOT EXISTS idx_hitos_anios ON hitos_historicos(anio_inicio, anio_fin);

-- ── Seed: hitos históricos (idempotent) ───────────────────────────────────────

INSERT INTO hitos_historicos (anio_inicio, anio_fin, nombre_evento, descripcion, tipo_shock)
SELECT anio_inicio, anio_fin, nombre_evento, descripcion, tipo_shock
FROM (VALUES
    (1973, 1974,
     'Primer Shock del Petróleo — OPEP / Yom Kippur',
     'Embargo petrolero de la OPEP tras la guerra del Yom Kippur. Cuadruplicación del precio del crudo. Fin del sistema de Bretton Woods. Estanflación generalizada en occidente y crisis fiscal en países no productores.',
     'Energético/Geopolítico'),
    (1979, 1982,
     'Segundo Shock del Petróleo + Shock Volcker',
     'Revolución iraní en 1979 dispara de nuevo el crudo. La Fed de Volcker sube tipos al 20% para domar la inflación. Recesión global severa 1980-82. Inicio de la crisis de deuda en Latinoamérica.',
     'Energético/Monetario'),
    (1982, 1989,
     'Crisis de Deuda Latinoamericana — Década Perdida',
     'Default mexicano de agosto 1982 desencadena crisis de deuda en toda América Latina. Década perdida con crecimiento nulo, hiperinflaciones y reformas estructurales del FMI. Recuperación tardía con el Plan Brady (1989).',
     'Financiero'),
    (1989, 1991,
     'Caída del Muro de Berlín y Colapso URSS',
     'Disolución del bloque soviético y reunificación alemana. Transición de planificación central a mercado en Europa del Este. Colapso del PIB en exrepúblicas soviéticas (-30% a -50%). Década de transición turbulenta.',
     'Geopolítico'),
    (1994, 1995,
     'Crisis del Tequila — México',
     'Devaluación abrupta del peso mexicano en diciembre 1994. Efecto tequila contagia a Argentina, Brasil. Rescate de US$50bn liderado por EE.UU. y FMI. Primer caso de crisis financiera moderna en mercado emergente.',
     'Financiero'),
    (1997, 1998,
     'Crisis Financiera Asiática',
     'Colapso en cadena de divisas asiáticas iniciado en Tailandia (baht). Contagio a Indonesia, Corea del Sur y Malasia. Intervención del FMI. Impacto severo en economías emergentes globales.',
     'Financiero'),
    (2000, 2001,
     'Burbuja Dot-com y Atentados 11S',
     'Crash del NASDAQ tecnológico tras la sobrevaluación de empresas internet. Compuesto por el shock geopolítico del 11 de septiembre de 2001 que paralizó el comercio global y disparo el gasto en defensa.',
     'Geopolítico/Financiero'),
    (2008, 2009,
     'Gran Crisis Financiera — Lehman Brothers',
     'Colapso del sistema financiero global tras la quiebra de Lehman Brothers en septiembre de 2008. Crisis de crédito, congelación del mercado interbancario y mayor recesión global desde la Gran Depresión de 1929.',
     'Financiero'),
    (2011, 2014,
     'Crisis de Deuda Soberana Eurozona',
     'Crisis de deuda pública en Grecia, Portugal, Irlanda, España e Italia. Riesgo real de ruptura del euro. Rescates del BCE y Troika. Austeridad fiscal severa con impacto prolongado en desempleo.',
     'Financiero'),
    (2014, 2016,
     'Shock del Precio del Crudo — OPEP',
     'Derrumbe del precio del petróleo superior al 50% entre 2014 y 2016. La OPEP, liderada por Arabia Saudí, mantuvo producción para defender cuota de mercado frente al auge del shale oil estadounidense.',
     'Energético'),
    (2020, 2021,
     'Pandemia COVID-19',
     'Parálisis económica global sin precedentes por la pandemia de coronavirus. Mayor contracción del PIB mundial desde la Segunda Guerra Mundial. Respuesta histórica de estímulo fiscal y monetario.',
     'Sanitario/Económico'),
    (2022, 2023,
     'Guerra de Ucrania y Crisis Inflacionaria',
     'Invasión rusa de Ucrania en febrero de 2022. Shock energético y alimentario global. Inflación en máximos de 40 años en economías occidentales. Respuesta de subidas agresivas de tipos por parte de la Fed y el BCE.',
     'Geopolítico/Energético')
) AS v(anio_inicio, anio_fin, nombre_evento, descripcion, tipo_shock)
WHERE NOT EXISTS (SELECT 1 FROM hitos_historicos LIMIT 1);
