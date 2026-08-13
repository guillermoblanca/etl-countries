import os
import time
import logging
import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


def _connect(retries: int = 10, delay: int = 3):
    for attempt in range(1, retries + 1):
        try:
            return psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                dbname=os.getenv("POSTGRES_DB"),
            )
        except psycopg2.OperationalError as e:
            logger.warning("DB not ready (attempt %d/%d): %s", attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(delay)


def _val(row: pd.Series, col: str):
    v = row[col]
    return None if pd.isna(v) else v


def load_countries(df: pd.DataFrame) -> None:
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("TRUNCATE TABLE countries RESTART IDENTITY CASCADE")

    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO countries (
                name, cca2, region, subregion, capital,
                population, area, population_density,
                currencies, gini, flag_emoji,
                latest_gdp_per_capita, latest_gdp_growth,
                latest_life_expectancy, latest_unemployment,
                latest_urban_pct, latest_education_pct,
                latest_trade_pct, latest_inflation,
                population_growth_10y,
                population_rank_region, density_rank_region,
                global_population_rank, global_density_rank
            ) VALUES (
                %s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,
                %s,%s,
                %s,%s,
                %s,%s,
                %s,
                %s,%s,
                %s,%s
            )
            """,
            (
                row["name"], row["cca2"], row["region"], row["subregion"], _val(row, "capital"),
                int(row["population"]), _val(row, "area"), _val(row, "population_density"),
                row["currencies"], _val(row, "gini"), row["flag_emoji"],
                _val(row, "latest_gdp_per_capita"), _val(row, "latest_gdp_growth"),
                _val(row, "latest_life_expectancy"), _val(row, "latest_unemployment"),
                _val(row, "latest_urban_pct"), _val(row, "latest_education_pct"),
                _val(row, "latest_trade_pct"), _val(row, "latest_inflation"),
                _val(row, "population_growth_10y"),
                int(row["population_rank_region"]), int(row["density_rank_region"]),
                int(row["global_population_rank"]), int(row["global_density_rank"]),
            ),
        )

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Loaded %d rows into countries", len(df))


def load_region_stats(df: pd.DataFrame) -> None:
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("TRUNCATE TABLE region_stats RESTART IDENTITY")

    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO region_stats (
                region, country_count, total_population, total_area,
                avg_density, avg_gini, avg_gdp_per_capita, avg_life_expectancy,
                avg_unemployment, avg_urban_pct
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                row["region"], int(row["country_count"]),
                int(row["total_population"]), _val(row, "total_area"),
                _val(row, "avg_density"), _val(row, "avg_gini"),
                _val(row, "avg_gdp_per_capita"), _val(row, "avg_life_expectancy"),
                _val(row, "avg_unemployment"), _val(row, "avg_urban_pct"),
            ),
        )

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Loaded %d rows into region_stats", len(df))


def load_indicators(df: pd.DataFrame) -> None:
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("TRUNCATE TABLE country_indicators RESTART IDENTITY")

    for _, row in df.iterrows():
        cur.execute(
            "INSERT INTO country_indicators (cca2, indicator_code, indicator_name, year, value) VALUES (%s,%s,%s,%s,%s)",
            (row["cca2"], row["indicator_code"], row["indicator_name"], int(row["year"]), float(row["value"])),
        )

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Loaded %d rows into country_indicators", len(df))


def load_correlations(df: pd.DataFrame) -> None:
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("TRUNCATE TABLE region_correlations RESTART IDENTITY")

    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO region_correlations
                (region, indicator_x, label_x, indicator_y, label_y, correlation, n_obs)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                row["region"], row["indicator_x"], row["label_x"],
                row["indicator_y"], row["label_y"],
                float(row["correlation"]), int(row["n_obs"]),
            ),
        )

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Loaded %d correlation pairs", len(df))


def load_global_context(df: pd.DataFrame) -> None:
    """
    Replaces contexto_global_anual with fresh data on every ETL run.
    Tolerates NaN values — individual metric columns may be NULL in the DB.
    """
    conn = _connect()
    cur  = conn.cursor()
    cur.execute("TRUNCATE TABLE contexto_global_anual RESTART IDENTITY")

    sql = """
        INSERT INTO contexto_global_anual
            (anio, precio_brent_avg, indice_vix_avg, fed_funds_rate,
             geopolitical_risk_idx, stress_financiero)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    rows = [
        (
            int(row["anio"]),
            _val(row, "precio_brent_avg"),
            _val(row, "indice_vix_avg"),
            _val(row, "fed_funds_rate"),
            _val(row, "geopolitical_risk_idx"),
            _val(row, "stress_financiero"),
        )
        for _, row in df.iterrows()
    ]
    cur.executemany(sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Loaded %d rows into contexto_global_anual", len(rows))


def load_hitos(seed_rows: list[dict] | None = None) -> None:
    """
    Idempotent seeding of hitos_historicos. Inserts any missing milestone by
    nombre_evento — keeps existing rows untouched, adds new ones (e.g. when the
    catalogue is extended without a full ETL reset).
    """
    conn = _connect()
    cur  = conn.cursor()

    # Full catalogue (12 hitos: 5 pre-1995 + 7 post-1995)
    full_seed = [
        (1973, 1974, "Primer Shock del Petróleo — OPEP / Yom Kippur",
         "Embargo OPEP tras Yom Kippur; precio crudo x4; fin Bretton Woods; estanflación.", "Energético/Geopolítico"),
        (1979, 1982, "Segundo Shock del Petróleo + Shock Volcker",
         "Revolución iraní + tipos Fed al 20% (Volcker); recesión global y crisis deuda Latam.", "Energético/Monetario"),
        (1982, 1989, "Crisis de Deuda Latinoamericana — Década Perdida",
         "Default mexicano 1982; década perdida en Latam; Plan Brady 1989.", "Financiero"),
        (1989, 1991, "Caída del Muro de Berlín y Colapso URSS",
         "Disolución bloque soviético; colapso PIB exURSS -30 a -50%; transición turbulenta.", "Geopolítico"),
        (1994, 1995, "Crisis del Tequila — México",
         "Devaluación peso 1994; efecto tequila Argentina/Brasil; rescate $50bn EE.UU./FMI.", "Financiero"),
        (1997, 1998, "Crisis Financiera Asiática",
         "Colapso de divisas asiáticas iniciado en Tailandia; contagio a mercados emergentes.", "Financiero"),
        (2000, 2001, "Burbuja Dot-com y Atentados 11S",
         "Crash del NASDAQ tecnológico y shock geopolítico del 11 de septiembre.", "Geopolítico/Financiero"),
        (2008, 2009, "Gran Crisis Financiera — Lehman Brothers",
         "Colapso del sistema financiero global; mayor recesión desde 1929.", "Financiero"),
        (2011, 2014, "Crisis de Deuda Soberana Eurozona",
         "Crisis de deuda pública en Grecia, Portugal, Irlanda y España.", "Financiero"),
        (2014, 2016, "Shock del Precio del Crudo — OPEP",
         "Derrumbe del petróleo >50%; OPEP mantiene producción para defender cuota.", "Energético"),
        (2020, 2021, "Pandemia COVID-19",
         "Parálisis económica global; mayor contracción del PIB desde la IIGM.", "Sanitario/Económico"),
        (2022, 2023, "Guerra de Ucrania y Crisis Inflacionaria",
         "Invasión rusa de Ucrania; inflación en máximos de 40 años.", "Geopolítico/Energético"),
    ]
    rows = seed_rows or full_seed

    # Insert only milestones whose nombre_evento doesn't already exist
    cur.execute("SELECT nombre_evento FROM hitos_historicos")
    existing = {r[0] for r in cur.fetchall()}
    missing = [r for r in rows if r[2] not in existing]

    if not missing:
        logger.info("Hitos históricos: all %d already present — no inserts", len(rows))
    else:
        cur.executemany(
            "INSERT INTO hitos_historicos (anio_inicio, anio_fin, nombre_evento, descripcion, tipo_shock) VALUES (%s,%s,%s,%s,%s)",
            missing,
        )
        conn.commit()
        logger.info("Hitos históricos: inserted %d new (total catalogue: %d)", len(missing), len(rows))

    cur.close()
    conn.close()
