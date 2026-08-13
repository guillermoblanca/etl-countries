"""
ETL Pipeline — v2
==================
Stage 1  Geo/demographic data       REST Countries API
Stage 2  Macroeconomic time-series  World Bank API (15 indicators, 30 years)
Stage 3  Global macro context       FRED (Brent, VIX, Fed Funds, Stress) + GPR Index
Stage 4  Load all tables + seed hitos históricos
"""

import logging
from pathlib import Path
from extract        import fetch_countries, fetch_indicators
from extract_global import fetch_global_context
from transform      import transform, transform_global_context
from load           import (
    load_countries,
    load_region_stats,
    load_indicators,
    load_correlations,
    load_global_context,
    load_hitos,
    _connect,
)


def apply_analytics_views(sql_path: Path = Path("/app/analytics_views.sql")) -> None:
    """Run analytics_views.sql against the DB (creates clean views, materialised
    views for crisis impact + strength score, and rebuilds region_correlations)."""
    if not sql_path.exists():
        logging.getLogger(__name__).warning(
            "analytics_views.sql not found at %s — skipping derived views", sql_path
        )
        return
    sql = sql_path.read_text(encoding="utf-8")
    conn = _connect()
    cur  = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close(); conn.close()
    logging.getLogger(__name__).info("Analytics views applied successfully")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== ETL pipeline v2 started ===")

    # ── Stage 1 & 2: Countries + World Bank indicators ────────────────────────
    logger.info("--- Stage 1: Extracting country & indicator data ---")
    raw_countries = fetch_countries()
    indicators_df = fetch_indicators()

    countries_df, region_stats_df, indicators_clean, correlations_df = transform(
        raw_countries, indicators_df
    )

    load_countries(countries_df)
    load_region_stats(region_stats_df)
    load_indicators(indicators_clean)
    load_correlations(correlations_df)
    logger.info("--- Stage 1 & 2 complete ---")

    # ── Stage 3: Global macro context (FRED + GPR) ────────────────────────────
    logger.info("--- Stage 3: Extracting global macro context ---")
    try:
        context_raw = fetch_global_context()
        context_df  = transform_global_context(context_raw)
        load_global_context(context_df)
    except Exception as exc:
        # Non-fatal: dashboard works without context data
        logger.error("Global context stage failed (non-fatal): %s", exc)
    logger.info("--- Stage 3 complete ---")

    # ── Stage 4: Seed reference tables ───────────────────────────────────────
    logger.info("--- Stage 4: Seeding hitos históricos ---")
    load_hitos()
    logger.info("--- Stage 4 complete ---")

    # ── Stage 5: Derived analytics views (clean, crisis_impact, strength) ────
    logger.info("--- Stage 5: Applying analytics views ---")
    try:
        apply_analytics_views()
    except Exception as exc:
        logger.error("Analytics views stage failed (non-fatal): %s", exc)
    logger.info("--- Stage 5 complete ---")

    logger.info("=== ETL pipeline v2 completed successfully ===")


if __name__ == "__main__":
    main()
