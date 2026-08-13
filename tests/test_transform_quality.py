"""
Data-quality checks on transform() output.

These assert the properties the warehouse depends on: keys are unique and
well-formed, derived metrics are arithmetically right, rankings are internally
consistent, and correlations are reported with the sample size behind them.
"""
import pandas as pd
import pytest

from transform import transform


@pytest.fixture
def result(valid_countries, indicators_df):
    countries, regions, indicators, correlations = transform(valid_countries, indicators_df)
    return countries, regions, indicators, correlations


# ── Keys and referential integrity ───────────────────────────────────────────

def test_country_codes_are_unique(result):
    countries, *_ = result
    assert countries["cca2"].is_unique


def test_country_codes_are_two_uppercase_letters(result):
    countries, *_ = result
    assert countries["cca2"].str.fullmatch(r"[A-Z]{2}").all()


def test_no_country_loses_its_name_or_region(result):
    countries, *_ = result
    assert countries["name"].notna().all()
    assert (countries["name"] != "Unknown").all()
    assert countries["region"].notna().all()


def test_indicator_rows_reference_known_countries(result):
    countries, _, indicators, _ = result
    assert set(indicators["cca2"]) <= set(countries["cca2"])


# ── Derived metrics ──────────────────────────────────────────────────────────

def test_population_comes_from_the_world_bank(result):
    """Population is not in the country dataset; it must arrive via the merge."""
    countries, *_ = result
    assert (countries["population"] > 0).all()


def test_gini_comes_from_the_world_bank(result):
    countries, *_ = result
    assert countries["gini"].notna().all()
    assert countries["gini"].between(0, 100).all()


def test_density_equals_population_over_area(result):
    countries, *_ = result
    expected = (countries["population"] / countries["area"]).round(4)
    pd.testing.assert_series_equal(
        countries["population_density"], expected, check_names=False
    )


def test_density_is_null_when_area_is_missing(valid_countries, indicators_df):
    valid_countries[3]["area"] = None
    valid_countries[4]["area"] = 0
    countries, *_ = transform(valid_countries, indicators_df)
    missing = countries[countries["cca2"].isin(["AD", "AE"])]
    assert missing["population_density"].isna().all()


# ── Rankings ─────────────────────────────────────────────────────────────────

def test_global_population_rank_is_dense_and_starts_at_one(result):
    countries, *_ = result
    ranks = countries["global_population_rank"]
    assert ranks.min() == 1
    assert ranks.max() <= len(countries)


def test_top_ranked_country_really_is_the_most_populous(result):
    countries, *_ = result
    top = countries.loc[countries["global_population_rank"] == 1, "population"].iloc[0]
    assert top == countries["population"].max()


def test_regional_rank_never_exceeds_region_size(result):
    countries, *_ = result
    sizes = countries.groupby("region")["cca2"].transform("count")
    assert (countries["population_rank_region"] <= sizes).all()


# ── Region aggregates ────────────────────────────────────────────────────────

def test_region_counts_match_the_country_table(result):
    countries, regions, _, _ = result
    expected = countries.groupby("region").size().sort_index()
    actual = regions.set_index("region")["country_count"].sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)


def test_region_populations_sum_to_the_world_total(result):
    countries, regions, _, _ = result
    assert regions["total_population"].sum() == countries["population"].sum()


# ── Correlations ─────────────────────────────────────────────────────────────

def test_correlations_stay_within_bounds(result):
    *_, correlations = result
    assert correlations["correlation"].between(-1, 1).all()


def test_every_correlation_reports_its_sample_size(result):
    """A correlation without n is not interpretable."""
    *_, correlations = result
    assert (correlations["n_obs"] >= 20).all()


def test_correlations_are_not_self_pairs(result):
    *_, correlations = result
    assert (correlations["indicator_x"] != correlations["indicator_y"]).all()
