"""
Guards on the raw country payload.

The first test here is a regression test for a real incident: the REST Countries
v3.1 API was deprecated and began returning

    {"success": false, "data": null, "errors": [{"message": "..."}]}

with HTTP 200. The pipeline called len() on that dict, got 3 (its key count),
logged "3 records" and carried on loading a warehouse built on three rows.
Nothing failed. The bad data reached the dashboard.
"""
import pytest

from extract import MIN_COUNTRIES, validate_countries


def test_rejects_api_error_envelope():
    """A dict must never pass as a dataset, however many keys it has."""
    payload = {
        "success": False,
        "data": None,
        "errors": [{"message": "This API version has been deprecated."}],
    }
    with pytest.raises(ValueError, match="Expected a list"):
        validate_countries(payload)


def test_rejects_truncated_list(valid_countries):
    with pytest.raises(ValueError, match="expected at least"):
        validate_countries(valid_countries[:10])


def test_rejects_missing_required_fields(valid_countries):
    del valid_countries[0]["area"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_countries(valid_countries)


def test_rejects_records_without_country_code(valid_countries):
    valid_countries[7]["cca2"] = ""
    with pytest.raises(ValueError, match="no cca2 code"):
        validate_countries(valid_countries)


def test_accepts_a_healthy_payload(valid_countries):
    assert len(valid_countries) >= MIN_COUNTRIES
    validate_countries(valid_countries)  # must not raise
