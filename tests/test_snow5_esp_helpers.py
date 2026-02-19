from __future__ import annotations

from datetime import datetime

from snowda.workflows.run_snow5_da import _replace_year, _resolve_esp_trigger_dates


def test_replace_year_handles_leap_day():
    leap = datetime(2012, 2, 29, 0, 0)
    shifted = _replace_year(leap, 2011)
    assert shifted == datetime(2011, 2, 28, 0, 0)


def test_resolve_esp_trigger_dates_filters_months():
    start = datetime(2011, 1, 1, 0, 0)
    end = datetime(2011, 4, 30, 0, 0)
    resolved = _resolve_esp_trigger_dates(
        start,
        end,
        explicit_dates=[datetime(2011, 2, 15, 0, 0)],
        frequency_days=30,
        months=[2, 4],
    )
    assert resolved == [
        datetime(2011, 1, 31, 0, 0).replace(month=2, day=15),
        datetime(2011, 4, 1, 0, 0),
    ]
