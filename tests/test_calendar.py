import pandas as pd

from xmforecast.calendar import day_type, holiday_dates


def _label(date: str, hours: int = 1) -> str:
    idx = pd.date_range(date, periods=hours, freq="h")
    return day_type(idx, holiday_dates(2023, 2026)).iloc[0]


def test_known_colombian_holidays_2024():
    # Dates checked by hand: Reyes moved to Monday, Holy Friday, Independence Day, Christmas
    for d in ["2024-01-08", "2024-03-29", "2024-07-20", "2024-12-25"]:
        assert _label(d) == "holiday", d


def test_regular_days_and_weekend():
    assert _label("2024-02-14") == "weekday"  # Wednesday, no holiday around
    assert _label("2024-02-17") == "sat"
    assert _label("2024-02-18") == "sun"


def test_holiday_on_weekend_is_still_holiday():
    assert _label("2024-07-20") == "holiday"  # a Saturday


def test_weekday_next_to_a_holiday_is_near_holiday():
    assert _label("2024-03-27") == "near_holiday"  # Wednesday before Holy Thursday
    assert _label("2024-01-09") == "near_holiday"  # Tuesday after the Monday holiday
