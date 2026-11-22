"""Day-type labels based on the Colombian public-holiday calendar (`holidays` package)."""
from __future__ import annotations

import holidays
import pandas as pd


def holiday_dates(start_year: int, end_year: int) -> set[pd.Timestamp]:
    co = holidays.country_holidays("CO", years=range(start_year, end_year + 1))
    return {pd.Timestamp(d) for d in co}


def day_type(index: pd.DatetimeIndex, holiday_set: set[pd.Timestamp]) -> pd.Series:
    """Label each timestamp: holiday, sat, sun, near_holiday (a weekday next to a holiday) or weekday.

    `holiday` wins over the weekday; `near_holiday` marks the days where bridges and
    long weekends distort the usual pattern.
    """
    days = index.normalize()
    one_day = pd.Timedelta(days=1)
    is_hol = days.isin(holiday_set)
    near = pd.Series(
        [(d - one_day) in holiday_set or (d + one_day) in holiday_set for d in days], index=index
    )
    dow = index.dayofweek
    label = pd.Series("weekday", index=index, dtype=object)
    label[near.values & (dow < 5)] = "near_holiday"
    label[dow == 5] = "sat"
    label[dow == 6] = "sun"
    label[is_hol] = "holiday"
    return label
