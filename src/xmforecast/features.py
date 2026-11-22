"""Day-ahead features. Every feature for hour t only uses values at least 24 h old.

At 00:00 of day D we forecast all 24 hours of D, so a lag of 24 h or more is always
known; any shorter lag would be the day being forecast (leak). `build_features` never
touches the future, and tests/test_features.py checks it by changing the future.
"""
from __future__ import annotations

import pandas as pd

from .calendar import day_type, holiday_dates

LAGS = (24, 48, 168, 336)
DAY_TYPES = ["weekday", "near_holiday", "sat", "sun", "holiday"]
FEATURES = [
    "hour", "dow", "month", "day_type", "ref24_holiday", "ref168_holiday", "ref336_holiday",
    "r24", "r48", "r336", "mean7_ratio",
]


def build_features(series: pd.Series, start_year: int | None = None, end_year: int | None = None) -> pd.DataFrame:
    """Return features plus `base` (= lag_168, the value the model scales) and `y`.

    The series may contain NaN (masked hours); they propagate to the features that use them.
    """
    s = series.sort_index()
    idx = s.index
    hol = holiday_dates((start_year or idx.min().year) - 1, (end_year or idx.max().year) + 1)

    lag = {k: s.shift(k, freq="h").reindex(idx) for k in LAGS}
    base = lag[168]
    # mean of the 168 h that end 24 h before t: all values are >= 24 h old
    mean7 = s.shift(24, freq="h").rolling("168h").mean().reindex(idx)

    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["day_type"] = pd.Categorical(day_type(idx, hol), categories=DAY_TYPES).codes
    for k in (24, 168, 336):
        f[f"ref{k}_holiday"] = (idx - pd.Timedelta(hours=k)).normalize().isin(hol).astype(int)
    f["r24"] = lag[24] / base
    f["r48"] = lag[48] / base
    f["r336"] = lag[336] / base
    f["mean7_ratio"] = mean7 / base
    f["base"] = base
    f["y"] = s
    return f
