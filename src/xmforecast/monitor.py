"""Drift monitoring on the hourly demand series and on the model's own errors.

Three independent signals, each turned into an alarm by a threshold that is calibrated on
a clean period (never picked by hand):
  - level:       log of (mean of the last 7 days / mean of the 7 days before)
  - shape:       largest hourly change in the average daily profile (share of the day's
                 demand per hour), last 7 days vs the 7 days before
  - performance: rolling mean of the daily MAPE of the deployed model

Comparing with the *previous* week, not a fixed reference, keeps the monitor from
alarming on the slow growth of demand that the model already handles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_mean(series: pd.Series) -> pd.Series:
    return series.groupby(series.index.normalize()).mean()


def level_signal(series: pd.Series, window: int = 7, ignore_days: set | None = None) -> pd.Series:
    """log(mean of the last `window` days / mean of the `window` days before), per day.

    `ignore_days` (e.g. public holidays) are left out of the means, so a holiday week is not
    mistaken for a change; a window needs at least `window - 2` usable days.
    """
    d = _daily_mean(series)
    if ignore_days:
        d = d.where(~d.index.isin(ignore_days))
    m = d.rolling(window, min_periods=window - 2 if ignore_days else window).mean()
    return np.log(m / m.shift(window))


def shape_signal(series: pd.Series, window: int = 7, ignore_days: set | None = None) -> pd.Series:
    """Largest hourly difference of the average daily profile vs. the previous window."""
    day = series.index.normalize()
    share = series / series.groupby(day).transform("sum")
    prof = share.groupby([day, share.index.hour]).mean().unstack()
    prof = prof.dropna()  # keep complete days only
    if ignore_days:
        prof = prof.mask(pd.Series(prof.index.isin(ignore_days), index=prof.index), axis=0)
    avg = prof.rolling(window, min_periods=window - 2 if ignore_days else window).mean()
    return (avg - avg.shift(window)).abs().max(axis=1)


def performance_signal(daily_mape: pd.Series, window: int = 14) -> pd.Series:
    """Rolling mean of the daily MAPE (percentage points)."""
    return daily_mape.rolling(window).mean()


def calibrate_threshold(clean_signal: pd.Series, q: float = 0.995, two_sided: bool = True) -> float:
    """Threshold = the q-quantile of the signal over a clean period."""
    x = clean_signal.dropna()
    return float((x.abs() if two_sided else x).quantile(q))


def alarms(signal: pd.Series, threshold: float, two_sided: bool = True) -> pd.Series:
    return (signal.abs() if two_sided else signal) > threshold


def first_alarm(alarm: pd.Series, after: str | pd.Timestamp) -> pd.Timestamp | None:
    hits = alarm[alarm.index >= pd.Timestamp(after)]
    hits = hits[hits]
    return hits.index[0] if len(hits) else None


def sustained_alarm(performance_alarm: pd.Series, consecutive_days: int = 3) -> bool:
    """True when an alarm has been on for the last `consecutive_days` days: open an investigation.

    This is deliberately NOT an automatic "retrain" trigger: in the simulation, retraining at
    an alarm gained at most 0.05 MAPE points and cannot fix a broken data pipeline
    (see docs/DEVLOG.md, Phase 10).
    """
    tail = performance_alarm.tail(consecutive_days)
    return bool(len(tail) == consecutive_days and tail.all())
