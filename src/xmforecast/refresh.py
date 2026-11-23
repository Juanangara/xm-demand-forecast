"""Daily refresh: update the history, forecast the next day, and keep a prospective track record.

The pure functions here have no network access; scripts/refresh.py does the downloading.
A forecast is only ever made for the first day that has no data yet, and the first forecast
made for a day is never overwritten, so the log is an honest out-of-sample record.
"""
from __future__ import annotations

import pandas as pd

from .intervals import cover


def complete_days(series: pd.Series) -> pd.Series:
    """Keep only calendar days that have all 24 hours with a value (a partial day is dropped)."""
    s = series.dropna()
    counts = s.groupby(s.index.normalize()).transform("size")
    return s[counts == 24]


# Daily-total ratio against the same weekday of the previous 4 weeks, over 9,794 days (2000-2026):
# from 0.70 (New Year's Day) to 1.25. XM's most recent days can arrive provisional and incomplete
# (observed: 24 % of normal), so a day outside this band, with margin, is not accepted.
GATE_LOW, GATE_HIGH = 0.6, 1.4


def daily_ratio(history: pd.Series, day: pd.Timestamp, values: pd.Series, weeks: int = 4) -> float | None:
    """Mean demand of `day` over the median of the same weekday in the previous `weeks` weeks.

    None when fewer than two reference days exist, in which case the day cannot be judged.
    """
    daily = history.groupby(history.index.normalize()).mean()
    same_weekday = [day - pd.Timedelta(weeks=k) for k in range(1, weeks + 1)]
    ref = [daily[d] for d in same_weekday if d in daily.index]
    if len(ref) < 2:
        return None
    return float(values.mean() / pd.Series(ref).median())


def accept_days(
    existing: pd.Series, new: pd.Series, low: float = GATE_LOW, high: float = GATE_HIGH
) -> tuple[pd.Series, list[dict]]:
    """Merge `new` into `existing`, day by day in order, keeping only days that pass the gate.

    Only complete days (24 hours) are considered; a day that revises an existing one replaces it.
    Each candidate is judged against the history accepted so far (never against later days).
    Returns the new history and the rejected days as [{"date", "ratio"}].
    """
    hist = existing.copy()
    cand = complete_days(new)
    rejected = []
    for day in sorted(set(cand.index.normalize())):
        vals = cand[cand.index.normalize() == day]
        ratio = daily_ratio(hist[hist.index < day], day, vals)
        if ratio is not None and not (low <= ratio <= high):
            rejected.append({"date": day, "ratio": ratio})
            continue
        hist = pd.concat([hist[~hist.index.isin(vals.index)], vals]).sort_index()
    return hist, rejected


def next_day(series: pd.Series) -> pd.Timestamp:
    """The first day after the last complete day of the history: the day to forecast."""
    return series.index.max().normalize() + pd.Timedelta(days=1)


def append_forecast(log: pd.DataFrame | None, forecast: pd.DataFrame, made_at: str, model: str) -> pd.DataFrame:
    """Add one day's forecast (index = hour) to the log unless that day is already there."""
    new = forecast.rename_axis("ts").reset_index().assign(made_at=made_at, model=model)
    if log is None or log.empty:
        return new
    day = new["ts"].dt.normalize().iloc[0]
    if (log["ts"].dt.normalize() == day).any():
        return log  # never overwrite the first forecast made for a day
    return pd.concat([log, new], ignore_index=True)


def scoreboard(log: pd.DataFrame | None, series: pd.Series) -> pd.DataFrame:
    """Per-day MAPE and interval coverage for every logged day whose 24 real values are now known."""
    if log is None or log.empty:
        return pd.DataFrame(columns=["date", "mape_pct", "in_80", "in_95"])
    rows = []
    for day, g in log.groupby(log["ts"].dt.normalize()):
        y = series.reindex(g["ts"])
        if y.isna().any():
            continue
        y.index = g.index
        rows.append({
            "date": day,
            "mape_pct": float(((g["forecast_kwh"] - y).abs() / y).mean() * 100),
            "in_80": float(cover(y, g["lower_80_kwh"], g["upper_80_kwh"]).mean()) if "lower_80_kwh" in g else None,
            "in_95": float(cover(y, g["lower_95_kwh"], g["upper_95_kwh"]).mean()) if "lower_95_kwh" in g else None,
        })
    return pd.DataFrame(rows)
