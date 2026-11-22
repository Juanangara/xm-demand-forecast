"""Download and cache hourly series from XM (Colombian power-system operator).

XM returns one row per day with 24 columns (Values_Hour01..24) and allows at most
31 days per request, so history is fetched month by month and cached as Parquet.
"""
from __future__ import annotations

import calendar
import datetime as dt
import time
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
HOUR_COLS = [f"Values_Hour{h:02d}" for h in range(1, 25)]


def _month_cache(metric: str, entity: str, year: int, month: int) -> Path:
    return RAW_DIR / f"{metric}_{entity}" / f"{year}-{month:02d}.parquet"


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide XM frame (one row per day) -> long frame with `ts` and `value`.

    Hour 01 is the hour ending at 01:00, so timestamps are the *start* of each hour.
    Rows with a missing value are kept as NaN so gaps stay visible.
    """
    cols = [c for c in HOUR_COLS if c in wide.columns]
    long = wide.melt(id_vars=["Date"], value_vars=cols, var_name="h", value_name="value")
    long["ts"] = pd.to_datetime(long["Date"]) + pd.to_timedelta(
        long["h"].str[-2:].astype(int) - 1, unit="h"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long[["ts", "value"]].sort_values("ts").reset_index(drop=True)


def fetch_month(metric: str, entity: str, year: int, month: int, retries: int = 3) -> pd.DataFrame:
    """Return the long frame for one month, from cache if present."""
    from pydataxm.pydataxm import ReadDB  # imported lazily: slow and network-bound

    path = _month_cache(metric, entity, year, month)
    if path.exists():
        return pd.read_parquet(path)

    last_day = calendar.monthrange(year, month)[1]
    start, end = dt.date(year, month, 1), dt.date(year, month, last_day)
    for attempt in range(retries):
        try:
            wide = ReadDB().request_data(metric, entity, start, end)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    if wide is None or wide.empty:
        long = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})
    else:
        long = to_long(wide)
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(path, index=False)
    return long


def download(metric: str, entity: str, start_year: int, end: dt.date) -> pd.DataFrame:
    """Download every month from `start_year` until the month containing `end`."""
    frames = []
    for year in range(start_year, end.year + 1):
        last = end.month if year == end.year else 12
        for month in range(1, last + 1):
            frames.append(fetch_month(metric, entity, year, month))
        print(f"{metric} {year} ok", flush=True)
    df = pd.concat(frames, ignore_index=True)
    # Months cached before to_long coerced values may still hold text.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df
