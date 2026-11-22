"""Anomaly audit and cleaning rules for the hourly demand series.

Principle: the raw series is never edited. Anomalies are *flagged* by a detector, a
human reviews them, and only reviewed events go into KNOWN_DISTURBANCES. Training masks
those hours (sets them to NaN); the test period is never masked, so reported errors
stay honest.
"""
from __future__ import annotations

import pandas as pd

# Reviewed events: sharp drop with gradual recovery over a few hours. They look like
# real grid disturbances rather than a corrupted reading, but the cause is NOT verified.
# (start, end inclusive, note)
KNOWN_DISTURBANCES: list[tuple[str, str, str]] = [
    ("2002-03-16 23:00", "2002-03-17 01:00", "drop to ~49-72% of last week, recovers by 02:00; cause unverified"),
    ("2007-04-26 10:00", "2007-04-26 13:00", "drop to 6% of last week at 10:00, recovers by 14:00; cause unverified"),
]


def local_ratio(series: pd.Series, weeks: int = 4) -> pd.Series:
    """Value divided by the median of the same hour and weekday, +-`weeks` weeks around it."""
    shifts = [k for k in range(-weeks, weeks + 1) if k != 0]
    neighbours = pd.concat([series.shift(168 * k, freq="h").reindex(series.index) for k in shifts], axis=1)
    return series / neighbours.median(axis=1)


def flag_anomalies(series: pd.Series, low: float = 0.6, high: float = 1.4) -> pd.DataFrame:
    """Hours whose value is far from the same hour in neighbouring weeks.

    Public holidays are legitimately low and WILL be flagged; the detector does not know
    the calendar. A flag is a candidate for review, not a verdict.
    """
    ratio = local_ratio(series)
    out = pd.DataFrame({"value": series, "ratio": ratio})
    return out[(out["ratio"] < low) | (out["ratio"] > high)]


def mask_disturbances(series: pd.Series) -> pd.Series:
    """Copy of `series` with the reviewed disturbance hours set to NaN (for training only)."""
    clean = series.copy()
    for start, end, _ in KNOWN_DISTURBANCES:
        clean.loc[start:end] = float("nan")
    return clean
