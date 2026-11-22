"""Data-quality checks for an hourly series (columns `ts`, `value`)."""
from __future__ import annotations

import pandas as pd


def quality_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year completeness: missing hours, duplicates, zeros, longest gap, outliers."""
    df = df.copy()
    df["year"] = df["ts"].dt.year
    last_ts = df["ts"].max()  # a partial final year must not count its future hours as gaps
    rows = []
    for year, g in df.groupby("year"):
        full = pd.date_range(f"{year}-01-01", min(pd.Timestamp(f"{year}-12-31 23:00"), last_ts), freq="h")
        s = g.drop_duplicates("ts").set_index("ts")["value"].reindex(full)
        missing = s.isna()
        # longest run of consecutive missing hours
        runs = missing.ne(missing.shift()).cumsum()
        longest = int(missing.groupby(runs).sum().max()) if missing.any() else 0
        med = s.median()
        mad = (s - med).abs().median()
        outliers = int((((s - med).abs() / (1.4826 * mad)) > 6).sum()) if mad else 0
        rows.append(
            {
                "year": year,
                "expected_h": len(full),
                "missing_h": int(missing.sum()),
                "missing_pct": round(100 * missing.mean(), 2),
                "duplicates": int(g["ts"].duplicated().sum()),
                "zeros": int((s == 0).sum()),
                "longest_gap_h": longest,
                "outliers_6mad": outliers,
            }
        )
    return pd.DataFrame(rows)


def usable_from(report: pd.DataFrame, max_missing_pct: float = 0.5) -> int | None:
    """First year from which every later year stays under the missing threshold."""
    ok = report["missing_pct"] <= max_missing_pct
    for i in range(len(report)):
        if ok.iloc[i:].all():
            return int(report["year"].iloc[i])
    return None
