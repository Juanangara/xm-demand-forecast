import pandas as pd

from xmforecast.data import HOUR_COLS, to_long
from xmforecast.quality import quality_by_year, usable_from


def _wide(days: int, nan_hours: tuple[int, ...] = ()) -> pd.DataFrame:
    rows = []
    for d in range(days):
        row = {"Date": f"2024-01-{d + 1:02d}"}
        for i, c in enumerate(HOUR_COLS):
            row[c] = float("nan") if i in nan_hours else 100.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def test_to_long_hour01_is_midnight_and_sorted():
    long = to_long(_wide(2))
    assert len(long) == 48
    assert long["ts"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    assert long["ts"].iloc[-1] == pd.Timestamp("2024-01-02 23:00")
    assert long["ts"].is_monotonic_increasing


def test_to_long_keeps_missing_values_as_nan():
    long = to_long(_wide(1, nan_hours=(5,)))
    assert long["value"].isna().sum() == 1


def test_quality_counts_absent_hours_as_missing():
    # Jan 1 and Jan 3 present, Jan 2 absent -> its 24 hours are one gap
    wide = _wide(3).drop(index=1)
    row = quality_by_year(to_long(wide)).iloc[0]
    assert row["expected_h"] == 72
    assert row["missing_h"] == 24
    assert row["longest_gap_h"] == 24


def test_usable_from_requires_all_later_years_clean():
    report = pd.DataFrame({"year": [2000, 2001, 2002], "missing_pct": [5.0, 0.1, 0.2]})
    assert usable_from(report) == 2001
    report.loc[2, "missing_pct"] = 3.0
    assert usable_from(report) is None


def test_to_long_coerces_text_values_to_numeric():
    wide = _wide(1).astype({c: object for c in HOUR_COLS})
    wide.loc[0, HOUR_COLS[0]] = "123.5"
    wide.loc[0, HOUR_COLS[1]] = "n/a"
    long = to_long(wide)
    assert pd.api.types.is_float_dtype(long["value"])
    assert long["value"].iloc[0] == 123.5
    assert long["value"].isna().sum() == 1


def test_quality_does_not_count_future_hours_of_partial_year_as_missing():
    report = quality_by_year(to_long(_wide(3)))
    row = report.iloc[0]
    assert row["expected_h"] == 72
    assert row["missing_h"] == 0
