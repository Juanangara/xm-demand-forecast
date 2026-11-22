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
    # 2 days present out of a 366-day year -> everything else is missing
    report = quality_by_year(to_long(_wide(2)))
    row = report.iloc[0]
    assert row["expected_h"] == 8784
    assert row["missing_h"] == 8784 - 48
    assert row["longest_gap_h"] > 8000


def test_usable_from_requires_all_later_years_clean():
    report = pd.DataFrame({"year": [2000, 2001, 2002], "missing_pct": [5.0, 0.1, 0.2]})
    assert usable_from(report) == 2001
    report.loc[2, "missing_pct"] = 3.0
    assert usable_from(report) is None
