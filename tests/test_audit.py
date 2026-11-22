import pandas as pd

from xmforecast.audit import flag_anomalies, local_ratio, mask_disturbances


def _weekly_series(weeks: int = 10) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=168 * weeks, freq="h")
    # identical weekly pattern, so the same hour of neighbouring weeks always matches
    return pd.Series([100.0 + (i % 168) for i in range(len(idx))], index=idx)


def test_local_ratio_is_one_for_a_perfectly_periodic_series():
    r = local_ratio(_weekly_series()).dropna()
    assert (r.round(6) == 1.0).all()


def test_flag_anomalies_finds_a_sudden_drop_and_nothing_else():
    s = _weekly_series()
    t = pd.Timestamp("2024-02-14 10:00")
    s[t] = s[t] * 0.1
    flagged = flag_anomalies(s)
    assert list(flagged.index) == [t]
    assert flagged["ratio"].iloc[0] < 0.2


def test_flag_anomalies_returns_empty_for_clean_series():
    assert flag_anomalies(_weekly_series()).empty


def test_mask_disturbances_sets_reviewed_hours_to_nan_without_touching_input():
    idx = pd.date_range("2007-04-25", "2007-04-27 23:00", freq="h")
    s = pd.Series(1.0, index=idx)
    clean = mask_disturbances(s)
    assert clean["2007-04-26 10:00":"2007-04-26 13:00"].isna().all()
    assert clean["2007-04-26 14:00"] == 1.0
    assert not s.isna().any()  # original untouched
