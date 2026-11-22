import numpy as np
import pandas as pd

from xmforecast.features import FEATURES, build_features


def _series(days: int = 60) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(0)
    return pd.Series(1000 + 50 * np.sin(np.arange(len(idx)) / 24) + rng.normal(0, 5, len(idx)), index=idx)


def test_features_for_day_d_do_not_depend_on_day_d_or_later():
    """The leak test: change everything from day D onwards; D's features must not move."""
    s = _series()
    day = pd.Timestamp("2024-02-10")
    changed = s.copy()
    changed[changed.index >= day] = changed[changed.index >= day] * 7 + 123
    a = build_features(s)
    b = build_features(changed)
    rows = (a.index >= day) & (a.index < day + pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(a.loc[rows, FEATURES + ["base"]], b.loc[rows, FEATURES + ["base"]])
    assert not a.loc[rows, "y"].equals(b.loc[rows, "y"])  # the target itself did change


def test_base_is_last_weeks_value_and_ratios_are_scale_free():
    s = _series()
    f = build_features(s)
    t = pd.Timestamp("2024-02-10 15:00")
    assert f.loc[t, "base"] == s[t - pd.Timedelta(hours=168)]
    assert f.loc[t, "r24"] == s[t - pd.Timedelta(hours=24)] / f.loc[t, "base"]
    scaled = build_features(s * 1000)
    assert np.isclose(scaled.loc[t, "r24"], f.loc[t, "r24"])
    assert np.isclose(scaled.loc[t, "mean7_ratio"], f.loc[t, "mean7_ratio"])


def test_masked_hours_propagate_as_nan_not_as_wrong_values():
    s = _series()
    s["2024-02-05 10:00"] = np.nan
    f = build_features(s)
    # one week later that hour is the 'base' of the next week's same hour
    assert np.isnan(f.loc["2024-02-12 10:00", "base"])


def test_holiday_reference_flags():
    s = _series()
    f = build_features(s)
    # 2024-01-08 (Monday) is a holiday; 2024-01-15 copies it as its 168 h reference
    assert f.loc["2024-01-15 12:00", "ref168_holiday"] == 1
    assert f.loc["2024-01-16 12:00", "ref168_holiday"] == 0
    assert f.loc["2024-01-09 12:00", "ref24_holiday"] == 1
