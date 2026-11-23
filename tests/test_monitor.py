import numpy as np
import pandas as pd

from xmforecast.monitor import (
    alarms,
    calibrate_threshold,
    first_alarm,
    level_signal,
    performance_signal,
    shape_signal,
    sustained_alarm,
)


def _series(days: int = 80, noise: float = 0.0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(0)
    base = 1000 + 200 * np.sin(2 * np.pi * (idx.hour - 6) / 24)
    return pd.Series(base + rng.normal(0, noise, len(idx)), index=idx)


def test_level_signal_is_zero_for_stable_series_and_jumps_on_a_step():
    s = _series()
    assert level_signal(s).dropna().abs().max() < 1e-9
    step = s.copy()
    step[step.index >= "2024-02-20"] *= 1.10
    sig = level_signal(step)
    # step on 02-20: on 02-26 the last 7 days are all new and the 7 before are all old
    assert abs(sig["2024-02-26"] - np.log(1.10)) < 1e-6
    assert abs(sig["2024-02-10"]) < 1e-9  # unchanged before the step
    # it is a change detector, not a level detector: two weeks later both windows are new
    assert abs(sig["2024-03-05"]) < 1e-9


def test_shape_signal_ignores_pure_level_changes_but_sees_profile_changes():
    s = _series()
    scaled = s * 3
    assert shape_signal(scaled).dropna().abs().max() < 1e-9  # share of the day is unchanged
    reshaped = s.copy()
    evening = (reshaped.index >= "2024-02-20") & reshaped.index.hour.isin([18, 19, 20, 21])
    reshaped[evening] *= 1.2
    sig = shape_signal(reshaped)
    assert sig["2024-02-26"] > 0.005  # peak: one week of new shape against one of old
    assert abs(sig["2024-03-05"]) < 1e-9  # back to zero once both windows hold the new shape


def test_calibrate_threshold_is_the_quantile_of_the_clean_signal():
    clean = pd.Series(np.linspace(-1, 1, 1001))
    assert abs(calibrate_threshold(clean, q=0.9) - 0.9) < 0.01


def test_alarms_first_alarm_and_retrain_rule():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    sig = pd.Series([0, 0, 0, 5, 0, 5, 5, 5, 5, 5], index=idx, dtype=float)
    a = alarms(sig, 1.0)
    assert first_alarm(a, "2024-01-01") == pd.Timestamp("2024-01-04")
    assert first_alarm(a, "2024-01-05") == pd.Timestamp("2024-01-06")
    assert sustained_alarm(a, 3) is True
    assert sustained_alarm(a.iloc[:5], 3) is False  # alarm was not on for 3 days in a row
    assert first_alarm(alarms(sig * 0, 1.0), "2024-01-01") is None


def test_performance_signal_is_a_rolling_mean():
    m = pd.Series(range(1, 21), index=pd.date_range("2024-01-01", periods=20, freq="D"), dtype=float)
    assert performance_signal(m, 14).iloc[-1] == m.iloc[-14:].mean()


def test_ignore_days_removes_a_holiday_week_false_alarm_without_hiding_a_real_step():
    s = _series()
    holiday = pd.Timestamp("2024-02-27")
    low = s.copy()
    low[low.index.normalize() == holiday] *= 0.5  # a holiday: half the usual demand
    plain = level_signal(low)["2024-03-01"]
    aware = level_signal(low, ignore_days={holiday})["2024-03-01"]
    assert abs(plain) > 0.05 and abs(aware) < 1e-9  # the false alarm disappears
    step = s.copy()
    step[step.index >= "2024-02-20"] *= 1.10
    assert abs(level_signal(step, ignore_days={holiday})["2024-02-26"] - np.log(1.10)) < 1e-6


def test_shape_signal_ignore_days_skips_an_odd_day_but_still_sees_a_real_shape_change():
    s = _series()
    odd = pd.Timestamp("2024-02-27")
    mask = s.index.normalize() == odd
    s2 = s.copy()
    s2[mask & (s2.index.hour == 12)] *= 3  # one strange day (a holiday with another profile)
    assert shape_signal(s2)["2024-03-01"] > 0.01  # plain monitor alarms on it
    assert abs(shape_signal(s2, ignore_days={odd})["2024-03-01"]) < 1e-9  # calendar-aware does not
    reshaped = s.copy()
    ev = (reshaped.index >= "2024-02-20") & reshaped.index.hour.isin([18, 19, 20, 21])
    reshaped[ev] *= 1.2
    assert shape_signal(reshaped, ignore_days={odd})["2024-02-26"] > 0.005
