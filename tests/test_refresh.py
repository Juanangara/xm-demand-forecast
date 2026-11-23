import numpy as np
import pandas as pd

from xmforecast.refresh import accept_days, append_forecast, complete_days, next_day, scoreboard


def _series(start, hours, value=100.0):
    return pd.Series(value, index=pd.date_range(start, periods=hours, freq="h"))


def test_complete_days_drops_a_partial_day():
    s = pd.concat([_series("2024-01-01", 48), _series("2024-01-03", 10)])
    out = complete_days(s)
    assert len(out) == 48 and out.index.max() == pd.Timestamp("2024-01-02 23:00")


def test_complete_days_treats_a_nan_hour_as_missing():
    s = _series("2024-01-01", 48)
    s.iloc[5] = np.nan
    assert len(complete_days(s)) == 24  # the first day has a hole, the second is complete


def test_accept_days_adds_sane_days_and_lets_a_sane_revision_replace_old_values():
    old = _series("2024-01-01", 24 * 35, 100.0)
    new = _series("2024-02-04", 48, 105.0)  # Feb 4 and Feb 5 are new; both within 5 % of normal
    out, rejected = accept_days(old, new)
    assert not rejected and out.index.max() == pd.Timestamp("2024-02-05 23:00")
    revised, rej = accept_days(out, _series("2024-02-04", 24, 110.0))  # XM revises Feb 4
    assert not rej and revised["2024-02-04 10:00"] == 110.0 and revised["2024-02-05 10:00"] == 105.0


def test_accept_days_rejects_a_provisional_day_at_a_quarter_of_normal():
    old = _series("2024-01-01", 24 * 35, 100.0)
    new = _series("2024-02-04", 48, 25.0)  # what XM returned on 18-19 Nov 2026: about 24 % of normal
    out, rejected = accept_days(old, new)
    assert [r["date"] for r in rejected] == [pd.Timestamp("2024-02-04"), pd.Timestamp("2024-02-05")]
    assert rejected[0]["ratio"] == 0.25
    assert out.index.max() == old.index.max()  # the history did not grow


def test_accept_days_judges_each_day_against_the_history_accepted_before_it():
    old = _series("2024-01-01", 24 * 35, 100.0)
    new = pd.concat([_series("2024-02-04", 24, 100.0), _series("2024-02-05", 24, 30.0)])
    out, rejected = accept_days(old, new)
    assert len(rejected) == 1 and rejected[0]["date"] == pd.Timestamp("2024-02-05")
    assert out.index.max() == pd.Timestamp("2024-02-04 23:00")


def test_accept_days_ignores_a_partial_day_and_cannot_judge_without_references():
    old = _series("2024-01-01", 24 * 35, 100.0)
    out, rejected = accept_days(old, _series("2024-02-04", 10, 100.0))  # only 10 hours
    assert not rejected and out.index.max() == old.index.max()
    fresh = _series("2024-01-01", 24 * 3, 100.0)  # fewer than two same-weekday references
    out2, rej2 = accept_days(fresh, _series("2024-01-04", 24, 5.0))
    assert not rej2 and out2.index.max() == pd.Timestamp("2024-01-04 23:00")  # accepted: it cannot be judged


def test_next_day_is_the_day_after_the_last_complete_one():
    assert next_day(_series("2024-01-01", 48)) == pd.Timestamp("2024-01-03")


def _forecast(day, value=100.0):
    idx = pd.date_range(day, periods=24, freq="h")
    return pd.DataFrame({"forecast_kwh": value, "lower_80_kwh": value - 10, "upper_80_kwh": value + 10,
                         "lower_95_kwh": value - 20, "upper_95_kwh": value + 20}, index=idx)


def test_the_first_forecast_for_a_day_is_never_overwritten():
    log = append_forecast(None, _forecast("2024-01-05", 100.0), "t1", "m")
    log = append_forecast(log, _forecast("2024-01-05", 999.0), "t2", "m")  # same day again
    assert len(log) == 24 and (log["forecast_kwh"] == 100.0).all() and (log["made_at"] == "t1").all()
    log = append_forecast(log, _forecast("2024-01-06"), "t3", "m")
    assert len(log) == 48


def test_scoreboard_only_scores_days_with_all_24_real_values():
    log = append_forecast(None, _forecast("2024-01-05", 100.0), "t1", "m")
    log = append_forecast(log, _forecast("2024-01-06", 100.0), "t2", "m")
    actual = pd.concat([_series("2024-01-05", 24, 105.0), _series("2024-01-06", 10, 105.0)])  # Jan 6 partial
    sb = scoreboard(log, actual)
    assert len(sb) == 1 and sb["date"].iloc[0] == pd.Timestamp("2024-01-05")
    assert abs(sb["mape_pct"].iloc[0] - 5 / 105 * 100) < 1e-9
    assert sb["in_80"].iloc[0] == 1.0 and sb["in_95"].iloc[0] == 1.0  # 105 is inside 100 +- 10


def test_scoreboard_of_an_empty_log_is_empty():
    assert scoreboard(None, _series("2024-01-01", 24)).empty
