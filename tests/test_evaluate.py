import pandas as pd
import pytest

from xmforecast.evaluate import metrics, seasonal_naive, summary, walk_forward


def _series(days: int = 20) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    return pd.Series(range(len(idx)), index=idx, dtype=float) + 100


def test_seasonal_naive_rejects_lags_that_leak_the_forecast_day():
    with pytest.raises(ValueError):
        seasonal_naive(_series(), 12)


def test_seasonal_naive_uses_value_from_lag_hours_before():
    s = _series()
    pred = seasonal_naive(s, 24)
    t = pd.Timestamp("2024-01-05 07:00")
    assert pred[t] == s[t - pd.Timedelta(hours=24)]
    assert pred.iloc[:24].isna().all()  # no history yet for the first day


def test_metrics_known_values():
    a = pd.Series([100.0, 200.0])
    p = pd.Series([110.0, 180.0])
    m = metrics(a, p)
    assert m["mae"] == 15.0
    assert m["mape"] == pytest.approx(10.0)  # (10% + 10%) / 2


def test_walk_forward_drops_nan_predictions_and_counts_them():
    s = _series(40)
    folds = walk_forward(s, seasonal_naive(s, 168), "2024-01-01")
    jan = folds[folds["fold"] == "2024-01"].iloc[0]
    assert jan["n_dropped"] == 168  # first week has no value one week earlier
    assert jan["n"] + jan["n_dropped"] == 31 * 24


def test_walk_forward_scores_each_month_separately():
    s = _series(60)
    folds = walk_forward(s, seasonal_naive(s, 24), "2024-01-01")
    assert list(folds["fold"]) == ["2024-01", "2024-02"]
    assert summary(folds)["mape_worst_fold"] >= summary(folds)["mape"]
