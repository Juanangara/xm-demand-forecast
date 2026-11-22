"""Walk-forward evaluation for day-ahead hourly forecasts.

Setting: at 00:00 of day D we know everything before D and must forecast the 24 hours
of D. Folds are calendar months of the test period; each fold is scored on its own, so
a model that only works in some seasons cannot hide behind one global number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - actual
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": float((err.abs() / actual.abs()).mean() * 100),
    }


def seasonal_naive(series: pd.Series, lag_hours: int) -> pd.Series:
    """Forecast every hour with the value `lag_hours` earlier.

    Day-ahead safe only if lag_hours >= 24: the forecast for any hour of day D then
    uses values from before D's midnight.
    """
    if lag_hours < 24:
        raise ValueError("lag_hours < 24 would use information not known at forecast time")
    return series.shift(lag_hours, freq="h").reindex(series.index)


def walk_forward(series: pd.Series, pred: pd.Series, test_start: str) -> pd.DataFrame:
    """Score `pred` against `series` month by month from `test_start`.

    Hours where the prediction is NaN are dropped and counted in `n_dropped`.
    """
    df = pd.DataFrame({"actual": series, "pred": pred}).loc[test_start:]
    rows = []
    for month, g in df.groupby(df.index.to_period("M")):
        ok = g.dropna()
        row = {"fold": str(month), "n": len(ok), "n_dropped": len(g) - len(ok)}
        row.update(metrics(ok["actual"], ok["pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def summary(folds: pd.DataFrame) -> dict[str, float]:
    """Pooled error over all folds plus the spread across folds."""
    w = folds["n"]
    return {
        "mae": float(np.average(folds["mae"], weights=w)),
        "mape": float(np.average(folds["mape"], weights=w)),
        "mape_worst_fold": float(folds["mape"].max()),
        "mape_fold_std": float(folds["mape"].std(ddof=0)),
    }
