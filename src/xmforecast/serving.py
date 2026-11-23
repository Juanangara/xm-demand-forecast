"""Load the trained model and forecast one day ahead from an hourly history.

A forecast for day D uses only the history strictly before D 00:00, exactly like the
walk-forward evaluation, so what the API returns is what was measured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, build_features

MIN_HISTORY_DAYS = 15  # the longest lag is 336 h (14 days) and it needs a value behind it


@dataclass
class Champion:
    booster: lgb.Booster
    meta: dict


def load_champion(model_dir: str | Path) -> Champion:
    d = Path(model_dir)
    return Champion(lgb.Booster(model_file=str(d / "model.txt")), json.loads((d / "meta.json").read_text("utf-8")))


class ForecastError(ValueError):
    """The requested day cannot be forecast from the available history."""


def _day_features(series: pd.Series, day: pd.Timestamp) -> pd.DataFrame:
    """Feature rows for the 24 hours of `day`, built only from the history before it."""
    day = pd.Timestamp(day).normalize()
    series = series.sort_index()
    first, last = series.index.min(), series.index.max()
    if day < first.normalize() + pd.Timedelta(days=MIN_HISTORY_DAYS):
        raise ForecastError(f"day {day.date()} is too early: needs {MIN_HISTORY_DAYS} days of history")
    if last < day - pd.Timedelta(hours=1):
        raise ForecastError(
            f"history ends at {last}; forecasting {day.date()} needs data up to {day - pd.Timedelta(hours=1)}"
        )

    history = series[series.index < day]  # never look at day D or later, even if present
    hours = pd.date_range(day, periods=24, freq="h")
    extended = history.reindex(pd.date_range(history.index.min(), hours[-1], freq="h"))
    f = build_features(extended).loc[hours]
    if f[FEATURES + ["base"]].isna().any().any():
        raise ForecastError("history has gaps that make some features unavailable for this day")
    return f


def forecast_day(champion: Champion, series: pd.Series, day: pd.Timestamp) -> pd.Series:
    """24 hourly forecasts for `day` (kWh), using only history before `day` 00:00."""
    f = _day_features(series, day)
    pred = champion.booster.predict(f[FEATURES]) * f["base"].to_numpy()
    return pd.Series(pred, index=f.index, name="forecast_kwh")


def group_code(day_type: pd.Series, ref168_holiday: pd.Series) -> pd.Series:
    """Error group of an hour: its day type, split by whether last week's copy was a holiday."""
    return day_type * 2 + ref168_holiday


def forecast_day_with_bands(champion: Champion, series: pd.Series, day: pd.Timestamp) -> pd.DataFrame:
    """Point forecast plus 80 % and 95 % intervals when the model carries an interval spec.

    The intervals are the point forecast times (1 + quantiles of the model's own relative errors
    on a recent year of walk-forward forecasts), by error group (see scripts/train_final.py).
    Columns: forecast_kwh and, if available, lower_80, upper_80, lower_95, upper_95.
    """
    f = _day_features(series, day)
    point = champion.booster.predict(f[FEATURES]) * f["base"].to_numpy()
    out = pd.DataFrame({"forecast_kwh": point}, index=f.index)
    spec = champion.meta.get("intervals")
    if not spec:
        return out
    codes = group_code(f["day_type"], f["ref168_holiday"]).astype(int).astype(str)
    quant = np.array([spec["groups"].get(c, spec["fallback"])["q"] for c in codes])  # (24, 4) relative errors
    # q = [q0.025, q0.1, q0.9, q0.975]
    out["lower_80"], out["upper_80"] = point * (1 + quant[:, 1]), point * (1 + quant[:, 2])
    out["lower_95"], out["upper_95"] = point * (1 + quant[:, 0]), point * (1 + quant[:, 3])
    return out
