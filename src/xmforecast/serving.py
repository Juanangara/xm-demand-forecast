"""Load the trained model and forecast one day ahead from an hourly history.

A forecast for day D uses only the history strictly before D 00:00, exactly like the
walk-forward evaluation, so what the API returns is what was measured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
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


def forecast_day(champion: Champion, series: pd.Series, day: pd.Timestamp) -> pd.Series:
    """24 hourly forecasts for `day` (kWh), using only history before `day` 00:00."""
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
    pred = champion.booster.predict(f[FEATURES]) * f["base"].to_numpy()
    return pd.Series(pred, index=hours, name="forecast_kwh")
