"""FastAPI service: day-ahead hourly demand forecast.

Config (environment): MODEL_DIR (default models/champion), DATA_PATH (default
data/DemaReal_Sistema.parquet). Run: uvicorn xmforecast.api:app
"""
from __future__ import annotations

import datetime as dt
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from .calendar import day_type, holiday_dates
from .serving import ForecastError, forecast_day_with_bands, load_champion


class HourForecast(BaseModel):
    hour: int
    forecast_kwh: float
    lower_80_kwh: float | None = None  # 80 % interval, present when the model carries an interval spec
    upper_80_kwh: float | None = None
    lower_95_kwh: float | None = None
    upper_95_kwh: float | None = None
    actual_kwh: float | None = None


class ForecastResponse(BaseModel):
    date: dt.date
    day_type: str
    is_holiday: bool
    model_version: str
    history_end: dt.datetime
    hours: list[HourForecast]
    mape_pct: float | None = None  # only when the real values for that day are in the history
    in_sample: bool  # True: the day is inside the final model's training window, so mape_pct is optimistic
    interval_method: str | None = None  # how the 80 % / 95 % intervals were built; None if there are none


def create_app(model_dir: str | None = None, data_path: str | None = None) -> FastAPI:
    model_dir = model_dir or os.environ.get("MODEL_DIR", "models/champion")
    data_path = data_path or os.environ.get("DATA_PATH", "data/DemaReal_Sistema.parquet")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.champion = load_champion(model_dir)
        df = pd.read_parquet(data_path)
        app.state.series = df.set_index("ts")["value"].sort_index().dropna()
        yield

    app = FastAPI(title="XM day-ahead demand forecast", version="1.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        s = request.app.state.series
        return {"status": "ok", "history_start": str(s.index.min()), "history_end": str(s.index.max())}

    @app.get("/model")
    def model_info(request: Request) -> dict:
        return request.app.state.champion.meta

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(request: Request, date: dt.date = Query(..., description="Day to forecast, YYYY-MM-DD")):
        st = request.app.state
        day = pd.Timestamp(date)
        try:
            bands = forecast_day_with_bands(st.champion, st.series, day)
        except ForecastError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        pred = bands["forecast_kwh"]

        hol = holiday_dates(day.year - 1, day.year + 1)
        label = str(day_type(pd.DatetimeIndex([day]), hol).iloc[0])
        actual = st.series.reindex(pred.index)
        mape = None
        if actual.notna().all():
            mape = float(((pred - actual).abs() / actual).mean() * 100)
        meta = st.champion.meta
        trained_until = pd.Timestamp(meta["trained_until"])
        window_start = trained_until - pd.DateOffset(years=meta["window_years"])
        in_sample = bool(window_start <= day <= trained_until)
        has_bands = "lower_80" in bands.columns

        def band(row, name):
            return float(row[name]) if has_bands else None

        hours = [
            HourForecast(
                hour=t.hour, forecast_kwh=float(row["forecast_kwh"]),
                lower_80_kwh=band(row, "lower_80"), upper_80_kwh=band(row, "upper_80"),
                lower_95_kwh=band(row, "lower_95"), upper_95_kwh=band(row, "upper_95"),
                actual_kwh=None if pd.isna(a) else float(a),
            )
            for (t, row), a in zip(bands.iterrows(), actual.to_numpy())
        ]
        return ForecastResponse(
            date=date, day_type=label, is_holiday=label == "holiday",
            model_version=f"{st.champion.meta['name']} (code {st.champion.meta['code_commit']})",
            history_end=st.series.index.max().to_pydatetime(), hours=hours, mape_pct=mape,
            in_sample=in_sample,
            interval_method=meta.get("intervals", {}).get("method") if has_bands else None,
        )

    return app


app = create_app()
