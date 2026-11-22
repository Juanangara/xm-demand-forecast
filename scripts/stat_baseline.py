"""Classical reference model: MSTL (daily + weekly seasonality) with an ETS trend.

For every day D in the test period it is fitted ONLY on the 6 weeks before D 00:00 and
forecasts the 24 hours of D. It knows nothing about holidays: it is the "good classical
model without calendar knowledge" to compare LightGBM against, not a strawman.

Usage: python scripts/stat_baseline.py [--test-start 2023-01-01] [--weeks 6]
Writes data/pred_mstl.parquet (column `pred`, indexed by hour).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoETS

    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--weeks", type=int, default=6)
    p.add_argument("--n-jobs", type=int, default=4)
    a = p.parse_args()

    s = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    window = pd.Timedelta(weeks=a.weeks)
    last_day = s.index.max().normalize()
    origins = pd.date_range(a.test_start, last_day, freq="D")
    model = MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))

    out, t0 = [], time.time()
    chunk = 100
    for i in range(0, len(origins), chunk):
        part = origins[i : i + chunk]
        frames = []
        for o in part:
            h = s[o - window : o - pd.Timedelta(hours=1)]  # strictly before the forecast day
            frames.append(pd.DataFrame({"unique_id": o.strftime("%Y-%m-%d"), "ds": h.index, "y": h.values}))
        fc = StatsForecast(models=[model], freq="h", n_jobs=a.n_jobs).forecast(df=pd.concat(frames), h=24)
        out.append(fc[["ds", "MSTL"]])
        print(f"{part[-1].date()} done ({time.time() - t0:.0f}s)", flush=True)

    pred = pd.concat(out).set_index("ds")["MSTL"].rename("pred").sort_index()
    pred.index.name = "ts"
    pred.to_frame().to_parquet("data/pred_mstl.parquet")
    print("saved", len(pred), "hours")


if __name__ == "__main__":
    main()
