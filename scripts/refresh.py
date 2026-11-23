"""Daily refresh: update the demand history, forecast the next day, score past forecasts.

1. Re-reads the last few days from XM (they can be revised) and merges them into the history.
2. Forecasts the first day that has no data yet, with the champion model and its intervals.
3. Appends that forecast to data/live_forecasts.parquet (the first forecast for a day is kept)
   and prints how the forecasts made so far did against the real values: a prospective,
   out-of-sample record that only grows with time.

Nothing is published: everything is written under data/, which git ignores.
Usage: python scripts/refresh.py [--days-back 7] [--dry-run]
Restart the API afterwards if it is running (it reads the history at startup).
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from xmforecast.audit import flag_anomalies  # noqa: E402
from xmforecast.data import fetch_range  # noqa: E402
from xmforecast.refresh import accept_days, append_forecast, next_day, scoreboard  # noqa: E402
from xmforecast.serving import ForecastError, forecast_day_with_bands, load_champion  # noqa: E402

DATA = Path("data/DemaReal_Sistema.parquet")
LOG = Path("data/live_forecasts.parquet")
LATEST = Path("data/forecast_latest.json")
METRIC, ENTITY = "DemaReal", "Sistema"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days-back", type=int, default=7, help="how many days before the last one to re-read")
    p.add_argument("--dry-run", action="store_true", help="query and report, but write nothing")
    a = p.parse_args()

    existing = pd.read_parquet(DATA).set_index("ts")["value"].sort_index().dropna()
    old_end = existing.index.max()
    start = (old_end - pd.Timedelta(days=a.days_back)).date()
    today = dt.date.today()
    print(f"History ends {old_end}. Querying XM from {start} to {today}.")

    fetched = fetch_range(METRIC, ENTITY, start, today).set_index("ts")["value"].dropna()
    print(f"XM returned {len(fetched)} hours" + (f", last {fetched.index.max()}" if len(fetched) else ""))
    series, rejected = accept_days(existing, fetched)
    for r in rejected:
        print(f"REJECTED {r['date'].date()}: its daily demand is {r['ratio']:.2f}x that of the same weekday in the "
              f"previous weeks (accepted range 0.6 to 1.4); probably provisional data. It stays out of the history "
              f"and is re-read on the next run.")
    new_end = series.index.max()
    added = series[series.index > old_end]
    changed = fetched[fetched.index.isin(existing.index)]
    revised = int((changed.reindex(changed.index) != existing.reindex(changed.index)).sum()) if len(changed) else 0
    print(f"History now ends {new_end}: {len(added)} new hours ({len(added) // 24} complete days), "
          f"{revised} earlier hours revised by XM.")

    # data-quality sanity check on the new rows (holidays are flagged too: a flag is a candidate, not a verdict)
    flagged = flag_anomalies(series).loc[old_end:] if len(added) else pd.DataFrame()
    if len(flagged):
        days = sorted({d.date().isoformat() for d in flagged.index})
        print(f"Check: {len(flagged)} accepted hours deviate more than 40% from the same hour in nearby weeks: {days}")

    try:
        champion = load_champion("models/champion")
    except OSError as e:
        raise SystemExit(f"no model found ({e}); run scripts/train_final.py first") from e
    target = next_day(series)
    try:
        bands = forecast_day_with_bands(champion, series, target)
    except ForecastError as e:
        raise SystemExit(f"cannot forecast {target.date()}: {e}") from e

    made_at = dt.datetime.now().isoformat(timespec="seconds")
    log = pd.read_parquet(LOG) if LOG.exists() else None
    named = bands.rename(columns={c: f"{c}_kwh" for c in bands.columns if c != "forecast_kwh"})
    log = append_forecast(log, named, made_at, champion.meta["name"])
    mean_w = float(((bands["upper_80"] - bands["lower_80"]) / bands["forecast_kwh"]).mean() * 100)
    print(f"\nForecast for {target.date()} ({bands['forecast_kwh'].sum() / 1e6:,.1f} GWh in the day; "
          f"80% interval about {mean_w:.1f}% wide):")
    print(f"  peak {bands['forecast_kwh'].max() / 1e3:,.0f} MWh at {bands['forecast_kwh'].idxmax().hour:02d}:00, "
          f"minimum {bands['forecast_kwh'].min() / 1e3:,.0f} MWh at {bands['forecast_kwh'].idxmin().hour:02d}:00")

    sb = scoreboard(log, series)
    if len(sb):
        print(f"\nProspective record: {len(sb)} forecast day(s) already compared with real data.")
        print(f"  mean MAPE {sb['mape_pct'].mean():.2f}%  |  inside 80% interval "
              f"{sb['in_80'].mean() * 100:.0f}% of hours  |  inside 95% {sb['in_95'].mean() * 100:.0f}%")
    else:
        print("\nProspective record: no forecast has its real values yet (it starts with today's forecast).")

    if a.dry_run:
        print("\nDry run: nothing written.")
        return
    tmp = DATA.with_suffix(".tmp")
    series.rename("value").rename_axis("ts").reset_index().to_parquet(tmp, index=False)
    tmp.replace(DATA)
    log.to_parquet(LOG, index=False)
    LATEST.write_text(json.dumps({
        "date": str(target.date()), "made_at": made_at, "model": champion.meta["name"],
        "hours": [{"hour": t.hour, **{k: float(v) for k, v in row.items()}} for t, row in bands.iterrows()],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {DATA}, {LOG} and {LATEST}.")


if __name__ == "__main__":
    main()
