"""Does the monitor work? Calibrate on clean data, then inject known failures.

Setup: a model trained up to 2022-12-31 is deployed FROZEN (never retrained) over
2023-01 to 2026-10. Thresholds are calibrated only on 2021-2022, using a model trained up to
2020-12-31, so nothing is tuned on the deployment period.

Scenarios injected from 2024-06-03 (a Monday):
  level_shift  : demand x1.10 (e.g. a new large customer or a metering change)
  evening_shape: hours 18-21 x1.12 (a change in the daily load profile)
  unit_bug     : values / 1000 (a pipeline bug: kWh turned into MWh)
Two versions of the level/shape monitors are compared: plain, and calendar-aware (public
holidays left out of the weekly means). The last block measures whether retraining when the
performance alarm fires would actually have paid off (out of sample, next 28 days).
Usage: python scripts/simulate_drift.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402

from xmforecast.audit import mask_disturbances  # noqa: E402
from xmforecast.calendar import holiday_dates  # noqa: E402
from xmforecast.features import FEATURES, build_features  # noqa: E402
from xmforecast.forecast import DEFAULT_PARAMS  # noqa: E402
from xmforecast.monitor import (alarms, calibrate_threshold, first_alarm, level_signal,  # noqa: E402
                                performance_signal, shape_signal, sustained_alarm)
from xmforecast.stats import daily_mape  # noqa: E402

T = pd.Timestamp("2024-06-03")
DEPLOY, CAL = "2023-01-01", ("2021-01-01", "2022-12-31")


def train_frozen(series: pd.Series, train_end: str, years: int = 5) -> lgb.LGBMRegressor:
    f = build_features(mask_disturbances(series.loc[:train_end]))
    f = f[f.index >= pd.Timestamp(train_end) - pd.DateOffset(years=years)]
    f = f.dropna(subset=FEATURES + ["base", "y"])
    return lgb.LGBMRegressor(**DEFAULT_PARAMS).fit(f[FEATURES], f["y"] / f["base"])


def daily_errors(model, series: pd.Series, start: str, end: str | None = None) -> pd.Series:
    """Daily MAPE of a frozen model on `series` (which may be altered) over [start, end]."""
    f = build_features(series).loc[start:end].dropna(subset=FEATURES + ["base", "y"])
    pred = pd.Series(model.predict(f[FEATURES]) * f["base"].to_numpy(), index=f.index)
    return daily_mape(f["y"], pred)


def scenarios(s: pd.Series) -> dict[str, pd.Series]:
    after = s.index >= T
    level = s.copy(); level[after] *= 1.10
    shape = s.copy(); shape[after & s.index.hour.isin([18, 19, 20, 21])] *= 1.12
    unit = s.copy(); unit[after] /= 1000
    return {"clean": s, "level_shift": level, "evening_shape": shape, "unit_bug": unit}


def episodes(a: pd.Series) -> int:
    d = a[a].index.to_series()
    return int(d.diff().dt.days.fillna(99).gt(5).sum())


def main() -> None:
    s = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    hol = holiday_dates(2019, 2027)
    variants = {"plain": None, "calendar": hol}

    # ---- calibration on clean 2021-2022 (model A trained up to 2020-12-31)
    model_a = train_frozen(s, "2020-12-31")
    cal = slice(*CAL)
    w = s.loc["2020-12-01":CAL[1]]
    thr = {"performance": calibrate_threshold(
        performance_signal(daily_errors(model_a, w, "2020-12-01", CAL[1])).loc[cal], q=0.99, two_sided=False)}
    for v, ign in variants.items():
        thr[f"level_{v}"] = calibrate_threshold(level_signal(w, ignore_days=ign).loc[cal], q=0.995)
        thr[f"shape_{v}"] = calibrate_threshold(shape_signal(w, ignore_days=ign).loc[cal], q=0.995)
    print("Thresholds (calibrated on 2021-2022): " + ", ".join(f"{k}={x:.4f}" for k, x in thr.items()))

    # ---- frozen deployment from 2023-01-01 (model B trained up to 2022-12-31)
    model_b = train_frozen(s, "2022-12-31")
    rows, payoff = [], []
    for name, alt in scenarios(s).items():
        a0 = alt.loc["2022-12-01":]
        err = daily_errors(model_b, alt, "2022-12-01")
        sig = {"performance": performance_signal(err).loc[DEPLOY:]}
        for v, ign in variants.items():
            sig[f"level_{v}"] = level_signal(a0, ignore_days=ign).loc[DEPLOY:]
            sig[f"shape_{v}"] = shape_signal(a0, ignore_days=ign).loc[DEPLOY:]
        al = {k: alarms(x, thr[k], two_sided=(k != "performance")) for k, x in sig.items()}
        for k, a in al.items():
            fa = first_alarm(a, T)
            rows.append({"scenario": name, "monitor": k,
                         "false_alarm_days_before_T": int(a[a.index < T].sum()),
                         "false_episodes_before_T": episodes(a[a.index < T]),
                         "alarm_days_all_period": int(a.sum()),
                         "delay_days": None if fa is None or (fa - T).days > 60 else (fa - T).days})
        # would retraining at the first performance alarm after T have helped? (next 28 days, out of sample)
        pa = al["performance"]
        d0 = first_alarm(pa, T) if name != "clean" else first_alarm(pa, DEPLOY)
        for label, d in (("first alarm after T" if name != "clean" else "first alarm in 2023-26", d0),):
            if d is not None:
                chal = train_frozen(alt, str((d - pd.Timedelta(days=1)).date()))
                nxt = (d + pd.Timedelta(days=1), d + pd.Timedelta(days=28))
                payoff.append({"scenario": name, "alarm_date": d.date(),
                               "frozen_mape": daily_errors(model_b, alt, str(nxt[0].date()), str(nxt[1].date())).mean(),
                               "retrained_mape": daily_errors(chal, alt, str(nxt[0].date()), str(nxt[1].date())).mean()})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nDetection delay in days after the injection on 2024-06-03 "
          "(None = no alarm within 60 days):\n")
    print(res.pivot(index="monitor", columns="scenario", values="delay_days").to_string())
    print("\nClean series: false alarm days / episodes in the 520 days before 2024-06-03, "
          "and alarm days over the whole run (1,400 days):")
    c = res[res.scenario == "clean"].set_index("monitor")[["false_alarm_days_before_T", "false_episodes_before_T", "alarm_days_all_period"]]
    print(c.to_string())
    pf = pd.DataFrame(payoff)
    pf["gain_points"] = pf["frozen_mape"] - pf["retrained_mape"]
    print("\nDoes retraining at the performance alarm pay off? (MAPE over the following 28 days)\n")
    print(pf.round(3).to_string(index=False))
    res.to_csv("data/drift_simulation.csv", index=False)
    pf.to_csv("data/drift_retrain_payoff.csv", index=False)


if __name__ == "__main__":
    main()
