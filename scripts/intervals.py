"""Prediction intervals: LightGBM quantile regression vs. intervals from recent model errors.

Both produce 80 % and 95 % central intervals for the day-ahead forecast, walk-forward:
  - quantile regression: one LightGBM per quantile, retrained every month (5-year window)
  - residual baseline: the point forecast of Phase 6 times (1 + empirical quantiles of its own
    relative errors over the previous 365 days), computed separately for holiday-affected and
    ordinary days. It needs a history of errors, so everything is scored from 2023-04-01.

The question is calibration: does an 80 % interval contain the real value about 80 % of the
time, also on holidays? Coverage intervals are block-bootstrapped over days.
Usage: python scripts/intervals.py [--reuse]   (--reuse loads data/pred_quantiles_w5.parquet)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xmforecast.audit import mask_disturbances  # noqa: E402
from xmforecast.features import build_features  # noqa: E402
from xmforecast.forecast import QUANTILES, walk_forward_quantiles  # noqa: E402
from xmforecast.intervals import cover, cqr_bands, pinball, residual_quantiles  # noqa: E402
from xmforecast.stats import block_bootstrap_ci  # noqa: E402

TEST_START, EVAL_START = "2023-01-01", "2023-04-01"
LEVELS = {"80%": (0.1, 0.9), "95%": (0.025, 0.975)}
DAY_NAMES = {0: "weekday", 1: "near holiday", 2: "Saturday", 3: "Sunday", 4: "holiday"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reuse", action="store_true")
    a = p.parse_args()

    series = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    feats = build_features(series)
    cache = Path("data/pred_quantiles_w5.parquet")
    if a.reuse and cache.exists():
        qr = pd.read_parquet(cache)
        qr.columns = [float(c) for c in qr.columns]
    else:
        train_feats = build_features(mask_disturbances(series))
        qr = walk_forward_quantiles(feats, train_feats, TEST_START)
        qr.columns = [float(c) for c in qr.columns]
        qr.astype(float).rename(columns=str).to_parquet(cache)
    point = pd.read_parquet("data/pred_lgbm_window_5.parquet")["pred"]
    res = residual_quantiles(feats, point, QUANTILES, EVAL_START)
    res_dt = residual_quantiles(feats, point, QUANTILES, EVAL_START,
                                groups=feats["day_type"] * 2 + feats["ref168_holiday"])
    all_affected = (feats["day_type"] == 4) | (feats["ref168_holiday"] == 1)
    cq = cqr_bands(qr, feats["y"], all_affected, LEVELS, EVAL_START)

    common = qr.dropna().index.intersection(res.dropna().index).intersection(point.dropna().index)
    common = common.intersection(cq.index).intersection(res_dt.dropna().index)
    common = common[common >= pd.Timestamp(EVAL_START)]
    y, pt = feats["y"].reindex(common), point.reindex(common)
    methods = {"quantile regression": qr.reindex(common), "recent residuals": res.reindex(common),
               "residuals by day type": res_dt.reindex(common), "conformal QR": cq.reindex(common)}
    f = feats.reindex(common)
    affected = ((f["day_type"] == 4) | (f["ref168_holiday"] == 1))
    day = common.normalize()
    print(f"Scored on {len(common)} hours ({len(common) // 24} days), {common.min().date()} to {common.max().date()}\n")

    rows = []
    for name, bands in methods.items():
        for level, (ql, qh) in LEVELS.items():
            lo, hi = bands[ql], bands[qh]
            c = cover(y, lo, hi)
            daily = c.groupby(day).mean()
            ci = block_bootstrap_ci(daily)
            row = {"method": name, "interval": level, "nominal": float(level[:-1]) / 100,
                   "coverage": c.mean(), "ci_lo": ci[0], "ci_hi": ci[1],
                   "ordinary": c[~affected.to_numpy()].mean(), "holiday_affected": c[affected.to_numpy()].mean(),
                   "width_pct": ((hi - lo) / pt).mean() * 100}
            for yr in sorted(set(common.year)):
                row[str(yr)] = c[common.year == yr].mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    pb = {n: pinball(y, b) for n, b in methods.items()}
    print("\nMean pinball loss, % of mean demand (lower is better):", {k: round(v, 3) for k, v in pb.items()})

    print("\nCoverage of the 80 % interval by day type:")
    dt = f["day_type"].map(DAY_NAMES)
    by_type = pd.DataFrame({n: cover(y, b[0.1], b[0.9]).groupby(dt.to_numpy()).mean() for n, b in methods.items()})
    print(by_type.round(3).to_string())
    hr = pd.DataFrame({n: cover(y, b[0.1], b[0.9]).groupby(common.hour).mean() for n, b in methods.items()})
    print("\nCoverage of the 80 % interval by hour of day: min / max")
    print(hr.agg(["min", "max"]).round(3).to_string())

    table.to_csv("data/intervals_summary.csv", index=False)
    by_type.to_csv("data/intervals_by_day_type.csv")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("day-ahead-demand-intervals")
    client = mlflow.MlflowClient()
    for old in client.search_runs([client.get_experiment_by_name("day-ahead-demand-intervals").experiment_id]):
        client.delete_run(old.info.run_id)  # rerunning replaces the previous results
    for r in rows:
        with mlflow.start_run(run_name=f"{r['method']} {r['interval']}".replace(" ", "_")):
            mlflow.log_params({"method": r["method"], "interval": r["interval"], "eval_start": EVAL_START})
            mlflow.log_metrics({k: float(v) for k, v in r.items() if isinstance(v, (float, np.floating))})
    print("\nsaved data/intervals_summary.csv and logged to MLflow")


if __name__ == "__main__":
    main()
