"""Break the naive-168h baseline error down by day type and by hour (no model, no MLflow)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from xmforecast.calendar import day_type, holiday_dates  # noqa: E402
from xmforecast.evaluate import seasonal_naive  # noqa: E402

TEST_START = "2023-01-01"


def main() -> None:
    s = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    pred = seasonal_naive(s, 168)
    df = pd.DataFrame({"actual": s, "pred": pred}).loc[TEST_START:].dropna()
    hol = holiday_dates(2022, 2026)
    df["type"] = day_type(df.index, hol)
    # the day the baseline copies from (7 days earlier) was a holiday
    ref = (df.index - pd.Timedelta(hours=168)).normalize()
    df["ref_holiday"] = ref.isin(hol)
    df["ape"] = (df["pred"] - df["actual"]).abs() / df["actual"] * 100
    df["abs_err"] = (df["pred"] - df["actual"]).abs()

    total = df["abs_err"].sum()
    g = df.groupby("type").agg(hours=("ape", "size"), mape=("ape", "mean"), err=("abs_err", "sum"))
    g["share_hours_%"] = (100 * g["hours"] / len(df)).round(1)
    g["share_error_%"] = (100 * g["err"] / total).round(1)
    print("By day type of the TARGET day:")
    print(g.drop(columns="err").round({"mape": 2}).sort_values("mape", ascending=False).to_string())

    ref_only = df[(df["ref_holiday"]) & (df["type"] != "holiday")]
    print(f"\nTarget is not a holiday but last week's copy was: {len(ref_only)} h, "
          f"MAPE {ref_only['ape'].mean():.2f}%, {100*ref_only['abs_err'].sum()/total:.1f}% of total error")
    clean = df[(df["type"].isin(["weekday", "sat", "sun"])) & (~df["ref_holiday"])]
    print(f"Ordinary days with an ordinary reference: {len(clean)} h, MAPE {clean['ape'].mean():.2f}%")
    flagged = df[(df["type"] == "holiday") | df["ref_holiday"]]
    print(f"All hours affected by a holiday (target or reference): {len(flagged)} h "
          f"({100*len(flagged)/len(df):.1f}% of hours), {100*flagged['abs_err'].sum()/total:.1f}% of total error")

    print("\nMAPE by hour of day (ordinary days only):")
    print(clean.groupby(clean.index.hour)["ape"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
