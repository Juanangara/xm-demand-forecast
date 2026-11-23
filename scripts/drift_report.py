"""Evidently data-drift report for the 'evening_shape' scenario (4 weeks before vs after).

One row per day, one column per hour with that hour's share of the day's demand, plus the
daily mean. Writes data/drift_report.html (open it in a browser) and prints the flagged columns.
Usage: python scripts/drift_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

from simulate_drift import T, scenarios  # noqa: E402


def daily_profile(series: pd.Series) -> pd.DataFrame:
    day = series.index.normalize()
    share = (series / series.groupby(day).transform("sum")).groupby([day, series.index.hour]).mean().unstack()
    share.columns = [f"h{h:02d}" for h in share.columns]
    share["daily_mean_kwh"] = series.groupby(day).mean()
    return share.dropna().reset_index(drop=True)


def main() -> None:
    s = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    alt = scenarios(s)["evening_shape"]
    ref = daily_profile(alt[T - pd.Timedelta(days=28): T - pd.Timedelta(hours=1)])
    cur = daily_profile(alt[T: T + pd.Timedelta(days=28) - pd.Timedelta(hours=1)])
    dd = DataDefinition(numerical_columns=list(ref.columns))
    snap = Report([DataDriftPreset()]).run(Dataset.from_pandas(cur, data_definition=dd),
                                           Dataset.from_pandas(ref, data_definition=dd))
    snap.save_html("data/drift_report.html")
    metrics = snap.dict()["metrics"]
    pvals = {m["metric_name"].split("column=")[1].split(",")[0]: m["value"]
             for m in metrics if str(m["metric_name"]).startswith("ValueDrift")}
    n_flag = sum(v < 0.05 for v in pvals.values())
    print(f"rows: reference={len(ref)} current={len(cur)}; columns with p < 0.05: {n_flag} of {len(pvals)}")
    print("smallest p-values (where the change is):",
          {k: float(f"{v:.1e}") for k, v in sorted(pvals.items(), key=lambda kv: kv[1])[:5]})
    print("daily_mean_kwh p-value:", round(pvals["daily_mean_kwh"], 2))
    print("report: data/drift_report.html")


if __name__ == "__main__":
    main()
