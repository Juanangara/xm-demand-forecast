"""Download an XM hourly metric and print a data-quality report.

Usage: python scripts/download.py [--metric DemaReal] [--entity Sistema] [--start-year 2000]
Months are cached under data/raw/, so re-running only fetches what is missing.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xmforecast.data import download  # noqa: E402
from xmforecast.quality import quality_by_year, usable_from  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="DemaReal")
    p.add_argument("--entity", default="Sistema")
    p.add_argument("--start-year", type=int, default=2000)
    a = p.parse_args()

    # The current month is incomplete and would be cached as final, so stop at last month.
    today = dt.date.today()
    end = today.replace(day=1) - dt.timedelta(days=1)
    df = download(a.metric, a.entity, a.start_year, end)

    out = Path("data") / f"{a.metric}_{a.entity}.parquet"
    df.to_parquet(out, index=False)

    report = quality_by_year(df)
    print(report.to_string(index=False))
    print(f"\nUsable from (<=0.5% missing every year after): {usable_from(report)}")
    report.to_csv(Path("data") / f"quality_{a.metric}_{a.entity}.csv", index=False)


if __name__ == "__main__":
    main()
