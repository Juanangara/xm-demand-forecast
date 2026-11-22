"""Evaluate seasonal-naive baselines with walk-forward folds and log them to MLflow.

Usage: python scripts/baselines.py [--test-start 2023-01-01]
Needs data/DemaReal_Sistema.parquet (run scripts/download.py first).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from xmforecast.evaluate import seasonal_naive, summary, walk_forward  # noqa: E402

BASELINES = {"naive_24h": 24, "naive_168h": 168}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2023-01-01")
    a = p.parse_args()

    df = pd.read_parquet("data/DemaReal_Sistema.parquet")
    series = df.set_index("ts")["value"].sort_index()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("day-ahead-demand")
    for name, lag in BASELINES.items():
        folds = walk_forward(series, seasonal_naive(series, lag), a.test_start)
        s = summary(folds)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "lag_hours": lag, "horizon_h": 24,
                               "test_start": a.test_start, "n_folds": len(folds)})
            mlflow.log_metrics(s)
            for _, r in folds.iterrows():  # one point per fold: step = fold index
                mlflow.log_metric("fold_mape", r["mape"], step=int(r.name))
            folds.to_csv("data/folds_tmp.csv", index=False)
            mlflow.log_artifact("data/folds_tmp.csv", artifact_path="folds")
        print(f"{name}: MAE={s['mae']/1e3:,.0f} MWh  MAPE={s['mape']:.2f}%  "
              f"worst fold={s['mape_worst_fold']:.2f}%  folds={len(folds)}")


if __name__ == "__main__":
    main()
