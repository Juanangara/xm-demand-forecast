"""Walk-forward LightGBM for day-ahead demand, logged to MLflow.

Usage: python scripts/train_lgbm.py [--test-start 2023-01-01] [--windows all,5,3]
Compares training windows (all history vs. the last N years). Scores overall and
separately on ordinary days vs. hours touched by a holiday (target or 168 h reference).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from xmforecast.audit import mask_disturbances  # noqa: E402
from xmforecast.evaluate import metrics, seasonal_naive, summary, walk_forward  # noqa: E402
from xmforecast.features import FEATURES, build_features  # noqa: E402
from xmforecast.forecast import DEFAULT_PARAMS, walk_forward_lgbm  # noqa: E402


def segment_metrics(feats: pd.DataFrame, pred: pd.Series, test_start: str) -> dict[str, float]:
    """MAPE on ordinary hours vs. holiday-affected hours (day_type==holiday or ref168 holiday)."""
    df = feats.loc[test_start:].assign(pred=pred).dropna(subset=["pred", "y"])
    affected = (df["day_type"] == 4) | (df["ref168_holiday"] == 1)
    out = {}
    for name, part in (("ordinary", df[~affected]), ("holiday_affected", df[affected])):
        out[f"mape_{name}"] = metrics(part["y"], part["pred"])["mape"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--windows", default="all,5,3")
    a = p.parse_args()

    series = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    feats = build_features(series)
    train_feats = build_features(mask_disturbances(series))  # reviewed disturbances masked, fit only

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("day-ahead-demand")

    base_pred = seasonal_naive(series, 168)
    base_seg = segment_metrics(feats, base_pred, a.test_start)
    print(f"naive_168h reference: MAPE ordinary={base_seg['mape_ordinary']:.2f}%  "
          f"holiday-affected={base_seg['mape_holiday_affected']:.2f}%")

    for w in a.windows.split(","):
        years = None if w == "all" else int(w)
        t0 = time.time()
        pred, imps = walk_forward_lgbm(feats, train_feats, a.test_start, train_years=years)
        folds = walk_forward(series, pred, a.test_start)
        s = {**summary(folds), **segment_metrics(feats, pred, a.test_start)}
        name = f"lgbm_window_{w}"
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": "lightgbm", "train_window_years": w, "horizon_h": 24,
                               "target": "y/lag_168", "test_start": a.test_start,
                               "n_folds": len(folds), "features": ",".join(FEATURES), **DEFAULT_PARAMS})
            mlflow.log_metrics(s)
            for _, r in folds.iterrows():
                mlflow.log_metric("fold_mape", r["mape"], step=int(r.name))
            folds.to_csv("data/folds_tmp.csv", index=False)
            imps.mean().sort_values(ascending=False).to_csv("data/importance_tmp.csv", header=["mean_importance"])
            mlflow.log_artifact("data/folds_tmp.csv", artifact_path="folds")
            mlflow.log_artifact("data/importance_tmp.csv", artifact_path="importance")
        pred.rename("pred").to_frame().to_parquet(f"data/pred_{name}.parquet")
        print(f"{name}: MAPE={s['mape']:.2f}%  worst fold={s['mape_worst_fold']:.2f}%  "
              f"ordinary={s['mape_ordinary']:.2f}%  holiday-affected={s['mape_holiday_affected']:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
