"""Train the final model on all history and export it for the API.

Writes models/champion/model.txt and meta.json (self-contained: the API does not need
MLflow), and logs the same run to MLflow and its model registry.
Usage: python scripts/train_final.py [--window-years 5]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lightgbm as lgb  # noqa: E402
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import pandas as pd  # noqa: E402

from xmforecast.audit import mask_disturbances  # noqa: E402
from xmforecast.features import FEATURES, build_features  # noqa: E402
from xmforecast.forecast import DEFAULT_PARAMS  # noqa: E402

# Walk-forward test results from docs/DEVLOG.md (Phase 6-7), stored with the model for reference.
REFERENCE_METRICS = {"walk_forward_test_mape": 1.64, "naive_168h_test_mape": 3.37, "mstl_test_mape": 2.25}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window-years", type=int, default=5)
    p.add_argument("--out", default="models/champion")
    a = p.parse_args()

    series = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    train = build_features(mask_disturbances(series))
    train = train[train.index >= train.index.max() - pd.DateOffset(years=a.window_years)]
    train = train.dropna(subset=FEATURES + ["base", "y"])

    model = lgb.LGBMRegressor(**DEFAULT_PARAMS).fit(train[FEATURES], train["y"] / train["base"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))
    meta = {
        "name": "xm-demand-day-ahead",
        "target": "demand_kwh_per_hour",
        "horizon_h": 24,
        "trained_until": str(series.index.max()),
        "train_rows": int(len(train)),
        "window_years": a.window_years,
        "features": FEATURES,
        "lightgbm": lgb.__version__,
        "code_commit": git_commit(),
        "reference_metrics": REFERENCE_METRICS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("day-ahead-demand")
    with mlflow.start_run(run_name="final_model") as run:
        mlflow.log_params({k: v for k, v in meta.items() if k not in ("reference_metrics", "features")})
        mlflow.log_params(DEFAULT_PARAMS)
        mlflow.log_metrics(REFERENCE_METRICS)
        mlflow.log_artifacts(str(out), artifact_path="champion")
        info = mlflow.lightgbm.log_model(
            model.booster_, name="model", registered_model_name="xm-demand-day-ahead"
        )
    print(f"saved {out}/ (trained until {meta['trained_until']}, {len(train)} rows); "
          f"registered {info.registered_model_version and 'xm-demand-day-ahead v' + str(info.registered_model_version)}")


if __name__ == "__main__":
    main()
