"""Choose the LightGBM training window on a validation period that precedes the test.

Validation: 2021-01 to 2022-12. The test period (2023 onwards) is never used here, so the
window picked is not tuned on the numbers we report as final.
Usage: python scripts/select_window.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from xmforecast.audit import mask_disturbances  # noqa: E402
from xmforecast.evaluate import summary, walk_forward  # noqa: E402
from xmforecast.features import build_features  # noqa: E402
from xmforecast.forecast import walk_forward_lgbm  # noqa: E402

VAL_START, VAL_END = "2021-01-01", "2022-12-31"
WINDOWS = ["all", "5", "3"]


def main() -> None:
    series = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    feats = build_features(series)
    train_feats = build_features(mask_disturbances(series))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("day-ahead-demand-window-selection")

    results = {}
    for w in WINDOWS:
        pred, _ = walk_forward_lgbm(feats, train_feats, VAL_START, None if w == "all" else int(w), test_end=VAL_END)
        folds = walk_forward(series.loc[:VAL_END], pred, VAL_START)
        s = summary(folds)
        results[w] = s["mape"]
        with mlflow.start_run(run_name=f"validation_window_{w}"):
            mlflow.log_params({"train_window_years": w, "val_start": VAL_START, "val_end": VAL_END})
            mlflow.log_metrics(s)
        print(f"window={w}: validation MAPE={s['mape']:.3f}%  worst fold={s['mape_worst_fold']:.2f}%", flush=True)
    best = min(results, key=results.get)
    print(f"\nSelected on validation: window={best}")


if __name__ == "__main__":
    main()
