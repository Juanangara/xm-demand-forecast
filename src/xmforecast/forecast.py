"""Walk-forward LightGBM: retrain at the start of every test month, predict that month.

The model predicts the ratio y / base (base = same hour last week), so it does not
need to extrapolate the 2x growth in demand since 2000. Training only uses rows whose
hour is strictly before the fold start.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES

DEFAULT_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1,
)


def walk_forward_lgbm(
    feats: pd.DataFrame,
    train_feats: pd.DataFrame,
    test_start: str,
    train_years: int | None = None,
    params: dict | None = None,
    test_end: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (predictions for every test hour, feature importance by fold).

    `feats` carries the real series (used to score and as base); `train_feats` is the
    same table built from the series with reviewed disturbances masked, used to fit only.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    preds, imps = [], []
    end_ts = pd.Timestamp(test_end) if test_end else feats.index.max()
    folds = pd.period_range(test_start, end_ts, freq="M")
    for month in folds:
        start, end = month.start_time, month.end_time
        train = train_feats[train_feats.index < start]
        if train_years:
            train = train[train.index >= start - pd.DateOffset(years=train_years)]
        train = train.dropna(subset=FEATURES + ["base", "y"])
        target = train["y"] / train["base"]

        model = lgb.LGBMRegressor(**p).fit(train[FEATURES], target)

        test = feats[(feats.index >= start) & (feats.index <= end)]
        ok = test.dropna(subset=FEATURES + ["base"])
        preds.append(pd.Series(model.predict(ok[FEATURES]) * ok["base"], index=ok.index))
        imps.append(pd.Series(model.feature_importances_, index=FEATURES, name=str(month)))
    pred = pd.concat(preds).reindex(feats.loc[test_start:end_ts].index)
    return pred, pd.DataFrame(imps)


QUANTILES = (0.025, 0.1, 0.9, 0.975)  # central 80 % and 95 % intervals


def walk_forward_quantiles(
    feats: pd.DataFrame,
    train_feats: pd.DataFrame,
    test_start: str,
    quantiles: tuple[float, ...] = QUANTILES,
    train_years: int | None = 5,
    params: dict | None = None,
    test_end: str | None = None,
) -> pd.DataFrame:
    """Walk-forward quantile regression of the ratio y / base, one model per quantile per month.

    Same protocol as `walk_forward_lgbm`: each test month is predicted by models fitted only on
    hours strictly before it. Returns one column per quantile, in demand units (kWh); each row is
    sorted so that a lower quantile never exceeds a higher one (LightGBM trains the quantiles
    independently, so they can cross).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    end_ts = pd.Timestamp(test_end) if test_end else feats.index.max()
    out = []
    for month in pd.period_range(test_start, end_ts, freq="M"):
        start, end = month.start_time, month.end_time
        train = train_feats[train_feats.index < start]
        if train_years:
            train = train[train.index >= start - pd.DateOffset(years=train_years)]
        train = train.dropna(subset=FEATURES + ["base", "y"])
        target = train["y"] / train["base"]

        test = feats[(feats.index >= start) & (feats.index <= end)].dropna(subset=FEATURES + ["base"])
        cols = {}
        for q in quantiles:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **p).fit(train[FEATURES], target)
            cols[q] = model.predict(test[FEATURES]) * test["base"].to_numpy()
        vals = np.sort(np.column_stack([cols[q] for q in quantiles]), axis=1)
        out.append(pd.DataFrame(vals, index=test.index, columns=list(quantiles)))
    return pd.concat(out).reindex(feats.loc[test_start:end_ts].index)
