"""Significance tests for comparing two forecasting models.

Both work on *daily* losses (one number per forecast day): the 24 hours of a day are
strongly correlated, and so are consecutive days, so treating hours as independent
would make every difference look significant.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def daily_mape(actual: pd.Series, pred: pd.Series) -> pd.Series:
    """Mean absolute percentage error per calendar day (NaN hours dropped)."""
    ape = ((pred - actual).abs() / actual.abs() * 100).dropna()
    return ape.groupby(ape.index.normalize()).mean()


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lags: int = 7) -> dict[str, float]:
    """Diebold-Mariano test on d_t = loss_a - loss_b with a Newey-West variance.

    Negative `mean_diff` means model A has the lower loss. H0: equal expected loss.
    `lags` is the autocorrelation window in days.
    """
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    dbar = d.mean()
    c = d - dbar
    lrv = float((c @ c) / n)
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * float(c[k:] @ c[:-k]) / n
    stat = dbar / math.sqrt(lrv / n) if lrv > 0 else 0.0
    p = math.erfc(abs(stat) / math.sqrt(2))  # two-sided normal
    return {"n_days": n, "mean_diff": float(dbar), "dm_stat": stat, "p_value": p}


def block_bootstrap_ci(
    d: pd.Series, block: int = 14, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of a daily series (keeps autocorrelation)."""
    x = d.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
