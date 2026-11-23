"""Helpers to build and score prediction intervals (used by scripts/intervals.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_DAYS = 365
MIN_GROUP_HOURS = 200


def residual_quantiles(
    feats: pd.DataFrame, point: pd.Series, quantiles: tuple, eval_start: str, groups: pd.Series | None = None
) -> pd.DataFrame:
    """Quantile forecasts = point * (1 + quantiles of the point forecast's recent relative errors).

    For each day, only errors from hours strictly before that day (the previous HISTORY_DAYS) are
    used, split into the same group as the day itself (`groups`, one label per hour; by default
    holiday-affected vs. ordinary); with fewer than MIN_GROUP_HOURS errors in the group, all
    recent errors are used instead.
    """
    idx = point.dropna().index
    pt = point.reindex(idx).to_numpy()
    rel = (feats["y"].reindex(idx).to_numpy() - pt) / pt
    if groups is None:
        groups = (feats["day_type"] == 4) | (feats["ref168_holiday"] == 1)
    affected = groups.reindex(idx).to_numpy()
    rows, row_idx = [], []
    for day in pd.date_range(eval_start, idx.max().normalize(), freq="D"):
        d0, d1 = idx.searchsorted(day), idx.searchsorted(day + pd.Timedelta(days=1))
        if d1 == d0:
            continue
        lo = idx.searchsorted(day - pd.Timedelta(days=HISTORY_DAYS))
        hist, hist_g = rel[lo:d0], affected[lo:d0]
        sel = hist[hist_g == affected[d0]]
        if len(sel) < MIN_GROUP_HOURS:
            sel = hist
        qs = np.quantile(sel[~np.isnan(sel)], quantiles)
        rows.append(pt[d0:d1, None] * (1 + qs[None, :]))
        row_idx.append(idx[d0:d1])
    return pd.DataFrame(np.vstack(rows), index=row_idx[0].append(row_idx[1:]), columns=list(quantiles))


def pinball(y: pd.Series, q_pred: pd.DataFrame) -> float:
    """Mean pinball loss over the quantiles, as a percentage of the mean demand."""
    losses = []
    for q in q_pred.columns:
        e = y - q_pred[q]
        losses.append(np.maximum(q * e, (q - 1) * e).mean())
    return float(np.mean(losses) / y.mean() * 100)


def cover(y: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    """1.0 where lo <= y <= hi, else 0.0."""
    return ((y >= lo) & (y <= hi)).astype(float)


def cqr_bands(
    qr: pd.DataFrame, y: pd.Series, affected: pd.Series, levels: dict, eval_start: str
) -> pd.DataFrame:
    """Conformalized quantile regression: widen each (low, high) pair by a recent out-of-sample margin.

    For one level, the score of an hour is max(low - y, y - high): how far the real value fell
    outside the band (negative if inside). For each day the margin is the
    (1 - alpha)(1 + 1/n) quantile of the scores of the previous HISTORY_DAYS (only hours before
    that day, same holiday-affected / ordinary group when there are enough of them) and the band
    becomes [low - margin, high + margin]. `qr` must itself be out-of-sample (walk-forward).
    """
    idx = qr.dropna().index.intersection(y.dropna().index)
    out = pd.DataFrame(index=idx, columns=qr.columns, dtype=float)
    aff = affected.reindex(idx).to_numpy()
    for ql, qh in levels.values():
        lo_v, hi_v, yv = qr.loc[idx, ql].to_numpy(), qr.loc[idx, qh].to_numpy(), y.reindex(idx).to_numpy()
        score = np.maximum(lo_v - yv, yv - hi_v)
        alpha = 1 - (qh - ql)
        lo_adj, hi_adj = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
        for day in pd.date_range(eval_start, idx.max().normalize(), freq="D"):
            d0, d1 = idx.searchsorted(day), idx.searchsorted(day + pd.Timedelta(days=1))
            if d1 == d0:
                continue
            lo = idx.searchsorted(day - pd.Timedelta(days=HISTORY_DAYS))
            hist, hist_g = score[lo:d0], aff[lo:d0]
            sel = hist[hist_g == aff[d0]]
            if len(sel) < MIN_GROUP_HOURS:
                sel = hist
            n = len(sel)
            margin = np.quantile(sel, min(1.0, (1 - alpha) * (1 + 1 / n)))
            lo_adj[d0:d1], hi_adj[d0:d1] = lo_v[d0:d1] - margin, hi_v[d0:d1] + margin
        out[ql], out[qh] = lo_adj, hi_adj
    return out.dropna()


def interval_spec(
    feats: pd.DataFrame, point: pd.Series, quantiles: tuple, history_days: int = HISTORY_DAYS
) -> dict:
    """Relative-error quantiles per error group, from the last `history_days` of walk-forward forecasts.

    Group = day_type * 2 + ref168_holiday (see serving.group_code). Groups with fewer than
    MIN_GROUP_HOURS errors are left out and the serving code falls back to the `fallback` entry
    (all errors). `point` must be out-of-sample (walk-forward), never in-sample fits.
    """
    p = point.dropna()
    p = p[p.index >= p.index.max() - pd.Timedelta(days=history_days)]
    rel = ((feats["y"].reindex(p.index) - p) / p).dropna()
    code = (feats["day_type"] * 2 + feats["ref168_holiday"]).reindex(rel.index).astype(int)
    groups = {
        str(c): {"n": int(len(g)), "q": [float(v) for v in np.quantile(g, quantiles)]}
        for c, g in rel.groupby(code)
        if len(g) >= MIN_GROUP_HOURS
    }
    return {
        "method": "residuals_by_day_type",
        "history_days": history_days,
        "history_end": str(p.index.max()),
        "quantiles": list(quantiles),
        "groups": groups,
        "fallback": {"n": int(len(rel)), "q": [float(v) for v in np.quantile(rel, quantiles)]},
    }
