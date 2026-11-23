import numpy as np
import pandas as pd

from xmforecast.features import build_features
from xmforecast.forecast import walk_forward_quantiles
from xmforecast.intervals import cover, pinball, residual_quantiles


def test_cover_counts_values_inside_the_closed_interval():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    lo, hi = pd.Series([1.0] * 4), pd.Series([3.0] * 4)
    assert list(cover(y, lo, hi)) == [1.0, 1.0, 1.0, 0.0]


def test_pinball_known_value():
    y = pd.Series([100.0, 100.0])
    q = pd.DataFrame({0.1: [110.0, 110.0], 0.9: [90.0, 90.0]})
    # q=0.9 predicting 90: e=10, loss 9; q=0.1 predicting 110: e=-10, loss 9 -> mean 9 -> 9 % of 100
    assert abs(pinball(y, q) - 9.0) < 1e-9


def _feats_and_point(days=120, seed=0):
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(seed)
    y = pd.Series(1000 + 100 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 20, len(idx)), index=idx)
    feats = pd.DataFrame({"y": y, "day_type": 0, "ref168_holiday": 0}, index=idx)
    point = y * (1 + rng.normal(0, 0.03, len(idx)))  # a forecast with ~3 % noise
    return feats, point


def test_residual_quantiles_are_ordered_and_wider_for_outer_quantiles():
    feats, point = _feats_and_point()
    q = residual_quantiles(feats, point, (0.025, 0.1, 0.9, 0.975), "2024-03-01")
    assert (q[0.025] <= q[0.1]).all() and (q[0.1] <= q[0.9]).all() and (q[0.9] <= q[0.975]).all()
    assert q.index.min() == pd.Timestamp("2024-03-01") and len(q) > 24 * 50


def test_residual_quantiles_for_a_day_do_not_depend_on_that_day_or_later():
    feats, point = _feats_and_point()
    day = pd.Timestamp("2024-03-10")
    a = residual_quantiles(feats, point, (0.1, 0.9), "2024-03-01")
    changed = feats.copy()
    changed.loc[changed.index >= day, "y"] *= 5  # rewrite the actuals from that day on
    b = residual_quantiles(changed, point, (0.1, 0.9), "2024-03-01")
    day_rows = (a.index >= day) & (a.index < day + pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(a[day_rows], b[day_rows])


def test_walk_forward_quantiles_sorted_and_blind_to_data_after_the_month():
    idx = pd.date_range("2023-01-01", periods=24 * 130, freq="h")
    rng = np.random.default_rng(1)
    s = pd.Series(1000 + 100 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 15, len(idx)), index=idx)
    params = {"n_estimators": 20}
    out = walk_forward_quantiles(build_features(s), build_features(s), "2023-04-01",
                                 (0.1, 0.5, 0.9), train_years=None, params=params, test_end="2023-04-30")
    assert out.dropna().shape[1] == 3
    v = out.dropna().to_numpy()
    assert (np.diff(v, axis=1) >= 0).all()  # a lower quantile never exceeds a higher one
    s2 = s.copy()
    s2[s2.index >= "2023-05-01"] *= 10  # later data must not matter for April
    out2 = walk_forward_quantiles(build_features(s2), build_features(s2), "2023-04-01",
                                  (0.1, 0.5, 0.9), train_years=None, params=params, test_end="2023-04-30")
    pd.testing.assert_frame_equal(out, out2)


def _toy_qr(days=150, narrow=0.5, seed=2):
    """Quantile forecasts that are too narrow by a factor: the truth is N(1000, 30) noise."""
    from statistics import NormalDist

    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(seed)
    y = pd.Series(1000 + rng.normal(0, 30, len(idx)), index=idx)
    nd = NormalDist()
    qr = pd.DataFrame({q: 1000 + nd.inv_cdf(q) * 30 * narrow for q in (0.1, 0.9)}, index=idx)
    return y, qr


def test_cqr_repairs_bands_that_are_too_narrow():
    from xmforecast.intervals import cqr_bands

    y, qr = _toy_qr()
    levels = {"80%": (0.1, 0.9)}
    raw = cover(y, qr[0.1], qr[0.9])["2024-03-01":].mean()
    fixed = cqr_bands(qr, y, pd.Series(False, index=y.index), levels, "2024-03-01")
    cov = cover(y.reindex(fixed.index), fixed[0.1], fixed[0.9]).mean()
    assert raw < 0.5 and abs(cov - 0.8) < 0.03  # from about 38 % back to about 80 %
    assert (fixed[0.1] < fixed[0.9]).all()


def test_cqr_margin_for_a_day_does_not_depend_on_that_day_or_later():
    from xmforecast.intervals import cqr_bands

    y, qr = _toy_qr()
    levels = {"80%": (0.1, 0.9)}
    aff = pd.Series(False, index=y.index)
    day = pd.Timestamp("2024-03-10")
    a = cqr_bands(qr, y, aff, levels, "2024-03-01")
    y2 = y.copy()
    y2[y2.index >= day] += 500  # rewrite the actuals from that day on
    b = cqr_bands(qr, y2, aff, levels, "2024-03-01")
    rows = (a.index >= day) & (a.index < day + pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(a[rows], b[rows])


def test_interval_spec_groups_and_fallback():
    from xmforecast.intervals import MIN_GROUP_HOURS, interval_spec

    feats, point = _feats_and_point(days=60)
    feats["day_type"] = np.where(feats.index.dayofweek >= 5, 2, 0)  # weekdays vs Saturdays/Sundays as type 2
    spec = interval_spec(feats, point, (0.025, 0.1, 0.9, 0.975), history_days=40)
    assert set(spec["groups"]) <= {"0", "4"}
    assert spec["fallback"]["n"] >= MIN_GROUP_HOURS
    for g in list(spec["groups"].values()) + [spec["fallback"]]:
        assert g["q"] == sorted(g["q"])  # quantiles are ordered
    assert spec["history_end"] == str(point.index.max())
