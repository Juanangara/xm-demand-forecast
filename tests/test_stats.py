import numpy as np
import pandas as pd

from xmforecast.stats import block_bootstrap_ci, daily_mape, diebold_mariano


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def _daily(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))


def test_dm_identical_models_have_zero_difference():
    a = _daily(np.random.default_rng(0).normal(3, 1, 300))
    r = diebold_mariano(a, a)
    assert r["mean_diff"] == 0 and r["dm_stat"] == 0 and r["p_value"] == 1.0


def test_dm_detects_a_clear_improvement_even_with_autocorrelated_losses():
    noise = _ar1(400, 0.5, 1)
    a = _daily(3 + noise)
    b = _daily(2 + noise + _ar1(400, 0.5, 2) * 0.3)  # model B is ~1 point better
    r = diebold_mariano(a, b)
    assert r["mean_diff"] > 0.8 and r["p_value"] < 0.001


def test_dm_does_not_reject_when_models_are_equally_good():
    a = _daily(3 + _ar1(400, 0.5, 3))
    b = _daily(3 + _ar1(400, 0.5, 4))
    assert diebold_mariano(a, b)["p_value"] > 0.05


def test_block_bootstrap_is_wider_than_iid_for_autocorrelated_data():
    x = _daily(_ar1(500, 0.8, 5))
    lo_b, hi_b = block_bootstrap_ci(x, block=20)
    lo_i, hi_i = block_bootstrap_ci(x, block=1)  # block of 1 = ordinary bootstrap
    assert (hi_b - lo_b) > 1.5 * (hi_i - lo_i)


def test_block_bootstrap_ci_contains_the_sample_mean():
    x = _daily(np.random.default_rng(6).normal(2, 1, 300))
    lo, hi = block_bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_daily_mape_groups_hours_by_day():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    actual = pd.Series(100.0, index=idx)
    pred = pd.Series([110.0] * 24 + [95.0] * 24, index=idx)
    r = daily_mape(actual, pred)
    assert list(r.round(6)) == [10.0, 5.0]
