import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from xmforecast.api import create_app
from xmforecast.features import FEATURES, build_features
from xmforecast.serving import ForecastError, forecast_day, load_champion


def _series(days: int = 80) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(1)
    daily = 1000 + 200 * np.sin(2 * np.pi * (idx.hour - 6) / 24) + 80 * (idx.dayofweek >= 5) * -1
    return pd.Series(daily + rng.normal(0, 5, len(idx)), index=idx)


@pytest.fixture()
def env(tmp_path):
    s = _series()
    f = build_features(s).dropna(subset=FEATURES + ["base", "y"])
    # native LightGBM API on purpose: the serving image (and CI) do not install scikit-learn
    booster = lgb.train({"verbose": -1, "seed": 0}, lgb.Dataset(f[FEATURES], f["y"] / f["base"]), num_boost_round=30)
    mdir = tmp_path / "champion"
    mdir.mkdir()
    booster.save_model(str(mdir / "model.txt"))
    (mdir / "meta.json").write_text(json.dumps({
        "name": "test-model", "code_commit": "abc123", "window_years": 5,
        "trained_until": str(s.index.max()), "features": FEATURES}), encoding="utf-8")
    data = tmp_path / "data.parquet"
    s.rename("value").reset_index().rename(columns={"index": "ts"}).to_parquet(data)
    return s, str(mdir), str(data)


def test_forecast_day_returns_24_hours_and_ignores_the_future(env):
    s, mdir, _ = env
    champ = load_champion(mdir)
    day = pd.Timestamp("2024-02-20")
    a = forecast_day(champ, s, day)
    assert len(a) == 24 and a.index[0] == day and a.index[-1] == day + pd.Timedelta(hours=23)
    tampered = s.copy()
    tampered[tampered.index >= day] = tampered[tampered.index >= day] * 9  # rewrite day D and later
    pd.testing.assert_series_equal(a, forecast_day(champ, tampered, day))  # forecast cannot depend on it


def test_forecast_day_rejects_missing_history_and_too_early_days(env):
    s, mdir, _ = env
    champ = load_champion(mdir)
    with pytest.raises(ForecastError, match="history ends"):
        forecast_day(champ, s, s.index.max().normalize() + pd.Timedelta(days=3))
    with pytest.raises(ForecastError, match="too early"):
        forecast_day(champ, s, pd.Timestamp("2024-01-05"))


def test_api_endpoints(env):
    s, mdir, data = env
    with TestClient(create_app(mdir, data)) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/model").json()["name"] == "test-model"

        r = c.get("/forecast", params={"date": "2024-02-20"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["hours"]) == 24 and body["mape_pct"] is not None
        assert body["hours"][0]["actual_kwh"] is not None
        assert body["in_sample"] is True  # inside the window of this model

        # the day after the history ends: forecastable, but there is no actual to compare with
        nxt = (s.index.max().normalize() + pd.Timedelta(days=1)).date().isoformat()
        r2 = c.get("/forecast", params={"date": nxt})
        assert r2.status_code == 200 and r2.json()["mape_pct"] is None
        assert r2.json()["hours"][0]["actual_kwh"] is None


def test_api_errors(env):
    _, mdir, data = env
    with TestClient(create_app(mdir, data)) as c:
        assert c.get("/forecast", params={"date": "2030-01-01"}).status_code == 422  # beyond history
        assert c.get("/forecast", params={"date": "not-a-date"}).status_code == 422  # invalid input
        assert c.get("/forecast").status_code == 422  # missing parameter


def test_api_flags_holidays(env):
    _, mdir, data = env
    with TestClient(create_app(mdir, data)) as c:
        r = c.get("/forecast", params={"date": "2024-02-19"})  # an ordinary Monday
        assert r.json()["is_holiday"] is False


def test_api_returns_ordered_intervals_when_the_model_has_a_spec(env, tmp_path):
    from pathlib import Path

    _, mdir, data = env
    meta_file = Path(mdir) / "meta.json"
    meta = json.loads(meta_file.read_text("utf-8"))
    q = [-0.05, -0.02, 0.02, 0.05]  # q0.025, q0.1, q0.9, q0.975 as relative errors
    meta["intervals"] = {"method": "residuals_by_day_type", "quantiles": [0.025, 0.1, 0.9, 0.975],
                         "groups": {"0": {"n": 500, "q": q}}, "fallback": {"n": 900, "q": [-0.08, -0.03, 0.03, 0.08]}}
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    with TestClient(create_app(mdir, data)) as c:
        body = c.get("/forecast", params={"date": "2024-02-20"}).json()
        assert body["interval_method"] == "residuals_by_day_type"
        for h in body["hours"]:
            assert h["lower_95_kwh"] < h["lower_80_kwh"] < h["forecast_kwh"] < h["upper_80_kwh"] < h["upper_95_kwh"]
        # an ordinary Tuesday uses group 0: its 80 % band is exactly forecast * (1 - 0.02, 1 + 0.02)
        h0 = body["hours"][0]
        assert abs(h0["lower_80_kwh"] / h0["forecast_kwh"] - 0.98) < 1e-9


def test_api_without_a_spec_has_no_intervals(env):
    _, mdir, data = env
    with TestClient(create_app(mdir, data)) as c:
        body = c.get("/forecast", params={"date": "2024-02-20"}).json()
        assert body["interval_method"] is None
        assert body["hours"][0]["lower_80_kwh"] is None and body["hours"][0]["upper_95_kwh"] is None
