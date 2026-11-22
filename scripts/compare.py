"""Final comparison on the test period: errors, Diebold-Mariano tests and bootstrap CIs.

Needs: data/pred_lgbm_window_{all,5,3}.parquet (scripts/train_lgbm.py) and
data/pred_mstl.parquet (scripts/stat_baseline.py).
All models are scored on exactly the same hours.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from xmforecast.evaluate import seasonal_naive  # noqa: E402
from xmforecast.features import build_features  # noqa: E402
from xmforecast.stats import block_bootstrap_ci, daily_mape, diebold_mariano  # noqa: E402

TEST_START = "2023-01-01"
SELECTED = "lgbm_5"  # chosen on 2021-2022 validation (scripts/select_window.py)


def main() -> None:
    s = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    feats = build_features(s).loc[TEST_START:]
    preds = {"naive_168": seasonal_naive(s, 168).reindex(feats.index)}
    preds["mstl"] = pd.read_parquet("data/pred_mstl.parquet")["pred"].reindex(feats.index)
    for w in ("all", "5", "3"):
        preds[f"lgbm_{w}"] = pd.read_parquet(f"data/pred_lgbm_window_{w}.parquet")["pred"].reindex(feats.index)

    ok = pd.concat(preds, axis=1).notna().all(axis=1)  # same hours for every model
    y, P = feats["y"][ok], {k: v[ok] for k, v in preds.items()}
    print(f"Scored on {ok.sum()} hours ({ok.sum() // 24} days)\n")

    daily = {k: daily_mape(y, p) for k, p in P.items()}
    # a day is holiday-affected if its target is a holiday or last week's copy was one
    h12 = feats[ok][feats[ok].index.hour == 12]
    affected = ((h12["day_type"] == 4) | (h12["ref168_holiday"] == 1)).groupby(h12.index.normalize()).first()

    rows = []
    for k, d in daily.items():
        a = affected.reindex(d.index)
        rows.append({"model": k, "MAPE_all": d.mean(), "MAPE_ordinary_days": d[~a].mean(),
                     "MAPE_holiday_affected_days": d[a].mean(), "worst_day": d.max()})
    table = pd.DataFrame(rows).set_index("model").round(2)
    print(table.to_string())
    print(f"\n(ordinary days: {int((~affected).sum())}, holiday-affected days: {int(affected.sum())})\n")

    print("Paired comparisons (daily MAPE; mean_diff < 0 means A is better; 95% block-bootstrap CI):")
    pairs = [(SELECTED, "naive_168"), (SELECTED, "mstl"), ("mstl", "naive_168"),
             (SELECTED, "lgbm_all"), (SELECTED, "lgbm_3")]
    out = []
    for a_name, b_name in pairs:
        for scope, mask in (("all days", None), ("ordinary", ~affected), ("holiday-affected", affected)):
            la, lb = daily[a_name], daily[b_name]
            if mask is not None:
                m = mask.reindex(la.index)
                la, lb = la[m], lb[m]
            dm = diebold_mariano(la, lb)
            lo, hi = block_bootstrap_ci(la - lb)
            out.append({"A": a_name, "B": b_name, "scope": scope, "days": dm["n_days"],
                        "mean_diff": round(dm["mean_diff"], 3), "CI95": f"[{lo:.3f}, {hi:.3f}]",
                        "p": f"{dm['p_value']:.3g}"})
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    table.to_csv("data/comparison_table.csv")
    res.to_csv("data/comparison_tests.csv", index=False)


if __name__ == "__main__":
    main()
