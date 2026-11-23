"""Draw the README figures from the saved predictions.

Needs data/pred_lgbm_window_5.parquet and data/pred_mstl.parquet (scripts/train_lgbm.py and
scripts/stat_baseline.py) plus data/DemaReal_Sistema.parquet. Writes docs/results.png and
docs/holy_week.png. Every number in the figures is recomputed here, not copied.
Usage: python scripts/make_figures.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xmforecast.evaluate import seasonal_naive  # noqa: E402
from xmforecast.features import build_features  # noqa: E402
from xmforecast.stats import block_bootstrap_ci, daily_mape  # noqa: E402

DOCS = HERE.parent / "docs"
TEST_START = "2023-01-01"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODELS = {"naive_168": ("Naive, last week", MUTED), "mstl": ("MSTL", ORANGE), "lgbm": ("LightGBM", BLUE)}
DAY_NAMES = {0: "weekday", 1: "near holiday", 2: "Saturday", 3: "Sunday", 4: "holiday"}


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)


def load():
    s = pd.read_parquet("data/DemaReal_Sistema.parquet").set_index("ts")["value"].sort_index()
    feats = build_features(s).loc[TEST_START:]
    preds = {
        "naive_168": seasonal_naive(s, 168).reindex(feats.index),
        "mstl": pd.read_parquet("data/pred_mstl.parquet")["pred"].reindex(feats.index),
        "lgbm": pd.read_parquet("data/pred_lgbm_window_5.parquet")["pred"].reindex(feats.index),
    }
    ok = pd.concat(preds, axis=1).notna().all(axis=1)  # the same hours for every model
    return feats[ok], {k: v[ok] for k, v in preds.items()}


def bars(ax, cats, values, fmt="{:.1f}"):
    """Grouped bars: values[model] is a list aligned with cats."""
    w = 0.26
    x = np.arange(len(cats))
    for i, (key, (label, color)) in enumerate(MODELS.items()):
        pos = x + (i - 1) * w
        ax.bar(pos, values[key], width=w * 0.92, color=color, label=label)
        for px, v in zip(pos, values[key]):
            ax.text(px, v + 0.12, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, cats)
    ax.margins(y=0.12)


def results_figure(feats, preds):
    y = feats["y"]
    daily = {k: daily_mape(y, p) for k, p in preds.items()}
    h12 = feats[feats.index.hour == 12]
    affected = ((h12["day_type"] == 4) | (h12["ref168_holiday"] == 1)).groupby(h12.index.normalize()).first()
    day_type = h12["day_type"].map(DAY_NAMES).groupby(h12.index.normalize()).first()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)
    (a, b), (c, d) = axes

    style(a, "MAPE by kind of day", "MAPE, %")
    cats = ["all days", "ordinary days", "holiday-affected"]
    vals = {k: [v.mean(), v[~affected.reindex(v.index)].mean(), v[affected.reindex(v.index)].mean()]
            for k, v in daily.items()}
    bars(a, cats, vals)
    a.set_ylim(bottom=0)

    style(b, "MAPE by month", "MAPE, %")
    for key, (label, color) in MODELS.items():
        m = daily[key].groupby(daily[key].index.to_period("M")).mean()
        b.plot(m.index.to_timestamp(), m.values, color=color, linewidth=2, label=label)
    b.set_xticks(pd.to_datetime([f"{yr}-01-01" for yr in range(2023, 2027)]), ["2023", "2024", "2025", "2026"])
    b.set_ylim(bottom=0)

    style(c, "MAPE by day type", "MAPE, %")
    types = ["weekday", "Saturday", "Sunday", "near holiday", "holiday"]
    vals_t = {k: [v[day_type.reindex(v.index) == t].mean() for t in types] for k, v in daily.items()}
    bars(c, types, vals_t)
    c.set_ylim(bottom=0)
    c.tick_params(axis="x", labelsize=8)

    style(d, "LightGBM minus the other model (95% CI)")
    d.grid(axis="y", visible=False)
    d.grid(axis="x", color=GRID, linewidth=0.8)
    rows = []
    for other, other_label in (("naive_168", "naive"), ("mstl", "MSTL")):
        for scope, mask in (("all", None), ("ordinary", ~affected), ("holiday", affected)):
            diff = daily["lgbm"] - daily[other]
            if mask is not None:
                diff = diff[mask.reindex(diff.index)]
            lo, hi = block_bootstrap_ci(diff)
            rows.append((f"{other_label}, {scope}", diff.mean(), lo, hi, MODELS[other][1]))
    for i, (label, m, lo, hi, color) in enumerate(rows):
        yy = len(rows) - 1 - i
        d.plot([lo, hi], [yy, yy], color=color, linewidth=2.2, solid_capstyle="round")
        d.plot(m, yy, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    d.axvline(0, color=INK_2, linewidth=1)
    d.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    d.tick_params(axis="y", labelsize=8.5, colors=INK_2)
    d.set_xlabel("daily MAPE points (negative: LightGBM is better)", fontsize=9, color=INK_2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c_) for _, c_ in MODELS.values()]
    fig.legend(handles, [lbl for lbl, _ in MODELS.values()], loc="upper right", ncol=3, frameon=False,
               fontsize=10, labelcolor=INK, bbox_to_anchor=(0.98, 0.99))
    fig.suptitle("xm-demand-forecast: error on 2023-2026", x=0.02, y=0.99,
                 ha="left", va="top", fontsize=13, color=INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.09, hspace=0.42, wspace=0.3)
    fig.savefig(DOCS / "results.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def holy_week_figure(feats, preds, start="2025-04-13", end="2025-04-27 23:00"):
    """Two weeks around Holy Week 2025: Thursday 17 and Friday 18 April are holidays."""
    sl = slice(start, end)
    y = feats["y"][sl] / 1e6  # kWh per hour -> GWh per hour
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    style(ax, "Holy Week 2025: forecast against actual, hour by hour", "demand, GWh per hour")
    ax.grid(axis="x", visible=False)
    for day, text in (("2025-04-17", "Holy Thursday\nand Good Friday\n(holidays)"),
                      ("2025-04-24", "naive copies the\nholidays onto\nnormal days")):
        x0 = pd.Timestamp(day)
        ax.axvspan(x0, x0 + pd.Timedelta(days=2), color=ORANGE if "naive" in text else MUTED, alpha=0.10, lw=0)
        ax.text(x0 + pd.Timedelta(days=1), 0.97, text, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8.5, color=INK_2)
    ax.plot(y.index, y.values, color=INK, linewidth=2.2, label="actual", zorder=3)
    ends = [("actual", INK, y)]
    for key in ("naive_168", "lgbm"):
        label, color = MODELS[key]
        p = preds[key][sl] / 1e6
        ape = ((preds[key][sl] - feats["y"][sl]).abs() / feats["y"][sl]).mean() * 100
        ax.plot(p.index, p.values, color=color, linewidth=1.6, linestyle="--" if key == "naive_168" else "-",
                label=f"{label}  (MAPE {ape:.1f}%)", zorder=2)
        ends.append((label, color, p))
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK, ncol=3, bbox_to_anchor=(0, -0.2))
    ax.set_xlim(y.index[0], y.index[-1])
    days = pd.date_range(start, end, freq="D")
    ax.set_xticks(days, [d.strftime("%a %d") for d in days])
    ax.tick_params(axis="x", labelsize=8)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 1.1)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    fig.savefig(DOCS / "holy_week.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    feats, preds = load()
    results_figure(feats, preds)
    holy_week_figure(feats, preds)
    print("wrote docs/results.png and docs/holy_week.png")


if __name__ == "__main__":
    main()
